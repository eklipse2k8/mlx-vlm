"""GVM spatio-temporal UNet (port of diffusers `UNetSpatioTemporalConditionModel`).

Layout is NHWC throughout. Only the configuration used by GVM checkpoints is
implemented: no class embedding and no additional time-id embedding.
"""

from __future__ import annotations


import mlx.core as mx
import mlx.nn as nn

from .config import UNetConfig
from .layers import (
    Downsample2D,
    GroupNorm,
    SpatioTemporalResBlock,
    TimestepEmbedding,
    Timesteps,
    TransformerSpatioTemporalModel,
    Upsample2D,
)


class UNetMidBlockSpatioTemporal(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        num_layers: int = 1,
        transformer_layers_per_block: int = 1,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
    ):
        super().__init__()
        self.has_cross_attention = True
        resnets = [
            SpatioTemporalResBlock(in_channels, in_channels, temb_channels, eps=1e-5)
        ]
        attentions = []
        for _ in range(num_layers):
            attentions.append(
                TransformerSpatioTemporalModel(
                    num_attention_heads,
                    in_channels // num_attention_heads,
                    in_channels=in_channels,
                    num_layers=transformer_layers_per_block,
                    cross_attention_dim=cross_attention_dim,
                )
            )
            resnets.append(
                SpatioTemporalResBlock(
                    in_channels, in_channels, temb_channels, eps=1e-5
                )
            )
        self.attentions = attentions
        self.resnets = resnets

    def __call__(
        self, hidden_states, temb, encoder_hidden_states, image_only_indicator
    ):
        hidden_states = self.resnets[0](hidden_states, temb, image_only_indicator)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(
                hidden_states, encoder_hidden_states, image_only_indicator
            )
            hidden_states = resnet(hidden_states, temb, image_only_indicator)
        return hidden_states


class DownBlockSpatioTemporal(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        temb_channels,
        num_layers=1,
        add_downsample=True,
    ):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            resnets.append(
                SpatioTemporalResBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    temb_channels,
                    eps=1e-5,
                )
            )
        self.resnets = resnets
        self.downsamplers = (
            [Downsample2D(out_channels, out_channels, padding=1)]
            if add_downsample
            else None
        )

    def __call__(self, hidden_states, temb, image_only_indicator):
        output_states = ()
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb, image_only_indicator)
            output_states += (hidden_states,)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states += (hidden_states,)
        return hidden_states, output_states


class CrossAttnDownBlockSpatioTemporal(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        temb_channels,
        num_layers=1,
        transformer_layers_per_block=1,
        num_attention_heads=1,
        cross_attention_dim=1280,
        add_downsample=True,
    ):
        super().__init__()
        self.has_cross_attention = True
        resnets, attentions = [], []
        for i in range(num_layers):
            resnets.append(
                SpatioTemporalResBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    temb_channels,
                    eps=1e-6,
                )
            )
            attentions.append(
                TransformerSpatioTemporalModel(
                    num_attention_heads,
                    out_channels // num_attention_heads,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block,
                    cross_attention_dim=cross_attention_dim,
                )
            )
        self.resnets = resnets
        self.attentions = attentions
        self.downsamplers = (
            [Downsample2D(out_channels, out_channels, padding=1)]
            if add_downsample
            else None
        )

    def __call__(
        self, hidden_states, temb, encoder_hidden_states, image_only_indicator
    ):
        output_states = ()
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb, image_only_indicator)
            hidden_states = attn(
                hidden_states, encoder_hidden_states, image_only_indicator
            )
            output_states += (hidden_states,)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states += (hidden_states,)
        return hidden_states, output_states


class UpBlockSpatioTemporal(nn.Module):
    def __init__(
        self,
        in_channels,
        prev_output_channel,
        out_channels,
        temb_channels,
        num_layers=1,
        add_upsample=True,
    ):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnets.append(
                SpatioTemporalResBlock(
                    resnet_in_channels + res_skip_channels,
                    out_channels,
                    temb_channels,
                    eps=1e-6,
                )
            )
        self.resnets = resnets
        self.upsamplers = (
            [Upsample2D(out_channels, out_channels)] if add_upsample else None
        )

    def __call__(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb,
        image_only_indicator,
        upsample_size=None,
    ):
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = mx.concatenate([hidden_states, res_hidden_states], axis=-1)
            hidden_states = resnet(hidden_states, temb, image_only_indicator)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
        return hidden_states


class CrossAttnUpBlockSpatioTemporal(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        prev_output_channel,
        temb_channels,
        num_layers=1,
        transformer_layers_per_block=1,
        num_attention_heads=1,
        cross_attention_dim=1280,
        add_upsample=True,
    ):
        super().__init__()
        self.has_cross_attention = True
        resnets, attentions = [], []
        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnets.append(
                SpatioTemporalResBlock(
                    resnet_in_channels + res_skip_channels,
                    out_channels,
                    temb_channels,
                    eps=1e-6,
                )
            )
            attentions.append(
                TransformerSpatioTemporalModel(
                    num_attention_heads,
                    out_channels // num_attention_heads,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block,
                    cross_attention_dim=cross_attention_dim,
                )
            )
        self.resnets = resnets
        self.attentions = attentions
        self.upsamplers = (
            [Upsample2D(out_channels, out_channels)] if add_upsample else None
        )

    def __call__(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb,
        encoder_hidden_states,
        image_only_indicator,
        upsample_size=None,
    ):
        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = mx.concatenate([hidden_states, res_hidden_states], axis=-1)
            hidden_states = resnet(hidden_states, temb, image_only_indicator)
            hidden_states = attn(
                hidden_states, encoder_hidden_states, image_only_indicator
            )
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
        return hidden_states


class GVMUNet(nn.Module):
    """Spatio-temporal UNet as used by GVM (SVD-XT architecture)."""

    def __init__(self, config: UNetConfig):
        super().__init__()
        self.config = config

        down_block_types = config.down_block_types
        up_block_types = config.up_block_types
        block_out_channels = config.block_out_channels
        layers_per_block = config.layers_per_block
        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)
        num_attention_heads = config.num_attention_heads
        cross_attention_dim = (config.cross_attention_dim,) * len(down_block_types)
        transformer_layers_per_block = (config.transformer_layers_per_block,) * len(
            down_block_types
        )

        self.conv_in = nn.Conv2d(
            config.in_channels, block_out_channels[0], 3, padding=1
        )

        time_embed_dim = block_out_channels[0] * 4
        self.time_proj = Timesteps(block_out_channels[0], True, 0)
        self.time_embedding = TimestepEmbedding(block_out_channels[0], time_embed_dim)

        # down
        self.down_blocks = []
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            common = dict(
                temb_channels=time_embed_dim,
                num_layers=layers_per_block[i],
                add_downsample=not is_final_block,
            )
            if down_block_type == "CrossAttnDownBlockSpatioTemporal":
                block = CrossAttnDownBlockSpatioTemporal(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    transformer_layers_per_block=transformer_layers_per_block[i],
                    num_attention_heads=num_attention_heads[i],
                    cross_attention_dim=cross_attention_dim[i],
                    **common,
                )
            elif down_block_type == "DownBlockSpatioTemporal":
                block = DownBlockSpatioTemporal(
                    in_channels=input_channel, out_channels=output_channel, **common
                )
            else:
                raise ValueError(f"Unsupported down block: {down_block_type}")
            self.down_blocks.append(block)

        # mid
        self.mid_block = UNetMidBlockSpatioTemporal(
            block_out_channels[-1],
            temb_channels=time_embed_dim,
            transformer_layers_per_block=transformer_layers_per_block[-1],
            num_attention_heads=num_attention_heads[-1],
            cross_attention_dim=cross_attention_dim[-1],
        )

        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))
        reversed_transformer_layers = list(reversed(transformer_layers_per_block))

        self.up_blocks = []
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[
                min(i + 1, len(block_out_channels) - 1)
            ]
            add_upsample = not is_final_block
            if add_upsample:
                self.num_upsamplers += 1
            common = dict(
                num_layers=reversed_layers_per_block[i] + 1,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
            )
            if up_block_type == "CrossAttnUpBlockSpatioTemporal":
                block = CrossAttnUpBlockSpatioTemporal(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    transformer_layers_per_block=reversed_transformer_layers[i],
                    num_attention_heads=reversed_num_attention_heads[i],
                    cross_attention_dim=reversed_cross_attention_dim[i],
                    **common,
                )
            elif up_block_type == "UpBlockSpatioTemporal":
                block = UpBlockSpatioTemporal(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    **common,
                )
            else:
                raise ValueError(f"Unsupported up block: {up_block_type}")
            self.up_blocks.append(block)
            prev_output_channel = output_channel

        self.conv_norm_out = GroupNorm(32, block_out_channels[0], eps=1e-5)
        self.conv_out = nn.Conv2d(
            block_out_channels[0], config.out_channels, 3, padding=1
        )

    def __call__(
        self,
        sample: mx.array,
        timestep: mx.array,
        encoder_hidden_states: mx.array,
    ) -> mx.array:
        """sample: (B, F, H, W, C) noisy latents; returns (B, F, H, W, out_channels)."""
        batch_size, num_frames = sample.shape[:2]

        default_overall_up_factor = 2**self.num_upsamplers
        forward_upsample_size = any(
            s % default_overall_up_factor != 0 for s in sample.shape[2:4]
        )

        # 1. time
        if timestep.ndim == 0:
            timestep = timestep[None]
        timesteps = mx.broadcast_to(timestep, (batch_size,))
        t_emb = self.time_proj(timesteps)
        emb = self.time_embedding(t_emb.astype(sample.dtype))

        # flatten frames into the batch axis
        sample = sample.reshape(batch_size * num_frames, *sample.shape[2:])
        emb = mx.repeat(emb, num_frames, axis=0)
        encoder_hidden_states = mx.repeat(encoder_hidden_states, num_frames, axis=0)

        # 2. pre-process
        sample = self.conv_in(sample)

        image_only_indicator = mx.zeros((batch_size, num_frames), dtype=sample.dtype)

        # 3. down
        down_block_res_samples = (sample,)
        for block in self.down_blocks:
            if getattr(block, "has_cross_attention", False):
                sample, res_samples = block(
                    sample, emb, encoder_hidden_states, image_only_indicator
                )
            else:
                sample, res_samples = block(sample, emb, image_only_indicator)
            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(
            sample, emb, encoder_hidden_states, image_only_indicator
        )

        # 5. up
        for i, block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1
            res_samples = down_block_res_samples[-len(block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(block.resnets)]
            upsample_size = None
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[1:3]

            if getattr(block, "has_cross_attention", False):
                sample = block(
                    sample,
                    res_samples,
                    emb,
                    encoder_hidden_states,
                    image_only_indicator,
                    upsample_size,
                )
            else:
                sample = block(
                    sample, res_samples, emb, image_only_indicator, upsample_size
                )

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = nn.silu(sample)
        sample = self.conv_out(sample)

        return sample.reshape(batch_size, num_frames, *sample.shape[1:])
