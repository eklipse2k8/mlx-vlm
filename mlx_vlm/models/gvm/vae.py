"""GVM temporal VAE (port of diffusers `AutoencoderKLTemporalDecoder`).

NHWC layout. Encoding returns the posterior mean (mode); decoding runs the
temporal decoder with a final 3D time convolution.
"""

from __future__ import annotations


import mlx.core as mx
import mlx.nn as nn

from .config import VAEConfig
from .layers import (
    Attention,
    Downsample2D,
    GroupNorm,
    ResnetBlock2D,
    SpatioTemporalResBlock,
    Upsample2D,
)


class DownEncoderBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1, add_downsample=True):
        super().__init__()
        self.resnets = [
            ResnetBlock2D(
                in_channels if i == 0 else out_channels,
                out_channels,
                temb_channels=None,
                eps=1e-6,
            )
            for i in range(num_layers)
        ]
        # VAE encoder downsamples with asymmetric padding (padding=0)
        self.downsamplers = (
            [Downsample2D(out_channels, out_channels, padding=0)]
            if add_downsample
            else None
        )

    def __call__(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
        return hidden_states


class UNetMidBlock2D(nn.Module):
    """VAE encoder mid block: resnet, attention, resnet."""

    def __init__(self, in_channels, attention_head_dim, resnet_eps=1e-6):
        super().__init__()
        self.resnets = [
            ResnetBlock2D(in_channels, in_channels, temb_channels=None, eps=resnet_eps),
            ResnetBlock2D(in_channels, in_channels, temb_channels=None, eps=resnet_eps),
        ]
        self.attentions = [
            Attention(
                query_dim=in_channels,
                heads=in_channels // attention_head_dim,
                dim_head=attention_head_dim,
                norm_num_groups=32,
                residual_connection=True,
                bias=True,
                eps=resnet_eps,
            )
        ]

    def __call__(self, hidden_states):
        hidden_states = self.resnets[0](hidden_states)
        hidden_states = self.attentions[0](hidden_states)
        hidden_states = self.resnets[1](hidden_states)
        return hidden_states


class Encoder(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        block_out_channels = config.block_out_channels
        self.conv_in = nn.Conv2d(
            config.in_channels, block_out_channels[0], 3, padding=1
        )

        self.down_blocks = []
        output_channel = block_out_channels[0]
        for i in range(len(config.down_block_types)):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            self.down_blocks.append(
                DownEncoderBlock2D(
                    input_channel,
                    output_channel,
                    num_layers=config.layers_per_block,
                    add_downsample=not is_final_block,
                )
            )

        self.mid_block = UNetMidBlock2D(
            block_out_channels[-1], attention_head_dim=block_out_channels[-1]
        )
        self.conv_norm_out = GroupNorm(32, block_out_channels[-1], eps=1e-6)
        self.conv_out = nn.Conv2d(
            block_out_channels[-1], 2 * config.latent_channels, 3, padding=1
        )

    def __call__(self, sample):
        sample = self.conv_in(sample)
        for down_block in self.down_blocks:
            sample = down_block(sample)
        sample = self.mid_block(sample)
        return self.conv_out(nn.silu(self.conv_norm_out(sample)))


class MidBlockTemporalDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, attention_head_dim=512, num_layers=1):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            resnets.append(
                SpatioTemporalResBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    temb_channels=None,
                    eps=1e-6,
                    temporal_eps=1e-5,
                    merge_factor=0.0,
                    merge_strategy="learned",
                    switch_spatial_to_temporal_mix=True,
                )
            )
        self.resnets = resnets
        self.attentions = [
            Attention(
                query_dim=in_channels,
                heads=in_channels // attention_head_dim,
                dim_head=attention_head_dim,
                norm_num_groups=32,
                residual_connection=True,
                bias=True,
                eps=1e-6,
            )
        ]

    def __call__(self, hidden_states, image_only_indicator):
        hidden_states = self.resnets[0](
            hidden_states, image_only_indicator=image_only_indicator
        )
        for resnet, attn in zip(self.resnets[1:], self.attentions):
            hidden_states = attn(hidden_states)
            hidden_states = resnet(
                hidden_states, image_only_indicator=image_only_indicator
            )
        return hidden_states


class UpBlockTemporalDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1, add_upsample=True):
        super().__init__()
        self.resnets = [
            SpatioTemporalResBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                temb_channels=None,
                eps=1e-6,
                temporal_eps=1e-5,
                merge_factor=0.0,
                merge_strategy="learned",
                switch_spatial_to_temporal_mix=True,
            )
            for i in range(num_layers)
        ]
        self.upsamplers = (
            [Upsample2D(out_channels, out_channels)] if add_upsample else None
        )

    def __call__(self, hidden_states, image_only_indicator):
        for resnet in self.resnets:
            hidden_states = resnet(
                hidden_states, image_only_indicator=image_only_indicator
            )
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        return hidden_states


class TemporalDecoder(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        block_out_channels = config.block_out_channels
        self.conv_in = nn.Conv2d(
            config.latent_channels, block_out_channels[-1], 3, padding=1
        )
        self.mid_block = MidBlockTemporalDecoder(
            num_layers=config.layers_per_block,
            in_channels=block_out_channels[-1],
            out_channels=block_out_channels[-1],
            attention_head_dim=block_out_channels[-1],
        )

        self.up_blocks = []
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i in range(len(block_out_channels)):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            self.up_blocks.append(
                UpBlockTemporalDecoder(
                    num_layers=config.layers_per_block + 1,
                    in_channels=prev_output_channel,
                    out_channels=output_channel,
                    add_upsample=i != len(block_out_channels) - 1,
                )
            )
            prev_output_channel = output_channel

        self.conv_norm_out = GroupNorm(32, block_out_channels[0], eps=1e-6)
        self.conv_out = nn.Conv2d(
            block_out_channels[0], config.out_channels, 3, padding=1
        )
        self.time_conv_out = nn.Conv3d(
            config.out_channels, config.out_channels, (3, 1, 1), padding=(1, 0, 0)
        )

    def __call__(self, sample, image_only_indicator, num_frames=1):
        sample = self.conv_in(sample)
        sample = self.mid_block(sample, image_only_indicator)
        for up_block in self.up_blocks:
            sample = up_block(sample, image_only_indicator)
        sample = self.conv_out(nn.silu(self.conv_norm_out(sample)))

        batch_frames, height, width, channels = sample.shape
        batch_size = batch_frames // num_frames
        # (B*F, H, W, C) -> (B, F, H, W, C) for the temporal conv
        sample = sample.reshape(batch_size, num_frames, height, width, channels)
        sample = self.time_conv_out(sample)
        return sample.reshape(batch_frames, height, width, channels)


class GVMVAE(nn.Module):
    """AutoencoderKLTemporalDecoder port (encode = posterior mode, chunked decode)."""

    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = TemporalDecoder(config)
        self.quant_conv = nn.Conv2d(
            2 * config.latent_channels, 2 * config.latent_channels, 1
        )

    def encode(self, x: mx.array) -> mx.array:
        """x: (B*F, H, W, 3) -> posterior mean latents (B*F, H/8, W/8, 4)."""
        moments = self.quant_conv(self.encoder(x))
        mean, logvar = mx.split(moments, 2, axis=-1)
        logvar = mx.clip(logvar, -30.0, 20.0)  # kept for parity with diffusers
        return mean

    def decode(self, z: mx.array, num_frames: int) -> mx.array:
        """z: (B*F, h, w, 4) -> (B*F, H, W, 3)."""
        batch_size = z.shape[0] // num_frames
        image_only_indicator = mx.zeros((batch_size, num_frames), dtype=z.dtype)
        return self.decoder(
            z, image_only_indicator=image_only_indicator, num_frames=num_frames
        )
