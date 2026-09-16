"""Shared building blocks for the GVM spatio-temporal UNet and temporal VAE.

All modules use MLX's native NHWC / NDHWC layout. Weight loading from the
diffusers checkpoint is handled by the sanitizers in `weights.py`.
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


def get_timestep_embedding(
    timesteps: mx.array,
    embedding_dim: int,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0.0,
    scale: float = 1.0,
    max_period: int = 10000,
) -> mx.array:
    """Sinusoidal timestep embedding, matching diffusers `get_timestep_embedding`."""
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * mx.arange(half_dim, dtype=mx.float32)
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = mx.exp(exponent)
    emb = timesteps.astype(mx.float32)[:, None] * emb[None, :]
    emb = scale * emb
    emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
    if flip_sin_to_cos:
        emb = mx.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)
    return emb


class Timesteps(nn.Module):
    def __init__(
        self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def __call__(self, timesteps: mx.array) -> mx.array:
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )


class TimestepEmbedding(nn.Module):
    def __init__(
        self, in_channels: int, time_embed_dim: int, out_dim: Optional[int] = None
    ):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, out_dim or time_embed_dim)

    def __call__(self, sample: mx.array) -> mx.array:
        return self.linear_2(nn.silu(self.linear_1(sample)))


class GroupNorm(nn.GroupNorm):
    def __init__(self, num_groups: int, dims: int, eps: float = 1e-5):
        super().__init__(
            num_groups, dims, eps=eps, affine=True, pytorch_compatible=True
        )


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states, gate = mx.split(self.proj(hidden_states), 2, axis=-1)
        return hidden_states * nn.gelu(gate)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        inner_dim: Optional[int] = None,
    ):
        super().__init__()
        inner_dim = inner_dim or int(dim * mult)
        # Keep the dropout slot so weight names match the diffusers checkpoint
        self.net = [
            GEGLU(dim, inner_dim),
            nn.Dropout(0.0),
            nn.Linear(inner_dim, dim_out or dim),
        ]

    def __call__(self, hidden_states: mx.array) -> mx.array:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class Attention(nn.Module):
    """Scaled dot-product attention, matching diffusers `Attention` + AttnProcessor2_0."""

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        bias: bool = False,
        out_bias: bool = True,
        norm_num_groups: Optional[int] = None,
        residual_connection: bool = False,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.residual_connection = residual_connection
        inner_dim = heads * dim_head
        kv_dim = cross_attention_dim if cross_attention_dim is not None else query_dim

        self.group_norm = (
            GroupNorm(norm_num_groups, query_dim, eps=eps)
            if norm_num_groups is not None
            else None
        )
        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(kv_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(kv_dim, inner_dim, bias=bias)
        # List keeps the `to_out.0.weight` naming of the checkpoint
        self.to_out = [nn.Linear(inner_dim, query_dim, bias=out_bias), nn.Dropout(0.0)]

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: Optional[mx.array] = None,
    ) -> mx.array:
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:  # (B, H, W, C) -> (B, H*W, C)
            batch, height, width, _ = hidden_states.shape
            hidden_states = hidden_states.reshape(batch, height * width, -1)

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states)

        context = (
            hidden_states if encoder_hidden_states is None else encoder_hidden_states
        )
        batch = hidden_states.shape[0]

        query = self.to_q(hidden_states)
        key = self.to_k(context)
        value = self.to_v(context)

        query = query.reshape(batch, -1, self.heads, self.dim_head).transpose(
            0, 2, 1, 3
        )
        key = key.reshape(batch, -1, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        value = value.reshape(batch, -1, self.heads, self.dim_head).transpose(
            0, 2, 1, 3
        )

        out = mx.fast.scaled_dot_product_attention(query, key, value, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(batch, -1, self.heads * self.dim_head)
        out = self.to_out[0](out)

        if input_ndim == 4:
            out = out.reshape(batch, height, width, -1)

        if self.residual_connection:
            out = out + residual
        return out


class BasicTransformerBlock(nn.Module):
    """diffusers `BasicTransformerBlock` with layer_norm, self-attn + cross-attn + GEGLU FF."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: Optional[int] = None,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn1 = Attention(
            query_dim=dim, heads=num_attention_heads, dim_head=attention_head_dim
        )
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
        )
        self.norm3 = nn.LayerNorm(dim, eps=norm_eps)
        self.ff = FeedForward(dim)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: Optional[mx.array] = None,
    ) -> mx.array:
        hidden_states = self.attn1(self.norm1(hidden_states)) + hidden_states
        hidden_states = (
            self.attn2(self.norm2(hidden_states), encoder_hidden_states) + hidden_states
        )
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
        return hidden_states


class TemporalBasicTransformerBlock(nn.Module):
    """diffusers `TemporalBasicTransformerBlock`; input (B*F, S, C)."""

    def __init__(
        self,
        dim: int,
        time_mix_inner_dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: Optional[int] = None,
    ):
        super().__init__()
        self.is_res = dim == time_mix_inner_dim
        self.norm_in = nn.LayerNorm(dim)
        self.ff_in = FeedForward(dim, dim_out=time_mix_inner_dim)
        self.norm1 = nn.LayerNorm(time_mix_inner_dim)
        self.attn1 = Attention(
            query_dim=time_mix_inner_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
        )
        if cross_attention_dim is not None:
            self.norm2 = nn.LayerNorm(time_mix_inner_dim)
            self.attn2 = Attention(
                query_dim=time_mix_inner_dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
            )
        else:
            self.norm2 = None
            self.attn2 = None
        self.norm3 = nn.LayerNorm(time_mix_inner_dim)
        self.ff = FeedForward(time_mix_inner_dim)

    def __call__(
        self,
        hidden_states: mx.array,
        num_frames: int,
        encoder_hidden_states: Optional[mx.array] = None,
    ) -> mx.array:
        batch_frames, seq_length, channels = hidden_states.shape
        batch_size = batch_frames // num_frames

        # (B*F, S, C) -> (B*S, F, C): attend over the frame axis
        hidden_states = (
            hidden_states.reshape(batch_size, num_frames, seq_length, channels)
            .transpose(0, 2, 1, 3)
            .reshape(batch_size * seq_length, num_frames, channels)
        )

        residual = hidden_states
        hidden_states = self.ff_in(self.norm_in(hidden_states))
        if self.is_res:
            hidden_states = hidden_states + residual

        hidden_states = self.attn1(self.norm1(hidden_states)) + hidden_states
        if self.attn2 is not None:
            hidden_states = (
                self.attn2(self.norm2(hidden_states), encoder_hidden_states)
                + hidden_states
            )

        ff_output = self.ff(self.norm3(hidden_states))
        hidden_states = ff_output + hidden_states if self.is_res else ff_output

        # back to (B*F, S, C)
        return (
            hidden_states.reshape(batch_size, seq_length, num_frames, -1)
            .transpose(0, 2, 1, 3)
            .reshape(batch_frames, seq_length, -1)
        )


class TransformerSpatioTemporalModel(nn.Module):
    """diffusers `TransformerSpatioTemporalModel` (spatial + temporal transformer pair)."""

    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        in_channels: int,
        num_layers: int = 1,
        cross_attention_dim: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = num_attention_heads * attention_head_dim

        self.norm = GroupNorm(32, in_channels, eps=1e-6)
        self.proj_in = nn.Linear(in_channels, inner_dim)
        self.transformer_blocks = [
            BasicTransformerBlock(
                inner_dim,
                num_attention_heads,
                attention_head_dim,
                cross_attention_dim=cross_attention_dim,
            )
            for _ in range(num_layers)
        ]
        self.temporal_transformer_blocks = [
            TemporalBasicTransformerBlock(
                inner_dim,
                inner_dim,
                num_attention_heads,
                attention_head_dim,
                cross_attention_dim=cross_attention_dim,
            )
            for _ in range(num_layers)
        ]
        time_embed_dim = in_channels * 4
        self.time_pos_embed = TimestepEmbedding(
            in_channels, time_embed_dim, out_dim=in_channels
        )
        self.time_proj = Timesteps(in_channels, True, 0)
        self.time_mixer = AlphaBlender(0.5, "learned_with_images")
        self.proj_out = nn.Linear(inner_dim, in_channels)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        image_only_indicator: mx.array,
    ) -> mx.array:
        batch_frames, height, width, channels = hidden_states.shape
        num_frames = image_only_indicator.shape[-1]
        batch_size = batch_frames // num_frames

        # Cross-attention context for temporal blocks: first-frame encoder state
        time_context = encoder_hidden_states.reshape(
            batch_size, num_frames, -1, encoder_hidden_states.shape[-1]
        )[:, 0]
        time_context = mx.broadcast_to(
            time_context[:, None, :, :],
            (batch_size, height * width, *time_context.shape[1:]),
        )
        time_context = time_context.reshape(
            batch_size * height * width, -1, encoder_hidden_states.shape[-1]
        )

        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.reshape(batch_frames, height * width, channels)
        hidden_states = self.proj_in(hidden_states)

        # Per-frame sinusoidal position embedding
        frame_ids = mx.tile(mx.arange(num_frames), batch_size)
        t_emb = self.time_proj(frame_ids).astype(hidden_states.dtype)
        emb = self.time_pos_embed(t_emb)[:, None, :]

        for block, temporal_block in zip(
            self.transformer_blocks, self.temporal_transformer_blocks
        ):
            hidden_states = block(hidden_states, encoder_hidden_states)
            hidden_states_mix = temporal_block(
                hidden_states + emb,
                num_frames=num_frames,
                encoder_hidden_states=time_context,
            )
            hidden_states = self.time_mixer(
                hidden_states, hidden_states_mix, image_only_indicator
            )

        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch_frames, height, width, -1)
        return hidden_states + residual


class AlphaBlender(nn.Module):
    """Blends spatial and temporal features (diffusers `AlphaBlender`)."""

    def __init__(
        self,
        alpha: float,
        merge_strategy: str = "learned_with_images",
        switch_spatial_to_temporal_mix: bool = False,
    ):
        super().__init__()
        if merge_strategy not in ("learned", "fixed", "learned_with_images"):
            raise ValueError(f"Unknown merge_strategy {merge_strategy}")
        self.merge_strategy = merge_strategy
        self.switch_spatial_to_temporal_mix = switch_spatial_to_temporal_mix
        self.mix_factor = mx.array([alpha])

    def _alpha(self, image_only_indicator: Optional[mx.array], ndims: int) -> mx.array:
        if self.merge_strategy == "fixed":
            return self.mix_factor
        if self.merge_strategy == "learned":
            return mx.sigmoid(self.mix_factor)
        if image_only_indicator is None:
            raise ValueError("image_only_indicator is required for learned_with_images")
        alpha = mx.where(
            image_only_indicator.astype(mx.bool_),
            mx.ones((1, 1)),
            mx.sigmoid(self.mix_factor)[..., None],
        )
        if ndims == 5:  # (B, F, H, W, C) in NHWC
            return alpha[:, :, None, None, None]
        if ndims == 3:  # (B*F, S, C)
            return alpha.reshape(-1)[:, None, None]
        raise ValueError(f"Unexpected ndims {ndims}")

    def __call__(
        self,
        x_spatial: mx.array,
        x_temporal: mx.array,
        image_only_indicator: Optional[mx.array] = None,
    ) -> mx.array:
        alpha = self._alpha(image_only_indicator, x_spatial.ndim).astype(
            x_spatial.dtype
        )
        if self.switch_spatial_to_temporal_mix:
            alpha = 1.0 - alpha
        return alpha * x_spatial + (1.0 - alpha) * x_temporal


class ResnetBlock2D(nn.Module):
    """diffusers `ResnetBlock2D` (default time embedding norm) in NHWC."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: Optional[int] = 512,
        eps: float = 1e-6,
        groups: int = 32,
    ):
        super().__init__()
        self.norm1 = GroupNorm(groups, in_channels, eps=eps)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_emb_proj = (
            nn.Linear(temb_channels, out_channels)
            if temb_channels is not None
            else None
        )
        self.norm2 = GroupNorm(groups, out_channels, eps=eps)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv_shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def __call__(self, x: mx.array, temb: Optional[mx.array] = None) -> mx.array:
        hidden_states = self.conv1(nn.silu(self.norm1(x)))
        if self.time_emb_proj is not None and temb is not None:
            hidden_states = (
                hidden_states + self.time_emb_proj(nn.silu(temb))[:, None, None, :]
            )
        hidden_states = self.conv2(nn.silu(self.norm2(hidden_states)))
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + hidden_states


class TemporalResnetBlock(nn.Module):
    """diffusers `TemporalResnetBlock`; input (B, F, H, W, C)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: Optional[int] = 512,
        eps: float = 1e-6,
    ):
        super().__init__()
        kernel_size = (3, 1, 1)
        padding = (1, 0, 0)
        self.norm1 = GroupNorm(32, in_channels, eps=eps)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.time_emb_proj = (
            nn.Linear(temb_channels, out_channels)
            if temb_channels is not None
            else None
        )
        self.norm2 = GroupNorm(32, out_channels, eps=eps)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding)
        self.conv_shortcut = (
            nn.Conv3d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def __call__(self, x: mx.array, temb: Optional[mx.array]) -> mx.array:
        hidden_states = self.conv1(nn.silu(self.norm1(x)))
        if self.time_emb_proj is not None and temb is not None:
            # temb (B, F, C) -> (B, F, 1, 1, out)
            hidden_states = (
                hidden_states + self.time_emb_proj(nn.silu(temb))[:, :, None, None, :]
            )
        hidden_states = self.conv2(nn.silu(self.norm2(hidden_states)))
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + hidden_states


class SpatioTemporalResBlock(nn.Module):
    """diffusers `SpatioTemporalResBlock`: spatial resnet, temporal resnet, blend."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: Optional[int] = 512,
        eps: float = 1e-6,
        temporal_eps: Optional[float] = None,
        merge_factor: float = 0.5,
        merge_strategy: str = "learned_with_images",
        switch_spatial_to_temporal_mix: bool = False,
    ):
        super().__init__()
        self.spatial_res_block = ResnetBlock2D(
            in_channels, out_channels, temb_channels, eps=eps
        )
        self.temporal_res_block = TemporalResnetBlock(
            out_channels,
            out_channels,
            temb_channels,
            eps=temporal_eps if temporal_eps is not None else eps,
        )
        self.time_mixer = AlphaBlender(
            merge_factor, merge_strategy, switch_spatial_to_temporal_mix
        )

    def __call__(
        self,
        hidden_states: mx.array,
        temb: Optional[mx.array] = None,
        image_only_indicator: Optional[mx.array] = None,
    ) -> mx.array:
        num_frames = image_only_indicator.shape[-1]
        hidden_states = self.spatial_res_block(hidden_states, temb)

        batch_frames, height, width, channels = hidden_states.shape
        batch_size = batch_frames // num_frames

        # (B*F, H, W, C) -> (B, F, H, W, C) for the temporal convs
        hidden_states_5d = hidden_states.reshape(
            batch_size, num_frames, height, width, channels
        )
        temb_5d = temb.reshape(batch_size, num_frames, -1) if temb is not None else None

        temporal = self.temporal_res_block(hidden_states_5d, temb_5d)
        hidden_states_5d = self.time_mixer(
            hidden_states_5d, temporal, image_only_indicator
        )
        return hidden_states_5d.reshape(batch_frames, height, width, channels)


class Downsample2D(nn.Module):
    """diffusers `Downsample2D` with use_conv=True (name="op")."""

    def __init__(self, channels: int, out_channels: int, padding: int = 1):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv2d(channels, out_channels, 3, stride=2, padding=padding)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        if self.padding == 0:
            # Asymmetric (bottom, right) pad, as in diffusers
            hidden_states = mx.pad(hidden_states, [(0, 0), (0, 1), (0, 1), (0, 0)])
        return self.conv(hidden_states)


class Upsample2D(nn.Module):
    """diffusers `Upsample2D` with use_conv=True (name="conv")."""

    def __init__(self, channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, out_channels, 3, padding=1)

    def __call__(
        self, hidden_states: mx.array, output_size: Optional[tuple[int, int]] = None
    ) -> mx.array:
        if output_size is None:
            hidden_states = nn.Upsample(scale_factor=2.0, mode="nearest")(hidden_states)
        else:
            hidden_states = nn.Upsample(
                scale_factor=(
                    output_size[0] / hidden_states.shape[1],
                    output_size[1] / hidden_states.shape[2],
                ),
                mode="nearest",
            )(hidden_states)
        return self.conv(hidden_states)


class LoRALinear(nn.Module):
    """Linear layer with an additive LoRA update (peft convention)."""

    def __init__(self, base: nn.Linear, rank: int, scale: float):
        super().__init__()
        self.base = base
        self.scale = scale
        out_dim, in_dim = base.weight.shape
        self.lora_a = mx.zeros((rank, in_dim))
        self.lora_b = mx.zeros((out_dim, rank))

    def __call__(self, x: mx.array) -> mx.array:
        delta = (x @ self.lora_a.T) @ self.lora_b.T * self.scale
        return self.base(x) + delta.astype(x.dtype)


class LoRAConv2d(nn.Module):
    """Conv2d with an additive LoRA update (peft convention)."""

    def __init__(self, base: nn.Conv2d, rank: int, scale: float):
        super().__init__()
        self.base = base
        self.scale = scale
        out_dim, kh, kw, in_dim = base.weight.shape
        self.lora_a = nn.Conv2d(
            in_dim, rank, (kh, kw), padding=(kh // 2, kw // 2), bias=False
        )
        self.lora_b = nn.Conv2d(rank, out_dim, 1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.base(x) + self.lora_b(self.lora_a(x)) * self.scale
