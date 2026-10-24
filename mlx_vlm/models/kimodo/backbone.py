# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Lazy MLX transformer and two-stage Kimodo motion denoiser."""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import scaled_dot_product_attention
from .config import ModelConfig


def pad_x_and_mask_to_fixed_size(x: mx.array, mask: mx.array, size: int):
    """Crop or right-pad feature and validity tensors without reading the GPU."""
    length = x.shape[1]
    if length >= size:
        return x[:, :size], mask[:, :size]
    return (
        mx.pad(x, ((0, 0), (0, size - length), (0, 0))),
        mx.pad(mask, ((0, 0), (0, size - length))),
    )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = mx.arange(max_len, dtype=mx.float32)[:, None]
        frequency = mx.power(
            10000.0, -mx.arange(0, d_model, 2, dtype=mx.float32) / d_model
        )
        phase = position * frequency
        # Private state matches the source's non-persistent position buffer.
        self._pe = mx.stack((mx.sin(phase), mx.cos(phase)), axis=-1).reshape(
            1, max_len, d_model
        )

    @property
    def pe(self):
        return self._pe

    def __call__(self, x: mx.array):
        if x.shape[1] > self._pe.shape[1]:
            raise ValueError("Motion and text tokens exceed the position table")
        return self.dropout(x + self._pe[:, : x.shape[1]].astype(x.dtype))


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim: int, sequence_pos_encoder: PositionalEncoding):
        super().__init__()
        self.sequence_pos_encoder = sequence_pos_encoder
        # A module list preserves the source's numeric Sequential weight paths.
        self.time_embed = [
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        ]

    def __call__(self, timesteps: mx.array):
        x = self.sequence_pos_encoder.pe[0, timesteps][:, None]
        for layer in self.time_embed:
            x = layer(x)
        return x


class SelfAttention(nn.Module):
    """Packed QKV projection with the source MultiheadAttention weight names."""

    def __init__(self, dims: int, num_heads: int):
        super().__init__()
        projection = nn.Linear(dims, 3 * dims)
        self.in_proj_weight = projection.weight
        self.in_proj_bias = projection.bias
        self.out_proj = nn.Linear(dims, dims)
        self.num_heads = num_heads

    def __call__(self, x: mx.array, mask: mx.array):
        batch, length, dims = x.shape
        qkv = mx.addmm(self.in_proj_bias, x, self.in_proj_weight.T)
        q, k, v = (
            value.reshape(batch, length, self.num_heads, -1).transpose(0, 2, 1, 3)
            for value in mx.split(qkv, 3, axis=-1)
        )
        output = scaled_dot_product_attention(
            q, k, v, cache=None, scale=q.shape[-1] ** -0.5, mask=mask
        )
        return self.out_proj(output.transpose(0, 2, 1, 3).reshape(batch, length, dims))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self_attn = SelfAttention(config.latent_dim, config.num_heads)
        self.linear1 = nn.Linear(config.latent_dim, config.ff_size)
        self.linear2 = nn.Linear(config.ff_size, config.latent_dim)
        self.norm1 = nn.LayerNorm(config.latent_dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(config.latent_dim, eps=1e-5)
        self.dropout = nn.Dropout(config.dropout)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.activation = nn.gelu if config.activation == "gelu" else nn.relu
        self.norm_first = config.norm_first

    def _ff(self, x):
        return self.dropout2(
            self.linear2(self.dropout(self.activation(self.linear1(x))))
        )

    def __call__(self, x: mx.array, mask: mx.array):
        if self.norm_first:
            x = x + self.dropout1(self.self_attn(self.norm1(x), mask))
            return x + self._ff(self.norm2(x))
        x = self.norm1(x + self.dropout1(self.self_attn(x, mask)))
        return self.norm2(x + self._ff(x))


class TransformerEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.layers = [
            TransformerEncoderLayer(config) for _ in range(config.num_layers)
        ]

    def __call__(self, x: mx.array, mask: mx.array):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, config: ModelConfig, input_dim: int, output_dim: int):
        super().__init__()
        self.config = config
        self.embed_text = nn.Linear(config.llm_shape[-1], config.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(
            config.latent_dim, config.pe_dropout, config.max_position_embeddings
        )
        self.embed_timestep = TimestepEmbedder(
            config.latent_dim, self.sequence_pos_encoder
        )
        self.input_linear = nn.Linear(input_dim, config.latent_dim)
        self.output_linear = nn.Linear(config.latent_dim, output_dim)
        self.linear_first_heading_angle = nn.Linear(2, config.latent_dim)
        self.seqTransEncoder = TransformerEncoder(config)
        self.num_text_tokens = (
            config.num_text_tokens_override
            if config.num_text_tokens_override is not None
            else config.llm_shape[0]
        )

    def __call__(
        self,
        x: mx.array,
        x_pad_mask: mx.array,
        text_feat: mx.array,
        text_feat_pad_mask: mx.array,
        timesteps: mx.array,
        first_heading_angle: Optional[mx.array] = None,
    ):
        batch = x.shape[0]
        x = self.input_linear(x)
        text_feat, text_feat_pad_mask = pad_x_and_mask_to_fixed_size(
            text_feat, text_feat_pad_mask, self.num_text_tokens
        )
        emb_text = self.embed_text(text_feat)
        emb_time = self.embed_timestep(timesteps)
        prefix_feats = [emb_text, emb_time]
        text_mask = (
            text_feat_pad_mask.astype(mx.bool_)
            if self.config.use_text_mask
            else mx.ones((batch, emb_text.shape[1]), dtype=mx.bool_)
        )
        prefix_masks = [text_mask, mx.ones((batch, 1), dtype=mx.bool_)]
        if self.config.input_first_heading_angle:
            if first_heading_angle is None:
                raise ValueError(
                    "The first heading angle is required by this checkpoint"
                )
            angle_features = mx.stack(
                (mx.cos(first_heading_angle), mx.sin(first_heading_angle)), axis=-1
            )
            prefix_feats.append(
                self.linear_first_heading_angle(angle_features)[:, None]
            )
            prefix_masks.append(mx.ones((batch, 1), dtype=mx.bool_))
        prefix = mx.concatenate(prefix_feats, axis=1)
        valid = mx.concatenate((*prefix_masks, x_pad_mask.astype(mx.bool_)), axis=1)
        xseq = self.sequence_pos_encoder(mx.concatenate((prefix, x), axis=1))
        output = self.seqTransEncoder(xseq, valid[:, None, None, :])
        return self.output_linear(output[:, prefix.shape[1] :])


class TwostageDenoiser(nn.Module):
    def __init__(self, config: ModelConfig, motion_rep):
        super().__init__()
        # Motion geometry and statistics are not denoiser checkpoint parameters.
        self._motion_rep = motion_rep
        self.motion_mask_mode = config.motion_mask_mode
        dim = motion_rep.motion_rep_dim
        mask_dim = dim if config.motion_mask_mode == "concat" else 0
        self.root_model = TransformerEncoderBlock(
            config, dim + mask_dim, motion_rep.global_root_dim
        )
        self.body_model = TransformerEncoderBlock(
            config,
            dim - motion_rep.global_root_dim + motion_rep.local_root_dim + mask_dim,
            dim - motion_rep.global_root_dim,
        )

    @property
    def motion_rep(self):
        return self._motion_rep

    def __call__(
        self,
        x: mx.array,
        x_pad_mask: mx.array,
        text_feat: mx.array,
        text_feat_pad_mask: mx.array,
        timesteps: mx.array,
        first_heading_angle: Optional[mx.array] = None,
        motion_mask: Optional[mx.array] = None,
        observed_motion: Optional[mx.array] = None,
    ):
        if self.motion_mask_mode == "concat":
            if motion_mask is None or observed_motion is None:
                motion_mask = mx.zeros_like(x)
                observed_motion = mx.zeros_like(x)
            x = x * (1 - motion_mask) + observed_motion * motion_mask
            x_extended = mx.concatenate((x, motion_mask), axis=-1)
        else:
            x_extended = x
        root_motion_pred = self.root_model(
            x_extended,
            x_pad_mask,
            text_feat,
            text_feat_pad_mask,
            timesteps,
            first_heading_angle,
        )
        root_motion_local = self.motion_rep.global_root_to_local_root(
            root_motion_pred, normalized=True, lengths=mx.sum(x_pad_mask, axis=-1)
        )
        x_new = mx.concatenate(
            (root_motion_local, x[..., self.motion_rep.body_slice]), axis=-1
        )
        if self.motion_mask_mode == "concat":
            x_new = mx.concatenate((x_new, motion_mask), axis=-1)
        predicted_body = self.body_model(
            x_new,
            x_pad_mask,
            text_feat,
            text_feat_pad_mask,
            timesteps,
            first_heading_angle,
        )
        return mx.concatenate((root_motion_pred, predicted_body), axis=-1)


__all__ = [
    "PositionalEncoding",
    "TimestepEmbedder",
    "TransformerEncoderBlock",
    "TwostageDenoiser",
    "pad_x_and_mask_to_fixed_size",
]
