"""Shared DINOv2 vision transformer (channel-last MLX port).

Used as the frozen backbone of dense-prediction models (Video Depth
Anything, MoGe-3). Parameter names match the official ``dinov2`` checkpoints
(``patch_embed.proj``, ``cls_token``, ``pos_embed``, ``blocks.N.*``,
``norm``), so weights load without renaming.

The ``config`` object only needs these attributes: ``embed_dim``, ``depth``,
``num_heads``, ``img_size``, ``patch_size``, ``mlp_ratio``,
``layer_norm_eps``, ``interpolate_offset``, and optionally ``ffn``
(``"mlp"`` or ``"swiglu"``, the latter used by the ViT-g preset).
``DINOV2_PRESETS`` holds those architecture parameters for the official
ViT-*/14 checkpoints so model configs can resolve them from a variant name.
"""

import math
from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn

from .interpolate import resize_bicubic_nhwc

# Architecture presets for the official DINOv2 ViT-*/14 (LVD-142M) checkpoints.
DINOV2_PRESETS = {
    "vits14": dict(embed_dim=384, depth=12, num_heads=6, ffn="mlp"),
    "vitb14": dict(embed_dim=768, depth=12, num_heads=12, ffn="mlp"),
    "vitl14": dict(embed_dim=1024, depth=24, num_heads=16, ffn="mlp"),
    "vitg14": dict(embed_dim=1536, depth=40, num_heads=24, ffn="swiglu"),
}


class PatchEmbed(nn.Module):
    """2D image to patch embedding: (B, H, W, C) -> (B, N, D)."""

    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def __call__(self, x: mx.array) -> mx.array:
        B, H, W, _ = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        x = self.proj(x)  # (B, H/p, W/p, D)
        return x.reshape(B, -1, x.shape[-1])


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        x = x.transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class SwiGLUFFN(nn.Module):
    """Fused SwiGLU FFN (dinov2 vit-g): hidden dim rounded up to a multiple of 8."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        hidden_dim = (int(hidden_dim * 2 / 3) + 7) // 8 * 8
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=True)
        self.w3 = nn.Linear(hidden_dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x1, x2 = mx.split(self.w12(x), 2, axis=-1)
        return self.w3(nn.silu(x1) * x2)


_FFN_LAYERS = {"mlp": Mlp, "swiglu": SwiGLUFFN}


class LayerScale(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.gamma


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.embed_dim
        hidden = int(dim * config.mlp_ratio)
        self.norm1 = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.attn = Attention(dim, config.num_heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        ffn = getattr(config, "ffn", None) or "mlp"
        if ffn not in _FFN_LAYERS:
            raise ValueError(
                f"Unknown DINOv2 ffn {ffn!r}; expected one of {list(_FFN_LAYERS)}"
            )
        self.mlp = _FFN_LAYERS[ffn](dim, hidden)
        self.ls2 = LayerScale(dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DINOv2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.patch_size = config.patch_size
        self.patch_embed = PatchEmbed(
            config.img_size, config.patch_size, 3, config.embed_dim
        )
        self.cls_token = mx.zeros((1, 1, config.embed_dim))
        self.pos_embed = mx.zeros(
            (1, self.patch_embed.num_patches + 1, config.embed_dim)
        )
        self.mask_token = mx.zeros((1, config.embed_dim))
        self.blocks = [Block(config) for _ in range(config.depth)]
        self.norm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)

    def interpolate_pos_encoding(self, x: mx.array, h: int, w: int) -> mx.array:
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        dim = x.shape[-1]
        class_pos_embed = self.pos_embed[:, :1]
        patch_pos_embed = self.pos_embed[:, 1:]
        # Small offset to avoid floating point error in the interpolation
        w0 = w // self.patch_size + self.config.interpolate_offset
        h0 = h // self.patch_size + self.config.interpolate_offset
        sqrt_N = math.sqrt(N)
        # (sy, sx) because the reference derives w0 from the pixel height and
        # applies it to the W axis.
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        patch_pos_embed = patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim)
        patch_pos_embed = resize_bicubic_nhwc(
            patch_pos_embed.astype(mx.float32), sy, sx
        )
        patch_pos_embed = patch_pos_embed.reshape(1, -1, dim)
        return mx.concatenate([class_pos_embed, patch_pos_embed], axis=1).astype(
            x.dtype
        )

    def prepare_tokens(self, x: mx.array) -> mx.array:
        B, H, W, _ = x.shape
        x = self.patch_embed(x)
        cls = mx.broadcast_to(self.cls_token, (B, 1, self.embed_dim))
        x = mx.concatenate([cls, x], axis=1)
        return x + self.interpolate_pos_encoding(x, H, W)

    def get_intermediate_layers(
        self, x: mx.array, indices: List[int]
    ) -> List[Tuple[mx.array, mx.array]]:
        """Run the backbone and return (patch_tokens, cls_token) per index."""
        x = self.prepare_tokens(x)
        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in indices:
                out = self.norm(x)
                outputs.append((out[:, 1:], out[:, 0]))
        assert len(outputs) == len(indices)
        return outputs
