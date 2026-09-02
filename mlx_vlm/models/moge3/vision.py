"""DINOv2 encoder for MoGe-3 (channel-last MLX port).

Ports ``moge.model.modules.dinov2_encoder.DINOv2Encoder``: resize with
antialias to the token grid, run the shared DINOv2 backbone, project the
requested intermediate layers to ``dim_out`` and sum them. Input is an RGB
image in [0, 1]; ImageNet normalization happens inside the module.
"""

from typing import Tuple

import mlx.core as mx
import mlx.nn as nn

from ..dinov2_backbone import DINOv2
from ..interpolate import resize_bilinear_nhwc
from .config import EncoderConfig


class DINOv2Encoder(nn.Module):
    """DINOv2 backbone plus per-layer 1x1 projections summed to dim_out."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.backbone = DINOv2(config)
        self.output_projections = [
            nn.Conv2d(config.embed_dim, config.dim_out, kernel_size=1)
            for _ in config.intermediate_layers
        ]
        self.image_mean = mx.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
        self.image_std = mx.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)

    def __call__(
        self, image: mx.array, token_rows: int, token_cols: int
    ) -> Tuple[mx.array, mx.array]:
        """image: (B, H, W, 3) in [0, 1] -> features (B, rows, cols, dim_out), cls."""
        p = self.config.patch_size
        x = resize_bilinear_nhwc(
            image, (token_rows * p, token_cols * p), antialias=True
        )
        x = (x - self.image_mean) / self.image_std
        features = self.backbone.get_intermediate_layers(
            x, self.config.intermediate_layers
        )
        projected = [
            proj(feat.reshape(feat.shape[0], token_rows, token_cols, -1))
            for proj, (feat, _) in zip(self.output_projections, features)
        ]
        return sum(projected[1:], projected[0]), features[-1][1]
