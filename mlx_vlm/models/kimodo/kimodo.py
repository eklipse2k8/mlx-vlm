"""Kimodo motion diffusion model and MLX-VLM checkpoint interface."""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .backbone import TwostageDenoiser
from .config import ModelConfig


class Model(nn.Module):
    def __init__(self, config: ModelConfig, motion_rep=None):
        super().__init__()
        self.config = config
        if motion_rep is None:
            from .motion import ArdyMotionRep, KimodoMotionRep, Skeleton

            if config.model_path is None:
                raise ValueError(
                    "Kimodo needs model_path with converted skeleton/statistics assets"
                )
            assets = Path(config.model_path) / "assets"
            skeleton = Skeleton(**json.loads((assets / "skeleton.json").read_text()))
            stats = mx.load(str(assets / "motion_stats.safetensors"))
            rep_class = {
                "KimodoMotionRep": KimodoMotionRep,
                "ArdyMotionRep": ArdyMotionRep,
            }.get(config.motion_rep)
            if rep_class is None:
                raise ValueError(
                    f"Unknown Kimodo motion representation: {config.motion_rep}"
                )
            motion_rep = rep_class(skeleton, fps=config.fps, stats=stats)
        self.motion_rep = motion_rep
        self.denoiser = TwostageDenoiser(config, motion_rep)

    def __call__(self, *args, **kwargs):
        return self.denoiser(*args, **kwargs)

    def generate(
        self,
        prompts,
        num_frames,
        num_denoising_steps=100,
        *,
        text_encoder=None,
        **kwargs,
    ):
        from .pipeline import KimodoPipeline

        if not hasattr(self, "_pipeline") or text_encoder is not None:
            self._pipeline = KimodoPipeline(self, text_encoder=text_encoder)
        return self._pipeline.generate(
            prompts, num_frames, num_denoising_steps, **kwargs
        )

    @staticmethod
    def sanitize(weights):
        result = {}
        for name, value in weights.items():
            name = name.removeprefix("denoiser.backbone.")
            if name.startswith(("root_model.", "body_model.")):
                name = "denoiser." + name
            result[name] = value
        return result


def load(path_or_repo, *, lazy=True, **kwargs):
    """Load converted Kimodo weights using the shared MLX-VLM loader."""
    from ...utils import get_model_path, load_model

    path = get_model_path(path_or_repo)
    return load_model(path, lazy=lazy, **kwargs)
