"""Configuration for Kimodo's two-stage motion diffusion model."""

from dataclasses import dataclass, fields
from typing import Any, Optional

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "kimodo"
    model_path: Optional[str] = None  # Local skeleton and statistics asset directory.
    skeleton: str = "SOMASkeleton30"  # Skeleton class in the source checkpoint.
    motion_rep: str = "KimodoMotionRep"  # Global and local root feature layout.
    fps: float = 30.0  # Motion sampling rate.
    motion_mask_mode: Optional[str] = "concat"  # Concatenate observed feature masks.
    llm_shape: tuple[int, int] = (1, 4096)  # Number and width of text features.
    use_text_mask: bool = False  # Preserve the official unmasked text prefix.
    latent_dim: int = 1024  # Transformer hidden width.
    ff_size: int = 2048  # Feed-forward hidden width.
    num_layers: int = 16  # Transformer layers per motion stage.
    num_heads: int = 8  # Attention heads per motion stage.
    activation: str = "gelu"  # PyTorch TransformerEncoderLayer activation.
    dropout: float = 0.0  # Transformer dropout, disabled for inference.
    pe_dropout: float = 0.0  # Position encoding dropout.
    norm_first: bool = False  # Use the reference post-norm transformer.
    num_text_tokens_override: Optional[int] = 50  # Fixed text prefix length.
    input_first_heading_angle: bool = True  # Condition on initial heading.
    max_position_embeddings: int = 5000  # Shared position and timestep table size.
    num_base_steps: int = 1000  # Training diffusion timesteps.
    cfg_type: str = "separated"  # Independent text and motion guidance.

    def __post_init__(self):
        self.llm_shape = tuple(self.llm_shape)
        if len(self.llm_shape) != 2 or min(self.llm_shape) <= 0:
            raise ValueError("llm_shape must contain two positive dimensions")
        if self.latent_dim <= 0 or self.num_heads <= 0:
            raise ValueError("latent_dim and num_heads must be positive")
        if self.latent_dim % self.num_heads:
            raise ValueError("latent_dim must be divisible by num_heads")
        if self.latent_dim % 2:
            raise ValueError("latent_dim must be even for sinusoidal positions")
        if self.num_layers < 1 or self.ff_size < 1:
            raise ValueError("num_layers and ff_size must be positive")
        if self.activation not in ("gelu", "relu"):
            raise ValueError("activation must be 'gelu' or 'relu'")
        if (
            self.num_text_tokens_override is not None
            and self.num_text_tokens_override < 1
        ):
            raise ValueError("num_text_tokens_override must be positive")
        if self.max_position_embeddings < 1 or self.num_base_steps < 1:
            raise ValueError("Position and diffusion step counts must be positive")
        if self.num_base_steps > self.max_position_embeddings:
            raise ValueError("The position table must cover all diffusion timesteps")
        if self.motion_mask_mode not in (None, "none", "concat"):
            raise ValueError("motion_mask_mode must be 'concat', 'none', or None")
        if not 0 <= self.dropout < 1 or not 0 <= self.pe_dropout < 1:
            raise ValueError("Dropout probabilities must be in [0, 1)")

    @classmethod
    def from_dict(cls, params: Optional[dict[str, Any]]):
        """Read MLX JSON or the original nested Hydra configuration."""
        data = dict(params or {})
        denoiser = data.pop("denoiser", None)
        if denoiser is not None:
            data.update(denoiser)
        motion_rep = data.get("motion_rep")
        if isinstance(motion_rep, dict):
            data["motion_rep"] = motion_rep.get("_target_", "KimodoMotionRep").rsplit(
                ".", 1
            )[-1]
            data["fps"] = motion_rep.get("fps", 30.0)
            skeleton = motion_rep.get("skeleton", {})
            data["skeleton"] = (
                skeleton.get("_target_", "SOMASkeleton30").rsplit(".", 1)[-1]
                if isinstance(skeleton, dict)
                else skeleton
            )
        allowed = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in allowed})


__all__ = ["ModelConfig"]
