"""GVM (Generative Video Matting) configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple


def _tuple(value: Any) -> Tuple:
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


@dataclass
class UNetConfig:
    """Config for the spatio-temporal UNet (SVD-XT style)."""

    sample_size: Optional[int] = None  # expected latent H/W; unused at inference
    in_channels: int = 8  # 4 noisy latent + 4 image latent channels
    out_channels: int = 4
    down_block_types: Tuple[str, ...] = (
        "CrossAttnDownBlockSpatioTemporal",
        "CrossAttnDownBlockSpatioTemporal",
        "CrossAttnDownBlockSpatioTemporal",
        "DownBlockSpatioTemporal",
    )
    up_block_types: Tuple[str, ...] = (
        "UpBlockSpatioTemporal",
        "CrossAttnUpBlockSpatioTemporal",
        "CrossAttnUpBlockSpatioTemporal",
        "CrossAttnUpBlockSpatioTemporal",
    )
    block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280)
    layers_per_block: int = 2
    cross_attention_dim: int = 1024
    transformer_layers_per_block: int = 1
    num_attention_heads: Tuple[int, ...] = (5, 10, 20, 20)
    num_frames: int = 25  # training-time frame count; not enforced
    act_fn: str = "silu"

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "UNetConfig":
        keys = cls.__dataclass_fields__
        kwargs = {k: v for k, v in params.items() if k in keys}
        for name in (
            "down_block_types",
            "up_block_types",
            "block_out_channels",
            "num_attention_heads",
        ):
            if name in kwargs:
                kwargs[name] = _tuple(kwargs[name])
        return cls(**kwargs)


@dataclass
class VAEConfig:
    """Config for the AutoencoderKLTemporalDecoder."""

    in_channels: int = 3
    out_channels: int = 3
    down_block_types: Tuple[str, ...] = ("DownEncoderBlock2D",) * 4
    block_out_channels: Tuple[int, ...] = (128, 256, 512, 512)
    layers_per_block: int = 2
    latent_channels: int = 4
    sample_size: int = 768
    scaling_factor: float = 0.18215
    force_upcast: bool = True

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "VAEConfig":
        keys = cls.__dataclass_fields__
        kwargs = {k: v for k, v in params.items() if k in keys}
        for name in ("down_block_types", "block_out_channels"):
            if name in kwargs:
                kwargs[name] = _tuple(kwargs[name])
        return cls(**kwargs)


@dataclass
class SchedulerConfig:
    """Config for the FlowMatchEulerDiscreteScheduler."""

    num_train_timesteps: int = 1000
    shift: float = 3.0
    use_dynamic_shifting: bool = False
    base_shift: float = 0.5
    max_shift: float = 1.15
    base_image_seq_len: int = 256
    max_image_seq_len: int = 4096

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "SchedulerConfig":
        keys = cls.__dataclass_fields__
        return cls(**{k: v for k, v in params.items() if k in keys})


@dataclass
class LoRAConfig:
    """LoRA adapter config applied on top of the UNet."""

    rank: int = 4
    alpha: float = 2.0
    target_modules: Tuple[str, ...] = ("to_k", "to_q", "to_v", "conv_in", "conv_out")

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "LoRAConfig":
        return cls(
            rank=int(params.get("r", 4)),
            alpha=float(params.get("lora_alpha", 2.0)),
            target_modules=_tuple(
                params.get(
                    "target_modules", ("to_k", "to_q", "to_v", "conv_in", "conv_out")
                )
            ),
        )


@dataclass
class ModelConfig:
    """Top-level GVM config; weights dir holds unet/, vae/, scheduler/."""

    model_type: str = "gvm"
    unet: UNetConfig = field(default_factory=UNetConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    lora: Optional[LoRAConfig] = None  # optional UNet LoRA adapter

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "ModelConfig":
        return cls(
            model_type=params.get("model_type", "gvm"),
            unet=UNetConfig.from_dict(params.get("unet", {})),
            vae=VAEConfig.from_dict(params.get("vae", {})),
            scheduler=SchedulerConfig.from_dict(params.get("scheduler", {})),
            lora=(LoRAConfig.from_dict(params["lora"]) if params.get("lora") else None),
        )

    @classmethod
    def from_model_path(cls, model_path: str | Path) -> "ModelConfig":
        root = Path(model_path).expanduser()
        unet_cfg = json.loads((root / "unet" / "config.json").read_text())
        vae_cfg = json.loads((root / "vae" / "config.json").read_text())
        sched_cfg = json.loads(
            (root / "scheduler" / "scheduler_config.json").read_text()
        )
        lora = None
        adapter_path = root / "unet" / "adapter_config.json"
        lora_path = root / "unet" / "lora_config.json"
        if lora_path.exists():
            lora = LoRAConfig.from_dict(json.loads(lora_path.read_text()))
        elif adapter_path.exists():
            lora = LoRAConfig.from_dict(json.loads(adapter_path.read_text()))
        return cls(
            unet=UNetConfig.from_dict(unet_cfg),
            vae=VAEConfig.from_dict(vae_cfg),
            scheduler=SchedulerConfig.from_dict(sched_cfg),
            lora=lora,
        )
