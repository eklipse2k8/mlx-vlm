"""Weight loading/sanitizing for GVM checkpoints (diffusers layout and MLX layout)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_unflatten

from .config import ModelConfig
from .layers import LoRAConv2d, LoRALinear
from .pipeline import GVMPipeline
from .scheduler import FlowMatchEulerDiscreteScheduler
from .unet import GVMUNet
from .vae import GVMVAE


def _load_safetensors(directory: Path) -> tuple[dict[str, mx.array], dict[str, Any]]:
    skip = {"lora.safetensors", "pytorch_lora_weights.safetensors"}
    files = sorted(
        p
        for p in directory.glob("*.safetensors")
        if not p.name.startswith("._") and p.name not in skip
    )
    if not files:
        raise FileNotFoundError(f"No safetensors found in {directory}")
    weights: dict[str, mx.array] = {}
    for path in files:
        weights.update(mx.load(str(path)))
    index_path = directory / "model.safetensors.index.json"
    metadata = (
        json.loads(index_path.read_text()).get("metadata", {})
        if index_path.exists()
        else {}
    )
    return weights, metadata


def sanitize_weights(
    weights: dict[str, mx.array], source_layout: Optional[bool] = None
) -> dict[str, mx.array]:
    """Transpose torch conv weights (OIHW/OIDHW) to MLX layout (OHWI/ODHWI).

    `source_layout=False` marks an already-converted MLX checkpoint.
    """
    if source_layout is False:
        return weights
    sanitized = {}
    for key, value in weights.items():
        if key.endswith(".weight") and value.ndim == 4:
            value = value.transpose(0, 2, 3, 1)
        elif key.endswith(".weight") and value.ndim == 5:
            value = value.transpose(0, 2, 3, 4, 1)
        sanitized[key] = value
    return sanitized


def _apply_weights(model: nn.Module, weights: dict[str, mx.array]) -> nn.Module:
    model.update(tree_unflatten(list(weights.items())), strict=True)
    return model


def load_unet(model_path: str | Path, config: Optional[ModelConfig] = None) -> GVMUNet:
    root = Path(model_path).expanduser()
    config = config or ModelConfig.from_model_path(root)
    unet = GVMUNet(config.unet)
    weights, metadata = _load_safetensors(root / "unet")
    weights = sanitize_weights(
        weights,
        source_layout=(False if metadata.get("mlx_vlm_format") == "gvm" else None),
    )
    _apply_weights(unet, weights)
    if config.lora is not None:
        apply_lora(unet, root, config.lora)
    return unet


def load_vae(model_path: str | Path, config: Optional[ModelConfig] = None) -> GVMVAE:
    root = Path(model_path).expanduser()
    config = config or ModelConfig.from_model_path(root)
    vae = GVMVAE(config.vae)
    weights, metadata = _load_safetensors(root / "vae")
    weights = sanitize_weights(
        weights,
        source_layout=(False if metadata.get("mlx_vlm_format") == "gvm" else None),
    )
    return _apply_weights(vae, weights)


def load_lora_weights(unet_dir: Path) -> dict[str, mx.array]:
    """Load LoRA weights from `lora.safetensors`, `pytorch_lora_weights.safetensors`
    or `pytorch_lora_weights.pt` (torch required for the .pt format)."""
    for name in ("lora.safetensors", "pytorch_lora_weights.safetensors"):
        path = unet_dir / name
        if path.exists():
            return mx.load(str(path))
    pt_path = unet_dir / "pytorch_lora_weights.pt"
    if pt_path.exists():
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "Loading pytorch_lora_weights.pt requires torch; "
                "convert it to safetensors first"
            ) from exc
        state = torch.load(pt_path, map_location="cpu")
        return {key: mx.array(value.float().numpy()) for key, value in state.items()}
    raise FileNotFoundError(f"No LoRA weights found in {unet_dir}")


def apply_lora(unet: GVMUNet, model_path: str | Path, lora_config) -> GVMUNet:
    """Wrap target UNet modules with LoRA adapters, in place."""
    root = Path(model_path).expanduser()
    raw = load_lora_weights(root / "unet")
    # Strip the peft "model." / "base_model.model." prefixes
    raw = {
        key.split("model.", 1)[-1] if "model." in key else key: value
        for key, value in raw.items()
    }

    modules = dict(unet.named_modules())
    targets = set(lora_config.target_modules)
    for path in sorted(modules):
        leaf = path.rsplit(".", 1)[-1]
        if leaf not in targets:
            continue
        a_key, b_key = f"{path}.lora_A.weight", f"{path}.lora_B.weight"
        if a_key not in raw or b_key not in raw:
            continue
        module = modules[path]
        if isinstance(module, nn.Linear):
            wrapper = LoRALinear(module, lora_config.rank, lora_config.scale)
            wrapper.lora_a = raw[a_key]
            wrapper.lora_b = raw[b_key]
        elif isinstance(module, nn.Conv2d):
            wrapper = LoRAConv2d(module, lora_config.rank, lora_config.scale)
            wrapper.lora_a.weight = raw[a_key].transpose(0, 2, 3, 1)
            wrapper.lora_b.weight = raw[b_key].transpose(0, 2, 3, 1)
        else:
            continue
        _set_module(unet, path, wrapper)
    return unet


def _set_module(root: nn.Module, path: str, module: nn.Module) -> None:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    leaf = parts[-1]
    if leaf.isdigit():
        parent[int(leaf)] = module
    else:
        setattr(parent, leaf, module)


def load_pipeline(model_path: str | Path) -> GVMPipeline:
    """Load the full GVM pipeline (UNet + LoRA, temporal VAE, scheduler)."""
    root = Path(model_path).expanduser()
    config = ModelConfig.from_model_path(root)
    vae = load_vae(root, config)
    unet = load_unet(root, config)
    scheduler = FlowMatchEulerDiscreteScheduler(config.scheduler)
    return GVMPipeline(vae=vae, unet=unet, scheduler=scheduler)


__all__ = [
    "apply_lora",
    "load_lora_weights",
    "load_pipeline",
    "load_unet",
    "load_vae",
    "sanitize_weights",
]
