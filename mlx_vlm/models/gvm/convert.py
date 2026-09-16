"""Convert a GVM diffusers checkpoint (geyongtao/gvm layout) to MLX format.

Layout produced:
    output/
        unet/model.safetensors(.index.json)   bf16, MLX conv layout
        unet/config.json
        unet/lora.safetensors                  bf16 (if present in source)
        unet/lora_config.json
        vae/model.safetensors                  bf16, MLX conv layout
        vae/config.json
        scheduler/scheduler_config.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import mlx.core as mx

from mlx_vlm.utils import (
    MODEL_CONVERSION_DTYPES,
    get_model_path,
    save_weights,
    upload_to_hub,
)

from .config import ModelConfig
from .unet import GVMUNet
from .vae import GVMVAE
from .weights import load_lora_weights, sanitize_weights


def is_gvm_model_path(model_path: str | Path) -> bool:
    root = Path(model_path)
    unet_cfg = root / "unet" / "config.json"
    vae_cfg = root / "vae" / "config.json"
    if not (unet_cfg.exists() and vae_cfg.exists()):
        return False
    try:
        unet = json.loads(unet_cfg.read_text())
        vae = json.loads(vae_cfg.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        unet.get("_class_name") == "UNetSpatioTemporalConditionModel"
        and vae.get("_class_name") == "AutoencoderKLTemporalDecoder"
    )


def _load_safetensors(directory: Path) -> dict[str, mx.array]:
    files = sorted(
        p for p in directory.glob("*.safetensors") if not p.name.startswith("._")
    )
    if not files:
        raise FileNotFoundError(f"No safetensors found in {directory}")
    weights: dict[str, mx.array] = {}
    for path in files:
        weights.update(mx.load(str(path)))
    return weights


def _cast_weights(weights: dict[str, mx.array], dtype: mx.Dtype) -> dict[str, mx.array]:
    return {
        key: value.astype(dtype) if mx.issubdtype(value.dtype, mx.floating) else value
        for key, value in weights.items()
    }


def _write_index_metadata(component_path: Path) -> None:
    index_path = component_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index.setdefault("metadata", {})["mlx_vlm_format"] = "gvm"
    index_path.write_text(json.dumps(index, indent=4, sort_keys=True) + "\n")


def _convert_component(
    name: str,
    model,
    source_dir: Path,
    output_dir: Path,
    dtype: mx.Dtype,
) -> None:
    print(f"[INFO] Converting GVM {name}")
    weights = sanitize_weights(_load_safetensors(source_dir))
    weights = _cast_weights(weights, dtype)
    model.load_weights(list(weights.items()), strict=True)

    component_path = output_dir / name
    component_path.mkdir(parents=True, exist_ok=True)
    save_weights(component_path, model, donate_weights=True)
    shutil.copy2(source_dir / "config.json", component_path / "config.json")
    _write_index_metadata(component_path)
    del weights


def _convert_lora(source_dir: Path, output_dir: Path, dtype: mx.Dtype) -> bool:
    try:
        lora = load_lora_weights(source_dir)
    except FileNotFoundError:
        return False
    lora = _cast_weights(lora, dtype)
    mx.save_safetensors(str(output_dir / "lora.safetensors"), lora)
    adapter_config = source_dir / "adapter_config.json"
    if adapter_config.exists():
        shutil.copy2(adapter_config, output_dir / "lora_config.json")
    print(f"[INFO] Converted GVM LoRA ({len(lora)} tensors)")
    return True


def convert_gvm(
    model_path: str | Path,
    output_path: str | Path,
    *,
    dtype: str | None = None,
    upload_repo: str | None = None,
) -> Path:
    source = Path(model_path).expanduser()
    destination = Path(output_path).expanduser()
    if not is_gvm_model_path(source):
        raise ValueError(f"Not a GVM diffusers checkpoint: {source}")
    if source.resolve() == destination.resolve():
        raise ValueError("GVM conversion output must differ from the source")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"GVM conversion output is not empty: {destination}")

    precision_name = dtype or "bfloat16"
    try:
        precision = getattr(mx, precision_name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported GVM dtype: {precision_name}") from exc

    config = ModelConfig.from_model_path(source)
    destination.mkdir(parents=True, exist_ok=True)

    _convert_component(
        "vae", GVMVAE(config.vae), source / "vae", destination, precision
    )
    _convert_component(
        "unet", GVMUNet(config.unet), source / "unet", destination, precision
    )
    has_lora = _convert_lora(source / "unet", destination / "unet", precision)

    scheduler_dir = destination / "scheduler"
    scheduler_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        source / "scheduler" / "scheduler_config.json",
        scheduler_dir / "scheduler_config.json",
    )

    top_config: dict[str, Any] = {
        "model_type": "gvm",
        "unet": json.loads((destination / "unet" / "config.json").read_text()),
        "vae": json.loads((destination / "vae" / "config.json").read_text()),
        "scheduler": json.loads((scheduler_dir / "scheduler_config.json").read_text()),
        "lora": config.lora.__dict__ if has_lora and config.lora else None,
    }
    (destination / "config.json").write_text(
        json.dumps(top_config, indent=2, sort_keys=True) + "\n"
    )

    if upload_repo is not None:
        upload_to_hub(destination, upload_repo)
    return destination


def convert(
    model: str,
    output_path: str | Path,
    *,
    revision: str | None = None,
    dtype: str | None = None,
    upload_repo: str | None = None,
) -> Path:
    model_path = get_model_path(model, revision=revision)
    return convert_gvm(model_path, output_path, dtype=dtype, upload_repo=upload_repo)


def configure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a GVM diffusers checkpoint to MLX format."
    )
    parser.add_argument(
        "--hf-path",
        "--model",
        dest="model",
        required=True,
        help="Local checkpoint path or Hugging Face repository ID.",
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--mlx-path",
        dest="output_path",
        default="mlx_model",
        help="Directory for the converted MLX model.",
    )
    parser.add_argument(
        "--dtype",
        choices=MODEL_CONVERSION_DTYPES,
        default="bfloat16",
        help="Floating-point dtype for converted weights (default: bfloat16).",
    )
    parser.add_argument(
        "--upload-repo",
        default=None,
        help="Hugging Face repository for the converted model.",
    )
    return parser


def main() -> None:
    args = configure_parser().parse_args()
    convert(**vars(args))


if __name__ == "__main__":
    main()


__all__ = ["configure_parser", "convert", "convert_gvm", "is_gvm_model_path"]
