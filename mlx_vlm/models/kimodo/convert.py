"""Convert official Kimodo safetensors and skeleton assets to MLX-VLM."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx

from ...utils import get_model_path, save_weights
from .config import ModelConfig
from .kimodo import Model


def skeleton_metadata(skeleton):
    """Export source skeleton buffers once; inference never imports PyTorch."""
    data = {
        "name": skeleton.name,
        "neutral_joints": skeleton.neutral_joints.tolist(),
        "joint_parents": skeleton.joint_parents.tolist(),
        "joint_names": skeleton.bone_order_names,
        "hip_joint_idx": skeleton.hip_joint_idx,
        "left_foot_joint_idx": skeleton.left_foot_joint_idx,
        "right_foot_joint_idx": skeleton.right_foot_joint_idx,
        "left_hand_joint_idx": [
            skeleton.bone_index[name] for name in skeleton.left_hand_joint_names
        ],
        "right_hand_joint_idx": [
            skeleton.bone_index[name] for name in skeleton.right_hand_joint_names
        ],
    }
    if hasattr(skeleton, "rest_local_rots"):
        data["rest_local_rots"] = skeleton.rest_local_rots.tolist()
    if skeleton.name == "somaskel30":
        output = skeleton.somaskel77
        data.update(
            output_skeleton=skeleton_metadata(output),
            output_joint_indices=[
                output.bone_index[name] for name in skeleton.bone_order_names
            ],
            output_rest_local_rots=output.relaxed_hands_rest_pose.tolist(),
        )
    return data


def convert(
    model,
    output_path,
    *,
    kimodo_source,
    dtype="float32",
    text_encoder=None,
    text_base=None,
    text_adapter="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
):
    from omegaconf import OmegaConf

    if dtype not in ("float32", "float16", "bfloat16"):
        raise ValueError("dtype must be float32, float16 or bfloat16")
    source = get_model_path(
        model, allow_patterns=["*.safetensors", "*.yaml", "*.npy", "*.json"]
    )
    destination = Path(output_path).expanduser()
    if source.resolve() == destination.resolve():
        raise ValueError("Conversion output must differ from its source")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Conversion output is not empty: {destination}")
    config = ModelConfig.from_dict(
        OmegaConf.to_container(OmegaConf.load(source / "config.yaml"))
    )
    source_repo = Path(kimodo_source).expanduser().resolve()
    if not (source_repo / "kimodo" / "skeleton").is_dir():
        raise FileNotFoundError(f"No Kimodo source package at {source_repo}")
    # Only conversion needs the source package and its asset loaders.
    sys.path.insert(0, str(source_repo))
    try:
        from kimodo import skeleton as source_skeletons
        from kimodo.motion_rep import ArdyMotionRep as SourceArdyRep
        from kimodo.motion_rep import KimodoMotionRep as SourceKimodoRep

        skeleton_type = getattr(source_skeletons, config.skeleton)
        skeleton = skeleton_type()
        rep_type = {"KimodoMotionRep": SourceKimodoRep, "ArdyMotionRep": SourceArdyRep}[
            config.motion_rep
        ]
        reference = rep_type(
            skeleton, config.fps, stats_path=str(source / "stats" / "motion")
        )
        stats = {
            f"{name}.{stat}": mx.array(
                getattr(getattr(reference, name + "_stats"), stat).numpy()
            )
            for name in ("global_root", "local_root", "body")
            for stat in ("mean", "std")
        }
        metadata = skeleton_metadata(skeleton)
    finally:
        sys.path.remove(str(source_repo))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=f".{destination.name}-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary)
        assets = staging / "assets"
        assets.mkdir()
        (assets / "skeleton.json").write_text(json.dumps(metadata) + "\n")
        mx.save_safetensors(str(assets / "motion_stats.safetensors"), stats)
        config.model_path = str(staging)
        converted = Model(config)
        weights = {}
        for file in sorted(source.glob("*.safetensors")):
            weights.update(mx.load(str(file)))
        if not weights:
            raise FileNotFoundError(
                f"No safetensors in {source}; convert the source weights first"
            )
        weights = {
            name: (
                value.astype(getattr(mx, dtype))
                if mx.issubdtype(value.dtype, mx.floating)
                else value
            )
            for name, value in converted.sanitize(weights).items()
        }
        converted.load_weights(list(weights.items()), strict=True)
        save_weights(staging, converted)
        if text_encoder:
            from .language import LLM2VecEncoder

            encoder = LLM2VecEncoder.from_pretrained(
                text_encoder,
                text_adapter,
                base_model_override=text_base,
            )
            encoder.save_pretrained(staging / "text_encoder")
        data = asdict(config)
        data.pop("model_path")
        data["torch_dtype"] = dtype
        data["source_model"] = str(model)
        (staging / "config.json").write_text(json.dumps(data, indent=2) + "\n")
        if destination.exists():
            destination.rmdir()
        staging.rename(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", "--hf-path", default="nvidia/Kimodo-SOMA-RP-v1.2")
    parser.add_argument("--mlx-path", dest="output_path", required=True)
    parser.add_argument("--kimodo-source", required=True)
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    parser.add_argument(
        "--text-encoder", help="LLM2Vec MNTP checkpoint; also convert its text encoder"
    )
    parser.add_argument(
        "--text-base", help="Local or Hugging Face Llama base, including MLX exports"
    )
    parser.add_argument(
        "--text-adapter",
        default="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
    )
    args = parser.parse_args()
    print(convert(**vars(args)))


if __name__ == "__main__":
    main()
