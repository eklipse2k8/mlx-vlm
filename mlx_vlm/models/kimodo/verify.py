"""Compare an MLX Kimodo checkpoint with the original PyTorch implementation.

Example:
    python -m mlx_vlm.models.kimodo.verify --mlx-path ./kimodo-mlx \
        --kimodo-source /path/to/kimodo
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from ...utils import get_model_path
from .kimodo import load
from .pipeline import KimodoPipeline, sanitize_text


def verify(
    mlx_path,
    kimodo_source,
    model="nvidia/Kimodo-SOMA-RP-v1.2",
    frames=8,
    steps=4,
    prompt=None,
    warmup=2,
    repeat=0,
    feature_atol=3e-3,
    output=None,
):
    """Check real weights, masked denoising, DDIM sampling, and motion decoding."""
    import torch
    from omegaconf import OmegaConf

    torch.set_num_threads(4)
    sys.path.insert(0, str(Path(kimodo_source).expanduser().resolve()))
    from kimodo import skeleton as source_skeletons
    from kimodo.model.cfg import ClassifierFreeGuidedModel
    from kimodo.model.diffusion import DDIMSampler
    from kimodo.model.diffusion import Diffusion as SourceDiffusion
    from kimodo.model.twostage_denoiser import TwostageDenoiser
    from kimodo.motion_rep import ArdyMotionRep, KimodoMotionRep

    source_path = get_model_path(
        model, allow_patterns=["*.safetensors", "*.yaml", "*.npy", "*.json"]
    )
    source_config = OmegaConf.to_container(OmegaConf.load(source_path / "config.yaml"))
    backbone_config = dict(source_config["denoiser"])
    backbone_config.pop("_target_")
    backbone_config.pop("ckpt_path", None)
    rep_config = backbone_config.pop("motion_rep")
    skeleton = getattr(
        source_skeletons, rep_config["skeleton"]["_target_"].rsplit(".", 1)[-1]
    )()
    rep_type = {"KimodoMotionRep": KimodoMotionRep, "ArdyMotionRep": ArdyMotionRep}[
        rep_config["_target_"].rsplit(".", 1)[-1]
    ]
    motion_rep = rep_type(
        skeleton, rep_config["fps"], stats_path=str(source_path / "stats" / "motion")
    )
    reference = TwostageDenoiser(
        motion_rep,
        ckpt_path=str(source_path / "model.safetensors"),
        **backbone_config,
    ).eval()
    mlx_model = load(mlx_path, lazy=True)
    mlx_model.eval()
    mx.eval(mlx_model.parameters())
    pipeline = KimodoPipeline(mlx_model)
    rng = np.random.default_rng(12)
    prompts = [sanitize_text(text) for text in prompt] if prompt else None
    batch = len(prompts) if prompts else 1
    shape = (batch, frames, motion_rep.motion_rep_dim)
    noise = rng.normal(size=shape).astype(np.float32)
    if prompts:
        features, feature_lengths = pipeline._encoder()(prompts)
        text = np.array(features.astype(mx.float32))
        text_mask = np.arange(text.shape[1])[None] < np.array(feature_lengths)[:, None]
    else:
        text = rng.normal(size=(1, 1, backbone_config["llm_shape"][-1])).astype(
            np.float32
        )
        text_mask = np.ones((1, 1), dtype=bool)
    valid = np.ones((batch, frames), dtype=bool)
    heading = np.full((batch,), 0.25, dtype=np.float32)
    observed = np.zeros(shape, dtype=np.float32)
    mask = np.zeros(shape, dtype=np.float32)
    observed[:, 0, :5] = rng.normal(size=(batch, 5)).astype(np.float32)
    mask[:, 0, :5] = 1
    report = {
        "model": model,
        "frames": frames,
        "steps": steps,
        "prompts": prompts,
        "text_features": (
            "shared native MLX LLM2Vec embeddings"
            if prompts
            else "shared random features"
        ),
        "torch_device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "mlx_device": str(mx.default_device()),
        "feature_atol": feature_atol,
    }
    failures = []

    def compare(name, actual, expected, atol=2e-4):
        actual = np.asarray(actual)
        expected = expected.detach().numpy()
        error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        report[name] = {"max_abs": float(error.max()), "mean_abs": float(error.mean())}
        try:
            np.testing.assert_allclose(
                actual, expected, rtol=2e-4, atol=atol, err_msg=name
            )
        except AssertionError:
            failures.append(name)

    torch_args = tuple(
        torch.from_numpy(value) for value in (noise, valid, text, text_mask)
    )
    mlx_args = tuple(mx.array(value) for value in (noise, valid, text, text_mask))
    started = time.perf_counter()
    with torch.inference_mode():
        expected = reference(
            *torch_args,
            torch.full((batch,), source_config["num_base_steps"] - 1),
            torch.from_numpy(heading),
            torch.from_numpy(mask),
            torch.from_numpy(observed),
        )
    report["torch_forward_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    actual = mlx_model(
        *mlx_args,
        mx.full((batch,), source_config["num_base_steps"] - 1, mx.int32),
        mx.array(heading),
        mx.array(mask),
        mx.array(observed),
    )
    mx.eval(actual)
    report["mlx_forward_seconds"] = time.perf_counter() - started
    compare("denoiser", actual, expected)

    diffusion = SourceDiffusion(source_config["num_base_steps"])
    sampler = DDIMSampler(diffusion)
    guided = ClassifierFreeGuidedModel(reference, cfg_type="separated")
    use_timesteps, mapped = diffusion.space_timesteps(steps)

    def sample_torch():
        motion = torch.from_numpy(noise)
        with torch.inference_mode():
            for index in reversed(range(steps)):
                step = torch.full((batch,), index)
                clean = guided(
                    (2.0, 2.0),
                    motion,
                    torch_args[1],
                    torch_args[2],
                    torch_args[3],
                    mapped[step],
                    torch.from_numpy(heading),
                    torch.from_numpy(mask),
                    torch.from_numpy(observed),
                )
                motion = sampler(use_timesteps, motion, clean, step)
            decoded = motion_rep.inverse(motion, is_normalized=True)
        return motion, decoded

    def sample_mlx():
        generated = pipeline.sample(
            mlx_args[2],
            mlx_args[3],
            [frames] * batch,
            steps,
            initial_noise=mx.array(noise),
            first_heading_angle=mx.array(heading),
            motion_mask=mx.array(mask),
            observed_motion=mx.array(observed),
        )
        decoded = mlx_model.motion_rep.inverse(generated, is_normalized=True)
        mx.eval(generated, decoded)
        return generated, decoded

    started = time.perf_counter()
    motion, expected_output = sample_torch()
    report["torch_sample_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    generated, decoded = sample_mlx()
    report["mlx_sample_seconds"] = time.perf_counter() - started
    compare("sample_features", generated, motion, atol=feature_atol)
    raw_error = np.abs(np.array(generated) - motion.numpy())
    report["sample_features"]["feature_max_abs"] = {
        name: float(raw_error[..., region].max())
        for name, region in motion_rep.slice_dict.items()
    }
    for key in ("local_rot_mats", "global_rot_mats", "root_positions", "posed_joints"):
        compare(key, decoded[key], expected_output[key])
    if prompts:
        report["per_prompt"] = [
            {
                "prompt": text,
                "joints_max_abs": float(
                    np.abs(
                        np.array(decoded["posed_joints"][index])
                        - expected_output["posed_joints"][index].numpy()
                    ).max()
                ),
            }
            for index, text in enumerate(prompts)
        ]
    # Isolate decoding accuracy from denoising's accumulated rounding error.
    shared_decode = mlx_model.motion_rep.inverse(
        mx.array(motion.numpy()), is_normalized=True
    )
    for key in ("local_rot_mats", "global_rot_mats", "root_positions", "posed_joints"):
        compare(
            "shared_features_" + key,
            shared_decode[key],
            expected_output[key],
            atol=1e-5,
        )
    if repeat:
        timings = {}
        for name, function in (("torch_cpu", sample_torch), ("mlx_gpu", sample_mlx)):
            for _ in range(warmup):
                function()
            seconds = []
            for _ in range(repeat):
                started = time.perf_counter()
                function()
                seconds.append(time.perf_counter() - started)
            timings[name] = {
                "seconds": seconds,
                "median_seconds": statistics.median(seconds),
            }
        report["benchmark"] = {
            "warmup": warmup,
            "repeat": repeat,
            "includes": "denoising plus compact skeleton decode; shared precomputed text embeddings",
            **timings,
        }
    serialized = json.dumps(report, indent=2)
    print(serialized)
    if output:
        Path(output).write_text(serialized + "\n")
    if failures:
        raise AssertionError("Parity tolerance exceeded: " + ", ".join(failures))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-path", required=True)
    parser.add_argument("--kimodo-source", required=True)
    parser.add_argument("--model", default="nvidia/Kimodo-SOMA-RP-v1.2")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--prompt",
        action="append",
        help="Use native text embeddings; repeat for a batch",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--repeat",
        type=int,
        default=0,
        help="Measured CPU and GPU sampling repetitions",
    )
    parser.add_argument("--feature-atol", type=float, default=3e-3)
    parser.add_argument("--output", help="Save the parity and timing report as JSON")
    args = parser.parse_args()
    verify(**vars(args))


if __name__ == "__main__":
    main()
