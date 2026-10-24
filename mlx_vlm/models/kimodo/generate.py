"""Generate Kimodo motion and save an NPZ at the explicit output boundary."""

import argparse
from pathlib import Path
from time import perf_counter

import mlx.core as mx

from .kimodo import load
from .pipeline import KimodoPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", nargs="+", required=True)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile denoising and cleanup for repeated inference (slower first call)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Sample count (default: one per prompt, or one sequential motion)",
    )
    parser.add_argument(
        "--multi-prompt",
        action="store_true",
        help="Join prompts into sequential segments",
    )
    parser.add_argument("--transition-frames", type=int, default=5)
    parser.add_argument("--constraints", help="Kimodo constraint JSON file")
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="Apply native GPU trajectory and contact cleanup",
    )
    parser.add_argument("--root-margin", type=float, default=0.04)
    parser.add_argument("--contact-threshold", type=float, default=0.5)
    parser.add_argument(
        "--cfg-type", choices=("nocfg", "regular", "separated"), default="separated"
    )
    parser.add_argument("--cfg-weight", nargs="+", type=float, default=[2.0, 2.0])
    parser.add_argument("--output", default="motion.npz")
    args = parser.parse_args()
    if not mx.metal.is_available():
        parser.error("Kimodo generation requires an Apple GPU")
    if args.duration <= 0 or (args.num_samples is not None and args.num_samples < 1):
        parser.error("duration and num-samples must be positive")
    num_samples = args.num_samples
    if num_samples is None:
        num_samples = 1 if args.multi_prompt else len(args.prompt)
    mx.set_default_device(mx.gpu)
    start = perf_counter()
    model = load(args.model, lazy=True)
    pipeline = KimodoPipeline(model)
    guidance = (
        args.cfg_weight[0] if args.cfg_type == "regular" else tuple(args.cfg_weight)
    )
    result = pipeline.generate(
        args.prompt[0] if len(args.prompt) == 1 else args.prompt,
        max(1, round(args.duration * model.config.fps)),
        args.steps,
        num_samples=num_samples,
        seed=args.seed,
        cfg_type=args.cfg_type,
        cfg_weight=guidance,
        multi_prompt=args.multi_prompt,
        num_transition_frames=args.transition_frames,
        constraint_lst=args.constraints,
        post_processing=args.postprocess,
        root_margin=args.root_margin,
        contact_threshold=args.contact_threshold,
        compile=args.compile,
    )
    # Saving is the only synchronization and host transfer in generation.
    import numpy as np

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **{name: np.asarray(value) for name, value in result.items()})
    print(
        f"Saved {output} in {perf_counter() - start:.2f}s; peak GPU memory {mx.get_peak_memory() / 1e9:.2f} GB"
    )


if __name__ == "__main__":
    main()
