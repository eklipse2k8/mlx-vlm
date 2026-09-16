"""GVM video matting inference entry point (port of the upstream `demo.py`).

Reads a video (mp4 via PyAV, or an image-sequence directory), runs the
windowed one-step diffusion matting pipeline, and writes the alpha matte.

Usage:
    uv run python -m mlx_vlm.models.gvm.generate \
        --model-path data/weights-mlx-bf16 \
        --data-dir data/demo_videos/xxx.mp4 \
        --output-dir output/
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Optional

import mlx.core as mx
import numpy as np
from PIL import Image

from .weights import load_pipeline

UPPER_BOUND = 240.0 / 255.0
LOWER_BOUND = 25.0 / 255.0


def resize_max(frame: Image.Image, size: int, max_size: int) -> Image.Image:
    """torchvision Resize(size=size, max_size=max_size) equivalent."""
    w, h = frame.size
    smaller, larger = (h, w) if h < w else (w, h)
    new_smaller = size
    new_larger = int(larger * size / smaller)
    if new_larger > max_size:
        new_smaller = int(smaller * max_size / larger)
        new_larger = max_size
    if h < w:
        new_h, new_w = new_smaller, new_larger
    else:
        new_h, new_w = new_larger, new_smaller
    return frame.resize((new_w, new_h), Image.BILINEAR)


def impad_multi(frames: np.ndarray, multiple: int = 32) -> tuple[np.ndarray, tuple]:
    """Pad (N, H, W, C) to a multiple of `multiple`, centered. Returns pads."""
    _, h, w, _ = frames.shape
    target_h = int(math.ceil(h / multiple) * multiple)
    target_w = int(math.ceil(w / multiple) * multiple)
    pad_top = (target_h - h) // 2
    pad_left = (target_w - w) // 2
    padded = np.zeros(
        (frames.shape[0], target_h, target_w, frames.shape[3]), dtype=frames.dtype
    )
    padded[:, pad_top : pad_top + h, pad_left : pad_left + w] = frames
    return padded, (pad_top, pad_left, target_h - h - pad_top, target_w - w - pad_left)


def read_video(path: str, max_frames: Optional[int] = None) -> tuple[np.ndarray, float]:
    """Read frames as (N, H, W, 3) float32 in [0, 1]. Returns (frames, fps)."""
    if os.path.isdir(path):
        files = sorted(
            f for f in os.listdir(path) if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        frames = [
            np.asarray(
                Image.open(os.path.join(path, f)).convert("RGB"), dtype=np.float32
            )
            / 255.0
            for f in files
        ]
        return np.stack(frames), 30.0
    try:
        import av
    except ImportError as exc:
        raise ImportError(
            "Reading video files requires PyAV (`pip install av`)"
        ) from exc
    container = av.open(path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate)
    frames = []
    for frame in container.decode(stream):
        frames.append(frame.to_ndarray(format="rgb24"))
        if max_frames is not None and len(frames) >= max_frames:
            break
    container.close()
    return np.stack(frames).astype(np.float32) / 255.0, fps


def write_output(alpha: np.ndarray, output_dir: str, fps: float) -> None:
    """Write alpha (N, H, W) in [0, 1] as PNG sequence and, if PyAV exists, mp4."""
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(alpha):
        Image.fromarray((frame * 255).astype(np.uint8)).save(
            os.path.join(output_dir, f"{i:04d}.png")
        )
    try:
        import av
    except ImportError:
        return
    out_path = os.path.join(output_dir, "alpha.mp4")
    container = av.open(out_path, mode="w")
    stream = container.add_stream("h264", rate=f"{fps:.4f}")
    stream.pix_fmt = "yuv420p"
    stream.width = alpha.shape[2]
    stream.height = alpha.shape[1]
    for frame in alpha:
        rgb = np.repeat((frame * 255).astype(np.uint8)[..., None], 3, axis=-1)
        container.mux(stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")))
    container.mux(stream.encode())
    container.close()


def matte_video(
    model_path: str,
    frames: np.ndarray,
    fps: float = 30.0,
    *,
    num_frames_per_batch: int = 8,
    num_overlap_frames: int = 1,
    denoise_steps: int = 1,
    decode_chunk_size: int = 8,
    size: int = 720,
    max_resolution: int = 960,
    noise_type: str = "zeros",
    progress: bool = True,
) -> np.ndarray:
    """Run GVM matting on (N, H, W, 3) float32 frames in [0, 1].

    Returns the alpha matte (N, H, W) float32 in [0, 1] at input resolution.
    """
    pipe = load_pipeline(model_path)
    origin_h, origin_w = frames.shape[1:3]

    resized = np.stack(
        [
            np.asarray(
                resize_max(
                    Image.fromarray((f * 255).astype(np.uint8)), size, max_resolution
                ),
                dtype=np.float32,
            )
            / 255.0
            for f in frames
        ]
    )

    alphas = []
    for start in range(0, len(resized), num_frames_per_batch):
        batch = resized[start : start + num_frames_per_batch]
        batch, pad = impad_multi(batch)
        out = pipe(
            mx.array(batch),
            num_frames=num_frames_per_batch,
            num_overlap_frames=num_overlap_frames,
            decode_chunk_size=decode_chunk_size,
            num_inference_steps=denoise_steps,
            noise_type=noise_type,
        )
        alpha = np.array(out.alpha.astype(mx.float32))  # (n, Hp, Wp, 1)
        pt, pl, pb, pr = pad
        alpha = alpha[:, pt : alpha.shape[1] - pb, pl : alpha.shape[2] - pr, 0]
        # back to the original resolution
        alpha = np.stack(
            [
                np.asarray(
                    Image.fromarray((a * 255).astype(np.uint8)).resize(
                        (origin_w, origin_h), Image.BILINEAR
                    ),
                    dtype=np.float32,
                )
                / 255.0
                for a in alpha
            ]
        )
        alpha = np.clip(alpha, 0, 1)
        alpha[alpha >= UPPER_BOUND] = 1.0
        alpha[alpha <= LOWER_BOUND] = 0.0
        alphas.append(alpha)
        if progress:
            print(f"[INFO] frames {start}..{start + len(batch) - 1} done", flush=True)
    return np.concatenate(alphas, axis=0)


def configure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GVM video matting (MLX).")
    parser.add_argument("--model-path", required=True, help="Converted MLX model dir")
    parser.add_argument(
        "--data-dir", required=True, help="Input video file or image-sequence dir"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-frames-per-batch", type=int, default=8)
    parser.add_argument("--num-overlap-frames", type=int, default=1)
    parser.add_argument("--denoise-steps", type=int, default=1)
    parser.add_argument("--decode-chunk-size", type=int, default=8)
    parser.add_argument("--size", type=int, default=720)
    parser.add_argument("--max-resolution", type=int, default=960)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--noise-type", choices=("zeros", "gaussian"), default="zeros")
    return parser


def main() -> None:
    args = configure_parser().parse_args()
    frames, fps = read_video(args.data_dir, max_frames=args.max_frames)
    print(
        f"[INFO] {len(frames)} frames ({frames.shape[2]}x{frames.shape[1]}) @ {fps:.2f} fps"
    )
    alpha = matte_video(
        args.model_path,
        frames,
        fps,
        num_frames_per_batch=args.num_frames_per_batch,
        num_overlap_frames=args.num_overlap_frames,
        denoise_steps=args.denoise_steps,
        decode_chunk_size=args.decode_chunk_size,
        size=args.size,
        max_resolution=args.max_resolution,
        noise_type=args.noise_type,
    )
    write_output(alpha, args.output_dir, fps)
    print(f"[INFO] wrote alpha matte to {args.output_dir}")


if __name__ == "__main__":
    main()
