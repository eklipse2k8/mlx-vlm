# GVM — Generative Video Matting (MLX)

MLX port of [GVM](https://github.com/aim-uofa/GVM) (SIGGRAPH 2025), a one-step
diffusion video matting model built on the Stable Video Diffusion architecture:
a spatio-temporal UNet (`UNetSpatioTemporalConditionModel`), a temporal VAE
(`AutoencoderKLTemporalDecoder`), a `FlowMatchEulerDiscreteScheduler`, and a
rank-4 LoRA adapter on the UNet (`to_q`/`to_k`/`to_v`, `conv_in`, `conv_out`).

## Weights

- Source checkpoint: [`geyongtao/gvm`](https://huggingface.co/geyongtao/gvm)
  (diffusers layout: `unet/`, `vae/`, `scheduler/` + `pytorch_lora_weights`).

Convert to MLX bf16:

```bash
uv run --with torch python -m mlx_vlm.models.gvm.convert \
  --hf-path data/weights \
  --mlx-path data/weights-mlx-bf16 \
  --dtype bfloat16
```

(`torch` is only needed when the LoRA adapter ships as a `.pt` pickle; the
`safetensors` variant is read directly.)

## Inference

```python
from mlx_vlm.models.gvm.weights import load_pipeline
from mlx_vlm.models.gvm.generate import matte_video

alpha = matte_video("data/weights-mlx-bf16", frames)  # frames: (N, H, W, 3) float32 [0, 1]
```

or from the CLI (video file needs PyAV; image-sequence dirs work out of the box):

```bash
uv run python -m mlx_vlm.models.gvm.generate \
  --model-path data/weights-mlx-bf16 \
  --data-dir data/demo_videos/xxx.mp4 \
  --output-dir output/
```

Key flags mirror upstream `demo.py`: `--num-frames-per-batch` (8),
`--num-overlap-frames` (1), `--denoise-steps` (1), `--decode-chunk-size` (8),
`--size`/`--max-resolution` (720/960), `--noise-type` (`zeros`).

## Notes

- Modules run in MLX's native NHWC layout; `weights.sanitize_weights`
  transposes torch OIHW/OIDHW conv kernels on load/convert.
- The pipeline defaults to the upstream one-step `zeros`-noise regime; the
  LoRA adapter is applied as additive wrappers (scale = alpha/r = 0.5), not
  fused, so the base UNet weights stay untouched.
- Validated against the torch/diffusers reference: exact in fp32 (max abs
  diff 2.5e-6 on a tiny UNet forward), and bf16 vs fp32 end-to-end alpha
  max diff 0.004 with 100% mask agreement at threshold 0.5.
