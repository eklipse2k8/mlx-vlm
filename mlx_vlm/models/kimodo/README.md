# Kimodo

Native MLX text-to-motion diffusion for NVIDIA Kimodo. The port reuses MLX-VLM's
bidirectional Llama, pooling, attention, processor registration, checkpoint loader,
and safetensors writer. The two-stage denoiser preserves the original weight names.

Supported architectures include the Kimodo and Ardy motion representations with
SOMA, SMPL-X, and Unitree G1 skeletons. Official source checkpoints include:

- `nvidia/Kimodo-SOMA-RP-v1`, `v1.1`, and `v1.2`
- `nvidia/Kimodo-SOMA-RP-ArdyRep-v1.2`
- `nvidia/Kimodo-SOMA-SEED-v1` and `v1.1`
- `nvidia/Kimodo-SMPLX-RP-v1`
- `nvidia/Kimodo-G1-RP-v1` and `nvidia/Kimodo-G1-SEED-v1`

The model generates motion arrays. It does not support chat or tool calling.

## Convert

Conversion reads the original skeleton assets from a Kimodo checkout. Install
its Python dependencies for conversion and reference verification; runtime
generation needs only MLX-VLM. The original checkpoints already use safetensors.

```bash
uv pip install torch peft omegaconf hydra-core einops
```

```bash
python -m mlx_vlm.models.kimodo.convert \
  --model nvidia/Kimodo-SOMA-RP-v1.2 \
  --kimodo-source /path/to/kimodo \
  --mlx-path ./kimodo-mlx \
  --text-encoder McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp \
  --text-base mlx-community/Meta-Llama-3-8B-Instruct
```

The text conversion merges the MNTP adapter followed by the supervised LLM2Vec
adapter. The supplied MLX Llama base avoids a download from the gated Meta
repository. A plain Llama checkpoint alone does not reproduce LLM2Vec embeddings.
The default motion precision is float32; the text encoder uses bfloat16. The full
text encoder occupies about 15 GB, plus about 1.1 GB for the motion model.

The converted directory contains the motion weights and `config.json`, static
skeleton and normalization data under `assets/`, and the native text encoder
and tokenizer under `text_encoder/`. Omit `--text-encoder` to convert just the
motion model and supply a reusable encoder or precomputed text features in Python.

## Generate

```bash
python -m mlx_vlm.models.kimodo.generate \
  --model ./kimodo-mlx \
  --prompt 'A person walks forward.' \
  --duration 3 --steps 100 --seed 42 \
  --output motion.npz
```

Use `--constraints /path/to/constraints.json` for Kimodo's native root paths,
waypoints, full-body keyframes, and end-effector constraints. Multiple prompts
produce independent samples; `--multi-prompt` joins them as sequential segments
with an overlapping transition (`--transition-frames`, default 5).

```python
from mlx_vlm import load
from mlx_vlm.models.kimodo import KimodoPipeline

model, processor = load('./kimodo-mlx', lazy=True)
pipeline = KimodoPipeline(model)  # Reuse the resident text encoder between calls.
motion = pipeline.generate('A person waves.', num_frames=90,
                           num_denoising_steps=100, seed=42)
# motion['posed_joints']: [90, 77, 3] for SOMA
# All returned values are lazy MLX arrays.
```

Tokenization and input-file parsing run on the host. Text-model inference,
guidance, every DDIM step, normalization, rotation conversion, skeleton expansion,
and forward kinematics build a lazy graph on the GPU stream. There are no explicit
evaluations, scalar reads, NumPy conversions, or CPU tensor fallbacks in generation.
The CLI materializes the result only when exporting the NPZ file. Python callers
choose their own evaluation or export boundary.

Outputs follow the original NPZ convention: `posed_joints`, `global_rot_mats`,
`local_rot_mats`, `root_positions`, `smooth_root_pos`, `global_root_heading`, and
`foot_contacts`, plus `posed_joints_norootpos`. SOMA's 30 predicted joints expand
to the 77-joint output skeleton with relaxed fingers. Batched variable durations
are padded to the longest requested motion, as in the source implementation.

Optional motion cleanup is also native to MLX and Metal. Enable it with
`--postprocess` on the CLI or `post_processing=True` in `pipeline.generate`.
It corrects root and pose constraints, pins end effectors with inverse kinematics,
stabilizes planted feet and toes, and handles G1's double ankle chains. Sequential
prompts are corrected per segment with their transition history before that
history conditions the next segment.

```bash
python -m mlx_vlm.models.kimodo.generate \
  --model ./kimodo-mlx --prompt 'A person walks forward.' \
  --duration 3 --steps 100 --postprocess --output clean-motion.npz
```

The defaults preserve generation without cleanup. `--root-margin` sets the root
path tolerance in meters (default `0.04`); `--contact-threshold` sets the predicted
contact cutoff (default `0.5`). Python also exposes `post_process_motion` for
cleaning existing local rotation matrices, root positions, contacts, and native
constraints using a converted skeleton. Existing G1 generation remains opt-in
for cleanup, consistent with the source CLI's caution about its quality.

The cleanup uses Metal kernels for pentadiagonal factorization, ADMM trajectory
solves, ordered boundary smoothing, contact interval scans, forward kinematics,
and one/two-bone inverse kinematics. Its runtime has no C++, PyTorch, SciPy, NumPy,
or CPU tensor dependency. It remains lazy through the final output. The source
mesh and robot format exporters are separate tools.

The port preserves the source's solver behavior, including its currently unused
velocity weights. The CPU reference uses double precision in the trajectory
solver; Metal uses float32, so ill-conditioned long trajectories can differ
slightly. Trajectory hard pins remain exact. Reference checks compare both
rotations and physical joint positions.

On the M2 Ultra, full cleanup for synthetic foot-contact clips measured the
following (two warmups, median of three runs, input arrays resident on GPU,
graph construction and output evaluation included):

| Frames | C++ CPU (ms) | MLX GPU (ms) | Peak GPU memory (MB) |
| ---: | ---: | ---: | ---: |
| 30 | 1.748 | 6.374 | 1.125 |
| 90 | 4.268 | 6.808 | 3.573 |
| 180 | 8.090 | 7.393 | 7.877 |

The original C++ reference was built on ARM with SSE2NEON compatibility headers;
its cleanup algorithms were unchanged. Across 64 cleanup fixtures covering
SOMA30/77, SMPL-X, G1 double ankles, contacts, and constraints, maximum joint
position error stayed below 0.8 mm. Near-parallel rotation thresholds account for
the largest differences; the real walking sample agreed within 3.7 micrometers.
On that sample, mean horizontal foot speed during consecutive contact frames
fell from 0.0578 to 0.0450 m/s, a 22.1% reduction. This metric describes this
sample, rather than guaranteeing perfectly planted feet for every generated clip.

## Performance

Reuse a `KimodoPipeline` to keep model weights, the text encoder, and compiled
functions resident. Repeated requests can also pass precomputed `text_features`
and `text_mask` to avoid encoding the same text again.

Unconstrained separated guidance automatically evaluates two branches instead
of three during inference: its motion-conditioned and unconditional branches
are identical when no motion constraints are supplied. Constrained sampling
retains all three branches.

With the official SOMA-RP-v1.2 float32 weights on an Apple M2 Ultra with 128 GiB
RAM and MLX 0.32.4, 100-step denoising measured:

| Frames | Original (s) | Optimized (s) | Speedup | Original peak (GB) | Optimized peak (GB) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | 1.868 | 1.516 | 1.23x | 1.594 | 1.460 |
| 90 | 2.812 | 2.023 | 1.39x | 1.878 | 1.667 |
| 180 | 4.203 | 3.218 | 1.31x | 1.943 | 1.930 |

These are medians of three runs after two warmups, with resident weights,
identical initial noise, and shared native LLM2Vec features for "A person walks
forward." Timings include graph construction and evaluation of denoised features;
text encoding, skeleton decoding, and export are excluded. Maximum decoded joint
position differences stayed below 13 micrometers across the 4- and 100-step cases.
Batch shape changes can select different matrix multiplication kernels and cause
small floating-point rounding differences.

Pass `compile=True` to `pipeline.generate` or `pipeline.sample`, or use the CLI's
`--compile`, to compile the denoiser. With `post_processing=True`, the same option
also compiles cleanup. Standalone `post_process_motion(..., compile=True)` is
supported. Compilation requires the motion model to be in evaluation mode;
checkpoint loading already calls `model.eval()`.

```python
motion = pipeline.generate('A person waves.', 90, 100,
                           post_processing=True, compile=True)
```

Compilation is opt-in. On this hardware, compiling the denoiser alone improved
100-step latency by roughly 1–2%. Compiling an entire four-step sampling loop
improved latency by about 2%, so the pipeline keeps the Python sampling loop and
caches the compiled denoiser. Cleanup benefits more from fusing its smaller
operations. Targets, masks, contact thresholds, and root margins remain dynamic
inputs to compiled cleanup.

Synthetic foot-contact cleanup with the same skeleton measured the following
(two warmups, median of five runs, graph construction and output evaluation
included; memory excludes the motion denoiser and text encoder):

| Frames | Uncompiled (ms) | Compiled (ms) | Uncompiled peak (MB) | Compiled peak (MB) |
| ---: | ---: | ---: | ---: | ---: |
| 30 | 6.17 | 3.55 | 1.12 | 1.31 |
| 90 | 6.33 | 4.03 | 3.22 | 3.90 |
| 180 | 6.98 | 4.66 | 7.45 | 8.49 |

Maximum array differences between compiled and uncompiled cleanup stayed below
`7e-7` on these fixtures. The tests also compare compiled cleanup with the C++
reference outputs, including G1's double ankle behavior.

A 90-frame run including denoising, cleanup, and SOMA77 decoding, with shared
precomputed text features, measured the following (two warmups, median of three
runs; model loading, text encoding, and export excluded):

| Steps | Original (s) | Optimized (s) | Optimized + compile (s) |
| ---: | ---: | ---: | ---: |
| 4 | 0.120 | 0.089 | 0.103 |
| 100 | 2.821 | 2.042 | 1.995 |

Compilation can add overhead for short runs. Across these full-pipeline cases,
the maximum joint position difference from the original path was 0.43 mm.

The first compiled call can take several seconds to generate Metal kernels.
Changes to input shapes can trigger another trace. Prefer compilation for
repeated inference, and measure startup separately from warm latency; see the
[MLX compilation guide](https://ml-explore.github.io/mlx/build/html/usage/compile.html).

## Validate

```bash
python -m pytest mlx_vlm/tests/test_models.py -q -k Kimodo
python -m mlx_vlm.models.kimodo.verify \
  --mlx-path ./kimodo-mlx --kimodo-source /path/to/kimodo
```

The tests cover PyTorch transformer and bidirectional Llama parity, both LoRA
merges, shared checkpoint loading, all CFG modes, DDIM, motion geometry,
conditioning, seed reproducibility, and lazy text-to-skeleton execution. The
reference verifier uses the official motion weights and identical initial noise
to compare denoising, sampled features, and decoded joints and rotations.

CPU PyTorch and GPU MLX can accumulate different floating-point rounding errors;
compare decoded motion as well as normalized channels. The verifier reports
both. BF16 text encoding likewise requires a looser tolerance than FP32 synthetic
architecture checks.

For tighter reference comparisons, load the native text encoder and call
`encoder.set_dtype(mx.float32)` before passing it to `KimodoPipeline`. This uses
more memory. The full 8B encoder's maximum FP32 error against CPU Transformers
was `4.96e-5` on three fixed prompts; BF16 cosine similarity was at least `0.99991`.

On an Apple M2 Ultra with 128 GiB RAM, BF16 text encoding plus FP32 motion
sampling and SOMA77 decoding measured the following with four DDIM steps
(two warmups, median of three runs, resident weights, fresh text encoding each
run, and evaluation at the output boundary):

| Frames | Seconds | Peak GPU memory (GB) |
| ---: | ---: | ---: |
| 30 | 0.139 | 16.605 |
| 90 | 0.177 | 16.883 |
| 180 | 0.233 | 16.955 |

A separate CLI run generating 90 frames with 100 steps, including loading and
NPZ export, completed in 4.17 seconds with 16.83 GB peak GPU memory. Four steps
serve as a smoke test; use more steps for motion quality.

The implementation adapts [NVIDIA Kimodo](https://github.com/nv-tlabs/kimodo),
licensed under Apache-2.0. Model weights retain their original licenses.
