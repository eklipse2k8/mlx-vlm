"""GVM pipeline: windowed one-step video matting inference."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from .scheduler import FlowMatchEulerDiscreteScheduler
from .unet import GVMUNet
from .vae import GVMVAE


@dataclass
class GVMOutput:
    alpha: mx.array  # (N, H, W, 1) matte in [0, 1]
    image: mx.array  # (N, H, W, 3) input frames in [0, 1]


class GVMPipeline:
    def __init__(
        self,
        vae: GVMVAE,
        unet: GVMUNet,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        self.vae = vae
        self.unet = unet
        self.scheduler = scheduler

    def encode(self, input: mx.array) -> mx.array:
        """(B, F, H, W, 3) frames in [-1, 1] -> scaled latents (B, F, h, w, 4)."""
        num_frames = input.shape[1]
        flat = input.reshape(-1, *input.shape[2:])
        latent = self.vae.encode(flat.astype(self._vae_dtype))
        latent = latent * self.vae.config.scaling_factor
        return latent.reshape(-1, num_frames, *latent.shape[1:])

    def decode(self, latents: mx.array, decode_chunk_size: int = 16) -> mx.array:
        """(B, F, h, w, 4) -> (B, F, H, W, 3) in [-1, 1], float32."""
        num_frames = latents.shape[1]
        flat = latents.reshape(-1, *latents.shape[2:])
        flat = flat / self.vae.config.scaling_factor

        frames = []
        for i in range(0, flat.shape[0], decode_chunk_size):
            chunk = flat[i : i + decode_chunk_size]
            frames.append(
                self.vae.decode(
                    chunk.astype(self._vae_dtype), num_frames=chunk.shape[0]
                )
            )
            mx.eval(frames[-1])
        frames = mx.concatenate(frames, axis=0)
        return frames.reshape(-1, num_frames, *frames.shape[1:]).astype(mx.float32)

    @property
    def _vae_dtype(self) -> mx.Dtype:
        return self.vae.encoder.conv_in.weight.dtype

    def single_infer(
        self,
        rgb: mx.array,
        num_inference_steps: int,
        noise_type: str = "zeros",
    ) -> mx.array:
        """One window of frames: (B, F, H, W, 3) in [-1, 1] -> latents (B, F, h, w, 4)."""
        rgb_latent = self.encode(rgb)

        if noise_type == "gaussian":
            noise_latent = mx.random.normal(rgb_latent.shape)
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.scheduler.timesteps
        elif noise_type == "zeros":
            noise_latent = mx.zeros_like(rgb_latent)
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = [
                mx.array(self.scheduler.config.num_train_timesteps - 1, dtype=mx.int64)
            ] * num_inference_steps
        else:
            raise NotImplementedError(f"Unknown noise_type {noise_type!r}")

        # GVM uses a zero image embedding (no CLIP conditioning)
        image_embeddings = mx.zeros(
            (noise_latent.shape[0], 1, self.unet.config.cross_attention_dim),
            dtype=rgb_latent.dtype,
        )

        for t in timesteps:
            latent_model_input = mx.concatenate([noise_latent, rgb_latent], axis=-1)
            model_output = self.unet(latent_model_input, t, image_embeddings)
            if noise_type == "zeros":
                noise_latent = model_output
            else:
                noise_latent = self.scheduler.step(model_output, t, noise_latent)
            mx.eval(noise_latent)

        return noise_latent

    def __call__(
        self,
        image: mx.array,
        num_frames: int = 8,
        num_overlap_frames: int = 1,
        decode_chunk_size: int = 8,
        num_inference_steps: int = 1,
        noise_type: str = "zeros",
        ensemble_size: int = 3,
    ) -> GVMOutput:
        """image: (N, H, W, 3) frames in [0, 1]."""
        assert ensemble_size >= 1
        image = image[None]  # (1, N, H, W, 3)
        B, N = image.shape[:2]
        rgb_norm = image * 2 - 1  # [-1, 1]
        rgb = mx.broadcast_to(rgb_norm, (ensemble_size, N, *image.shape[2:]))

        if N <= num_frames:
            latent_all = self.single_infer(
                rgb, num_inference_steps=num_inference_steps, noise_type=noise_type
            )
        else:
            assert num_frames % 2 == 0
            key_frame_indices = []
            for i in range(0, N, num_frames - num_overlap_frames):
                key_frame_indices.append(i)
                key_frame_indices.append(min(N - 1, i + num_frames - 1))

            latent_all = None
            for i in range(0, len(key_frame_indices), 2):
                start, end = key_frame_indices[i], key_frame_indices[i + 1]
                latent = self.single_infer(
                    rgb[:, start : end + 1],
                    num_inference_steps=num_inference_steps,
                    noise_type=noise_type,
                )

                if latent_all is not None:
                    overlap = min(num_overlap_frames, latent.shape[1])
                    ratio = mx.linspace(0, 1, overlap).reshape(1, -1, 1, 1, 1)
                    latent_all[:, -overlap:] = latent[:, :overlap] * ratio + latent_all[
                        :, -overlap:
                    ] * (1 - ratio)
                    latent_all = mx.concatenate(
                        [latent_all, latent[:, overlap:]], axis=1
                    )
                else:
                    latent_all = latent

            assert latent_all.shape[1] == image.shape[1]

        alpha = self.decode(latent_all, decode_chunk_size=decode_chunk_size)

        # mean over channels, best of ensemble, map [-1, 1] -> [0, 1]
        alpha = alpha.mean(axis=-1, keepdims=True)
        alpha = mx.max(alpha, axis=0)
        alpha = mx.clip(alpha * 0.5 + 0.5, 0.0, 1.0)

        return GVMOutput(alpha=alpha, image=image[0])
