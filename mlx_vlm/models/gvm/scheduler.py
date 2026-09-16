"""FlowMatchEulerDiscreteScheduler port (inference-only, deterministic sampling)."""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import numpy as np

from .config import SchedulerConfig


class FlowMatchEulerDiscreteScheduler:
    def __init__(self, config: SchedulerConfig):
        self.config = config
        num_train_timesteps = config.num_train_timesteps

        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps)[
            ::-1
        ].astype(np.float32)
        sigmas = timesteps / num_train_timesteps
        if not config.use_dynamic_shifting:
            sigmas = config.shift * sigmas / (1 + (config.shift - 1) * sigmas)

        self.timesteps = mx.array(sigmas * num_train_timesteps)
        self.sigmas = mx.array(sigmas)
        self.sigma_min = float(sigmas[-1])
        self.sigma_max = float(sigmas[0])
        self._step_index: Optional[int] = None

    def set_timesteps(
        self, num_inference_steps: int, timesteps: Optional[mx.array] = None
    ) -> None:
        num_train_timesteps = self.config.num_train_timesteps
        if timesteps is None:
            # bounds are the (already shifted) sigma extremes from __init__
            sigma_max = self.sigma_max
            sigma_min = self.sigma_min
            timesteps = np.linspace(
                sigma_max * num_train_timesteps,
                sigma_min * num_train_timesteps,
                num_inference_steps,
            ).astype(np.float32)
        sigmas = np.array(timesteps, dtype=np.float32) / num_train_timesteps

        # timestep shifting (fixed shift)
        sigmas = self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)

        self.timesteps = mx.array(sigmas * num_train_timesteps)
        self.sigmas = mx.concatenate([mx.array(sigmas), mx.zeros(1)])
        self._step_index = None

    def _init_step_index(self, timestep: mx.array) -> None:
        matches = np.flatnonzero(
            np.array(self.timesteps) == np.float32(timestep.item())
        )
        index = int(matches[1]) if matches.shape[0] > 1 else int(matches[0])
        self._step_index = index

    def step(
        self, model_output: mx.array, timestep: mx.array, sample: mx.array
    ) -> mx.array:
        """One Euler step; returns prev_sample in the dtype of model_output."""
        if self._step_index is None:
            self._init_step_index(timestep)

        sample = sample.astype(mx.float32)
        sigma = self.sigmas[self._step_index]
        sigma_next = self.sigmas[self._step_index + 1]
        prev_sample = sample + (sigma_next - sigma) * model_output

        self._step_index += 1
        return prev_sample.astype(model_output.dtype)
