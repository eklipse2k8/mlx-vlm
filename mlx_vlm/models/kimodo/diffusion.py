# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from NVIDIA Kimodo (Apache-2.0).
"""Lazy cosine-schedule DDIM sampling and batched classifier-free guidance."""

import math

import mlx.core as mx


class Diffusion:
    def __init__(self, num_base_steps=1000):
        self.num_base_steps = num_base_steps

        def alpha(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

        betas = mx.array(
            [
                min(
                    1 - alpha((i + 1) / num_base_steps) / alpha(i / num_base_steps),
                    0.999,
                )
                for i in range(num_base_steps)
            ],
            dtype=mx.float32,
        )
        self.alphas_cumprod_base = mx.cumprod(1 - betas)

    def schedule(self, steps):
        if not isinstance(steps, int) or not 1 <= steps <= self.num_base_steps:
            raise ValueError(f"steps must be between 1 and {self.num_base_steps}")
        # Match source float32 spacing, including rounding at half timesteps.
        stride = (self.num_base_steps - 1) / max(1, steps - 1)
        indices = mx.round(mx.arange(steps, dtype=mx.float32) * stride).astype(mx.int32)
        base = self.alphas_cumprod_base[indices]
        base_previous = mx.concatenate([mx.ones((1,)), base[:-1]])
        # Preserve float32 rounding in the source's respaced beta schedule.
        betas = 1 - base / base_previous
        cumulative = mx.maximum(mx.cumprod(1 - betas), 1e-9)
        previous = mx.concatenate([mx.ones((1,)), cumulative[:-1]])
        return indices, cumulative, previous

    @staticmethod
    def step(noisy, clean, alpha, previous_alpha):
        eps = (mx.rsqrt(alpha) * noisy - clean) / mx.rsqrt(alpha / (1 - alpha))
        return mx.sqrt(previous_alpha) * clean + mx.sqrt(1 - previous_alpha) * eps


def guided_prediction(
    denoiser,
    x,
    pad_mask,
    text_feat,
    text_mask,
    timesteps,
    *,
    first_heading_angle=None,
    motion_mask=None,
    observed_motion=None,
    cfg_type="separated",
    cfg_weight=(2.0, 2.0),
):
    if cfg_type not in ("nocfg", "regular", "separated"):
        raise ValueError(f"Unknown CFG strategy: {cfg_type}")
    if cfg_type == "regular" and not isinstance(cfg_weight, (float, int)):
        raise ValueError("regular CFG requires a single weight")
    if cfg_type == "separated" and (
        not isinstance(cfg_weight, (tuple, list)) or len(cfg_weight) != 2
    ):
        raise ValueError("separated CFG requires (text_weight, constraint_weight)")
    if (
        cfg_type == "separated"
        and motion_mask is None
        and not getattr(denoiser, "training", False)
    ):
        # Without motion conditions, the constrained and unconditional branches
        # are identical. Evaluate that branch once during inference.
        cfg_type, cfg_weight = "regular", cfg_weight[0]
    branches = {"nocfg": 1, "regular": 2, "separated": 3}[cfg_type]

    def repeat(value):
        return None if value is None else mx.concatenate([value] * branches, axis=0)

    if branches > 1:
        text_feat = mx.concatenate(
            [text_feat] + [mx.zeros_like(text_feat)] * (branches - 1)
        )
        text_mask = mx.concatenate(
            [text_mask] + [mx.zeros_like(text_mask)] * (branches - 1)
        )
        if motion_mask is not None:
            masks = (
                [motion_mask, mx.zeros_like(motion_mask)]
                if cfg_type == "regular"
                else [
                    mx.zeros_like(motion_mask),
                    motion_mask,
                    mx.zeros_like(motion_mask),
                ]
            )
            motion_mask = mx.concatenate(masks)
    predictions = denoiser(
        repeat(x),
        repeat(pad_mask),
        text_feat,
        text_mask,
        repeat(timesteps),
        first_heading_angle=repeat(first_heading_angle),
        motion_mask=motion_mask,
        observed_motion=repeat(observed_motion),
    )
    if branches == 1:
        return predictions
    if branches == 2:
        conditional, unconditional = mx.split(predictions, 2)
        return unconditional + cfg_weight * (conditional - unconditional)
    text, constrained, unconditional = mx.split(predictions, 3)
    return (
        unconditional
        + cfg_weight[0] * (text - unconditional)
        + cfg_weight[1] * (constrained - unconditional)
    )
