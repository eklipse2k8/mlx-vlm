# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from NVIDIA Kimodo MotionCorrection (Apache-2.0).
"""Lazy GPU trajectory constraints used by Kimodo motion correction."""

import math

import mlx.core as mx

from .trajectory_kernels import correct_trajectories, smooth_trajectories


def trajectory_correct(
    reference: mx.array,
    observations: mx.array,
    margins: mx.array,
    pos_weight: float = 0.001,
    vel_weight: float = 1.0,
    acc_weight: float = 10.0,
    velocity_weights: mx.array | None = None,
    admm_iters: int = 100,
) -> mx.array:
    """Preserve a trajectory's shape while satisfying position constraints.

    The final two axes are time and channels; arbitrary leading batch axes
    broadcast across ``reference``, ``observations`` and ``margins``. A negative
    margin leaves a frame free, zero pins all channels exactly, and a positive
    margin constrains their Euclidean distance from the observation. The source
    ADMM update order and iteration count are preserved, including returning
    the final x update rather than the projected z update.

    ``velocity_weights`` is accepted for source API compatibility. NVIDIA's
    ``multVelWeights`` does not assign its triplets back to the velocity matrix,
    so those weights do not affect the source solve and are likewise unused.
    Factorization, solves, projections and boundary scans remain lazy on GPU.
    """
    if reference.ndim < 2 or observations.ndim < 2 or margins.ndim < 1:
        raise ValueError(
            "Trajectories require [..., time, channels] and [..., time] margins"
        )
    if reference.shape[-2:] != observations.shape[-2:]:
        raise ValueError(
            "Reference and observations must have matching time and channel axes"
        )
    frames, channels = reference.shape[-2:]
    if frames == 0 or channels == 0:
        raise ValueError("Trajectory time and channel axes must be nonempty")
    if margins.shape[-1] != frames:
        raise ValueError("Margins must have one value per trajectory frame")
    if channels > 1024:
        raise ValueError("Trajectory correction supports at most 1024 channels")
    if pos_weight <= 0 or vel_weight < 0 or acc_weight < 0:
        raise ValueError(
            "Position weight must be positive and derivative weights nonnegative"
        )
    if admm_iters < 0:
        raise ValueError("ADMM iteration count must be nonnegative")
    leading = mx.broadcast_shapes(
        reference.shape[:-2], observations.shape[:-2], margins.shape[:-1]
    )
    shape = leading + (frames, channels)
    batch = math.prod(leading)
    ref = (
        mx.broadcast_to(reference, shape)
        .astype(mx.float32)
        .reshape(batch, frames, channels)
    )
    obs = (
        mx.broadcast_to(observations, shape)
        .astype(mx.float32)
        .reshape(batch, frames, channels)
    )
    margin = (
        mx.broadcast_to(margins, leading + (frames,))
        .astype(mx.float32)
        .reshape(batch, frames)
    )
    weights = mx.array([pos_weight, vel_weight, acc_weight], dtype=mx.float32)
    return correct_trajectories(ref, obs, margin, weights, admm_iters).reshape(shape)


def smooth_channels(reference: mx.array, mask: mx.array) -> mx.array:
    """Smooth constrained boundary frames in source order, without host reads."""
    if reference.ndim < 2 or mask.ndim < 1 or reference.shape[-2] != mask.shape[-1]:
        raise ValueError(
            "Channels require [..., time, channels] and a matching [..., time] mask"
        )
    frames, channels = reference.shape[-2:]
    if frames == 0 or channels == 0:
        return reference
    leading = mx.broadcast_shapes(reference.shape[:-2], mask.shape[:-1])
    shape = leading + (frames, channels)
    batch = math.prod(leading)
    ref = (
        mx.broadcast_to(reference, shape)
        .astype(mx.float32)
        .reshape(batch, frames, channels)
    )
    mask = (
        mx.broadcast_to(mask, leading + (frames,))
        .astype(mx.float32)
        .reshape(batch, frames)
    )
    return smooth_trajectories(ref, mask).reshape(shape)
