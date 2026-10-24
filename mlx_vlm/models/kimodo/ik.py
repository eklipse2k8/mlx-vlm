# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from NVIDIA Kimodo MotionCorrection (Apache-2.0).
"""Lazy GPU limb IK and source-compatible rotation trajectory correction."""

import mlx.core as mx

from ..sam3d_body.mhr_utils import cross, quat_to_rotmat, rotmat_to_quat
from .ik_kernels import joint_global, limb_ik


def one_bone_ik(rotations, roots, rig, joint_index, target, weight=1.0):
    """Rotate the target joint's parent, preserving source NLERP behavior."""
    parent = rig.joint_parents[joint_index]
    if parent < 0:
        return rotations
    weights = mx.broadcast_to(mx.array(weight, mx.float32), roots.shape[:-1])
    updates = limb_ik(
        rotations, roots, rig, joint_index, target, weights, mx.zeros((3,)), False
    )
    result = mx.array(rotations)
    result[..., parent, :, :] = updates[..., 0, :, :]
    return result


def two_bone_ik(
    rotations, roots, rig, joint_index, target, weight=1.0, hint_offset=(0, 0, 0)
):
    """Apply the original analytic two-bone solver in one Metal dispatch."""
    parent = rig.joint_parents[joint_index]
    if parent < 0 or rig.joint_parents[parent] < 0:
        return rotations
    ancestor = rig.joint_parents[parent]
    weights = mx.broadcast_to(mx.array(weight, mx.float32), roots.shape[:-1])
    updates = limb_ik(
        rotations,
        roots,
        rig,
        joint_index,
        target,
        weights,
        mx.array(hint_offset, mx.float32),
        True,
    )
    result = mx.array(rotations)
    result[..., ancestor, :, :] = updates[..., 0, :, :]
    result[..., parent, :, :] = updates[..., 1, :, :]
    return result


def rotation_slerp(start, end, weight):
    """SLERP with the reference's linear branch near identical quaternions."""
    a, b = rotmat_to_quat(start), rotmat_to_quat(end)
    cosine = mx.sum(a * b, axis=-1, keepdims=True)
    sign = mx.where(cosine < 0, -1.0, 1.0)
    cosine = mx.clip(cosine * sign, 0, 1)
    sine = mx.sqrt(mx.maximum(1 - cosine * cosine, 0))
    angle = mx.arctan2(sine, cosine)
    w = weight[..., None]
    denominator = mx.maximum(sine, 1e-12)
    s0 = mx.where(cosine < 1 - 1e-5, mx.sin((1 - w) * angle) / denominator, 1 - w)
    s1 = mx.where(cosine < 1 - 1e-5, mx.sin(w * angle) / denominator, w)
    return quat_to_rotmat(a * s0 + b * s1 * sign)


def target_reach_falloff(rotations, roots, rig, joint_index, target, two_bone):
    """Source quartic Gaussian falloff outside 99% of the limb reach."""
    reach = mx.linalg.norm(rig.local_offsets[joint_index])
    joint_index = rig.joint_parents[joint_index]
    if two_bone:
        reach = reach + mx.linalg.norm(rig.local_offsets[joint_index])
        joint_index = rig.joint_parents[joint_index]
    _, base = joint_global(rotations, roots, rig, joint_index)
    distance = mx.linalg.norm(target - base, axis=-1)
    tmp = mx.maximum(distance / reach - 0.99, 0) / 0.01
    return mx.exp(-2 * tmp**4)


def correct_rotation_channels(
    local_rotations,
    target_rotations,
    masks,
    perform_channel_smoothing=False,
    velocity_weights=None,
):
    """Correct each bone's six forward/up channels without quaternion flips."""
    from .trajectory import smooth_channels, trajectory_correct

    b, t, j = local_rotations.shape[:3]
    reference = mx.concatenate(
        (local_rotations[..., 2], local_rotations[..., 1]), axis=-1
    )
    target = mx.concatenate(
        (target_rotations[..., 2], target_rotations[..., 1]), axis=-1
    )
    reference = reference.transpose(0, 2, 1, 3).reshape(b * j, t, 6)
    target = target.transpose(0, 2, 1, 3).reshape(b * j, t, 6)
    mask = masks.transpose(0, 2, 1).reshape(b * j, t)
    if perform_channel_smoothing:
        reference = smooth_channels(reference, mask)
    margins = mx.where(mask != 0, 0.0, -1.0)
    if velocity_weights is not None:
        velocity_weights = mx.broadcast_to(
            velocity_weights[:, None, :], (b, j, velocity_weights.shape[-1])
        ).reshape(b * j, -1)
    fixed = trajectory_correct(
        reference,
        target * mask[..., None],
        margins,
        pos_weight=0.01,
        vel_weight=1.0,
        acc_weight=10.0,
        velocity_weights=velocity_weights,
    )
    forward, up = fixed[..., :3], fixed[..., 3:]
    forward = forward / mx.maximum(
        mx.linalg.norm(forward, axis=-1, keepdims=True), 1e-30
    )
    up = up / mx.maximum(mx.linalg.norm(up, axis=-1, keepdims=True), 1e-30)
    right = cross(up, forward)
    right = right / mx.maximum(mx.linalg.norm(right, axis=-1, keepdims=True), 1e-30)
    # LookRotation rebuilds the orthogonal up vector from forward and right.
    result = mx.stack((right, cross(forward, right), forward), axis=-1)
    return result.reshape(b, j, t, 3, 3).transpose(0, 2, 1, 3, 4)
