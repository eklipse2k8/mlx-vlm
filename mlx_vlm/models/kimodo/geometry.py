# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from NVIDIA Kimodo (Apache-2.0).
"""Lazy rotation and forward kinematics operations for Kimodo."""

import mlx.core as mx

from ..sam3d_body.mhr_utils import cross


def matrix_to_cont6d(matrix: mx.array) -> mx.array:
    return mx.concatenate((matrix[..., 0], matrix[..., 1]), axis=-1)


def cont6d_to_matrix(cont6d: mx.array) -> mx.array:
    # Kimodo uses exact normalization; SAM3D's rot6d helper adds an epsilon.
    x = cont6d[..., :3]
    x = x / mx.linalg.norm(x, axis=-1, keepdims=True)
    z = cross(x, cont6d[..., 3:])
    z = z / mx.linalg.norm(z, axis=-1, keepdims=True)
    return mx.stack((x, cross(z, x), z), axis=-1)


def angle_to_y_rotation_matrix(angle: mx.array) -> mx.array:
    c, s = mx.cos(angle), mx.sin(angle)
    z, o = mx.zeros_like(c), mx.ones_like(c)
    return mx.stack((c, z, s, z, o, z, -s, z, c), axis=-1).reshape(angle.shape + (3, 3))


def global_rots_to_local_rots(global_joint_rots: mx.array, skeleton) -> mx.array:
    parent = mx.take(global_joint_rots, skeleton.parent_indices, axis=-3)
    parent = mx.where(
        skeleton.root_mask[:, None, None],
        mx.eye(3, dtype=global_joint_rots.dtype),
        parent,
    )
    return mx.swapaxes(parent, -1, -2) @ global_joint_rots


def fk(local_joint_rots, root_positions, skeleton, root_positions_is_global=True):
    """Compose rotations and bone offsets level by level without host reads."""
    if skeleton.rest_local_rots is not None:
        local_joint_rots = (
            skeleton.rest_local_rots.astype(local_joint_rots.dtype) @ local_joint_rots
        )
    neutral = skeleton.neutral_joints.astype(local_joint_rots.dtype)
    if root_positions_is_global:
        neutral = neutral - neutral[skeleton.root_idx]
    offsets = neutral - mx.take(neutral, skeleton.parent_indices, axis=0)
    offsets = mx.where(skeleton.root_mask[:, None], neutral, offsets)
    positions = mx.broadcast_to(offsets, local_joint_rots.shape[:-2] + (3,))
    rotations = local_joint_rots
    for indices, parents in skeleton.levels:
        parent_rots = mx.take(rotations, parents, axis=-3)
        new_rots = parent_rots @ mx.take(local_joint_rots, indices, axis=-3)
        new_pos = (
            mx.take(positions, parents, axis=-2)
            + (parent_rots @ mx.take(offsets, indices, axis=0)[..., None])[..., 0]
        )
        rotations = mx.array(rotations)
        rotations[..., indices, :, :] = new_rots
        positions = mx.array(positions)
        positions[..., indices, :] = new_pos
    return rotations, positions + root_positions[..., None, :], positions


def joints_from_global_rots(
    global_joint_rots,
    root_positions,
    skeleton,
    root_positions_is_global=True,
    return_norootpos=False,
):
    neutral = skeleton.neutral_joints.astype(global_joint_rots.dtype)
    if root_positions_is_global:
        neutral = neutral - neutral[skeleton.root_idx]
    offsets = neutral - mx.take(neutral, skeleton.parent_indices, axis=0)
    offsets = mx.where(skeleton.root_mask[:, None], neutral, offsets)
    parent_rots = mx.take(global_joint_rots, skeleton.parent_indices, axis=-3)
    steps = (parent_rots @ offsets[..., None])[..., 0]
    positions = mx.broadcast_to(offsets, global_joint_rots.shape[:-2] + (3,))
    for indices, parents in skeleton.levels:
        new = mx.take(positions, parents, axis=-2) + mx.take(steps, indices, axis=-2)
        positions = mx.array(positions)
        positions[..., indices, :] = new
    posed = positions + root_positions[..., None, :]
    return (posed, positions) if return_norootpos else posed


def axis_angle_to_matrix(axis_angle):
    """Rodrigues conversion with the same small-angle convention as Kimodo."""
    angle = mx.linalg.norm(axis_angle, axis=-1, keepdims=True)
    axis = axis_angle / (angle + 1e-6)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = mx.zeros_like(x)
    skew = mx.stack((zero, -z, y, z, zero, -x, -y, x, zero), axis=-1)
    skew = skew.reshape(axis.shape[:-1] + (3, 3))
    return (
        mx.eye(3, dtype=axis.dtype)
        + mx.sin(angle)[..., None] * skew
        + (1 - mx.cos(angle)[..., None]) * (skew @ skew)
    )
