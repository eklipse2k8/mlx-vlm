# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from NVIDIA Kimodo (Apache-2.0).
"""Sparse feature conditioning without GPU synchronization or dynamic shapes."""

import mlx.core as mx

from .geometry import matrix_to_cont6d


def create_conditions(rep, index_dict, data_dict, length, to_normalize=True):
    """Write frame/joint conditions using the reference feature block convention.

    Index dictionaries contain lists of integer arrays (frame indices or
    frame/joint pairs). Values contain matching lists of MLX arrays. Joint
    positions require a root planar condition on the same frame.
    """
    # Ardy routes excluded root-joint writes into three discarded columns.
    width = rep.motion_rep_dim
    observed = mx.zeros((length, width + 3), mx.float32)
    mask = mx.zeros((length, width + 3), mx.bool_)

    def gather(name):
        entries = index_dict.get(name)
        if entries is None or isinstance(entries, (tuple, list)) and not entries:
            return None
        values = data_dict[name]
        if isinstance(entries, mx.array):
            return entries.astype(mx.int32), mx.array(values, mx.float32)
        return (
            mx.concatenate([mx.array(x, mx.int32) for x in entries], axis=0),
            mx.concatenate([mx.array(x, mx.float32) for x in values], axis=0),
        )

    def frames(name, indices, values, components=None):
        nonlocal observed, mask
        s = rep.slice_dict[name]
        cols = list(range(s.stop - s.start)) if components is None else components
        cols = mx.array([s.start + c for c in cols], mx.int32)
        observed = mx.array(observed)
        observed[indices[:, None], cols] = values.reshape(-1, len(cols))
        mask = mx.array(mask)
        mask[indices[:, None], cols] = True

    def joints(name, pairs, values, exclude_root=False):
        nonlocal observed, mask
        s = rep.slice_dict[name]
        stride = rep.size_dict[name][-1]
        joint = pairs[:, 1] - int(exclude_root)
        cols = s.start + joint[:, None] * stride + mx.arange(stride, dtype=mx.int32)
        if exclude_root:
            cols = mx.where(
                (joint < 0)[:, None], width + mx.arange(stride, dtype=mx.int32), cols
            )
        observed = mx.array(observed)
        observed[pairs[:, 0, None], cols] = values.reshape(-1, stride)
        mask = mx.array(mask)
        mask[pairs[:, 0, None], cols] = True

    root_key = (
        "root_2d"
        if rep._ardy and index_dict.get("root_2d") is not None
        else "smooth_root_2d"
    )
    for source, destination, components in (
        (root_key, rep.root_pos_name, [0, 2]),
        ("root_y_pos", rep.root_pos_name, [1]),
        ("global_root_heading", "global_root_heading", None),
    ):
        entry = gather(source)
        if entry is not None:
            frames(destination, *entry, components=components)
    entry = gather("global_joints_rots")
    if entry is not None:
        indices, rotations = entry
        joints("global_rot_data", indices, matrix_to_cont6d(rotations))
    entry = gather("global_joints_positions")
    if entry is not None:
        pairs, positions = entry
        root = observed[pairs[:, 0], rep.slice_dict[rep.root_pos_name]]
        ground_root = root * mx.array([1, 0, 1], root.dtype)
        joints(
            "local_joints_positions",
            pairs,
            positions - ground_root,
            exclude_root=rep._ardy,
        )
    observed, mask = observed[:, :width], mask[:, :width]
    if to_normalize:
        observed = rep.normalize(observed)
        if rep._ardy:
            observed = observed * mask
    return observed, mask.astype(mx.float32) if rep._ardy else mask


def decode_pose_constraint(constraint, skeleton):
    """Decode a native pose constraint once for conditioning or motion cleanup."""
    from .geometry import axis_angle_to_matrix

    if "global_joints_positions" in constraint and "global_joints_rots" in constraint:
        positions = mx.array(constraint["global_joints_positions"], mx.float32)
        rotations = mx.array(constraint["global_joints_rots"], mx.float32)
    else:
        local = axis_angle_to_matrix(
            mx.array(constraint["local_joints_rot"], mx.float32)
        )
        if local.shape[-3] != skeleton.nbjoints:
            if (
                local.shape[-3] == 77
                and skeleton.nbjoints == 30
                and skeleton.output_joint_indices is not None
            ):
                local = mx.take(local, skeleton.output_joint_indices, axis=-3)
            else:
                raise ValueError(
                    "Constraint joint count does not match the converted skeleton."
                )
        rotations, positions, _ = skeleton.fk(
            local, mx.array(constraint["root_positions"], mx.float32)
        )
    count = len(constraint["frame_indices"])
    if positions.shape != (count, skeleton.nbjoints, 3):
        raise ValueError(
            "Constraint poses must match the number of frames and skeleton joints."
        )
    smooth = constraint.get("smooth_root_2d")
    smooth = (
        positions[:, skeleton.root_idx, :][:, [0, 2]]
        if smooth is None
        else mx.array(smooth, mx.float32)[..., :2]
    )
    return positions, rotations, smooth


def build_condition_dicts(constraints, skeleton):
    """Accept source constraint JSON or prebuilt feature index/data dictionaries."""
    import json
    from collections import defaultdict
    from pathlib import Path

    if isinstance(constraints, (str, Path)):
        with open(constraints) as f:
            constraints = json.load(f)
    if isinstance(constraints, dict):
        if "index_dict" in constraints and "data_dict" in constraints:
            return constraints["index_dict"], constraints["data_dict"]
        constraints = [constraints]
    indices, data = defaultdict(list), defaultdict(list)

    def append(name, index, values):
        indices[name].append(index)
        data[name].append(values)

    def pairs(frames, joints):
        frame = mx.broadcast_to(frames[:, None], (frames.shape[0], len(joints)))
        joint = mx.broadcast_to(mx.array(joints, mx.int32)[None], frame.shape)
        return mx.stack((frame, joint), axis=-1).reshape(-1, 2)

    for constraint in constraints:
        if not isinstance(constraint, dict):
            constraint.update_constraints(data, indices)
            continue
        if "index_dict" in constraint and "data_dict" in constraint:
            for name, values in constraint["index_dict"].items():
                entries = [values] if isinstance(values, mx.array) else values
                targets = constraint["data_dict"][name]
                targets = [targets] if isinstance(targets, mx.array) else targets
                indices[name].extend(entries)
                data[name].extend(targets)
            continue
        kind = constraint["type"]
        frames = mx.array(constraint["frame_indices"], mx.int32)
        if frames.ndim != 1:
            raise ValueError("Constraint frame_indices must be one-dimensional.")
        smooth = constraint.get("smooth_root_2d")
        if kind == "root2d":
            if smooth is None:
                raise ValueError("A root2d constraint requires smooth_root_2d.")
            smooth = mx.array(smooth, mx.float32)
            # Preserve the source JSON convention for legacy three-component data.
            append("smooth_root_2d", frames, smooth[..., :2])
            if constraint.get("global_root_heading") is not None:
                append(
                    "global_root_heading",
                    frames,
                    mx.array(constraint["global_root_heading"], mx.float32),
                )
            continue
        if kind not in (
            "fullbody",
            "end-effector",
            "left-hand",
            "right-hand",
            "left-foot",
            "right-foot",
        ):
            raise ValueError(f"Unknown constraint type: {kind}")
        positions, rotations, smooth = decode_pose_constraint(constraint, skeleton)
        if kind == "fullbody":
            append(
                "global_joints_positions",
                pairs(frames, list(range(skeleton.nbjoints))),
                positions.reshape(-1, 3),
            )
        else:
            semantic = {
                "left-hand": ["LeftHand"],
                "right-hand": ["RightHand"],
                "left-foot": ["LeftFoot"],
                "right-foot": ["RightFoot"],
            }
            names = semantic.get(kind, constraint.get("joint_names", []))
            rotation_names, position_names = skeleton.expand_joint_names(names)
            pos_ids = [skeleton.bone_index[name] for name in position_names]
            rot_ids = [skeleton.bone_index[name] for name in rotation_names]
            append(
                "global_joints_positions",
                pairs(frames, pos_ids),
                mx.take(positions, mx.array(pos_ids, mx.int32), axis=1).reshape(-1, 3),
            )
            append(
                "global_joints_rots",
                pairs(frames, rot_ids),
                mx.take(rotations, mx.array(rot_ids, mx.int32), axis=1).reshape(
                    -1, 3, 3
                ),
            )
        right, left = skeleton.hip_joint_idx
        hip = positions[:, right] - positions[:, left]
        angle = mx.arctan2(hip[:, 2], -hip[:, 0])
        append("smooth_root_2d", frames, smooth)
        append("root_y_pos", frames, positions[:, skeleton.root_idx, 1:2])
        append(
            "global_root_heading",
            frames,
            mx.stack((mx.cos(angle), mx.sin(angle)), axis=-1),
        )
    return indices, data
