# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU motion cleanup: trajectory constraints, joint pins, and foot contacts."""

import json
from functools import lru_cache, partial
from pathlib import Path

import mlx.core as mx

from .conditioning import decode_pose_constraint

_MASK_NAMES = ("FullBody", "LeftFoot", "RightFoot", "LeftHand", "RightHand", "Root")
_CONSTRAINT_MASKS = {
    "fullbody": "FullBody",
    "left-foot": "LeftFoot",
    "right-foot": "RightFoot",
    "left-hand": "LeftHand",
    "right-hand": "RightHand",
    "root2d": "Root",
}


def constraint_batches(constraints, batch):
    if isinstance(constraints, (str, Path)):
        constraints = json.loads(Path(constraints).read_text())
    if constraints is None:
        constraints = []
    if isinstance(constraints, dict):
        constraints = [constraints]
    if constraints and isinstance(constraints[0], list):
        if len(constraints) != batch:
            raise ValueError("Per-sample cleanup constraints must match the batch size")
        return constraints
    return [constraints] * batch


def crop_constraints(constraints, start, end):
    """Shift a constraint timeline; fixed-size invalid rows are discarded on GPU."""
    result = []
    for constraint in constraints:
        value = dict(constraint)
        value["frame_indices"] = mx.array(constraint["frame_indices"], mx.int32) - start
        value["_valid_frames"] = value["frame_indices"] < end - start
        result.append(value)
    return result


def prepare_constraint_targets(constraints, skeleton, frames, batch):
    """Build padded targets and named masks without reading tensor indices."""
    target_roots, target_rots, target_masks = [], [], []
    for sample_constraints in constraint_batches(constraints, batch):
        roots = mx.zeros((frames + 1, 3), mx.float32)
        rotations = mx.broadcast_to(mx.eye(3), (frames + 1, skeleton.nbjoints, 3, 3))
        masks = {name: mx.zeros((frames + 1,), mx.bool_) for name in _MASK_NAMES}
        # Full-body targets must take precedence over root and limb targets.
        for constraint in sorted(
            sample_constraints, key=lambda c: c.get("type") == "fullbody"
        ):
            if "type" not in constraint:
                raise ValueError(
                    "Motion cleanup requires native pose/root constraints, not feature-only constraints"
                )
            kind = constraint["type"]
            indices = mx.array(constraint["frame_indices"], mx.int32)
            valid = (indices >= 0) & (indices < frames)
            if "_valid_frames" in constraint:
                valid = valid & constraint["_valid_frames"]
            indices = mx.where(valid, indices, frames)
            names = []
            if kind == "end-effector":
                names = [
                    "Root" if name == "Hips" else name
                    for name in constraint.get("joint_names", [])
                ]
            elif kind in _CONSTRAINT_MASKS:
                names = [_CONSTRAINT_MASKS[kind]]
            for name in names:
                if name in masks:
                    masks[name] = mx.array(masks[name])
                    masks[name][indices] = True
            if kind == "root2d":
                planar = mx.array(constraint["smooth_root_2d"], mx.float32)[..., :2]
                roots = mx.array(roots)
                roots[indices[:, None], mx.array([0, 2])] = planar
                continue
            if kind not in (
                "fullbody",
                "end-effector",
                "left-hand",
                "right-hand",
                "left-foot",
                "right-foot",
            ):
                raise ValueError(f"Unknown cleanup constraint: {kind}")
            positions, global_rots, smooth = decode_pose_constraint(
                constraint, skeleton
            )
            target = positions[:, skeleton.root_idx]
            if kind != "fullbody" and "Hips" not in constraint.get("joint_names", []):
                target = mx.stack((smooth[:, 0], target[:, 1], smooth[:, 1]), axis=-1)
            roots = mx.array(roots)
            roots[indices] = target
            rotations = mx.array(rotations)
            rotations[indices] = skeleton.global_rots_to_local_rots(global_rots)
        target_roots.append(roots[:frames])
        target_rots.append(rotations[:frames])
        target_masks.append(masks)
    return (
        mx.stack(target_roots),
        mx.stack(target_rots),
        {
            name: mx.stack([mask[name][:frames] for mask in target_masks])
            for name in _MASK_NAMES
        },
    )


def create_cleanup_rig(skeleton):
    from .contact_cleanup import CleanupRig

    neutral = skeleton.neutral_joints.astype(mx.float32)
    offsets = neutral - neutral[skeleton.parent_indices]
    offset = 0.02 if skeleton.name.startswith("soma") else 0.007
    root = neutral[0] + mx.stack(
        (mx.array(0.0), -mx.min(neutral[:, 1]) + offset, mx.array(0.0))
    )
    offsets = mx.array(offsets)
    offsets[0] = root
    return CleanupRig(skeleton.joint_parents, offsets)


def _contacts_and_pins(skeleton, labels, masks, rig):
    from .contact_cleanup import ONE_BONE, TWO_BONE, ContactInfo

    groups = (
        skeleton.left_hand_joint_idx,
        skeleton.right_hand_joint_idx,
        skeleton.left_foot_joint_idx,
        skeleton.right_foot_joint_idx,
    )
    if any(not group for group in groups):
        raise ValueError(
            "Motion cleanup requires left/right hand and foot skeleton metadata"
        )
    left_hand, right_hand, left_foot, right_foot = [group[0] for group in groups]
    pins = [
        ContactInfo(index, TWO_BONE, masks[name] & ~masks["FullBody"], hint_offset=hint)
        for index, name, hint in (
            (left_hand, "LeftHand", (0, 0, -0.1)),
            (right_hand, "RightHand", (0, 0, -0.1)),
            (left_foot, "LeftFoot", (0, 0, 0.1)),
            (right_foot, "RightFoot", (0, 0, 0.1)),
        )
    ]
    # Default-pose heights include the source rig's small ground clearance.
    neutral = skeleton.neutral_joints
    height_shift = rig.local_offsets[0, 1] - neutral[0, 1]
    contacts = [
        ContactInfo(
            right_foot,
            TWO_BONE,
            mx.where(
                masks["RightFoot"], 0, mx.minimum(labels[..., 2] + labels[..., 3], 1)
            ),
            hint_offset=(0, 0, 0.1),
            min_height=neutral[right_foot, 1] + height_shift,
        ),
        ContactInfo(
            left_foot,
            TWO_BONE,
            mx.where(
                masks["LeftFoot"], 0, mx.minimum(labels[..., 0] + labels[..., 1], 1)
            ),
            hint_offset=(0, 0, 0.1),
            min_height=neutral[left_foot, 1] + height_shift,
        ),
    ]
    left_children = [j for j, p in enumerate(skeleton.joint_parents) if p == left_foot]
    right_children = [
        j for j, p in enumerate(skeleton.joint_parents) if p == right_foot
    ]
    if left_children and right_children:
        left_toe, right_toe = left_children[-1], right_children[-1]
        # The reference uses the right toe's height for both toes.
        toe_height = neutral[right_toe, 1] + height_shift
        contacts.extend(
            (
                ContactInfo(
                    right_toe,
                    ONE_BONE,
                    mx.where(masks["RightFoot"], 0, labels[..., 3]),
                    min_height=toe_height,
                ),
                ContactInfo(
                    left_toe,
                    ONE_BONE,
                    mx.where(masks["LeftFoot"], 0, labels[..., 1]),
                    min_height=toe_height,
                ),
            )
        )
    return contacts, pins


def post_process_motion(
    local_rot_mats,
    root_positions,
    contacts,
    skeleton,
    constraint_lst=None,
    contact_threshold=0.5,
    root_margin=0.04,
    *,
    compile=False,
):
    """Return corrected motion arrays, evaluated lazily on the GPU stream."""
    if not 0 <= contact_threshold <= 1 or root_margin < 0:
        raise ValueError(
            "contact_threshold must be in [0, 1] and root_margin nonnegative"
        )
    unbatched = local_rot_mats.ndim == 4
    if unbatched:
        local_rot_mats, root_positions, contacts = (
            local_rot_mats[None],
            root_positions[None],
            contacts[None],
        )
    if local_rot_mats.ndim != 5 or local_rot_mats.shape[-3:] != (
        skeleton.nbjoints,
        3,
        3,
    ):
        raise ValueError("local_rot_mats must have shape [B,T,J,3,3]")
    batch, frames = local_rot_mats.shape[:2]
    if (
        frames < 1
        or root_positions.shape != (batch, frames, 3)
        or contacts.shape[:2] != (batch, frames)
        or contacts.shape[-1] not in (4, 6)
    ):
        raise ValueError(
            "Root positions and foot contacts must match the motion's batch and frames"
        )
    with mx.stream(mx.gpu):
        local_rot_mats = local_rot_mats.astype(mx.float32)
        roots = root_positions.astype(mx.float32)
        labels = contacts.astype(mx.float32)
        if labels.shape[-1] == 6:
            labels = labels[..., [0, 1, 3, 4]]
        target_roots, target_rotations, masks = prepare_constraint_targets(
            constraint_lst, skeleton, frames, batch
        )
        cleanup = (
            _compiled_cleanup(skeleton)
            if compile
            else partial(_cleanup_motion, skeleton=skeleton)
        )
        result = cleanup(
            local_rot_mats,
            roots,
            labels,
            target_roots,
            target_rotations,
            masks,
            mx.array(contact_threshold, mx.float32),
            mx.array(root_margin, mx.float32),
        )
    return {name: value[0] for name, value in result.items()} if unbatched else result


@lru_cache(maxsize=8)
def _compiled_cleanup(skeleton):
    # Targets, masks and tuning values are explicit inputs, so later calls can
    # change constraints without retaining values from the first trace.
    return mx.compile(partial(_cleanup_motion, skeleton=skeleton))


def _cleanup_motion(
    local_rot_mats,
    roots,
    labels,
    target_roots,
    target_rotations,
    masks,
    contact_threshold,
    root_margin,
    *,
    skeleton,
):
    from .contact_cleanup import (
        correct_rotation_channels,
        do_contact_ik,
        do_effector_ik,
    )
    from .trajectory import smooth_channels, trajectory_correct

    rig = create_cleanup_rig(skeleton)
    foot_contacts, pins = _contacts_and_pins(skeleton, labels, masks, rig)
    hard = masks["FullBody"]
    y = trajectory_correct(
        roots[..., 1:2],
        target_roots[..., 1:2],
        mx.where(hard, 0.0, -1.0),
        pos_weight=0.01,
        vel_weight=1.0,
        acc_weight=1.0,
    )
    pin_mask = mx.stack([pin.mask for pin in pins]).any(axis=0)
    margins = mx.where(hard, 0.0, mx.where(masks["Root"] | pin_mask, root_margin, -1.0))
    xz = trajectory_correct(
        smooth_channels(roots[..., [0, 2]], hard | masks["Root"]),
        target_roots[..., [0, 2]],
        margins,
        pos_weight=0.001,
        vel_weight=1.0,
        acc_weight=10.0,
    )
    roots = mx.stack((xz[..., 0], y[..., 0], xz[..., 1]), axis=-1)
    rotations = correct_rotation_channels(
        local_rot_mats,
        target_rotations,
        mx.broadcast_to(hard[..., None], local_rot_mats.shape[:3]),
        perform_channel_smoothing=True,
    )
    rotations = do_effector_ik(
        rotations, roots, target_rotations, target_roots, hard, pins, rig
    )
    rotations = do_contact_ik(
        rotations,
        roots,
        hard,
        foot_contacts,
        pins,
        rig,
        contact_threshold=contact_threshold,
        has_double_ankle_joints=skeleton.name == "g1skel34",
    )
    global_rots, joints, _ = skeleton.fk(rotations, roots)
    return dict(
        local_rot_mats=rotations,
        root_positions=roots,
        global_rot_mats=global_rots,
        posed_joints=joints,
    )
