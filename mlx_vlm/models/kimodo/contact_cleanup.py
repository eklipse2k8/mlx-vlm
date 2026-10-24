# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from NVIDIA Kimodo MotionCorrection (Apache-2.0).
"""GPU contact intervals, effector pins, and human/robot foot stabilization."""

from dataclasses import dataclass
from functools import lru_cache

import mlx.core as mx

from .ik import (
    correct_rotation_channels,
    one_bone_ik,
    rotation_slerp,
    target_reach_falloff,
    two_bone_ik,
)
from .ik_kernels import joint_global

ONE_BONE = 0
TWO_BONE = 1


@dataclass
class CleanupRig:
    joint_parents: tuple
    local_offsets: mx.array

    def __post_init__(self):
        self.joint_parents = tuple(self.joint_parents)
        self.local_offsets = self.local_offsets.astype(mx.float32)
        self.parent_indices = mx.array(self.joint_parents, mx.int32)


@dataclass
class ContactInfo:
    joint_index: int
    contact_type: int
    mask: mx.array
    hint_offset: tuple = (0, 0, 0)
    min_height: float = 0.0


@lru_cache(None)
def _interval_kernel():
    return mx.fast.metal_kernel(
        name="kimodo_contact_intervals",
        input_names=["contacts", "mask", "threshold"],
        output_names=["active", "representatives", "boundary"],
        source=r"""
            uint batch=thread_position_in_grid.x;
            if (batch >= B) return;
            uint base=batch*T;
            for (int i=0; i<T; ++i) {
                active[base+i]=0.0f; representatives[base+i]=i;
                boundary[base+i]=false;
            }
            int start=-1;
            for (int i=0; i<=T; ++i) {
                bool contact=i<T && (mask[base+i] ? 0.0f : contacts[base+i])>threshold[0];
                if (contact && start<0) start=i;
                else if (!contact && start>=0) {
                    bool sc=start>0 && mask[base+start-1];
                    bool ec=i<T && mask[base+i];
                    if (ONE_BONE ? !(sc||ec) : !(sc&&ec)) {
                        int representative=sc ? start-1 : (ec ? i : (start+i)/2);
                        for (int f=start; f<i; ++f) {
                            active[base+f]=1.0f;
                            representatives[base+f]=representative;
                            boundary[base+f]=sc||ec;
                        }
                    }
                    start=-1;
                }
            }
        """,
    )


def contact_intervals(contacts, mask, threshold=0.5, one_bone=False):
    """Find retained intervals and representative frames on the GPU."""
    b, t = contacts.shape
    return tuple(
        _interval_kernel()(
            inputs=[
                contacts.astype(mx.float32),
                mask,
                mx.array(threshold, mx.float32).reshape(1),
            ],
            template=[("B", b), ("T", t), ("ONE_BONE", one_bone)],
            grid=(b, 1, 1),
            threadgroup=(min(b, 32), 1, 1),
            output_shapes=[contacts.shape, contacts.shape, contacts.shape],
            output_dtypes=[mx.float32, mx.int32, mx.bool_],
        )
    )


def _joint_mask(full_body_mask, pins, joint, parent):
    matches = [p for p in pins if p.joint_index == joint]
    if not matches:
        matches = [p for p in pins if p.joint_index == parent]
    return mx.logical_or(full_body_mask, matches[0].mask) if matches else full_body_mask


def _select_frames(original, corrected, frames):
    return mx.where(frames[..., None, None, None], corrected, original)


def _smooth_joints(rotations, targets, masks):
    if not masks:
        return rotations
    indices = mx.array(sorted(masks), mx.int32)
    source = mx.take(rotations, indices, axis=-3)
    target = mx.take(targets, indices, axis=-3)
    mask = mx.stack([masks[j] for j in sorted(masks)], axis=-1)
    corrected = correct_rotation_channels(source, target, mask)
    result = mx.array(rotations)
    result[..., indices, :, :] = corrected
    return result


def _restore_global(rotations, roots, rig, joint, desired_rotation, frames):
    parent = rig.joint_parents[joint]
    parent_global, _ = joint_global(rotations, roots, rig, parent)
    local = mx.swapaxes(parent_global, -1, -2) @ desired_rotation
    result = mx.array(rotations)
    result[..., joint, :, :] = mx.where(
        frames[..., None, None], local, rotations[..., joint, :, :]
    )
    return result


def do_effector_ik(
    rotations, roots, target_rotations, target_roots, full_body_mask, pins, rig
):
    """Pin effectors with limb IK, restore their orientation, then smooth."""
    fixed = rotations
    masks = {}
    for pin in pins:
        joint = pin.joint_index
        parent = rig.joint_parents[joint]
        ancestor = rig.joint_parents[parent]
        touched = [joint, parent] + ([ancestor] if pin.contact_type == TWO_BONE else [])
        for j in touched:
            masks.setdefault(j, full_body_mask)
            masks[j] = mx.logical_or(masks[j], pin.mask)
        target_global, target_point = joint_global(
            target_rotations, target_roots, rig, joint
        )
        if pin.contact_type == TWO_BONE:
            updated = two_bone_ik(
                fixed, roots, rig, joint, target_point, hint_offset=pin.hint_offset
            )
        else:
            updated = one_bone_ik(fixed, roots, rig, joint, target_point)
        fixed = _select_frames(fixed, updated, pin.mask != 0)
        fixed = _restore_global(fixed, roots, rig, joint, target_global, pin.mask != 0)
    return _smooth_joints(rotations, fixed, masks)


def do_contact_ik(
    rotations,
    roots,
    full_body_mask,
    contacts,
    pins,
    rig,
    contact_threshold=0.5,
    has_double_ankle_joints=False,
):
    """Preserve Kimodo's ordered two-bone, toe, and double-ankle passes."""
    original = rotations
    fixed = rotations
    masks = {}
    contact_frames = {}
    for contact in contacts:
        if contact.contact_type != TWO_BONE:
            continue
        joint = contact.joint_index
        parent = rig.joint_parents[joint]
        ancestor = rig.joint_parents[parent]
        joint_mask = _joint_mask(full_body_mask, pins, joint, parent)
        active, representatives, boundary = contact_intervals(
            contact.mask, joint_mask, contact_threshold
        )
        frames = active != 0
        contact_frames[joint] = frames
        _, world_points = joint_global(rotations, roots, rig, joint)
        target = mx.take_along_axis(world_points, representatives[..., None], axis=1)
        target = mx.array(target)
        target[..., 1] = mx.where(
            boundary,
            target[..., 1],
            mx.maximum(world_points[..., 1], contact.min_height),
        )
        desired_rotation, _ = joint_global(fixed, roots, rig, joint)
        falloff = target_reach_falloff(fixed, roots, rig, joint, target, True)
        updated = two_bone_ik(
            fixed, roots, rig, joint, target, hint_offset=contact.hint_offset
        )
        updated = mx.array(updated)
        for j in (parent, ancestor):
            updated[..., j, :, :] = rotation_slerp(
                fixed[..., j, :, :], updated[..., j, :, :], falloff
            )
        fixed = _select_frames(fixed, updated, frames)
        fixed = _restore_global(fixed, roots, rig, joint, desired_rotation, frames)
        for j in (parent, ancestor, joint):
            masks.setdefault(j, joint_mask)
            masks[j] = mx.logical_or(masks[j], frames)
    rotations = _smooth_joints(rotations, fixed, masks)
    masks = {}
    for contact in contacts:
        if contact.contact_type != ONE_BONE:
            continue
        joint = contact.joint_index
        parent = rig.joint_parents[joint]
        joint_mask = _joint_mask(full_body_mask, pins, joint, parent)
        active, _, _ = contact_intervals(
            contact.mask, joint_mask, contact_threshold, True
        )
        frames = active != 0
        _, parent_point = joint_global(fixed, roots, rig, parent)
        _, target = joint_global(fixed, roots, rig, joint)
        length = mx.linalg.norm(target - parent_point, axis=-1, keepdims=True)
        for _ in range(10):
            target = mx.array(target)
            target[..., 1] = contact.min_height
            direction = target - parent_point
            direction = direction / mx.maximum(
                mx.linalg.norm(direction, axis=-1, keepdims=True), 1e-30
            )
            target = parent_point + direction * length
        updated = one_bone_ik(fixed, roots, rig, joint, target)
        fixed = _select_frames(fixed, updated, frames)
        masks.setdefault(parent, joint_mask)
        masks[parent] = mx.logical_or(masks[parent], frames)
    rotations = _smooth_joints(rotations, fixed, masks)
    if not has_double_ankle_joints:
        return rotations

    toes = {
        rig.joint_parents[c.joint_index]: c.joint_index
        for c in contacts
        if c.contact_type == ONE_BONE
    }
    saved_ankles, saved_toes = {}, {}
    for contact in contacts:
        first = contact.joint_index
        if contact.contact_type != TWO_BONE or first not in contact_frames:
            continue
        second = rig.joint_parents[first]
        knee = rig.joint_parents[second]
        hip = rig.joint_parents[knee]
        if hip < 0:
            continue
        frames = contact_frames[first]
        joint_mask = _joint_mask(full_body_mask, pins, first, second)
        first_rotation, first_point = joint_global(fixed, roots, rig, first)
        saved_ankles[first] = first_point
        if first in toes:
            _, saved_toes[first] = joint_global(fixed, roots, rig, toes[first])
        original_first_rotation, original_first_point = joint_global(
            original, roots, rig, first
        )
        _, original_second_point = joint_global(original, roots, rig, second)
        delta = (
            mx.swapaxes(original_first_rotation, -1, -2)
            @ (original_second_point - original_first_point)[..., None]
        )[..., 0]
        target = first_point + (first_rotation @ delta[..., None])[..., 0]
        updated = two_bone_ik(
            fixed, roots, rig, second, target, hint_offset=contact.hint_offset
        )
        fixed = _select_frames(fixed, updated, frames)
        for j in (knee, hip):
            masks.setdefault(j, joint_mask)
            masks[j] = mx.logical_or(masks[j], frames)
    rotations = _smooth_joints(rotations, fixed, masks)
    masks = {}
    for contact in contacts:
        first = contact.joint_index
        if contact.contact_type != TWO_BONE or first not in saved_ankles:
            continue
        second = rig.joint_parents[first]
        frames = contact_frames[first]
        joint_mask = _joint_mask(full_body_mask, pins, first, second)
        updated = one_bone_ik(fixed, roots, rig, first, saved_ankles[first])
        fixed = _select_frames(fixed, updated, frames)
        if first in saved_toes:
            updated = one_bone_ik(fixed, roots, rig, toes[first], saved_toes[first])
            fixed = _select_frames(fixed, updated, frames)
        for j in (second, first):
            masks.setdefault(j, joint_mask)
            masks[j] = mx.logical_or(masks[j], frames)
    return _smooth_joints(rotations, fixed, masks)
