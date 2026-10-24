# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from NVIDIA Kimodo (Apache-2.0).
"""Motion representations and skeleton buffers, kept outside trainable parameters."""

import json
import math
from functools import lru_cache

import mlx.core as mx

from .geometry import (
    angle_to_y_rotation_matrix,
    cont6d_to_matrix,
    fk,
    global_rots_to_local_rots,
    joints_from_global_rots,
    matrix_to_cont6d,
)


class Skeleton:
    """Static skeleton metadata and GPU arrays loaded from converted assets."""

    def __init__(
        self,
        neutral_joints,
        joint_parents,
        joint_names=None,
        name="skeleton",
        hip_joint_idx=None,
        left_foot_joint_idx=None,
        right_foot_joint_idx=None,
        left_hand_joint_idx=None,
        right_hand_joint_idx=None,
        rest_local_rots=None,
        output_skeleton=None,
        output_joint_indices=None,
        output_rest_local_rots=None,
        **metadata,
    ):
        self.name = name
        self.joint_parents = tuple(joint_parents)
        self.nbjoints = len(self.joint_parents)
        if not self.nbjoints or self.joint_parents[0] != -1:
            raise ValueError("Kimodo requires a skeleton with root at joint zero.")
        if any(p < 0 or p >= i for i, p in enumerate(self.joint_parents[1:], 1)):
            raise ValueError("Joint parents must precede their children.")
        self.root_idx = 0
        self.neutral_joints = mx.array(neutral_joints, dtype=mx.float32).reshape(
            self.nbjoints, 3
        )
        self.rest_local_rots = (
            None
            if rest_local_rots is None
            else mx.array(rest_local_rots, dtype=mx.float32)
        )
        self.bone_order_names = list(
            joint_names or [str(i) for i in range(self.nbjoints)]
        )
        self.bone_index = {n: i for i, n in enumerate(self.bone_order_names)}
        self.hip_joint_idx = list(hip_joint_idx or [2, 1])
        self.left_foot_joint_idx = list(left_foot_joint_idx or [])
        self.right_foot_joint_idx = list(right_foot_joint_idx or [])
        self.foot_joint_idx = self.left_foot_joint_idx + self.right_foot_joint_idx
        for side, supplied in (
            ("left", left_hand_joint_idx),
            ("right", right_hand_joint_idx),
        ):
            title = side.capitalize()
            names = [n for n in self.bone_order_names if n.startswith(title + "Hand")]
            if self.nbjoints == 30:
                names = [
                    n for n in names if n in (title + "Hand", title + "HandMiddleEnd")
                ]
            if not names:
                names = [
                    n
                    for n in (
                        side + "_wrist_yaw_skel",
                        side + "_hand_roll_skel",
                        side + "_wrist",
                    )
                    if n in self.bone_index
                ]
            setattr(
                self,
                side + "_hand_joint_idx",
                (
                    list(supplied)
                    if supplied is not None
                    else [self.bone_index[n] for n in names]
                ),
            )
        self.parent_indices = mx.array(
            [max(0, p) for p in self.joint_parents], mx.int32
        )
        self.root_mask = mx.arange(self.nbjoints) == self.root_idx
        depths = [0] * self.nbjoints
        for j, parent in enumerate(self.joint_parents[1:], 1):
            depths[j] = depths[parent] + 1
        self.levels = []
        for depth in range(1, max(depths) + 1):
            indices = [j for j, d in enumerate(depths) if d == depth]
            parents = [self.joint_parents[j] for j in indices]
            self.levels.append(
                (mx.array(indices, mx.int32), mx.array(parents, mx.int32))
            )
        self.output_skeleton = Skeleton(**output_skeleton) if output_skeleton else None
        self.output_joint_indices = (
            None
            if output_joint_indices is None
            else mx.array(output_joint_indices, mx.int32)
        )
        self.output_rest_local_rots = (
            None
            if output_rest_local_rots is None
            else mx.array(output_rest_local_rots, mx.float32)
        )

    @classmethod
    def from_file(cls, path):
        with open(path) as f:
            return cls(**json.load(f))

    def fk(self, local_joint_rots, root_positions):
        return fk(local_joint_rots, root_positions, self)

    def global_rots_to_local_rots(self, rotations):
        return global_rots_to_local_rots(rotations, self)

    def expand_joint_names(self, joint_names):
        groups = {
            "LeftFoot": self.left_foot_joint_idx,
            "RightFoot": self.right_foot_joint_idx,
            "LeftHand": self.left_hand_joint_idx,
            "RightHand": self.right_hand_joint_idx,
            "Hips": [self.root_idx],
        }
        positions, rotations = [], []
        for name in joint_names:
            group = groups[name] if name in groups else [self.bone_index[name]]
            rotations.extend(group[:1])
            positions.extend(group)
        return (
            [self.bone_order_names[j] for j in rotations],
            [self.bone_order_names[j] for j in positions],
        )

    def output_to_SOMASkeleton77(self, output):
        """Expand compact SOMA rotations with the checkpoint's relaxed hand pose."""
        if self.output_skeleton is None:
            if self.nbjoints == 77:
                return output
            raise ValueError("This checkpoint has no SOMA77 output assets.")
        local = output["local_rot_mats"]
        expanded = mx.broadcast_to(
            self.output_rest_local_rots.astype(local.dtype),
            local.shape[:-3] + (self.output_skeleton.nbjoints, 3, 3),
        )
        expanded = mx.array(expanded)
        expanded[..., self.output_joint_indices, :, :] = local
        global_rots, positions, _ = self.output_skeleton.fk(
            expanded, output["root_positions"]
        )
        result = dict(
            output,
            local_rot_mats=expanded,
            global_rot_mats=global_rots,
            posed_joints=positions,
        )
        if "foot_contacts" in result:
            fc = result["foot_contacts"]
            result["foot_contacts"] = mx.concatenate(
                (fc[..., :2], fc[..., 1:2], fc[..., 2:4], fc[..., 3:4]), axis=-1
            )
        return result


class Stats:
    def __init__(self, mean, std, eps=1e-5):
        self.mean = mx.array(mean)
        self.std = mx.array(std)
        if self.mean.shape != self.std.shape:
            raise ValueError("Normalization mean and std must have matching shapes.")
        self.std_eps = mx.sqrt(self.std**2 + eps)

    def normalize(self, data):
        return (data - self.mean.astype(data.dtype)) / self.std_eps.astype(data.dtype)

    def unnormalize(self, data):
        return data * self.std_eps.astype(data.dtype) + self.mean.astype(data.dtype)


def _velocity(data, fps, lengths):
    """Differences along time; copy the previous velocity at each last valid frame."""
    velocity = fps * (data[:, 1:] - data[:, :-1])
    velocity = mx.concatenate((velocity, mx.zeros_like(data[:, :1])), axis=1)
    previous = mx.concatenate(
        (mx.zeros_like(velocity[:, :1]), velocity[:, :-1]), axis=1
    )
    last = mx.arange(data.shape[1])[None, :] == (lengths[:, None] - 1)
    return mx.where(
        last.reshape(last.shape + (1,) * (data.ndim - 2)), previous, velocity
    )


def _angular_velocity(angles, fps, lengths):
    c, s = mx.cos(angles), mx.sin(angles)
    diff = fps * mx.arctan2(
        s[:, 1:] * c[:, :-1] - c[:, 1:] * s[:, :-1],
        c[:, 1:] * c[:, :-1] + s[:, 1:] * s[:, :-1],
    )
    velocity = mx.concatenate((diff, mx.zeros_like(angles[:, :1])), axis=1)
    previous = mx.concatenate(
        (mx.zeros_like(velocity[:, :1]), velocity[:, :-1]), axis=1
    )
    last = mx.arange(angles.shape[1])[None, :] == lengths[:, None] - 1
    return mx.where(last, previous, velocity)


@lru_cache(maxsize=32)
def _smoothing_operator(n):
    """Build a shape-only ADMM solve operator; motion data never leaves the GPU.

    MLX's solve and inverse currently require CPU streams. The pentadiagonal
    coefficients depend only on sequence length, so factor them as static metadata.
    """
    diagonal = [0.0] * n
    lower1, lower2 = [0.0] * n, [0.0] * n
    for j in range(1, n - 1):
        diagonal[j - 1] += 1.0
        diagonal[j] += 4.0
        diagonal[j + 1] += 1.0
        lower1[j] -= 2.0
        lower1[j + 1] -= 2.0
        lower2[j + 1] += 1.0
    rho = 0.25 * math.sqrt(max(diagonal))
    d, l1, l2 = [], [], []
    for i in range(n):
        b2 = lower2[i] / d[i - 2] if i > 1 else 0.0
        b1 = (lower1[i] - b2 * l1[i - 1]) / d[i - 1] if i else 0.0
        l2.append(b2)
        l1.append(b1)
        d.append(math.sqrt(diagonal[i] + rho - b1 * b1 - b2 * b2))
    inverse = [[0.0] * n for _ in range(n)]
    for column in range(n):
        y = []
        for i in range(n):
            v = rho if i == column else 0.0
            if i:
                v -= l1[i] * y[i - 1]
            if i > 1:
                v -= l2[i] * y[i - 2]
            y.append(v / d[i])
        for i in range(n - 1, -1, -1):
            v = y[i]
            if i + 1 < n:
                v -= l1[i + 1] * inverse[i + 1][column]
            if i + 2 < n:
                v -= l2[i + 2] * inverse[i + 2][column]
            inverse[i][column] = v / d[i]
    return mx.array(inverse, mx.float32)


def get_smooth_root_pos(root_positions, admm_iters=500):
    """Reference multigrid ADMM smoothing, with all trajectory arithmetic on GPU."""
    nframes = root_positions.shape[1]
    if nframes < 3:
        return root_positions
    target = root_positions[..., [0, 2]]
    smoothed = mx.broadcast_to(mx.mean(target, axis=1, keepdims=True), target.shape)
    step = 2 ** max(math.floor(math.log2(nframes)) - 4, 1)
    while True:
        x_target = target[:, ::step]
        x = smoothed[:, ::step]
        count = x.shape[1]
        if count >= 3:
            solve = _smoothing_operator(count).astype(root_positions.dtype)
            u = mx.zeros_like(x)
            for _ in range(admm_iters):
                delta = x + u - x_target
                norm = mx.linalg.norm(delta, axis=-1, keepdims=True)
                scale = mx.minimum(1.0, 0.06 / mx.maximum(norm, 1e-12))
                z = delta * scale + x_target
                u = u + 1.8 * (x - z)
                x = solve @ (z - u)
        smoothed = mx.array(smoothed)
        smoothed[:, ::step] = x
        half = step // 2
        interleaved = smoothed[:, half::step].shape[1]
        if interleaved == count:
            indices = half + (interleaved - 1) * step
            smoothed = mx.array(smoothed)
            smoothed[:, indices] = x[:, -1] + (x[:, -1] - x[:, -2]) / 2
            interleaved -= 1
        # The final half-step interpolation is part of the reference algorithm.
        base = smoothed[:, ::step]
        interpolation = (base[:, :-1] + base[:, 1:]) / 2
        ids = mx.arange(interleaved, dtype=mx.int32) * step + half
        smoothed = mx.array(smoothed)
        smoothed[:, ids] = interpolation[:, :interleaved]
        if step == 1:
            break
        step = half
    return mx.stack(
        (smoothed[..., 0], root_positions[..., 1], smoothed[..., 1]), axis=-1
    )


class KimodoMotionRep:
    """Kimodo smooth-root features with lazy normalization, encoding and decoding."""

    _ardy = False

    def __init__(
        self, skeleton, fps=30, stats=None, stats_path=None, name=None, **kwargs
    ):
        self.skeleton = skeleton
        self.fps = fps
        self.name = name
        self.nbjoints = skeleton.nbjoints
        self.root_pos_name = "root_pos" if self._ardy else "smooth_root_pos"
        self.size_dict = {
            self.root_pos_name: (3,),
            "global_root_heading": (2,),
            "local_joints_positions": (self.nbjoints - int(self._ardy), 3),
            "global_rot_data": (self.nbjoints, 6),
            "velocities": (self.nbjoints, 3),
            "foot_contacts": (4,),
        }
        self.feature_names = list(self.size_dict)
        self.nfeats_dict = {k: math.prod(shape) for k, shape in self.size_dict.items()}
        self.slice_dict = {}
        offset = 0
        for k, n in self.nfeats_dict.items():
            self.slice_dict[k] = slice(offset, offset + n)
            offset += n
        self.ps = list(self.size_dict.values())
        self.motion_rep_dim = offset
        self.global_root_dim = 5
        self.local_root_dim = 4
        self.root_slice = slice(0, self.global_root_dim)
        self.body_slice = slice(self.global_root_dim, self.motion_rep_dim)
        self.body_dim = self.motion_rep_dim - self.global_root_dim
        self.root_dim = self.global_root_dim
        self.motion_root_dim = self.global_root_dim
        self.motion_body_dim = self.body_dim
        self.local_motion_rep_dim = self.local_root_dim + self.body_dim
        self.global_motion_rep_dim = self.motion_rep_dim
        if stats_path is not None:
            stats = mx.load(str(stats_path))
        self.set_stats(stats)

    def set_stats(self, stats=None):
        parts = {}
        for name, size in (
            ("global_root", 5),
            ("local_root", 4),
            ("body", self.body_dim),
        ):
            mean = mx.zeros((size,)) if stats is None else stats[f"{name}.mean"]
            std = mx.ones((size,)) if stats is None else stats[f"{name}.std"]
            if mean.shape != (size,) or std.shape != (size,):
                raise ValueError(f"{name} stats must have shape ({size},).")
            parts[name] = Stats(mean, std)
            setattr(self, f"{name}_stats", parts[name])
        self.stats = Stats(
            mx.concatenate((parts["global_root"].mean, parts["body"].mean)),
            mx.concatenate((parts["global_root"].std, parts["body"].std)),
        )

    def normalize(self, features):
        return self.stats.normalize(features)

    def unnormalize(self, features):
        return self.stats.unnormalize(features)

    def _unpack(self, features):
        leading = features.shape[:-1]
        return [
            features[..., self.slice_dict[k]].reshape(leading + self.size_dict[k])
            for k in self.feature_names
        ]

    @staticmethod
    def _pack(parts):
        return mx.concatenate([p.reshape(p.shape[:2] + (-1,)) for p in parts], axis=-1)

    def global_root_to_local_root(self, root_features, normalized, lengths=None):
        unbatched = root_features.ndim == 2
        if unbatched:
            root_features = root_features[None]
        if lengths is None:
            lengths = mx.full(
                (root_features.shape[0],), root_features.shape[1], mx.int32
            )
        else:
            lengths = mx.array(lengths, mx.int32).reshape(-1)
        if normalized:
            root_features = self.global_root_stats.unnormalize(root_features)
        angles = mx.arctan2(root_features[..., 4], root_features[..., 3])
        angular = _angular_velocity(angles, self.fps, lengths)
        linear = _velocity(root_features[..., :3], self.fps, lengths)[..., [0, 2]]
        result = mx.concatenate(
            (angular[..., None], linear, root_features[..., 1:2]), axis=-1
        )
        if normalized:
            result = self.local_root_stats.normalize(result)
        return result[0] if unbatched else result

    def local_root_to_global_root(self, local_root_features, normalized, lengths=None):
        if normalized:
            local_root_features = self.local_root_stats.unnormalize(local_root_features)
        delta = local_root_features[..., :3] / self.fps
        integrated = mx.cumsum(
            mx.concatenate(
                (mx.zeros_like(delta[..., :1, :]), delta[..., :-1, :]), axis=-2
            ),
            axis=-2,
        )
        angle, x, z = integrated[..., 0], integrated[..., 1], integrated[..., 2]
        result = mx.stack(
            (x, local_root_features[..., 3], z, mx.cos(angle), mx.sin(angle)), axis=-1
        )
        return self.global_root_stats.normalize(result) if normalized else result

    def recenter_root_motion(
        self,
        root_motion,
        center_frame_index,
        is_normalized,
        to_normalize,
        return_center_pos=False,
    ):
        center_frame_index = mx.array(center_frame_index, mx.int32).reshape(-1)
        if is_normalized:
            root_motion = self.global_root_stats.unnormalize(root_motion)
        center = root_motion[mx.arange(root_motion.shape[0]), center_frame_index, :3]
        center = center * mx.array([1, 0, 1], root_motion.dtype)
        result = mx.concatenate(
            (root_motion[..., :3] - center[:, None], root_motion[..., 3:]), axis=-1
        )
        if to_normalize:
            result = self.global_root_stats.normalize(result)
        return (result, center) if return_center_pos else result

    def inverse(
        self,
        features,
        is_normalized=True,
        posed_joints_from="rotations",
        return_numpy=False,
        return_all=False,
        need_local_rots=True,
        contact_threshold=0.5,
    ):
        if return_numpy:
            raise ValueError(
                "Kimodo decoding returns lazy MLX arrays; export arrays explicitly after generation."
            )
        if posed_joints_from not in ("rotations", "positions"):
            raise ValueError("posed_joints_from must be 'rotations' or 'positions'.")
        unbatched = features.ndim == 2
        if unbatched:
            features = features[None]
        if is_normalized:
            features = self.unnormalize(features)
        smooth, heading, local_pos, rot6d, _, contacts = self._unpack(features)
        rotations = cont6d_to_matrix(rot6d)
        has_rest = self.skeleton.rest_local_rots is not None
        local_rotations = (
            global_rots_to_local_rots(rotations, self.skeleton)
            if need_local_rots or has_rest
            else None
        )
        if self._ardy:
            root = smooth
            root_row = mx.zeros_like(local_pos[..., :1, :])
            root_row = mx.array(root_row)
            root_row[..., 1] = root[..., None, 1]
            local_pos = mx.concatenate((root_row, local_pos), axis=-2)
        translation = smooth * mx.array([1, 0, 1], smooth.dtype)
        from_positions = local_pos + translation[..., None, :]
        if not self._ardy:
            root = from_positions[..., self.skeleton.root_idx, :]
        from_rotations = no_root = None
        if return_all or posed_joints_from == "rotations":
            if has_rest:
                _, from_rotations, no_root = self.skeleton.fk(local_rotations, root)
            else:
                from_rotations, no_root = joints_from_global_rots(
                    rotations, root, self.skeleton, return_norootpos=True
                )
        result = dict(
            local_rot_mats=local_rotations,
            global_rot_mats=rotations,
            posed_joints=(
                from_rotations if posed_joints_from == "rotations" else from_positions
            ),
            posed_joints_norootpos=no_root,
            root_positions=root,
            smooth_root_pos=smooth,
            foot_contacts=contacts > contact_threshold,
            global_root_heading=heading,
        )
        if return_all:
            result["all_positions"] = dict(
                positions=from_positions, rotations=from_rotations
            )
        if unbatched:
            result = {
                k: (
                    {n: x[0] if x is not None else None for n, x in v.items()}
                    if isinstance(v, dict)
                    else v[0] if v is not None else None
                )
                for k, v in result.items()
            }
        return result

    def __call__(
        self,
        local_joint_rots,
        root_positions,
        to_normalize=True,
        to_canonicalize=False,
        lengths=None,
    ):
        unbatched = local_joint_rots.ndim == 4
        if unbatched:
            local_joint_rots, root_positions = (
                local_joint_rots[None],
                root_positions[None],
            )
        if lengths is None:
            lengths = mx.full(
                (root_positions.shape[0],), root_positions.shape[1], mx.int32
            )
        else:
            lengths = mx.array(lengths, mx.int32).reshape(-1)
        global_rots, positions, root_local = self.skeleton.fk(
            local_joint_rots, root_positions
        )
        right, left = self.skeleton.hip_joint_idx
        hip = positions[..., right, :] - positions[..., left, :]
        angle = mx.arctan2(hip[..., 2], -hip[..., 0])
        heading = mx.stack((mx.cos(angle), mx.sin(angle)), axis=-1)
        smooth = root_positions if self._ardy else get_smooth_root_pos(root_positions)
        offset = root_positions - smooth
        offset = mx.stack(
            (offset[..., 0], root_positions[..., 1], offset[..., 2]), axis=-1
        )
        local_pos = root_local[..., int(self._ardy) :, :] + offset[..., None, :]
        velocities = _velocity(positions, self.fps, lengths)
        feet = (
            self.skeleton.left_foot_joint_idx[:2]
            + self.skeleton.right_foot_joint_idx[:2]
        )
        if len(feet) != 4:
            raise ValueError("Motion encoding requires two foot joints per side.")
        foot_velocity = mx.take(velocities, mx.array(feet, mx.int32), axis=-2)
        foot_height = mx.take(positions, mx.array(feet, mx.int32), axis=-2)[..., 1]
        contacts = (
            (mx.linalg.norm(foot_velocity, axis=-1) < 0.15) & (foot_height < 0.10)
        ).astype(positions.dtype)
        result = self._pack(
            (
                smooth,
                heading,
                local_pos,
                matrix_to_cont6d(global_rots),
                velocities,
                contacts,
            )
        )
        if to_canonicalize:
            result = self.canonicalize(result)
        if to_normalize:
            result = self.normalize(result)
        return result[0] if unbatched else result

    def get_root_pos(self, features, fallback_to_smooth=True):
        root = features[..., :3]
        if self._ardy or fallback_to_smooth:
            return root
        offset = features[..., self.slice_dict["local_joints_positions"]][..., :3]
        return root + offset * mx.array([1, 0, 1], root.dtype)

    def get_root_heading_angle(self, features, normalized=False):
        heading = features[..., 3:5]
        if normalized:
            heading = heading * self.global_root_stats.std_eps[3:5].astype(
                features.dtype
            ) + self.global_root_stats.mean[3:5].astype(features.dtype)
        return mx.arctan2(heading[..., 1], heading[..., 0])

    def get_first_heading_angle(self, features, normalized=False):
        return self.get_root_heading_angle(features, normalized)[..., 0]

    def rotate(self, features, angle):
        unbatched = features.ndim == 2
        if unbatched:
            features = features[None]
        angle = mx.broadcast_to(mx.array(angle, features.dtype), (features.shape[0],))
        rotation = angle_to_y_rotation_matrix(angle)
        transposed = mx.swapaxes(rotation, -1, -2)
        root, heading, pos, rot6d, vel, fc = self._unpack(features)
        c, s = mx.cos(angle), mx.sin(angle)
        planar = mx.stack((c, s, -s, c), axis=-1).reshape(-1, 2, 2)
        result = self._pack(
            (
                root @ transposed,
                heading @ planar,
                pos @ transposed[:, None],
                matrix_to_cont6d(rotation[:, None, None] @ cont6d_to_matrix(rot6d)),
                vel @ transposed[:, None],
                fc,
            )
        )
        return result[0] if unbatched else result

    def translate_2d(self, features, translation_2d):
        unbatched = features.ndim == 2
        if unbatched:
            features = features[None]
        translation_2d = mx.broadcast_to(
            mx.array(translation_2d, features.dtype), (features.shape[0], 2)
        )
        shift = mx.stack(
            (
                translation_2d[:, 0],
                mx.zeros((features.shape[0],), features.dtype),
                translation_2d[:, 1],
            ),
            axis=-1,
        )
        result = mx.concatenate(
            (features[..., :3] + shift[:, None], features[..., 3:]), axis=-1
        )
        return result[0] if unbatched else result

    def canonicalize(self, features, normalized=False):
        if normalized:
            features = self.unnormalize(features)
        result = self.rotate(features, -self.get_first_heading_angle(features))
        result = self.translate_2d(result, -result[..., 0, [0, 2]])
        return self.normalize(result) if normalized else result

    def extract_root(self, features):
        return features[..., self.root_slice]

    def extract_body(self, features):
        return features[..., self.body_slice]

    def concat_root_body(self, root, body):
        return mx.concatenate((root, body), axis=-1)

    def feature_indices(self, feature_name):
        s = self.slice_dict[feature_name]
        return list(range(s.start, s.stop))

    def joint_feature_indices(self, feature_name, joint_names, joint_order=None):
        order = joint_order or self.skeleton.bone_order_names
        s = self.slice_dict[feature_name]
        stride = (s.stop - s.start) // len(order)
        return [
            s.start + order.index(n) * stride + i
            for n in joint_names
            for i in range(stride)
        ]

    def create_conditions_from_constraints(
        self, constraints_lst, length, to_normalize=True, device=None
    ):
        from .conditioning import build_condition_dicts

        indices, data = build_condition_dicts(constraints_lst, self.skeleton)
        return self.create_conditions(indices, data, length, to_normalize)

    def create_conditions_from_constraints_batched(
        self, constraints_lst, lengths, to_normalize=True, device=None
    ):
        if not isinstance(lengths, (tuple, list)):
            raise TypeError("Pass Python sequence lengths so conditioning stays lazy.")
        length = max(lengths)
        if (
            isinstance(constraints_lst, (tuple, list))
            and constraints_lst
            and isinstance(constraints_lst[0], list)
        ):
            if len(constraints_lst) != len(lengths):
                raise ValueError(
                    "Per-sample constraint lists must match the batch size."
                )
            results = [
                self.create_conditions_from_constraints(c, length, to_normalize)
                for c in constraints_lst
            ]
            return tuple(mx.stack([result[i] for result in results]) for i in (0, 1))
        observed, mask = self.create_conditions_from_constraints(
            constraints_lst, length, to_normalize
        )
        shape = (len(lengths),) + observed.shape
        return mx.broadcast_to(observed, shape), mx.broadcast_to(mask, shape)

    def create_conditions(
        self, index_dict, data_dict, length, to_normalize=True, device=None
    ):
        from .conditioning import create_conditions

        return create_conditions(self, index_dict, data_dict, length, to_normalize)


class ArdyMotionRep(KimodoMotionRep):
    """Root-position variant used by Kimodo autoregressive checkpoints."""

    _ardy = True
