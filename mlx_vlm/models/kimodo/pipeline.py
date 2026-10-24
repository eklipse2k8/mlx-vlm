# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from NVIDIA Kimodo (Apache-2.0).
"""Text-to-motion generation with a single lazy GPU graph through decoding."""

from functools import wraps
from pathlib import Path

import mlx.core as mx

from .diffusion import Diffusion, guided_prediction


def gpu_inference(method):
    @wraps(method)
    def call(*args, **kwargs):
        with mx.stream(mx.gpu):
            return method(*args, **kwargs)

    return call


def sanitize_text(text):
    text = " ".join(text.strip().split())
    if not text:
        return text
    first = next((i for i, char in enumerate(text) if char.isalnum()), len(text) - 1)
    text = text[first:].capitalize()
    last = next(
        (
            i
            for i in range(len(text) - 1, -1, -1)
            if text[i].isalnum() or text[i] in ".!?\"])'"
        ),
        0,
    )
    text = text[: last + 1]
    if text[-1] not in ".!?":
        text += "."
    for punctuation in ".!?":
        text = (
            (punctuation + " ")
            .join(
                part[0].capitalize() + part[1:] if part else part
                for piece in text.split(punctuation)
                for part in [piece.strip()]
            )
            .strip()
        )
    return text


class KimodoPipeline:
    def __init__(self, model, text_encoder=None):
        self.model = model
        self.motion_rep = model.motion_rep
        self.text_encoder = text_encoder
        self.diffusion = Diffusion(model.config.num_base_steps)
        self._compiled_model = None

    def _encoder(self):
        if self.text_encoder is None:
            from .language import LLM2VecEncoder

            path = Path(self.model.config.model_path or ".") / "text_encoder"
            if not path.is_dir():
                raise FileNotFoundError(
                    f"Missing native LLM2Vec text encoder at {path}. Convert it with "
                    "--text-encoder, or pass text_encoder/text_features explicitly."
                )
            self.text_encoder = LLM2VecEncoder.from_converted(path)
        return self.text_encoder

    @gpu_inference
    def sample(
        self,
        text_features,
        text_mask,
        lengths,
        steps=100,
        *,
        seed=0,
        first_heading_angle=None,
        motion_mask=None,
        observed_motion=None,
        cfg_type=None,
        cfg_weight=(2.0, 2.0),
        initial_noise=None,
        compile=False,
    ):
        lengths = list(lengths)
        if not lengths or any(not isinstance(n, int) or n < 1 for n in lengths):
            raise ValueError("lengths must be a nonempty sequence of positive integers")
        batch, frames = len(lengths), max(lengths)
        if (
            text_features.shape[0] != batch
            or text_mask.shape != text_features.shape[:2]
        ):
            raise ValueError("Text features and masks must match the motion batch")
        shape = (batch, frames, self.motion_rep.motion_rep_dim)
        if initial_noise is None:
            key = None if seed is None else mx.random.key(seed)
            motion = mx.random.normal(shape, key=key)
        else:
            if initial_noise.shape != shape:
                raise ValueError(f"initial_noise must have shape {shape}")
            motion = initial_noise
        if (motion_mask is None) != (observed_motion is None):
            raise ValueError(
                "motion_mask and observed_motion must be supplied together"
            )
        if motion_mask is not None and (
            motion_mask.shape != shape or observed_motion.shape != shape
        ):
            raise ValueError(f"Constraint features and masks must have shape {shape}")
        pad_mask = mx.arange(frames)[None] < mx.array(lengths)[:, None]
        if first_heading_angle is None:
            first_heading_angle = mx.zeros((batch,))
        else:
            first_heading_angle = mx.broadcast_to(
                mx.array(first_heading_angle), (batch,)
            )
        indices, alphas, previous = self.diffusion.schedule(steps)
        denoiser = self.model
        if compile:
            if self.model.training:
                raise ValueError("Call model.eval() before compiled inference")
            if self._compiled_model is None:
                self._compiled_model = mx.compile(self.model, inputs=self.model.state)
            denoiser = self._compiled_model
        # Only Python metadata controls this loop. No tensor is read back or evaluated.
        for i in reversed(range(steps)):
            prediction = guided_prediction(
                denoiser,
                motion,
                pad_mask,
                text_features,
                text_mask,
                mx.full((batch,), indices[i], dtype=mx.int32),
                first_heading_angle=first_heading_angle,
                motion_mask=motion_mask,
                observed_motion=observed_motion,
                cfg_type=cfg_type or self.model.config.cfg_type,
                cfg_weight=cfg_weight,
            )
            motion = self.diffusion.step(motion, prediction, alphas[i], previous[i])
        return motion

    @gpu_inference
    def generate(
        self,
        prompts,
        num_frames,
        num_denoising_steps=100,
        *,
        num_samples=None,
        seed=0,
        text_features=None,
        text_mask=None,
        constraint_lst=None,
        return_features=False,
        output_skeleton=True,
        multi_prompt=False,
        num_transition_frames=5,
        post_processing=False,
        root_margin=0.04,
        contact_threshold=0.5,
        compile=False,
        **kwargs,
    ):
        if num_samples is not None and (
            not isinstance(num_samples, int) or num_samples < 1
        ):
            raise ValueError("num_samples must be a positive integer")
        if multi_prompt:
            return self._multiprompt(
                prompts,
                num_frames,
                num_denoising_steps,
                num_samples=num_samples,
                seed=seed,
                text_features=text_features,
                text_mask=text_mask,
                constraint_lst=constraint_lst,
                return_features=return_features,
                output_skeleton=output_skeleton,
                num_transition_frames=num_transition_frames,
                post_processing=post_processing,
                root_margin=root_margin,
                contact_threshold=contact_threshold,
                compile=compile,
                **kwargs,
            )
        squeeze = (
            isinstance(prompts, str)
            and isinstance(num_frames, int)
            and num_samples is None
        )
        texts = [prompts] if isinstance(prompts, str) else list(prompts)
        lengths = [num_frames] if isinstance(num_frames, int) else list(num_frames)
        batch = num_samples or max(len(texts), len(lengths))
        if len(texts) == 1:
            texts *= batch
        if len(lengths) == 1:
            lengths *= batch
        if not texts or len(texts) != batch or len(lengths) != batch:
            raise ValueError("Prompts and frame counts must match the sample count")
        texts = [sanitize_text(text) for text in texts]
        if text_features is None:
            text_features, text_lengths = self._encoder()(texts)
            text_mask = (
                mx.arange(text_features.shape[1])[None]
                < mx.array(text_lengths)[:, None]
            )
            active = mx.array([bool(text.strip()) for text in texts])
            text_features = mx.where(active[:, None, None], text_features, 0)
            text_mask = text_mask & active[:, None]
        elif text_mask is None:
            text_mask = mx.ones(text_features.shape[:2], dtype=mx.bool_)
        if constraint_lst:
            observed, mask = self.motion_rep.create_conditions_from_constraints_batched(
                constraint_lst, lengths, to_normalize=True
            )
            kwargs.update(observed_motion=observed, motion_mask=mask)
        features = self.sample(
            text_features,
            text_mask,
            lengths,
            num_denoising_steps,
            seed=seed,
            compile=compile,
            **kwargs,
        )
        if return_features and not post_processing:
            return features[0] if squeeze else features
        output = self.motion_rep.inverse(
            features, is_normalized=True, contact_threshold=contact_threshold
        )
        if post_processing:
            from .postprocess import post_process_motion

            output.update(
                post_process_motion(
                    output["local_rot_mats"],
                    output["root_positions"],
                    output["foot_contacts"],
                    self.motion_rep.skeleton,
                    constraint_lst,
                    root_margin=root_margin,
                    contact_threshold=contact_threshold,
                    compile=compile,
                )
            )
            if return_features:
                corrected = self.motion_rep(
                    output["local_rot_mats"],
                    output["root_positions"],
                    to_normalize=True,
                    lengths=mx.array(lengths, mx.int32),
                )
                return corrected[0] if squeeze else corrected
        if output_skeleton and self.motion_rep.skeleton.output_skeleton is not None:
            output = self.motion_rep.skeleton.output_to_SOMASkeleton77(output)
        if squeeze:
            output = {key: value[0] for key, value in output.items()}
        return output

    def _multiprompt(
        self,
        prompts,
        num_frames,
        steps,
        *,
        num_samples=None,
        seed=0,
        text_features=None,
        text_mask=None,
        constraint_lst=None,
        return_features=False,
        output_skeleton=True,
        num_transition_frames=5,
        post_processing=False,
        root_margin=0.04,
        contact_threshold=0.5,
        compile=False,
        **kwargs,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]
        frames = (
            [num_frames] * len(prompts)
            if isinstance(num_frames, int)
            else list(num_frames)
        )
        if not prompts or len(frames) != len(prompts) or any(n < 1 for n in frames):
            raise ValueError("Sequential prompts need matching positive frame counts")
        if num_transition_frames < 1 or any(
            n < num_transition_frames for n in frames[:-1]
        ):
            raise ValueError("Each preceding segment must cover the transition frames")
        batch = num_samples or 1
        dim = self.motion_rep.motion_rep_dim
        total = sum(frames)
        if post_processing:
            from .postprocess import (
                constraint_batches,
                crop_constraints,
                post_process_motion,
            )

            cleanup_constraints = constraint_batches(constraint_lst, batch)
        if constraint_lst:
            observed, masks = (
                self.motion_rep.create_conditions_from_constraints_batched(
                    constraint_lst, [total] * batch, to_normalize=False
                )
            )
        else:
            observed = mx.zeros((batch, total, dim))
            masks = mx.zeros((batch, total, dim))
        segments = []
        offset = 0
        heading = kwargs.pop("first_heading_angle", None)
        for index, (prompt, length) in enumerate(zip(prompts, frames)):
            segment_observed = observed[:, offset : offset + length]
            segment_mask = masks[:, offset : offset + length]
            origin = None
            transitions = [[] for _ in range(batch)]
            if segments:
                previous = segments.pop()
                history = previous[:, -num_transition_frames:]
                segments.append(previous[:, :-num_transition_frames])
                decoded = self.motion_rep.inverse(history, is_normalized=False)
                planar = decoded["smooth_root_pos"][..., [0, 2]]
                origin = planar[:, 0]
                constraints = []
                for sample in range(batch):
                    pose = dict(
                        frame_indices=list(range(num_transition_frames)),
                        global_joints_positions=decoded["posed_joints"][sample],
                        global_joints_rots=decoded["global_rot_mats"][sample],
                        smooth_root_2d=planar[sample],
                    )
                    constraints.append(
                        [
                            dict(pose, type="fullbody"),
                            dict(
                                pose,
                                type="end-effector",
                                joint_names=[
                                    "LeftHand",
                                    "RightHand",
                                    "LeftFoot",
                                    "RightFoot",
                                ],
                            ),
                        ]
                    )
                transitions = constraints
                past_observed, past_mask = (
                    self.motion_rep.create_conditions_from_constraints_batched(
                        constraints, [num_transition_frames] * batch, to_normalize=False
                    )
                )
                segment_observed = mx.concatenate(
                    (past_observed, segment_observed), axis=1
                )
                segment_mask = mx.concatenate((past_mask, segment_mask), axis=1)
                segment_observed = (
                    self.motion_rep.translate_2d(segment_observed, -origin)
                    * segment_mask
                )
                right, left = self.motion_rep.skeleton.hip_joint_idx
                hip = (
                    decoded["posed_joints"][:, 0, right]
                    - decoded["posed_joints"][:, 0, left]
                )
                heading = mx.arctan2(hip[:, 2], -hip[:, 0])
            normalized = self.generate(
                prompt,
                segment_observed.shape[1],
                steps,
                num_samples=batch,
                seed=None if seed is None else seed + index,
                text_features=text_features,
                text_mask=text_mask,
                compile=compile,
                observed_motion=self.motion_rep.normalize(segment_observed),
                motion_mask=segment_mask,
                first_heading_angle=heading,
                return_features=True,
                **kwargs,
            )
            motion = self.motion_rep.unnormalize(normalized)
            if origin is not None:
                motion = self.motion_rep.translate_2d(motion, origin)
            if post_processing:
                # Correct each segment together with its history before it becomes
                # the next segment's constraint, as in the source pipeline.
                begin = offset - (num_transition_frames if origin is not None else 0)
                segment_constraints = [
                    list(transitions[sample])
                    + crop_constraints(
                        cleanup_constraints[sample], begin, offset + length
                    )
                    for sample in range(batch)
                ]
                decoded = self.motion_rep.inverse(
                    motion, is_normalized=False, contact_threshold=contact_threshold
                )
                corrected = post_process_motion(
                    decoded["local_rot_mats"],
                    decoded["root_positions"],
                    decoded["foot_contacts"],
                    self.motion_rep.skeleton,
                    segment_constraints,
                    root_margin=root_margin,
                    contact_threshold=contact_threshold,
                    compile=compile,
                )
                motion = self.motion_rep(
                    corrected["local_rot_mats"],
                    corrected["root_positions"],
                    to_normalize=False,
                    lengths=mx.full((batch,), motion.shape[1], dtype=mx.int32),
                )
            elif origin is not None:
                blend = mx.linspace(1, 0, num_transition_frames)[:, None]
                segments.append(
                    history * blend + motion[:, :num_transition_frames] * (1 - blend)
                )
                motion = motion[:, num_transition_frames:]
            segments.append(motion)
            offset += length
        features = self.motion_rep.normalize(mx.concatenate(segments, axis=1))
        if return_features:
            return features[0] if num_samples is None else features
        output = self.motion_rep.inverse(
            features, is_normalized=True, contact_threshold=contact_threshold
        )
        if output_skeleton and self.motion_rep.skeleton.output_skeleton is not None:
            output = self.motion_rep.skeleton.output_to_SOMASkeleton77(output)
        return (
            {key: value[0] for key, value in output.items()}
            if num_samples is None
            else output
        )
