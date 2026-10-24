# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from NVIDIA Kimodo MotionCorrection (Apache-2.0).
"""Metal band solves and ordered scans for lazy motion correction."""

import mlx.core as mx

_BAND_HEADER = """
inline float trajectory_diag(int t, int n, float p, float v, float a) {
    return p + v * float((t > 0) + (t + 1 < n))
        + a * float((t + 2 < n) + 4 * (t > 0 && t + 1 < n) + (t > 1));
}
inline float trajectory_lower(int t, int n, float v, float a) {
    return -v - 2.0f * a * float((t + 1 < n) + (t > 1));
}
"""

_factor_kernel = mx.fast.metal_kernel(
    name="kimodo_trajectory_factor",
    input_names=["margins", "weights"],
    output_names=["factors", "metadata"],
    header=_BAND_HEADER,
    source="""
    uint b = thread_position_in_grid.x;
    if (b >= B) return;
    const float p = weights[0], v = weights[1], a = weights[2];
    size_t offset = size_t(b) * N;
    float max_diag = 0.0f;
    bool has_constraint = false, has_margin = false;
    for (int t = 0; t < N; ++t) {
        max_diag = max(max_diag, trajectory_diag(t, N, p, v, a));
        has_constraint |= margins[offset + t] >= 0.0f;
        has_margin |= margins[offset + t] > 0.0f;
    }
    float rho = 0.5f * metal::precise::sqrt(max_diag);
    metadata[3 * b] = rho;
    metadata[3 * b + 1] = float(has_constraint);
    metadata[3 * b + 2] = float(has_margin);
    float d_prev = 1.0f, d_prev2 = 1.0f, l_prev = 0.0f;
    for (int t = 0; t < N; ++t) {
        bool free = margins[offset + t] != 0.0f && has_constraint;
        float d = free ? trajectory_diag(t, N, p, v, a)
            + (margins[offset + t] > 0.0f ? rho : 0.0f) : 1.0f;
        float l1 = 0.0f, l2 = 0.0f;
        if (t > 1 && free && margins[offset + t - 2] != 0.0f) {
            l2 = a / d_prev2;
        }
        if (t > 0) {
            float lower = free && margins[offset + t - 1] != 0.0f
                ? trajectory_lower(t, N, v, a) : 0.0f;
            l1 = (lower - l2 * d_prev2 * l_prev) / d_prev;
        }
        d = d - l1 * l1 * d_prev - l2 * l2 * d_prev2;
        size_t f = (offset + t) * 3;
        factors[f] = d;
        factors[f + 1] = l1;
        factors[f + 2] = l2;
        d_prev2 = d_prev;
        d_prev = d;
        l_prev = l1;
    }
    """,
)

_correct_kernel = mx.fast.metal_kernel(
    name="kimodo_trajectory_correct",
    input_names=[
        "reference",
        "observations",
        "margins",
        "weights",
        "factors",
        "metadata",
    ],
    output_names=["result", "rhs", "z", "u"],
    header=_BAND_HEADER,
    source="""
    uint b = threadgroup_position_in_grid.x;
    uint d = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint simd = simdgroup_index_in_threadgroup;
    threadgroup float partial_norms[32];
    const float p = weights[0], v = weights[1], a = weights[2];
    size_t offset = size_t(b) * N;
    float rho = metadata[3 * b];
    bool active = metadata[3 * b + 1] != 0.0f;
    bool has_margin = metadata[3 * b + 2] != 0.0f;
    if (!active) {
        if (d < D) {
            for (int t = 0; t < N; ++t) {
                size_t i = (offset + t) * D + d;
                result[i] = reference[i];
                rhs[i] = z[i] = u[i] = 0.0f;
            }
        }
        return;
    }
    if (d < D) {
        // Solve for displacement relative to the reference. This avoids
        // cancellation in N * reference for an almost constant trajectory.
        for (int t = 0; t < N; ++t) {
            size_t i = (offset + t) * D + d;
            float r = 0.0f;
            for (int j = max(0, t - 2); j <= min(N - 1, t + 2); ++j) {
                if (margins[offset + j] == 0.0f) {
                    float coefficient = abs(j - t) == 2 ? a
                        : (j == t ? trajectory_diag(t, N, p, v, a)
                            : trajectory_lower(max(j, t), N, v, a));
                    size_t k = (offset + j) * D + d;
                    r -= coefficient * (observations[k] - reference[k]);
                }
            }
            rhs[i] = margins[offset + t] == 0.0f ? 0.0f : r;
            z[i] = reference[i];
            u[i] = 0.0f;
            result[i] = reference[i];
        }
    }
    int iterations = has_margin ? ITERATIONS : 1;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        if (d < D) {
            float prev = 0.0f, prev2 = 0.0f;
            // L y = rhs.
            for (int t = 0; t < N; ++t) {
                size_t i = (offset + t) * D + d;
                size_t f = (offset + t) * 3;
                float r = rhs[i];
                if (margins[offset + t] > 0.0f) {
                    r += rho * (z[i] - u[i] - reference[i]);
                }
                float y = r - factors[f + 1] * prev - factors[f + 2] * prev2;
                result[i] = y;
                prev2 = prev;
                prev = y;
            }
            // L^T delta = D^-1 y.
            prev = prev2 = 0.0f;
            for (int t = N - 1; t >= 0; --t) {
                size_t i = (offset + t) * D + d;
                float delta = result[i] / factors[(offset + t) * 3];
                if (t + 1 < N) delta -= factors[(offset + t + 1) * 3 + 1] * prev;
                if (t + 2 < N) delta -= factors[(offset + t + 2) * 3 + 2] * prev2;
                result[i] = delta;
                prev2 = prev;
                prev = delta;
            }
            for (int t = 0; t < N; ++t) {
                size_t i = (offset + t) * D + d;
                result[i] = margins[offset + t] == 0.0f
                    ? observations[i] : reference[i] + result[i];
            }
        }
        if (has_margin) {
            for (int t = 0; t < N; ++t) {
                // Every lane takes the same branch. A single SIMD group
                // handles the usual 1/2/3/6-channel trajectories.
                if (margins[offset + t] > 0.0f) {
                    size_t i = (offset + t) * D + d;
                    float difference = d < D ? result[i] + u[i] - observations[i] : 0.0f;
                    float norm_sq = simd_sum(difference * difference);
                    if (TG > 32) {
                        if (lane == 0) partial_norms[simd] = norm_sq;
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        norm_sq = simd_sum(lane < TG / 32 ? partial_norms[lane] : 0.0f);
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }
                    float norm = metal::precise::sqrt(norm_sq);
                    float scale = norm > margins[offset + t] ? margins[offset + t] / norm : 1.0f;
                    if (d < D) {
                        z[i] = observations[i] + difference * scale;
                        u[i] += result[i] - z[i];
                    }
                }
            }
        }
    }
    """,
)

_smooth_kernel = mx.fast.metal_kernel(
    name="kimodo_smooth_channels",
    input_names=["reference", "mask"],
    output_names=["result"],
    source="""
    uint bd = thread_position_in_grid.x;
    if (bd >= B * D) return;
    uint b = bd / D, d = bd % D;
    size_t offset = size_t(b) * N;
    float previous = reference[offset * D + d];
    for (int t = 0; t < N; ++t) {
        size_t i = (offset + t) * D + d;
        int next = min(t + 1, N - 1);
        float value = reference[i];
        float following = reference[(offset + next) * D + d];
        if (t > 0 && mask[offset + t] > 0.0f && mask[offset + t - 1] == 0.0f) {
            value = 0.5f * (previous + following);
        }
        if (mask[offset + t] > 0.0f && mask[offset + next] == 0.0f) {
            value = 0.5f * (previous + following);
        }
        result[i] = value;
        previous = value;
    }
    """,
)


def correct_trajectories(reference, observations, margins, weights, iterations):
    """Run a masked pentadiagonal LDLᵀ solve and ADMM entirely on the GPU."""
    batch, frames, channels = reference.shape
    factors, metadata = _factor_kernel(
        inputs=[margins, weights],
        template=[("B", batch), ("N", frames)],
        grid=(batch, 1, 1),
        threadgroup=(min(batch, 32), 1, 1),
        output_shapes=[(batch, frames, 3), (batch, 3)],
        output_dtypes=[mx.float32, mx.float32],
    )
    threads = max(32, 1 << (channels - 1).bit_length())
    return _correct_kernel(
        inputs=[reference, observations, margins, weights, factors, metadata],
        template=[
            ("N", frames),
            ("D", channels),
            ("TG", threads),
            ("ITERATIONS", iterations),
        ],
        grid=(batch * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[reference.shape] * 4,
        output_dtypes=[mx.float32] * 4,
    )[0]


def smooth_trajectories(reference, mask):
    """Apply the source's two independent boundary tests in temporal order."""
    batch, frames, channels = reference.shape
    return _smooth_kernel(
        inputs=[reference, mask],
        template=[("B", batch), ("N", frames), ("D", channels)],
        grid=(batch * channels, 1, 1),
        threadgroup=(min(256, batch * channels), 1, 1),
        output_shapes=[reference.shape],
        output_dtypes=[mx.float32],
    )[0]
