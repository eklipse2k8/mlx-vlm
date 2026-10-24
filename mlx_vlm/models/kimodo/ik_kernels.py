# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from NVIDIA Kimodo MotionCorrection (Apache-2.0).
"""Fused, lazy Metal forward kinematics and limb solvers."""

from functools import lru_cache

import mlx.core as mx

_HEADER = r"""
float3 normalized3(float3 v) {
    float l = length(v);
    return l == 0.0f ? float3(0) : v / l;
}
float4 normalized4(float4 v) {
    float l=length(v);
    return l == 0.0f ? float4(0) : v/l;
}
float3x3 read_rotation(const device float *r, int j) {
    r += j * 9;
    return float3x3(float3(r[0],r[3],r[6]),
                    float3(r[1],r[4],r[7]), float3(r[2],r[5],r[8]));
}
void write_rotation(device float *r, float3x3 m) {
    for (int row=0; row<3; ++row)
        for (int col=0; col<3; ++col) r[row*3+col] = m[col][row];
}
template<typename P> float3 read_vector(P v, int j) {
    return float3(v[j*3],v[j*3+1],v[j*3+2]);
}
float3x3 quaternion_matrix(float4 q) {
    float x=q.x,y=q.y,z=q.z,w=q.w;
    return float3x3(float3(1-2*(y*y+z*z),2*(x*y+w*z),2*(x*z-w*y)),
                    float3(2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)),
                    float3(2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)));
}
float4 between_vectors(float3 from, float3 to) {
    from=normalized3(from); to=normalized3(to);
    float d=dot(from,to);
    if (d >= 1.0f-1e-6f) return float4(0,0,0,1);
    if (d <= -1.0f+1e-6f) return normalized4(float4(-from.z,from.y,from.x,0));
    float3 c=cross(from,to);
    float4 q=float4(c,d);
    q.w += length(q);
    return normalized4(q);
}
float3x3 axis_rotation(float3 axis, float angle) {
    float ha=angle*.5f;
    return quaternion_matrix(float4(axis*sin(ha),cos(ha)));
}
template<typename P> void joint_global(const device float *rotations, const device float *offsets,
                  P parents, float3 root, int joint,
                  thread float3x3 &rotation, thread float3 &position) {
    rotation=float3x3(1.0f); position=float3(0.0f);
    while (joint >= 0) {
        float3x3 local=read_rotation(rotations,joint);
        position=local*position + (parents[joint] < 0 ? root : read_vector(offsets,joint));
        rotation=local*rotation;
        joint=parents[joint];
    }
}
float vector_angle(float3 l, float3 r) {
    l=normalized3(l); r=normalized3(r);
    return atan2(length(cross(l,r)),dot(l,r));
}
float cosine_angle(float l, float r, float opposite) {
    return acos(clamp((l*l+r*r-opposite*opposite)/(2*l*r),-1.0f,1.0f));
}
"""


@lru_cache(None)
def _limb_kernel(two_bone):
    body = r"""
        uint frame=thread_position_in_grid.x;
        if (frame >= N) return;
        const device float *r=rotations + frame*J*9;
        float3 root=read_vector(roots,frame);
        int b=parents[C], a=TWO_BONE ? parents[b] : b;
        float3x3 ar=read_rotation(r,a), br=read_rotation(r,b);
        float3x3 aw,bw,cw; float3 ap,bp,cp;
        joint_global(r,offsets,parents,root,a,aw,ap);
        joint_global(r,offsets,parents,root,b,bw,bp);
        joint_global(r,offsets,parents,root,C,cw,cp);
        float weight=clamp(weights[frame],0.0f,1.0f);
        float3 target=read_vector(targets,frame);
        if (weight > 0.0f) {
            if (!TWO_BONE) {
                float3 ab=transpose(aw)*(cp-ap), at=transpose(aw)*(target-ap);
                float4 q=between_vectors(ab,at);
                q=normalized4(mix(float4(0,0,0,1),q,weight));
                ar=ar*quaternion_matrix(q);
            } else {
                float3 t=mix(cp,target,weight);
                float lab=length(bp-ap), lbc=length(cp-bp);
                float lat=clamp(length(ap-t),.0001f,(lab+lbc)*.999f);
                if (lab >= .0001f && lbc >= .0001f && lat >= .0001f) {
                    float current_a=vector_angle(ap-bp,ap-cp);
                    float current_b=vector_angle(bp-ap,bp-cp);
                    float desired_a=cosine_angle(lab,lat,lbc);
                    float desired_b=cosine_angle(lab,lbc,lat);
                    float3 h=read_vector(hint,0);
                    float3 axis=cross(cp-ap,bp+bw*h-ap);
                    float axis_length=length(axis);
                    axis=axis_length==0 ? float3(0,0,1) : axis/axis_length;
                    br=br*axis_rotation(transpose(bw)*axis,desired_b-current_b);
                    ar=ar*axis_rotation(transpose(aw)*axis,desired_a-current_a);
                    // Source stage two uses the original end-joint direction.
                    float3x3 a_parent=aw*transpose(read_rotation(r,a));
                    aw=a_parent*ar;
                    float4 q=between_vectors(transpose(aw)*(cp-ap),transpose(aw)*(target-ap));
                    ar=ar*quaternion_matrix(q);
                }
            }
        }
        write_rotation(updates+frame*18,ar);
        write_rotation(updates+frame*18+9,br);
    """
    return mx.fast.metal_kernel(
        name="kimodo_two_bone_ik" if two_bone else "kimodo_one_bone_ik",
        input_names=[
            "rotations",
            "roots",
            "offsets",
            "parents",
            "targets",
            "weights",
            "hint",
        ],
        output_names=["updates"],
        source=body,
        header=_HEADER,
    )


@lru_cache(None)
def _global_kernel():
    return mx.fast.metal_kernel(
        name="kimodo_joint_global",
        input_names=["rotations", "roots", "offsets", "parents"],
        output_names=["world_rotation", "world_position"],
        header=_HEADER,
        source=r"""
            uint frame=thread_position_in_grid.x;
            if (frame >= N) return;
            float3x3 r; float3 p;
            joint_global(rotations+frame*J*9,offsets,parents,read_vector(roots,frame),C,r,p);
            write_rotation(world_rotation+frame*9,r);
            world_position[frame*3]=p.x;
            world_position[frame*3+1]=p.y;
            world_position[frame*3+2]=p.z;
        """,
    )


def joint_global(rotations, roots, rig, joint_index):
    """Return one joint's world rotation and position for all leading frames."""
    shape = rotations.shape[:-3]
    n = roots.size // 3
    result = _global_kernel()(
        inputs=[rotations, roots, rig.local_offsets, rig.parent_indices],
        template=[("N", n), ("J", rotations.shape[-3]), ("C", joint_index)],
        grid=(n, 1, 1),
        threadgroup=(min(n, 128), 1, 1),
        output_shapes=[shape + (3, 3), shape + (3,)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return tuple(result)


def limb_ik(rotations, roots, rig, joint_index, targets, weights, hint, two_bone):
    """Solve a limb per frame and return the two modified local rotations."""
    shape = rotations.shape[:-3]
    n = roots.size // 3
    return _limb_kernel(two_bone)(
        inputs=[
            rotations,
            roots,
            rig.local_offsets,
            rig.parent_indices,
            targets,
            weights,
            hint,
        ],
        template=[
            ("N", n),
            ("J", rotations.shape[-3]),
            ("C", joint_index),
            ("TWO_BONE", two_bone),
        ],
        grid=(n, 1, 1),
        threadgroup=(min(n, 128), 1, 1),
        output_shapes=[shape + (2, 3, 3)],
        output_dtypes=[mx.float32],
    )[0]
