/**
 * Adapter from a cricket-traced robot header to the IK kernels' residual/Jacobian layout.
 *
 * Included only in builds compiled with -DPYROFFI_TRACED_ROBOT. `_traced_robot_gen.cuh` is written
 * per robot by pyroffi.cuda_kernels.ik._sqp_ik_traced: cricket's `language="cuda"` output for one
 * end-effector, plus gather_q()/scatter_jacobian() between cricket's joint order and the kernel's
 * solved variables (`cfg`, n_solved of them) and frozen joints (`frz`, carried from the seed).
 *
 * Single end-effector only; the launch must match the traced robot (checked host-side).
 */
#pragma once

#include "_traced_robot_gen.cuh"

namespace pyroffi::traced {

constexpr int n_q = robot::n_q;             // every actuated joint
constexpr int n_frames = robot::n_frames;   // one per pyroffi joint, in joint-index order

static __device__ __forceinline__ void residual(
    const float* __restrict__ cfg, const float* __restrict__ frz,
    const float* __restrict__ target_T, float* __restrict__ r)
{
    float q[n_q], pose[7];
    gather_q(cfg, frz, q);
    robot::ee_pose(q, pose);
    pose_residual(pose, target_T, r);
}

// J is (6, n_solved) row-major, matching compute_residual_and_jacobian's layout.
static __device__ __forceinline__ void residual_and_jacobian(
    const float* __restrict__ cfg, const float* __restrict__ frz,
    const float* __restrict__ target_T, float* __restrict__ r, float* __restrict__ J)
{
    float q[n_q], pose[7], Jt[6 * n_q];
    gather_q(cfg, frz, q);
    robot::ee_pose_jacobian(q, pose, Jt);
    pose_residual(pose, target_T, r);
    scatter_jacobian(Jt, J);
}

// T_world layout of fk_single: (n_joints, 7) [w, x, y, z, tx, ty, tz] per joint.
static __device__ __forceinline__ void frame_poses(
    const float* __restrict__ cfg, const float* __restrict__ frz, float* __restrict__ T_world)
{
    float q[n_q];
    gather_q(cfg, frz, q);
    robot::frame_poses(q, T_world);
}

}  // namespace pyroffi::traced
