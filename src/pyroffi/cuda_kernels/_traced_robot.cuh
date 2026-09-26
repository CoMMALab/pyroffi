/**
 * Adapter from a cricket-traced robot header to the IK kernels' residual/Jacobian layout.
 *
 * Included only in builds compiled with -DPYROFFI_TRACED_ROBOT. `_traced_robot_gen.cuh` is written
 * per robot by pyroffi.cuda_kernels._traced: cricket's `language="cuda"` output for one
 * end-effector, plus gather_q()/scatter_jacobian() between cricket's joint order and the kernel's
 * solved variables (`cfg`, n_solved of them) and frozen joints (`frz`, carried from the seed).
 *
 * Single end-effector only; the launch must match the traced robot (launch_matches, host-side).
 * Needs pose_residual() from _ik_cuda_helpers.cuh.
 */
#pragma once

#include "_ik_cuda_helpers.cuh"
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

constexpr int frz_len = n_frozen > 0 ? n_frozen : 1;  // size for the kernel's frz[] array

// One seed/output row (every actuated joint, pyroffi order) <-> solved variables + frozen joints.
static __device__ __forceinline__ void load_state(
    const float* __restrict__ row, float* __restrict__ cfg, float* __restrict__ frz)
{
    for (int i = 0; i < n_solved; ++i) cfg[i] = row[solved_idx(i)];
    for (int k = 0; k < n_frozen; ++k) frz[k] = row[frozen_idx(k)];
}

static __device__ __forceinline__ void store_state(
    const float* __restrict__ cfg, const float* __restrict__ frz, float* __restrict__ row)
{
    for (int i = 0; i < n_solved; ++i) row[solved_idx(i)] = cfg[i];
    for (int k = 0; k < n_frozen; ++k) row[frozen_idx(k)] = frz[k];
}

// The call's collision tables, baked into the build, replace the runtime buffers.
static __device__ __forceinline__ void bind_collision_tables(
    const float* __restrict__& robot_spheres_local, const int* __restrict__& robot_sphere_joint_idx,
    const float* __restrict__& self_sph_local, const int* __restrict__& self_link_start,
    const int* __restrict__& self_link_joint, const int* __restrict__& self_pair_i,
    const int* __restrict__& self_pair_j)
{
    robot_spheres_local    = kRobotSpheres;
    robot_sphere_joint_idx = kRobotSphereJoint;
    self_sph_local  = kSelfSph;
    self_link_start = kSelfLinkStart;
    self_link_joint = kSelfLinkJoint;
    self_pair_i     = kSelfPairI;
    self_pair_j     = kSelfPairJ;
}

// Host-side check that a launch matches what this build was traced for.
static inline bool launch_matches(int n_ee, int n_act, int n_joints, int n_robot_spheres_rt,
                                  int n_self_pairs_rt, int n_ws, int n_wc, int n_wb, int n_wh)
{
    return n_ee == 1 && n_act == n_q && n_joints == n_frames &&
           n_robot_spheres_rt == n_robot_spheres && n_self_pairs_rt == n_self_pairs &&
           n_ws == n_world_spheres && n_wc == n_world_capsules && n_wb == n_world_boxes &&
           n_wh == n_world_halfspaces;
}

}  // namespace pyroffi::traced
