/**
 * Stein Variational Gradient Descent (SVGD) IK CUDA kernel with Jacobian guidance.
 *
 * Implements Jacobian-guided SVGD for region-based inverse kinematics.
 * The algorithm transports particles to cover the kinematic constraint manifold
 * uniformly using:
 *   - Gradient of log-target (task residuals)
 *   - RBF kernel for particle repulsion
 *
 * Correct SVGD update:
 *   phi(x_i) = (1/N) sum_j [ k(x_j, x_i) * grad_logp(x_j)
 *                            + grad_{x_j} k(x_j, x_i) ]
 * where:
 *   log p(q) = -||r(q)||^2
 *   grad log p(q) = -2 * J^T * r
 *   grad_{x_j} k(x_j, x_i) = k(x_j, x_i) * (x_i - x_j) / h^2
 *
 * Particles live in shared memory so all threads see the same evolving state.
 *
 * Multi-EE support: stacked residuals and Jacobians for all EEs simultaneously.
 *
 * Adaptive bandwidth: at each iteration the RBF bandwidth is set via the median
 * heuristic over current pairwise distances:
 *   h = median(||x_i - x_j||^2) / log(N),   bandwidth = sqrt(h)
 * Each thread sorts its own row of pairwise distances to find the per-particle
 * median; thread 0 then picks the median of those N values.  If the computed
 * bandwidth is degenerate (particles collapsed) the caller-supplied fallback
 * value is used instead.
 *
 * Build with:  bash src/pyroffi/cuda_kernels/build_svgd_region_ik_cuda.sh
 */

#include "_ik_cuda_helpers.cuh"
#include "_collision_cuda_helpers.cuh"

#include "xla/ffi/api/ffi.h"

#include <cmath>
#include <cstring>

namespace ffi = xla::ffi;

// ---------------------------------------------------------------------------
// Compile-time limits (same as other IK kernels)
// ---------------------------------------------------------------------------

#ifndef MAX_PARTICLES
#define MAX_PARTICLES 32
#endif

#ifndef MAX_LBFGS_M
#define MAX_LBFGS_M 8
#endif

// ---------------------------------------------------------------------------
// CUDA kernel: one thread per particle, one block per problem
// ---------------------------------------------------------------------------

__global__
void svgd_region_ik_kernel(
    const float* __restrict__ seeds,         // (n_problems, n_particles, n_act)
    const float* __restrict__ twists,
    const float* __restrict__ parent_tf,
    const int*   __restrict__ parent_idx,
    const int*   __restrict__ act_idx,
    const float* __restrict__ mimic_mul,
    const float* __restrict__ mimic_off,
    const int*   __restrict__ mimic_act_idx,
    const int*   __restrict__ topo_inv,
    const int*   __restrict__ target_jnts,
    const int*   __restrict__ ancestor_masks,
    const float* __restrict__ target_Ts,
    const float* __restrict__ robot_spheres_local,    // (n_rs, 4)
    const int*   __restrict__ robot_sphere_joint_idx, // (n_rs,)
    const float* __restrict__ world_spheres,    // (Ms, 4)
    const float* __restrict__ world_capsules,   // (Mc, 7)
    const float* __restrict__ world_boxes,      // (Mb, 15)
    const float* __restrict__ world_halfspaces, // (Mh, 6)
    const int*   __restrict__ self_pair_i,      // (n_self,)
    const int*   __restrict__ self_pair_j,      // (n_self,)
    const float* __restrict__ lower,
    const float* __restrict__ upper,
    const int*   __restrict__ fixed_mask,
    int64_t n_iters,
    float bandwidth,
    float step_size,
    float*       __restrict__ out,
    float*       __restrict__ out_err,
    float*       __restrict__ out_ee,     // (n_problems * n_particles, 3)
    float*       __restrict__ out_target, // (n_problems * n_particles, 3)
    int n_problems, int n_particles, int n_joints, int n_act, int n_ee,
    int n_robot_spheres, int n_world_spheres, int n_world_capsules,
    int n_world_boxes, int n_world_halfspaces, int n_self_pairs,
    int enable_collision,
    float collision_weight, float collision_margin)
{
    // ---- Shared memory: robot parameters (loaded once per block) ----
    __shared__ float s_twists       [MAX_JOINTS * 6];
    __shared__ float s_parent_tf    [MAX_JOINTS * 7];
    __shared__ int   s_parent_idx   [MAX_JOINTS];
    __shared__ int   s_act_idx      [MAX_JOINTS];
    __shared__ float s_mimic_mul    [MAX_JOINTS];
    __shared__ float s_mimic_off    [MAX_JOINTS];
    __shared__ int   s_mimic_act_idx[MAX_JOINTS];
    __shared__ int   s_topo_inv     [MAX_JOINTS];
    __shared__ float s_target_Ts   [MAX_EE * 7];
    __shared__ int   s_target_jnts [MAX_EE];
    __shared__ int   s_ancestor_masks[MAX_EE * MAX_JOINTS];
    __shared__ float s_lower   [MAX_ACT];
    __shared__ float s_upper   [MAX_ACT];
    __shared__ int   s_fixed_mask[MAX_ACT];

    // ---- Shared memory: particle state, gradients, and adaptive bandwidth ----
    __shared__ float s_particles[MAX_PARTICLES * MAX_ACT];
    __shared__ float s_grad_logp[MAX_PARTICLES * MAX_ACT];
    // Per-particle median pairwise distance² (used for bandwidth estimation)
    __shared__ float s_row_med[MAX_PARTICLES];
    // Adaptive bandwidth (written by thread 0, read by all)
    __shared__ float s_bandwidth;

    // Cooperative load of robot parameters
    for (int i = threadIdx.x; i < n_joints * 6; i += blockDim.x) s_twists[i]    = twists[i];
    for (int i = threadIdx.x; i < n_joints * 7; i += blockDim.x) s_parent_tf[i] = parent_tf[i];
    for (int i = threadIdx.x; i < n_joints;     i += blockDim.x) {
        s_parent_idx[i]    = parent_idx[i];
        s_act_idx[i]       = act_idx[i];
        s_mimic_mul[i]     = mimic_mul[i];
        s_mimic_off[i]     = mimic_off[i];
        s_mimic_act_idx[i] = mimic_act_idx[i];
        s_topo_inv[i]      = topo_inv[i];
    }
    for (int i = threadIdx.x; i < n_act; i += blockDim.x) {
        s_lower[i]      = lower[i];
        s_upper[i]      = upper[i];
        s_fixed_mask[i] = fixed_mask[i];
    }
    const int p = blockIdx.y;
    for (int i = threadIdx.x; i < n_ee * 7; i += blockDim.x)
        s_target_Ts[i] = target_Ts[p * n_ee * 7 + i];
    for (int i = threadIdx.x; i < n_ee; i += blockDim.x)
        s_target_jnts[i] = target_jnts[i];
    for (int i = threadIdx.x; i < n_ee * n_joints; i += blockDim.x)
        s_ancestor_masks[i] = ancestor_masks[i];

    // Each thread handles one particle
    const int t = blockIdx.x * blockDim.x + threadIdx.x;

    // Load initial particle positions into shared memory (before first sync)
    if (t < n_particles) {
        for (int a = 0; a < n_act; a++)
            s_particles[t * n_act + a] = seeds[(p * n_particles + t) * n_act + a];
    }

    __syncthreads();

    if (t >= n_particles) return;

    // Per-thread best-solution tracking
    float best_cfg[MAX_ACT];
    float best_err = 1e30f;
    for (int a = 0; a < n_act; a++)
        best_cfg[a] = s_particles[t * n_act + a];

    // Scratch buffers (thread-private to avoid shared-memory pressure)
    float T_world[MAX_JOINTS * 7];
    float r[6 * MAX_EE];
    float J[6 * MAX_EE * MAX_ACT];

    // ---- Main SVGD loop ----
    for (int iter = 0; iter < n_iters; iter++) {

        // ------------------------------------------------------------------
        // Step 1: compute grad_logp for particle t and store in shared memory.
        //
        // Also compute the per-particle median pairwise distance² for the
        // adaptive bandwidth estimate.  Each thread sorts its own row of
        // distances (N ≤ 32 elements) with a simple insertion sort and writes
        // the median to s_row_med[t].  Thread 0 then finds the median of
        // those N values and updates s_bandwidth.
        // ------------------------------------------------------------------
        float* cfg = s_particles + t * n_act;

        // --- Pairwise distances for bandwidth (insertion-sort row t) ---
        float local_dists[MAX_PARTICLES];
        for (int j = 0; j < n_particles; j++) {
            float* p_j = s_particles + j * n_act;
            float dsq = 0.0f;
            for (int a = 0; a < n_act; a++) {
                float d = cfg[a] - p_j[a];
                dsq += d * d;
            }
            local_dists[j] = dsq;
        }
        // Insertion sort — O(N²) but N ≤ 32 so very cheap
        for (int i = 1; i < n_particles; i++) {
            float key = local_dists[i];
            int k = i - 1;
            while (k >= 0 && local_dists[k] > key) {
                local_dists[k + 1] = local_dists[k];
                --k;
            }
            local_dists[k + 1] = key;
        }
        // After sorting, local_dists[0] == 0 (self-distance).
        // Use index n_particles/2 so the self-distance doesn't dominate.
        s_row_med[t] = local_dists[n_particles / 2];

        // --- Jacobian / residual for grad_logp ---
        compute_multi_ee_residual_and_jacobian(
            cfg, T_world,
            s_twists, s_parent_tf, s_parent_idx, s_act_idx,
            s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
            s_target_jnts, s_ancestor_masks, s_target_Ts,
            n_joints, n_act, n_ee, r, J);

        // grad_logp = d/dq [-||r||^2] = -2 * J^T * r
        for (int a = 0; a < n_act; a++) {
            float g = 0.0f;
            for (int k = 0; k < 6 * n_ee; k++)
                g += J[k * n_act + a] * r[k];
            s_grad_logp[t * n_act + a] = -2.0f * g;
        }

        // ------------------------------------------------------------------
        // Collision: scalar penalty + analytical gradient (option B).
        //
        // log p(q) = -||r||^2 - collision_weight * sum_{pairs} max(0, margin-d)^2
        // grad_logp -= collision_weight * sum_{active pairs} 2*(d-margin) * dd/dq
        // For each robot sphere, dd/dq is computed by walking up the parent
        // chain and accumulating n^T (z_j × arm) (revolute) or n^T z_j
        // (prismatic), where n is the unit normal at the obstacle surface.
        // n is obtained via finite differences on the dist function w.r.t.
        // the sphere center (analytical for sphere-sphere self-pairs).
        // ------------------------------------------------------------------
        float coll_pen = 0.0f;
        if (enable_collision && n_robot_spheres > 0) {
            // Cache world position + radius for every robot sphere using the
            // T_world produced by the task FK above.
            float sphere_world[MAX_ROBOT_SPHERES * 4];
            for (int i = 0; i < n_robot_spheres; i++) {
                const int jidx = robot_sphere_joint_idx[i];
                if (jidx < 0 || jidx >= n_joints) {
                    sphere_world[i*4 + 3] = -1.0f;
                    continue;
                }
                const float* sp = robot_spheres_local + i * 4;
                float local_p[3] = {sp[0], sp[1], sp[2]};
                apply_se3_point(T_world + jidx * 7, local_p, sphere_world + i * 4);
                sphere_world[i*4 + 3] = sp[3];
            }

            // Walk the kinematic ancestor chain of `sphere_jnt` and add
            //   contrib = factor * n^T J_p_sphere * sign
            // to grad_logp.  J_p column for joint j at sphere position
            // (sx,sy,sz):
            //   revolute  → z_j × (sphere_pos - p_j)
            //   prismatic → z_j
            auto add_grad_for_sphere = [&](
                int sphere_jnt,
                float sx, float sy, float sz,
                float nx, float ny, float nz,
                float factor, float sign)
            {
                int j = sphere_jnt;
                while (j >= 0 && j < n_joints) {
                    const int a1 = s_act_idx[j];
                    const int a2 = s_mimic_act_idx[j];
                    if (a1 < 0 && a2 < 0) { j = s_parent_idx[j]; continue; }

                    const float* tw = s_twists + j * 6;
                    const float ang_sq = tw[3]*tw[3] + tw[4]*tw[4] + tw[5]*tw[5];
                    const float lin_sq = tw[0]*tw[0] + tw[1]*tw[1] + tw[2]*tw[2];
                    const float* T_j = T_world + j * 7;
                    const float* p_j = T_j + 4;

                    float jg_lin[3];
                    if (ang_sq > 1e-6f) {
                        const float inv_ang = 1.0f / sqrtf(ang_sq);
                        const float body_ax[3] = { tw[3]*inv_ang, tw[4]*inv_ang, tw[5]*inv_ang };
                        float z_j[3];
                        quat_rotate(T_j, body_ax, z_j);
                        const float arm_j[3] = { sx - p_j[0], sy - p_j[1], sz - p_j[2] };
                        cross3(z_j, arm_j, jg_lin);
                    } else if (lin_sq > 1e-6f) {
                        const float inv_lin = 1.0f / sqrtf(lin_sq);
                        const float body_ax[3] = { tw[0]*inv_lin, tw[1]*inv_lin, tw[2]*inv_lin };
                        float z_j[3];
                        quat_rotate(T_j, body_ax, z_j);
                        jg_lin[0] = z_j[0]; jg_lin[1] = z_j[1]; jg_lin[2] = z_j[2];
                    } else {
                        j = s_parent_idx[j]; continue;
                    }

                    const float n_dot_jg = nx*jg_lin[0] + ny*jg_lin[1] + nz*jg_lin[2];
                    const float ms = s_mimic_mul[j];
                    const float contrib = sign * factor * ms * n_dot_jg;
                    if (a1 >= 0) s_grad_logp[t*n_act + a1] -= contrib;
                    if (a2 >= 0) s_grad_logp[t*n_act + a2] -= contrib;

                    j = s_parent_idx[j];
                }
            };

            const float fd_h = 1e-4f;
            const float fd_inv = 0.5f / fd_h;

            // Robot-sphere vs world primitives.
            for (int i = 0; i < n_robot_spheres; i++) {
                const float rr = sphere_world[i*4 + 3];
                if (rr < 0.0f) continue;
                const float wx = sphere_world[i*4 + 0];
                const float wy = sphere_world[i*4 + 1];
                const float wz = sphere_world[i*4 + 2];
                const int sphere_jnt = robot_sphere_joint_idx[i];

                for (int m = 0; m < n_world_spheres; m++) {
                    const float* o = world_spheres + m * 4;
                    const float d = sphere_sphere_dist(wx, wy, wz, rr,
                                                       o[0], o[1], o[2], o[3]);
                    if (d >= collision_margin) continue;
                    const float diff = d - collision_margin;
                    coll_pen += diff * diff;
                    const float nx = (sphere_sphere_dist(wx+fd_h, wy, wz, rr, o[0], o[1], o[2], o[3])
                                    - sphere_sphere_dist(wx-fd_h, wy, wz, rr, o[0], o[1], o[2], o[3])) * fd_inv;
                    const float ny = (sphere_sphere_dist(wx, wy+fd_h, wz, rr, o[0], o[1], o[2], o[3])
                                    - sphere_sphere_dist(wx, wy-fd_h, wz, rr, o[0], o[1], o[2], o[3])) * fd_inv;
                    const float nz = (sphere_sphere_dist(wx, wy, wz+fd_h, rr, o[0], o[1], o[2], o[3])
                                    - sphere_sphere_dist(wx, wy, wz-fd_h, rr, o[0], o[1], o[2], o[3])) * fd_inv;
                    const float factor = 2.0f * collision_weight * diff;
                    add_grad_for_sphere(sphere_jnt, wx, wy, wz, nx, ny, nz, factor, 1.0f);
                }
                for (int m = 0; m < n_world_capsules; m++) {
                    const float* o = world_capsules + m * 7;
                    const float d = sphere_capsule_dist(wx, wy, wz, rr,
                                                        o[0], o[1], o[2], o[3], o[4], o[5], o[6]);
                    if (d >= collision_margin) continue;
                    const float diff = d - collision_margin;
                    coll_pen += diff * diff;
                    const float nx = (sphere_capsule_dist(wx+fd_h, wy, wz, rr, o[0], o[1], o[2], o[3], o[4], o[5], o[6])
                                    - sphere_capsule_dist(wx-fd_h, wy, wz, rr, o[0], o[1], o[2], o[3], o[4], o[5], o[6])) * fd_inv;
                    const float ny = (sphere_capsule_dist(wx, wy+fd_h, wz, rr, o[0], o[1], o[2], o[3], o[4], o[5], o[6])
                                    - sphere_capsule_dist(wx, wy-fd_h, wz, rr, o[0], o[1], o[2], o[3], o[4], o[5], o[6])) * fd_inv;
                    const float nz = (sphere_capsule_dist(wx, wy, wz+fd_h, rr, o[0], o[1], o[2], o[3], o[4], o[5], o[6])
                                    - sphere_capsule_dist(wx, wy, wz-fd_h, rr, o[0], o[1], o[2], o[3], o[4], o[5], o[6])) * fd_inv;
                    const float factor = 2.0f * collision_weight * diff;
                    add_grad_for_sphere(sphere_jnt, wx, wy, wz, nx, ny, nz, factor, 1.0f);
                }
                for (int m = 0; m < n_world_boxes; m++) {
                    const float* o = world_boxes + m * 15;
                    const float d = sphere_box_dist(wx, wy, wz, rr,
                                                    o[0], o[1], o[2], o[3], o[4], o[5],
                                                    o[6], o[7], o[8], o[9], o[10], o[11],
                                                    o[12], o[13], o[14]);
                    if (d >= collision_margin) continue;
                    const float diff = d - collision_margin;
                    coll_pen += diff * diff;
                    const float nx = (sphere_box_dist(wx+fd_h, wy, wz, rr, o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7], o[8], o[9], o[10], o[11], o[12], o[13], o[14])
                                    - sphere_box_dist(wx-fd_h, wy, wz, rr, o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7], o[8], o[9], o[10], o[11], o[12], o[13], o[14])) * fd_inv;
                    const float ny = (sphere_box_dist(wx, wy+fd_h, wz, rr, o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7], o[8], o[9], o[10], o[11], o[12], o[13], o[14])
                                    - sphere_box_dist(wx, wy-fd_h, wz, rr, o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7], o[8], o[9], o[10], o[11], o[12], o[13], o[14])) * fd_inv;
                    const float nz = (sphere_box_dist(wx, wy, wz+fd_h, rr, o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7], o[8], o[9], o[10], o[11], o[12], o[13], o[14])
                                    - sphere_box_dist(wx, wy, wz-fd_h, rr, o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7], o[8], o[9], o[10], o[11], o[12], o[13], o[14])) * fd_inv;
                    const float factor = 2.0f * collision_weight * diff;
                    add_grad_for_sphere(sphere_jnt, wx, wy, wz, nx, ny, nz, factor, 1.0f);
                }
                for (int m = 0; m < n_world_halfspaces; m++) {
                    const float* o = world_halfspaces + m * 6;
                    const float d = sphere_halfspace_dist(wx, wy, wz, rr,
                                                          o[0], o[1], o[2], o[3], o[4], o[5]);
                    if (d >= collision_margin) continue;
                    const float diff = d - collision_margin;
                    coll_pen += diff * diff;
                    // sphere-halfspace SDF gradient w.r.t. sphere center is the
                    // halfspace normal — analytical, no FD needed.
                    const float nx = o[0], ny = o[1], nz = o[2];
                    const float factor = 2.0f * collision_weight * diff;
                    add_grad_for_sphere(sphere_jnt, wx, wy, wz, nx, ny, nz, factor, 1.0f);
                }
            }

            // Self-collision pairs (analytical normal: unit vector along
            // sphere-center separation; gradient contributions from both spheres).
            for (int i = 0; i < n_self_pairs; i++) {
                const int a = self_pair_i[i];
                const int b = self_pair_j[i];
                if (a < 0 || a >= n_robot_spheres || b < 0 || b >= n_robot_spheres) continue;
                const float ra = sphere_world[a*4 + 3];
                const float rb = sphere_world[b*4 + 3];
                if (ra < 0.0f || rb < 0.0f) continue;
                const float ax = sphere_world[a*4 + 0];
                const float ay = sphere_world[a*4 + 1];
                const float az = sphere_world[a*4 + 2];
                const float bx = sphere_world[b*4 + 0];
                const float by = sphere_world[b*4 + 1];
                const float bz = sphere_world[b*4 + 2];
                const float d = sphere_sphere_dist(ax, ay, az, ra, bx, by, bz, rb);
                if (d >= collision_margin) continue;
                const float diff = d - collision_margin;
                coll_pen += diff * diff;
                const float dx = ax - bx, dy = ay - by, dz = az - bz;
                const float dist = sqrtf(dx*dx + dy*dy + dz*dz) + 1e-12f;
                const float nx = dx / dist, ny = dy / dist, nz = dz / dist;
                const float factor = 2.0f * collision_weight * diff;
                const int joint_a = robot_sphere_joint_idx[a];
                const int joint_b = robot_sphere_joint_idx[b];
                add_grad_for_sphere(joint_a, ax, ay, az, nx, ny, nz, factor, +1.0f);
                add_grad_for_sphere(joint_b, bx, by, bz, nx, ny, nz, factor, -1.0f);
            }

            coll_pen *= collision_weight;
        }

        // Update best solution (collision-aware).
        float curr_err = 0.0f;
        for (int k = 0; k < 6 * n_ee; k++) curr_err += r[k] * r[k];
        curr_err += coll_pen;
        if (curr_err < best_err) {
            best_err = curr_err;
            for (int a = 0; a < n_act; a++) best_cfg[a] = cfg[a];
        }

        __syncthreads();

        // Thread 0: compute adaptive bandwidth from median of row medians.
        // All other threads are idle here, so the in-place sort of s_row_med
        // is safe — no other thread reads it after the sync above.
        if (t == 0) {
            // Insertion sort of n_particles row-medians
            for (int i = 1; i < n_particles; i++) {
                float key = s_row_med[i];
                int k = i - 1;
                while (k >= 0 && s_row_med[k] > key) {
                    s_row_med[k + 1] = s_row_med[k];
                    --k;
                }
                s_row_med[k + 1] = key;
            }
            float med = s_row_med[n_particles / 2];
            float log_n = logf((float)n_particles);
            // bandwidth = sqrt(median_dist² / log N).
            // Fall back to caller-supplied value when particles have collapsed.
            float h = med / (log_n + 1e-8f);
            s_bandwidth = (h > 1e-8f) ? sqrtf(h) : bandwidth;
        }

        __syncthreads();   // all threads wait for s_bandwidth before Step 2

        // ------------------------------------------------------------------
        // Step 2: compute SVGD phi for particle t using all particles
        //
        //   phi(x_i) = (1/N) sum_j [ k(x_j, x_i) * grad_logp(x_j)
        //                           + grad_{x_j} k(x_j, x_i)        ]
        //
        // where:
        //   k(x_j, x_i)              = exp(-||x_j - x_i||^2 / (2h^2))
        //   grad_{x_j} k(x_j, x_i)  = k * (x_i - x_j) / h^2
        // ------------------------------------------------------------------
        const float bw      = s_bandwidth;
        const float bw_sq   = bw * bw + 1e-8f;

        float phi[MAX_ACT];
        for (int a = 0; a < n_act; a++) phi[a] = 0.0f;

        for (int j = 0; j < n_particles; j++) {
            float* p_j     = s_particles + j * n_act;
            float* glogp_j = s_grad_logp + j * n_act;

            // RBF kernel value k(x_j, x_i)
            float dist_sq = 0.0f;
            for (int a = 0; a < n_act; a++) {
                float d = p_j[a] - cfg[a];
                dist_sq += d * d;
            }
            float k_val = expf(-dist_sq / (2.0f * bw_sq));

            for (int a = 0; a < n_act; a++) {
                // k(x_j, x_i) * grad_logp(x_j)
                phi[a] += k_val * glogp_j[a];
                // grad_{x_j} k(x_j, x_i) = k * (x_i - x_j) / h^2
                phi[a] += k_val * (cfg[a] - p_j[a]) / bw_sq;
            }
        }

        // Normalize by N
        float n_inv = 1.0f / (float)n_particles;
        for (int a = 0; a < n_act; a++) phi[a] *= n_inv;

        __syncthreads();

        // ------------------------------------------------------------------
        // Step 3: update shared particle state
        // ------------------------------------------------------------------
        for (int a = 0; a < n_act; a++) {
            if (!s_fixed_mask[a]) {
                s_particles[t * n_act + a] = clampf(
                    s_particles[t * n_act + a] + step_size * phi[a],
                    s_lower[a], s_upper[a]);
            }
        }

        __syncthreads();
    }

    // ---- Write outputs ----
    const int gs = p * n_particles + t;
    for (int a = 0; a < n_act; a++) out[gs * n_act + a] = best_cfg[a];
    out_err[gs] = best_err;

    // FK on best_cfg to get final EE position
    compute_multi_ee_residual_only(
        best_cfg, T_world,
        s_twists, s_parent_tf, s_parent_idx, s_act_idx,
        s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
        s_target_jnts, s_target_Ts, n_joints, n_act, n_ee, r);
    int tgt0 = s_target_jnts[0];
    out_ee[gs * 3 + 0] = T_world[tgt0 * 7 + 4];
    out_ee[gs * 3 + 1] = T_world[tgt0 * 7 + 5];
    out_ee[gs * 3 + 2] = T_world[tgt0 * 7 + 6];
    out_target[gs * 3 + 0] = s_target_Ts[4];
    out_target[gs * 3 + 1] = s_target_Ts[5];
    out_target[gs * 3 + 2] = s_target_Ts[6];
}

// ---------------------------------------------------------------------------
// XLA FFI handler
// ---------------------------------------------------------------------------

static ffi::Error SvgdRegionIkCudaImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F32> seeds,
    ffi::Buffer<ffi::DataType::F32> twists,
    ffi::Buffer<ffi::DataType::F32> parent_tf,
    ffi::Buffer<ffi::DataType::S32> parent_idx,
    ffi::Buffer<ffi::DataType::S32> act_idx,
    ffi::Buffer<ffi::DataType::F32> mimic_mul,
    ffi::Buffer<ffi::DataType::F32> mimic_off,
    ffi::Buffer<ffi::DataType::S32> mimic_act_idx,
    ffi::Buffer<ffi::DataType::S32> topo_inv,
    ffi::Buffer<ffi::DataType::S32> target_jnts,
    ffi::Buffer<ffi::DataType::S32> ancestor_masks,
    ffi::Buffer<ffi::DataType::F32> target_Ts,
    ffi::Buffer<ffi::DataType::F32> robot_spheres_local,
    ffi::Buffer<ffi::DataType::S32> robot_sphere_joint_idx,
    ffi::Buffer<ffi::DataType::F32> world_spheres,
    ffi::Buffer<ffi::DataType::F32> world_capsules,
    ffi::Buffer<ffi::DataType::F32> world_boxes,
    ffi::Buffer<ffi::DataType::F32> world_halfspaces,
    ffi::Buffer<ffi::DataType::S32> self_pair_i,
    ffi::Buffer<ffi::DataType::S32> self_pair_j,
    ffi::Buffer<ffi::DataType::F32> lower,
    ffi::Buffer<ffi::DataType::F32> upper,
    ffi::Buffer<ffi::DataType::S32> fixed_mask,
    int64_t n_iters,
    float bandwidth,
    float step_size,
    int64_t enable_collision,
    float   collision_weight,
    float   collision_margin,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_err,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_ee,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_target)
{
    const int n_problems = static_cast<int>(seeds.dimensions()[0]);
    const int n_particles = static_cast<int>(seeds.dimensions()[1]);
    const int n_act = static_cast<int>(seeds.dimensions()[2]);
    const int n_joints = static_cast<int>(twists.dimensions()[0]);
    const int n_ee = static_cast<int>(target_jnts.dimensions()[0]);
    const int n_robot_spheres   = static_cast<int>(robot_spheres_local.dimensions()[0]);
    const int n_world_spheres   = static_cast<int>(world_spheres.dimensions()[0]);
    const int n_world_capsules  = static_cast<int>(world_capsules.dimensions()[0]);
    const int n_world_boxes     = static_cast<int>(world_boxes.dimensions()[0]);
    const int n_world_halfspaces= static_cast<int>(world_halfspaces.dimensions()[0]);
    const int n_self_pairs      = static_cast<int>(self_pair_i.dimensions()[0]);

    constexpr int THREADS_MAX = 32;
    const int threads = n_particles < THREADS_MAX ? n_particles : THREADS_MAX;
    const int blocks_x = (n_particles + threads - 1) / threads;

    svgd_region_ik_kernel<<<dim3(blocks_x, n_problems), threads, 0, stream>>>(
        seeds.typed_data(),
        twists.typed_data(),
        parent_tf.typed_data(),
        parent_idx.typed_data(),
        act_idx.typed_data(),
        mimic_mul.typed_data(),
        mimic_off.typed_data(),
        mimic_act_idx.typed_data(),
        topo_inv.typed_data(),
        target_jnts.typed_data(),
        ancestor_masks.typed_data(),
        target_Ts.typed_data(),
        robot_spheres_local.typed_data(),
        robot_sphere_joint_idx.typed_data(),
        world_spheres.typed_data(),
        world_capsules.typed_data(),
        world_boxes.typed_data(),
        world_halfspaces.typed_data(),
        self_pair_i.typed_data(),
        self_pair_j.typed_data(),
        lower.typed_data(),
        upper.typed_data(),
        fixed_mask.typed_data(),
        n_iters,
        bandwidth,
        step_size,
        out->typed_data(),
        out_err->typed_data(),
        out_ee->typed_data(),
        out_target->typed_data(),
        n_problems, n_particles, n_joints, n_act, n_ee,
        n_robot_spheres, n_world_spheres, n_world_capsules,
        n_world_boxes, n_world_halfspaces, n_self_pairs,
        static_cast<int>(enable_collision),
        collision_weight, collision_margin);

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    return ffi::Error::Success();
}

// ---------------------------------------------------------------------------
// Handler registration
// ---------------------------------------------------------------------------

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SvgdRegionIkCudaFfi, SvgdRegionIkCudaImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // seeds
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // twists
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // parent_tf
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // parent_idx
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // act_idx
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // mimic_mul
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // mimic_off
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // mimic_act_idx
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // topo_inv
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // target_jnts
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // ancestor_masks
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // target_Ts
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // robot_spheres_local
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // robot_sphere_joint_idx
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_spheres
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_capsules
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_boxes
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_halfspaces
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // self_pair_i
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // self_pair_j
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // lower
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // upper
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // fixed_mask
        .Attr<int64_t>("n_iters")
        .Attr<float>("bandwidth")
        .Attr<float>("step_size")
        .Attr<int64_t>("enable_collision")
        .Attr<float>("collision_weight")
        .Attr<float>("collision_margin")
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // out cfgs
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // out errors
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // out ee_points
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // out target_points
);
