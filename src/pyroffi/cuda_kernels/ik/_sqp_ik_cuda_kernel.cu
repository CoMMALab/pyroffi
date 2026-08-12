/**
 * Sequential Quadratic Programming IK CUDA kernel with XLA FFI binding.
 *
 * Implements multi-seed SQP-IK directly (no coarse phase):
 *   - One CUDA thread per seed.
 *   - Fixed pos_weight / ori_weight.
 *   - Jacobi column scaling in the QP matrices.
 *   - Box-constrained QP solved by Cholesky (unconstrained Newton step)
 *     then clamped to joint limits, followed by n_inner_iters active-set
 *     refinement steps that fix bound-hitting joints and re-solve.
 *   - 5-point line search (early exit on sufficient descent).
 *   - Trust-region step-size schedule.
 *   - All-time best-config tracking.
 *   - Multi-EE support: stacked residuals and Jacobians for all EEs.
 *
 * Algorithmic difference from LS-IK
 *   The LM unconstrained Cholesky solve is replaced by an inner projected
 *   gradient loop that enforces joint limits as hard constraints on the step.
 *   This means the step p always satisfies lower <= q+p <= upper, rather
 *   than clamping q+p after the fact.
 *
 * Reuses _ik_cuda_helpers.cuh for SE(3) math, FK, and IK helpers
 * (residual/Jacobian, Cholesky, small math).
 *
 * Numerical stability:
 *   - FK and Jacobian in float32.
 *   - Normal-equation matrix H and gradient g in float64.
 *   - Inner projected gradient loop in float32.
 *
 * Build with:
 *   bash build_kernels/build_sqp_ik_cuda.sh
 */

#include "_ik_cuda_helpers.cuh"
#include "_glass_solve.cuh"
#include "_tier_kernel.cuh"
#include "_collision_cuda_helpers.cuh"
#include "xla/ffi/api/ffi.h"

#include <cmath>
#include <cstring>

namespace ffi = xla::ffi;

// ---------------------------------------------------------------------------
// SQP-IK kernel — one thread per seed
// ---------------------------------------------------------------------------

/**
 * Multi-seed SQP-IK with multi-EE support.
 *
 * Each thread independently refines one seed for max_iter outer iterations.
 * Each outer iteration solves a box-constrained QP via n_inner_iters steps
 * of projected gradient descent, then applies a line search.
 *
 * @param seeds         (n_problems, n_seeds, n_act)   initial configurations
 * @param target_jnts   (n_ee,)                        joint index per EE
 * @param ancestor_masks (n_ee, n_joints)              ancestor bitmask per EE
 * @param target_Ts     (n_problems, n_ee, 7)          target poses
 * @param lower/upper   (n_act,)                       joint limits
 * @param fixed_mask    (n_act,) int32                 1 = frozen joint
 * @param out           (n_problems, n_seeds, n_act)   best configurations
 * @param out_err       (n_problems, n_seeds)          best weighted sq. errors
 * @param out_feasible  (n_problems, n_seeds) int32     1 if the returned config
 *                                                      satisfies the collision
 *                                                      constraint, else 0
 * @param n_inner_iters int                            projected-gradient steps
 * @param pos_weight    scalar                         weight on position residual
 * @param ori_weight    scalar                         weight on orientation residual
 * @param lambda_init   scalar                         initial damping factor
 * @param eps_pos       scalar                         position convergence [m]
 * @param eps_ori       scalar                         orientation convergence [rad]
 * @param max_iter      int                            outer SQP iteration budget
 */
// Launch shape per tier (SQP is register/local-memory heavy; keep blocks modest).
// Hard-constraint stage. The cap on simultaneously-enforced collision rows is a
// storage/occupancy trade (MAX_ACT floats each, thread-local); sweeps bound the
// alternating projection when the constraints and the box conflict.
// Added to a seed's reported error when it ends infeasible, so host-side winner
// selection puts every feasible seed ahead of every infeasible one.
// TODO(task 8): silent correctness cliff -- past this many simultaneously
// violated constraints the extras degrade to the soft term with no runtime
// warning, so the "hard" guarantee quietly weakens. Report it at least.
#define PYROFFI_MAX_COLL_CON   4
#define PYROFFI_COLL_POCS_SWEEPS 8

// OSQP-style ADMM inner QP solve (osqp.org/docs/solver). Indirect form, so the
// system stays SPD and GLASS's Cholesky is used unmodified. Defaults follow
// OSQP's: sigma 1e-6, rho 1, over-relaxation 1.6.
// Fixed sweep count, MEASURED not assumed. Adding OSQP's primal/dual residual
// termination made this ~8% SLOWER at both 1e-5/1e-4 and 1e-3/1e-2 tolerances
// (sqp self-only 381 -> 412-419 ms, across four runs): the criteria never fire
// inside 25 sweeps, so the check is pure overhead. That is itself the finding --
// the QP is NOT solved to convergence here, and the POCS polish below is
// load-bearing rather than cosmetic.
//
// TODO(task 8): if the subproblem needs to be solved properly, the lever is rho
// (rho <- rho*sqrt(r_prim/r_dual)), which requires re-forming AND re-factoring M
// since rho sits inside it. Raising PYROFFI_ADMM_ITERS alone would only pay if
// convergence is close, and the residuals say it is not.
#define PYROFFI_ADMM_ITERS  25
#define PYROFFI_ADMM_SIGMA  1.0e-6f
#define PYROFFI_ADMM_RHO    1.0f
#define PYROFFI_ADMM_ALPHA  1.6f

#define PYROFFI_SQP_THREAD_TPB 32
#define PYROFFI_SQP_BLOCK_TPB  64

// Tiered: one seed per thread / warp / block. See _tier_kernel.cuh — every cooperative
// loop is `for (i = rank; i < n; i += size)`, which collapses to the original
// sequential code at Tier::Thread.
//
// SCOPE NOTE: the ACTIVE-SET box-QP algorithm below is deliberately left as-is (it
// outperforms glass::box_qp). Only the plain Cholesky solves inside it are routed
// through GLASS; the projection, active-set selection, and refinement loop are
// untouched.
template <pyroffi::Tier TIER, uint32_t N>
__global__
void sqp_ik_kernel(
    const float* __restrict__ seeds,
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
    const float* __restrict__ robot_spheres_local,
    const int*   __restrict__ robot_sphere_joint_idx,
    const float* __restrict__ world_spheres,
    const float* __restrict__ world_capsules,
    const float* __restrict__ world_boxes,
    const float* __restrict__ world_halfspaces,
    const float* __restrict__ self_sph_local,   // (K, 4) link-local (x,y,z,r)
    const int*   __restrict__ self_link_start,  // (N + 1) CSR sphere runs
    const int*   __restrict__ self_link_joint,  // (N)     posing joint per link
    const int*   __restrict__ self_pair_i,      // (Ps)    active link pairs
    const int*   __restrict__ self_pair_j,      // (Ps)
    const float* __restrict__ lower,
    const float* __restrict__ upper,
    const int*   __restrict__ fixed_mask,
    float*       __restrict__ out,
    float*       __restrict__ out_err,
    int*         __restrict__ out_feasible,   // (n_problems, n_seeds) 1 = satisfies
    int   n_problems, int n_seeds, int n_joints, int n_act, int n_ee,
    int   n_robot_spheres, int n_world_spheres, int n_world_capsules,
    int   n_world_boxes, int n_world_halfspaces, int n_self_pairs,
    int   max_iter, int n_inner_iters,
    int   enable_collision,
    float collision_weight, float collision_margin,
    float pos_weight, float ori_weight, float lambda_init,
    float eps_pos, float eps_ori)
{
    // ── Shared memory: robot parameters loaded once per block ───────────────
    __shared__ float s_twists        [MAX_JOINTS * 6];
    __shared__ float s_parent_tf     [MAX_JOINTS * 7];
    __shared__ int   s_parent_idx    [MAX_JOINTS];
    __shared__ int   s_act_idx       [MAX_JOINTS];
    __shared__ float s_mimic_mul     [MAX_JOINTS];
    __shared__ float s_mimic_off     [MAX_JOINTS];
    __shared__ int   s_mimic_act_idx [MAX_JOINTS];
    __shared__ int   s_topo_inv      [MAX_JOINTS];
    __shared__ float s_target_Ts     [MAX_EE * 7];
    __shared__ int   s_target_jnts   [MAX_EE];
    __shared__ int   s_ancestor_masks[MAX_EE * MAX_JOINTS];
    __shared__ float s_lower   [MAX_ACT];
    __shared__ float s_upper   [MAX_ACT];
    __shared__ int   s_fixed_mask[MAX_ACT];

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
    __syncthreads();

    PYROFFI_TIER_GROUP_VARS(TIER);
    const int s  = PYROFFI_TIER_SEED_INDEX(TIER);
    if (s >= n_seeds) return;   // group-uniform: whole thread/warp/block retires
    const int gs = p * n_seeds + s;

    // One shared solve slot per resident seed, REUSED for the initial Newton solve and
    // for every active-set refinement solve (they are strictly sequential, so they
    // cannot alias). Thread tier: local, so nvcc can promote it.
    constexpr int SLOTS  = PYROFFI_TIER_SLOTS(TIER, N);
    constexpr int SMEM_N = PYROFFI_TIER_SMEM_N(TIER, N);
    __shared__ double sh_A   [SLOTS][SMEM_N * SMEM_N];
    __shared__ double sh_rhs [SLOTS][SMEM_N];
    __shared__ int    sh_fail[SLOTS];
    const int slot = PYROFFI_TIER_SLOT(TIER);
    double  A_local[(TIER == pyroffi::Tier::Thread) ? N * N : 1];
    double  rhs_local[(TIER == pyroffi::Tier::Thread) ? N : 1];
    double* A_g   = (TIER == pyroffi::Tier::Thread) ? A_local   : sh_A[slot];
    double* rhs_g = (TIER == pyroffi::Tier::Thread) ? rhs_local : sh_rhs[slot];

    // ── Thread-private weight vector ─────────────────────────────────────
    float W[6];
    W[0] = pos_weight; W[1] = pos_weight; W[2] = pos_weight;
    W[3] = ori_weight; W[4] = ori_weight; W[5] = ori_weight;

    // ── Thread-private state ─────────────────────────────────────────────
    float cfg[MAX_ACT], best_cfg[MAX_ACT];
    float T_world[MAX_JOINTS * 7];
    float r[6 * MAX_EE];
    float J[6 * MAX_EE * MAX_ACT];

    for (int a = 0; a < n_act; a++) cfg[a] = seeds[gs * n_act + a];
    for (int a = 0; a < n_act; a++) best_cfg[a] = cfg[a];

    // Initial weighted error.
    compute_multi_ee_residual_and_jacobian(
        cfg, T_world,
        s_twists, s_parent_tf, s_parent_idx, s_act_idx,
        s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
        s_target_jnts, s_ancestor_masks, s_target_Ts,
        n_joints, n_act, n_ee, r, J);
    float best_err = 0.0f;
    for (int ee = 0; ee < n_ee; ee++)
        for (int k = 0; k < 6; k++) { float rw = r[ee*6+k] * W[k]; best_err += rw * rw; }

    // Hoisted out of the merit lambda: the QP assembly below needs them too, so
    // that collision enters the *step*, not only the accept/reject test.
    const bool want_self  = n_self_pairs > 0;
    const bool want_world = enable_collision && n_robot_spheres > 0;

    auto collision_raw = [&](const float* cfg_eval, float* T_eval) {
        // Self-collision is independent of world geometry: an arm folded into
        // itself is invalid whether or not there are obstacles. Gating the whole
        // penalty on `enable_collision` (which tracks *world* obstacles) skipped
        // it entirely for obstacle-free problems -- the common case for plain
        // reachability IK, and exactly where a folded solution is most likely to
        // be returned unnoticed.
        if (!want_world && !want_self) return 0.0f;

        fk_single(
            cfg_eval,
            s_twists, s_parent_tf, s_parent_idx, s_act_idx,
            s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
            T_eval,
            n_joints, n_act);

        float pen = 0.0f;
        // `want_self` can be true with world collision off, so the world loop
        // needs its own guard -- otherwise enabling self-collision would
        // silently switch on world penalties the caller did not ask for.
        for (int i = 0; want_world && i < n_robot_spheres; i++) {
            const int jidx = robot_sphere_joint_idx[i];
            if (jidx < 0 || jidx >= n_joints) continue;

            const float* sp = robot_spheres_local + i * 4;
            float local_p[3] = {sp[0], sp[1], sp[2]};
            float world_p[3];
            apply_se3_point(T_eval + jidx * 7, local_p, world_p);
            const float rr = sp[3];

            for (int m = 0; m < n_world_spheres; m++) {
                const float* o = world_spheres + m * 4;
                const float d = sphere_sphere_dist(world_p[0], world_p[1], world_p[2], rr,
                                                   o[0], o[1], o[2], o[3]);
                if (d < collision_margin) {
                    const float diff = d - collision_margin;
                    pen += diff * diff;
                }
            }
            for (int m = 0; m < n_world_capsules; m++) {
                const float* o = world_capsules + m * 7;
                const float d = sphere_capsule_dist(world_p[0], world_p[1], world_p[2], rr,
                                                    o[0], o[1], o[2], o[3], o[4], o[5], o[6]);
                if (d < collision_margin) {
                    const float diff = d - collision_margin;
                    pen += diff * diff;
                }
            }
            for (int m = 0; m < n_world_boxes; m++) {
                const float* o = world_boxes + m * 15;
                const float d = sphere_box_dist(world_p[0], world_p[1], world_p[2], rr,
                                                o[0], o[1], o[2],
                                                o[3], o[4], o[5],
                                                o[6], o[7], o[8],
                                                o[9], o[10], o[11],
                                                o[12], o[13], o[14]);
                if (d < collision_margin) {
                    const float diff = d - collision_margin;
                    pen += diff * diff;
                }
            }
            for (int m = 0; m < n_world_halfspaces; m++) {
                const float* o = world_halfspaces + m * 6;
                const float d = sphere_halfspace_dist(world_p[0], world_p[1], world_p[2], rr,
                                                      o[0], o[1], o[2], o[3], o[4], o[5]);
                if (d < collision_margin) {
                    const float diff = d - collision_margin;
                    pen += diff * diff;
                }
            }
        }

        // Self-collision. This solver checked the robot against the WORLD but
        // never against itself, so a returned configuration could have the arm
        // folded through its own links and still report collision-free. On the
        // Panda, 6.5% of random in-limit configurations self-collide.
        //
        // Shares `self_collision_penalty` with the fused collision kernel so
        // both evaluate identical geometry. `self_sph_local` travels with the
        // tables because `link_start` indexes IT, not `robot_spheres_local`.
        // n_self_pairs == 0 disables the whole thing, and that is the default --
        // existing callers are unaffected until they pass a pair table, which
        // must be SRDF-filtered (without an SRDF the spherized model treats
        // adjacent links as permanently overlapping and rejects everything).
        if (want_self) {
            pen += self_collision_penalty(
                T_eval, self_sph_local, self_link_start, self_link_joint,
                self_pair_i, self_pair_j, n_self_pairs, collision_margin);
        }
        return pen;
    };

    // Weighted merit term (unchanged behaviour for every existing call site).
    auto collision_penalty = [&](const float* cfg_eval, float* T_eval) {
        return collision_weight * collision_raw(cfg_eval, T_eval);
    };

    // Constraint violation, independent of `collision_weight`. This is what makes
    // the constraint HARD: acceptance below is lexicographic on it, so no choice
    // of weight can trade a collision away against pose error.
    auto constraint_violation = [&](const float* cfg_eval, float* T_eval) {
        return collision_raw(cfg_eval, T_eval);
    };

    best_err += collision_penalty(cfg, T_world);
    float best_viol = constraint_violation(cfg, T_world);

    float lam = lambda_init;

    for (int iter = 0; iter < max_iter; iter++) {

        // ── Residual + Jacobian ─────────────────────────────────────────
        compute_multi_ee_residual_and_jacobian(
            cfg, T_world,
            s_twists, s_parent_tf, s_parent_idx, s_act_idx,
            s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
            s_target_jnts, s_ancestor_masks, s_target_Ts,
            n_joints, n_act, n_ee, r, J);

        // Early exit if ALL EEs converged.
        {
            bool all_conv = true;
            for (int ee = 0; ee < n_ee; ee++) {
                if (norm3(r + ee*6) >= eps_pos || norm3(r + ee*6 + 3) >= eps_ori) {
                    all_conv = false;
                    break;
                }
            }
            if (all_conv) break;
        }

        // Apply weights to residual and Jacobian rows.
        float fw[6 * MAX_EE];
        for (int k = 0; k < 6 * n_ee; k++) fw[k] = r[k] * W[k % 6];
        for (int ee = 0; ee < n_ee; ee++)
            for (int k = 0; k < 6; k++)
                for (int a = 0; a < n_act; a++)
                    J[(ee*6+k)*n_act+a] *= W[k];

        float curr_err = 0.0f;
        for (int k = 0; k < 6 * n_ee; k++) curr_err += fw[k] * fw[k];
        curr_err += collision_penalty(cfg, T_world);

        // ── Jacobi column scaling ───────────────────────────────────────
        float col_scale[MAX_ACT];
        for (int a = 0; a < n_act; a++) {
            float sq = 0.0f;
            for (int k = 0; k < 6 * n_ee; k++) { float v = J[k*n_act+a]; sq += v*v; }
            col_scale[a] = sqrtf(sq) + 1e-8f;
        }
        for (int k = 0; k < 6 * n_ee; k++)
            for (int a = 0; a < n_act; a++)
                J[k*n_act+a] /= col_scale[a];

        // ── Form H_s = Js^T Js + λI and g_s = Js^T fw (float64) ───────
        double H_s[MAX_ACT * MAX_ACT];
        double g_s[MAX_ACT];

        // NOT group-parallel, deliberately. H_s/g_s are THREAD-LOCAL buffers that must
        // survive the destructive solves and be re-read by every active-set refinement
        // step, so each lane needs its OWN complete copy. Distributing the (i,j) entries
        // across lanes would leave 31/32 of each lane's H_s uninitialized (caught in
        // testing: thread tier correct, warp/block off by ~1000x). Every lane therefore
        // rebuilds all of H_s, the same redundancy the FK/Jacobian above already has.
        // Only the SOLVE is group-parallel here; its buffer is packed at stride N below.
        for (int i = 0; i < n_act; i++) {
            for (int j = 0; j < n_act; j++) {
                double acc = 0.0;
                for (int k = 0; k < 6 * n_ee; k++)
                    acc += (double)J[k*n_act+i] * (double)J[k*n_act+j];
                H_s[i*n_act + j] = acc;
            }
            double gb = 0.0;
            for (int k = 0; k < 6 * n_ee; k++)
                gb += (double)J[k*n_act+i] * (double)fw[k];
            g_s[i] = gb;
            H_s[i*n_act + i] += (double)lam;
        }

        // ── Collision constraints in the QP ─────────────────────────────
        // Linearised rows kept for the HARD enforcement stage further down:
        //     grad(d)^T p >= margin - d
        // Capped, because each row costs MAX_ACT floats of thread-local storage
        // and these kernels are register-bound. The cap holds the MOST VIOLATED
        // constraints; anything beyond it still contributes its Gauss-Newton
        // penalty term below, so excess constraints degrade to soft rather than
        // vanishing. In practice the simultaneously-violated set is 1-2 pairs.
        float coll_A[PYROFFI_MAX_COLL_CON][MAX_ACT];
        float coll_b[PYROFFI_MAX_COLL_CON];
        int   n_coll = 0;

        // Keep `row` (already Jacobi-scaled) if it is among the worst violations.
        auto record_constraint = [&](const float* row, float viol) {
            int slot_i = n_coll;
            if (n_coll < PYROFFI_MAX_COLL_CON) {
                n_coll++;
            } else {
                int worst = -1; float smallest = viol;
                for (int c = 0; c < PYROFFI_MAX_COLL_CON; c++)
                    if (coll_b[c] < smallest) { smallest = coll_b[c]; worst = c; }
                if (worst < 0) return;      // every stored row is more violated
                slot_i = worst;
            }
            for (int a = 0; a < n_act; a++) coll_A[slot_i][a] = row[a];
            coll_b[slot_i] = viol;
        };

        // Each violated constraint enters as a Gauss-Newton residual
        //     c = sqrt(w) * (margin - d),  active only while d < margin
        // with Jacobian row -sqrt(w) * dd/dq, contributing
        //     H += w * g g^T,   g_s -= w * (margin - d) * g
        // (g_s is +J^T fw here; the solve below negates it). Previously the
        // penalty reached only the merit function, so the QP was built from the
        // pose residual alone and the step could not leave a collision.
        //
        // NOTE: this is still a PENALTY, not a hard constraint. It gives the
        // solver a descent direction; it does not guarantee feasibility. Proper
        // hard constraints belong as linearised inequality rows
        //     grad(d)^T p >= margin - d
        // in the box-QP below, which currently projects onto joint-limit boxes
        // only. That needs a general inequality active-set solve.
        //
        // Not group-strided, matching the H_s/g_s build above: these are
        // thread-local and every lane must hold a complete copy.
        if (want_self || want_world) {
            auto qp_accumulate = [&](float* gg, float viol) {
                for (int a = 0; a < n_act; a++) gg[a] /= col_scale[a];
                // `gg` is now in the same scaled space as p_s, which is the space
                // the hard-enforcement stage projects in.
                record_constraint(gg, viol);
                const double w = (double)collision_weight;
                for (int i = 0; i < n_act; i++) {
                    for (int j = 0; j < n_act; j++)
                        H_s[i*n_act + j] += w * (double)gg[i] * (double)gg[j];
                    g_s[i] -= w * (double)viol * (double)gg[i];
                }
            };

            // Shared descriptions of the collision geometry; the sweep itself lives
            // in _collision_cuda_helpers.cuh so LS and SQP cannot drift apart.
            const RobotChainRefs chain_refs = {
                s_twists, s_parent_idx, s_act_idx, s_mimic_mul, s_mimic_act_idx, n_joints };
            const SelfCollisionRefs self_refs = {
                self_sph_local, self_link_start, self_link_joint,
                self_pair_i, self_pair_j, n_self_pairs };
            const WorldCollisionRefs world_refs = {
                robot_spheres_local, robot_sphere_joint_idx, n_robot_spheres,
                world_spheres, n_world_spheres, world_capsules, n_world_capsules,
                world_boxes, n_world_boxes, world_halfspaces, n_world_halfspaces };

            {
                float g[MAX_ACT];
                collision_gauss_newton_terms(
                    T_world, chain_refs, self_refs, world_refs,
                    want_self, want_world, n_act, collision_margin, g,
                    [&](float* gg, float viol) { qp_accumulate(gg, viol); });
            }
        }

        // ── Box bounds in scaled space ──────────────────────────────────
        float lb_s[MAX_ACT], ub_s[MAX_ACT];
        for (int a = 0; a < n_act; a++) {
            lb_s[a] = (s_lower[a] - cfg[a]) * col_scale[a];
            ub_s[a] = (s_upper[a] - cfg[a]) * col_scale[a];
            if (s_fixed_mask[a]) { lb_s[a] = 0.0f; ub_s[a] = 0.0f; }
        }

        // ── Step 1: unconstrained Newton step via Cholesky (same as LM) ─
        // chol_solve modifies A/b in-place; preserve H_s and g_s for the
        // active-set refinement steps that follow.
        // Copy H_s -> the group's solve buffer, repacking stride n_act -> N. The solve
        // is destructive (A is overwritten with its factor) and the active-set steps
        // below re-read H_s, hence the copy.
        for (int idx = rank; idx < n_act * n_act; idx += size) {
            const int i = idx / n_act, j = idx % n_act;
            A_g[i*(int)N + j] = H_s[i*n_act + j];
        }
        for (int a = rank; a < n_act; a += size) rhs_g[a] = -g_s[a];
        group_sync();

        // Handle fixed joints: unit row/col, zero rhs. Lane `a` owns row a and col a;
        // two masked lanes overlap only where both write 0 — benign.
        for (int a = rank; a < n_act; a += size) {
            if (!s_fixed_mask[a]) continue;
            for (int j = 0; j < n_act; j++)
                A_g[a*(int)N + j] = A_g[j*(int)N + a] = 0.0;
            A_g[a*(int)N + a] = 1.0;
            rhs_g[a] = 0.0;
        }
        pyroffi::pad_tail_identity<double, N>(rank, size, n_act, A_g, rhs_g);
        group_sync();

        pyroffi::tier_posv<TIER, double, N>(A_g, rhs_g, &sh_fail[slot]);
        group_sync();

        float p_s[MAX_ACT];
        for (int a = 0; a < n_act; a++) p_s[a] = (float)rhs_g[a];
        // Every lane must finish reading the solution out of the shared rhs_g before
        // the active-set loop below rebuilds it in place. Without this, a fast lane
        // clobbers rhs_g while a slow lane is still copying it into p_s.
        group_sync();

        // ── Step 2: project onto the feasible set ───────────────────────
        // The joint-limit box AND the linearised collision half-spaces, by
        // alternating projection (POCS). Projecting onto a half-space can push a
        // joint out of its box and clamping to the box can re-violate a
        // half-space, so the two projections alternate until both hold.
        //
        // This makes collision a HARD constraint ON THE SUBPROBLEM: the returned
        // step satisfies every recorded linearised row. Two honest limits. First,
        // the projection is Euclidean rather than in the H-metric, so the step is
        // feasible but is not the exact constrained-QP minimiser -- SQP stays
        // sound because the outer trust region and line search still work on the
        // merit function. Second, feasibility is of the LINEARISATION; the true
        // d(q + p) >= margin follows only as the outer iteration converges and the
        // step shrinks, which is the standard SQP guarantee, not a per-step one.
        //
        // If the rows and the box have no common point the sweeps cannot satisfy
        // everything; the iteration cap bounds the work and leaves the step at the
        // last iterate, with the Gauss-Newton penalty still pulling in the right
        // direction rather than the solver stalling.
        auto project_feasible = [&](float* p) {
            for (int sweep = 0; sweep < PYROFFI_COLL_POCS_SWEEPS; sweep++) {
                bool feasible = true;
                for (int c = 0; c < n_coll; c++) {
                    float ap = 0.0f, nn = 0.0f;
                    for (int a = 0; a < n_act; a++) {
                        ap += coll_A[c][a] * p[a];
                        nn += coll_A[c][a] * coll_A[c][a];
                    }
                    const float viol = coll_b[c] - ap;
                    if (viol <= 1e-7f || nn < 1e-12f) continue;
                    feasible = false;
                    const float step = viol / nn;
                    for (int a = 0; a < n_act; a++) p[a] += step * coll_A[c][a];
                }
                for (int a = 0; a < n_act; a++) p[a] = clampf(p[a], lb_s[a], ub_s[a]);
                if (feasible) break;
            }
        };

        for (int a = 0; a < n_act; a++)
            p_s[a] = clampf(p_s[a], lb_s[a], ub_s[a]);
        project_feasible(p_s);

        // ── Active-set refinement steps ─────────────────────────────────
        // Fix joints that hit their bounds, re-solve for free joints.
        // Converges in 1-2 steps; no-op when joints are not near limits.
        // Box-only refinement. With collision rows present the ADMM solve below
        // replaces it: this loop convexifies bounds ALONE, and re-solving here
        // would just undo the constrained step.
        for (int k = 0; (n_coll == 0) && k < n_inner_iters; k++) {
            float active[MAX_ACT], p_bounded[MAX_ACT];
            for (int a = 0; a < n_act; a++) {
                active[a]    = (p_s[a] <= lb_s[a] + 1e-8f || p_s[a] >= ub_s[a] - 1e-8f)
                               ? 1.0f : 0.0f;
                p_bounded[a] = clampf(p_s[a], lb_s[a], ub_s[a]) * active[a];
            }

            // g_adj = g_s + H_s @ p_bounded; masked system for free joints. Rebuilt
            // into the same group buffer each step (the previous solve destroyed it).
            // `active` is identical on every lane of the group, so the mask each lane
            // applies agrees. Assembled at stride N for GLASS.
            for (int a = rank; a < n_act; a += size) {
                double adj = 0.0;
                for (int b = 0; b < n_act; b++)
                    adj += H_s[a*n_act+b] * (double)p_bounded[b];
                rhs_g[a] = -(g_s[a] + adj) * (double)(1.0f - active[a]);

                for (int b = 0; b < n_act; b++) {
                    A_g[a*(int)N + b] = (active[a] > 0.5f || active[b] > 0.5f)
                                        ? 0.0 : H_s[a*n_act+b];
                }
                if (active[a] > 0.5f) A_g[a*(int)N + a] = 1.0;
            }
            pyroffi::pad_tail_identity<double, N>(rank, size, n_act, A_g, rhs_g);
            group_sync();

            pyroffi::tier_posv<TIER, double, N>(A_g, rhs_g, &sh_fail[slot]);
            group_sync();

            for (int a = 0; a < n_act; a++) {
                p_s[a] = (active[a] > 0.5f)
                         ? p_bounded[a]
                         : clampf((float)rhs_g[a], lb_s[a], ub_s[a]);
            }
            // The re-solve ignores the collision rows, so restore feasibility.
            project_feasible(p_s);
            group_sync();   // p_s consumed before the next step rebuilds A_g
        }

        // ── Hard-constrained QP by ADMM (OSQP indirect form) ────────────
        // Replaces the POCS projection with an actual solve of
        //     min 1/2 p' H p + g' p   s.t.  l <= A p <= u
        // with A = [I ; A_c]: joint-limit box rows plus one linearised collision
        // row per active constraint, grad(d)' p >= margin - d (upper bound +inf).
        //
        // OSQP's INDIRECT form is what makes this fit the existing kernel:
        //     (H + sigma*I + rho*A'A) p = sigma*p - g + A'(rho*z - y)
        // is symmetric positive definite, so GLASS's Cholesky solves it as-is.
        // The direct KKT form is quasi-definite and would need an LDL' that GLASS
        // does not expose -- and GLASS is not to be modified. Because A = [I ; A_c],
        // A'A = I + A_c'A_c, so the matrix never has to be formed explicitly.
        //
        // The matrix is constant across ADMM sweeps, so it is factored ONCE via
        // tier_potrf and each sweep is a tier_potrs against that factor. Without
        // the factor/solve split this would cost 25 Choleskys per SQP step.
        if (n_coll > 0) {
            const float sigma = PYROFFI_ADMM_SIGMA;
            const float rho   = PYROFFI_ADMM_RHO;
            const float alpha = PYROFFI_ADMM_ALPHA;

            for (int idx = rank; idx < n_act * n_act; idx += size) {
                const int i = idx / n_act, j = idx % n_act;
                double m = H_s[i*n_act + j];
                for (int c = 0; c < n_coll; c++)
                    m += (double)rho * (double)coll_A[c][i] * (double)coll_A[c][j];
                if (i == j) m += (double)sigma + (double)rho;
                A_g[i*(int)N + j] = m;
            }
            for (int a = rank; a < n_act; a += size) {
                if (!s_fixed_mask[a]) continue;
                for (int j = 0; j < n_act; j++)
                    A_g[a*(int)N + j] = A_g[j*(int)N + a] = 0.0;
                A_g[a*(int)N + a] = 1.0;
            }
            // Identity-pad the tail. Matrix only -- the rhs is rebuilt per sweep.
            for (int idx = rank; idx < (int)N * (int)N; idx += size) {
                const int i = idx / (int)N, j = idx % (int)N;
                if (i >= n_act || j >= n_act)
                    A_g[i*(int)N + j] = (i == j) ? 1.0 : 0.0;
            }
            group_sync();

            const bool factored = pyroffi::tier_potrf<TIER, double, N>(A_g, &sh_fail[slot]);
            group_sync();

            if (factored) {
                float x[MAX_ACT], zb[MAX_ACT], yb[MAX_ACT];
                float zc[PYROFFI_MAX_COLL_CON], yc[PYROFFI_MAX_COLL_CON];

                // Warm start from the box-projected Newton step already in hand.
                for (int a = 0; a < n_act; a++) {
                    x[a]  = p_s[a];
                    zb[a] = clampf(x[a], lb_s[a], ub_s[a]);
                    yb[a] = 0.0f;
                }
                for (int c = 0; c < n_coll; c++) {
                    float ax = 0.0f;
                    for (int a = 0; a < n_act; a++) ax += coll_A[c][a] * x[a];
                    zc[c] = fmaxf(ax, coll_b[c]);
                    yc[c] = 0.0f;
                }

                for (int it = 0; it < PYROFFI_ADMM_ITERS; it++) {
                    for (int a = rank; a < n_act; a += size) {
                        double v = (double)sigma * (double)x[a] - g_s[a]
                                 + ((double)rho * (double)zb[a] - (double)yb[a]);
                        for (int c = 0; c < n_coll; c++)
                            v += (double)coll_A[c][a]
                               * ((double)rho * (double)zc[c] - (double)yc[c]);
                        rhs_g[a] = s_fixed_mask[a] ? 0.0 : v;
                    }
                    for (int a = rank + n_act; a < (int)N; a += size) rhs_g[a] = 0.0;
                    group_sync();

                    pyroffi::tier_potrs<TIER, double, N>(A_g, rhs_g);
                    group_sync();

                    for (int a = 0; a < n_act; a++) x[a] = (float)rhs_g[a];
                    group_sync();   // rhs_g is rebuilt in place next sweep

                    // z/y updates with over-relaxation. Cheap and identical on
                    // every lane, so they are done redundantly rather than
                    // strided-and-broadcast.
                    for (int a = 0; a < n_act; a++) {
                        const float ax = alpha * x[a] + (1.0f - alpha) * zb[a];
                        const float zn = clampf(ax + yb[a] / rho, lb_s[a], ub_s[a]);
                        yb[a] += rho * (ax - zn);
                        zb[a]  = zn;
                    }
                    for (int c = 0; c < n_coll; c++) {
                        float ax = 0.0f;
                        for (int a = 0; a < n_act; a++) ax += coll_A[c][a] * x[a];
                        ax = alpha * ax + (1.0f - alpha) * zc[c];
                        // Projection onto [b_c, +inf).
                        const float zn = fmaxf(ax + yc[c] / rho, coll_b[c]);
                        yc[c] += rho * (ax - zn);
                        zc[c]  = zn;
                    }

                }
                for (int a = 0; a < n_act; a++) p_s[a] = x[a];
            }

            // ADMM converges TO the feasible set without landing exactly on it,
            // so finish with the exact projection. Cheap, and it is what lets the
            // constraint be stated as satisfied rather than nearly satisfied.
            //
            // No ADMM-level infeasibility certificate is computed, deliberately.
            // Feasibility is verified downstream on the ACTUAL nonlinear distances
            // by `constraint_violation`, and that verdict is what reaches
            // `out_feasible` and the filter. A certificate here would describe the
            // linearised subproblem, which is not the thing callers need to trust;
            // an infeasible subproblem simply yields a poor step that the filter
            // then declines.
            project_feasible(p_s);
        }

        // ── Unscale to original joint space ─────────────────────────────
        float delta[MAX_ACT];
        for (int a = 0; a < n_act; a++)
            delta[a] = p_s[a] / col_scale[a];

        // ── Trust-region step clipping ──────────────────────────────────
        {
            float max_p = 0.0f, max_o = 0.0f;
            for (int ee = 0; ee < n_ee; ee++) {
                max_p = fmaxf(max_p, norm3(r + ee*6));
                max_o = fmaxf(max_o, norm3(r + ee*6 + 3));
            }
            float R;
            if      (max_p > 1e-2f || max_o > 0.6f)  R = 0.38f;
            else if (max_p > 1e-3f || max_o > 0.25f) R = 0.22f;
            else if (max_p > 2e-4f || max_o > 0.08f) R = 0.12f;
            else                                       R = 0.05f;

            float dnorm = 0.0f;
            for (int a = 0; a < n_act; a++) dnorm += delta[a]*delta[a];
            dnorm = sqrtf(dnorm);
            if (dnorm > R) {
                const float scale = R / (dnorm + 1e-18f);
                for (int a = 0; a < n_act; a++) delta[a] *= scale;
            }
        }

        // ── Line search over 5 step sizes ──────────────────────────────
        const float alphas[5] = { 1.0f, 0.5f, 0.25f, 0.1f, 0.025f };
        float best_alpha_err = 1e30f;
        int   best_alpha_idx = 0;
        float r_trial[6 * MAX_EE];

        for (int ai = 0; ai < 5; ai++) {
            float cfg_trial[MAX_ACT];
            for (int a = 0; a < n_act; a++)
                cfg_trial[a] = clampf(cfg[a] + alphas[ai] * delta[a],
                                      s_lower[a], s_upper[a]);

            compute_multi_ee_residual_only(
                cfg_trial, T_world,
                s_twists, s_parent_tf, s_parent_idx, s_act_idx,
                s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
                s_target_jnts, s_target_Ts, n_joints, n_act, n_ee, r_trial);

            float err_trial = 0.0f;
            for (int ee = 0; ee < n_ee; ee++)
                for (int k = 0; k < 6; k++) {
                    float rw = r_trial[ee*6+k] * W[k];
                    err_trial += rw * rw;
                }
            err_trial += collision_penalty(cfg_trial, T_world);
            if (err_trial < best_alpha_err) {
                best_alpha_err = err_trial;
                best_alpha_idx = ai;
            }
        }

        // Best trial configuration.
        float trial_cfg[MAX_ACT];
        for (int a = 0; a < n_act; a++)
            trial_cfg[a] = clampf(cfg[a] + alphas[best_alpha_idx] * delta[a],
                                  s_lower[a], s_upper[a]);

        // ── Accept / reject ─────────────────────────────────────────────
        const bool improved = best_alpha_err < curr_err * (1.0f - 1e-4f);
        if (improved) {
            for (int a = 0; a < n_act; a++) cfg[a] = trial_cfg[a];
            lam = fmaxf(lam * 0.5f, 1e-10f);
        } else {
            lam = fminf(lam * 3.0f, 1e6f);
        }

        // ── Track all-time best ─────────────────────────────────────────
        // Filter acceptance: feasibility first, then pose error. A feasible
        // iterate always beats an infeasible one however small the penalty
        // weight; among infeasible ones the least-violating wins, which is what
        // drives an infeasible seed toward the feasible set. Seeds are sampled
        // at random and many START in collision, so a method that only preserves
        // feasibility would never get off the ground -- this is the restoration
        // phase, folded into the acceptance test.
        const float trial_viol = constraint_violation(trial_cfg, T_world);
        const bool  trial_feas = (trial_viol <= 0.0f);
        const bool  best_feas  = (best_viol  <= 0.0f);
        bool accept_trial;
        if (trial_feas != best_feas)  accept_trial = trial_feas;
        else if (trial_feas)          accept_trial = (best_alpha_err < best_err);
        else                          accept_trial = (trial_viol < best_viol);

        if (accept_trial) {
            best_err  = best_alpha_err;
            best_viol = trial_viol;
            for (int a = 0; a < n_act; a++) best_cfg[a] = trial_cfg[a];
        }
    }

    // Write output.
    // Every lane of the group holds bit-identical state (same seed, same FK, same
    // solved step), so the leader guard avoids a redundant same-value write race.
    if (leader) {
        for (int a = 0; a < n_act; a++) out[gs * n_act + a] = best_cfg[a];
        // Feasibility travels as its own value rather than folded into the error.
        // The host still has to order lexicographically -- an infeasible seed must
        // lose to every feasible one regardless of pose error -- but it now does
        // that against a flag it can also report, instead of against a magic
        // constant baked into a metric callers may threshold on.
        out_err[gs]      = best_err;
        out_feasible[gs] = (best_viol > 0.0f) ? 0 : 1;
    }
}

// ---------------------------------------------------------------------------
// XLA FFI handler
// ---------------------------------------------------------------------------

static ffi::Error SqpIkCudaImpl(
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
    ffi::Buffer<ffi::DataType::F32> self_sph_local,
    ffi::Buffer<ffi::DataType::S32> self_link_start,
    ffi::Buffer<ffi::DataType::S32> self_link_joint,
    ffi::Buffer<ffi::DataType::S32> self_pair_i,
    ffi::Buffer<ffi::DataType::S32> self_pair_j,
    ffi::Buffer<ffi::DataType::F32> lower,
    ffi::Buffer<ffi::DataType::F32> upper,
    ffi::Buffer<ffi::DataType::S32> fixed_mask,
    int64_t max_iter,
    int64_t n_inner_iters,
    float   pos_weight,
    float   ori_weight,
    float   lambda_init,
    float   eps_pos,
    float   eps_ori,
    int64_t enable_collision,
    float   collision_weight,
    float   collision_margin,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_err,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> out_feasible)
{
    const int n_problems = static_cast<int>(seeds.dimensions()[0]);
    const int n_seeds    = static_cast<int>(seeds.dimensions()[1]);
    const int n_act      = static_cast<int>(seeds.dimensions()[2]);
    const int n_joints   = static_cast<int>(twists.dimensions()[0]);
    const int n_ee       = static_cast<int>(target_jnts.dimensions()[0]);
    const int n_robot_spheres = static_cast<int>(robot_spheres_local.dimensions()[0]);
    const int n_world_spheres = static_cast<int>(world_spheres.dimensions()[0]);
    const int n_world_capsules = static_cast<int>(world_capsules.dimensions()[0]);
    const int n_world_boxes = static_cast<int>(world_boxes.dimensions()[0]);
    const int n_world_halfspaces = static_cast<int>(world_halfspaces.dimensions()[0]);
    const int n_self_pairs = static_cast<int>(self_pair_i.dimensions()[0]);

    // N: the compile-time bucket holding n_act (identity-padded). 0 => past MAX_ACT's
    // ceiling, which _build_params.py should already have refused.
    const int bucket = pyroffi::solve_bucket(n_act);
    if (bucket == 0)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "sqp_ik_cuda: n_act exceeds the largest solve bucket (" PYROFFI_SOLVE_MAX_N_STR ").");
    const pyroffi::Tier tier = pyroffi_tier_from_env();

#define PYROFFI_SQP_ARGS                                                       \
        seeds.typed_data(), twists.typed_data(), parent_tf.typed_data(),       \
        parent_idx.typed_data(), act_idx.typed_data(), mimic_mul.typed_data(), \
        mimic_off.typed_data(), mimic_act_idx.typed_data(),                    \
        topo_inv.typed_data(), target_jnts.typed_data(),                       \
        ancestor_masks.typed_data(), target_Ts.typed_data(),                   \
        robot_spheres_local.typed_data(), robot_sphere_joint_idx.typed_data(), \
        world_spheres.typed_data(), world_capsules.typed_data(),               \
        world_boxes.typed_data(), world_halfspaces.typed_data(),               \
        self_sph_local.typed_data(), self_link_start.typed_data(),             \
        self_link_joint.typed_data(), self_pair_i.typed_data(),                \
        self_pair_j.typed_data(),                                              \
        lower.typed_data(), upper.typed_data(), fixed_mask.typed_data(),       \
        out->typed_data(), out_err->typed_data(),                              \
        out_feasible->typed_data(),                                            \
        n_problems, n_seeds, n_joints, n_act, n_ee,                            \
        n_robot_spheres, n_world_spheres, n_world_capsules,                    \
        n_world_boxes, n_world_halfspaces, n_self_pairs,                       \
        static_cast<int>(max_iter), static_cast<int>(n_inner_iters),           \
        static_cast<int>(enable_collision),                                    \
        collision_weight, collision_margin,                                    \
        pos_weight, ori_weight, lambda_init, eps_pos, eps_ori

    PYROFFI_TIER_DISPATCH(sqp_ik_kernel, bucket, tier, n_problems, n_seeds,
                          PYROFFI_SQP_THREAD_TPB, PYROFFI_SQP_BLOCK_TPB, stream,
                          PYROFFI_SQP_ARGS);
#undef PYROFFI_SQP_ARGS

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SqpIkCudaFfi, SqpIkCudaImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // seeds
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // twists
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // parent_tf
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // parent_idx
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // act_idx
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // mimic_mul
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // mimic_off
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // mimic_act_idx
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // topo_inv
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // target_jnts
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // ancestor_masks
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // target_Ts
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // robot_spheres_local
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // robot_sphere_joint_idx
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_spheres
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_capsules
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_boxes
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_halfspaces
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // self_sph_local
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // self_link_start
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // self_link_joint
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // self_pair_i
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // self_pair_j
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // lower
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // upper
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // fixed_mask
        .Attr<int64_t>("max_iter")
        .Attr<int64_t>("n_inner_iters")
        .Attr<float>("pos_weight")
        .Attr<float>("ori_weight")
        .Attr<float>("lambda_init")
        .Attr<float>("eps_pos")
        .Attr<float>("eps_ori")
        .Attr<int64_t>("enable_collision")
        .Attr<float>("collision_weight")
        .Attr<float>("collision_margin")
        .Ret<ffi::Buffer<ffi::DataType::F32>>()   // out cfgs
        .Ret<ffi::Buffer<ffi::DataType::F32>>()   // out errors
        .Ret<ffi::Buffer<ffi::DataType::S32>>()   // out feasible (1 = satisfies)
);
