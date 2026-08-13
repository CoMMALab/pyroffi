/**
 * Gauss-Newton Least Squares IK CUDA kernel with XLA FFI binding.
 *
 * Implements multi-seed Levenberg-Marquardt IK directly (no coarse phase):
 *   - One CUDA thread per seed.
 *   - Fixed pos_weight / ori_weight instead of adaptive row-equilibration.
 *   - Jacobi column scaling in the normal equations.
 *   - 5-point line search (early exit on sufficient descent).
 *   - Trust-region step-size schedule.
 *   - All-time best-config tracking.
 *   - No stall kicks, no joint-limit prior.
 *   - Multi-EE support: stacked residuals and Jacobians for all EEs.
 *
 * Reuses _ik_cuda_helpers.cuh for SE(3) math, FK, and IK helpers
 * (residual/Jacobian, Cholesky, small math).
 *
 * Numerical stability:
 *   - FK and Jacobian in float32.
 *   - Normal-equation matrix and Cholesky solve in float64.
 *
 * Build with:
 *   bash build_kernels/build_ls_ik_cuda.sh
 */

#include "_ik_cuda_helpers.cuh"
#include "_collision_cuda_helpers.cuh"
#include "_glass_solve.cuh"
#include "_tier_kernel.cuh"
#include "xla/ffi/api/ffi.h"

#include <cmath>
#include <cstring>

namespace ffi = xla::ffi;

// ---------------------------------------------------------------------------
// Compile-time limits (must cover the largest robot you plan to use)
// ---------------------------------------------------------------------------
// Defined in _ik_cuda_helpers.cuh (override by defining before include).

// ---------------------------------------------------------------------------
// LS-IK LM kernel — one seed per thread / warp / block (see pyroffi::Tier)
// ---------------------------------------------------------------------------

// Launch shape per tier. These must agree between the kernel's shared-memory
// sizing and the host launch, so they live here rather than at either site.
//
// LM is register/local-memory heavy (T_world[MAX_JOINTS*7] + J + a double A), so
// every tier keeps blocks modest.
// The warp tier's warps-per-block is N-dependent (its shared A[N*N] per warp caps
// it) — see pyroffi::warp_tier_warps_per_block, which host and device share.
#define PYROFFI_LS_IK_THREAD_TPB 32   // thread tier: 32 seeds/block, 1 warp
#define PYROFFI_LS_IK_BLOCK_TPB  64   // block tier: 1 seed/block, 64 lanes cooperate

// Trial step sizes in the LM line search. The warp/block tiers spread these across
// lanes, so the count also sizes their reduction buffer.
#define N_LS_ALPHAS 5

/**
 * Multi-seed Levenberg-Marquardt IK with multi-EE support.
 *
 * Templated on the PARALLELISM TIER that owns one seed, and on the compile-time
 * normal-equation size `N` (the padded bucket for `n_act` — see _glass_solve.cuh):
 *
 *   Tier::Thread  1 seed per thread  — A_s thread-local; glass::thread::potrf
 *   Tier::Warp    1 seed per warp    — A_s shared per warp; glass::warp::potrf
 *   Tier::Block   1 seed per block   — A_s shared per block; glass::posv
 *
 * The LM algorithm is IDENTICAL across tiers; only (a) which group owns a seed,
 * (b) where the normal equations live, and (c) which GLASS surface factors them
 * differ. Everything outside the solve runs on the group's LEADER (lane 0 / thread
 * 0); for Tier::Thread every thread is its own leader, so that path is exactly the
 * original kernel. This means the warp/block tiers currently buy a parallel SOLVE
 * against a serial FK/Jacobian — if the autotune shows either tier winning, the
 * next step is to lane-parallelize the Jacobian build and the 5-way line search
 * (both are embarrassingly parallel; the FK chain is not, being parent->child).
 *
 * Ragged-tail safety: the seed index is GROUP-uniform at every tier (warp tier
 * derives it from the warp id, block tier from blockIdx.x), so the `>= n_seeds`
 * early return retires a whole group at once and never leaves a barrier with
 * divergent participation. Do not make the seed index depend on the lane.
 *
 * @param seeds        (n_problems, n_seeds, n_act)  initial configurations
 * @param target_jnts  (n_ee,)                       joint index per EE
 * @param ancestor_masks (n_ee, n_joints)             ancestor bitmask per EE
 * @param target_Ts    (n_problems, n_ee, 7)          target poses
 * @param lower/upper  (n_act,)                       joint limits
 * @param fixed_mask   (n_act,) int32                 1 = frozen joint
 * @param out          (n_problems, n_seeds, n_act)   best configurations
 * @param out_err      (n_problems, n_seeds)           best weighted squared errors
 * @param n_ee         int                             number of end-effectors
 * @param pos_weight   scalar                          weight on position residual
 * @param ori_weight   scalar                          weight on orientation residual
 * @param lambda_init  scalar                          initial LM damping
 * @param eps_pos      scalar                          position convergence threshold [m]
 * @param eps_ori      scalar                          orientation convergence threshold [rad]
 * @param max_iter     int                             LM iteration budget
 */
template <pyroffi::Tier TIER, uint32_t N>
__global__
void ls_ik_lm_kernel(
    const float* __restrict__ seeds,
    const float* __restrict__ twists,
    const float* __restrict__ parent_tf,
    const int*   __restrict__ parent_idx,
    const int*   __restrict__ act_idx,
    const float* __restrict__ mimic_mul,
    const float* __restrict__ mimic_off,
    const int*   __restrict__ mimic_act_idx,
    const int*   __restrict__ topo_inv,
    const int*   __restrict__ target_jnts,     // (n_ee,) NEW
    const int*   __restrict__ ancestor_masks,  // (n_ee, n_joints) NEW
    const float* __restrict__ target_Ts,       // (n_problems, n_ee, 7) NEW
    const float* __restrict__ robot_spheres_local,   // (n_rs, 4) [x,y,z,r] in joint frame
    const int*   __restrict__ robot_sphere_joint_idx, // (n_rs,)
    const float* __restrict__ world_spheres,    // (Ms, 4)
    const float* __restrict__ world_capsules,   // (Mc, 7)
    const float* __restrict__ world_boxes,      // (Mb, 15)
    const float* __restrict__ world_halfspaces, // (Mh, 6)
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
    int   n_problems, int n_seeds, int n_joints, int n_act, int n_ee, int max_iter,
    int   n_robot_spheres, int n_world_spheres, int n_world_capsules, int n_world_boxes, int n_world_halfspaces,
    int   n_self_pairs,
    int   enable_collision,
    float pos_weight, float ori_weight, float lambda_init,
    float eps_pos, float eps_ori,
    float collision_weight, float collision_margin)
{
    // ── Shared memory: robot parameters loaded once per block ───────────────
    __shared__ float s_twists       [MAX_JOINTS * 6];
    __shared__ float s_parent_tf    [MAX_JOINTS * 7];
    __shared__ int   s_parent_idx   [MAX_JOINTS];
    __shared__ int   s_act_idx      [MAX_JOINTS];
    __shared__ float s_mimic_mul    [MAX_JOINTS];
    __shared__ float s_mimic_off    [MAX_JOINTS];
    __shared__ int   s_mimic_act_idx[MAX_JOINTS];
    __shared__ int   s_topo_inv     [MAX_JOINTS];
    __shared__ float s_target_Ts    [MAX_EE * 7];
    __shared__ int   s_target_jnts  [MAX_EE];
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

    // ── Group -> seed mapping ────────────────────────────────────────────
    // GROUP-uniform by construction (see the ragged-tail note in the docstring):
    // the warp tier keys off the warp id and the block tier off blockIdx.x, so a
    // group retires together and no barrier sees divergent participation.
    PYROFFI_TIER_GROUP_VARS(TIER);
    const int s = PYROFFI_TIER_SEED_INDEX(TIER);
    if (s >= n_seeds) return;   // group-uniform: whole thread/warp/block retires
    const int gs = p * n_seeds + s;

    // ── Per-seed normal equations ────────────────────────────────────────
    // Residency is dictated by the tier and is NOT interchangeable: the thread
    // tier keeps A in thread-local storage (the whole point — nvcc can promote a
    // small N*N to registers), while the warp/block tiers must place it in SHARED
    // so every cooperating lane sees the same buffer.
    constexpr int SLOTS  = PYROFFI_TIER_SLOTS(TIER, N);
    constexpr int SMEM_N = PYROFFI_TIER_SMEM_N(TIER, N);
    __shared__ double sh_A   [SLOTS][SMEM_N * SMEM_N];
    __shared__ double sh_rhs [SLOTS][SMEM_N];
    __shared__ int    sh_fail[SLOTS];
    // Line-search reduction: one error slot per trial step size. Tiny, and only the
    // warp/block tiers read it (the thread tier reduces in registers).
    __shared__ float  sh_ls  [SLOTS][N_LS_ALPHAS];
    const int slot = PYROFFI_TIER_SLOT(TIER);

    // ── Thread-private weight vector ─────────────────────────────────────
    // W = [pos_weight x3, ori_weight x3] — applied per-EE row block
    float W[6];
    W[0] = pos_weight; W[1] = pos_weight; W[2] = pos_weight;
    W[3] = ori_weight; W[4] = ori_weight; W[5] = ori_weight;

    // ── Thread-private state ─────────────────────────────────────────────
    float cfg[MAX_ACT], best_cfg[MAX_ACT];
    float T_world[MAX_JOINTS * 7];
    float r[6 * MAX_EE];
    float J[6 * MAX_EE * MAX_ACT];  // stacked Jacobian buffer

    for (int a = 0; a < n_act; a++) cfg[a] = seeds[gs * n_act + a];
    for (int a = 0; a < n_act; a++) best_cfg[a] = cfg[a];

    // Initial weighted error (sum over all EEs).
    compute_multi_ee_residual_and_jacobian(
        cfg, T_world,
        s_twists, s_parent_tf, s_parent_idx, s_act_idx,
        s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
        s_target_jnts, s_ancestor_masks, s_target_Ts,
        n_joints, n_act, n_ee, r, J);
    float best_err = 0.0f;
    for (int ee = 0; ee < n_ee; ee++)
        for (int k = 0; k < 6; k++) { float rw = r[ee*6+k] * W[k]; best_err += rw * rw; }

    // Hoisted out of the merit lambda: the normal-equation assembly below needs
    // it too, so that self-collision enters the *step direction* and not only
    // the accept/reject test.
    const bool want_self  = n_self_pairs > 0;
    const bool want_world = enable_collision && n_robot_spheres > 0;

    // Merit term for a config whose FK is ALREADY in `T_eval`.
    //
    // Every caller here has just computed that FK -- the line search via
    // compute_multi_ee_residual_only, the iteration head via
    // compute_multi_ee_residual_and_jacobian -- and the old shape of this
    // lambda recomputed it internally. That is one redundant full FK per merit
    // evaluation, and the merit is evaluated ~6x per LM iteration (once at the
    // head plus once per line-search alpha) against ONE gradient sweep.
    //
    // Measured: ablating the line-search merit entirely cut self-collision cost
    // 250 -> 135 ms, so these evaluations dominate. Reusing the FK takes the
    // same work out without changing a single result.
    auto collision_penalty_at = [&](float* T_eval) {
        // Self-collision is independent of world geometry: an arm folded into
        // itself is invalid whether or not there are obstacles. Gating the
        // whole penalty on `enable_collision` (which tracks *world* obstacles)
        // skipped it entirely for obstacle-free problems -- the common case for
        // plain reachability IK, and exactly where a folded solution is most
        // likely to be returned unnoticed.
        if (!want_world && !want_self) return 0.0f;

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
        // both evaluate identical geometry. n_self_pairs == 0 disables it, and
        // that is the default -- existing callers are unaffected until they
        // pass a pair table. That table must be SRDF-filtered: without an SRDF
        // the spherized model treats adjacent links as permanently overlapping
        // and every configuration would be rejected.
        if (want_self) {
            pen += self_collision_penalty(
                T_eval, self_sph_local, self_link_start, self_link_joint,
                self_pair_i, self_pair_j, n_self_pairs, collision_margin);
        }
        return collision_weight * pen;
    };

    // Kept for callers whose FK is not already current; none remain in this
    // kernel, but the two forms must not silently diverge.
    auto collision_penalty = [&](const float* cfg_eval, float* T_eval) {
        if (!want_world && !want_self) return 0.0f;
        fk_single(cfg_eval,
                  s_twists, s_parent_tf, s_parent_idx, s_act_idx,
                  s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
                  T_eval, n_joints, n_act);
        return collision_penalty_at(T_eval);
    };
    (void)collision_penalty;

    // T_world holds cfg's FK from the residual/Jacobian evaluation above.
    best_err += collision_penalty_at(T_world);

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
            // Pose convergence is not convergence when a collision constraint is
            // active: the arm can sit exactly on target and folded through
            // itself, and exiting here would return that as a converged answer
            // with the constraint never worked off. Open loop means running
            // until everything being solved for has converged, not until the
            // pose has.
            if (all_conv && (want_self || want_world))
                all_conv = collision_penalty_at(T_world) <= 1e-12f;
            if (all_conv) break;
        }

        // Weighted residual fw and apply weights in-place to J rows.
        float fw[6 * MAX_EE];
        for (int k = 0; k < 6 * n_ee; k++) fw[k] = r[k] * W[k % 6];
        for (int ee = 0; ee < n_ee; ee++)
            for (int k = 0; k < 6; k++)
                for (int a = 0; a < n_act; a++)
                    J[(ee*6+k)*n_act+a] *= W[k];

        float curr_err = 0.0f;
        for (int k = 0; k < 6 * n_ee; k++) curr_err += fw[k] * fw[k];
        curr_err += collision_penalty_at(T_world);   // FK from the Jacobian eval

        // ── Jacobi column scaling ───────────────────────────────────────
        float col_scale[MAX_ACT];
        for (int a = 0; a < n_act; a++) {
            float sq = 0.0f;
            for (int k = 0; k < 6 * n_ee; k++) { float v = J[k*n_act+a]; sq += v*v; }
            col_scale[a] = sqrtf(sq) + 1e-8f;
        }
        // Scale J in-place → Js (reuse J buffer).
        for (int k = 0; k < 6 * n_ee; k++)
            for (int a = 0; a < n_act; a++)
                J[k*n_act+a] /= col_scale[a];

        // ── Normal equations + LM damping (float64) ────────────────────
        // Built at stride n_act (as before), then padded up to the compile-time
        // bucket N by pad_identity before the GLASS solve. Thread tier: local, so
        // nvcc can promote it. Warp/block: the group's shared slot.
        double  A_local[(TIER == pyroffi::Tier::Thread) ? N * N : 1];
        double  rhs_local[(TIER == pyroffi::Tier::Thread) ? N : 1];
        double* A_s   = (TIER == pyroffi::Tier::Thread) ? A_local   : sh_A[slot];
        double* rhs_s = (TIER == pyroffi::Tier::Thread) ? rhs_local : sh_rhs[slot];

        // Assemble A = J^T J + lam*I. Every lane of the group holds an identical J
        // (it ran the same FK/Jacobian on the same seed), so any lane can compute any
        // entry — the (i,j) pairs distribute freely with no communication. This is
        // the O(n_act^2 * 6*n_ee) inner product, the heaviest step after the FKs.
        //
        // Assembled at stride N (not n_act) so the GLASS solve reads it directly and
        // no serial repack is needed.
        for (int idx = rank; idx < n_act * n_act; idx += size) {
            const int i = idx / n_act, j = idx % n_act;
            double acc = 0.0;
            for (int k = 0; k < 6 * n_ee; k++)
                acc += (double)J[k*n_act+i] * (double)J[k*n_act+j];
            A_s[i*(int)N + j] = acc + ((i == j) ? (double)lam : 0.0);
        }
        for (int i = rank; i < n_act; i += size) {
            double rb = 0.0;
            for (int k = 0; k < 6 * n_ee; k++)
                rb += (double)J[k*n_act+i] * (double)fw[k];
            rhs_s[i] = -rb;
        }

        // ── Self-collision Gauss-Newton contribution ────────────────────
        // Treat each violated pair as an extra least-squares residual
        //     c_p = sqrt(w) * (margin - d_p),   active only while d_p < margin
        // whose Jacobian row is -sqrt(w) * dd_p/dq. Folding it in gives
        //     A   += w * g g^T
        //     rhs += w * (margin - d_p) * g
        // which steers `delta` toward increasing clearance. Previously the
        // penalty reached only the merit function, so the step direction was
        // computed from the pose residual alone and the solver could not move
        // away from a self-collision -- only refuse to move further in.
        //
        // The rows are never materialised into J: each pair is reduced straight
        // into A/rhs as a rank-1 update, so register use is one g[MAX_ACT] and
        // J keeps its pose-only size.
        //
        // Striding matches the assembly loops above, so at the warp/block tiers
        // the lane that wrote an entry is the lane that updates it. Every lane
        // holds an identical J and recomputes an identical g, exactly as the
        // base assembly already assumes.
        // Fold one violated constraint into the normal equations as a rank-1
        // update. `g` is dd/dq for the constraint and `viol` its violation
        // (margin - d) > 0.
        auto gn_accumulate = [&](float* gg, float viol) {
            // J was Jacobi-scaled in place above; scale g the same way so the
            // collision rows live in the same column space as the pose rows.
            for (int a = 0; a < n_act; a++) gg[a] /= col_scale[a];

            const double w = (double)collision_weight;
            for (int idx = rank; idx < n_act * n_act; idx += size) {
                const int i = idx / n_act, j = idx % n_act;
                A_s[i*(int)N + j] += w * (double)gg[i] * (double)gg[j];
            }
            for (int i = rank; i < n_act; i += size)
                rhs_s[i] += w * (double)viol * (double)gg[i];
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
                [&](float* gg, float viol) { gn_accumulate(gg, viol); });
        }
        group_sync();

        // Mask fixed joints (zero row+col, unit diagonal, zero rhs), one lane per
        // masked index. Lane `a` owns row a and column a; two masked lanes a != a'
        // overlap only at A[a][a'] and A[a'][a], where both write 0 — same value, so
        // the race is benign. No lane but `a` ever writes the diagonal A[a][a].
        for (int a = rank; a < n_act; a += size) {
            if (!s_fixed_mask[a]) continue;
            for (int j = 0; j < n_act; j++)
                A_s[a*(int)N + j] = A_s[j*(int)N + a] = 0.0;
            A_s[a*(int)N + a] = 1.0;
            rhs_s[a] = 0.0;
        }
        // Identity-pad [n_act, N). Same construction as the masking above, so
        // delta[a] == 0 for every padded index and the real DOF are untouched.
        pyroffi::pad_tail_identity<double, N>(rank, size, n_act, A_s, rhs_s);
        group_sync();

        pyroffi::tier_posv<TIER, double, N>(A_s, rhs_s, &sh_fail[slot]);
        group_sync();

        // Unscale. Only indices [0, n_act) are real; the padded tail solved to 0.
        float delta[MAX_ACT];
        for (int a = 0; a < n_act; a++)
            delta[a] = (float)rhs_s[a] / col_scale[a];

        // ── Trust-region step clipping ──────────────────────────────────
        {
            // Use max pos/ori error across all EEs.
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

        // ── Line search over the trial step sizes ──────────────────────
        // Each alpha is an INDEPENDENT full FK + residual evaluation, and the FKs
        // dominate an LM step (5 here vs 1 for the residual/Jacobian above). At the
        // warp/block tiers the trials therefore go one-per-lane rather than all-five
        // on every lane. Only N_LS_ALPHAS lanes have work — the algorithm offers no
        // more independent trials than that, and the FK chain itself is parent->child
        // and cannot be split further — but it removes the redundancy that made these
        // tiers ~10x slower than thread.
        //
        // The thread tier (rank=0, size=1) walks all five in sequence, exactly as before.
        const float alphas[N_LS_ALPHAS] = { 1.0f, 0.5f, 0.25f, 0.1f, 0.025f };
        float best_alpha_err = 1e30f;
        int   best_alpha_idx = 0;
        float r_trial[6 * MAX_EE];

        for (int ai = rank; ai < N_LS_ALPHAS; ai += size) {
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
            err_trial += collision_penalty_at(T_world);  // FK from the residual eval

            if constexpr (TIER == pyroffi::Tier::Thread) {
                // Sole owner of every trial — reduce in registers, as before.
                if (err_trial < best_alpha_err) {
                    best_alpha_err = err_trial;
                    best_alpha_idx = ai;
                }
            } else {
                sh_ls[slot][ai] = err_trial;
            }
        }

        // Reduce the trials. Every lane rescans all N_LS_ALPHAS in index order with a
        // strict `<`, which (a) reproduces the thread tier's tie-break exactly — the
        // LOWEST index wins a tie — and (b) leaves every lane holding the same winner,
        // so no broadcast is needed and the group stays in lockstep for the next
        // iteration's FK. Rescanning 5 floats is cheaper than a shuffle reduction.
        if constexpr (TIER != pyroffi::Tier::Thread) {
            group_sync();
            for (int ai = 0; ai < N_LS_ALPHAS; ai++) {
                if (sh_ls[slot][ai] < best_alpha_err) {
                    best_alpha_err = sh_ls[slot][ai];
                    best_alpha_idx = ai;
                }
            }
        }

        // Compute winning trial configuration.
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
        if (best_alpha_err < best_err) {
            best_err = best_alpha_err;
            for (int a = 0; a < n_act; a++) best_cfg[a] = trial_cfg[a];
        }
    }

    // Write output. At the warp/block tiers every lane of the group ran the same
    // FK/Jacobian on the same seed and read the same solved rhs_s, so all lanes
    // hold bit-identical state here — the leader guard is to avoid a redundant
    // same-value write race, not to select a winner among differing lanes.
    if (leader) {
        for (int a = 0; a < n_act; a++) out[gs * n_act + a] = best_cfg[a];
        out_err[gs] = best_err;
    }
}

// ---------------------------------------------------------------------------
// XLA FFI handler
// ---------------------------------------------------------------------------

static ffi::Error LsIkCudaImpl(
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
    ffi::Buffer<ffi::DataType::S32> target_jnts,     // (n_ee,) NEW
    ffi::Buffer<ffi::DataType::S32> ancestor_masks,  // (n_ee, n_joints) NEW
    ffi::Buffer<ffi::DataType::F32> target_Ts,       // (n_problems, n_ee, 7) NEW
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
    float   pos_weight,
    float   ori_weight,
    float   lambda_init,
    float   eps_pos,
    float   eps_ori,
    int64_t enable_collision,
    float   collision_weight,
    float   collision_margin,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_err)
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

    // ── Tier x N dispatch ────────────────────────────────────────────────
    // N: the compile-time bucket holding n_act (identity-padded). 0 => n_act is
    // past MAX_ACT's ceiling, which _build_params.py should already have refused.
    const int bucket = pyroffi::solve_bucket(n_act);
    if (bucket == 0)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "ls_ik_cuda: n_act exceeds the largest solve bucket (" PYROFFI_SOLVE_MAX_N_STR ").");

    // Tier is a launch-shape decision (grid/block differ per tier), so it cannot be
    // chosen on device.
    const pyroffi::Tier tier = pyroffi_tier_from_env();

// The kernel's argument list, shared verbatim by all (tier, N) instantiations
// below so they cannot drift apart.
#define PYROFFI_LS_IK_ARGS                                                     \
        seeds.typed_data(), twists.typed_data(), parent_tf.typed_data(),       \
        parent_idx.typed_data(), act_idx.typed_data(), mimic_mul.typed_data(), \
        mimic_off.typed_data(), mimic_act_idx.typed_data(),                    \
        topo_inv.typed_data(), target_jnts.typed_data(),                       \
        ancestor_masks.typed_data(), target_Ts.typed_data(),                   \
        robot_spheres_local.typed_data(), robot_sphere_joint_idx.typed_data(), \
        world_spheres.typed_data(), world_capsules.typed_data(),               \
        world_boxes.typed_data(), world_halfspaces.typed_data(),               \
        self_sph_local.typed_data(),                                           \
        self_link_start.typed_data(), self_link_joint.typed_data(),            \
        self_pair_i.typed_data(), self_pair_j.typed_data(),                    \
        lower.typed_data(), upper.typed_data(), fixed_mask.typed_data(),       \
        out->typed_data(), out_err->typed_data(),                              \
        n_problems, n_seeds, n_joints, n_act, n_ee,                            \
        static_cast<int>(max_iter),                                            \
        n_robot_spheres, n_world_spheres, n_world_capsules, n_world_boxes,     \
        n_world_halfspaces, n_self_pairs, static_cast<int>(enable_collision),  \
        pos_weight, ori_weight, lambda_init, eps_pos, eps_ori,                 \
        collision_weight, collision_margin

    PYROFFI_TIER_DISPATCH(ls_ik_lm_kernel, bucket, tier, n_problems, n_seeds,
                          PYROFFI_LS_IK_THREAD_TPB, PYROFFI_LS_IK_BLOCK_TPB, stream,
                          PYROFFI_LS_IK_ARGS);
#undef PYROFFI_LS_IK_ARGS

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    LsIkCudaFfi, LsIkCudaImpl,
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
);
