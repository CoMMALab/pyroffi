/**
 * Fused FK + *binary* collision-check CUDA kernel for pyroffi, with XLA/JAX FFI.
 *
 * This is the binary edge-validation counterpart to the differentiable signed-
 * distance kernels in _collision_cuda_kernel.cu.  It mirrors pRRTC's SIMT
 * collision checker (https://github.com/CoMMALab/pRRTC):
 *
 *   - Forward kinematics is *fused into* the collision kernel (pRRTC's `fkcc`):
 *     each block runs FK for one configuration in shared memory, then transforms
 *     and checks the collision spheres in the same kernel — no intermediate
 *     sphere-position array is ever written to global memory.
 *
 *   - Two-stage approximate→fine checking (pRRTC's `*_approx` then fine pass):
 *     a coarse sphere model (typically one enclosing sphere per link) is checked
 *     first.  Because the coarse spheres enclose the fine spheres, a coarse
 *     "clear" result proves the fine geometry is clear too, so the fine pass is
 *     skipped entirely.  Where the coarse model *does* report a possible
 *     collision, only the flagged links are re-checked with the fine geometry.
 *
 *   - Early exit: the instant any fine sphere is found in collision the block
 *     stops doing collision work (shared `cc` flag + per-thread `break`) and the
 *     configuration is reported as invalid.  This is the speedup that the
 *     distance-matrix SDF kernels fundamentally cannot provide.
 *
 * Output is a single int32 per configuration: 1 == collision-free, 0 == in
 * collision (world OR self).  This matches pRRTC's per-edge "edge_good".
 *
 * Robot model is supplied per-call (pyroffi is robot-agnostic):
 *   FK model arrays — identical to _fk_cuda_kernel.cu (twists, parent_tf, ...).
 *   link_parent_joint [NL]  — joint index whose world transform is link l's pose
 *                              (-1 → identity / base link).
 *   Sphere geometry is link-LOCAL (pre-FK), laid out uniformly [S, NL, 4] →
 *   flattened to [K, 4] with k = s * NL + n, component (x, y, z, r).  Padding
 *   spheres have radius < 0 and are skipped (matches RobotCollisionSpherized).
 *
 * World geometry arrays match _collision_cuda_kernel.cu / _cuda_collision.py:
 *   spheres   [Ms, 4], capsules [Mc, 7], boxes [Mb, 15], halfspaces [Mh, 6].
 *
 * Build: bash build_kernels/build_collision_binary_cuda.sh
 */

#include "xla/ffi/api/ffi.h"
#include "_collision_cuda_helpers.cuh"  // *_dist primitives, apply_se3_point, fk_single

namespace ffi = xla::ffi;

// Static shared-memory bounds.  Increase and rebuild for larger robots.
#define BCC_MAX_JOINTS 64
#define BCC_MAX_LINKS  64
#define BCC_THREADS    64

// ── Device predicate: robot sphere vs. the whole environment ──────────────────
//
// Returns true on the first obstacle found in collision (distance < 0).  Walking
// obstacle types in order with an early `return` mirrors pRRTC's
// `sphere_environment_in_collision`.

__device__ __forceinline__ bool sphere_world_hit(
    float px, float py, float pz, float r,
    const float* __restrict__ ws, int Ms,
    const float* __restrict__ wc, int Mc,
    const float* __restrict__ wb, int Mb,
    const float* __restrict__ wh, int Mh)
{
    for (int i = 0; i < Ms; i++) {
        const float* o = ws + i * 4;
        if (sphere_sphere_dist(px, py, pz, r, o[0], o[1], o[2], o[3]) < 0.0f) return true;
    }
    for (int i = 0; i < Mc; i++) {
        const float* o = wc + i * 7;
        if (sphere_capsule_dist(px, py, pz, r,
                o[0], o[1], o[2], o[3], o[4], o[5], o[6]) < 0.0f) return true;
    }
    for (int i = 0; i < Mb; i++) {
        const float* o = wb + i * 15;
        if (sphere_box_dist(px, py, pz, r,
                o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7], o[8],
                o[9], o[10], o[11], o[12], o[13], o[14]) < 0.0f) return true;
    }
    for (int i = 0; i < Mh; i++) {
        const float* o = wh + i * 6;
        if (sphere_halfspace_dist(px, py, pz, r,
                o[0], o[1], o[2], o[3], o[4], o[5]) < 0.0f) return true;
    }
    return false;
}

// ── Fused FK + binary collision kernel ────────────────────────────────────────
//
// Grid:  B blocks (one per configuration).
// Block: BCC_THREADS threads cooperating over spheres / pairs.

__global__ void binary_collision_kernel(
    const float* __restrict__ cfg,               // [B, n_act]
    const float* __restrict__ twists,            // [J, 6]
    const float* __restrict__ parent_tf,         // [J, 7]
    const int*   __restrict__ parent_idx,        // [J]
    const int*   __restrict__ act_idx,           // [J]
    const float* __restrict__ mimic_mul,         // [J]
    const float* __restrict__ mimic_off,         // [J]
    const int*   __restrict__ mimic_act_idx,     // [J]
    const int*   __restrict__ topo_inv,          // [J]
    const int*   __restrict__ link_parent_joint, // [NL]
    const float* __restrict__ f_local,           // [Kf, 4]  fine spheres (local)
    const float* __restrict__ c_local,           // [Kc, 4]  coarse spheres (local)
    const float* __restrict__ ws,                // [Ms, 4]
    const float* __restrict__ wc,                // [Mc, 7]
    const float* __restrict__ wb,                // [Mb, 15]
    const float* __restrict__ wh,                // [Mh, 6]
    const int*   __restrict__ f_pi,              // [Pf]
    const int*   __restrict__ f_pj,              // [Pf]
    const int*   __restrict__ c_pi,              // [Pc]
    const int*   __restrict__ c_pj,              // [Pc]
    int*         __restrict__ out_free,          // [B]  1 = free, 0 = collision
    int B, int n_act, int J, int NL,
    int Kf, int Kc, int Pf, int Pc,
    int Ms, int Mc, int Mb, int Mh)
{
    const int b = blockIdx.x;
    if (b >= B) return;
    const int tid = threadIdx.x;
    const int nt  = blockDim.x;

    __shared__ float Tw[BCC_MAX_JOINTS * 7];  // world transforms, joint-indexed
    __shared__ float Tl[BCC_MAX_LINKS * 7];   // world transforms, link-indexed
    __shared__ int   link_hit[BCC_MAX_LINKS]; // coarse "needs fine check" flags
    __shared__ volatile int cc;               // 1 once a fine collision is found
    __shared__ int   approx_env;              // any coarse env collision?
    __shared__ int   approx_self;             // any coarse self collision?

    // ── FK (single thread; sequential topological walk) ──────────────────────
    if (tid == 0) {
        fk_single(cfg + (long long)b * n_act,
                  twists, parent_tf, parent_idx, act_idx,
                  mimic_mul, mimic_off, mimic_act_idx, topo_inv,
                  Tw, J, n_act);
        cc = 0; approx_env = 0; approx_self = 0;
    }
    __syncthreads();

    // ── Link poses: T_link[l] = T_joint[parent] (or identity for base) ────────
    for (int l = tid; l < NL; l += nt) {
        const int pj = link_parent_joint[l];
        float* dst = Tl + l * 7;
        if (pj < 0) {
            dst[0] = 1.0f; dst[1] = 0.0f; dst[2] = 0.0f; dst[3] = 0.0f;
            dst[4] = 0.0f; dst[5] = 0.0f; dst[6] = 0.0f;
        } else {
            #pragma unroll
            for (int k = 0; k < 7; k++) dst[k] = Tw[pj * 7 + k];
        }
        link_hit[l] = 0;
    }
    __syncthreads();

    const int  Sf = (NL > 0) ? Kf / NL : 0;
    const int  Sc = (NL > 0) ? Kc / NL : 0;
    const bool have_coarse = (Kc > 0);

    // ── ENV stage 1: coarse spheres → flag links / detect possible collision ──
    if (have_coarse) {
        for (int k = tid; k < Kc; k += nt) {
            const int n = k % NL;
            const float* lp = c_local + k * 4;
            const float r = lp[3];
            if (r < 0.0f) continue;
            float p[3] = {lp[0], lp[1], lp[2]}, w[3];
            apply_se3_point(Tl + n * 7, p, w);
            if (sphere_world_hit(w[0], w[1], w[2], r, ws, Ms, wc, Mc, wb, Mb, wh, Mh)) {
                link_hit[n] = 1;
                approx_env = 1;
            }
        }
        __syncthreads();
    } else {
        // No coarse model: every link must be checked with the fine geometry.
        for (int l = tid; l < NL; l += nt) link_hit[l] = 1;
        if (tid == 0) approx_env = 1;
        __syncthreads();
    }

    // ── ENV stage 2: fine spheres on flagged links, with early exit ───────────
    if (approx_env) {
        for (int k = tid; k < Kf; k += nt) {
            if (cc) break;
            const int n = k % NL;
            if (link_hit[n] == 0) continue;
            const float* lp = f_local + k * 4;
            const float r = lp[3];
            if (r < 0.0f) continue;
            float p[3] = {lp[0], lp[1], lp[2]}, w[3];
            apply_se3_point(Tl + n * 7, p, w);
            if (sphere_world_hit(w[0], w[1], w[2], r, ws, Ms, wc, Mc, wb, Mb, wh, Mh))
                cc = 1;
        }
        __syncthreads();
    }

    if (cc) {
        if (tid == 0) out_free[b] = 0;
        return;
    }

    // ── SELF stage 1: coarse self-pairs → flag links ──────────────────────────
    bool run_self_fine;
    if (Pc > 0) {
        for (int p = tid; p < Pc; p += nt) {
            const int li = c_pi[p], lj = c_pj[p];
            bool hit = false;
            for (int si = 0; si < Sc && !hit; si++) {
                const float* lpi = c_local + (si * NL + li) * 4;
                if (lpi[3] < 0.0f) continue;
                float pi[3] = {lpi[0], lpi[1], lpi[2]}, wi[3];
                apply_se3_point(Tl + li * 7, pi, wi);
                for (int sj = 0; sj < Sc; sj++) {
                    const float* lpj = c_local + (sj * NL + lj) * 4;
                    if (lpj[3] < 0.0f) continue;
                    float pj[3] = {lpj[0], lpj[1], lpj[2]}, wj[3];
                    apply_se3_point(Tl + lj * 7, pj, wj);
                    if (sphere_sphere_dist(wi[0], wi[1], wi[2], lpi[3],
                                           wj[0], wj[1], wj[2], lpj[3]) < 0.0f) {
                        hit = true;
                        break;
                    }
                }
            }
            if (hit) { link_hit[li] = 1; link_hit[lj] = 1; approx_self = 1; }
        }
        __syncthreads();
        run_self_fine = (approx_self != 0);
    } else {
        // No coarse self-pairs → no guard available; run the fine self pass.
        run_self_fine = (Pf > 0);
    }

    // ── SELF stage 2: fine self-pairs on flagged links, with early exit ───────
    if (run_self_fine) {
        for (int p = tid; p < Pf; p += nt) {
            if (cc) break;
            const int li = f_pi[p], lj = f_pj[p];
            if (Pc > 0 && link_hit[li] == 0 && link_hit[lj] == 0) continue;
            bool hit = false;
            for (int si = 0; si < Sf && !hit; si++) {
                const float* lpi = f_local + (si * NL + li) * 4;
                if (lpi[3] < 0.0f) continue;
                float pi[3] = {lpi[0], lpi[1], lpi[2]}, wi[3];
                apply_se3_point(Tl + li * 7, pi, wi);
                for (int sj = 0; sj < Sf; sj++) {
                    const float* lpj = f_local + (sj * NL + lj) * 4;
                    if (lpj[3] < 0.0f) continue;
                    float pj[3] = {lpj[0], lpj[1], lpj[2]}, wj[3];
                    apply_se3_point(Tl + lj * 7, pj, wj);
                    if (sphere_sphere_dist(wi[0], wi[1], wi[2], lpi[3],
                                           wj[0], wj[1], wj[2], lpj[3]) < 0.0f) {
                        hit = true;
                        break;
                    }
                }
            }
            if (hit) cc = 1;
        }
        __syncthreads();
    }

    if (tid == 0) out_free[b] = cc ? 0 : 1;
}

// ── XLA FFI handler ───────────────────────────────────────────────────────────

static ffi::Error CollisionBinaryImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F32> cfg,               // [B, n_act]
    ffi::Buffer<ffi::DataType::F32> twists,            // [J, 6]
    ffi::Buffer<ffi::DataType::F32> parent_tf,         // [J, 7]
    ffi::Buffer<ffi::DataType::S32> parent_idx,        // [J]
    ffi::Buffer<ffi::DataType::S32> act_idx,           // [J]
    ffi::Buffer<ffi::DataType::F32> mimic_mul,         // [J]
    ffi::Buffer<ffi::DataType::F32> mimic_off,         // [J]
    ffi::Buffer<ffi::DataType::S32> mimic_act_idx,     // [J]
    ffi::Buffer<ffi::DataType::S32> topo_inv,          // [J]
    ffi::Buffer<ffi::DataType::S32> link_parent_joint, // [NL]
    ffi::Buffer<ffi::DataType::F32> f_local,           // [Kf, 4]
    ffi::Buffer<ffi::DataType::F32> c_local,           // [Kc, 4]
    ffi::Buffer<ffi::DataType::F32> world_spheres,     // [Ms, 4]
    ffi::Buffer<ffi::DataType::F32> world_capsules,    // [Mc, 7]
    ffi::Buffer<ffi::DataType::F32> world_boxes,       // [Mb, 15]
    ffi::Buffer<ffi::DataType::F32> world_halfspaces,  // [Mh, 6]
    ffi::Buffer<ffi::DataType::S32> f_pair_i,          // [Pf]
    ffi::Buffer<ffi::DataType::S32> f_pair_j,          // [Pf]
    ffi::Buffer<ffi::DataType::S32> c_pair_i,          // [Pc]
    ffi::Buffer<ffi::DataType::S32> c_pair_j,          // [Pc]
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> out)  // [B]
{
    const int B     = static_cast<int>(cfg.dimensions()[0]);
    const int n_act = static_cast<int>(cfg.dimensions()[1]);
    const int J     = static_cast<int>(twists.dimensions()[0]);
    const int NL    = static_cast<int>(link_parent_joint.dimensions()[0]);
    const int Kf    = static_cast<int>(f_local.dimensions()[0]);
    const int Kc    = static_cast<int>(c_local.dimensions()[0]);
    const int Pf    = static_cast<int>(f_pair_i.dimensions()[0]);
    const int Pc    = static_cast<int>(c_pair_i.dimensions()[0]);
    const int Ms    = static_cast<int>(world_spheres.dimensions()[0]);
    const int Mc    = static_cast<int>(world_capsules.dimensions()[0]);
    const int Mb    = static_cast<int>(world_boxes.dimensions()[0]);
    const int Mh    = static_cast<int>(world_halfspaces.dimensions()[0]);

    if (B <= 0) return ffi::Error::Success();
    if (J > BCC_MAX_JOINTS || NL > BCC_MAX_LINKS) {
        return ffi::Error(
            ffi::ErrorCode::kInvalidArgument,
            "binary collision kernel: J/NL exceed BCC_MAX_JOINTS/BCC_MAX_LINKS; "
            "increase the bounds and rebuild.");
    }

    binary_collision_kernel<<<B, BCC_THREADS, 0, stream>>>(
        cfg.typed_data(), twists.typed_data(), parent_tf.typed_data(),
        parent_idx.typed_data(), act_idx.typed_data(),
        mimic_mul.typed_data(), mimic_off.typed_data(), mimic_act_idx.typed_data(),
        topo_inv.typed_data(), link_parent_joint.typed_data(),
        f_local.typed_data(), c_local.typed_data(),
        world_spheres.typed_data(), world_capsules.typed_data(),
        world_boxes.typed_data(), world_halfspaces.typed_data(),
        f_pair_i.typed_data(), f_pair_j.typed_data(),
        c_pair_i.typed_data(), c_pair_j.typed_data(),
        out->typed_data(),
        B, n_act, J, NL, Kf, Kc, Pf, Pc, Ms, Mc, Mb, Mh);

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CollisionBinaryFfi, CollisionBinaryImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // cfg
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // twists
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // parent_tf
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // parent_idx
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // act_idx
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // mimic_mul
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // mimic_off
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // mimic_act_idx
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // topo_inv
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // link_parent_joint
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // f_local
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // c_local
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_spheres
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_capsules
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_boxes
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_halfspaces
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // f_pair_i
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // f_pair_j
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // c_pair_i
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // c_pair_j
        .Ret<ffi::Buffer<ffi::DataType::S32>>()); // out [B]
