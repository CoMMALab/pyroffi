/**
 * Levenberg-Marquardt / augmented Lagrangian over a compiled least-squares problem, at one
 * GLASS tier.
 *
 * `_cost_gen.cuh` is written per problem structure (and per tier variant) by
 * pyroffi.costs._compile. It provides tier-templated device functions
 *
 *   residual(x, p, r, scratch, rank, size)
 *   residual_jacobian(x, p, r, J, scratch, rank, size)     J: (n_r, n_x) row-major
 *
 * written in GLASS's form: straight-line code every lane of a group runs redundantly, plus
 * loops `for (i = rank; i < n; i += size)` that the group splits, each followed by
 * PYROFFI_COST_SYNC(). Each residual row has a kind:
 *
 *   0  cost              minimise r^2
 *   1  equality          r == 0      (augmented Lagrangian)
 *   2  inequality        r <= 0      (augmented Lagrangian; >= 0 rows arrive negated)
 *
 * With multipliers `lam` and penalty `rho` the merit is
 *
 *   f(x) = sum_cost r^2 + sum_con (rho / 2) t^2,   t = r + lam / rho   (t = max(t, 0) for <=)
 *
 * and its Gauss-Newton model is H = sum w J J^T, g = sum w (r + s) J, with w = 1, s = 0 on
 * cost rows and w = rho / 2 (0 on an inactive inequality), s = lam / rho on constraint rows.
 *
 * Tiers (PYROFFI_COST_TIER, one per build): one problem per thread (Thread), per warp
 * (Warp), or per block (Block). Per-problem state lives in registers/local memory at the
 * thread tier and in shared memory at the cooperative tiers. Scalar control flow (LM
 * damping, acceptance, AL updates) is computed redundantly and identically by every lane,
 * so every branch is group-uniform.
 */

#include "_glass_solve.cuh"
#include "xla/ffi/api/ffi.h"

#include <cmath>

// One build per tier: the generated code is plain functions (templates of this size are
// pathologically slow in nvcc's front end), and syncs through this macro.
#if PYROFFI_COST_TIER == 0
#define PYROFFI_COST_SYNC() ((void)0)
#elif PYROFFI_COST_TIER == 1
#define PYROFFI_COST_SYNC() __syncwarp()
#else
#define PYROFFI_COST_SYNC() __syncthreads()
#endif

#include "_cost_gen.cuh"

namespace ffi = xla::ffi;
namespace G = pyroffi::costs_gen;
using pyroffi::Tier;

constexpr Tier TIER = static_cast<Tier>(PYROFFI_COST_TIER);
constexpr int NX = G::n_x;
constexpr int NR = G::n_r;
constexpr int NS = G::scratch_size > 0 ? G::scratch_size : 1;

// Per-problem state: x, xt, dx, g (NX each), H (NX*NX), r, rt, lam, w, s (NR each),
// J (NR*NX). The generated code's scratch (NS per problem) lives in a global workspace: it
// can be far larger than shared memory, and one problem's slice stays in L1/L2.
constexpr int SLOT_FLOATS = 4 * NX + NX * NX + 5 * NR + NR * NX;
constexpr int SHARED_BUDGET = 48 * 1024;
constexpr int slots_that_fit(int want)
{
    while (want > 1 && want * (SLOT_FLOATS * 4 + 16) > SHARED_BUDGET) want /= 2;
    return want;
}
constexpr int WARPS_PER_BLOCK = slots_that_fit(8);
constexpr int BLOCK_TPB = 128;
constexpr int THREAD_TPB = 64;
constexpr int SLOTS = TIER == Tier::Warp ? WARPS_PER_BLOCK : 1;
constexpr int TPB = TIER == Tier::Thread ? THREAD_TPB
                  : TIER == Tier::Warp   ? 32 * WARPS_PER_BLOCK : BLOCK_TPB;

static_assert(TIER == Tier::Block || NX <= pyroffi::TIER_CHOICE_MAX_N,
              "pyroffi.costs: the thread and warp tiers support up to 32 variables; "
              "use the block tier");
#ifdef PYROFFI_COST_JVP
static_assert(TIER == Tier::Thread || NX <= TPB, "one Jacobian column per lane");
#endif
static_assert(TIER == Tier::Thread || SLOT_FLOATS * 4 + 16 <= SHARED_BUDGET,
              "pyroffi.costs: per-problem state exceeds the shared-memory budget at this tier");

struct CostLmSettings {
    int max_iters;
    int al_iters;
    float lambda_initial;
    float tolerance;
    float al_tolerance;
};

static __device__ __forceinline__ float merit(const float* r, const float* lam, float rho)
{
    float f = 0.0f;
    #pragma unroll
    for (int i = 0; i < NR; ++i) {
        const int k = G::row_kind(i);
        if (k == 0) {
            f += r[i] * r[i];
        } else {
            float t = r[i] + lam[i] / rho;
            if (k == 2) t = fmaxf(t, 0.0f);
            f += 0.5f * rho * t * t;
        }
    }
    return f;
}

__global__ void __launch_bounds__(TPB)
cost_lm_kernel(const float* __restrict__ x0, const float* __restrict__ params, int param_stride,
               float* __restrict__ x_out, float* __restrict__ cost_out,
               float* __restrict__ viol_out, float* __restrict__ workspace, int n_problems,
               CostLmSettings st)
{
    // Compile-time constants at the thread tier, so every per-row loop below unrolls and the
    // solver state stays in registers.
    const int rank = TIER == Tier::Thread ? 0 : TIER == Tier::Warp ? (int)(threadIdx.x & 31u)
                                                                    : (int)threadIdx.x;
    const int size = TIER == Tier::Thread ? 1 : TIER == Tier::Warp ? 32 : (int)blockDim.x;
    const int slot = TIER == Tier::Warp ? (int)(threadIdx.x >> 5) : 0;
    const int b = TIER == Tier::Thread ? (int)(blockIdx.x * blockDim.x + threadIdx.x)
                : TIER == Tier::Warp   ? (int)(blockIdx.x * WARPS_PER_BLOCK + slot)
                                       : (int)blockIdx.x;
    if (b >= n_problems) return;  // group-uniform: a whole thread/warp/block retires together

#if PYROFFI_COST_TIER == 0
    // Separate small arrays, indexed by constants after unrolling: registers, not local memory.
    float x[NX], xt[NX], dx[NX], g[NX], H[NX * NX], r[NR], rt[NR], lam[NR], w[NR], s[NR];
#ifdef PYROFFI_COST_NORMAL_EQUATIONS
    float* J = nullptr;
#else
    float J[NR * NX];
#endif
    int fail_slot = 0;
    int* s_fail = &fail_slot;
#else
    __shared__ float smem[SLOTS][SLOT_FLOATS];
    __shared__ int sfail[SLOTS];
    int* s_fail = &sfail[slot];
    float* x = smem[slot];
    float* xt = x + NX;
    float* dx = xt + NX;
    float* g = dx + NX;
    float* H = g + NX;
    float* r = H + NX * NX;
    float* rt = r + NR;
    float* lam = rt + NR;
    float* w = lam + NR;
    float* s = w + NR;
    float* J = s + NR;
#endif
    float* scratch = workspace + (size_t)b * NS;
    const float* p = params + (size_t)b * param_stride;
    const auto sync = [] { PYROFFI_COST_SYNC(); };

    #pragma unroll

    for (int j = rank; j < NX; j += size) x[j] = x0[(size_t)b * NX + j];
    #pragma unroll
    for (int i = rank; i < NR; i += size) lam[i] = 0.0f;
    sync();
    G::residual(x, p, r, scratch, rank, size);
    sync();

    // ALGENCAN-style initial penalty, as jaxls: rho = 10 max(1, |f|) / max(1, c^2 / 2).
    float rho = 1.0f;
    if constexpr (G::has_constraints) {
        float f0 = 0.0f, c2 = 0.0f;
        #pragma unroll
        for (int i = 0; i < NR; ++i) {
            const int k = G::row_kind(i);
            if (k == 0) f0 += r[i] * r[i];
            else { const float v = (k == 2) ? fmaxf(r[i], 0.0f) : r[i]; c2 += v * v; }
        }
        rho = 10.0f * fmaxf(1.0f, fabsf(f0)) / fmaxf(1.0f, 0.5f * c2);
    }

    // One LM loop with one iteration budget, as jaxls: the multipliers are updated whenever
    // the inner solve has converged, not by restarting a full solve per AL round (which, on a
    // constraint that cannot be met, would spend al_iters x max_iters iterations).
    float prev_viol = INFINITY;
    int al_updates = 0;
    float mu = st.lambda_initial;
    float f = merit(r, lam, rho);
    {
        for (int it = 0; it < st.max_iters; ++it) {
            #pragma unroll
            for (int i = rank; i < NR; i += size) {
                const int k = G::row_kind(i);
                if (k == 0) { w[i] = 1.0f; s[i] = 0.0f; continue; }
                s[i] = lam[i] / rho;
                w[i] = ((k == 1) || (r[i] + s[i] > 0.0f)) ? 0.5f * rho : 0.0f;
            }
#ifdef PYROFFI_COST_NORMAL_EQUATIONS
            // Scalar variant: H and g emitted symbolically (Jacobian sparsity folded away).
            G::normal_equations(x, p, w, s, r, H, g, scratch, rank, size);
            #pragma unroll
            for (int a = rank; a < NX; a += size) {
                H[a * NX + a] += mu;
                dx[a] = -g[a];
            }
#elif defined(PYROFFI_COST_JVP)
            // Column-parallel forward mode: lane j computes J[:, j] as the directional
            // derivative along e_j, running the same straight-line code as every other lane.
            if (rank < NX) {
                float t[NX], rr[NR], jt[NR];
#pragma unroll
                for (int k = 0; k < NX; ++k) t[k] = (k == rank) ? 1.0f : 0.0f;
                G::residual_jvp(x, p, t, rr, jt);
                for (int i = 0; i < NR; ++i) J[i * NX + rank] = jt[i];
            }
            sync();
            for (int e = rank; e < NX * NX; e += size) {
                const int a = e / NX, c = e % NX;
                float acc = 0.0f;
                for (int i = 0; i < NR; ++i) acc += w[i] * J[i * NX + a] * J[i * NX + c];
                H[e] = acc + (a == c ? mu : 0.0f);
            }
            for (int a = rank; a < NX; a += size) {
                float acc = 0.0f;
                for (int i = 0; i < NR; ++i) acc += w[i] * (r[i] + s[i]) * J[i * NX + a];
                dx[a] = -acc;
            }
#else
            G::residual_jacobian(x, p, r, J, scratch, rank, size);  // syncs (w, s too)
            #pragma unroll
            for (int e = rank; e < NX * NX; e += size) {
                const int a = e / NX, c = e % NX;
                float acc = 0.0f;
                #pragma unroll
                for (int i = 0; i < NR; ++i) acc += w[i] * J[i * NX + a] * J[i * NX + c];
                H[e] = acc + (a == c ? mu : 0.0f);
            }
            #pragma unroll
            for (int a = rank; a < NX; a += size) {
                float acc = 0.0f;
                #pragma unroll
                for (int i = 0; i < NR; ++i) acc += w[i] * (r[i] + s[i]) * J[i * NX + a];
                dx[a] = -acc;
            }
#endif
            sync();
            const bool ok = pyroffi::tier_posv<TIER, float, NX>(H, dx, s_fail);
            sync();
            float step2 = 0.0f, x2 = 0.0f;
            #pragma unroll
            for (int j = 0; j < NX; ++j) {
                step2 += dx[j] * dx[j];
                x2 += x[j] * x[j];
            }
            #pragma unroll
            for (int j = rank; j < NX; j += size) xt[j] = x[j] + dx[j];
            sync();
            G::residual(xt, p, rt, scratch, rank, size);
            sync();
            const float ft = merit(rt, lam, rho);
            bool inner_done;
            if (ok && ft < f) {
                #pragma unroll
                for (int j = rank; j < NX; j += size) x[j] = xt[j];
                #pragma unroll
                for (int i = rank; i < NR; i += size) r[i] = rt[i];
                sync();
                const float rel = (f - ft) / fmaxf(f, 1e-30f);
                f = ft;
                mu = fmaxf(mu * 0.25f, 1e-10f);
                inner_done = rel < st.tolerance || step2 < st.tolerance * st.tolerance * (x2 + 1e-12f);
            } else {
                mu *= 4.0f;
                inner_done = mu > 1e10f;
            }
            if (!inner_done) continue;
            if constexpr (!G::has_constraints) break;
            if (al_updates++ >= st.al_iters) break;
            float viol = 0.0f;
            #pragma unroll
            for (int i = 0; i < NR; ++i) {
                const int k = G::row_kind(i);
                if (k == 1) viol = fmaxf(viol, fabsf(r[i]));
                if (k == 2) viol = fmaxf(viol, r[i]);
            }
            #pragma unroll
            for (int i = rank; i < NR; i += size) {
                const int k = G::row_kind(i);
                if (k == 1) lam[i] = fminf(fmaxf(lam[i] + rho * r[i], -1e7f), 1e7f);
                if (k == 2) lam[i] = fminf(fmaxf(lam[i] + rho * r[i], 0.0f), 1e7f);
            }
            sync();
            if (viol < st.al_tolerance) break;
            if (viol > 0.5f * prev_viol) rho = fminf(rho * 4.0f, 1e7f);
            prev_viol = viol;
            mu = st.lambda_initial;
            f = merit(r, lam, rho);  // the merit changed with lam / rho
        }
    }

    #pragma unroll

    for (int j = rank; j < NX; j += size) x_out[(size_t)b * NX + j] = x[j];
    if (rank == 0) {
        float cost = 0.0f, viol = 0.0f;
        #pragma unroll
        for (int i = 0; i < NR; ++i) {
            const int k = G::row_kind(i);
            if (k == 0) cost += r[i] * r[i];
            if (k == 1) viol = fmaxf(viol, fabsf(r[i]));
            if (k == 2) viol = fmaxf(viol, r[i]);
        }
        cost_out[b] = cost;
        viol_out[b] = viol;
    }
}

static ffi::Error CostLmImpl(cudaStream_t stream, ffi::Buffer<ffi::DataType::F32> x0,
                             ffi::Buffer<ffi::DataType::F32> params,
                             ffi::ResultBuffer<ffi::DataType::F32> x_out,
                             ffi::ResultBuffer<ffi::DataType::F32> cost_out,
                             ffi::ResultBuffer<ffi::DataType::F32> viol_out,
                             ffi::ResultBuffer<ffi::DataType::F32> workspace,
                             int32_t max_iters, int32_t al_iters, float lambda_initial,
                             float tolerance, float al_tolerance)
{
    const int n_problems = static_cast<int>(x0.dimensions()[0]);
    const int param_stride = params.dimensions()[0] == 1 ? 0 : static_cast<int>(params.dimensions()[1]);
    const CostLmSettings st{max_iters, al_iters, lambda_initial, tolerance, al_tolerance};
    const int per_block = TIER == Tier::Thread ? THREAD_TPB : (TIER == Tier::Warp ? WARPS_PER_BLOCK : 1);
    const int blocks = (n_problems + per_block - 1) / per_block;
    cost_lm_kernel<<<blocks, TPB, 0, stream>>>(
        x0.typed_data(), params.typed_data(), param_stride, x_out->typed_data(),
        cost_out->typed_data(), viol_out->typed_data(), workspace->typed_data(), n_problems, st);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CostLmFfi, CostLmImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // x0      (B, n_x)
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // params  (B | 1, max(n_p, 1))
        .Ret<ffi::Buffer<ffi::DataType::F32>>()   // x       (B, n_x)
        .Ret<ffi::Buffer<ffi::DataType::F32>>()   // cost    (B,)  sum of squared cost rows
        .Ret<ffi::Buffer<ffi::DataType::F32>>()   // viol    (B,)  max constraint violation
        .Ret<ffi::Buffer<ffi::DataType::F32>>()   // workspace (B, max(scratch, 1)), scratch only
        .Attr<int32_t>("max_iters")
        .Attr<int32_t>("al_iters")
        .Attr<float>("lambda_initial")
        .Attr<float>("tolerance")
        .Attr<float>("al_tolerance"));
