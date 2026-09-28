/**
 * Levenberg-Marquardt over a compiled jaxls-style least-squares problem.
 *
 * `_cost_gen.cuh` is written per problem structure by pyroffi.costs._compile. It holds the
 * problem's residual and its weighted Gauss-Newton normal equations as straight-line code
 * (robot kinematics, costs and Jacobians folded together), plus each residual row's kind:
 *
 *   0  cost              minimise r^2
 *   1  equality          r == 0      (augmented Lagrangian)
 *   2  inequality        r <= 0      (augmented Lagrangian; >= 0 rows arrive negated)
 *
 * With multipliers `lam` and penalty `rho` the merit is
 *
 *   f(x) = sum_cost r^2 + sum_con (rho / 2) t^2,   t = r + lam / rho   (t = max(t, 0) for <=)
 *
 * and its Gauss-Newton model has H = sum w J J^T, g = sum w (r + s) J, with w = 1, s = 0 on
 * cost rows and w = rho / 2 (0 on an inactive inequality), s = lam / rho on constraint rows.
 * normal_equations() is handed w and s, so the generated code never sees rho or lam.
 *
 * One thread solves one problem (a row of x0/params): the thread tier, which wins at the
 * large batches this is meant for (see _glass_solve.cuh). Everything happens in this one
 * launch; nothing returns to JAX until the solve is done.
 */

#include "_cost_gen.cuh"
#include "_glass_solve.cuh"
#include "xla/ffi/api/ffi.h"

#include <cmath>

namespace ffi = xla::ffi;
namespace G = pyroffi::costs_gen;

static_assert(G::n_x <= pyroffi::TIER_CHOICE_MAX_N,
              "pyroffi.costs: the thread-tier dense solve supports up to 32 variables");

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
    for (int i = 0; i < G::n_r; ++i) {
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

__global__ void cost_lm_kernel(const float* __restrict__ x0, const float* __restrict__ params,
                               int param_stride, float* __restrict__ x_out,
                               float* __restrict__ cost_out, float* __restrict__ viol_out,
                               int n_problems, CostLmSettings st)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_problems) return;
    const float* p = params + (size_t)b * param_stride;

    float x[G::n_x], xt[G::n_x], g[G::n_x], dx[G::n_x], H[G::n_x * G::n_x];
    float r[G::n_r], rt[G::n_r], lam[G::n_r], w[G::n_r], s[G::n_r];
#pragma unroll
    for (int j = 0; j < G::n_x; ++j) x[j] = x0[(size_t)b * G::n_x + j];
#pragma unroll
    for (int i = 0; i < G::n_r; ++i) lam[i] = 0.0f;

    G::residual(x, p, r);

    // ALGENCAN-style initial penalty, as jaxls: rho = 10 max(1, |f|) / max(1, c^2 / 2).
    float rho = 1.0f;
    if constexpr (G::has_constraints) {
        float f0 = 0.0f, c2 = 0.0f;
#pragma unroll
        for (int i = 0; i < G::n_r; ++i) {
            const int k = G::row_kind(i);
            if (k == 0) f0 += r[i] * r[i];
            else { const float v = (k == 2) ? fmaxf(r[i], 0.0f) : r[i]; c2 += v * v; }
        }
        rho = 10.0f * fmaxf(1.0f, fabsf(f0)) / fmaxf(1.0f, 0.5f * c2);
    }

    float prev_viol = INFINITY;
    int fail = 0;
    const int n_outer = G::has_constraints ? st.al_iters : 1;
    for (int outer = 0; outer < n_outer; ++outer) {
        float mu = st.lambda_initial;
        float f = merit(r, lam, rho);
        for (int it = 0; it < st.max_iters; ++it) {
#pragma unroll
            for (int i = 0; i < G::n_r; ++i) {
                const int k = G::row_kind(i);
                if (k == 0) { w[i] = 1.0f; s[i] = 0.0f; continue; }
                s[i] = lam[i] / rho;
                const bool active = (k == 1) || (r[i] + s[i] > 0.0f);
                w[i] = active ? 0.5f * rho : 0.0f;
            }
            G::normal_equations(x, p, w, s, r, H, g);
#pragma unroll
            for (int j = 0; j < G::n_x; ++j) {
                H[j * G::n_x + j] += mu;
                dx[j] = -g[j];
            }
            const bool ok = pyroffi::tier_posv<pyroffi::Tier::Thread, float, G::n_x>(H, dx, &fail);
            float step2 = 0.0f, x2 = 0.0f;
#pragma unroll
            for (int j = 0; j < G::n_x; ++j) {
                xt[j] = x[j] + dx[j];
                step2 += dx[j] * dx[j];
                x2 += x[j] * x[j];
            }
            G::residual(xt, p, rt);
            const float ft = merit(rt, lam, rho);
            if (ok && ft < f) {
#pragma unroll
                for (int j = 0; j < G::n_x; ++j) x[j] = xt[j];
#pragma unroll
                for (int i = 0; i < G::n_r; ++i) r[i] = rt[i];
                const float rel = (f - ft) / fmaxf(f, 1e-30f);
                f = ft;
                mu = fmaxf(mu * 0.25f, 1e-10f);
                if (rel < st.tolerance || step2 < st.tolerance * st.tolerance * (x2 + 1e-12f)) break;
            } else {
                mu *= 4.0f;
                if (mu > 1e10f) break;
            }
        }
        if constexpr (G::has_constraints) {
            float viol = 0.0f;
#pragma unroll
            for (int i = 0; i < G::n_r; ++i) {
                const int k = G::row_kind(i);
                if (k == 1) { lam[i] = fminf(fmaxf(lam[i] + rho * r[i], -1e7f), 1e7f); viol = fmaxf(viol, fabsf(r[i])); }
                if (k == 2) { lam[i] = fminf(fmaxf(lam[i] + rho * r[i], 0.0f), 1e7f); viol = fmaxf(viol, r[i]); }
            }
            if (viol < st.al_tolerance) break;
            if (viol > 0.5f * prev_viol) rho = fminf(rho * 4.0f, 1e7f);
            prev_viol = viol;
        }
    }

    float cost = 0.0f, viol = 0.0f;
#pragma unroll
    for (int i = 0; i < G::n_r; ++i) {
        const int k = G::row_kind(i);
        if (k == 0) cost += r[i] * r[i];
        if (k == 1) viol = fmaxf(viol, fabsf(r[i]));
        if (k == 2) viol = fmaxf(viol, r[i]);
    }
#pragma unroll
    for (int j = 0; j < G::n_x; ++j) x_out[(size_t)b * G::n_x + j] = x[j];
    cost_out[b] = cost;
    viol_out[b] = viol;
}

static ffi::Error CostLmImpl(cudaStream_t stream, ffi::Buffer<ffi::DataType::F32> x0,
                             ffi::Buffer<ffi::DataType::F32> params,
                             ffi::ResultBuffer<ffi::DataType::F32> x_out,
                             ffi::ResultBuffer<ffi::DataType::F32> cost_out,
                             ffi::ResultBuffer<ffi::DataType::F32> viol_out,
                             int32_t max_iters, int32_t al_iters, float lambda_initial,
                             float tolerance, float al_tolerance)
{
    const int n_problems = static_cast<int>(x0.dimensions()[0]);
    const int param_stride = params.dimensions()[0] == 1 ? 0 : static_cast<int>(params.dimensions()[1]);
    const CostLmSettings st{max_iters, al_iters, lambda_initial, tolerance, al_tolerance};
    constexpr int tpb = 64;
    cost_lm_kernel<<<(n_problems + tpb - 1) / tpb, tpb, 0, stream>>>(
        x0.typed_data(), params.typed_data(), param_stride, x_out->typed_data(),
        cost_out->typed_data(), viol_out->typed_data(), n_problems, st);
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
        .Attr<int32_t>("max_iters")
        .Attr<int32_t>("al_iters")
        .Attr<float>("lambda_initial")
        .Attr<float>("tolerance")
        .Attr<float>("al_tolerance"));
