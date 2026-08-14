/*
 * Canonicalisation: slide an IK solution along the self-motion manifold to the
 * point nearest a reference configuration.
 *
 *     q* = argmin 1/2 ||q - q_ref||^2   subject to   r(q, t) = 0
 *
 * This is what makes IK derivatives correct on a redundant arm. r = 0 alone
 * does not determine q* (a 7-DOF arm with 6 task constraints leaves a 1-D
 * solution curve), so differentiating it answers a question the solver did not
 * ask -- pinv silently supplies the missing information by picking the
 * minimum-norm tangent. Pinning q* to a definite point of that curve makes the
 * KKT sensitivity exact. See optimization_engines/_canonical_ik.py.
 *
 * Why this is a kernel and not JAX. The JAX loop cost ~4 ms PER ITERATION of
 * XLA dispatch overhead -- 1615 ms at B=64, 13.7 s at B=1024, i.e. 61-149x the
 * IK solve it was correcting. The arithmetic is trivial: each iteration is one
 * FK + task Jacobian + a small SPD solve, which is exactly what the IK solve
 * kernels already do many times over in 26 ms. The overhead was the whole cost.
 *
 * The iteration is Gauss-Newton on the constrained problem:
 *
 *     dq = -J^+ r  +  step * (I - J^+ J) (q_ref - q)
 *
 * The first term restores r = 0; the second walks toward q_ref along the null
 * space WITHOUT moving the end-effector. `step` must be damped: at step = 1
 * (the exact Gauss-Newton step) the iteration diverges on manifold curvature --
 * measured ||q - q_ref|| growing 6.0 -> 34.9 -- because a full step leaves the
 * linearisation's region of validity. Larger trust-region variants were tried
 * and also failed to converge; the small damped step is required, which is
 * precisely why the iteration count is high and why it belongs in CUDA.
 *
 * PRECISION. This runs in float32, because the shared FK/Jacobian helper is
 * float32. That converges to |r| ~ 1e-5, not the ~1e-16 the derivative rule
 * wants, so the Python side finishes with a few float64 Newton steps. Bulk here,
 * polish there.
 *
 * J is the GEOMETRIC Jacobian (the helper's orientation rows are the angular
 * Jacobian, not d(log R)/dq). That is harmless here: the two coincide wherever
 * r = 0, so the FIXED POINT is unchanged, and only the path taken to reach it
 * differs slightly.
 */

#include "_ik_cuda_helpers.cuh"

#include <xla/ffi/api/ffi.h>
#include <cuda_runtime.h>

namespace ffi = xla::ffi;

#define CANON_MAX_RES (6 * MAX_EE)

// Cholesky of a small SPD matrix, in place, lower triangular. Serial and
// per-thread on purpose: m is 6 for a single end-effector, so a cooperative
// GLASS factorisation would cost more in synchronisation than it saves. (GLASS
// is also vendored and must not be modified.)
__device__ __forceinline__ void canon_chol(float* A, int m)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j <= i; j++) {
            float s = A[i * m + j];
            for (int k = 0; k < j; k++) s -= A[i * m + k] * A[j * m + k];
            if (i == j) {
                A[i * m + j] = sqrtf(fmaxf(s, 1e-12f));
            } else {
                A[i * m + j] = s / A[j * m + j];
            }
        }
    }
}

// Solve (L L^T) x = b in place on x, given the factor from canon_chol.
__device__ __forceinline__ void canon_chol_solve(const float* L, float* x, int m)
{
    for (int i = 0; i < m; i++) {
        float s = x[i];
        for (int k = 0; k < i; k++) s -= L[i * m + k] * x[k];
        x[i] = s / L[i * m + i];
    }
    for (int i = m - 1; i >= 0; i--) {
        float s = x[i];
        for (int k = i + 1; k < m; k++) s -= L[k * m + i] * x[k];
        x[i] = s / L[i * m + i];
    }
}

__global__ void canonical_ik_kernel(
    const float* __restrict__ cfgs,        // (n_problems, n_act)
    const float* __restrict__ cfg_refs,    // (n_problems, n_act)
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
    const float* __restrict__ target_Ts,   // (n_problems, n_ee, 7)
    int n_problems, int n_joints, int n_act, int n_ee,
    int max_iters, float step, float tol, float damping,
    float* __restrict__ out_q,             // (n_problems, n_act)
    int*   __restrict__ out_iters)         // (n_problems,)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_problems) return;

    const int m = 6 * n_ee;

    float q[MAX_ACT], qref[MAX_ACT], w[MAX_ACT];
    float T_world[MAX_JOINTS * 7];
    float r[CANON_MAX_RES], J[CANON_MAX_RES * MAX_ACT];
    float G[CANON_MAX_RES * CANON_MAX_RES], y[CANON_MAX_RES];

    for (int a = 0; a < n_act; a++) {
        q[a]    = cfgs[(size_t)p * n_act + a];
        qref[a] = cfg_refs[(size_t)p * n_act + a];
    }

    int used = max_iters;
    for (int it = 0; it < max_iters; it++) {
        compute_multi_ee_residual_and_jacobian(
            q, T_world, twists, parent_tf, parent_idx, act_idx,
            mimic_mul, mimic_off, mimic_act_idx, topo_inv,
            target_jnts, ancestor_masks,
            target_Ts + (size_t)p * n_ee * 7,
            n_joints, n_act, n_ee, r, J);

        // G = J J^T + damping I, then factor once and reuse for both solves.
        // Forming the m x m normal matrix (m = 6) rather than an SVD is the
        // point: pinv-by-SVD was measured to be the wrong tool here.
        for (int i = 0; i < m; i++) {
            for (int j = 0; j <= i; j++) {
                float s = 0.0f;
                for (int a = 0; a < n_act; a++) s += J[i * n_act + a] * J[j * n_act + a];
                if (i == j) s += damping;
                G[i * m + j] = s;
                G[j * m + i] = s;
            }
        }
        canon_chol(G, m);

        // Pose term: dp = -J^T (G^-1 r)
        for (int i = 0; i < m; i++) y[i] = r[i];
        canon_chol_solve(G, y, m);

        // Null-space term: w = q_ref - q, then dn = w - J^T (G^-1 (J w)),
        // which is (I - J^+ J) w without ever forming the n x n projector.
        for (int a = 0; a < n_act; a++) w[a] = qref[a] - q[a];
        float Jw[CANON_MAX_RES];
        for (int i = 0; i < m; i++) {
            float s = 0.0f;
            for (int a = 0; a < n_act; a++) s += J[i * n_act + a] * w[a];
            Jw[i] = s;
        }
        canon_chol_solve(G, Jw, m);

        float step_sq = 0.0f;
        for (int a = 0; a < n_act; a++) {
            float dp = 0.0f, dn = 0.0f;
            for (int i = 0; i < m; i++) {
                dp -= J[i * n_act + a] * y[i];
                dn -= J[i * n_act + a] * Jw[i];
            }
            dn += w[a];
            const float d = dp + step * dn;
            q[a] += d;
            step_sq += d * d;
        }

        // Early exit. The JAX version ran a FIXED count and so both wasted work
        // on easy problems and silently failed to converge on hard ones at large
        // batch (|r| degraded to 1.8e-2 at B=1024).
        if (step_sq < tol * tol) { used = it + 1; break; }
    }

    for (int a = 0; a < n_act; a++) out_q[(size_t)p * n_act + a] = q[a];
    out_iters[p] = used;
}

// ---------------------------------------------------------------------------
// XLA FFI handler
// ---------------------------------------------------------------------------

static ffi::Error CanonicalIkImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F32> cfgs,
    ffi::Buffer<ffi::DataType::F32> cfg_refs,
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
    int64_t max_iters,
    float   step,
    float   tol,
    float   damping,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_q,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> out_iters)
{
    const int n_problems = static_cast<int>(cfgs.dimensions()[0]);
    const int n_act      = static_cast<int>(cfgs.dimensions()[1]);
    const int n_joints   = static_cast<int>(twists.dimensions()[0]);
    const int n_ee       = static_cast<int>(target_jnts.dimensions()[0]);

    if (n_act > MAX_ACT || n_joints > MAX_JOINTS || n_ee > MAX_EE)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "canonical_ik: robot exceeds compiled capacity "
                          "(rebuild with --max-act / --max-joints)");

    const int block = 64;
    const int grid  = (n_problems + block - 1) / block;

    canonical_ik_kernel<<<grid, block, 0, stream>>>(
        cfgs.typed_data(), cfg_refs.typed_data(),
        twists.typed_data(), parent_tf.typed_data(),
        parent_idx.typed_data(), act_idx.typed_data(),
        mimic_mul.typed_data(), mimic_off.typed_data(),
        mimic_act_idx.typed_data(), topo_inv.typed_data(),
        target_jnts.typed_data(), ancestor_masks.typed_data(),
        target_Ts.typed_data(),
        n_problems, n_joints, n_act, n_ee,
        static_cast<int>(max_iters), step, tol, damping,
        out_q->typed_data(), out_iters->typed_data());

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CanonicalIkFfi, CanonicalIkImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // cfgs
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // cfg_refs
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
        .Attr<int64_t>("max_iters")
        .Attr<float>("step")
        .Attr<float>("tol")
        .Attr<float>("damping")
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // out_q
        .Ret<ffi::Buffer<ffi::DataType::S32>>()); // out_iters
