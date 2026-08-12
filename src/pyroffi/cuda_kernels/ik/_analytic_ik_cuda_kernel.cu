/**
 * Analytic IK CUDA kernel (canonical subproblem decomposition) with XLA FFI.
 *
 * Solves the 7-DOF `spherical shoulder + intersecting axes 5-6 + offset axis 7`
 * family (Franka Panda / FR3) in closed form for each value of the redundancy
 * parameter q7. See pyroffi/kinematics/_analytic_ik.py for the derivation; this
 * file is a direct transcription of it, so the two paths must agree to
 * numerical precision and a test asserts exactly that.
 *
 * Why this kernel looks nothing like _ls_ik / _hjcd_ik:
 *
 *   - **No iteration.** Every candidate is a fixed, branch-free arithmetic
 *     sequence. There is no convergence check, no line search, no stall
 *     detection, and therefore no warp divergence from threads finishing at
 *     different times. This is the property that makes analytic IK a much
 *     better GPU fit than the LM solvers already in the suite.
 *   - **No FK helper headers.** The arm geometry (axis lines, shoulder/wrist
 *     points, home pose) is precomputed host-side and passed in, so the kernel
 *     needs no robot-model traversal, no mimic-joint handling and no
 *     topological ordering — just 3-vector algebra.
 *   - **No linear solves.** The decomposition is built from Paden-Kahan
 *     subproblems, which are closed-form trigonometry. Nothing here needs
 *     GLASS or a Cholesky factorisation.
 *
 * Parallel decomposition: one block per target pose; threads stride over the
 * `n_q7 * 8` candidates (8 branches per q7: 2 elbow x 2 sign x 2 shoulder
 * pair). Each thread keeps its own best candidate, then a shared-memory
 * argmin-reduction picks the block's winner. Candidates are completely
 * independent, so there is no synchronisation inside the solve itself.
 *
 * Precision: the solve runs in `real` (double by default). The arithmetic
 * volume per candidate is a few hundred flops, so even at the 1:64 FP64 rate
 * of a consumer GA102 this is bandwidth- rather than FP64-bound; correctness of
 * an *analytic* method is worth more than the throughput. Compile with
 * -DANALYTIC_IK_USE_FLOAT to switch to float32.
 *
 * Build with:  bash build_kernels/build_analytic_ik_cuda.sh
 */

#include "_collision_cuda_helpers.cuh"

#include "xla/ffi/api/ffi.h"

#include <cmath>
#include <cfloat>
#include <type_traits>

namespace ffi = xla::ffi;

#ifdef ANALYTIC_IK_USE_FLOAT
typedef float real;
#define R_SQRT sqrtf
#define R_ATAN2 atan2f
#define R_ACOS acosf
#define R_COS cosf
#define R_SIN sinf
#define R_HYPOT hypotf
#define R_FABS fabsf
#define R_BIG FLT_MAX
#else
typedef double real;
#define R_SQRT sqrt
#define R_ATAN2 atan2
#define R_ACOS acos
#define R_COS cos
#define R_SIN sin
#define R_HYPOT hypot
#define R_FABS fabs
#define R_BIG DBL_MAX
#endif

#define N_JOINT 7
#define N_BRANCH 8

// Matches _subproblems._TOL; loose enough for f32 FK round-off on metre links.
#define SP_TOL ((real)1e-6)
#define TINY ((real)1e-12)

// ---------------------------------------------------------------------------
// 3-vector / 3x3 helpers (row-major 3x3)
// ---------------------------------------------------------------------------

__device__ __forceinline__ real dot3(const real* a, const real* b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__device__ __forceinline__ void cross3(const real* a, const real* b, real* out)
{
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ __forceinline__ real norm3(const real* a)
{
    return R_SQRT(dot3(a, a));
}

__device__ __forceinline__ void normalize3(real* a)
{
    real n = norm3(a);
    n = n > TINY ? n : (real)1;
    a[0] /= n; a[1] /= n; a[2] /= n;
}

/** out = M v, M row-major 3x3. */
__device__ __forceinline__ void mat3_vec(const real* M, const real* v, real* out)
{
    out[0] = M[0] * v[0] + M[1] * v[1] + M[2] * v[2];
    out[1] = M[3] * v[0] + M[4] * v[1] + M[5] * v[2];
    out[2] = M[6] * v[0] + M[7] * v[1] + M[8] * v[2];
}

/** out = M^T v. */
__device__ __forceinline__ void mat3T_vec(const real* M, const real* v, real* out)
{
    out[0] = M[0] * v[0] + M[3] * v[1] + M[6] * v[2];
    out[1] = M[1] * v[0] + M[4] * v[1] + M[7] * v[2];
    out[2] = M[2] * v[0] + M[5] * v[1] + M[8] * v[2];
}

/** C = A B, all row-major 3x3. */
__device__ __forceinline__ void mat3_mul(const real* A, const real* B, real* C)
{
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j)
            C[3 * i + j] = A[3 * i + 0] * B[j]
                         + A[3 * i + 1] * B[3 + j]
                         + A[3 * i + 2] * B[6 + j];
}

/** C = A^T B. */
__device__ __forceinline__ void mat3T_mul(const real* A, const real* B, real* C)
{
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j)
            C[3 * i + j] = A[i] * B[j]
                         + A[3 + i] * B[3 + j]
                         + A[6 + i] * B[6 + j];
}

/** Rodrigues rotation about unit axis k by theta, row-major 3x3. */
__device__ __forceinline__ void rot3(const real* k, real theta, real* R)
{
    real c = R_COS(theta), s = R_SIN(theta), v = (real)1 - c;
    real x = k[0], y = k[1], z = k[2];
    R[0] = c + x * x * v;      R[1] = x * y * v - z * s;  R[2] = x * z * v + y * s;
    R[3] = x * y * v + z * s;  R[4] = c + y * y * v;      R[5] = y * z * v - x * s;
    R[6] = x * z * v - y * s;  R[7] = y * z * v + x * s;  R[8] = c + z * z * v;
}

__device__ __forceinline__ real wrap_pi(real t)
{
    return R_ATAN2(R_SIN(t), R_COS(t));
}

// ---------------------------------------------------------------------------
// Canonical subproblems (mirrors pyroffi/kinematics/_subproblems.py)
// ---------------------------------------------------------------------------

/**
 * Subproblem 4: h . rot(k, theta) p = d.
 * Writes two roots; sets *is_ls when the sinusoid never reaches d, in which
 * case both roots hold the residual-minimising angle rather than NaN.
 */
__device__ __forceinline__ void subproblem4(const real* h, const real* p,
                                            const real* k, real d,
                                            real* theta, bool* is_ls)
{
    real kp = dot3(k, p);
    real hk = dot3(h, k);
    real kxp[3]; cross3(k, p, kxp);

    real A = dot3(h, p) - hk * kp;
    real B = dot3(h, kxp);
    real C = hk * kp;

    real R = R_HYPOT(A, B);
    real phi = R_ATAN2(B, A);
    real rhs = d - C;

    *is_ls = R_FABS(rhs) > R + SP_TOL;
    if (*is_ls) {
        real t = rhs > (real)0 ? phi : phi + (real)M_PI;
        theta[0] = wrap_pi(t);
        theta[1] = theta[0];
    } else {
        real ratio = R > SP_TOL ? rhs / R : (real)0;
        ratio = ratio > (real)1 ? (real)1 : (ratio < (real)-1 ? (real)-1 : ratio);
        real delta = R_ACOS(ratio);
        theta[0] = wrap_pi(phi + delta);
        theta[1] = wrap_pi(phi - delta);
    }
}

/** Subproblem 3: ||rot(k, theta) p1 - p2|| = d, via subproblem 4. */
__device__ __forceinline__ void subproblem3(const real* p1, const real* p2,
                                            const real* k, real d,
                                            real* theta, bool* is_ls)
{
    real rhs = (real)0.5 * (dot3(p1, p1) + dot3(p2, p2) - d * d);
    subproblem4(p2, p1, k, rhs, theta, is_ls);
}

/** Subproblem 1: rot(k, theta) p1 = p2. Unique root (or its LS minimiser). */
__device__ __forceinline__ real subproblem1(const real* p1, const real* p2,
                                            const real* k, bool* is_ls)
{
    real kp1 = dot3(k, p1), kp2 = dot3(k, p2);
    real a[3], b[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        a[i] = p1[i] - k[i] * kp1;
        b[i] = p2[i] - k[i] * kp2;
    }
    real axb[3]; cross3(a, b, axb);
    real theta = R_ATAN2(dot3(k, axb), dot3(a, b));

    *is_ls = (R_FABS(kp1 - kp2) > SP_TOL) ||
             (R_FABS(norm3(p1) - norm3(p2)) > SP_TOL);
    return wrap_pi(theta);
}

/**
 * Subproblem 2: rot(k1, t1) p1 = rot(k2, t2) p2.
 * Projecting onto k1 removes t1, leaving subproblem 4 for t2; each root then
 * fixes t1 by subproblem 1. Two solution pairs.
 */
__device__ __forceinline__ void subproblem2(const real* p1, const real* p2,
                                            const real* k1, const real* k2,
                                            real* t1, real* t2, bool* is_ls)
{
    bool ls4 = false;
    subproblem4(k1, p2, k2, dot3(k1, p1), t2, &ls4);

    bool ls1a = false, ls1b = false;
    real R[9], rp[3];
    rot3(k2, t2[0], R); mat3_vec(R, p2, rp);
    t1[0] = subproblem1(p1, rp, k1, &ls1a);
    rot3(k2, t2[1], R); mat3_vec(R, p2, rp);
    t1[1] = subproblem1(p1, rp, k1, &ls1b);

    bool norm_mismatch = R_FABS(norm3(p1) - norm3(p2)) > SP_TOL;
    *is_ls = ls4 || norm_mismatch || ls1a || ls1b;
}

// ---------------------------------------------------------------------------
// Geometry, shared by every candidate of a block
// ---------------------------------------------------------------------------

// Number of scalars in the packed geometry blob; must match ArmGeom's layout
// and the host-side packer in _analytic_ik_cuda.py.
#define GEOM_N_SCALARS 95

struct ArmGeom {
    real axes[N_JOINT * 3];      // k1..k7 world directions at home
    real points[N_JOINT * 3];    // a point on each axis
    real shoulder[3];            // S
    real wrist[3];               // W0
    real m_home[16];             // M (row-major 4x4)
    real m_home_inv[16];         // M^-1, precomputed host-side
    real cos_alpha;              // k5 . k6
    real lower[N_JOINT];
    real upper[N_JOINT];
};

/** T = exp([S] theta) as row-major 4x4, for the axis k through p0. */
__device__ __forceinline__ void screw_matrix(const real* k, const real* p0,
                                             real theta, real* T)
{
    real R[9];
    rot3(k, theta, R);
    real Rp[3];
    mat3_vec(R, p0, Rp);
#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = 0; j < 3; ++j) T[4 * i + j] = R[3 * i + j];
        T[4 * i + 3] = p0[i] - Rp[i];
    }
    T[12] = T[13] = T[14] = (real)0;
    T[15] = (real)1;
}

/** C = A B for row-major 4x4. */
__device__ __forceinline__ void mat4_mul(const real* A, const real* B, real* C)
{
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            C[4 * i + j] = A[4 * i + 0] * B[j]
                         + A[4 * i + 1] * B[4 + j]
                         + A[4 * i + 2] * B[8 + j]
                         + A[4 * i + 3] * B[12 + j];
}

/**
 * Rotation carrying a1 -> b1 and a2 -> b2, via Gram-Schmidt frames.
 * Two non-parallel correspondences determine a rotation uniquely.
 */
__device__ __forceinline__ void rotation_from_two_vectors(
    const real* a1, const real* a2, const real* b1, const real* b2, real* R)
{
    real E[9], F[9];
    // Columns of E are the frame built from (a1, a2); same for F from (b1, b2).
    real e1[3], e2[3], e3[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) e1[i] = a1[i];
    normalize3(e1);
    real d = dot3(e1, a2);
#pragma unroll
    for (int i = 0; i < 3; ++i) e2[i] = a2[i] - e1[i] * d;
    normalize3(e2);
    cross3(e1, e2, e3);

    real f1[3], f2[3], f3[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) f1[i] = b1[i];
    normalize3(f1);
    d = dot3(f1, b2);
#pragma unroll
    for (int i = 0; i < 3; ++i) f2[i] = b2[i] - f1[i] * d;
    normalize3(f2);
    cross3(f1, f2, f3);

#pragma unroll
    for (int i = 0; i < 3; ++i) {
        E[3 * i + 0] = e1[i]; E[3 * i + 1] = e2[i]; E[3 * i + 2] = e3[i];
        F[3 * i + 0] = f1[i]; F[3 * i + 1] = f2[i]; F[3 * i + 2] = f3[i];
    }
    // R = F E^T
    real Et[9];
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j) Et[3 * i + j] = E[3 * j + i];
    mat3_mul(F, Et, R);
}

/** Any unit vector perpendicular to k, chosen the same way as the JAX path. */
__device__ __forceinline__ void perp_to(const real* k, real* out)
{
    real alt[3] = {(real)0, (real)0, (real)0};
    if (R_FABS(k[0]) < (real)0.9) alt[0] = (real)1; else alt[1] = (real)1;
    cross3(k, alt, out);
    normalize3(out);
}

/** Combined position + angle error of a candidate configuration. */
__device__ real pose_error(const ArmGeom* g, const real* q, const real* target)
{
    real T[16], tmp[16], E[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) T[i] = (real)0;
    T[0] = T[5] = T[10] = T[15] = (real)1;

    for (int i = 0; i < N_JOINT; ++i) {
        screw_matrix(&g->axes[3 * i], &g->points[3 * i], q[i], E);
        mat4_mul(T, E, tmp);
#pragma unroll
        for (int j = 0; j < 16; ++j) T[j] = tmp[j];
    }
    mat4_mul(T, g->m_home, tmp);

    real dp = (real)0;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        real d = tmp[4 * i + 3] - target[4 * i + 3];
        dp += d * d;
    }
    // Chordal rotation error ||Re - I||_F / sqrt(2), matching the JAX path.
    // Deliberately NOT arccos((tr-1)/2): arccos has infinite derivative at 1,
    // which is precisely where a correct solution lives, so rounding error
    // there inflates enormously -- badly enough in f32 to reject most valid
    // solutions. The chordal form is smooth and well-conditioned near zero.
    real fro = (real)0;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            real re = (real)0;
#pragma unroll
            for (int k = 0; k < 3; ++k) re += tmp[4 * i + k] * target[4 * j + k];
            if (i == j) re -= (real)1;
            fro += re * re;
        }
    }
    return R_SQRT(dp) + R_SQRT(fro / (real)2);
}

// ---------------------------------------------------------------------------
// One candidate: (q7 sample, branch) -> q[7], validity
// ---------------------------------------------------------------------------

__device__ bool solve_candidate(const ArmGeom* g, const real* G,
                                real q7, int branch, real* q_out)
{
    const int i4    = branch & 1;          // elbow root
    const int isign = (branch >> 1) & 1;   // z sign
    const int ipair = (branch >> 2) & 1;   // shoulder pair

    const real* k1 = &g->axes[0];
    const real* k2 = &g->axes[3];
    const real* k3 = &g->axes[6];
    const real* k4 = &g->axes[9];
    const real* k5 = &g->axes[12];
    const real* k6 = &g->axes[15];

    // --- strip E7 to expose the link-6 body transform: GB = E1..E6 --------
    real E7inv[16], GB[16];
    screw_matrix(&g->axes[18], &g->points[18], -q7, E7inv);
    mat4_mul(G, E7inv, GB);

    real R6[9];
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j) R6[3 * i + j] = GB[4 * i + j];

    real W[3], a6[3];
    mat3_vec(R6, g->wrist, W);
#pragma unroll
    for (int i = 0; i < 3; ++i) W[i] += GB[4 * i + 3];
    mat3_vec(R6, k6, a6);

    real w[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) w[i] = W[i] - g->shoulder[i];
    real wn = norm3(w);
    real w_hat[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) w_hat[i] = w[i] / (wn > TINY ? wn : (real)1);

    // --- 1. elbow q4 from the shoulder-to-wrist distance -------------------
    real p1[3], p2[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        p1[i] = g->wrist[i] - g->points[9 + i];
        p2[i] = g->shoulder[i] - g->points[9 + i];
    }
    real q4c[2]; bool ls4 = false;
    subproblem3(p1, p2, k4, wn, q4c, &ls4);
    real q4 = q4c[i4];

    // --- 2. v3, m3 ---------------------------------------------------------
    real R4q[9], v3[3], m3[3];
    rot3(k4, q4, R4q);
    real tmp3[3];
    mat3_vec(R4q, p1, tmp3);                 // rot(k4,q4)(W0 - p4)
#pragma unroll
    for (int i = 0; i < 3; ++i)
        v3[i] = tmp3[i] + g->points[9 + i] - g->shoulder[i];
    mat3_vec(R4q, k5, m3);

    real v3n = norm3(v3);
    real v3_hat[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) v3_hat[i] = v3[i] / (v3n > TINY ? v3n : (real)1);

    // --- 3. z = R3 m3 from two linear constraints on a unit vector ---------
    real c1 = dot3(m3, v3_hat);              // z . w_hat  (angle preserved)
    real c2 = g->cos_alpha;                  // z . a6     (mechanism constant)
    real gg = dot3(w_hat, a6);
    real det = (real)1 - gg * gg;
    real det_safe = R_FABS(det) > TINY ? det : TINY;
    real alpha_c = (c1 - gg * c2) / det_safe;
    real beta_c  = (c2 - gg * c1) / det_safe;

    real z_par[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) z_par[i] = alpha_c * w_hat[i] + beta_c * a6[i];

    real perp[3]; cross3(w_hat, a6, perp);
    real perp_n = norm3(perp);
    real perp_hat[3];
#pragma unroll
    for (int i = 0; i < 3; ++i)
        perp_hat[i] = perp[i] / (perp_n > TINY ? perp_n : (real)1);

    real c_sq = (real)1 - dot3(z_par, z_par);
    bool degenerate = (perp_n < (real)1e-9) || (c_sq < (real)-1e-9);
    real c_mag = R_SQRT(c_sq > (real)0 ? c_sq : (real)0);

    real sign = isign ? (real)-1 : (real)1;
    real z[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) z[i] = z_par[i] + sign * c_mag * perp_hat[i];

    // --- 4. R3 from the two correspondences, then q1 q2 q3 -----------------
    real R3[9];
    rotation_from_two_vectors(v3_hat, m3, w_hat, z, R3);

    real R3k3[3];
    mat3_vec(R3, k3, R3k3);
    real t1[2], t2[2]; bool ls2 = false;
    subproblem2(R3k3, k3, k1, k2, t1, t2, &ls2);

    real q1 = -t1[ipair];
    real q2 =  t2[ipair];

    real Ra[9], Rb[9], P[9];
    rot3(k1, -q1, Ra);
    rot3(k2, -q2, Rb);
    mat3_mul(Rb, Ra, P);                     // rot(k2,-q2) rot(k1,-q1)
    real Pfull[9];
    mat3_mul(P, R3, Pfull);

    real u[3], Pu[3];
    perp_to(k3, u);
    mat3_vec(Pfull, u, Pu);
    bool ls1 = false;
    real q3 = subproblem1(u, Pu, k3, &ls1);

    // --- 5. wrist q5 q6 ----------------------------------------------------
    real Rq4[9], R4m[9];
    rot3(k4, q4, Rq4);
    mat3_mul(R3, Rq4, R4m);                  // R4 = R3 rot(k4, q4)

    real Q[9];
    mat3T_mul(R4m, R6, Q);                   // Q = R4^T R6

    real Qk6[3];
    mat3_vec(Q, k6, Qk6);
    bool ls5 = false;
    real q5 = subproblem1(k6, Qk6, k5, &ls5);

    real R5inv[9], Pw[9];
    rot3(k5, -q5, R5inv);
    mat3_mul(R5inv, Q, Pw);
    real uw[3], Pwu[3];
    perp_to(k6, uw);
    mat3_vec(Pw, uw, Pwu);
    bool ls6 = false;
    real q6 = subproblem1(uw, Pwu, k6, &ls6);

    q_out[0] = q1; q_out[1] = q2; q_out[2] = q3; q_out[3] = q4;
    q_out[4] = q5; q_out[5] = q6; q_out[6] = q7;

    return !(ls4 || ls2 || ls1 || ls5 || ls6 || degenerate);
}

// ---------------------------------------------------------------------------
// Collision
// ---------------------------------------------------------------------------
// Sphere positions are staged in shared memory as float rather than `real`.
// Collision decisions live at millimetre scale, where float carries ~1e-7 m of
// error -- four orders below the margin -- while halving the shared-memory
// footprint. That footprint is the binding constraint: 59 spheres x 3 coords
// per thread is what caps the block size once collision is enabled.
//
// Spheres are placed by the SAME cumulative screw transforms the pose-error
// scorer walks, so collision costs one gather and one transform per sphere on
// top of scoring rather than a second forward-kinematics pass.

/** World-frame positions of every collision sphere at configuration `q`. */
__device__ void place_spheres(const ArmGeom* g, const real* q,
                              const double* __restrict__ spheres_home,
                              const int* __restrict__ sphere_joint,
                              int n_sph, float* out /* [n_sph * 3] */)
{
    // T[i] = E1..Ei, built incrementally; spheres attached at depth i are
    // transformed as soon as T[i] is available, so only one 4x4 is ever live.
    real T[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) T[i] = (real)0;
    T[0] = T[5] = T[10] = T[15] = (real)1;

    for (int depth = 0; depth <= N_JOINT; ++depth) {
        if (depth > 0) {
            real E[16], tmp[16];
            screw_matrix(&g->axes[3 * (depth - 1)], &g->points[3 * (depth - 1)],
                         q[depth - 1], E);
            mat4_mul(T, E, tmp);
#pragma unroll
            for (int j = 0; j < 16; ++j) T[j] = tmp[j];
        }
        for (int k = 0; k < n_sph; ++k) {
            if (sphere_joint[k] != depth) continue;
            const double* p = &spheres_home[4 * k];
#pragma unroll
            for (int r3 = 0; r3 < 3; ++r3) {
                out[3 * k + r3] = (float)(T[4 * r3 + 0] * (real)p[0]
                                        + T[4 * r3 + 1] * (real)p[1]
                                        + T[4 * r3 + 2] * (real)p[2]
                                        + T[4 * r3 + 3]);
            }
        }
    }
}

/** Minimum signed clearance: robot-vs-world and robot self-collision.
 *
 * World primitives use the same buffer layouts and the same distance helpers as
 * every other CUDA IK solver in the suite (`_collision_cuda_helpers.cuh`), so a
 * world built for `ls`/`hjcd`/`sqp`/`mppi` works here unchanged:
 *   spheres    (Ms, 4)  = centre, radius
 *   capsules   (Mc, 7)
 *   boxes      (Mb, 15)
 *   halfspaces (Mh, 6)
 * The robot side is spheres-only by construction — `RobotCollisionSpherized`
 * *is* a sphere model — so only sphere-vs-X is ever needed.
 */
__device__ float clearance_of(const float* __restrict__ pos, int n_sph,
                              const double* __restrict__ spheres_home,
                              const float* __restrict__ world_spheres, int n_ws,
                              const float* __restrict__ world_capsules, int n_wc,
                              const float* __restrict__ world_boxes, int n_wb,
                              const float* __restrict__ world_halfspaces, int n_wh,
                              const int* __restrict__ self_pairs, int n_pairs)
{
    float best = 1e30f;

    for (int k = 0; k < n_sph; ++k) {
        const float x = pos[3 * k + 0], y = pos[3 * k + 1], z = pos[3 * k + 2];
        const float rr = (float)spheres_home[4 * k + 3];

        for (int m = 0; m < n_ws; ++m) {
            const float* o = world_spheres + m * 4;
            best = fminf(best, sphere_sphere_dist(x, y, z, rr, o[0], o[1], o[2], o[3]));
        }
        for (int m = 0; m < n_wc; ++m) {
            const float* o = world_capsules + m * 7;
            best = fminf(best, sphere_capsule_dist(x, y, z, rr,
                                                   o[0], o[1], o[2], o[3], o[4], o[5], o[6]));
        }
        for (int m = 0; m < n_wb; ++m) {
            const float* o = world_boxes + m * 15;
            best = fminf(best, sphere_box_dist(x, y, z, rr,
                                               o[0], o[1], o[2], o[3], o[4], o[5],
                                               o[6], o[7], o[8], o[9], o[10], o[11],
                                               o[12], o[13], o[14]));
        }
        for (int m = 0; m < n_wh; ++m) {
            const float* o = world_halfspaces + m * 6;
            best = fminf(best, sphere_halfspace_dist(x, y, z, rr,
                                                     o[0], o[1], o[2], o[3], o[4], o[5]));
        }
    }

    for (int p = 0; p < n_pairs; ++p) {
        const int a = self_pairs[2 * p + 0];
        const int b = self_pairs[2 * p + 1];
        float dx = pos[3 * a + 0] - pos[3 * b + 0];
        float dy = pos[3 * a + 1] - pos[3 * b + 1];
        float dz = pos[3 * a + 2] - pos[3 * b + 2];
        float d = sqrtf(dx * dx + dy * dy + dz * dz)
                - (float)spheres_home[4 * a + 3] - (float)spheres_home[4 * b + 3];
        best = fminf(best, d);
    }
    return best;
}

// ---------------------------------------------------------------------------
// Kernel: one block per target
// ---------------------------------------------------------------------------

// Templated on whether collision is enabled so nvcc emits two specialisations.
// This is not cosmetic: registers are allocated statically for the whole kernel,
// so a runtime `if (n_sph > 0)` guard still charges the collision path's
// footprint to every launch. Measured at 200 registers/thread with the
// collision code inlined, which caps occupancy near 17% on sm_86 and made the
// *plain* solve ~1.8x slower than before collision was added. `if constexpr`
// removes the code entirely from the no-collision specialisation.
template <bool WITH_COLLISION>
__global__ void analytic_ik_kernel(
    const double* __restrict__ geom_blob,   // [GEOM_N_SCALARS], ArmGeom layout
    const double* __restrict__ targets,     // [B, 4, 4] row-major
    const float* __restrict__ q7_samples,   // [S]
    const float* __restrict__ prev_cfg,     // [B, 7] or null
    const double* __restrict__ spheres_home, // [K, 4] world (x,y,z,r) at q=0
    const int* __restrict__ sphere_joint,    // [K] last joint moving each sphere
    const int* __restrict__ self_pairs,      // [P, 2] sphere-index pairs
    const float* __restrict__ world_spheres,    // [Ms, 4]
    const float* __restrict__ world_capsules,   // [Mc, 7]
    const float* __restrict__ world_boxes,      // [Mb, 15]
    const float* __restrict__ world_halfspaces, // [Mh, 6]
    int n_sph, int n_pairs,
    int n_ws, int n_wc, int n_wb, int n_wh,
    int n_q7, int batch, int respect_limits, int use_prev,
    float err_tol, float margin,
    float* __restrict__ q_out,              // [B, 7]
    float* __restrict__ err_out,            // [B]
    int*   __restrict__ found_out,          // [B]
    float* __restrict__ clear_out)          // [B]
{
    extern __shared__ char smem_raw[];
    real* s_score = (real*)smem_raw;                       // [blockDim.x]
    int*  s_idx   = (int*)&s_score[blockDim.x];            // [blockDim.x]
    float* s_clear = (float*)&s_idx[blockDim.x];           // [blockDim.x]  (collision only)
    real* s_G     = (real*)&s_clear[blockDim.x];           // [16]
    real* s_tgt   = &s_G[16];                              // [16]
    real* s_geom  = &s_tgt[16];                            // [GEOM_N_SCALARS]
    real* s_best  = &s_geom[GEOM_N_SCALARS];               // [blockDim.x * 7]
    // Per-thread sphere scratch. Sized by the host from n_sph; zero-length when
    // collision is disabled, so the no-collision path pays nothing.
    float* s_pos  = (float*)&s_best[blockDim.x * N_JOINT];  // [blockDim.x*n_sph*3]

    const int b = blockIdx.x;
    if (b >= batch) return;

    // Stage the geometry into shared memory once per block. It is read by every
    // candidate, so this turns ~95 repeated global loads per candidate into one
    // coalesced cooperative copy. Arriving as f64 keeps the blob layout
    // independent of whether `real` is float or double.
    for (int i = threadIdx.x; i < GEOM_N_SCALARS; i += blockDim.x)
        s_geom[i] = (real)geom_blob[i];
    __syncthreads();
    const ArmGeom* geom = (const ArmGeom*)s_geom;

    // Load target and form G = T M^-1 once per block. Targets arrive in f64:
    // they feed the closed form directly, so rounding them to f32 would cap the
    // achievable pose accuracy near 1e-4 no matter how precise the solve is.
    if (threadIdx.x == 0) {
        real T[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) T[i] = (real)targets[b * 16 + i];
#pragma unroll
        for (int i = 0; i < 16; ++i) s_tgt[i] = T[i];
        mat4_mul(T, geom->m_home_inv, s_G);
    }
    __syncthreads();

    const int n_cand = n_q7 * N_BRANCH;

    real best_score = R_BIG;
    float best_clear = 1e30f;
    real best_q[N_JOINT];
#pragma unroll
    for (int i = 0; i < N_JOINT; ++i) best_q[i] = (real)0;

    for (int c = threadIdx.x; c < n_cand; c += blockDim.x) {
        const int is = c / N_BRANCH;
        const int br = c % N_BRANCH;
        const real q7 = (real)q7_samples[is];

        real q[N_JOINT];
        bool clean = solve_candidate(geom, s_G, q7, br, q);

        bool finite = true;
#pragma unroll
        for (int i = 0; i < N_JOINT; ++i) finite = finite && isfinite((double)q[i]);

        real err = finite ? pose_error(geom, q, s_tgt) : R_BIG;

        bool in_lim = true;
        if (respect_limits) {
#pragma unroll
            for (int i = 0; i < N_JOINT; ++i)
                in_lim = in_lim && (q[i] >= geom->lower[i] - (real)1e-6)
                                && (q[i] <= geom->upper[i] + (real)1e-6);
        }
        bool ok = finite && (err < (real)err_tol) && in_lim;

        // Collision is ~10x the cost of the analytic solve per candidate, and
        // only a small fraction of candidates are pose-valid, so it is gated
        // behind `ok`. The divergence this costs is far cheaper than checking
        // collision for candidates that are already rejected.
        float clr = 1e30f;
        bool collision_free = true;
        if constexpr (WITH_COLLISION) {
            if (ok) {
                float* mypos = &s_pos[threadIdx.x * n_sph * 3];
                place_spheres(geom, q, spheres_home, sphere_joint, n_sph, mypos);
                clr = clearance_of(mypos, n_sph, spheres_home,
                                   world_spheres, n_ws, world_capsules, n_wc,
                                   world_boxes, n_wb, world_halfspaces, n_wh,
                                   self_pairs, n_pairs);
            }
            collision_free = (clr > margin);
        }

        // Ranking: pose error, or joint-space distance to the previous config
        // when continuity resolution is requested. Invalid branches are pushed
        // behind every valid one by a large additive penalty rather than being
        // skipped, so a block with no valid branch still returns its best
        // near-miss instead of garbage.
        real score;
        if (use_prev) {
            real d = (real)0;
#pragma unroll
            for (int i = 0; i < N_JOINT; ++i) {
                real dd = q[i] - (real)prev_cfg[b * N_JOINT + i];
                d += dd * dd;
            }
            score = R_SQRT(d);
        } else {
            score = err;
        }
        // Lexicographic: pose-invalid worst, then colliding, then the base
        // criterion. Collision must dominate continuity so a trajectory gets
        // the closest *collision-free* branch, not a compromise between the two.
        if (!collision_free) score += (real)1e3;
        if (!ok) score += (real)1e6;

        if (score < best_score) {
            best_score = score;
            best_clear = clr;
#pragma unroll
            for (int i = 0; i < N_JOINT; ++i) best_q[i] = q[i];
        }
        (void)clean; (void)ok;
    }

    s_score[threadIdx.x] = best_score;
    s_idx[threadIdx.x] = threadIdx.x;
    s_clear[threadIdx.x] = best_clear;
#pragma unroll
    for (int i = 0; i < N_JOINT; ++i) s_best[threadIdx.x * N_JOINT + i] = best_q[i];
    __syncthreads();

    // Argmin reduction over threads.
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (s_score[threadIdx.x + stride] < s_score[threadIdx.x]) {
                s_score[threadIdx.x] = s_score[threadIdx.x + stride];
                s_idx[threadIdx.x] = s_idx[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const int win = s_idx[0];
#pragma unroll
        for (int i = 0; i < N_JOINT; ++i)
            q_out[b * N_JOINT + i] = (float)s_best[win * N_JOINT + i];
        // The winner's own error/ok are recomputed rather than shuffled through
        // shared memory: cheaper than another reduction and avoids a race.
        real qw[N_JOINT];
#pragma unroll
        for (int i = 0; i < N_JOINT; ++i) qw[i] = s_best[win * N_JOINT + i];
        real e = pose_error(geom, qw, s_tgt);
        bool lim = true;
        if (respect_limits) {
#pragma unroll
            for (int i = 0; i < N_JOINT; ++i)
                lim = lim && (qw[i] >= geom->lower[i] - (real)1e-6)
                          && (qw[i] <= geom->upper[i] + (real)1e-6);
        }
        err_out[b] = (float)e;
        found_out[b] = (e < (real)err_tol && lim) ? 1 : 0;
        clear_out[b] = s_clear[win];
    }
}

// ---------------------------------------------------------------------------
// FFI entry point
// ---------------------------------------------------------------------------

static ffi::Error AnalyticIkCudaImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F64> geom_blob,
    ffi::Buffer<ffi::DataType::F64> targets,
    ffi::Buffer<ffi::DataType::F32> q7_samples,
    ffi::Buffer<ffi::DataType::F32> prev_cfg,
    ffi::Buffer<ffi::DataType::F64> spheres_home,
    ffi::Buffer<ffi::DataType::S32> sphere_joint,
    ffi::Buffer<ffi::DataType::S32> self_pairs,
    ffi::Buffer<ffi::DataType::F32> world_spheres,
    ffi::Buffer<ffi::DataType::F32> world_capsules,
    ffi::Buffer<ffi::DataType::F32> world_boxes,
    ffi::Buffer<ffi::DataType::F32> world_halfspaces,
    int64_t respect_limits,
    int64_t use_prev,
    float err_tol,
    float margin,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> q_out,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> err_out,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> found_out,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> clear_out)
{
    const auto tdims = targets.dimensions();
    if (tdims.size() != 3 || tdims[1] != 4 || tdims[2] != 4)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "analytic_ik_cuda: targets must be [B, 4, 4]");

    const int batch = static_cast<int>(tdims[0]);
    const int n_q7 = static_cast<int>(q7_samples.dimensions()[0]);

    // Packed host-side into exactly ArmGeom's field order (see the packer in
    // _analytic_ik_cuda.py). Checked rather than assumed: a silent layout drift
    // here would produce plausible-looking wrong joint angles.
    if (static_cast<size_t>(geom_blob.dimensions()[0]) != GEOM_N_SCALARS)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "analytic_ik_cuda: geometry blob has the wrong size; "
                          "rebuild the kernel or regenerate the blob");

    const int n_sph = static_cast<int>(spheres_home.dimensions()[0]);
    const int n_pairs = static_cast<int>(self_pairs.dimensions()[0]);
    const int n_ws = static_cast<int>(world_spheres.dimensions()[0]);
    const int n_wc = static_cast<int>(world_capsules.dimensions()[0]);
    const int n_wb = static_cast<int>(world_boxes.dimensions()[0]);
    const int n_wh = static_cast<int>(world_halfspaces.dimensions()[0]);

    const int n_cand = n_q7 * N_BRANCH;
    int threads = 128;
    while (threads > 32 && threads > n_cand) threads >>= 1;

    // Per-thread shared cost. The sphere scratch dominates once collision is
    // enabled (59 spheres x 3 floats = 708 B/thread), so the block size is
    // shrunk until the request fits. Kept a power of two: the argmin reduction
    // below halves the block each step.
    auto shmem_for = [&](int t) -> size_t {
        return t * sizeof(real)                     // s_score
             + t * sizeof(int)                      // s_idx
             + t * sizeof(float)                    // s_clear
             + 32 * sizeof(real)                    // s_G + s_tgt
             + GEOM_N_SCALARS * sizeof(real)        // s_geom
             + t * N_JOINT * sizeof(real)           // s_best
             + (size_t)t * n_sph * 3 * sizeof(float);  // s_pos
    };
    int shmem_limit = 48 * 1024;
    while (threads > 32 && shmem_for(threads) > (size_t)shmem_limit) threads >>= 1;
    const size_t shmem = shmem_for(threads);
    if (shmem > (size_t)shmem_limit)
        return ffi::Error(ffi::ErrorCode::kResourceExhausted,
                          "analytic_ik_cuda: collision scratch exceeds shared "
                          "memory; reduce the sphere count or disable collision");

    auto launch = [&](auto with_collision) {
        analytic_ik_kernel<decltype(with_collision)::value>
            <<<batch, threads, shmem, stream>>>(
        geom_blob.typed_data(),
        targets.typed_data(),
        q7_samples.typed_data(),
        prev_cfg.typed_data(),
        spheres_home.typed_data(),
        sphere_joint.typed_data(),
        self_pairs.typed_data(),
        world_spheres.typed_data(),
        world_capsules.typed_data(),
        world_boxes.typed_data(),
        world_halfspaces.typed_data(),
        n_sph, n_pairs, n_ws, n_wc, n_wb, n_wh,
        n_q7, batch,
        static_cast<int>(respect_limits),
        static_cast<int>(use_prev),
        err_tol, margin,
        q_out->typed_data(),
        err_out->typed_data(),
        found_out->typed_data(),
        clear_out->typed_data());
    };

    if (n_sph > 0) launch(std::true_type{});
    else           launch(std::false_type{});

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AnalyticIkCudaFfi, AnalyticIkCudaImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // geom_blob
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // targets
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // q7_samples
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // prev_cfg
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // spheres_home
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // sphere_joint
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // self_pairs
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_spheres
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_capsules
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_boxes
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_halfspaces
        .Attr<int64_t>("respect_limits")
        .Attr<int64_t>("use_prev")
        .Attr<float>("err_tol")
        .Attr<float>("margin")
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // q_out
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // err_out
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // found_out
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // clear_out
);
