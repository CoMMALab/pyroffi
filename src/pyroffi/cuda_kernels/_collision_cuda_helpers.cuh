#pragma once

#include "_fk_cuda_helpers.cuh"

#include <cmath>

__device__ __forceinline__ float sql2(
    float ax, float ay, float az, float bx, float by, float bz)
{
    float dx = ax - bx, dy = ay - by, dz = az - bz;
    return dx * dx + dy * dy + dz * dz;
}

__device__ __forceinline__ float sphere_sphere_dist(
    float ax, float ay, float az, float ar,
    float bx, float by, float bz, float br)
{
    return sqrtf(sql2(ax, ay, az, bx, by, bz)) - (ar + br);
}

__device__ __forceinline__ float sphere_capsule_dist(
    float sx, float sy, float sz, float sr,
    float x1, float y1, float z1,
    float x2, float y2, float z2, float cr)
{
    float vx = x2 - x1, vy = y2 - y1, vz = z2 - z1;
    float len2 = vx * vx + vy * vy + vz * vz;
    float t = 0.0f;
    if (len2 > 1e-12f) {
        t = ((sx - x1) * vx + (sy - y1) * vy + (sz - z1) * vz) / len2;
        t = fmaxf(0.0f, fminf(1.0f, t));
    }
    float cx = x1 + t * vx, cy = y1 + t * vy, cz = z1 + t * vz;
    return sqrtf(sql2(sx, sy, sz, cx, cy, cz)) - (sr + cr);
}

__device__ __forceinline__ float box_sdf_local(
    float p1, float p2, float p3,
    float hl1, float hl2, float hl3)
{
    float q1 = fabsf(p1) - hl1, q2 = fabsf(p2) - hl2, q3 = fabsf(p3) - hl3;
    float mq1 = fmaxf(q1, 0.0f), mq2 = fmaxf(q2, 0.0f), mq3 = fmaxf(q3, 0.0f);
    return sqrtf(mq1 * mq1 + mq2 * mq2 + mq3 * mq3)
           + fminf(fmaxf(fmaxf(q1, q2), q3), 0.0f);
}

__device__ __forceinline__ float sphere_box_dist(
    float sx, float sy, float sz, float sr,
    float bcx, float bcy, float bcz,
    float a1x, float a1y, float a1z,
    float a2x, float a2y, float a2z,
    float a3x, float a3y, float a3z,
    float hl1, float hl2, float hl3)
{
    float dx = sx - bcx, dy = sy - bcy, dz = sz - bcz;
    return box_sdf_local(dx * a1x + dy * a1y + dz * a1z,
                         dx * a2x + dy * a2y + dz * a2z,
                         dx * a3x + dy * a3y + dz * a3z,
                         hl1, hl2, hl3) - sr;
}

__device__ __forceinline__ float capsule_box_dist(
    float x1, float y1, float z1,
    float x2, float y2, float z2, float cr,
    float bcx, float bcy, float bcz,
    float a1x, float a1y, float a1z,
    float a2x, float a2y, float a2z,
    float a3x, float a3y, float a3z,
    float hl1, float hl2, float hl3)
{
    float d1x = x1 - bcx, d1y = y1 - bcy, d1z = z1 - bcz;
    float al1 = d1x * a1x + d1y * a1y + d1z * a1z;
    float al2 = d1x * a2x + d1y * a2y + d1z * a2z;
    float al3 = d1x * a3x + d1y * a3y + d1z * a3z;
    float d2x = x2 - bcx, d2y = y2 - bcy, d2z = z2 - bcz;
    float bl1 = d2x * a1x + d2y * a1y + d2z * a1z;
    float bl2 = d2x * a2x + d2y * a2y + d2z * a2z;
    float bl3 = d2x * a3x + d2y * a3y + d2z * a3z;
    float ab1 = bl1 - al1, ab2 = bl2 - al2, ab3 = bl3 - al3;

    // box_sdf_local(a + t*ab) is convex in t (the rounded-box SDF is convex
    // both outside the box and inside, where it reduces to
    // max_i(|p_i| - hl_i)), so ternary search over t in [0, 1] finds the
    // segment's true closest approach to the box. Projecting onto the point
    // closest to the box center (the previous approach) is only exact when
    // the box degenerates to a sphere; for an off-center penetration it can
    // report the capsule as separated when it actually intersects a face.
    float lo = 0.0f, hi = 1.0f;
    #pragma unroll
    for (int i = 0; i < 30; i++) {
        float m1 = lo + (hi - lo) / 3.0f;
        float m2 = hi - (hi - lo) / 3.0f;
        float f1 = box_sdf_local(al1 + m1 * ab1, al2 + m1 * ab2, al3 + m1 * ab3, hl1, hl2, hl3);
        float f2 = box_sdf_local(al1 + m2 * ab1, al2 + m2 * ab2, al3 + m2 * ab3, hl1, hl2, hl3);
        if (f1 > f2) { lo = m1; } else { hi = m2; }
    }
    float t = 0.5f * (lo + hi);
    return box_sdf_local(al1 + t * ab1, al2 + t * ab2, al3 + t * ab3,
                         hl1, hl2, hl3) - cr;
}

__device__ __forceinline__ float capsule_capsule_dist(
    float ax1, float ay1, float az1,
    float ax2, float ay2, float az2, float ar,
    float bx1, float by1, float bz1,
    float bx2, float by2, float bz2, float br)
{
    float d1x = ax2 - ax1, d1y = ay2 - ay1, d1z = az2 - az1;
    float d2x = bx2 - bx1, d2y = by2 - by1, d2z = bz2 - bz1;
    float rx = ax1 - bx1,  ry = ay1 - by1,  rz = az1 - bz1;
    float a = d1x * d1x + d1y * d1y + d1z * d1z;
    float e = d2x * d2x + d2y * d2y + d2z * d2z;
    float f = d2x * rx  + d2y * ry  + d2z * rz;
    const float EPS = 1e-10f;
    float s, t;
    if (a <= EPS && e <= EPS) {
        s = t = 0.0f;
    } else if (a <= EPS) {
        s = 0.0f;
        t = fmaxf(0.0f, fminf(1.0f, f / e));
    } else {
        float c = d1x * rx + d1y * ry + d1z * rz;
        if (e <= EPS) {
            t = 0.0f;
            s = fmaxf(0.0f, fminf(1.0f, -c / a));
        } else {
            float b = d1x * d2x + d1y * d2y + d1z * d2z;
            float denom = a * e - b * b;
            s = (fabsf(denom) > EPS) ? fmaxf(0.0f, fminf(1.0f, (b * f - c * e) / denom)) : 0.0f;
            t = (b * s + f) / e;
            if (t < 0.0f) {
                t = 0.0f;
                s = fmaxf(0.0f, fminf(1.0f, -c / a));
            } else if (t > 1.0f) {
                t = 1.0f;
                s = fmaxf(0.0f, fminf(1.0f, (b - c) / a));
            }
        }
    }
    float px = ax1 + s * d1x - (bx1 + t * d2x);
    float py = ay1 + s * d1y - (by1 + t * d2y);
    float pz = az1 + s * d1z - (bz1 + t * d2z);
    return sqrtf(px * px + py * py + pz * pz) - (ar + br);
}

__device__ __forceinline__ float sphere_halfspace_dist(
    float sx, float sy, float sz, float sr,
    float nx, float ny, float nz,
    float px, float py, float pz)
{
    return (sx - px) * nx + (sy - py) * ny + (sz - pz) * nz - sr;
}

__device__ __forceinline__ float capsule_halfspace_dist(
    float x1, float y1, float z1,
    float x2, float y2, float z2, float cr,
    float nx, float ny, float nz,
    float px, float py, float pz)
{
    float d1 = (x1 - px) * nx + (y1 - py) * ny + (z1 - pz) * nz;
    float d2 = (x2 - px) * nx + (y2 - py) * ny + (z2 - pz) * nz;
    return fminf(d1, d2) - cr;
}

__device__ __forceinline__ void apply_se3_point(
    const float* __restrict__ T,
    const float* __restrict__ p,
    float* __restrict__ out)
{
    quat_rotate(T, p, out);
    out[0] += T[4];
    out[1] += T[5];
    out[2] += T[6];
}

__device__ __forceinline__ float colldist_from_sdf(float d, float margin)
{
    d = fminf(d, margin);
    float val;
    if (d < 0.0f) {
        val = d - 0.5f * margin;
    } else {
        float diff = d - margin;
        val = -0.5f / (margin + 1e-6f) * diff * diff;
    }
    return fminf(val, 0.0f);
}

// ---------------------------------------------------------------------------
// Self-collision (shared)
// ---------------------------------------------------------------------------
// One implementation of the robot-vs-itself check, used by every kernel that
// needs it. It was previously written twice -- once in the standalone
// self-collision kernel and once in the fused FK+collision kernel -- and the IK
// solvers omitted it entirely, which was the bug this consolidates away:
// `ls`/`hjcd`/`sqp` checked world collision only, so "collision-free IK" could
// return a configuration with the arm folded through itself. Measured on the
// Panda, 6.5% of random in-limit configurations are self-colliding, so that is
// roughly 1 in 15 solutions.
//
// Inputs are the same flattened tables the fused kernel uses:
//   T          [n_joints, 7]  FK output, wxyz_xyz per joint
//   sph_local  [K, 4]         link-local (x, y, z, r), grouped by link
//   link_start [N + 1]        CSR offsets into the per-link sphere runs
//   link_joint [N]            posing joint per link; -1 for the root link,
//                             which is fixed at the world origin
//   pair_i/j   [P]            active link pairs (SRDF-filtered)
//
// Sphere positions are formed in registers from the FK transforms already in
// hand; nothing is written to global memory.

/** Identity transform (wxyz_xyz), for links with no parent joint. */
__device__ __constant__ float kIdentityTf[7] = {1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

__device__ __forceinline__ const float* link_tf(const float* __restrict__ T,
                                                const int* __restrict__ link_joint,
                                                int link)
{
    const int j = link_joint[link];
    return (j >= 0) ? T + (size_t)j * 7 : kIdentityTf;
}

/** Minimum signed distance over one active link pair. */
__device__ __forceinline__ float self_collision_pair_dist(
    const float* __restrict__ T,
    const float* __restrict__ sph_local,
    const int*   __restrict__ link_start,
    const int*   __restrict__ link_joint,
    int li, int lj)
{
    const float* Ti = link_tf(T, link_joint, li);
    const float* Tj = link_tf(T, link_joint, lj);

    float best = 1e9f;
    for (int a = link_start[li]; a < link_start[li + 1]; ++a) {
        float ci[3];
        apply_se3_point(Ti, sph_local + (size_t)a * 4, ci);
        const float ri = sph_local[(size_t)a * 4 + 3];
        for (int c = link_start[lj]; c < link_start[lj + 1]; ++c) {
            float cj[3];
            apply_se3_point(Tj, sph_local + (size_t)c * 4, cj);
            const float rj = sph_local[(size_t)c * 4 + 3];
            best = fminf(best, sphere_sphere_dist(ci[0], ci[1], ci[2], ri,
                                                  cj[0], cj[1], cj[2], rj));
        }
    }
    return best;
}

/** Minimum signed distance over ALL active pairs (negative == penetrating). */
__device__ __forceinline__ float self_collision_min_dist(
    const float* __restrict__ T,
    const float* __restrict__ sph_local,
    const int*   __restrict__ link_start,
    const int*   __restrict__ link_joint,
    const int*   __restrict__ pair_i,
    const int*   __restrict__ pair_j,
    int P)
{
    float best = 1e9f;
    for (int p = 0; p < P; ++p)
        best = fminf(best, self_collision_pair_dist(T, sph_local, link_start,
                                                    link_joint, pair_i[p], pair_j[p]));
    return best;
}

/**
 * Hinge penalty for an IK/trajopt cost term: sum of squared violation over
 * pairs closer than `margin`. Zero when the arm is clear, matching the shape of
 * the world-collision penalty the IK kernels already accumulate.
 */
__device__ __forceinline__ float self_collision_penalty(
    const float* __restrict__ T,
    const float* __restrict__ sph_local,
    const int*   __restrict__ link_start,
    const int*   __restrict__ link_joint,
    const int*   __restrict__ pair_i,
    const int*   __restrict__ pair_j,
    int P, float margin)
{
    float pen = 0.0f;
    for (int p = 0; p < P; ++p) {
        const float d = self_collision_pair_dist(T, sph_local, link_start,
                                                 link_joint, pair_i[p], pair_j[p]);
        if (d < margin) {
            const float v = d - margin;
            pen += v * v;
        }
    }
    return pen;
}

// ---------------------------------------------------------------------------
// Self-collision gradients
// ---------------------------------------------------------------------------
//
// `self_collision_penalty` above only produces a scalar. Added to a solver's
// merit function it can rank candidates and reject steps, but it contributes
// nothing to the Gauss-Newton normal equations or to an SQP subproblem, so the
// solver never *steps away* from a self-collision -- it only declines to step
// further in. Driving the penalty to zero then needs an enormous weight, which
// turns the term into a rejection filter and puts a cliff in the line search.
//
// The routines below supply the missing derivative. For the witness sphere pair
// (the closest pair of spheres across two links),
//
//     d(q) = ||c_a(q) - c_b(q)|| - r_a - r_b
//     dd/dq = u^T (dc_a/dq - dc_b/dq),   u = (c_a - c_b) / ||c_a - c_b||
//
// The point Jacobians are never formed: each joint's 3-vector contribution is
// projected onto `u` as it is produced, so the cost is one scalar per joint per
// pair rather than a 3 x n_act block.
//
// TODO(task 9): the non-smoothness below is documented but NOT mitigated -- no
// hysteresis on the active set, so the gradient can chatter near ties.
//
// NOTE ON SMOOTHNESS: the witness pair is an argmin, so d(q) is only piecewise
// smooth and dd/dq jumps when the witness switches. Gauss-Newton tolerates this
// (it is the same situation as any min-distance collision cost), but it is why
// the caller should keep an active set -- only pairs inside the margin -- rather
// than differentiating everything.

/** Accumulate `sign * u . (dpt/dq_a)` into g[a] for every joint on `j_start`'s
 *  chain to the root. Mirrors the EE-Jacobian construction in
 *  `_ik_cuda_helpers.cuh` (same twist/mimic handling), but walks `parent_idx`
 *  instead of consuming a precomputed ancestor mask, so it needs no extra
 *  host-side table. */
__device__ __forceinline__ void collision_point_grad_chain(
    const float* __restrict__ T,
    const float* __restrict__ twists,
    const int*   __restrict__ parent_idx,
    const int*   __restrict__ act_idx,
    const float* __restrict__ mimic_mul,
    const int*   __restrict__ mimic_act_idx,
    int j_start,
    const float* __restrict__ pt,
    const float* __restrict__ u,
    float sign,
    float* __restrict__ g)
{
    for (int j = j_start; j >= 0; j = parent_idx[j]) {
        const int a1 = act_idx[j];
        const int a2 = mimic_act_idx[j];
        if (a1 < 0 && a2 < 0) continue;

        const float* tw = twists + (size_t)j * 6;
        const float ang_sq = tw[3]*tw[3] + tw[4]*tw[4] + tw[5]*tw[5];
        const float lin_sq = tw[0]*tw[0] + tw[1]*tw[1] + tw[2]*tw[2];

        const float* T_j = T + (size_t)j * 7;
        const float* q_j = T_j;
        const float* p_j = T_j + 4;

        float contrib;
        if (ang_sq > 1e-6f) {
            const float inv_ang = rsqrtf(ang_sq);
            const float body_ax[3] = { tw[3]*inv_ang, tw[4]*inv_ang, tw[5]*inv_ang };
            float z[3];
            quat_rotate(q_j, body_ax, z);
            // u . (z x arm), arm = pt - p_j. Cross product written out so this
            // header keeps its current dependencies.
            const float ax = pt[0]-p_j[0], ay = pt[1]-p_j[1], az = pt[2]-p_j[2];
            const float cx = z[1]*az - z[2]*ay;
            const float cy = z[2]*ax - z[0]*az;
            const float cz = z[0]*ay - z[1]*ax;
            contrib = u[0]*cx + u[1]*cy + u[2]*cz;
        } else if (lin_sq > 1e-6f) {
            const float inv_lin = rsqrtf(lin_sq);
            const float body_ax[3] = { tw[0]*inv_lin, tw[1]*inv_lin, tw[2]*inv_lin };
            float z[3];
            quat_rotate(q_j, body_ax, z);
            contrib = u[0]*z[0] + u[1]*z[1] + u[2]*z[2];
        } else {
            continue;
        }

        const float s = sign * mimic_mul[j];
        if (a1 >= 0) g[a1] += s * contrib;
        if (a2 >= 0) g[a2] += s * contrib;
    }
}

// TODO(task 4): callers build the row `grad(d)^T p >= margin - d`, which demands
// full clearance recovery in ONE step -- the reason deeply-penetrating seeds need
// a restoration branch. The Faverjon-Tournassoud velocity damper used by Pink /
// mc_rtc / Stack-of-Tasks relaxes the RHS smoothly near the safety distance and
// carries an invariance proof. Prefer it.

/**
 * Signed distance for one active link pair, plus its gradient w.r.t. q.
 *
 * `g` is ACCUMULATED into, not overwritten, and is only touched when the pair
 * is closer than `margin` -- callers zero it once per pair and skip inactive
 * pairs entirely, which is what keeps the active set small.
 *
 * Returns the pair's minimum signed distance (negative == penetrating).
 */
__device__ __forceinline__ float self_collision_pair_dist_grad(
    const float* __restrict__ T,
    const float* __restrict__ sph_local,
    const int*   __restrict__ link_start,
    const int*   __restrict__ link_joint,
    int li, int lj,
    const float* __restrict__ twists,
    const int*   __restrict__ parent_idx,
    const int*   __restrict__ act_idx,
    const float* __restrict__ mimic_mul,
    const int*   __restrict__ mimic_act_idx,
    int n_act, float margin,
    float* __restrict__ g)
{
    const float* Ti = link_tf(T, link_joint, li);
    const float* Tj = link_tf(T, link_joint, lj);

    // Witness pair search. Same loop as `self_collision_pair_dist`, but the
    // winning centres are kept so the gradient can be taken at the argmin.
    float best = 1e9f;
    float ci[3] = {0.f, 0.f, 0.f}, cj[3] = {0.f, 0.f, 0.f};
    for (int a = link_start[li]; a < link_start[li + 1]; ++a) {
        float pa[3];
        apply_se3_point(Ti, sph_local + (size_t)a * 4, pa);
        const float ra = sph_local[(size_t)a * 4 + 3];
        for (int b = link_start[lj]; b < link_start[lj + 1]; ++b) {
            float pb[3];
            apply_se3_point(Tj, sph_local + (size_t)b * 4, pb);
            const float rb = sph_local[(size_t)b * 4 + 3];
            const float d = sphere_sphere_dist(pa[0], pa[1], pa[2], ra,
                                               pb[0], pb[1], pb[2], rb);
            if (d < best) {
                best = d;
                ci[0] = pa[0]; ci[1] = pa[1]; ci[2] = pa[2];
                cj[0] = pb[0]; cj[1] = pb[1]; cj[2] = pb[2];
            }
        }
    }
    if (best >= margin) return best;

    float ux = ci[0]-cj[0], uy = ci[1]-cj[1], uz = ci[2]-cj[2];
    const float nrm = sqrtf(ux*ux + uy*uy + uz*uz);
    // Coincident centres: the separation direction is undefined, so there is no
    // meaningful gradient. Report the distance and leave `g` alone rather than
    // manufacturing a direction (which would be an arbitrary push).
    if (nrm < 1e-9f) return best;
    const float inv = 1.0f / nrm;
    const float u[3] = { ux*inv, uy*inv, uz*inv };

    // A joint shared by both chains moves both spheres; its two contributions
    // carry opposite signs and cancel to the correct net effect.
    collision_point_grad_chain(T, twists, parent_idx, act_idx, mimic_mul,
                                    mimic_act_idx, link_joint[li], ci, u, +1.0f, g);
    collision_point_grad_chain(T, twists, parent_idx, act_idx, mimic_mul,
                                    mimic_act_idx, link_joint[lj], cj, u, -1.0f, g);
    return best;
}

// ---------------------------------------------------------------------------
// World-collision gradients
// ---------------------------------------------------------------------------
//
// The robot-vs-world penalty in the IK kernels has the same defect the
// self-collision penalty had: it reaches the merit function but not the normal
// equations, so the solver declines to step deeper into an obstacle without
// ever computing a step that leaves one.
//
// For a robot sphere centred at c(q), every primitive distance here is a
// function of the centre alone, so
//
//     dd/dq = (grad_c d)^T (dc/dq)
//
// and `collision_point_grad_chain` above already projects dc/dq onto a
// direction. All that is needed is grad_c d.

/** Primitive kinds, matching the buffer order used by every IK kernel. */
enum WorldPrimKind { kWorldSphere = 0, kWorldCapsule = 1, kWorldBox = 2, kWorldHalfspace = 3 };

/** Signed distance from a robot sphere (centre `c`, radius `rr`) to one world
 *  primitive. Thin dispatch over the existing per-type helpers so the gradient
 *  below cannot drift from the penalty. */
__device__ __forceinline__ float world_prim_dist(
    const float* __restrict__ c, float rr, int kind, const float* __restrict__ o)
{
    switch (kind) {
        case kWorldSphere:
            return sphere_sphere_dist(c[0], c[1], c[2], rr, o[0], o[1], o[2], o[3]);
        case kWorldCapsule:
            return sphere_capsule_dist(c[0], c[1], c[2], rr,
                                       o[0], o[1], o[2], o[3], o[4], o[5], o[6]);
        case kWorldBox:
            return sphere_box_dist(c[0], c[1], c[2], rr,
                                   o[0], o[1], o[2], o[3], o[4], o[5],
                                   o[6], o[7], o[8], o[9], o[10], o[11],
                                   o[12], o[13], o[14]);
        default:
            return sphere_halfspace_dist(c[0], c[1], c[2], rr,
                                         o[0], o[1], o[2], o[3], o[4], o[5]);
    }
}

/**
 * `grad_c d` for one robot-sphere/world-primitive pair, written to `u`.
 *
 * Central differences on `world_prim_dist` rather than four hand-derived
 * closest-point formulas. The deliberate trade: boxes and capsules need the
 * nearest surface point to differentiate analytically, which is a second
 * geometry implementation that can disagree with the penalty it is supposed to
 * be the derivative of -- the exact class of bug that makes a solver quietly
 * push the wrong way. Finite differences cannot drift, and only pairs already
 * inside the margin are ever differentiated, so the six extra distance
 * evaluations land on a small active set.
 *
 * These are true signed-distance functions, so the gradient is a unit vector
 * wherever it is defined; `u` is normalised to shed discretisation error.
 * Returns false at a non-differentiable point (centre exactly on the medial
 * axis, where the finite difference collapses) so the caller can skip the pair
 * instead of pushing in an arbitrary direction.
 */
__device__ __forceinline__ bool world_prim_grad_dir(
    const float* __restrict__ c, float rr, int kind, const float* __restrict__ o,
    float* __restrict__ u)
{
    // Comfortably above float32 noise on metre-scale geometry, and far below the
    // ~2cm margins these kernels run with.
    const float h = 1e-4f;
    float cp[3] = { c[0], c[1], c[2] };
    for (int ax = 0; ax < 3; ++ax) {
        cp[ax] = c[ax] + h;
        const float dp = world_prim_dist(cp, rr, kind, o);
        cp[ax] = c[ax] - h;
        const float dm = world_prim_dist(cp, rr, kind, o);
        cp[ax] = c[ax];
        u[ax] = (dp - dm) * (0.5f / h);
    }
    const float n2 = u[0]*u[0] + u[1]*u[1] + u[2]*u[2];
    if (n2 < 1e-12f) return false;
    const float inv = rsqrtf(n2);
    u[0] *= inv; u[1] *= inv; u[2] *= inv;
    return true;
}

// ---------------------------------------------------------------------------
// Shared Gauss-Newton assembly
// ---------------------------------------------------------------------------
//
// The self- and world-collision constraint sweeps were duplicated in the LS and
// SQP kernels. They must not drift: both are meant to be the derivative of the
// SAME penalty those kernels add to their merit function, and a divergence would
// have a solver descending a gradient that does not match the cost it is being
// scored on -- silently, and only under some geometries.
//
// The two kernels differ ONLY in how a constraint is absorbed:
//   LS   strides its rank-1 update by lane into the group's shared A_s/rhs_s
//   SQP  updates thread-local H_s/g_s on every lane, and also records the row
//        for its hard-constraint stage
// so that step is the callback and everything else is shared.

/** Model arrays needed to walk a joint chain. Usually the shared-memory copies. */
struct RobotChainRefs {
    const float* __restrict__ twists;
    const int*   __restrict__ parent_idx;
    const int*   __restrict__ act_idx;
    const float* __restrict__ mimic_mul;
    const int*   __restrict__ mimic_act_idx;
    int n_joints;
};

/** SRDF-filtered self-collision tables. `n_pairs == 0` disables self-collision. */
struct SelfCollisionRefs {
    const float* __restrict__ sph_local;   // (K, 4); link_start indexes THIS
    const int*   __restrict__ link_start;  // (N + 1)
    const int*   __restrict__ link_joint;  // (N)
    const int*   __restrict__ pair_i;      // (P)
    const int*   __restrict__ pair_j;      // (P)
    int n_pairs;
};

/** Robot spheres and the world primitives they are tested against. */
struct WorldCollisionRefs {
    const float* __restrict__ robot_spheres_local;    // (n_rs, 4), joint-frame
    const int*   __restrict__ robot_sphere_joint_idx; // (n_rs,)
    int n_robot_spheres;
    const float* __restrict__ spheres;    int n_spheres;
    const float* __restrict__ capsules;   int n_capsules;
    const float* __restrict__ boxes;      int n_boxes;
    const float* __restrict__ halfspaces; int n_halfspaces;
};

/**
 * Walk every ACTIVE collision constraint and hand each one to `accumulate`.
 *
 * A constraint is active only while its distance is inside `margin`; inactive
 * ones are skipped before any gradient is computed, which is what keeps the
 * differentiated set small (and is also why the argmin non-smoothness noted
 * above is tolerable).
 *
 * @param g_scratch  caller-owned, at least `n_act` floats. Zeroed per constraint
 *                   and filled with dd/dq. Passed in rather than declared here
 *                   so this header does not depend on the IK helpers' MAX_ACT.
 * @param accumulate `void(float* g, float viol)` — g is dd/dq (unscaled), viol is
 *                   `margin - d > 0`. The callee may modify `g` in place, e.g. to
 *                   apply Jacobi column scaling before absorbing it.
 */
template <typename AccumFn>
__device__ __forceinline__ void collision_gauss_newton_terms(
    const float* __restrict__ T,
    const RobotChainRefs&    chain,
    const SelfCollisionRefs& self,
    const WorldCollisionRefs& world,
    bool want_self, bool want_world,
    int n_act, float margin,
    float* __restrict__ g_scratch,
    AccumFn accumulate)
{
    if (want_self) {
        for (int p = 0; p < self.n_pairs; ++p) {
            for (int a = 0; a < n_act; ++a) g_scratch[a] = 0.0f;
            const float d = self_collision_pair_dist_grad(
                T, self.sph_local, self.link_start, self.link_joint,
                self.pair_i[p], self.pair_j[p],
                chain.twists, chain.parent_idx, chain.act_idx,
                chain.mimic_mul, chain.mimic_act_idx, n_act, margin, g_scratch);
            if (d >= margin) continue;
            accumulate(g_scratch, margin - d);
        }
    }

    if (want_world) {
        const float* prim_base[4]   = { world.spheres, world.capsules,
                                        world.boxes, world.halfspaces };
        const int    prim_count[4]  = { world.n_spheres, world.n_capsules,
                                        world.n_boxes, world.n_halfspaces };
        const int    prim_stride[4] = { 4, 7, 15, 6 };

        for (int i = 0; i < world.n_robot_spheres; ++i) {
            const int jidx = world.robot_sphere_joint_idx[i];
            if (jidx < 0 || jidx >= chain.n_joints) continue;

            const float* sp = world.robot_spheres_local + (size_t)i * 4;
            const float local_p[3] = { sp[0], sp[1], sp[2] };
            float cw[3];
            apply_se3_point(T + (size_t)jidx * 7, local_p, cw);
            const float rr = sp[3];

            for (int kind = 0; kind < 4; ++kind) {
                for (int m = 0; m < prim_count[kind]; ++m) {
                    const float* o = prim_base[kind] + (size_t)m * prim_stride[kind];
                    const float d = world_prim_dist(cw, rr, kind, o);
                    if (d >= margin) continue;

                    float u[3];
                    if (!world_prim_grad_dir(cw, rr, kind, o, u)) continue;

                    for (int a = 0; a < n_act; ++a) g_scratch[a] = 0.0f;
                    collision_point_grad_chain(
                        T, chain.twists, chain.parent_idx, chain.act_idx,
                        chain.mimic_mul, chain.mimic_act_idx, jidx, cw, u,
                        +1.0f, g_scratch);
                    accumulate(g_scratch, margin - d);
                }
            }
        }
    }
}
