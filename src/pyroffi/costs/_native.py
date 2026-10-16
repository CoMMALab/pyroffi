"""Native residuals: costs whose rows and Jacobian come from hand-written device code.

The cost compiler traces a residual's jaxpr and emits it as straight-line CUDA, which is the
right form for kinematic chains but explodes for residuals that are naturally loops: a
collision residual is (spheres x obstacles) distances and their derivatives, and its fully
unrolled body grows to megabytes that ptxas takes minutes on. A ``NativeResidual`` instead
supplies one ``__device__`` function per cost instance; the traced residual only reserves its
rows (zeros), so row kinds, weights and constraint handling are unchanged, and the generated
header calls the function after the traced body to fill those rows of ``r`` and ``J`` (or
``jt = J t`` in the column-parallel variant).

Device function contract (one cost instance, one variable of ``dof`` coordinates)::

    __device__ void <fn>(const float* q, float* r, float* J)   // J: (rows, dof) or nullptr
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class NativeResidual:
    fn: str          # device function name
    source: str      # its CUDA definition (and the tables it reads)
    rows: int        # residual rows per instance
    dof: int         # coordinates of the instance's variable
    # Runtime data the device code reads from mutable __device__ buffers, uploaded before every
    # solve by the exported host function ``setter`` (so the data is not part of the source, and
    # the compiled kernel is shared by every problem whose data differs only in value).
    setter: str = ""
    data: tuple = field(default=(), compare=False, hash=False)

    @property
    def digest(self) -> str:
        return hashlib.sha1(self.source.encode()).hexdigest()[:16]


def _arr(name: str, a, ctype: str = "float") -> str:
    a = np.asarray(a).ravel()
    if a.size == 0:
        a = np.zeros(1)
    body = ", ".join((f"{float(v):.9e}f" if ctype == "float" else str(int(v))) for v in a)
    return f"__device__ const {ctype} {name}[{a.size}] = {{{body}}};"


def world_collision_source(robot, robot_coll, world_geoms, margin: float, tag: str) -> NativeResidual:
    """Device code for ``r = sum_{sphere, obstacle} max(0, margin - d)`` of one config, and its
    exact Jacobian through the kinematic chain (sphere-centre distance gradients by central
    differences, which covers every primitive GRiM's distance functions support)."""
    from ..collision._cuda_collision import _extract_world_arrays, _spherized_local_geometry
    from ..collision._cuda_collision import link_parent_joint_for

    j = robot.joints
    NJ = int(j.num_joints)
    dof = int(j.num_actuated_joints)
    twists = np.asarray(j.twists, np.float32)                              # (NJ, 6) [v; w]
    ptf = np.asarray(j.parent_transforms, np.float32)                      # (NJ, 7) wxyz_xyz
    parent = np.asarray(j.parent_indices, np.int32)
    order = np.asarray(j._topo_sort_inv, np.int32)                         # processing order
    act = np.asarray(j.actuated_indices, np.int32)
    mim = np.asarray(j.mimic_act_indices, np.int32)
    src = np.where(mim == -1, act, mim)
    mul = np.asarray(j.mimic_multiplier, np.float32)
    off = np.asarray(j.mimic_offset, np.float32)

    sph = np.asarray(_spherized_local_geometry(robot_coll), np.float32)    # (S*N, 4), k = s*N + n
    lpj = np.asarray(link_parent_joint_for(robot, robot_coll), np.int32)   # (N,)
    N = len(lpj)
    keep = sph[:, 3] > 0
    k_idx = np.arange(len(sph))
    sph_joint = lpj[k_idx % N][keep]
    sph = sph[keep]

    items = world_geoms if isinstance(world_geoms, (list, tuple)) else (world_geoms,)
    chunks = [[], [], [], []]
    for w in items:
        w = w.reshape((-1,))
        for dst, a in zip(chunks, _extract_world_arrays(w)):
            if a.shape[0]:
                dst.append(np.asarray(a, np.float32))
    ws, wc, wb, wh = (np.concatenate(c) if c else np.zeros((0, n), np.float32)
                      for c, n in zip(chunks, (4, 7, 15, 6)))
    # The world is RUNTIME data: buffers of a capacity bucket per primitive type (the source,
    # hence the compile, depends only on the robot, the margin and the buckets), filled by
    # <ns>_set_world before each solve.
    cap = [_bucket(len(a)) for a in (ws, wc, wb, wh)]

    ns = f"pyroffi_native_{tag}"
    fn = f"{ns}::world_collision"
    source = f"""
#include "collision.cuh"
namespace {ns} {{
constexpr int NJ = {NJ}, DOF = {dof}, NSPH = {len(sph)};
__device__ int NWS, NWC, NWB, NWH;
constexpr float MARGIN = {float(margin):.9e}f;
{_arr("TW", twists)}
{_arr("PTF", ptf)}
{_arr("PAR", parent, "int")}
{_arr("ORD", order, "int")}
{_arr("SRC", src, "int")}
{_arr("MUL", mul)}
{_arr("OFF", off)}
{_arr("SPH", sph)}
{_arr("SJ", sph_joint, "int")}
__device__ float WS[{max(cap[0], 1) * 4}], WC[{max(cap[1], 1) * 7}], WB[{max(cap[2], 1) * 15}], WH[{max(cap[3], 1) * 6}];

__device__ __forceinline__ void quat_to_mat(const float* q, float* R) {{
  float w = q[0], x = q[1], y = q[2], z = q[3];
  R[0] = 1 - 2*(y*y + z*z); R[1] = 2*(x*y - w*z);     R[2] = 2*(x*z + w*y);
  R[3] = 2*(x*y + w*z);     R[4] = 1 - 2*(x*x + z*z); R[5] = 2*(y*z - w*x);
  R[6] = 2*(x*z - w*y);     R[7] = 2*(y*z + w*x);     R[8] = 1 - 2*(x*x + y*y);
}}

// Signed distance of a sphere (centre c, radius rr) to every obstacle; hinge sum.
__device__ __forceinline__ float hinge_sum(const float* c, float rr) {{
  float acc = 0.f;
  for (int o = 0; o < NWS; ++o) {{ const float* w = WS + 4*o;
    acc += fmaxf(MARGIN - sphere_sphere_dist(c[0], c[1], c[2], rr, w[0], w[1], w[2], w[3]), 0.f); }}
  for (int o = 0; o < NWC; ++o) {{ const float* w = WC + 7*o;
    acc += fmaxf(MARGIN - sphere_capsule_dist(c[0], c[1], c[2], rr, w[0], w[1], w[2], w[3], w[4], w[5], w[6]), 0.f); }}
  for (int o = 0; o < NWB; ++o) {{ const float* w = WB + 15*o;
    acc += fmaxf(MARGIN - sphere_box_dist(c[0], c[1], c[2], rr, w[0], w[1], w[2], w[3], w[4], w[5],
                                          w[6], w[7], w[8], w[9], w[10], w[11], w[12], w[13], w[14]), 0.f); }}
  for (int o = 0; o < NWH; ++o) {{ const float* w = WH + 6*o;
    acc += fmaxf(MARGIN - sphere_halfspace_dist(c[0], c[1], c[2], rr, w[0], w[1], w[2], w[3], w[4], w[5]), 0.f); }}
  return acc;
}}

__device__ void world_collision(const float* q, float* r, float* J) {{
  // FK: world pose of each joint (R, t), plus each joint's motion axis and origin.
  float Rw[NJ * 9], tw[NJ * 3], ax[NJ * 3], og[NJ * 3];
  for (int i = 0; i < NJ; ++i) {{
    const int jj = ORD[i];
    const int pj = PAR[jj];
    float Rp[9], tp[3];
    if (pj < 0) {{ for (int a = 0; a < 9; ++a) Rp[a] = (a % 4 == 0); tp[0] = tp[1] = tp[2] = 0.f; }}
    else {{ for (int a = 0; a < 9; ++a) Rp[a] = Rw[pj*9 + a]; for (int a = 0; a < 3; ++a) tp[a] = tw[pj*3 + a]; }}
    float Rc[9]; quat_to_mat(PTF + jj*7, Rc);
    const float* tc = PTF + jj*7 + 4;
    float F[9], fo[3];
    for (int a = 0; a < 3; ++a) {{
      for (int b = 0; b < 3; ++b) F[a*3+b] = Rp[a*3]*Rc[b] + Rp[a*3+1]*Rc[3+b] + Rp[a*3+2]*Rc[6+b];
      fo[a] = Rp[a*3]*tc[0] + Rp[a*3+1]*tc[1] + Rp[a*3+2]*tc[2] + tp[a];
    }}
    const float* v = TW + jj*6; const float* w = TW + jj*6 + 3;
    const int s = SRC[jj];
    const float qj = (s >= 0 ? q[s] : 0.f) * MUL[jj] + OFF[jj];
    // delta: Rodrigues about w (revolute) or translation along v (prismatic).
    const float sn = sinf(qj), cs = 1.f - cosf(qj);
    float K[9] = {{0.f, -w[2], w[1], w[2], 0.f, -w[0], -w[1], w[0], 0.f}};
    float D[9];
    for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) {{
      float k2 = K[a*3]*K[b] + K[a*3+1]*K[3+b] + K[a*3+2]*K[6+b];
      D[a*3+b] = (a == b) + sn * K[a*3+b] + cs * k2;
    }}
    for (int a = 0; a < 3; ++a) {{
      for (int b = 0; b < 3; ++b) Rw[jj*9 + a*3+b] = F[a*3]*D[b] + F[a*3+1]*D[3+b] + F[a*3+2]*D[6+b];
      tw[jj*3 + a] = fo[a] + qj * (F[a*3]*v[0] + F[a*3+1]*v[1] + F[a*3+2]*v[2]);
      const bool rev = (w[0]*w[0] + w[1]*w[1] + w[2]*w[2]) > 0.5f;
      const float* u = rev ? w : v;
      ax[jj*3 + a] = F[a*3]*u[0] + F[a*3+1]*u[1] + F[a*3+2]*u[2];
      og[jj*3 + a] = fo[a];
    }}
  }}
  float acc = 0.f;
  if (J) for (int a = 0; a < DOF; ++a) J[a] = 0.f;
  for (int k = 0; k < NSPH; ++k) {{
    const int pj = SJ[k];
    const float* ls = SPH + 4*k;
    float c[3];
    for (int a = 0; a < 3; ++a)
      c[a] = (pj < 0) ? ls[a] : Rw[pj*9 + a*3]*ls[0] + Rw[pj*9 + a*3+1]*ls[1] + Rw[pj*9 + a*3+2]*ls[2] + tw[pj*3 + a];
    const float h = hinge_sum(c, ls[3]);
    if (h <= 0.f) continue;
    acc += h;
    if (!J) continue;
    // d h / d c by central differences, then through the chain: dc/dq_j = ax x (c - og)
    // (revolute) or ax (prismatic), routed to the actuated / mimic source column.
    float g[3];
    for (int a = 0; a < 3; ++a) {{
      const float e = 1e-4f; float cp[3] = {{c[0], c[1], c[2]}}, cm[3] = {{c[0], c[1], c[2]}};
      cp[a] += e; cm[a] -= e;
      g[a] = (hinge_sum(cp, ls[3]) - hinge_sum(cm, ls[3])) / (2.f * e);
    }}
    for (int jj = pj; jj >= 0; jj = PAR[jj]) {{
      const int s = SRC[jj];
      if (s < 0) continue;
      const float* u = ax + jj*3;
      const float* w = TW + jj*6 + 3;
      float dc[3];
      if ((w[0]*w[0] + w[1]*w[1] + w[2]*w[2]) > 0.5f) {{
        const float d0 = c[0] - og[jj*3], d1 = c[1] - og[jj*3+1], d2 = c[2] - og[jj*3+2];
        dc[0] = u[1]*d2 - u[2]*d1; dc[1] = u[2]*d0 - u[0]*d2; dc[2] = u[0]*d1 - u[1]*d0;
      }} else {{ dc[0] = u[0]; dc[1] = u[1]; dc[2] = u[2]; }}
      J[s] += MUL[jj] * (g[0]*dc[0] + g[1]*dc[1] + g[2]*dc[2]);
    }}
  }}
  r[0] = acc;
}}
}}  // namespace {ns}
// Upload a world (counts <= the buckets). Synchronizes first: no in-flight solve reads the
// buffers while they change.
extern "C" __attribute__((visibility("default"))) int {ns}_set_world(
    const float* ws, int nws, const float* wc, int nwc, const float* wb, int nwb, const float* wh, int nwh) {{
  using namespace {ns};
  if (nws > {cap[0]} || nwc > {cap[1]} || nwb > {cap[2]} || nwh > {cap[3]}) return -1;
  cudaDeviceSynchronize();
  if (nws) cudaMemcpyToSymbol(WS, ws, sizeof(float) * 4 * nws);
  if (nwc) cudaMemcpyToSymbol(WC, wc, sizeof(float) * 7 * nwc);
  if (nwb) cudaMemcpyToSymbol(WB, wb, sizeof(float) * 15 * nwb);
  if (nwh) cudaMemcpyToSymbol(WH, wh, sizeof(float) * 6 * nwh);
  cudaMemcpyToSymbol(NWS, &nws, sizeof(int)); cudaMemcpyToSymbol(NWC, &nwc, sizeof(int));
  cudaMemcpyToSymbol(NWB, &nwb, sizeof(int)); cudaMemcpyToSymbol(NWH, &nwh, sizeof(int));
  cudaDeviceSynchronize();
  return (int)cudaGetLastError();
}}
"""
    data = tuple(np.ascontiguousarray(a, np.float32) for a in (ws, wc, wb, wh))
    return NativeResidual(fn=fn, source=source, rows=1, dof=dof, setter=f"{ns}_set_world", data=data)


def _bucket(n: int) -> int:
    """Capacity for n primitives: 0, else the next power of two >= max(n, 8)."""
    return 0 if n == 0 else 1 << (max(n, 8) - 1).bit_length()


def upload(native: NativeResidual, lib) -> None:
    """Write a native residual's runtime data into the library that runs it."""
    import ctypes
    if not native.setter:
        return
    ptr = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    args = []
    for a in native.data:
        args += [ptr(a), ctypes.c_int(len(a))]
    err = getattr(lib, native.setter)(*args)
    if err != 0:
        raise RuntimeError(f"{native.setter} failed ({err})")


def world_collision_native(robot, robot_coll, joint_var, world_geoms, margin: float):
    """Collision constraint ``sum max(0, margin - d) <= 0`` per configuration, as a native
    residual: one row per batch instance of ``joint_var``, compact device code. The world is
    runtime data (uploaded per solve), so a new scene of the same size bucket does not compile."""
    import jax.numpy as jnp
    from ._problem import Cost
    if robot.floating_base:
        # ponytail: the device FK walks the joints from a fixed root; add the base pose to it
        # when a floating-base planner needs native collision.
        raise NotImplementedError("world_collision_native: fixed-base robots only")
    tag = hashlib.sha1(repr((float(margin),)).encode()).hexdigest()[:6]
    nat = world_collision_source(robot, robot_coll, world_geoms, margin, tag)
    tag = nat.digest[:10]
    nat = world_collision_source(robot, robot_coll, world_geoms, margin, tag)

    def residual(vals, var):
        return jnp.zeros(nat.rows, jnp.float32) + 0.0 * jnp.sum(vals[var])
    return Cost(compute_residual=lambda vals, a, kw: residual(vals, *a), args=((joint_var,), {}),
                kind="constraint_leq_zero", name=f"native_world_collision_{tag}", native=nat)
