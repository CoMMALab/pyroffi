"""Analytic inverse kinematics by canonical subproblem decomposition.

For the ``7dof_spherical_shoulder_offset_wrist`` family — a spherical shoulder,
an intersecting joint-5/6 pair, and joint 7 offset from that pair. The Franka
Panda and FR3 are both in it (see :mod:`._structure`). Shimizu et al.'s
arm-angle method does not apply to these arms: their wrists are not spherical,
so Pieper's criterion fails and no pure closed form exists over all seven
joints.

What *does* exist is a closed form **given q7**. Fixing the last joint pins
both the wrist point and the link-6 orientation, after which the remaining six
joints fall out of four subproblem solves with no iteration. q7 is then the
redundancy parameter, which a 7-DOF arm needs anyway — the "search" is
redundancy resolution, not a numerical fallback.

The derivation
--------------
Write forward kinematics in space-frame product-of-exponentials form,

    T(q) = E₁(q₁) E₂(q₂) … E₇(q₇) M,     Eᵢ(θ) = exp([Sᵢ] θ)

with ``M`` the home pose and ``Sᵢ`` the world screw axis of joint *i* at home.
Two points do the work:

* **S**, where axes 1-2-3 meet. Fixed in the base, so E₁E₂E₃ leave it alone.
* **W**, where axes 5-6 meet. Fixed in links 4, 5 and 6, so E₅ and E₆ leave it
  alone and its position depends only on q₁…q₄.

Given a target ``T`` let ``G = T M⁻¹ = E₁…E₇``. Then:

1. **Wrist point.** ``W = G E₇(−q₇) W₀`` — known once q₇ is chosen, because
   E₅ and E₆ fix W₀. Likewise the link-6 rotation ``R₆ = Rot(G E₇(−q₇))``,
   giving the world direction of axis 6 as ``a₆ = R₆ k₆``.

2. **Elbow, q₄.** Since E₁E₂E₃ fix S, the distance ‖W − S‖ depends on q₄ alone:
   ``‖E₄(q₄) W₀ − S‖ = ‖W − S‖``. That is subproblem 3. *(2 branches)*

3. **Shoulder, q₁q₂q₃.** With q₄ known, both ``v₃ = E₄(q₄) W₀ − S`` and
   ``m₃ = rot(k₄, q₄) k₅`` are known, and the shoulder rotation R₃ must satisfy
   ``R₃ v₃ = w`` (with ``w = W − S``) and ``z := R₃ m₃``. The extra constraint
   pinning the remaining freedom is that the angle between axes 5 and 6 is a
   mechanism constant: ``z · a₆ = k₅ · k₆``. Together with ``z · ŵ = m₃ · v̂₃``
   — an angle R₃ must preserve — that is two linear constraints on a unit
   vector, so z is determined up to a sign. *(2 branches)* Two vector
   correspondences ``(v₃ → w, m₃ → z)`` then fix R₃ outright, and q₁q₂ come
   from subproblem 2 *(2 branches)* with q₃ from subproblem 1.

4. **Wrist, q₅q₆.** ``rot(k₅,q₅) rot(k₆,q₆) = R₄ᵀ R₆``. Right-multiplying by
   k₆ kills q₆, leaving subproblem 1 for q₅, then another for q₆.

Eight branches per q₇, of which the geometrically consistent ones are kept by
scoring every candidate against forward kinematics — the decomposition can emit
spurious roots when a subproblem falls into its least-squares regime, and
scoring is cheaper and more robust than trying to prove them away analytically.

The least-squares fallbacks in :mod:`._subproblems` mean an unreachable target
yields a *nearest* configuration with a raised flag rather than NaN, which is
what makes this usable as a task-planner oracle: a near-miss reports as a
near-miss instead of poisoning the search.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float

from ._structure import detect
from ._subproblems import rot, subproblem1, subproblem2, subproblem3

if TYPE_CHECKING:
    from .._robot import Robot

#: Branches per q7 sample: 2 (elbow) x 2 (z sign) x 2 (shoulder pair).
N_BRANCH = 8


@dataclass(frozen=True)
class ArmGeometry:
    """Home-configuration geometry the solver needs, extracted once per robot.

    All quantities are in the world frame at ``q = 0``. Nothing here is
    robot-specific beyond being read off the model, so the same solver serves
    Panda, FR3, or any other arm in the family.
    """

    axes: Float[Array, "7 3"]        # k1..k7, unit world axis directions
    points: Float[Array, "7 3"]      # a point on each axis
    shoulder: Float[Array, "3"]      # S, where axes 1-2-3 meet
    wrist: Float[Array, "3"]         # W0, where axes 5-6 meet
    m_home: Float[Array, "4 4"]      # M, home pose of the target link
    cos_alpha: Float[Array, ""]      # k5 . k6, the fixed axis-5/6 angle
    lower: Float[Array, "7"]
    upper: Float[Array, "7"]

    @property
    def n_joints(self) -> int:
        return 7


# Registered as a pytree rather than passed as a static jit argument: every
# field is a JAX array, and arrays are not hashable. This also lets the
# geometry be closed over, vmapped, or differentiated through if a caller wants
# gradients with respect to link geometry.
jax.tree_util.register_pytree_node(
    ArmGeometry,
    lambda g: (
        (g.axes, g.points, g.shoulder, g.wrist, g.m_home, g.cos_alpha,
         g.lower, g.upper),
        None,
    ),
    lambda _, children: ArmGeometry(*children),
)


def build_geometry(robot: "Robot", ee_link_name: str) -> ArmGeometry:
    """Extract :class:`ArmGeometry`, validating the structural preconditions."""
    st = detect(robot, ee_link_name)
    if st.family != "7dof_spherical_shoulder_offset_wrist":
        raise ValueError(
            f"analytic IK: robot chain to {ee_link_name!r} has family "
            f"{st.family!r}, but this solver implements "
            f"'7dof_spherical_shoulder_offset_wrist' (spherical shoulder, "
            f"intersecting axes 5-6, offset axis 7).\n{st.describe()}")

    shoulder = st.concurrent(0, 3)
    wrist = st.concurrent(4, 6)
    if shoulder is None or wrist is None:      # classify() already checked
        raise ValueError(f"analytic IK: structure check failed\n{st.describe()}")

    axes = np.stack([a.direction for a in st.axes])
    points = np.stack([a.point for a in st.axes])

    n_act = len(robot.joints.actuated_names)
    m_home = _home_pose(robot, ee_link_name, jnp.zeros(n_act))

    lo, hi = robot.joints.lower_limits, robot.joints.upper_limits
    return ArmGeometry(
        axes=jnp.asarray(axes),
        points=jnp.asarray(points),
        shoulder=jnp.asarray(shoulder[0]),
        wrist=jnp.asarray(wrist[0]),
        m_home=jnp.asarray(m_home),
        cos_alpha=jnp.asarray(float(np.dot(axes[4], axes[5]))),
        lower=jnp.asarray(np.asarray(lo)[:7]),
        upper=jnp.asarray(np.asarray(hi)[:7]),
    )


def _home_pose(robot: "Robot", ee_link_name: str, cfg) -> np.ndarray:
    """``[4,4]`` world pose of ``ee_link_name`` at ``cfg``."""
    import jaxlie

    idx = list(robot.links.names).index(ee_link_name)
    Ts = jaxlie.SE3(robot.forward_kinematics(cfg))
    return np.asarray(Ts.as_matrix()[idx])


# --------------------------------------------------------------------------- #
# Screw helpers (pure rotation screws; every joint here is revolute)
# --------------------------------------------------------------------------- #

def _screw_apply_point(k, p0, theta, x):
    """Apply ``exp([S] theta)`` to point ``x``, for the axis through ``p0``."""
    return rot(k, theta) @ (x - p0) + p0


def _screw_matrix(k, p0, theta):
    """``exp([S] theta)`` as a ``[4,4]`` transform."""
    R = rot(k, theta)
    T = jnp.eye(4)
    T = T.at[:3, :3].set(R)
    T = T.at[:3, 3].set(p0 - R @ p0)
    return T


# --------------------------------------------------------------------------- #
# Core: all branches for one (target, q7)
# --------------------------------------------------------------------------- #

def _solve_fixed_q7(geom: ArmGeometry, target: Float[Array, "4 4"], q7):
    """Closed-form q1..q6 for a fixed q7. Returns ``(q[8,7], is_ls[8])``."""
    k = geom.axes
    pt = geom.points
    S = geom.shoulder
    W0 = geom.wrist

    # G = T M^-1 = E1..E7, then strip E7 to expose the link-6 body transform.
    G = target @ jnp.linalg.inv(geom.m_home)
    E7_inv = _screw_matrix(k[6], pt[6], -q7)
    GB = G @ E7_inv                      # = E1..E6

    W = GB[:3, :3] @ W0 + GB[:3, 3]      # wrist point in the world
    R6 = GB[:3, :3]
    a6 = R6 @ k[5]                       # world direction of axis 6

    w = W - S
    w_norm = jnp.linalg.norm(w)
    w_hat = w / jnp.maximum(w_norm, 1e-12)

    # --- 1. elbow q4 from the shoulder-to-wrist distance ------------------- #
    q4_cands, _, ls4 = subproblem3(W0 - pt[3], S - pt[3], k[3], w_norm)

    def per_q4(q4):
        v3 = _screw_apply_point(k[3], pt[3], q4, W0) - S
        m3 = rot(k[3], q4) @ k[4]
        v3_hat = v3 / jnp.maximum(jnp.linalg.norm(v3), 1e-12)

        # --- 2. z = R3 m3, from two linear constraints on a unit vector --- #
        # R3 preserves angles, so z.w_hat is known; the axis-5/6 angle is a
        # mechanism constant, giving z.a6. Solve in the (w_hat, a6) basis.
        c1 = jnp.dot(m3, v3_hat)         # z . w_hat
        c2 = geom.cos_alpha              # z . a6
        g = jnp.dot(w_hat, a6)
        det = 1.0 - g * g
        det_safe = jnp.where(jnp.abs(det) > 1e-12, det, 1e-12)
        alpha_c = (c1 - g * c2) / det_safe
        beta_c = (c2 - g * c1) / det_safe
        z_par = alpha_c * w_hat + beta_c * a6
        perp = jnp.cross(w_hat, a6)
        perp_n = jnp.linalg.norm(perp)
        perp_hat = perp / jnp.maximum(perp_n, 1e-12)
        c_sq = 1.0 - jnp.dot(z_par, z_par)
        # Degenerate (w_hat ∥ a6) or inconsistent (c_sq < 0) geometry falls back
        # to the in-plane part; the FK scoring pass rejects it if it is wrong.
        c_mag = jnp.sqrt(jnp.clip(c_sq, 0.0, jnp.inf))
        z_degenerate = (perp_n < 1e-9) | (c_sq < -1e-9)

        def per_sign(sign):
            z = z_par + sign * c_mag * perp_hat

            # --- 3. R3 from the two correspondences v3->w, m3->z ---------- #
            R3 = _rotation_from_two_vectors(v3_hat, m3, w_hat, z)

            # q1, q2 via subproblem 2; q3 via subproblem 1.
            t1, t2, _, ls2 = subproblem2(R3 @ k[2], k[2], k[0], k[1])

            def per_pair(theta1, theta2):
                q1 = -theta1
                q2 = theta2
                P = rot(k[1], -q2) @ rot(k[0], -q1) @ R3
                u = _perp_to(k[2])
                q3, ls1 = subproblem1(u, P @ u, k[2])

                # --- 4. wrist q5, q6 -------------------------------------- #
                R4 = R3 @ rot(k[3], q4)
                Q = R4.T @ R6
                q5, ls5 = subproblem1(k[5], Q @ k[5], k[4])
                Pw = rot(k[4], -q5) @ Q
                uw = _perp_to(k[5])
                q6, ls6 = subproblem1(uw, Pw @ uw, k[5])

                q = jnp.stack([q1, q2, q3, q4, q5, q6, q7])
                bad = ls4 | ls2 | ls1 | ls5 | ls6 | z_degenerate
                return q, bad

            return jax.vmap(per_pair)(t1, t2)

        return jax.vmap(per_sign)(jnp.array([1.0, -1.0]))

    q_all, ls_all = jax.vmap(per_q4)(q4_cands)      # [2, 2, 2, 7]
    return q_all.reshape(N_BRANCH, 7), ls_all.reshape(N_BRANCH)


def _rotation_from_two_vectors(a1, a2, b1, b2):
    """Rotation taking ``a1 -> b1`` and ``a2 -> b2`` (Gram-Schmidt frames).

    Two non-parallel correspondences determine a rotation uniquely. Built from
    orthonormal frames rather than by solving a Procrustes problem so it stays
    cheap and differentiable.
    """
    def frame(u, v):
        e1 = u / jnp.maximum(jnp.linalg.norm(u), 1e-12)
        v_perp = v - e1 * jnp.dot(e1, v)
        e2 = v_perp / jnp.maximum(jnp.linalg.norm(v_perp), 1e-12)
        return jnp.stack([e1, e2, jnp.cross(e1, e2)], axis=1)

    return frame(b1, b2) @ frame(a1, a2).T


def _perp_to(k):
    """Any unit vector perpendicular to ``k``, chosen smoothly and safely."""
    alt = jnp.where(jnp.abs(k[0]) < 0.9,
                    jnp.array([1.0, 0.0, 0.0]),
                    jnp.array([0.0, 1.0, 0.0]))
    v = jnp.cross(k, alt)
    return v / jnp.maximum(jnp.linalg.norm(v), 1e-12)


# --------------------------------------------------------------------------- #
# Scoring and public entry point
# --------------------------------------------------------------------------- #

def _pose_error(geom: ArmGeometry, q, target):
    """Combined position + orientation error of a candidate configuration."""
    T = jnp.eye(4)
    for i in range(7):
        T = T @ _screw_matrix(geom.axes[i], geom.points[i], q[i])
    T = T @ geom.m_home
    pos = jnp.linalg.norm(T[:3, 3] - target[:3, 3])
    # Chordal (Frobenius) rotation error rather than arccos((tr-1)/2).
    # arccos has infinite derivative at 1, which is exactly where a correct
    # solution sits: in float32 a ~1e-7 rounding error in the trace inflates to
    # ~4.5e-4 of apparent angle error, above any sane acceptance threshold. That
    # rejected 90% of valid solutions when SPaSM (which runs f32) called this.
    # ||Re - I||_F = 2*sqrt(2)*sin(theta/2) ~ sqrt(2)*theta for small theta, and
    # is smooth and well-conditioned there.
    R = T[:3, :3] @ target[:3, :3].T
    ang = jnp.linalg.norm(R - jnp.eye(3)) / jnp.sqrt(2.0)
    return pos + ang


def _within_limits(geom, q, slack=1e-6):
    return jnp.all((q >= geom.lower - slack) & (q <= geom.upper + slack))


#: Pose-error threshold below which a branch counts as a solution. The closed
#: form is exact, so this measures arithmetic precision, not solver quality —
#: and it must therefore track the working dtype. Under float32 the whole chain
#: (FK, screw products, target construction) carries ~1e-3 of position error on
#: a metre-scale arm; SPaSM's own hand-rolled IK lands at 1.6e-3 there. A 1e-4
#: threshold silently rejected 98% of *correct* solutions when called from
#: float32 code.
def default_err_tol() -> float:
    return 1e-4 if jax.config.jax_enable_x64 else 5e-3


@partial(jax.jit, static_argnums=(3, 4))
def solve_all(geom: ArmGeometry, target, q7_samples, respect_limits: bool = True,
              err_tol: float | None = None):
    """All candidate solutions over a set of q7 values.

    Returns ``(q[N, 8, 7], error[N, 8], ok[N, 8])`` where ``ok`` marks branches
    that both reproduce the target pose and respect joint limits.
    """
    tol = default_err_tol() if err_tol is None else err_tol

    def one(q7):
        q, ls = _solve_fixed_q7(geom, target, q7)
        err = jax.vmap(lambda qq: _pose_error(geom, qq, target))(q)
        lim = jax.vmap(lambda qq: _within_limits(geom, qq))(q)
        finite = jnp.all(jnp.isfinite(q), axis=-1)
        ok = finite & (err < tol) & jnp.where(respect_limits, lim, True)
        return q, err, ok

    return jax.vmap(one)(q7_samples)


def default_q7_samples(geom: ArmGeometry, n: int = 32):
    """Uniform sweep of the redundancy parameter across joint 7's range."""
    return jnp.linspace(geom.lower[6], geom.upper[6], n)


def analytic_ik_solve(
    robot: "Robot",
    target_link_name: str,
    target_pose,
    *,
    q7_samples=None,
    num_q7: int = 32,
    previous_cfg=None,
    respect_limits: bool = True,
    geometry: ArmGeometry | None = None,
):
    """Analytic IK for one target pose.

    Args:
        robot: the arm; must be in the supported structural family.
        target_link_name: link whose pose is being commanded.
        target_pose: ``jaxlie.SE3`` or ``[4,4]`` matrix.
        q7_samples: explicit redundancy-parameter values. Defaults to a uniform
            sweep of joint 7's range; pass a single value to pin it.
        previous_cfg: if given, the returned solution is the valid branch
            closest to it in joint space — continuity resolution for a
            trajectory, rather than an arbitrary branch.
        respect_limits: discard branches outside the joint limits.
        geometry: prebuilt :class:`ArmGeometry`, to avoid re-detecting the
            structure on every call.

    Returns:
        ``(q[7], found)`` — ``found`` is False when no branch reproduced the
        target, in which case ``q`` holds the lowest-error candidate.
    """
    import jaxlie

    geom = geometry if geometry is not None else build_geometry(robot, target_link_name)
    T = (target_pose.as_matrix() if isinstance(target_pose, jaxlie.SE3)
         else jnp.asarray(target_pose))
    samples = (default_q7_samples(geom, num_q7) if q7_samples is None
               else jnp.atleast_1d(jnp.asarray(q7_samples)))

    q, err, ok = solve_all(geom, T, samples, respect_limits)
    q_flat = q.reshape(-1, 7)
    err_flat = err.reshape(-1)
    ok_flat = ok.reshape(-1)

    if previous_cfg is None:
        score = jnp.where(ok_flat, err_flat, err_flat + 1e6)
    else:
        prev = jnp.asarray(previous_cfg)[:7]
        dist = jnp.linalg.norm(q_flat - prev[None, :], axis=-1)
        score = jnp.where(ok_flat, dist, dist + 1e6)

    best = jnp.argmin(score)
    return q_flat[best], jnp.any(ok_flat)


# --------------------------------------------------------------------------- #
# Backend dispatch
# --------------------------------------------------------------------------- #
# Measured on an RTX A5000, Panda, 32 q7 samples (256 candidates/target):
#
#     batch      JAX-GPU     CUDA     winner
#         1       0.99 ms   2.94 ms   JAX
#         8       1.74 ms   2.94 ms   JAX
#        20       2.99 ms   2.94 ms   ~even
#        32       4.02 ms   2.94 ms   CUDA
#       256      26.48 ms   6.55 ms   CUDA  (4.0x)
#      4096     403.6  ms  75.9  ms   CUDA  (5.3x)
#
# The CUDA kernel is flat until batch ~64 — that is FFI dispatch overhead, not
# work — while the JAX path costs ~0.99 ms fixed plus ~0.10 ms per target.
#
# The CUDA floor rose from 2.02 ms to 2.94 ms when the collision path was added,
# and this is dispatch cost, not kernel time: it is constant across every batch
# size, which is exactly how it was identified. Roughly half came from
# allocating the (empty) collision buffers per call and is fixed; the remaining
# ~0.9 ms is the marshalling of four extra input buffers and one extra output
# that the no-collision path does not need. Splitting the plain and collision
# FFI targets would recover it — the kernel is already templated on
# WITH_COLLISION, so only the handler is shared. Until then the crossover sits
# near batch 20 rather than 12.
CROSSOVER_BATCH = 20


def _cuda_available() -> bool:
    try:
        from ..cuda_kernels.ik._analytic_ik_cuda import _load_and_register

        _load_and_register()
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False


def analytic_ik_solve_batched(
    robot: "Robot",
    target_link_name: str,
    target_poses,
    *,
    q7_samples=None,
    num_q7: int = 32,
    previous_cfg=None,
    respect_limits: bool = True,
    geometry: ArmGeometry | None = None,
    backend: str = "auto",
    differentiable: bool = True,
    err_tol: float | None = None,
):
    """Analytic IK over a batch of target poses, on the faster backend.

    Args:
        target_poses: ``[B, 4, 4]`` (or a single ``[4, 4]``).
        backend: ``"auto"`` picks CUDA at or above :data:`CROSSOVER_BATCH` when a
            GPU kernel is available, else the JAX path. ``"jax"`` / ``"cuda"``
            force one.
        differentiable: attach the implicit-diff gradient w.r.t. the target
            poses. The CUDA kernel is opaque to autodiff, so without this a
            ``jax.grad`` through it silently yields zeros.

    Returns:
        ``(q[B, 7], found[B])``.
    """
    geom = geometry if geometry is not None else build_geometry(robot, target_link_name)

    T = jnp.asarray(target_poses)
    single = T.ndim == 2
    if single:
        T = T[None, ...]
    batch = T.shape[0]

    samples = (default_q7_samples(geom, num_q7) if q7_samples is None
               else jnp.atleast_1d(jnp.asarray(q7_samples)))

    # The solver is a black box for autodiff: the CUDA kernel is literally
    # non-differentiable, and differentiating the JAX decomposition would give
    # the wrong answer anyway (branch selection is piecewise constant). Cut the
    # gradient here and let _attach_implicit_gradient supply all of it.
    T_solve = jax.lax.stop_gradient(T)

    use_cuda = (backend == "cuda" or
                (backend == "auto" and batch >= CROSSOVER_BATCH and _cuda_available()))
    if backend == "cuda" and not _cuda_available():
        raise RuntimeError(
            "analytic IK: backend='cuda' requested but the kernel is "
            "unavailable; build it with build_kernels/build_analytic_ik_cuda.sh")

    if use_cuda:
        from ..cuda_kernels.ik._analytic_ik_cuda import (
            analytic_ik_cuda, pack_geometry)

        blob = jnp.asarray(pack_geometry(geom))
        q, _err, found, _clr = analytic_ik_cuda(
            blob, T_solve, samples, previous_cfg,
            respect_limits=respect_limits,
            err_tol=default_err_tol() if err_tol is None else err_tol)
    else:
        def one(Ti, prev):
            qq, ee, ok = solve_all(geom, Ti, samples, respect_limits)
            q_flat = qq.reshape(-1, 7)
            e_flat = ee.reshape(-1)
            ok_flat = ok.reshape(-1)
            if previous_cfg is None:
                score = jnp.where(ok_flat, e_flat, e_flat + 1e6)
            else:
                d = jnp.linalg.norm(q_flat - prev[None, :7], axis=-1)
                score = jnp.where(ok_flat, d, d + 1e6)
            i = jnp.argmin(score)
            return q_flat[i], jnp.any(ok_flat)

        prev = (jnp.zeros((batch, 7)) if previous_cfg is None
                else jnp.asarray(previous_cfg).reshape(batch, -1))
        q, found = jax.vmap(one)(T_solve, prev)

    if differentiable:
        q = _attach_implicit_gradient(q, robot, target_link_name, T)

    if single:
        return q[0], found[0]
    return q, found


def _attach_implicit_gradient(q, robot: "Robot", target_link_name: str, T):
    """Make an analytic solution differentiable w.r.t. the target poses.

    Reuses the suite's shared implicit-diff wrapper. Differentiating *through*
    the decomposition would be wrong as well as awkward: branch selection is
    piecewise constant, so its derivative is zero almost everywhere and
    undefined at branch switches. The implicit function theorem instead
    differentiates the solution *manifold* — ``FK(q*) = T`` gives
    ``dq* = J(q*)^+ dT`` — which is the correct derivative within a branch and
    the one callers actually want.
    """
    import jaxlie

    from ..optimization_engines._implicit_diff import differentiable_ik_solution

    link_idx = list(robot.links.names).index(target_link_name)
    n_act = len(robot.joints.actuated_names)

    def one(qi, Ti):
        q_full = jnp.zeros((n_act,), dtype=qi.dtype).at[:7].set(qi)
        q_full = differentiable_ik_solution(
            q_full, robot, link_idx, jaxlie.SE3.from_matrix(Ti))
        return q_full[:7]

    return jax.vmap(one)(q, T)


# --------------------------------------------------------------------------- #
# Collision-free selection
# --------------------------------------------------------------------------- #
# The enumeration trick: the solver already emits `n_q7 * 8` candidates per
# target and typically ~18 of them are geometrically valid. Collision-freeness
# is therefore a *selection* over an existing set, not a re-solve — there is no
# reseeding and no retry loop, so the cost stays bounded and deterministic.
#
# Self-collision is checked by default rather than as an option, because it is
# specific to this solver: the 8 branches per q7 are exactly the elbow-up/down,
# wrist-flip and shoulder-pair alternatives, which are the configurations that
# fold the arm into itself. A seeded numerical solver rarely lands on one; a
# branch enumerator hits them routinely.

#: Signed distance (m) below which a candidate counts as in collision. Positive
#: means separated, so a small positive margin buys clearance rather than
#: touching.
COLLISION_MARGIN = 0.005


def candidate_clearance(robot, robot_coll, q_candidates, world_geom=None,
                        check_self=True):
    """Minimum signed clearance for each candidate configuration.

    Args:
        q_candidates: ``[K, 7]`` joint configurations.
        world_geom: a ``CollGeom`` of obstacles, or ``None`` to skip world checks.
        check_self: include self-collision pairs.

    Returns:
        ``[K]`` clearance; negative means penetrating.

    The collision model must be built with an SRDF
    (``RobotCollisionSpherized.from_urdf(urdf, srdf_path=...)``). Without one,
    the spherized model's conservative enclosure leaves adjacent links
    overlapping by construction — self-clearance is about -0.03 m even at the
    neutral pose — so every candidate reports as colliding and this function
    silently returns "nothing is reachable". That is a wrong answer rather than
    a crash, which is why it is called out here.

    Passes the whole candidate block to the collision model as one batched
    ``cfg`` (a single batched FK), rather than mapping single configurations.
    """
    n_act = len(robot.joints.actuated_names)
    q_full = jax.vmap(
        lambda qq: jnp.zeros((n_act,), dtype=qq.dtype).at[:7].set(qq[:7])
    )(q_candidates)

    clearances = []
    if world_geom is not None:
        clearances.append(jnp.min(
            robot_coll.compute_world_collision_distance(robot, q_full, world_geom),
            axis=(-2, -1)))
    if check_self:
        clearances.append(jnp.min(
            robot_coll.compute_self_collision_distance(robot, q_full), axis=-1))

    if not clearances:
        return jnp.full((q_candidates.shape[0],), jnp.inf)
    return jnp.min(jnp.stack(clearances, axis=0), axis=0)


def analytic_ik_solve_collision_free(
    robot: "Robot",
    target_link_name: str,
    target_pose,
    robot_coll,
    world_geom=None,
    *,
    q7_samples=None,
    num_q7: int = 32,
    previous_cfg=None,
    respect_limits: bool = True,
    geometry: ArmGeometry | None = None,
    margin: float = COLLISION_MARGIN,
    check_self: bool = True,
):
    """Analytic IK returning a collision-free branch when one exists.

    Ranking is lexicographic — collision dominates, then the usual criterion
    (pose error, or joint distance to ``previous_cfg``). That ordering matters
    for trajectories: you want the closest *collision-free* branch, not a
    compromise between closeness and safety.

    Degrades gracefully. If no branch is collision-free the maximum-clearance
    one is returned with ``collision_free=False``, so a caller can tell
    "unreachable" from "reachable but blocked" — different facts, which a task
    planner should act on differently.

    Returns:
        ``(q[7], found, collision_free, clearance)``.
    """
    geom = geometry if geometry is not None else build_geometry(robot, target_link_name)

    import jaxlie

    T = (target_pose.as_matrix() if isinstance(target_pose, jaxlie.SE3)
         else jnp.asarray(target_pose))
    samples = (default_q7_samples(geom, num_q7) if q7_samples is None
               else jnp.atleast_1d(jnp.asarray(q7_samples)))

    q, err, ok = solve_all(geom, jax.lax.stop_gradient(T), samples, respect_limits)
    q_flat = q.reshape(-1, 7)
    err_flat = err.reshape(-1)
    ok_flat = ok.reshape(-1)

    clearance = candidate_clearance(robot, robot_coll, q_flat, world_geom, check_self)
    collision_free = clearance > margin

    if previous_cfg is None:
        base = err_flat
    else:
        prev = jnp.asarray(previous_cfg)[:7]
        base = jnp.linalg.norm(q_flat - prev[None, :], axis=-1)

    # Lexicographic: pose-invalid worst, then colliding, then the base criterion.
    score = base + jnp.where(ok_flat, 0.0, 1e6) + jnp.where(collision_free, 0.0, 1e3)

    best = jnp.argmin(score)
    any_valid = jnp.any(ok_flat)
    any_free = jnp.any(ok_flat & collision_free)

    # No collision-free branch: fall back to the roomiest valid one so the
    # caller gets the least-bad configuration rather than an arbitrary colliding
    # one.
    fallback = jnp.argmin(jnp.where(ok_flat, -clearance, jnp.inf))
    pick = jnp.where(any_free, best, jnp.where(any_valid, fallback, best))

    return q_flat[pick], any_valid, any_free, clearance[pick]
