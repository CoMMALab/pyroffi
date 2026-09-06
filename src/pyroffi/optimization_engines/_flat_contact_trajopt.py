"""Differential-flatness **contact-aware** (fixed-grasp) trajectory optimization.

.. note:: **Contact-aware, not contact-rich.** This solver assumes a *single,
   persistent contact mode* — the grasp captured at plan time is held rigid for
   the entire horizon. Contacts never make or break, the active set is fixed,
   and the contact forces are *allocated analytically* (not decided) from the
   object's motion via the grasp-map pseudo-inverse. It reasons *about* contact
   but never *decides* anything about it, which is exactly what makes the system
   differentially flat. A genuinely **contact-rich** solver, where the contact
   forces are first-class decision variables optimized subject to the object's
   Newton-Euler balance (and where flatness therefore does *not* hold), lives in
   :func:`~pyroffi.optimization_engines.contact_rich_trajopt`. See
   ``flat_contact_trajopt_theory.md`` for the full distinction.

A faster, better-conditioned reformulation of :mod:`_contact_trajopt`. The
insight is that a *rigidly-grasped* object is **differentially flat in the
object's SE(3) pose**: once the object-pose trajectory is chosen, everything
else that the augmented-Lagrangian solver used to fight over is *structurally
determined* rather than penalized:

* **Grasp closure** — every gripper pose is ``xi_t @ offset_i`` for a constant
  captured offset, so all grippers move as one rigid body *by construction*.
  The expensive relative-pose closure constraint (and its ``mu`` duals) is gone;
  in its place each arm's config merely *tracks* its object-derived gripper pose
  (a well-conditioned absolute-pose cost, not a coupled equality constraint).
* **Object Newton-Euler** — the contact forces are *allocated analytically*
  from the object's acceleration via the grasp-map pseudo-inverse, so the
  object-dynamics residual is **zero for every candidate trajectory**. The
  contact forces stop being decision variables and the ``nu`` duals disappear.

What remains is a single, mostly-unconstrained smooth problem solved with **one
L-BFGS pass** (with an optional light penalty-continuation on the pose-tracking
weight) — no nested dual-ascent outer loop. The decision vector is

    z = [ delta_obj (T x 6) | q (T x ndof) | squeeze (T) | time_scale (1) ]

where ``delta_obj_t`` is the object-pose twist relative to the grasp pose
(``xi_t = exp(delta_t) @ T_obj0``), ``q`` are the stacked joint configs (kept
so torque / joint-limit / (opt-in) collision costs can see them), and
``squeeze_t`` is a scalar internal grip force added *in the null space of the
grasp map* so it tightens the grip without disturbing the object dynamics, and
``time_scale`` sets a shared timestep ``dt`` (so the trajectory *duration* is a
decision variable, enabling a minimum-time objective).

The objective is a tunable blend — ``w_time`` (minimum time), ``w_smoothness``
(accel/jerk), and ``w_effort`` (torque) — defaulting to pure min-time.

This slashes the variable count (~4x fewer than ``[q | lambda]`` with duals),
removes the outer loop, and turns two hard equality constraints into
structure — addressing both the tractability and the speed of the plain
augmented-Lagrangian formulation.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jaxlie
from jax import Array
from jaxtyping import Float

from ..dynamics import _contact as C
from ..dynamics._contact import ContactSystem
from ._contact_trajopt import _fd_vel_acc
from ._sco_optimization import (
    _collision_dists_reduced,
    _limits_cost,
    _smoothness_cost,
)
from ._trajopt_core import _adaptive_trust_step, _lbfgs_driver, _make_trust


def _collision_cost(
    q: Float[Array, "T n"],
    system: ContactSystem,
    colls: tuple,
    world_geoms: tuple,
    cfg: "FlatContactTrajOptConfig",
) -> Array:
    """Hinge on the softmin clearance of every waypoint, per manipulator.

    ``colls[i]`` is the collision model for ``system.manipulators[i]``, or
    ``None`` to skip that arm. To sweep the *carried object* as well, pass
    ``model.with_attachments(aset)``: the attachment hangs off the grip link in
    that link's frame, so it rides the same FK the grasp-tracking cost uses and
    there is no second place for the object's pose to come from.

    ``_collision_dists_reduced`` returns one softmin scalar per group, self
    first and then one per world geometry; the two groups are weighted
    separately (see ``w_self_collision``).
    """
    total = jnp.zeros((), q.dtype)
    idx = 0
    for m, coll in zip(system.manipulators, colls):
        o, idx = idx, idx + m.num_dof
        if coll is None:
            continue
        q_i = q[:, o : o + m.num_dof]

        def groups(c, coll=coll, m=m):
            return _collision_dists_reduced(
                c, m.robot, coll, world_geoms, cfg.collision_temperature
            )

        d = jax.vmap(groups)(q_i)  # (T, 1 + n_world)
        hinge = jnp.maximum(0.0, cfg.collision_margin - d) ** 2
        total += cfg.w_self_collision * jnp.sum(hinge[:, 0])
        total += cfg.w_collision * jnp.sum(hinge[:, 1:])
    return total


def _collision_cost_lin(
    q: Float[Array, "T n"],
    q_k: Float[Array, "T n"],
    system: ContactSystem,
    colls: tuple,
    world_geoms: tuple,
    cfg: "FlatContactTrajOptConfig",
) -> Array:
    """Schulman-SCO linearization of :func:`_collision_cost`.

    Identical hinge, but the (non-convex) per-waypoint reduced clearance ``d(q)``
    is replaced by its first-order model ``d(q_k) + J(q_k)(q - q_k)`` about the
    frozen outer/stage iterate ``q_k`` — so the inner L-BFGS subproblem sees a
    convex hinge of an affine function, the defining move of SCO. Used only when
    ``cfg.use_sco`` is set (and collision is active); the exact nonlinear term
    above is the default."""
    total = jnp.zeros((), q.dtype)
    idx = 0
    for m, coll in zip(system.manipulators, colls):
        o, idx = idx, idx + m.num_dof
        if coll is None:
            continue
        q_i = q[:, o : o + m.num_dof]
        qk_i = q_k[:, o : o + m.num_dof]

        def groups_all(qq, coll=coll, m=m):
            return jax.vmap(lambda c: _collision_dists_reduced(
                c, m.robot, coll, world_geoms, cfg.collision_temperature
            ))(qq)

        d0, jvp = jax.linearize(groups_all, qk_i)
        d_lin = d0 + jvp(q_i - qk_i)  # (T, 1 + n_world), affine in q_i
        hinge = jnp.maximum(0.0, cfg.collision_margin - d_lin) ** 2
        total += cfg.w_self_collision * jnp.sum(hinge[:, 0])
        total += cfg.w_collision * jnp.sum(hinge[:, 1:])
    return total


@dataclass(frozen=True)
class FlatContactTrajOptConfig:
    """Hyper-parameters for the differential-flatness contact solver."""

    # --- Loop structure (note: NO dual-ascent outer loop) ---
    n_stages: int = 4
    """Penalty-continuation stages; each scales the pose-tracking weight up."""
    n_inner_iters: int = 40
    m_lbfgs: int = 8

    # --- Time parameterization (the trajectory *duration* is a decision var) ---
    dt: float = 0.1
    """Nominal / initial timestep. The per-step ``dt`` is optimized within
    ``[min_dt, max_dt]`` (a shared scalar), so the total horizon is
    ``(T - 1) * dt`` — this is what the min-time objective drives down."""
    min_dt: float = 0.01
    max_dt: float = 0.5

    # --- High-level objective weights (what the solve trades off) ------------
    #   These select *what* to optimize; the per-term weights below only shape
    #   the relative contribution *within* each group. Feasibility penalties
    #   (grasp tracking, joint/torque limits, grip validity) are always on and
    #   are NOT gated by these — they enforce constraints, not preferences.
    w_time: float = 1.0
    """Minimum-time: penalizes the total horizon ``(T - 1) * dt``."""
    w_smoothness: float = 0.0
    """Scales the object + joint accel/jerk smoothness group."""
    w_effort: float = 0.0
    """Scales the total torque effort ``sum(tau**2)``."""

    # --- Object flat-output smoothness (relative shape within the group) ---
    w_obj_smooth: float = 1.0
    w_obj_acc: float = 1.0
    w_obj_jerk: float = 0.2

    # --- Grasp tracking (replaces the AL grasp-closure constraint) ---
    w_track: float = 50.0
    """Initial weight pulling each arm's gripper onto its object-derived pose."""
    track_scale: float = 3.0
    """Per-stage multiplier on w_track (cheap penalty continuation)."""
    w_track_max: float = 1e4

    # --- Early termination -------------------------------------------------
    #   The inner solve runs as a `lax.while_loop` capped at `n_inner_iters`;
    #   enabling this stops a converged solve instead of burning the remaining
    #   iterations. OFF by default -- see the matching note in
    #   :mod:`_contact_rich_trajopt`: on these objectives the gradient norm
    #   never approaches the tolerance, so the check never fires.
    grad_tol: float = 0.0
    """Inner L-BFGS stops once ``max|grad|`` drops below this. 0 disables."""

    use_sco: bool = False
    """Schulman-SCO the (opt-in) collision term: linearize the per-waypoint
    clearance about each stage's incoming iterate so the inner subproblem sees a
    convex affine-hinge instead of the raw non-convex distance. No effect unless
    a collision weight is also nonzero; the default (False) leaves the exact
    nonlinear term and a byte-identical graph."""

    # --- Adaptive (Schulman) trust region ---------------------------------
    #     Adds a resized trust region ``coef ||z - z_k||²`` around each stage,
    #     accepting/rejecting the step by the actual-vs-predicted ratio test.
    #     Meaningful together with ``use_sco`` (the ratio judges the linearized
    #     collision model). Default off ⇒ the exact stage loop, byte-identical.
    adaptive_trust: bool = False
    tr_coef0: float = 1.0
    tr_tighten: float = 4.0
    tr_loosen: float = 0.25
    tr_shrink_ratio: float = 0.25
    tr_expand_ratio: float = 0.75
    tr_accept_ratio: float = 0.1
    tr_coef_min: float = 1e-2
    tr_coef_max: float = 1e4

    # --- Joint-space regularity ---
    w_q_smooth: float = 0.2
    w_q_acc: float = 0.5
    w_q_jerk: float = 0.1
    w_limits: float = 1.0

    # --- Dynamics / effort (forces are ALLOCATED, not optimized) ---
    w_torque_limit: float = 1.0
    tau_max: float = 87.0

    # --- Collision (OPT-IN: both weights default to 0, which traces the term
    #     out entirely, so a caller that does not ask for collision gets the
    #     same graph and the same numbers as before this term existed) --------
    w_collision: float = 0.0
    """Weight on the world-collision hinge. 0 disables the term (the default);
    a nonzero value requires ``colls``/``world_geoms`` at the call site."""
    w_self_collision: float = 0.0
    """Weight on the self-collision hinge, kept separate from ``w_collision``
    and off by default *on purpose*. A spherized model's base spheres can sit
    permanently inside each other (the Panda's baseline is about -0.03 m at
    every configuration), so folding self-collision in with the world would put
    a constant, irreducible force on ``q`` that has nothing to do with the
    obstacles. Turn this on only for a model whose baseline is actually
    positive."""
    collision_margin: float = 0.02
    """Clearance the hinge asks for, in metres."""
    collision_temperature: float = 0.05
    """Softmin temperature reducing each group's pair distances to one scalar.
    A hard ``min`` would be non-smooth, and this solver is L-BFGS: the kinks
    corrupt the inverse-Hessian estimate rather than merely slowing a step."""

    # --- Grip validity + squeeze ---
    w_grip: float = 1.0
    mu_friction: float | None = None
    f_min: float = 2.0
    w_squeeze_reg: float = 1e-4
    """Regularizer on the internal squeeze force (keeps it minimal)."""


# ---------------------------------------------------------------------------
# Object frame + flat kinematics
# ---------------------------------------------------------------------------

def object_frame_offsets(system: ContactSystem) -> tuple[jaxlie.SE3, ...]:
    """Constant object-frame -> gripper[i] transforms.

    The object frame is taken to be the *reference* gripper frame at grasp
    time, so ``offset_0`` is the identity and ``offset_i`` is exactly the
    captured reference->gripper[i] offset already stored on the system.
    """
    k = system.num_manipulators
    ident = jaxlie.SE3.identity()
    return (ident,) + tuple(system.grasp_offsets)  # length == k, checked below or trust


def _dt_from_scale(scale: Array, cfg: FlatContactTrajOptConfig) -> Array:
    """Smoothly map an unconstrained scalar to a timestep in ``[min_dt, max_dt]``.

    Using a sigmoid keeps the min-time optimization *unconstrained* (no clamp /
    projection needed) while structurally respecting the duration bounds.
    """
    return cfg.min_dt + (cfg.max_dt - cfg.min_dt) * jax.nn.sigmoid(scale)


def _scale_from_dt(dt: float, cfg: FlatContactTrajOptConfig) -> float:
    """Inverse of :func:`_dt_from_scale` (used to initialise ``dt = cfg.dt``)."""
    import math

    frac = (dt - cfg.min_dt) / (cfg.max_dt - cfg.min_dt)
    frac = min(max(frac, 1e-4), 1.0 - 1e-4)
    return math.log(frac / (1.0 - frac))


def _object_pose(delta: Float[Array, "6"], T_obj0: jaxlie.SE3) -> jaxlie.SE3:
    """Object SE(3) pose from a twist relative to the grasp pose."""
    return jaxlie.SE3.exp(delta) @ T_obj0


def _gripper_targets(
    xi: jaxlie.SE3, offsets: tuple[jaxlie.SE3, ...]
) -> tuple[jaxlie.SE3, ...]:
    return tuple(xi @ off for off in offsets)


# ---------------------------------------------------------------------------
# Analytic contact-force allocation (grasp-map pseudo-inverse + null squeeze)
# ---------------------------------------------------------------------------

def _skew(v: Array) -> Array:
    return jnp.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], v.dtype
    )


def allocate_forces(
    system: ContactSystem,
    q: Float[Array, "n"],
    c: Float[Array, "3"],
    R_obj: Array,
    a_lin: Float[Array, "3"],
    alpha: Float[Array, "3"],
    omega: Float[Array, "3"],
    squeeze: Array,
) -> Float[Array, "k 3"]:
    """Contact forces that satisfy the object's Newton-Euler balance exactly.

    Solves the (6 x 3k) grasp map ``G lam = w_req`` for the minimum-norm
    particular solution, then adds an internal ``squeeze`` force projected onto
    ``null(G)`` so the grip can be tightened without changing ``w_req``. The
    result therefore *always* satisfies object dynamics — there is no residual
    left to constrain.

    Runs its own FK for the contact points; callers holding them (via
    :func:`~pyroffi.dynamics._contact.grasp_kinematics`) should call
    :func:`allocate_forces_at`.
    """
    return allocate_forces_at(
        system,
        jnp.stack(C.contact_points_world(system, q)),
        c,
        R_obj,
        a_lin,
        alpha,
        omega,
        squeeze,
    )


def allocate_forces_at(
    system: ContactSystem,
    P: Float[Array, "k 3"],
    c: Float[Array, "3"],
    R_obj: Array,
    a_lin: Float[Array, "3"],
    alpha: Float[Array, "3"],
    omega: Float[Array, "3"],
    squeeze: Array,
) -> Float[Array, "k 3"]:
    """:func:`allocate_forces` from precomputed world contact points."""
    k = system.num_manipulators
    dtype = P.dtype

    # Required net wrench about the object centre c.
    g_vec = jnp.array([0.0, 0.0, -system.gravity], dtype)
    inertia_diag = jnp.asarray(system.body.inertia_diag, dtype)
    I_world = R_obj @ (inertia_diag[:, None] * R_obj.T)  # (3,3)
    f_req = system.body.mass * (a_lin - g_vec)
    tau_req = I_world @ alpha + jnp.cross(omega, I_world @ omega)
    w_req = jnp.concatenate([f_req, tau_req])  # (6,)

    # Grasp map G (6 x 3k): [ I3 ... ; skew(p_i - c) ... ].
    top = jnp.tile(jnp.eye(3, dtype=dtype), (1, k))  # (3, 3k)
    bot = jnp.concatenate([_skew(P[i] - c) for i in range(k)], axis=1)  # (3, 3k)
    G = jnp.concatenate([top, bot], axis=0)  # (6, 3k)

    GGt = G @ G.T + 1e-6 * jnp.eye(6, dtype=dtype)
    Gpinv = G.T @ jnp.linalg.inv(GGt)  # (3k, 6), min-norm right-inverse
    lam_part = Gpinv @ w_req  # (3k,)

    # Internal squeeze: push each contact toward the object centre, then remove
    # any component that would disturb w_req (project onto null(G)).
    #
    # The direction has to be built with a *where-guarded* norm, not
    # ``norm(d) + eps``. With one manipulator the object centre is the sole
    # contact point, so ``d`` is exactly zero: ``norm(d) + eps`` still returns a
    # finite 0/eps value, but ``norm`` is non-differentiable at the origin and
    # its gradient is NaN, which propagates into every ``q`` entry of the
    # objective's gradient and silently turns the whole solve into a no-op
    # (``best_cost`` can never be beaten by a NaN). A zero normal is also the
    # physically right answer there: a single contact at the object centre has
    # no internal squeeze direction to speak of.
    normals = jnp.stack([C._safe_unit(c - P[i]) for i in range(k)])  # (k, 3)
    raw = (squeeze * normals).reshape(-1)  # (3k,)
    proj = raw - Gpinv @ (G @ raw)  # null-space component
    lam = (lam_part + proj).reshape(k, 3)
    return lam


def _dof_offsets(system: ContactSystem) -> list[int]:
    offs, idx = [], 0
    for m in system.manipulators:
        offs.append(idx)
        idx += m.num_dof
    return offs


def _shares_one_grid(system: ContactSystem) -> bool:
    """True iff every manipulator is driven by the same GRiD model.

    The common bimanual case is two identical arms, where the per-arm inverse
    dynamics can be stacked into one batched kernel launch instead of ``k``.
    """
    g0 = system.manipulators[0].grid
    return all(m.grid is g0 for m in system.manipulators)


def _torque_cost(
    system: ContactSystem,
    q: Float[Array, "T n"],
    lam: Float[Array, "T k 3"],
    points: Float[Array, "T k 3"],
    dt: Array,
    cfg,
) -> Array:
    """Effort + torque-limit cost from GRiD inverse dynamics.

    ``cfg`` is any config exposing ``w_effort``, ``w_torque_limit`` and
    ``tau_max`` — shared verbatim by the flat and contact-rich solvers.

    When every manipulator shares one GRiD model the ``k`` inverse-dynamics
    calls are stacked into a single launch over a ``(k, T)`` leading batch —
    which also fuses the analytic-gradient and CRBA kernels behind the custom
    JVP. Heterogeneous systems fall back to the per-manipulator loop.
    """
    offs = _dof_offsets(system)
    manips = system.manipulators

    # Per-arm reaction wrenches (each arm has its own base transform and contact
    # point, so this part stays per-manipulator either way).
    fexts = [
        jax.vmap(C.manipulator_contact_fext_at, in_axes=(None, 0, 0, 0))(
            m, q[:, o : o + m.num_dof], points[:, i, :], lam[:, i, :]
        )
        for i, (m, o) in enumerate(zip(manips, offs))
    ]

    def _cost_from_tau(tau: Array) -> Array:
        over = jnp.maximum(0.0, jnp.abs(tau) - cfg.tau_max) ** 2
        return cfg.w_effort * jnp.sum(tau**2) + cfg.w_torque_limit * jnp.sum(over)

    if _shares_one_grid(system) and len({m.num_dof for m in manips}) == 1:
        q_all = jnp.stack(
            [q[:, o : o + m.num_dof] for m, o in zip(manips, offs)]
        )  # (k, T, ndof_arm)
        qd_all, qdd_all = jax.vmap(_fd_vel_acc, in_axes=(0, None))(q_all, dt)
        tau_all = manips[0].grid.inverse_dynamics(
            q_all, qd_all, qdd_all, f_ext=jnp.stack(fexts)
        )
        return _cost_from_tau(tau_all)

    cost = jnp.array(0.0, q.dtype)
    for i, (m, o) in enumerate(zip(manips, offs)):
        q_i = q[:, o : o + m.num_dof]
        qd_i, qdd_i = _fd_vel_acc(q_i, dt)
        cost += _cost_from_tau(
            m.grid.inverse_dynamics(q_i, qd_i, qdd_i, f_ext=fexts[i])
        )
    return cost


# ---------------------------------------------------------------------------
# Flat objective
# ---------------------------------------------------------------------------

def _angular_rates(R_seq_log: Float[Array, "T 3"], dt: float) -> tuple[Array, Array]:
    """omega, alpha from a sequence of world rotation vectors (small-motion FD)."""
    omega, alpha = _fd_vel_acc(R_seq_log, dt)
    return omega, alpha


def _flat_cost(
    z: Float[Array, "nz"],
    system: ContactSystem,
    T_obj0: jaxlie.SE3,
    offsets: tuple[jaxlie.SE3, ...],
    lower: Float[Array, "n"],
    upper: Float[Array, "n"],
    w_track: Array,
    T: int,
    cfg: FlatContactTrajOptConfig,
    colls: tuple = (),
    world_geoms: tuple = (),
    z_k: Float[Array, "nz"] | None = None,
) -> Array:
    ndof = system.num_dof
    k = system.num_manipulators
    n_delta = T * 6
    n_q = T * ndof

    delta = z[:n_delta].reshape(T, 6)
    q = z[n_delta : n_delta + n_q].reshape(T, ndof)
    squeeze = z[n_delta + n_q : n_delta + n_q + T]  # (T,)
    dt = _dt_from_scale(z[-1], cfg)  # shared, optimized timestep

    # --- Minimum-time: penalize the total horizon (T - 1) * dt -----------
    cost = cfg.w_time * (T - 1) * dt

    # --- Object pose trajectory (flat output) ----------------------------
    xis = jax.vmap(lambda d: _object_pose(d, T_obj0))(delta)  # SE3 batched
    obj_centers = xis.translation()  # (T, 3) flat-output translation
    phis = jax.vmap(lambda R: R.log())(xis.rotation())  # (T,3) world rotvec

    # --- Object flat-output smoothness (cheap: 6-D per waypoint) ---------
    cost += cfg.w_smoothness * cfg.w_obj_smooth * _smoothness_cost(
        obj_centers, 0.0, cfg.w_obj_acc, cfg.w_obj_jerk
    )
    cost += cfg.w_smoothness * cfg.w_obj_smooth * _smoothness_cost(
        phis, 0.0, cfg.w_obj_acc, cfg.w_obj_jerk
    )

    # --- Grasp kinematics: ONE forward-kinematics pass per manipulator -----
    #   The tracking residual, the object centre, the force allocation, the grip
    #   penalty and the reaction wrench are all projections of the same gripper
    #   poses. Deriving them together removes five of the six FK sweeps this cost
    #   used to run per timestep.
    fk_params, points, centers, _, _ = jax.vmap(
        C.grasp_kinematics_with_poses, in_axes=(None, 0)
    )(system, q)  # (T,k,7), (T,k,3), (T,3)

    # --- Grasp tracking: each arm follows its object-derived gripper pose --
    def track_res(delta_t, fk_params_t):
        xi = _object_pose(delta_t, T_obj0)
        targets = _gripper_targets(xi, offsets)
        errs = [
            (jaxlie.SE3(fk_params_t[i]).inverse() @ tgt).log()
            for i, tgt in enumerate(targets)
        ]
        return jnp.concatenate(errs)  # (6k,)

    track = jax.vmap(track_res)(delta, fk_params)  # (T, 6k)
    cost += w_track * jnp.sum(track**2)

    # --- Joint-space regularity ------------------------------------------
    cost += cfg.w_smoothness * cfg.w_q_smooth * _smoothness_cost(
        q, 0.0, cfg.w_q_acc, cfg.w_q_jerk
    )
    cost += cfg.w_limits * _limits_cost(q, lower, upper)

    # --- Collision (opt-in) ----------------------------------------------
    # Guarded on the *static* config, so with the default weights the term is
    # not traced at all and the graph is byte-for-byte what it was before.
    if cfg.w_collision or cfg.w_self_collision:
        if cfg.use_sco and z_k is not None:
            q_k = z_k[n_delta : n_delta + n_q].reshape(T, ndof)
            cost += _collision_cost_lin(q, q_k, system, colls, world_geoms, cfg)
        else:
            cost += _collision_cost(q, system, colls, world_geoms, cfg)

    # --- Allocate forces (exact object dynamics) -------------------------
    # Balance about the *actual* object centre (contact-point centroid), so the
    # allocated forces satisfy the same Newton-Euler residual a caller measures.
    _, a_lin = _fd_vel_acc(centers, dt)
    omega, alpha = _angular_rates(phis, dt)
    R_mats = jax.vmap(lambda R: R.as_matrix())(xis.rotation())  # (T,3,3)
    lam = jax.vmap(allocate_forces_at, in_axes=(None, 0, 0, 0, 0, 0, 0, 0))(
        system, points, centers, R_mats, a_lin, alpha, omega, squeeze
    )  # (T, k, 3)

    # --- Torque effort + limits via GRiD inverse dynamics ----------------
    cost += _torque_cost(system, q, lam, points, dt, cfg)

    # --- Grip validity + squeeze regularization --------------------------
    grip = jax.vmap(
        C.grip_validity_penalty_at, in_axes=(None, 0, 0, 0, None, None)
    )(system, centers, points, lam, cfg.mu_friction, cfg.f_min)
    cost += cfg.w_grip * jnp.sum(grip)
    cost += cfg.w_squeeze_reg * jnp.sum(squeeze**2)

    return cost


# ---------------------------------------------------------------------------
# Inner L-BFGS solve (shared scaffold; single unconstrained pass)
# ---------------------------------------------------------------------------

def _inner_solve(z0, endpoint_mask, cost_fn, cfg: FlatContactTrajOptConfig):
    """One inner L-BFGS solve, delegating to the shared driver.

    ``while_loop`` form (capped at ``n_inner_iters``, ``grad_tol``-gated early
    exit), best-by-cost, endpoint-masked — the same behavior this engine's own
    copy had. Safe because ``flat_contact_trajopt`` is only ever called under
    ``stop_gradient`` (the IOC gradient comes from the analytic adjoint, not from
    differentiating this loop).
    """
    return _lbfgs_driver(
        z0, cost_fn,
        n_iters=cfg.n_inner_iters, m_lbfgs=cfg.m_lbfgs,
        grad_tol=cfg.grad_tol, loop="while",
        endpoint_mask=endpoint_mask, best_by="cost", gd_dir="norm",
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=("system", "opt_cfg"))
def _flat_contact_jax(
    init_q: Float[Array, "T n"],
    delta_goal: Float[Array, "6"],
    init_squeeze: Float[Array, "T"],
    start: Float[Array, "n"],
    goal: Float[Array, "n"],
    system: ContactSystem,
    opt_cfg: FlatContactTrajOptConfig,
    colls: tuple = (),
    world_geoms: tuple = (),
) -> tuple[Array, Array, Array, Array]:
    T = init_q.shape[0]
    ndof = system.num_dof
    k = system.num_manipulators
    n_delta = T * 6
    n_q = T * ndof
    nz = n_delta + n_q + T + 1  # + shared time-scale scalar

    lower = jnp.concatenate([m.robot.joints.lower_limits for m in system.manipulators])
    upper = jnp.concatenate([m.robot.joints.upper_limits for m in system.manipulators])

    # Object frame = reference gripper frame at the start config.
    ref = system.manipulators[0]
    T_obj0 = C._gripper_world_pose(ref, system.split_q(start)[0])
    offsets = object_frame_offsets(system)

    # Initialise the object-pose deltas by linearly ramping to the goal twist.
    t = jnp.linspace(0.0, 1.0, T)[:, None]
    delta0 = t * delta_goal[None, :]  # (T,6)

    q = init_q.at[0].set(start).at[-1].set(goal)
    scale0 = jnp.array([_scale_from_dt(opt_cfg.dt, opt_cfg)], jnp.float32)
    z = jnp.concatenate([delta0.reshape(-1), q.reshape(-1), init_squeeze, scale0])

    # Pin: object-pose delta at start (=0) and goal; joint configs at start/goal.
    # The trailing time-scale scalar stays free (optimized).
    mask = jnp.ones(nz)
    mask = mask.at[:6].set(0.0).at[n_delta - 6 : n_delta].set(0.0)
    mask = mask.at[n_delta : n_delta + ndof].set(0.0)
    mask = mask.at[n_delta + n_q - ndof : n_delta + n_q].set(0.0)

    trust = _make_trust(opt_cfg)

    def _flat_merit(zz, z_k, w_track, linearize):
        """Stage cost for the trust ratio test: collision linearized about
        ``z_k`` (the model) or exact (the truth); no trust term."""
        return _flat_cost(
            zz, system, T_obj0, offsets, lower, upper, w_track, T, opt_cfg,
            colls, world_geoms, z_k if linearize else None,
        )

    if trust is None:
        def stage(carry, _):
            z, w_track = carry
            # Freeze the stage iterate for the (opt-in) SCO collision
            # linearization; ignored by the exact-collision default.
            z_k = jax.lax.stop_gradient(z)
            cost_fn = lambda zz: _flat_cost(
                zz, system, T_obj0, offsets, lower, upper, w_track, T, opt_cfg,
                colls, world_geoms, z_k,
            )
            z = _inner_solve(z, mask, cost_fn, opt_cfg)
            w_track = jnp.minimum(w_track * opt_cfg.track_scale, opt_cfg.w_track_max)
            return (z, w_track), None

        (z, _), _ = jax.lax.scan(
            stage, (z, jnp.array(opt_cfg.w_track, jnp.float32)),
            None, length=opt_cfg.n_stages,
        )
    else:
        def stage(carry, _):
            z, w_track, tr_coef = carry
            z_k = jax.lax.stop_gradient(z)
            cost_fn = lambda zz: _flat_cost(
                zz, system, T_obj0, offsets, lower, upper, w_track, T, opt_cfg,
                colls, world_geoms, z_k,
            ) + tr_coef * jnp.sum((zz - z_k) ** 2)
            z_trial = _inner_solve(z, mask, cost_fn, opt_cfg)
            # Schulman ratio test on the linearized-collision model.
            m_zk = _flat_merit(z_k, z_k, w_track, linearize=True)
            m_model = _flat_merit(z_trial, z_k, w_track, linearize=True)
            m_true = _flat_merit(z_trial, z_k, w_track, linearize=False)
            z, tr_coef, _ = _adaptive_trust_step(
                z_k, z_trial, m_zk, m_model, m_true, tr_coef, trust
            )
            w_track = jnp.minimum(w_track * opt_cfg.track_scale, opt_cfg.w_track_max)
            return (z, w_track, tr_coef), None

        (z, _, _), _ = jax.lax.scan(
            stage,
            (z, jnp.array(opt_cfg.w_track, jnp.float32),
             jnp.array(trust.coef0, jnp.float32)),
            None, length=opt_cfg.n_stages,
        )

    delta = z[:n_delta].reshape(T, 6)
    q = z[n_delta : n_delta + n_q].reshape(T, ndof)
    squeeze = z[n_delta + n_q : n_delta + n_q + T]
    dt = _dt_from_scale(z[-1], opt_cfg)  # optimized timestep

    # Recover the allocated forces for the returned trajectory.
    xis = jax.vmap(lambda d: _object_pose(d, T_obj0))(delta)
    phis = jax.vmap(lambda R: R.log())(xis.rotation())
    centers = jax.vmap(C.object_center_world, in_axes=(None, 0))(system, q)
    _, a_lin = _fd_vel_acc(centers, dt)
    omega, alpha = _angular_rates(phis, dt)
    R_mats = jax.vmap(lambda R: R.as_matrix())(xis.rotation())
    forces = jax.vmap(allocate_forces, in_axes=(None, 0, 0, 0, 0, 0, 0, 0))(
        system, q, centers, R_mats, a_lin, alpha, omega, squeeze
    )

    # Diagnostics: grasp-closure residual (should be tiny — tracked, not free).
    g = jax.vmap(C.grasp_closure_residual, in_axes=(None, 0))(system, q)
    if g.shape[-1] == 0:
        residuals = jnp.zeros((2,), g.dtype)
    else:
        residuals = jnp.array([jnp.sqrt(jnp.mean(g**2)), jnp.max(jnp.abs(g))])
    return q, forces, residuals, centers, dt


def flat_contact_trajopt(
    init_traj: Float[Array, "T n"],
    start: Float[Array, "n"],
    goal: Float[Array, "n"],
    system: ContactSystem,
    opt_cfg: FlatContactTrajOptConfig = FlatContactTrajOptConfig(),
    *,
    colls: tuple | None = None,
    world_geoms: tuple = (),
) -> tuple[Array, Array, Array, Array, Array]:
    """Differential-flatness **contact-aware** (fixed-grasp) trajectory optimization.

    Contact-aware, *not* contact-rich: the grasp is a fixed, persistent contact
    mode and the contact forces are allocated analytically (not optimized). For a
    solver that optimizes the contact forces as decision variables, use
    :func:`~pyroffi.optimization_engines.contact_rich_trajopt`.

    Optimizes the object's SE(3) pose trajectory (the flat output) together with
    slaved joint configs, a scalar grip-squeeze, and a shared trajectory
    timestep, so the grasp-closure and object-dynamics constraints are satisfied
    by construction. Returns ``(traj, forces, residuals, obj_centers, dt)``,
    where ``dt`` is the optimized timestep (total horizon ``(T - 1) * dt``).

    The objective is a tunable blend selected by ``opt_cfg``:

    * ``w_time``       — minimum time (drives the horizon ``(T - 1) * dt`` down),
    * ``w_smoothness`` — object + joint accel/jerk smoothness,
    * ``w_effort``     — total torque ``sum(tau**2)``.

    By default only ``w_time`` is active (pure min-time); ``w_smoothness`` and
    ``w_effort`` default to 0. Feasibility penalties (grasp tracking, joint /
    torque limits, grip validity) are always enforced regardless of the blend.

    ``start`` / ``goal`` are stacked joint configs (as in
    :func:`contact_sco_trajopt`); the goal object pose is derived from ``goal``'s
    reference gripper so a caller can reuse an IK'd goal directly.

    **Collision is opt-in and off by default.** Set ``opt_cfg.w_collision`` and
    pass ``colls`` (one collision model per manipulator, ``None`` to skip an
    arm) plus ``world_geoms``. Pass ``model.with_attachments(aset)`` to sweep
    the carried object too. With the default weights the term is not traced, so
    an existing caller's graph and results are unchanged.

    Two things this does *not* do, both worth knowing before relying on it:

    * It is a **penalty from a single seed**, not a search. This solver is one
      L-BFGS pass with penalty continuation, so the term pushes locally out of
      violation — it will not discover a different homotopy class (going *over*
      an obstacle rather than through it). Seed ``init_traj`` with a path from a
      multi-seed geometric planner such as
      :func:`~pyroffi.optimization_engines.ls_trajopt`; what this term then buys
      is that the min-time/effort solve cannot quietly undo that clearance.
    * A CUDA SDF checker buys **nothing** here. The kernel is opaque to autodiff
      and its ``custom_jvp`` takes both primal and tangent from the pure-JAX
      inner model, so under the ``value_and_grad`` this solver runs every
      iteration the kernel is bypassed entirely. Pass the JAX model; keep the
      CUDA one for forward-only verification of the result.
    """
    if colls is None:
        colls = (None,) * system.num_manipulators
    if len(colls) != system.num_manipulators:
        raise ValueError(
            f"colls must have one entry per manipulator (got {len(colls)} for "
            f"{system.num_manipulators}); use None to skip an arm."
        )
    if (opt_cfg.w_collision or opt_cfg.w_self_collision) and all(
        c is None for c in colls
    ):
        raise ValueError(
            "a nonzero collision weight needs at least one entry in `colls`; "
            "otherwise the term is silently zero."
        )
    T = init_traj.shape[0]
    ref = system.manipulators[0]
    # Goal object-pose twist relative to the start object pose.
    T_obj0 = C._gripper_world_pose(ref, system.split_q(start)[0])
    T_objT = C._gripper_world_pose(ref, system.split_q(goal)[0])
    # xi = exp(delta) @ T_obj0, so xi_T = T_objT  =>  delta_goal = log(T_objT @ T_obj0^-1).
    delta_goal = (T_objT @ T_obj0.inverse()).log()

    # Static initial squeeze so the grip starts closed.
    share = system.body.mass * system.gravity / system.num_manipulators
    init_squeeze = jnp.full((T,), float(share))

    return _flat_contact_jax(
        init_traj, delta_goal, init_squeeze, start, goal, system, opt_cfg,
        tuple(colls), tuple(world_geoms),
    )
