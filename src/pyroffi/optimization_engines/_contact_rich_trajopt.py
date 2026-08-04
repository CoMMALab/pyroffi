"""Genuinely **contact-rich** trajectory optimization (forces as decision vars).

This is the contact-rich counterpart to the differential-flatness
:mod:`_flat_contact_trajopt` (which is only contact-*aware*: it *allocates* the
contact forces analytically from the object motion, so the system is flat and
the forces are an output). Here the contact forces are promoted to **first-class
decision variables** and optimized subject to the grasped object's Newton-Euler
balance — so the system is *not* flat and there is no closed-form substitution.

See ``flat_contact_trajopt_theory.md`` §8 for the precise ladder. In short:

* **Contact mode is still fixed** — the same rigid grasp (who touches what, and
  the captured relative-pose offsets) holds for the whole horizon. Contacts
  never make or break, so this is contact-rich but *not* contact-implicit;
  discovering the contact schedule (via complementarity) is deliberately out of
  scope because it becomes intractable fast.
* **Forces are optimized, not allocated.** The object's Newton-Euler residual
  ``Σf − m(v̇ − g) = 0`` / ``Σ(p−c)×f − (Iω̇ + ω×Iω) = 0`` becomes an equality
  constraint enforced by an **augmented Lagrangian** (multiplier ``nu`` + penalty
  ``rho_obj``), instead of being satisfied by construction. The friction cone and
  minimum-normal-force feasibility (``grip_validity_penalty``) then shape *which*
  balancing forces are admissible — a live constraint on real decision variables.
* **Grasp closure** (for multi-manipulator systems) is likewise an augmented
  Lagrangian (``mu`` + ``rho_grasp``); it is empty for a single manipulator.

The decision vector is

    z = [ q (T x ndof) | lambda (T x k x 3) | time_scale (1) ]

where ``q`` are stacked joint configs, ``lambda`` are per-manipulator world-frame
contact forces, and ``time_scale`` sets a shared timestep ``dt`` (so the horizon
duration is a decision variable, enabling a minimum-time objective — matching the
flat solver's interface).

Loop structure: an **augmented-Lagrangian outer loop** (dual ascent on ``mu, nu``
+ penalty continuation on ``rho_grasp, rho_obj``), each outer step solving the
inner subproblem over the full ``z`` with L-BFGS. Returns
``(traj, forces, residuals, obj_centers, dt)`` — the same signature as
:func:`~pyroffi.optimization_engines.flat_contact_trajopt`, so callers (and the
``16_02`` example) can swap the two solvers directly.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from ..dynamics import _contact as C
from ..dynamics._contact import ContactSystem
from ._contact_trajopt import _fd_vel_acc
from ._flat_contact_trajopt import (
    _collision_cost,
    _dt_from_scale,
    _scale_from_dt,
    _torque_cost,
)
from ._sco_optimization import (
    _LS_ALPHAS,
    _lbfgs_two_loop,
    _limits_cost,
    _smoothness_cost,
)


@dataclass(frozen=True)
class ContactRichTrajOptConfig:
    """Hyper-parameters for the contact-rich (forces-as-variables) solver."""

    # --- Augmented-Lagrangian outer loop + inner L-BFGS ---
    n_outer_iters: int = 12
    n_inner_iters: int = 40
    m_lbfgs: int = 8

    # --- Time parameterization (the trajectory *duration* is a decision var) ---
    dt: float = 0.1
    """Nominal / initial timestep. The per-step ``dt`` is optimized within
    ``[min_dt, max_dt]`` (a shared scalar); total horizon ``(T - 1) * dt``."""
    min_dt: float = 0.01
    max_dt: float = 0.5

    # --- High-level objective weights ------------------------------------
    w_time: float = 1.0
    """Minimum-time: penalizes the total horizon ``(T - 1) * dt``."""
    w_smoothness: float = 1.0
    """Scales the joint accel/jerk smoothness group."""
    w_effort: float = 1e-3
    """Scales total manipulator torque effort ``sum(tau**2)``."""

    # --- Joint smoothness / limits ---
    w_acc: float = 0.5
    w_jerk: float = 0.1
    w_limits: float = 1.0

    # --- Torque limits ---
    w_torque_limit: float = 1.0
    tau_max: float = 87.0

    # --- Collision (OPT-IN, exactly as in FlatContactTrajOptConfig: both
    #     weights default to 0, which traces the term out entirely) ----------
    w_collision: float = 0.0
    """Weight on the world-collision hinge. 0 disables the term (the default);
    a nonzero value requires ``colls``/``world_geoms`` at the call site."""
    w_self_collision: float = 0.0
    """Weight on the self-collision hinge, off by default for the same reason as
    in the flat solver: a spherized model can have a permanently negative
    self-distance baseline, which would apply a constant force unrelated to the
    obstacles."""
    collision_margin: float = 0.02
    """Clearance the hinge asks for, in metres."""
    collision_temperature: float = 0.05
    """Softmin temperature reducing each group's pair distances to one scalar
    (the inner solve is L-BFGS, so a hard ``min``'s kinks corrupt the
    inverse-Hessian estimate)."""

    # --- Contact / grip ---
    w_grip: float = 1.0
    mu_friction: float | None = None
    f_min: float = 2.0
    w_force_reg: float = 1e-4
    """Regularizer on the contact forces (prefers minimal internal force)."""

    # --- Augmented-Lagrangian penalties (equality constraints) ---
    rho_grasp: float = 10.0
    rho_grasp_max: float = 1e4
    rho_obj: float = 1.0
    rho_obj_max: float = 1e3
    penalty_scale: float = 2.0
    """Per-outer-iteration multiplier on rho_grasp / rho_obj."""
    dual_scale: float = 1.0
    """Scaling on the dual-ascent step (mu += dual_scale * rho * residual)."""

    # --- Early termination -------------------------------------------------
    #   Both loops are fixed-length `lax.scan`s, so with these enabled a solve
    #   that converges early stops instead of burning the remaining iterations.
    #   OFF by default: on the AL subproblems this solver actually sees, the
    #   penalty terms keep the gradient norm far above any sensible tolerance,
    #   so the check never fires and the per-iteration branch measured ~11%
    #   *slower* on the bimanual box-lift benchmark. Turn them on for problems
    #   that do reach a stationary point before the iteration budget runs out.
    grad_tol: float = 0.0
    """Inner L-BFGS stops once ``max|grad|`` drops below this. 0 disables."""
    constraint_tol: float = 0.0
    """Outer AL loop stops once ``max|g|`` and ``max|b|`` are both below this
    (and the duals have therefore stopped moving). 0 disables."""


# ---------------------------------------------------------------------------
# Augmented-Lagrangian objective (forces are DECISION VARIABLES here)
# ---------------------------------------------------------------------------

def _contact_rich_cost(
    z: Float[Array, "nz"],
    system: ContactSystem,
    lower: Float[Array, "n"],
    upper: Float[Array, "n"],
    mu: Float[Array, "T 6*(k-1)"],
    nu: Float[Array, "T 6"],
    rho_g: Array,
    rho_o: Array,
    T: int,
    cfg: ContactRichTrajOptConfig,
    colls: tuple = (),
    world_geoms: tuple = (),
) -> Array:
    ndof = system.num_dof
    k = system.num_manipulators
    n_q = T * ndof
    n_lam = T * k * 3

    q = z[:n_q].reshape(T, ndof)
    lam = z[n_q : n_q + n_lam].reshape(T, k, 3)  # per-manipulator contact forces
    dt = _dt_from_scale(z[-1], cfg)  # shared, optimized timestep

    # --- Minimum-time: penalize the total horizon (T - 1) * dt -----------
    cost = cfg.w_time * (T - 1) * dt

    # --- Smoothness + joint limits ---------------------------------------
    cost += cfg.w_smoothness * _smoothness_cost(q, 0.0, cfg.w_acc, cfg.w_jerk)
    cost += cfg.w_limits * _limits_cost(q, lower, upper)

    # --- Collision (opt-in) ----------------------------------------------
    # Shares the flat solver's helper: the term depends only on q, which is the
    # leading block of z in both solvers, so there is one implementation to keep
    # honest. Guarded on the *static* config, so the default weights leave the
    # graph untouched.
    if cfg.w_collision or cfg.w_self_collision:
        cost += _collision_cost(q, system, colls, world_geoms, cfg)

    # --- Grasp kinematics: ONE forward-kinematics pass per manipulator -----
    #   Every contact term below (closure residual, object centre + Newton-Euler
    #   balance, grip penalty, and the reaction wrench handed to inverse
    #   dynamics) is a projection of the same gripper poses. Computing them
    #   together instead of letting each residual re-derive them from q removes
    #   five of the six FK sweeps this cost used to run per timestep.
    points, centers, axes, g = jax.vmap(C.grasp_kinematics, in_axes=(None, 0))(
        system, q
    )  # (T,k,3), (T,3), (T,k,3), (T,6(k-1))

    # --- Manipulator dynamics: torques via GRiD ID with contact reaction --
    cost += _torque_cost(system, q, lam, points, dt, cfg)

    # --- Fixed-contact (grasp closure), augmented Lagrangian --------------
    if g.shape[-1] > 0:
        cost += jnp.sum(mu * g) + 0.5 * rho_g * jnp.sum(g**2)

    # --- Object Newton-Euler balance, augmented Lagrangian ----------------
    #   THIS is the contact-rich core: the forces are free variables and this
    #   equality is what ties them to the object motion (vs. the flat solver's
    #   analytic G+ allocation that made the residual identically zero).
    _, a_obj = _fd_vel_acc(centers, dt)  # (T,3)
    b = jax.vmap(C.object_dynamics_residual_at, in_axes=(None, 0, 0, 0, 0))(
        system, centers, points, a_obj, lam
    )  # (T,6)
    cost += jnp.sum(nu * b) + 0.5 * rho_o * jnp.sum(b**2)

    # --- Grip validity (parallel-jaw pinch) + force regularization -------
    grip = jax.vmap(
        C.parallel_jaw_grip_penalty_at, in_axes=(None, 0, 0, None)
    )(system, axes, lam, cfg.mu_friction)
    cost += cfg.w_grip * jnp.sum(grip)
    cost += cfg.w_force_reg * jnp.sum(lam**2)

    return cost


def _constraint_residuals(
    system: ContactSystem,
    q: Float[Array, "T n"],
    lam: Float[Array, "T k 3"],
    dt: Array,
) -> tuple[Array, Array, Array]:
    """``(grasp_closure, object_dynamics, object_centers)`` over a trajectory.

    Both the dual-ascent step and the final diagnostics need exactly this
    triple; sharing one FK pass keeps each of them to a single pass instead of
    the three they used to run apiece.
    """
    points, centers, _, g = jax.vmap(C.grasp_kinematics, in_axes=(None, 0))(
        system, q
    )
    _, a_obj = _fd_vel_acc(centers, dt)
    b = jax.vmap(C.object_dynamics_residual_at, in_axes=(None, 0, 0, 0, 0))(
        system, centers, points, a_obj, lam
    )
    return g, b, centers


# ---------------------------------------------------------------------------
# Inner L-BFGS solve over z = [q | lambda | time_scale]
# ---------------------------------------------------------------------------

def _inner_solve(z0, endpoint_mask, cost_fn, cfg: ContactRichTrajOptConfig):
    m = cfg.m_lbfgs
    nz = z0.shape[0]
    cost0, g0 = jax.value_and_grad(cost_fn)(z0)
    g0 = g0 * endpoint_mask

    init = (
        z0, z0, cost0, z0, g0,
        jnp.zeros((m, nz)), jnp.zeros((m, nz)), jnp.zeros(m),
        jnp.int32(0), jnp.int32(0), jnp.int32(0),
        jnp.bool_(False),  # converged
    )

    def body(carry):
        (x, best_x, best_cost, x_prev, g_prev,
         s_buf, y_buf, rho_buf, m_used, newest, it, _) = carry
        cost_val, g = jax.value_and_grad(cost_fn)(x)
        g = g * endpoint_mask
        s_k = x - x_prev
        y_k = g - g_prev
        sy = jnp.dot(s_k, y_k)
        yy = jnp.dot(y_k, y_k)
        valid = (sy > 1e-10 * yy + 1e-30) & (it > 0)
        new_newest = (newest + 1) % m
        actual_newest = jnp.where(valid, new_newest, newest)
        # Update only the newest slot. Selecting on the *row* is O(nz); the
        # earlier `where` over the whole updated buffer was O(m * nz) for the
        # same result.
        s_buf = s_buf.at[new_newest].set(jnp.where(valid, s_k, s_buf[new_newest]))
        y_buf = y_buf.at[new_newest].set(jnp.where(valid, y_k, y_buf[new_newest]))
        rho_buf = jnp.where(valid, rho_buf.at[new_newest].set(1.0 / (sy + 1e-30)), rho_buf)
        m_used = jnp.where(valid & (m_used < m), m_used + 1, m_used)
        newest = actual_newest
        dir_lbfgs = _lbfgs_two_loop(g, s_buf, y_buf, rho_buf, m_used, newest, m)
        dir_gd = -g / (jnp.linalg.norm(g) + 1e-18)
        direction = jnp.where(m_used > 0, dir_lbfgs, dir_gd) * endpoint_mask
        suff = cost_val * (1.0 - 1e-4)
        trial = jax.vmap(lambda a: cost_fn(x + a * direction))(_LS_ALPHAS)
        has_suff = trial < suff
        idx = jnp.where(jnp.any(has_suff), jnp.argmax(has_suff), jnp.argmin(trial))
        alpha = _LS_ALPHAS[idx]
        x_new = x + alpha * direction
        new_cost = trial[idx]
        improved = new_cost < best_cost
        best_x = jnp.where(improved, x_new, best_x)
        best_cost = jnp.where(improved, new_cost, best_cost)
        # Stationary point of the (currently fixed) AL subproblem: further
        # L-BFGS steps cannot move `best_x`, so the remaining iterations are
        # pure waste. Static-gated so the disabled default emits no reduction.
        converged = (
            jnp.max(jnp.abs(g)) < cfg.grad_tol
            if cfg.grad_tol > 0.0
            else jnp.bool_(False)
        )
        return (
            x_new, best_x, best_cost, x, g,
            s_buf, y_buf, rho_buf, m_used, newest, it + 1,
            converged,
        )

    # Gated on the *static* config so the default (disabled) emits exactly the
    # graph it always did. `lax.cond` does genuinely skip the branch on GPU here
    # -- these solvers optimize a single z, so it is not vmapped down into a
    # `select` -- but on the AL subproblems this solver actually sees, the
    # gradient norm never approaches grad_tol, so the check only ever costs a
    # per-iteration branch. Enable it for problems that do converge early.
    if cfg.grad_tol > 0.0:
        def step(carry, _):
            return jax.lax.cond(carry[-1], lambda c: c, body, carry), None
    else:
        def step(carry, _):
            return body(carry), None

    (_, best_x, *_), _ = jax.lax.scan(step, init, None, length=cfg.n_inner_iters)
    return best_x


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=("system", "opt_cfg"))
def _contact_rich_jax(
    init_traj: Float[Array, "T n"],
    init_forces: Float[Array, "T k 3"],
    start: Float[Array, "n"],
    goal: Float[Array, "n"],
    system: ContactSystem,
    opt_cfg: ContactRichTrajOptConfig,
    colls: tuple = (),
    world_geoms: tuple = (),
) -> tuple[Array, Array, Array, Array, Array]:
    T = init_traj.shape[0]
    ndof = system.num_dof
    k = system.num_manipulators
    n_q = T * ndof
    n_lam = T * k * 3
    nz = n_q + n_lam + 1  # + shared time-scale scalar

    lower = jnp.concatenate([m.robot.joints.lower_limits for m in system.manipulators])
    upper = jnp.concatenate([m.robot.joints.upper_limits for m in system.manipulators])

    traj = init_traj.at[0].set(start).at[-1].set(goal)
    scale0 = jnp.array([_scale_from_dt(opt_cfg.dt, opt_cfg)], jnp.float32)
    z = jnp.concatenate([traj.reshape(-1), init_forces.reshape(-1), scale0])

    # Pin q at start/goal; forces + time-scale are free everywhere.
    mask = jnp.ones(nz)
    mask = mask.at[:ndof].set(0.0).at[n_q - ndof : n_q].set(0.0)

    n_grasp = 6 * (k - 1)

    def outer_body(carry):
        z, mu, nu, rho_g, rho_o, _ = carry
        cost_fn = lambda zz: _contact_rich_cost(
            zz, system, lower, upper, mu, nu, rho_g, rho_o, T, opt_cfg,
            colls, world_geoms,
        )
        z = _inner_solve(z, mask, cost_fn, opt_cfg)

        # Re-pin endpoints.
        q = z[:n_q].reshape(T, ndof).at[0].set(start).at[-1].set(goal)
        z = z.at[:n_q].set(q.reshape(-1))
        lam = z[n_q : n_q + n_lam].reshape(T, k, 3)
        dt = _dt_from_scale(z[-1], opt_cfg)

        # Dual ascent on the equality-constraint multipliers.
        g, b, _ = _constraint_residuals(system, q, lam, dt)
        if n_grasp > 0:
            mu = mu + opt_cfg.dual_scale * rho_g * g
        nu = nu + opt_cfg.dual_scale * rho_o * b
        rho_g = jnp.minimum(rho_g * opt_cfg.penalty_scale, opt_cfg.rho_grasp_max)
        rho_o = jnp.minimum(rho_o * opt_cfg.penalty_scale, opt_cfg.rho_obj_max)
        # Both equality constraints satisfied: the dual updates above are then
        # no-ops and every later outer iteration reproduces this same z.
        if opt_cfg.constraint_tol > 0.0:
            gmax = jnp.max(jnp.abs(g)) if n_grasp > 0 else jnp.array(0.0)
            converged = (
                jnp.maximum(gmax, jnp.max(jnp.abs(b))) < opt_cfg.constraint_tol
            )
        else:
            converged = jnp.bool_(False)
        return (z, mu, nu, rho_g, rho_o, converged)

    if opt_cfg.constraint_tol > 0.0:
        def outer(carry, _):
            return jax.lax.cond(carry[-1], lambda c: c, outer_body, carry), None
    else:
        def outer(carry, _):
            return outer_body(carry), None

    (z, mu, nu, rho_g, rho_o, _), _ = jax.lax.scan(
        outer,
        (
            z,
            jnp.zeros((T, max(n_grasp, 1))),
            jnp.zeros((T, 6)),
            jnp.array(opt_cfg.rho_grasp, jnp.float32),
            jnp.array(opt_cfg.rho_obj, jnp.float32),
            jnp.bool_(False),
        ),
        None,
        length=opt_cfg.n_outer_iters,
    )

    q = z[:n_q].reshape(T, ndof)
    forces = z[n_q : n_q + n_lam].reshape(T, k, 3)
    dt = _dt_from_scale(z[-1], opt_cfg)

    # Diagnostics: object-dynamics residual (this is the constraint the forces
    # are optimized to satisfy) and grasp-closure residual. One shared FK pass.
    g, b, centers = _constraint_residuals(system, q, forces, dt)
    grasp_rms = jnp.sqrt(jnp.mean(g**2)) if g.shape[-1] > 0 else jnp.array(0.0)
    residuals = jnp.array(
        [jnp.sqrt(jnp.mean(b**2)), jnp.max(jnp.abs(b)), grasp_rms]
    )
    return q, forces, residuals, centers, dt


def contact_rich_trajopt(
    init_traj: Float[Array, "T n"],
    start: Float[Array, "n"],
    goal: Float[Array, "n"],
    system: ContactSystem,
    opt_cfg: ContactRichTrajOptConfig = ContactRichTrajOptConfig(),
    init_forces: Float[Array, "T k 3"] | None = None,
    *,
    colls: tuple | None = None,
    world_geoms: tuple = (),
) -> tuple[Array, Array, Array, Array, Array]:
    """Contact-rich trajectory optimization with contact forces as decision vars.

    Unlike :func:`~pyroffi.optimization_engines.flat_contact_trajopt` (which is
    contact-*aware*: it allocates the forces analytically from the object motion
    so the system is differentially flat), this solver promotes the per-contact
    forces to first-class decision variables and enforces the grasped object's
    Newton-Euler balance as an augmented-Lagrangian equality constraint. The
    contact *mode* is still fixed (the same rigid grasp for the whole horizon) —
    this is contact-rich, not contact-implicit.

    Args:
        init_traj:   Initial stacked ``[q_0 | q_1 | ...]`` trajectory. ``[T, n]``.
        start, goal: Pinned start/goal configurations. ``[n]``.
        system:      Contact system (manipulators + GRiD kernels + grasped object).
        opt_cfg:     Hyper-parameters (static — changes trigger recompilation).
        init_forces: Optional initial contact forces ``[T, k, 3]``. Defaults to an
                     even static split of the object's weight.
        colls:       Optional per-manipulator collision models (``None`` to skip
                     an arm), required when a collision weight is nonzero. Pass
                     ``model.with_attachments(aset)`` to sweep the carried object
                     too.
        world_geoms: Obstacles the collision hinge is measured against.

    **Collision is opt-in and off by default** (``opt_cfg.w_collision``); with the
    default weights the term is not traced, so existing callers are unaffected.
    The same two caveats as in
    :func:`~pyroffi.optimization_engines.flat_contact_trajopt` apply, and the
    first one applies *more* strongly here: this is a penalty inside an
    augmented-Lagrangian loop seeded from one trajectory, so it pushes locally out
    of violation and will not find a different homotopy class — seed
    ``init_traj`` from a multi-seed geometric planner such as
    :func:`~pyroffi.optimization_engines.ls_trajopt`. And a CUDA SDF checker buys
    nothing, since its ``custom_jvp`` takes primal *and* tangent from the pure-JAX
    inner model and this solver differentiates every iteration.

    Returns:
        ``(traj, forces, residuals, obj_centers, dt)`` — matching the flat
        solver's signature. ``residuals`` is ``[obj_dyn_rms, obj_dyn_max,
        grasp_rms]``; ``dt`` is the optimized timestep (horizon ``(T - 1) * dt``).
    """
    k = system.num_manipulators
    if init_forces is None:
        share = system.body.mass * system.gravity / k
        f0 = jnp.array([0.0, 0.0, share])
        init_forces = jnp.tile(f0, (init_traj.shape[0], k, 1))
    if colls is None:
        colls = (None,) * k
    if len(colls) != k:
        raise ValueError(
            f"colls must have one entry per manipulator (got {len(colls)} for "
            f"{k}); use None to skip an arm."
        )
    if (opt_cfg.w_collision or opt_cfg.w_self_collision) and all(
        c is None for c in colls
    ):
        raise ValueError(
            "a nonzero collision weight needs at least one entry in `colls`; "
            "otherwise the term is silently zero."
        )
    return _contact_rich_jax(
        init_traj, init_forces, start, goal, system, opt_cfg,
        tuple(colls), tuple(world_geoms),
    )
