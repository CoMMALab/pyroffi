"""Contact-rich, dynamics-aware SCO trajectory optimization.

Extends the Sequential Convex Optimization trajopt scaffold
(:mod:`_sco_optimization`) into a *contact-rich, dynamics-aware* planner for
multi-manipulator manipulation of a rigidly-grasped object (see
:class:`pyroffi.dynamics._contact.ContactSystem`). The manipulator count and
morphology are not hardcoded here — a :class:`ContactSystem` may hold any
number of :class:`~pyroffi.dynamics._contact.ManipulatorSpec`.

Decision variables (per waypoint ``t``):

* ``q_t``      — stacked ``[q_0 | q_1 | ...]`` joint configuration of every
  manipulator.
* ``lambda_t`` — one world-frame contact force per manipulator, stacked as
  ``[f_0(3) | f_1(3) | ...]``.

The optimizer minimizes a smooth, low-effort trajectory subject to:

* **Fixed contact** — every non-reference gripper's relative pose stays equal
  to the grasp (``grasp_closure_residual``), enforced by an **augmented
  Lagrangian**: a per-timestep multiplier ``mu_t`` plus a quadratic penalty,
  with the multipliers updated by dual ascent between outer iterations.
* **Object dynamics** — the contact forces must satisfy the grasped object's
  Newton-Euler balance (``object_dynamics_residual``), likewise via an
  augmented Lagrangian with multipliers ``nu_t``.
* **Torque feasibility** — manipulator torques from GRiD inverse dynamics
  (with the contact reaction as ``f_ext``) stay within ``tau_max`` (squared
  hinge), and overall effort ``||tau||^2`` is penalized (``w_dynamics``).
* **Grip validity** — friction-cone + pushing-normal feasibility of every
  contact.

Outer loop (``n_outer_iters``): solve the inner subproblem with L-BFGS over the
full ``z = [q | lambda]`` vector, then update the duals ``mu, nu`` by dual
ascent and scale the penalty weights (penalty continuation).

The inner solver, smoothness and limit costs, and the L-BFGS two-loop are
imported unchanged from :mod:`_sco_optimization`.
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
from ._sco_optimization import (
    _LS_ALPHAS,
    _lbfgs_two_loop,
    _limits_cost,
    _smoothness_cost,
)


@dataclass(frozen=True)
class ContactTrajOptConfig:
    """Hyper-parameters for the contact-rich SCO solver."""

    # --- Outer / inner loop ---
    n_outer_iters: int = 12
    n_inner_iters: int = 40
    m_lbfgs: int = 6

    # --- Timing ---
    dt: float = 0.1
    """Timestep between waypoints (s) — used for finite-difference qd/qdd/a_obj."""

    # --- Smoothness / limits ---
    w_smooth: float = 1.0
    w_acc: float = 0.5
    w_jerk: float = 0.1
    w_limits: float = 1.0

    # --- Dynamics / effort ---
    w_dynamics: float = 1e-3
    """Weight on total manipulator effort ||tau||^2."""

    w_torque_limit: float = 1.0
    tau_max: float = 87.0
    """Per-joint torque limit (N·m) for the squared-hinge penalty."""

    # --- Contact / grip ---
    w_grip: float = 1.0
    mu_friction: float | None = None
    """Coulomb friction coefficient used for the grip-validity penalty. If
    ``None``, the grasped object's own ``geom.friction`` is used."""
    f_min: float = 2.0
    """Minimum inward normal force per contact (N)."""

    # --- Augmented-Lagrangian penalties (equality constraints) ---
    rho_grasp: float = 10.0
    rho_grasp_max: float = 1e4
    rho_obj: float = 1.0
    rho_obj_max: float = 1e3
    penalty_scale: float = 2.0
    """Per-outer-iteration multiplier on rho_grasp / rho_obj."""

    dual_scale: float = 1.0
    """Scaling on the dual-ascent step (mu += dual_scale * rho * residual)."""

    # --- Regularization on contact forces ---
    w_force_reg: float = 1e-4


# ---------------------------------------------------------------------------
# Finite-difference helpers (world-frame, uniform dt)
# ---------------------------------------------------------------------------

def _fd_vel_acc(x: Float[Array, "T d"], dt: float) -> tuple[Array, Array]:
    """First/second time derivatives via central differences (edges one-sided)."""
    dt = jnp.asarray(dt, x.dtype)  # avoid weak-float64 divisor in grad transpose
    T = x.shape[0]
    xd = jnp.zeros_like(x)
    xdd = jnp.zeros_like(x)
    xd = xd.at[1:-1].set((x[2:] - x[:-2]) / (2.0 * dt))
    xd = xd.at[0].set((x[1] - x[0]) / dt)
    xd = xd.at[-1].set((x[-1] - x[-2]) / dt)
    if T >= 3:
        xdd = xdd.at[1:-1].set((x[2:] - 2.0 * x[1:-1] + x[:-2]) / (dt * dt))
    return xd, xdd


# ---------------------------------------------------------------------------
# Augmented-Lagrangian objective
# ---------------------------------------------------------------------------

def _contact_cost(
    z: Float[Array, "nz"],
    system: ContactSystem,
    lower: Float[Array, "n"],
    upper: Float[Array, "n"],
    mu: Float[Array, "T 6k"],
    nu: Float[Array, "T 6"],
    rho_g: Array,
    rho_o: Array,
    T: int,
    cfg: ContactTrajOptConfig,
) -> Array:
    ndof = system.num_dof
    k = system.num_manipulators
    n_q = T * ndof

    q = z[:n_q].reshape(T, ndof)
    lam = z[n_q:].reshape(T, k, 3)  # per-manipulator contact forces

    # --- Smoothness + joint limits ---------------------------------------
    cost = cfg.w_smooth * _smoothness_cost(q, 0.0, cfg.w_acc, cfg.w_jerk)
    cost += cfg.w_limits * _limits_cost(q, lower, upper)

    # --- Manipulator dynamics: torques via GRiD ID with contact reaction --
    dof_offsets = []
    idx = 0
    for m in system.manipulators:
        dof_offsets.append(idx)
        idx += m.num_dof

    for i, m in enumerate(system.manipulators):
        o = dof_offsets[i]
        q_i = q[:, o : o + m.num_dof]
        f_i = lam[:, i, :]
        qd_i, qdd_i = _fd_vel_acc(q_i, cfg.dt)

        fext_i = jax.vmap(C.manipulator_contact_fext, in_axes=(None, 0, 0))(
            m, q_i, f_i
        )
        tau_i = m.grid.inverse_dynamics(q_i, qd_i, qdd_i, f_ext=fext_i)

        cost += cfg.w_dynamics * jnp.sum(tau_i**2)
        over_i = jnp.maximum(0.0, jnp.abs(tau_i) - cfg.tau_max) ** 2
        cost += cfg.w_torque_limit * jnp.sum(over_i)

    # --- Fixed-contact (grasp closure), augmented Lagrangian --------------
    g = jax.vmap(C.grasp_closure_residual, in_axes=(None, 0))(system, q)  # (T, 6(k-1))
    cost += jnp.sum(mu * g) + 0.5 * rho_g * jnp.sum(g**2)

    # --- Object Newton-Euler, augmented Lagrangian -------------------------
    centers = jax.vmap(C.object_center_world, in_axes=(None, 0))(system, q)  # (T,3)
    _, a_obj = _fd_vel_acc(centers, cfg.dt)  # (T,3)
    b = jax.vmap(C.object_dynamics_residual, in_axes=(None, 0, 0, 0))(
        system, q, a_obj, lam
    )  # (T,6)
    cost += jnp.sum(nu * b) + 0.5 * rho_o * jnp.sum(b**2)

    # --- Grip validity + force regularization ----------------------------
    grip = jax.vmap(
        C.grip_validity_penalty, in_axes=(None, 0, 0, None, None)
    )(system, q, lam, cfg.mu_friction, cfg.f_min)
    cost += cfg.w_grip * jnp.sum(grip)
    cost += cfg.w_force_reg * jnp.sum(lam**2)

    return cost


# ---------------------------------------------------------------------------
# Inner L-BFGS solve over z = [q | lambda]
# ---------------------------------------------------------------------------

def _inner_solve(
    z0: Float[Array, "nz"],
    endpoint_mask: Float[Array, "nz"],
    cost_fn,
    cfg: ContactTrajOptConfig,
) -> Array:
    m = cfg.m_lbfgs
    nz = z0.shape[0]

    cost0, g0 = jax.value_and_grad(cost_fn)(z0)
    g0 = g0 * endpoint_mask

    init = (
        z0, z0, cost0, z0, g0,
        jnp.zeros((m, nz)), jnp.zeros((m, nz)), jnp.zeros(m),
        jnp.int32(0), jnp.int32(0), jnp.int32(0),
    )

    def step(carry, _):
        (x, best_x, best_cost, x_prev, g_prev,
         s_buf, y_buf, rho_buf, m_used, newest, it) = carry

        cost_val, g = jax.value_and_grad(cost_fn)(x)
        g = g * endpoint_mask

        s_k = x - x_prev
        y_k = g - g_prev
        sy = jnp.dot(s_k, y_k)
        yy = jnp.dot(y_k, y_k)
        valid = (sy > 1e-10 * yy + 1e-30) & (it > 0)

        new_newest = (newest + 1) % m
        actual_newest = jnp.where(valid, new_newest, newest)
        # Update only the newest slot: O(nz) rather than O(m * nz).
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

        return (
            x_new, best_x, best_cost, x, g,
            s_buf, y_buf, rho_buf, m_used, newest, it + 1,
        ), None

    (_, best_x, *_), _ = jax.lax.scan(step, init, None, length=cfg.n_inner_iters)
    return best_x


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=("system", "opt_cfg"))
def _contact_sco_jax(
    init_traj: Float[Array, "T n"],
    init_forces: Float[Array, "T k 3"],
    start: Float[Array, "n"],
    goal: Float[Array, "n"],
    system: ContactSystem,
    opt_cfg: ContactTrajOptConfig,
) -> tuple[Array, Array, Array]:
    T = init_traj.shape[0]
    ndof = system.num_dof
    k = system.num_manipulators
    n_q = T * ndof
    nz = n_q + T * k * 3

    lower = jnp.concatenate([m.robot.joints.lower_limits for m in system.manipulators])
    upper = jnp.concatenate([m.robot.joints.upper_limits for m in system.manipulators])

    traj = init_traj.at[0].set(start).at[-1].set(goal)
    z = jnp.concatenate([traj.reshape(-1), init_forces.reshape(-1)])

    # Pin q at start/goal; contact forces are free everywhere.
    mask = jnp.ones(nz)
    mask = mask.at[:ndof].set(0.0).at[n_q - ndof:n_q].set(0.0)

    n_grasp = 6 * (k - 1)

    def outer(carry, _):
        z, mu, nu, rho_g, rho_o = carry

        cost_fn = lambda zz: _contact_cost(
            zz, system, lower, upper, mu, nu, rho_g, rho_o, T, opt_cfg
        )
        z = _inner_solve(z, mask, cost_fn, opt_cfg)

        # Re-pin endpoints.
        q = z[:n_q].reshape(T, ndof).at[0].set(start).at[-1].set(goal)
        z = z.at[:n_q].set(q.reshape(-1))
        lam = z[n_q:].reshape(T, k, 3)

        # Dual ascent on the equality-constraint multipliers.
        g = jax.vmap(C.grasp_closure_residual, in_axes=(None, 0))(system, q)
        centers = jax.vmap(C.object_center_world, in_axes=(None, 0))(system, q)
        _, a_obj = _fd_vel_acc(centers, opt_cfg.dt)
        b = jax.vmap(C.object_dynamics_residual, in_axes=(None, 0, 0, 0))(
            system, q, a_obj, lam
        )
        mu = mu + opt_cfg.dual_scale * rho_g * g
        nu = nu + opt_cfg.dual_scale * rho_o * b

        rho_g = jnp.minimum(rho_g * opt_cfg.penalty_scale, opt_cfg.rho_grasp_max)
        rho_o = jnp.minimum(rho_o * opt_cfg.penalty_scale, opt_cfg.rho_obj_max)

        return (z, mu, nu, rho_g, rho_o), None

    (z, mu, nu, rho_g, rho_o), _ = jax.lax.scan(
        outer,
        (
            z,
            jnp.zeros((T, n_grasp)),
            jnp.zeros((T, 6)),
            jnp.array(opt_cfg.rho_grasp, jnp.float32),
            jnp.array(opt_cfg.rho_obj, jnp.float32),
        ),
        None,
        length=opt_cfg.n_outer_iters,
    )

    q = z[:n_q].reshape(T, ndof)
    forces = z[n_q:].reshape(T, k, 3)

    # Report final constraint residuals for diagnostics.
    g = jax.vmap(C.grasp_closure_residual, in_axes=(None, 0))(system, q)
    residuals = jnp.array(
        [jnp.sqrt(jnp.mean(g**2)), jnp.max(jnp.abs(g))]
        if n_grasp > 0
        else [0.0, 0.0]
    )
    return q, forces, residuals


def contact_sco_trajopt(
    init_traj: Float[Array, "T n"],
    start: Float[Array, "n"],
    goal: Float[Array, "n"],
    system: ContactSystem,
    opt_cfg: ContactTrajOptConfig = ContactTrajOptConfig(),
    init_forces: Float[Array, "T k 3"] | None = None,
    *,
    use_cuda: bool = False,
) -> tuple[Array, Array, Array]:
    """Contact-rich, dynamics-aware SCO trajectory optimization.

    Args:
        init_traj:   Initial stacked ``[q_0 | q_1 | ...]`` trajectory. ``[T, n]``.
        start, goal: Pinned start/goal configurations. ``[n]``.
        system:      Contact system (manipulators + GRiD kernels + grasped object).
        opt_cfg:     Hyper-parameters (static — changes trigger recompilation).
        init_forces: Optional initial contact forces ``[T, k, 3]`` (one per
                     manipulator). Defaults to an even static split of the
                     object's weight.
        use_cuda:    Reserved for a future CUDA mirror; currently unsupported.

    Returns:
        traj:      Optimized joint trajectory.                  ``[T, n]``.
        forces:    Optimized contact forces, one per manipulator. ``[T, k, 3]``.
        residuals: ``[rms, max]`` grasp-closure residual for diagnostics.
    """
    if use_cuda:
        raise NotImplementedError(
            "The CUDA mirror of contact_sco_trajopt is not implemented yet; "
            "run the JAX backend (use_cuda=False) and profile before porting."
        )

    k = system.num_manipulators
    if init_forces is None:
        # Static split: each contact carries an even share of the object's weight.
        share = system.body.mass * system.gravity / k
        f0 = jnp.array([0.0, 0.0, share])
        init_forces = jnp.tile(f0, (init_traj.shape[0], k, 1))

    return _contact_sco_jax(
        init_traj, init_forces, start, goal, system, opt_cfg
    )
