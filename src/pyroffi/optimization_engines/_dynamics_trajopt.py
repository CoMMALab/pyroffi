"""Tier 1 — dynamics-aware L-BFGS trajopt over an arbitrary cost.

Built on the shared :mod:`_trajopt_core` primitives. In its default
configuration this is exactly the contact-free, single-arm L-BFGS solver that
``ioc`` / ``iosp`` (SPaSM / inverse-SPaSM) call as their forward solver: a single
L-BFGS minimize of the caller's ``cost_fn(x) -> scalar``, tracking the
best-*stationarity* iterate, with the ``early_stop`` / ``unroll_tail`` /
``soft_line_search`` / ``soft_curvature_gate`` semantics the implicit-adjoint IOC
pipeline depends on.

Opt-in, all defaulting to preserve exact current behavior, it also exposes:

* ``constraints`` — an arbitrary tuple of
  :class:`~pyroffi.optimization_engines._trajopt_core.AugmentedLagrangianTerm`,
  folded into a generic AL outer loop. Empty ⇒ the single-solve behavior above.
* ``use_sco`` — Sequential Convex Optimization: each AL outer iteration
  linearizes every supplied *inequality* residual at the current iterate before
  the inner solve (à la Schulman et al. 2013), generalized beyond collision.
* ``robot`` / ``grid`` — when supplied, tier 1 auto-adds kinematic smoothness /
  joint-limit terms and a dynamics-feasibility (torque-limit) AL term, so a
  standalone caller gets a dynamics-feasible solve without hand-writing the cost.
  Existing ``ioc`` / ``iosp`` callers pass neither, so their graph is untouched.
"""

from __future__ import annotations

import dataclasses
import functools
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as onp
from jax import Array
from jaxtyping import Float

from ._trajopt_core import (
    AugmentedLagrangianTerm,
    _al_outer_loop,
    _lbfgs_driver,
    _make_trust,
    _projected_gd,
)


@dataclass(frozen=True)
class DynamicsTrajOptConfig:
    """Hyper-parameters for the dynamics-aware L-BFGS trajopt engine."""

    n_iters: int = 200
    """Number of L-BFGS steps."""

    method: str = "lbfgs"
    """"lbfgs" (default, unchanged) or "projected_gd".  The projected-GD path is
    a SPaSM-style solver: plain gradient descent from the (already valid)
    straight-line seed with a linearly-decayed step, PROJECTED onto the joint
    box after every step.  It stays in the neighborhood of the feasible init and
    cannot diverge out of joint limits the way an unconstrained L-BFGS step can
    on the composed pick-place objective."""

    gd_lr: float = 0.1
    """projected_gd only: initial GD step, decayed linearly to 0 over n_iters."""

    q_lo: tuple = ()
    """projected_gd only: per-DOF lower joint limits (hashable tuple).  Empty
    disables the box projection (pure GD)."""

    q_hi: tuple = ()
    """projected_gd only: per-DOF upper joint limits (hashable tuple)."""

    dof: int = 0
    """projected_gd only: DOF, to reshape the flat iterate to (T_interior, dof)
    for per-waypoint box projection."""

    m_lbfgs: int = 8
    """L-BFGS history size (number of curvature pairs)."""

    grad_tol: float = 1e-6
    """Inner L-BFGS stops once ``max|grad|`` drops below this. 0 disables (falls
    back to the fixed ``n_iters`` budget). Only meaningful when
    ``early_stop=True``."""

    early_stop: bool = True
    """True: run the ``while_loop`` early-stopping form -- cheap, but not
    reverse-mode differentiable (JAX cannot backprop through a data-dependent
    trip count), so this form must only ever be called under `stop_gradient`.

    False: run a fixed-length, reverse-mode-differentiable form instead --
    `jax.lax.scan` for exactly ``n_iters`` steps, optionally split so only the
    last ``unroll_tail`` are actually unrolled and differentiated (Domke 2012
    truncated unrolling)."""

    unroll_tail: int = 0
    """Only used when ``early_stop=False``. Number of trailing steps that carry
    gradients; 0 (or >= n_iters) unrolls and differentiates the whole solve."""

    soft_line_search: bool = False
    """Opt-in, default False. Replaces the hard argmax over ``_LS_ALPHAS`` with a
    softmax blend over the trial costs, to keep the solve continuous in an
    upstream differentiable input (e.g. an IK boundary condition). See the shared
    driver for the mechanism."""

    soft_curvature_gate: bool = False
    """Opt-in, default False. Replaces the discrete curvature-pair admit/reject
    and the hard L-BFGS/GD direction switch with smooth blends, for the same
    reason as ``soft_line_search``. See the shared driver."""

    # --- Opt-in tier-1 SCO / augmented-Lagrangian extensions --------------
    #     All default to preserve the exact single-solve behavior above.

    constraints: tuple = ()
    """Tuple of
    :class:`~pyroffi.optimization_engines._trajopt_core.AugmentedLagrangianTerm`.
    Empty (default) ⇒ a single L-BFGS solve of ``cost_fn``, exactly as before.
    Non-empty ⇒ ``cost_fn`` becomes the AL *base* cost and these terms are folded
    into a generic AL outer loop with their own duals and penalties."""

    use_sco: bool = False
    """When True (and ``constraints`` non-empty), each AL outer iteration
    linearizes every *inequality* residual at the current iterate before the
    inner solve, à la Sequential Convex Optimization. Equality terms are left
    exact. No effect when ``constraints`` is empty."""

    n_outer_iters: int = 10
    """AL / SCO outer iterations (only used when ``constraints`` is non-empty)."""

    constraint_tol: float = 0.0
    """Outer loop stops once every term's ``max|residual|`` is below this. 0
    disables (runs the full ``n_outer_iters`` budget)."""

    dual_scale: float = 1.0
    """Scaling on the AL dual-ascent step."""

    robot: object = None
    """Optional :class:`~pyroffi.Robot`. When supplied together with ``grid`` (and
    ``dof`` for the reshape), tier 1 auto-adds kinematic smoothness / joint-limit
    terms and a dynamics-feasibility AL term for a standalone dynamics-feasible
    solve. Left ``None`` by every ``ioc`` / ``iosp`` caller."""

    grid: object = None
    """Optional :class:`~pyroffi.dynamics.GRiDDynamics`, paired with ``robot``."""

    tau_max: float = 87.0
    """Torque limit for the auto-added dynamics-feasibility term (robot/grid)."""

    auto_dt: float = 0.1
    """Timestep for finite-difference velocities/accelerations in the auto-added
    dynamics-feasibility term."""

    w_smooth: float = 1.0
    w_limits: float = 1.0
    """Weights for the auto-added kinematic terms (robot/grid path only)."""

    # --- Adaptive (Schulman) trust region for the AL/SCO outer loop --------
    #     Only active when ``constraints`` is non-empty (the AL path); best paired
    #     with ``use_sco`` (the ratio test judges the linearized model). Default
    #     off ⇒ no trust region, byte-identical to the plain AL loop.
    adaptive_trust: bool = False
    """Enable Schulman actual-vs-predicted trust-region resizing in the AL loop."""
    tr_coef0: float = 1.0
    """Initial trust-region coefficient."""
    tr_tighten: float = 4.0
    tr_loosen: float = 0.25
    tr_shrink_ratio: float = 0.25
    tr_expand_ratio: float = 0.75
    tr_accept_ratio: float = 0.1
    tr_coef_min: float = 1e-2
    tr_coef_max: float = 1e4


def dynamics_trajopt(
    x0: Float[Array, "n"],
    cost_fn: Callable[[Float[Array, "n"]], Array],
    opt_cfg: DynamicsTrajOptConfig = DynamicsTrajOptConfig(),
) -> Float[Array, "n"]:
    """Minimize ``cost_fn`` over a flat decision vector with L-BFGS.

    Default (``method="lbfgs"``, ``constraints=()``, ``robot=None``): a single
    L-BFGS solve tracking the best-stationarity iterate — the exact forward solver
    ``ioc.inner.make_inner_solver`` wires in. ``method="projected_gd"`` runs the
    SPaSM box-projected GD fallback instead.

    With ``constraints`` (and/or ``robot``/``grid``) supplied, the solve becomes
    a generic augmented-Lagrangian outer loop over those terms, optionally with
    SCO linearization of inequality residuals (``use_sco``).
    """
    if opt_cfg.method == "projected_gd":
        return _projected_gd(
            x0, cost_fn,
            n_iters=opt_cfg.n_iters, gd_lr=opt_cfg.gd_lr,
            q_lo=opt_cfg.q_lo, q_hi=opt_cfg.q_hi, dof=opt_cfg.dof,
        )

    loop = "while" if opt_cfg.early_stop else "unroll"

    def inner_solve(z0, cf):
        return _lbfgs_driver(
            z0, cf,
            n_iters=opt_cfg.n_iters, m_lbfgs=opt_cfg.m_lbfgs,
            grad_tol=opt_cfg.grad_tol, loop=loop,
            unroll_tail=opt_cfg.unroll_tail,
            soft_line_search=opt_cfg.soft_line_search,
            soft_curvature_gate=opt_cfg.soft_curvature_gate,
            best_by="grad",
        )

    constraints = _build_constraints(x0, opt_cfg)
    base_cost, extra = _augmented_base_cost(cost_fn, opt_cfg)

    if not constraints:
        # Exact legacy path: a single inner solve of the (possibly kinematic-
        # augmented) cost. With robot=None this is byte-identical to before.
        return inner_solve(x0, base_cost if extra else cost_fn)

    z, _, _ = _al_outer_loop(
        x0, inner_solve, constraints, lambda z, zk: base_cost(z),
        n_outer_iters=opt_cfg.n_outer_iters,
        dual_scale=opt_cfg.dual_scale,
        constraint_tol=opt_cfg.constraint_tol,
        sco_linearize=opt_cfg.use_sco,
        trust=_make_trust(opt_cfg),
    )
    return z


# ---------------------------------------------------------------------------
# Opt-in robot/grid auto-terms
# ---------------------------------------------------------------------------

def _augmented_base_cost(cost_fn, opt_cfg):
    """``(base_cost_fn, added_anything)``.

    When ``robot``/``grid`` are supplied, add kinematic smoothness + joint-limit
    penalties to the caller's cost (the AL base). Otherwise return ``cost_fn``
    unchanged and ``False`` (the legacy graph).
    """
    if opt_cfg.robot is None or opt_cfg.dof <= 0:
        return cost_fn, False

    from ._sco_optimization import _limits_cost, _smoothness_cost

    dof = opt_cfg.dof
    robot = opt_cfg.robot
    lower = robot.joints.lower_limits
    upper = robot.joints.upper_limits

    def base(x):
        t = x.reshape(-1, dof)
        c = cost_fn(x)
        c = c + opt_cfg.w_smooth * _smoothness_cost(t, 1.0, 0.5, 0.1)
        c = c + opt_cfg.w_limits * _limits_cost(t, lower, upper)
        return c

    return base, True


def _build_constraints(x0, opt_cfg) -> tuple:
    """Assemble the AL constraint tuple: caller-supplied plus, if robot/grid are
    given, an auto dynamics-feasibility (torque-limit) inequality term."""
    constraints = tuple(opt_cfg.constraints)
    if opt_cfg.robot is not None and opt_cfg.grid is not None and opt_cfg.dof > 0:
        from ..dynamics._contact import dynamics_feasibility_residual

        dof = opt_cfg.dof
        grid = opt_cfg.grid

        def residual(x):
            t = x.reshape(-1, dof)
            return dynamics_feasibility_residual(
                grid, t, opt_cfg.auto_dt, opt_cfg.tau_max
            )

        constraints = constraints + (
            AugmentedLagrangianTerm(
                residual_fn=residual, kind="ineq",
                rho0=1.0, rho_max=1e3, penalty_scale=2.0,
                name="torque_limit",
            ),
        )
    return constraints


# ---------------------------------------------------------------------------
# Batched, collision-aware, dynamically feasible trajopt (the motion-generator
# default). Same call signature as sco_trajopt / chomp_trajopt / stomp_trajopt.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DynamicsAwareTrajOptConfig:
    """Hyper-parameters for :func:`dynamics_aware_trajopt`."""

    dt: float = 0.1
    """Seconds between waypoints; sets the velocities/accelerations the
    velocity and torque limits see. ``n_timesteps * dt`` is the duration."""

    tau_max: tuple = ()
    """Per-joint torque limits (N·m / N). Empty ⇒ the URDF ``<limit effort>``
    of each actuated joint (joints without one are unlimited)."""

    w_vel: float = 1.0
    w_acc: float = 0.5
    w_jerk: float = 0.1
    """Smoothness weights (same finite differences as :class:`ScoTrajOptConfig`)."""

    collision_margin: float = 0.01
    """Required clearance (m) for every robot link/pair along the path."""

    limit_backoff: float = 0.01
    """The AL loop converges onto active limits from outside (PyRoki wall scene:
    velocity at 1.0001-1.001x, joint limits 5-10 mrad over). Velocity and torque
    are enforced at ``1 - limit_backoff`` of their limits and joint limits
    ``limit_backoff`` rad inside."""

    collision_substeps: int = 4
    """Collision is enforced at each waypoint and ``collision_substeps - 1``
    joint-space interpolants per segment, so the path cannot tunnel through an
    obstacle between waypoints."""

    w_collision: float = 1.0
    w_collision_max: float = 100.0
    penalty_scale: float = 3.0
    """Collision AL penalty: initial, maximum, and growth per outer iteration."""

    path_stage: bool = True
    """Stage 1: solve collision + joint limits alone under a strongly weighted
    smoothness cost, then stage 2 adds velocity and torque limits from that
    path. Solved jointly from a straight line through an obstacle, the
    velocity limit pins the trajectory near the seed and it never clears the
    obstacle (PyRoki wall scene, dt=0.02: 0/3 collision-free jointly, 3/3
    fully feasible staged)."""

    path_smooth_scale: float = 100.0
    """Smoothness weight multiplier in stage 1. A weakly smoothed stage-1 path
    is a jerky detour that stage 2 cannot retime within the velocity limit
    (scale 1: 15-18x the velocity limit after stage 1; 100: 5-7x, then 1.0x)."""

    extra_constraints: tuple = ()
    """Caller inequality constraints ``fn(traj (T, DOF)) -> residual`` (feasible where
    ``<= 0``) on the full trajectory, endpoints included, enforced in both stages (e.g. a
    task-space region on every waypoint). Must be hashable (module-level or cached
    callables): the config is a static jit argument."""

    solver: DynamicsTrajOptConfig = DynamicsTrajOptConfig(
        n_iters=15, use_sco=True, n_outer_iters=20, adaptive_trust=True, tr_coef0=0.5
    )
    """Tier-1 solver settings. ``constraints`` / ``robot`` / ``grid`` in it are
    ignored: the constraint terms are supplied here. The adaptive trust region
    keeps the linearized steps local; without it the solve takes rad-scale jumps."""


def effort_limits(robot) -> Float[Array, "n_act"]:
    """Per-actuated-joint URDF effort limits; ``inf`` where the URDF gives none."""
    joint_map = robot.urdf.joint_map
    lim = []
    for name in robot.joints.actuated_names:
        limit = joint_map[name].limit
        effort = None if limit is None else limit.effort
        lim.append(float(effort) if effort else onp.inf)
    return jnp.asarray(lim)


@functools.partial(jax.jit, static_argnames=("n_probe",))
def _changing_collision_terms(robot, robot_coll, world_geoms, n_probe: int = 4):
    """Masks of the self pairs / per-link world entries whose distance changes
    with the configuration. Fixed links (a base below a table) and rigidly
    attached pairs are constant: they can never be satisfied or improved, and
    as AL terms they only inflate penalties and duals."""
    lo = jnp.where(jnp.isfinite(robot.joints.lower_limits), robot.joints.lower_limits, -jnp.pi)
    hi = jnp.where(jnp.isfinite(robot.joints.upper_limits), robot.joints.upper_limits, jnp.pi)
    qs = jax.random.uniform(jax.random.PRNGKey(0), (n_probe, lo.shape[0]), minval=lo, maxval=hi)
    d_self = robot_coll.compute_self_collision_distance(robot, qs)
    self_mask = jnp.any(jnp.abs(d_self - d_self[:1]) > 1e-6, axis=0)
    world_masks = []
    for wg in world_geoms:
        d = robot_coll.compute_world_collision_distance(robot, qs, wg)  # (n_probe, N, M)
        world_masks.append(jnp.any(jnp.abs(d - d[:1]) > 1e-6, axis=0))
    return self_mask, tuple(world_masks)


@functools.partial(jax.jit, static_argnames=("opt_cfg", "use_cuda"))
def _dynamics_aware_trajopt_jax(
    init_trajs, start, goal, robot, robot_coll, world_geoms, tau_max, masks, opt_cfg,
    use_cuda,
):
    from ._contact_trajopt import _fd_vel_acc
    from ._sco_optimization import _smoothness_cost

    T, dof = init_trajs.shape[1:]
    lower, upper = robot.joints.lower_limits, robot.joints.upper_limits
    vel_max = robot.joints.velocity_limits
    self_mask, world_masks = masks
    cfg = opt_cfg
    K = cfg.collision_substeps

    # Endpoints are pinned by construction: only interior waypoints are decision
    # variables, so every term still sees the start/goal transitions.
    def full(x):
        return jnp.concatenate([start[None], x.reshape(T - 2, dof), goal[None]])

    def cost_fn(x):
        return _smoothness_cost(full(x), cfg.w_vel, cfg.w_acc, cfg.w_jerk)

    def path_points(t):
        a = (jnp.arange(K, dtype=t.dtype) / K)[:, None, None]
        return ((1 - a) * t[None, :-1] + a * t[None, 1:]).reshape(-1, dof)

    # Every robot link / self pair separately, not a smooth-min per group: a
    # reduced clearance lets one bad entry hide the rest.
    def clearance(q):
        parts = [jnp.where(self_mask, robot_coll.compute_self_collision_distance(robot, q), jnp.inf)]
        for wg, m in zip(world_geoms, world_masks):
            d = robot_coll.compute_world_collision_distance(robot, q, wg)
            parts.append(jnp.where(m, d, jnp.inf).reshape(-1))
        return jnp.concatenate(parts)

    def collision_residual(x):
        d = jax.vmap(clearance)(path_points(full(x)))
        return jnp.where(jnp.isfinite(d), cfg.collision_margin - d, -1.0)

    def limit_residual(x):
        t = full(x)[1:-1]
        b = cfg.limit_backoff
        return jnp.concatenate([t - (upper - b), (lower + b) - t])

    def velocity_residual(x):
        t = full(x)
        ok = jnp.isfinite(vel_max) & (vel_max > 0)  # URDFs often leave it 0 / unset
        v = jnp.abs(t[1:] - t[:-1]) / (cfg.dt * jnp.where(ok, vel_max, 1.0))
        return jnp.where(ok, v - (1.0 - cfg.limit_backoff), -1.0)

    def torque_residual(x):
        t = full(x)
        qd, qdd = _fd_vel_acc(t, cfg.dt)
        tau = robot.inverse_dynamics(t, qd, qdd, use_cuda=use_cuda)
        # Dimensionless (fraction over the limit), so the torque term does not
        # swamp the metre-scale collision term in the shared AL/trust-region loop.
        return jnp.abs(tau) / tau_max - (1.0 - cfg.limit_backoff)

    ineq = lambda fn, name, rho0=1.0, rho_max=1e3, scale=2.0: AugmentedLagrangianTerm(
        fn, kind="ineq", rho0=rho0, rho_max=rho_max, penalty_scale=scale, name=name)
    path_terms = (
        ineq(collision_residual, "collision", cfg.w_collision, cfg.w_collision_max,
             cfg.penalty_scale),
        ineq(limit_residual, "joint_limit"),
    ) + tuple(ineq(lambda x, f=f: f(full(x)), f"extra_{i}")
              for i, f in enumerate(cfg.extra_constraints))
    terms = path_terms + (
        ineq(velocity_residual, "velocity_limit"),
        ineq(torque_residual, "torque_limit"),
    )
    solver = dataclasses.replace(cfg.solver, robot=None, grid=None, constraints=terms)
    path_solver = dataclasses.replace(solver, constraints=path_terms)
    s = cfg.path_smooth_scale
    path_cost = lambda x: _smoothness_cost(full(x), s * cfg.w_vel, s * cfg.w_acc, s * cfg.w_jerk)

    def solve_one(traj):
        x = traj[1:-1].reshape(-1)
        if cfg.path_stage:
            x = dynamics_trajopt(x, path_cost, path_solver)
        return full(dynamics_trajopt(x, cost_fn, solver))

    trajs = init_trajs.at[:, 0].set(start).at[:, -1].set(goal)
    final_trajs = jax.vmap(solve_one)(trajs)

    # Rank by total constraint violation first, smoothness second.
    def violation(t):
        x = t[1:-1].reshape(-1)
        return sum(jnp.sum(jnp.maximum(term.residual_fn(x), 0.0)) for term in terms)

    viol = jax.vmap(violation)(final_trajs)
    costs = cfg.w_collision_max * viol + jax.vmap(lambda t: cost_fn(t[1:-1].reshape(-1)))(final_trajs)
    return final_trajs[jnp.argmin(costs)], costs, final_trajs


def dynamics_aware_trajopt(
    init_trajs: Float[Array, "B T DOF"],
    start: Float[Array, "DOF"],
    goal: Float[Array, "DOF"],
    robot,
    robot_coll,
    world_geoms: tuple,
    opt_cfg: DynamicsAwareTrajOptConfig = DynamicsAwareTrajOptConfig(),
    *,
    use_cuda: bool = False,
) -> tuple[Float[Array, "T DOF"], Float[Array, "B"], Float[Array, "B T DOF"]]:
    """Collision-free, limit-respecting, torque-feasible trajopt (tier 1).

    Each seed is solved by :func:`dynamics_trajopt` as an augmented-Lagrangian
    SCO loop over a smoothness cost (after a collision-only path stage, see
    ``path_stage``), with inequality constraints for
    collision clearance along the interpolated path (``collision_substeps``
    per segment), joint limits, joint velocity limits, and ``|tau| <= tau_max``
    (inverse dynamics on central-difference velocities/accelerations at
    ``opt_cfg.dt``). Collision terms that cannot change with the configuration
    (fixed links, rigid pairs) are dropped.

    If ``n_timesteps * dt`` is too short for any feasible motion, the
    least-violating trajectory is returned; ``costs`` ranks violation first.

    Same signature and returns as :func:`sco_trajopt`. ``use_cuda`` selects the
    GRiD inverse-dynamics kernels; otherwise dynamics run in pure JAX.
    """
    if robot.floating_base:
        raise NotImplementedError(
            "Dynamics-aware trajopt does not support floating-base robots yet; "
            "pass a ScoTrajOptConfig, ChompTrajOptConfig or StompTrajOptConfig "
            "for kinematics-only trajopt."
        )
    if robot.dynamics is None:
        raise ValueError(
            "This robot has no dynamics model (URDF without inertials or with an "
            "unsupported joint type), so torque limits cannot be checked. "
            "Pass a ScoTrajOptConfig, ChompTrajOptConfig or StompTrajOptConfig "
            "for kinematics-only trajopt."
        )
    tau_max = jnp.asarray(opt_cfg.tau_max) if opt_cfg.tau_max else effort_limits(robot)
    masks = _changing_collision_terms(robot, robot_coll, world_geoms)
    return _dynamics_aware_trajopt_jax(
        init_trajs, start, goal, robot, robot_coll, world_geoms, tau_max, masks,
        opt_cfg, use_cuda,
    )
