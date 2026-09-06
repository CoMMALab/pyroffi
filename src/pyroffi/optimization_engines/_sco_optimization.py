"""SCO TrajOpt: Sequential Convex Optimization for trajectory planning.

SCO pipeline (Schulman et al. 2013, "Finding Locally Optimal,
Collision-Free Trajectories with Sequential Convex Optimization"):

  Outer loop (n_outer_iters):
    1. Linearize collision constraints at the current trajectory q_k:
           d_lin(q_t) = d(q_k_t) + J_d(q_k_t) @ (q_t - q_k_t)
       Jacobians are computed once per outer iteration via jax.jacobian.
    2. Solve the inner convex subproblem with L-BFGS (n_inner_iters steps):
           min  w_smooth  * J_smooth(q)               [exact quadratic]
              + w_coll    * Σ max(0, margin-d_lin(q))² [convex hinge]
              + w_trust   * ||q - q_k||²               [trust region]
              + w_limits  * J_limits(q)                [quadratic]
       Start/goal endpoints are pinned via gradient masking.
    3. Set q_k ← solution.
    4. Scale w_coll by penalty_scale (penalty continuation).

The key distinction from plain gradient descent is that the non-convex
collision distances are *linearized* at each outer iterate and the
resulting convex subproblem is solved to near-optimality with L-BFGS,
rather than taking a single gradient step on the nonlinear objective.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from .._robot import Robot
from ..collision import RobotCollision, colldist_from_sdf
from ._trajopt_core import (
    _LS_ALPHAS,
    _al_outer_loop,
    _lbfgs_driver,
    _lbfgs_two_loop,
    AugmentedLagrangianTerm,
    TrustRegionConfig,
)

# ``_LS_ALPHAS`` / ``_lbfgs_two_loop`` now live in :mod:`_trajopt_core`; they are
# re-exported here because the legacy :mod:`_contact_trajopt` still imports them
# from this module. This module's own inner solve is a thin ``_lbfgs_driver``
# call.
_ = (_LS_ALPHAS, _lbfgs_two_loop)  # keep re-exports live for _contact_trajopt


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoTrajOptConfig:
    """Hyper-parameters for the SCO TrajOpt solver."""

    # --- Outer SCO loop ---
    n_outer_iters:   int   = 10
    """Number of linearize-and-solve outer iterations."""

    # --- Inner L-BFGS solver ---
    n_inner_iters:   int   = 30
    """L-BFGS steps per outer iteration."""

    m_lbfgs:         int   = 6
    """L-BFGS history size (number of curvature pairs)."""

    # --- Smoothness cost weights ---
    w_smooth:        float = 1.0
    """Overall smoothness weight."""

    w_vel:           float = 1.0
    """Unused; kept for API compatibility."""

    w_acc:           float = 0.5
    """Relative weight of acceleration within smoothness."""

    w_jerk:          float = 0.1
    """Relative weight of jerk within smoothness."""

    # --- Collision ---
    w_collision:     float = 1.0
    """Initial collision penalty weight (increased each outer iteration)."""

    w_collision_max: float = 100.0
    """Maximum collision penalty weight after continuation."""

    penalty_scale:   float = 3.0
    """Multiplicative increase in w_collision per outer iteration."""

    collision_margin: float = 0.01
    """Activation margin for the collision cost (metres)."""

    # --- Trust region ---
    w_trust:         float = 0.5
    """Penalty weight for deviating from the linearization point. With
    ``adaptive_trust`` this is the *initial* trust coefficient, then resized by
    the ratio test."""

    adaptive_trust:  bool  = False
    """Schulman-et-al. adaptive trust-region sizing. When True, the trust
    coefficient is grown/shrunk each outer iteration by the actual-vs-predicted
    merit-improvement ratio, and a poorly-predicted step is rejected instead of
    accepted. Default False keeps the fixed-weight trust region (byte-identical
    to the previous SCO behavior)."""
    tr_tighten:      float = 4.0
    """adaptive_trust only: multiplier (>1) on the trust coef for a rejected step."""
    tr_loosen:       float = 0.25
    """adaptive_trust only: multiplier (<1) on the trust coef for a good step."""
    tr_shrink_ratio: float = 0.25
    tr_expand_ratio: float = 0.75
    tr_accept_ratio: float = 0.1
    tr_coef_min:     float = 1e-2
    tr_coef_max:     float = 1e4

    # --- Joint limits ---
    w_limits:        float = 1.0
    """Weight for the soft joint-limit violation penalty."""

    # --- Collision dimensionality reduction ---
    smooth_min_temperature: float = 0.05
    """Temperature for the per-group smooth-minimum aggregation.

    Instead of keeping all P raw distances (potentially hundreds), one
    smooth-minimum scalar is computed per collision group (self-collision +
    one per world-geometry type).  This reduces the Jacobian from [P, DOF]
    to [n_groups, DOF], cutting Jacobian memory and compile time by ~50-100x.

    Smaller temperature → closer to the true minimum but steeper gradients.
    """


# Backward-compatible alias.
TrajOptConfig = ScoTrajOptConfig


# ---------------------------------------------------------------------------
# Cost components
# ---------------------------------------------------------------------------

def _smoothness_cost(
    traj:   Float[Array, "T DOF"],
    w_vel:  float,
    w_acc:  float,
    w_jerk: float,
) -> Array:
    """Velocity + 4th-order central-difference acceleration + jerk smoothness cost."""
    vel = traj[1:] - traj[:-1]
    acc = (
        -      traj[:-4]
        + 16.0 * traj[1:-3]
        - 30.0 * traj[2:-2]
        + 16.0 * traj[3:-1]
        -      traj[4:]
    ) / 12.0
    jerk  = acc[1:] - acc[:-1]
    cost  = w_vel  * jnp.sum(vel  ** 2)
    cost += w_acc  * jnp.sum(acc  ** 2)
    cost += w_jerk * jnp.sum(jerk ** 2)
    return cost


def _limits_cost(
    traj:  Float[Array, "T DOF"],
    lower: Float[Array, "DOF"],
    upper: Float[Array, "DOF"],
) -> Array:
    """Squared exceedance penalty for joint-limit violations."""
    viol_upper = jnp.maximum(0.0, traj - upper)
    viol_lower = jnp.maximum(0.0, lower - traj)
    return jnp.sum((viol_upper + viol_lower) ** 2)


def _collision_distances_all(
    cfg:        Float[Array, "DOF"],
    robot:      Robot,
    robot_coll: RobotCollision,
    world_geoms: tuple,
) -> Float[Array, "P"]:
    """Flat concatenation of all self + world collision distances.

    Each array is ravelled to 1-D first because ``compute_world_collision_distance``
    can return varying-rank tensors depending on geometry type.
    Used only in the final nonlinear evaluation cost.
    """
    self_dists = robot_coll.compute_self_collision_distance(robot, cfg)
    parts = [jnp.ravel(self_dists)]
    for wg in world_geoms:
        parts.append(jnp.ravel(
            robot_coll.compute_world_collision_distance(robot, cfg, wg)
        ))
    return jnp.concatenate(parts)


def _collision_dists_reduced(
    cfg:         Float[Array, "DOF"],
    robot:       Robot,
    robot_coll:  RobotCollision,
    world_geoms: tuple,
    temperature: float,
) -> Float[Array, "G"]:
    """Per-group smooth-minimum collision distances.

    Returns one scalar per collision group (self-collision + one per world
    geometry type), computed as the smooth-minimum over all pair distances in
    that group:

        smooth_min(d) = -temperature * logsumexp(-d / temperature)

    This reduces the Jacobian shape from [P, DOF] (P ≈ 100-300) to
    [G, DOF] (G = 1 + n_world_geoms, typically 3-5), cutting Jacobian
    memory and compile time by 50-100x.
    """
    def smooth_min(d_flat: Array) -> Array:
        return -temperature * jax.scipy.special.logsumexp(-d_flat / temperature)

    self_dists = robot_coll.compute_self_collision_distance(robot, cfg)
    groups = [smooth_min(jnp.ravel(self_dists))]
    for wg in world_geoms:
        dists = robot_coll.compute_world_collision_distance(robot, cfg, wg)
        groups.append(smooth_min(jnp.ravel(dists)))
    return jnp.stack(groups)  # [G]


# ---------------------------------------------------------------------------
# SCO solve for a single trajectory (Schulman et al. 2013)
# ---------------------------------------------------------------------------

def _collision_residual(
    x_flat:      Float[Array, "n"],
    robot:       Robot,
    robot_coll:  RobotCollision,
    world_geoms: tuple,
    cfg:         ScoTrajOptConfig,
    T:           int,
    DOF:         int,
) -> Float[Array, "T*G"]:
    """Per-waypoint, per-group signed collision violation ``margin - d`` as an
    inequality residual (feasible ⇔ ``<= 0``).

    ``d`` is the per-group smooth-minimum clearance (self + one per world
    geometry), so the residual is one scalar per ``(timestep, group)``. The
    augmented-Lagrangian term squares its positive part into the same convex
    hinge ``max(0, margin - d)²`` the old fixed-weight penalty used — but now
    with a dual and a per-term ``rho`` continuation, and *linearized* about the
    outer iterate by :func:`_al_outer_loop`'s ``sco_linearize`` (this is the
    Schulman convexification, via ``jax.linearize`` instead of a hand-built
    Jacobian einsum)."""
    t = x_flat.reshape(T, DOF)

    def per_cfg(c):
        return _collision_dists_reduced(
            c, robot, robot_coll, world_geoms, cfg.smooth_min_temperature
        )

    d = jax.vmap(per_cfg)(t)                       # [T, G]
    return (cfg.collision_margin - d).reshape(-1)  # [T*G]


def _sco_solve_one(
    traj:        Float[Array, "T DOF"],
    start:       Float[Array, "DOF"],
    goal:        Float[Array, "DOF"],
    robot:       Robot,
    robot_coll:  RobotCollision,
    world_geoms: tuple,
    lower:       Float[Array, "DOF"],
    upper:       Float[Array, "DOF"],
    cfg:         ScoTrajOptConfig,
) -> Float[Array, "T DOF"]:
    """One trajectory's SCO solve, routed through the shared AL outer loop.

    Traditional Schulman-et-al. SCO: each outer iteration linearizes the
    non-convex collision constraint at the current iterate (``sco_linearize``),
    then solves a convex subproblem — exact quadratic smoothness/limits, the
    convex linearized-collision hinge, and a trust region ``||q - q_k||²`` about
    that iterate — with L-BFGS. The collision constraint carries a real dual and
    a ``rho`` penalty continuation (the AL upgrade over the old duals-free
    penalty-only loop). Endpoints are pinned every inner step and re-pinned each
    outer step.
    """
    T = traj.shape[0]
    DOF = traj.shape[1]
    n = T * DOF

    endpoint_mask = jnp.ones(n).at[:DOF].set(0.0).at[n - DOF:].set(0.0)

    # With adaptive TR the trust term is owned + resized by `_al_outer_loop`; with
    # the fixed variant it stays a constant penalty here (byte-identical to before).
    if cfg.adaptive_trust:
        trust = TrustRegionConfig(
            coef0=cfg.w_trust, tighten=cfg.tr_tighten, loosen=cfg.tr_loosen,
            shrink_ratio=cfg.tr_shrink_ratio, expand_ratio=cfg.tr_expand_ratio,
            accept_ratio=cfg.tr_accept_ratio,
            coef_min=cfg.tr_coef_min, coef_max=cfg.tr_coef_max,
        )
    else:
        trust = None

    def base_cost(x_flat, x_k):
        t = x_flat.reshape(T, DOF)
        c = cfg.w_smooth * _smoothness_cost(t, cfg.w_vel, cfg.w_acc, cfg.w_jerk)
        c += cfg.w_limits * _limits_cost(t, lower, upper)
        if not cfg.adaptive_trust:
            c += cfg.w_trust * jnp.sum((x_flat - x_k) ** 2)  # fixed SCO trust region
        return c

    coll_term = AugmentedLagrangianTerm(
        residual_fn=lambda x: _collision_residual(
            x, robot, robot_coll, world_geoms, cfg, T, DOF
        ),
        kind="ineq",
        rho0=cfg.w_collision,
        rho_max=cfg.w_collision_max,
        penalty_scale=cfg.penalty_scale,
        name="collision",
    )

    def inner_solve(z0, cost_fn):
        return _lbfgs_driver(
            z0, cost_fn,
            n_iters=cfg.n_inner_iters, m_lbfgs=cfg.m_lbfgs,
            loop="scan", endpoint_mask=endpoint_mask, best_by="cost",
            gd_dir="norm",
        )

    def repin(z):
        t = z.reshape(T, DOF).at[0].set(start).at[-1].set(goal)
        return t.reshape(-1)

    z, _, _ = _al_outer_loop(
        traj.reshape(n), inner_solve, (coll_term,), base_cost,
        n_outer_iters=cfg.n_outer_iters, repin_fn=repin, sco_linearize=True,
        trust=trust,
    )
    return z.reshape(T, DOF)


# ---------------------------------------------------------------------------
# Final nonlinear cost  (used only for ranking at the end)
# ---------------------------------------------------------------------------

def _eval_cost(
    traj:       Float[Array, "T DOF"],
    lower:      Float[Array, "DOF"],
    upper:      Float[Array, "DOF"],
    robot:      Robot,
    robot_coll: RobotCollision,
    world_geoms: tuple,
    cfg:        ScoTrajOptConfig,
) -> Array:
    """Full nonlinear cost at the final w_collision_max weight."""
    cost = cfg.w_smooth * _smoothness_cost(traj, cfg.w_vel, cfg.w_acc, cfg.w_jerk)
    cost += cfg.w_limits * _limits_cost(traj, lower, upper)

    def per_step(c):
        dists = _collision_distances_all(c, robot, robot_coll, world_geoms)
        return jnp.sum(-jnp.minimum(colldist_from_sdf(dists, cfg.collision_margin), 0.0))

    cost += cfg.w_collision_max * jnp.sum(jax.vmap(per_step)(traj))
    return cost


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

@functools.partial(
    jax.jit,
    static_argnames=("opt_cfg",),
)
def _sco_trajopt_jax(
    init_trajs:  Float[Array, "B T DOF"],
    start:       Float[Array, "DOF"],
    goal:        Float[Array, "DOF"],
    robot:       Robot,
    robot_coll:  RobotCollision,
    world_geoms: tuple,
    opt_cfg:     ScoTrajOptConfig = ScoTrajOptConfig(),
) -> tuple[Float[Array, "T DOF"], Float[Array, "B"], Float[Array, "B T DOF"]]:
    lower = robot.joints.lower_limits
    upper = robot.joints.upper_limits

    # Pin endpoints in the initial batch
    trajs = init_trajs.at[:, 0, :].set(start).at[:, -1, :].set(goal)

    # SCO per trajectory, in parallel: each is its own augmented-Lagrangian
    # outer loop (linearize collision at the iterate -> convex subproblem with
    # trust region -> dual ascent + penalty continuation). The per-outer-iter
    # collision linearization that the old hand-built `_compute_coll_dists_and_jacs`
    # produced is now done inside `_al_outer_loop` via `sco_linearize`.
    final_trajs = jax.vmap(
        lambda traj: _sco_solve_one(
            traj, start, goal, robot, robot_coll, world_geoms, lower, upper, opt_cfg
        )
    )(trajs)

    # Rank by full nonlinear cost at the maximum collision weight
    costs = jax.vmap(
        lambda t: _eval_cost(t, lower, upper, robot, robot_coll, world_geoms, opt_cfg)
    )(final_trajs)
    best_idx  = jnp.argmin(costs)
    best_traj = final_trajs[best_idx]

    return best_traj, costs, final_trajs


def sco_trajopt(
    init_trajs:  Float[Array, "B T DOF"],
    start:       Float[Array, "DOF"],
    goal:        Float[Array, "DOF"],
    robot:       Robot,
    robot_coll:  RobotCollision,
    world_geoms: tuple,
    opt_cfg:     ScoTrajOptConfig = ScoTrajOptConfig(),
    *,
    use_cuda:    bool = False,
) -> tuple[Float[Array, "T DOF"], Float[Array, "B"], Float[Array, "B T DOF"]]:
    """True SCO trajectory optimization.

    Outer loop: linearize collision at current trajectory, solve convex
    inner subproblem with L-BFGS, repeat with scaled-up penalty.

    Args:
        init_trajs:  Initial trajectory batch.  Shape [B, T, DOF].
        start:       Start joint configuration.  Shape [DOF].
        goal:        Goal joint configuration.   Shape [DOF].
        robot:       Robot kinematics pytree.
        robot_coll:  Robot collision model pytree.
        world_geoms: Tuple of stacked world collision geometry objects.
        opt_cfg:     Hyper-parameters (static — changes trigger recompilation).
        use_cuda:    If True, run the CUDA kernel instead of JAX (requires
                     the compiled ``_sco_trajopt_cuda_lib.so``).

    Returns:
        best_traj:   Trajectory with lowest final nonlinear cost. [T, DOF].
        costs:       Final nonlinear cost per trajectory.         [B].
        final_trajs: All optimized trajectories.                  [B, T, DOF].

    Note:
        With per-trajectory endpoints, ``best_traj`` compares DIFFERENT problems
        and is meaningless — use ``final_trajs`` / ``costs`` and pick the best
        within each endpoint pair's own slice.
    """
    if use_cuda:
        from ..cuda_kernels.trajopt._sco_trajopt_cuda import sco_trajopt_cuda
        return sco_trajopt_cuda(
            init_trajs, start, goal, robot, robot_coll, world_geoms, opt_cfg
        )
    return _sco_trajopt_jax(
        init_trajs, start, goal, robot, robot_coll, world_geoms, opt_cfg
    )


# ---------------------------------------------------------------------------
# Convenience: initialise trajectory batch by interpolation
# ---------------------------------------------------------------------------

def make_init_trajs(
    start:       Float[Array, "DOF"],
    goal:        Float[Array, "DOF"],
    n_batch:     int,
    n_timesteps: int,
    key:         Array,
    noise_scale: float = 0.05,
) -> Float[Array, "B T DOF"]:
    """Create a batch of linearly-interpolated trajectories with small random noise."""
    t    = jnp.linspace(0.0, 1.0, n_timesteps)[:, None]
    base = start[None, :] * (1.0 - t) + goal[None, :] * t
    trajs = jnp.broadcast_to(base[None], (n_batch, n_timesteps, start.shape[0]))
    noise = jax.random.normal(key, trajs.shape) * noise_scale
    return trajs + noise
