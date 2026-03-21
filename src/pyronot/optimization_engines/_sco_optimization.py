"""SCO-style Batched Trajectory Optimization.

JAX-native implementation of a batched trajectory optimization (TrajOpt) pipeline
loosely based on curobo's accelerated TrajOpt.  All B candidate trajectories are
optimized in parallel using ``jax.vmap`` and ``jax.lax.scan``.

Pipeline
--------
1. Initialize trajectories  [B, T, DOF]
2. Run batched gradient descent  (lax.scan over N_iters steps)
3. Evaluate costs  [B]
4. Return best trajectory + all costs

Cost structure
--------------
  J = w_smooth     * J_smooth        (velocity + acceleration + jerk)
    + w_collision  * J_collision     (self + world, via colldist_from_sdf)
    + w_goal       * J_goal          (final-state distance to goal)
    + w_limits     * J_limits        (soft joint-limit penalty)

Optimization
------------
Gradient descent with optional parallel line-search over ``_LS_ALPHAS``.
The entire pipeline is wrapped in ``jax.jit`` and operates on static shapes,
making it fully JIT-compilable and XLA-optimizable.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import jaxlie
import optax
from jax import Array
from jaxtyping import Float

from .._robot import Robot
from ..collision import CollGeom, RobotCollision, colldist_from_sdf

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrajOptConfig:
    """Hyper-parameters for the SCO TrajOpt solver."""

    # --- Iteration budget ---
    n_iters: int = 100
    """Number of Adam iterations."""

    # --- Adam optimizer ---
    lr: float = 0.01
    """Adam learning rate."""

    adam_b1: float = 0.9
    """Adam beta1 (first-moment decay)."""

    adam_b2: float = 0.999
    """Adam beta2 (second-moment decay)."""

    adam_eps: float = 1e-8
    """Adam epsilon for numerical stability."""

    # --- Cost weights ---
    w_smooth: float = 1.0
    """Weight for the smoothness cost (vel + acc + jerk)."""

    w_vel: float = 1.0
    """Relative weight of velocity within smoothness."""

    w_acc: float = 0.5
    """Relative weight of acceleration within smoothness."""

    w_jerk: float = 0.1
    """Relative weight of jerk within smoothness."""

    w_collision: float = 10.0
    """Weight for the collision cost."""

    collision_margin: float = 0.01
    """Activation distance (margin) for colldist_from_sdf."""

    w_limits: float = 1.0
    """Weight for joint-limit violation penalty."""


# ---------------------------------------------------------------------------
# Individual cost components  (operate on a single trajectory [T, DOF])
# ---------------------------------------------------------------------------

def _smoothness_cost(
    traj: Float[Array, "T DOF"],
    w_vel: float,
    w_acc: float,
    w_jerk: float,
) -> Array:
    """Smoothness cost using 4th-order central-difference acceleration.

    Velocity and acceleration terms are replaced by the 4th-order central-difference
    stencil for acceleration (valid for interior points t in [2, T-3]):

        a[t] = -q[t-2] + 16·q[t-1] - 30·q[t] + 16·q[t+1] - q[t+2]

    Jerk is the finite difference of consecutive central-difference accelerations.
    w_vel is unused (no separate velocity term).
    """
    # 4th-order central-difference acceleration  [T-4, DOF]
    # Stencil: (-q[t-2] + 16q[t-1] - 30q[t] + 16q[t+1] - q[t+2]) / (12h²)
    # Source: Fornberg, B. (1988). "Generation of finite difference formulas on
    #   arbitrarily spaced grids." Mathematics of Computation, 51(184), 699–706.
    #   https://doi.org/10.1090/S0025-5718-1988-0935077-0  (Table 1, d=2, order=4)
    # Divided by 12 to match the standard f''(x) ≈ stencil/(12h²) scaling (h=1),
    # keeping gradient magnitudes comparable to the previous 2nd-order stencil.
    acc = (
        -      traj[:-4]
        + 16.0 * traj[1:-3]
        - 30.0 * traj[2:-2]
        + 16.0 * traj[3:-1]
        -      traj[4:]
    ) / 12.0
    jerk = acc[1:] - acc[:-1]                   # [T-5, DOF]

    cost  = w_acc  * jnp.sum(acc  ** 2)
    cost += w_jerk * jnp.sum(jerk ** 2)
    return cost



def _limits_cost(
    traj: Float[Array, "T DOF"],
    lower: Float[Array, "DOF"],
    upper: Float[Array, "DOF"],
) -> Array:
    """Soft penalty for joint-limit violations (sum of squared exceedances)."""
    viol_upper = jnp.maximum(0.0, traj - upper)   # [T, DOF]
    viol_lower = jnp.maximum(0.0, lower - traj)   # [T, DOF]
    return jnp.sum((viol_upper + viol_lower) ** 2)


def _collision_cost_single_cfg(
    cfg: Float[Array, "DOF"],
    robot: Robot,
    robot_coll: RobotCollision,
    world_geoms: tuple,
    margin: float,
) -> Array:
    """Collision cost for a single configuration (self + all world geometry types)."""
    cost = jnp.zeros(())

    # Self-collision
    self_dists = robot_coll.compute_self_collision_distance(robot, cfg)
    cost += jnp.sum(-jnp.minimum(colldist_from_sdf(self_dists, margin), 0.0))

    # World collision — one geometry type at a time (Python loop, unrolled at trace time)
    for world_geom in world_geoms:
        world_dists = robot_coll.compute_world_collision_distance(
            robot, cfg, world_geom
        )
        cost += jnp.sum(-jnp.minimum(colldist_from_sdf(world_dists, margin), 0.0))

    return cost


def _collision_cost_traj(
    traj: Float[Array, "T DOF"],
    robot: Robot,
    robot_coll: RobotCollision,
    world_geoms: tuple,
    margin: float,
) -> Array:
    """Sum of collision costs over all T timesteps."""
    per_step = jax.vmap(
        _collision_cost_single_cfg,
        in_axes=(0, None, None, None, None),
    )(traj, robot, robot_coll, world_geoms, margin)
    return jnp.sum(per_step)


# ---------------------------------------------------------------------------
# Combined cost  (single trajectory)
# ---------------------------------------------------------------------------

def _total_cost(
    traj: Float[Array, "T DOF"],
    lower: Float[Array, "DOF"],
    upper: Float[Array, "DOF"],
    robot: Robot,
    robot_coll: RobotCollision,
    world_geoms: tuple,
    cfg: TrajOptConfig,
) -> Array:
    """Scalar cost for a single trajectory.

    Start and goal waypoints are pinned externally via gradient masking,
    so no endpoint cost terms are needed here.
    """
    cost = jnp.zeros(())

    cost += cfg.w_smooth * _smoothness_cost(
        traj, cfg.w_vel, cfg.w_acc, cfg.w_jerk
    )
    cost += cfg.w_limits * _limits_cost(traj, lower, upper)
    cost += cfg.w_collision * _collision_cost_traj(
        traj, robot, robot_coll, world_geoms, cfg.collision_margin
    )

    return cost


# ---------------------------------------------------------------------------
# Batched cost + grad  (over B trajectories)
# ---------------------------------------------------------------------------

def _make_batched_fns(
    lower: Float[Array, "DOF"],
    upper: Float[Array, "DOF"],
    robot: Robot,
    robot_coll: RobotCollision,
    world_geoms: tuple,
    opt_cfg: TrajOptConfig,
):
    """Return (batched_cost_fn, batched_val_grad_fn) closed over all static data."""

    def single_cost(traj):
        return _total_cost(
            traj, lower, upper, robot, robot_coll, world_geoms, opt_cfg
        )

    # [B, T, DOF] -> [B]
    batched_cost_fn = jax.vmap(single_cost, in_axes=0)
    # [B, T, DOF] -> ([B], [B, T, DOF])  — one forward+backward pass for both
    batched_val_grad_fn = jax.vmap(jax.value_and_grad(single_cost), in_axes=0)
    return batched_cost_fn, batched_val_grad_fn


# ---------------------------------------------------------------------------
# Optimization step
# ---------------------------------------------------------------------------

def _make_step_fn(batched_val_grad_fn, tx):
    """Return a single lax.scan-compatible step function for Adam.

    Endpoints (traj[0] and traj[-1]) are pinned by zeroing their gradients
    before the optimizer update, so start and goal never move.
    """

    def step(carry, _):
        trajs, opt_state = carry          # [B, T, DOF], optax state
        _, grads = batched_val_grad_fn(trajs)   # [B, T, DOF]

        # Pin start and goal: zero out gradients at first and last waypoints
        grads = grads.at[:, 0, :].set(0.0)
        grads = grads.at[:, -1, :].set(0.0)

        updates, new_opt_state = tx.update(grads, opt_state, trajs)
        new_trajs = optax.apply_updates(trajs, updates)

        return (new_trajs, new_opt_state), None

    return step


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

@functools.partial(
    jax.jit,
    static_argnames=("opt_cfg",),
)
def sco_trajopt(
    init_trajs: Float[Array, "B T DOF"],
    start: Float[Array, "DOF"],
    goal: Float[Array, "DOF"],
    robot: Robot,
    robot_coll: RobotCollision,
    world_geoms: tuple,
    opt_cfg: TrajOptConfig = TrajOptConfig(),
) -> tuple[Float[Array, "T DOF"], Float[Array, "B"], Float[Array, "B T DOF"]]:
    """Batched SCO-style trajectory optimization using Adam.

    Start and goal waypoints are pinned: ``init_trajs[:, 0, :]`` and
    ``init_trajs[:, -1, :]`` are never modified by the optimizer.

    Args:
        init_trajs:  Initial trajectory batch.  Shape ``[B, T, DOF]``.
        start:       Start joint configuration (used to pin ``traj[:, 0, :]``).
        goal:        Goal joint configuration (used to pin ``traj[:, -1, :]``).
        robot:       Robot kinematics pytree.
        robot_coll:  Robot collision model pytree.
        world_geoms: Tuple of world collision geometry objects (one per obstacle type).
                     Pass an empty tuple ``()`` to skip world collisions.
        opt_cfg:     Hyper-parameters (static — changing them triggers recompilation).

    Returns:
        best_traj:   Optimized trajectory with lowest final cost.  Shape ``[T, DOF]``.
        costs:       Final cost per trajectory in the batch.  Shape ``[B]``.
        final_trajs: All optimized trajectories.  Shape ``[B, T, DOF]``.
    """
    lower = robot.joints.lower_limits   # [DOF]
    upper = robot.joints.upper_limits   # [DOF]

    # Pin endpoints in init_trajs so every trajectory starts and ends correctly
    init_trajs = init_trajs.at[:, 0, :].set(start)
    init_trajs = init_trajs.at[:, -1, :].set(goal)

    batched_cost_fn, batched_val_grad_fn = _make_batched_fns(
        lower, upper, robot, robot_coll, world_geoms, opt_cfg
    )

    tx = optax.adam(opt_cfg.lr, opt_cfg.adam_b1, opt_cfg.adam_b2, opt_cfg.adam_eps)
    opt_state = tx.init(init_trajs)

    step_fn = _make_step_fn(batched_val_grad_fn, tx)

    (final_trajs, _), _ = jax.lax.scan(
        step_fn, (init_trajs, opt_state), None, length=opt_cfg.n_iters
    )

    costs = batched_cost_fn(final_trajs)       # [B]
    best_idx = jnp.argmin(costs)
    best_traj = final_trajs[best_idx]          # [T, DOF]

    return best_traj, costs, final_trajs


# ---------------------------------------------------------------------------
# Convenience: initialise trajectory batch by interpolation
# ---------------------------------------------------------------------------

def make_init_trajs(
    start: Float[Array, "DOF"],
    goal: Float[Array, "DOF"],
    n_batch: int,
    n_timesteps: int,
    key: Array,
    noise_scale: float = 0.05,
) -> Float[Array, "B T DOF"]:
    """Create a batch of linearly-interpolated trajectories with small random noise.

    Args:
        start:        Start joint configuration.
        goal:         Goal joint configuration.
        n_batch:      Number of candidate trajectories.
        n_timesteps:  Number of waypoints (including start and end).
        key:          JAX PRNG key.
        noise_scale:  Standard deviation of additive Gaussian noise.

    Returns:
        Trajectory batch of shape ``[B, T, DOF]``.
    """
    # Linear interpolation: [T, DOF]
    t = jnp.linspace(0.0, 1.0, n_timesteps)[:, None]   # [T, 1]
    base = start[None, :] * (1.0 - t) + goal[None, :] * t  # [T, DOF]

    # Tile across batch and add noise
    trajs = jnp.broadcast_to(base[None], (n_batch, n_timesteps, start.shape[0]))
    noise = jax.random.normal(key, trajs.shape) * noise_scale
    return trajs + noise
