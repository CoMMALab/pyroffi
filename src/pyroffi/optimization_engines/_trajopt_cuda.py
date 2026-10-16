"""Trajectory optimization on the GPU: GRiM's SCO, STOMP, CHOMP and LS-trajopt kernels.

Each kernel is compiled per robot and per collision structure (sphere rows, self-collision
pairs, obstacle counts) on first use and cached on disk by GRiM. This module adapts pyroffi's
inputs -- a ``Robot``, a ``RobotCollisionSpherized`` model, world geometry and the engines'
config dataclasses -- and returns ``(best_traj, costs, final_trajs)`` like the JAX engines.

Requires a ``RobotCollisionSpherized`` collision model.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .. import _grim


class TrajoptPrepared(NamedTuple):
    """Everything about the scene GRiM bakes into a build. Built eagerly (numpy), so
    :func:`run` can be called inside ``jax.jit``; reuse one per scene."""
    mrobot: object
    collision: object        # grim.motion._build.TrajCollision
    world: object            # grim.motion.ik.World
    traced: bool
    rot_joint: int = -1      # a floating base's chart: its reference rotation, per call
    rot: object = None


def _world_arrays(world_geoms):
    from ..collision._cuda_collision import _extract_world_arrays

    items = world_geoms if isinstance(world_geoms, (list, tuple)) else (world_geoms,)
    chunks = ([], [], [], [])
    for w in items:
        for dst, arr in zip(chunks, _extract_world_arrays(w)):
            if arr.shape[0] > 0:
                dst.append(np.asarray(arr, np.float32))
    return tuple(np.concatenate(c) if c else None for c in chunks)


def prepare(robot, robot_coll, world_geoms, traced: bool = False) -> TrajoptPrepared:
    """Flatten the robot's spheres and the world for the kernels. Call OUTSIDE ``jax.jit``.

    The kernels pose sphere row ``r`` with joint ``r``'s world frame, or the identity for
    ``r == n_joints`` (a root link fixed in the world); rows no link maps to are padding
    (radius < 0). ``traced=True`` builds against cricket's straight-line FK.
    """
    from ..collision._cuda_collision import link_parent_joint_for
    from ..collision._robot_collision import RobotCollisionSpherized

    if not isinstance(robot_coll, RobotCollisionSpherized):
        raise TypeError("the CUDA trajectory optimizers require a RobotCollisionSpherized "
                        f"collision model, got {type(robot_coll).__name__}")
    _grim.ensure_importable()
    from grim.motion._build import TrajCollision
    from grim.motion.ik import World

    sphere_off = np.asarray(robot_coll.coll.pose.translation(), np.float32)
    sphere_rad = np.asarray(robot_coll.coll.radius, np.float32)
    n_joints = int(robot.joints.num_joints)
    row = np.asarray(link_parent_joint_for(robot, robot_coll), np.int32)
    row = np.where(row < 0, n_joints, row)
    off = np.zeros((n_joints + 1, sphere_rad.shape[1], 3), np.float32)
    rad = np.full((n_joints + 1, sphere_rad.shape[1]), -1.0, np.float32)
    off[row], rad[row] = sphere_off, sphere_rad
    collision = TrajCollision(off, rad,
                              row[np.asarray(robot_coll.active_idx_i)].astype(np.int32),
                              row[np.asarray(robot_coll.active_idx_j)].astype(np.int32))
    from .._floating_base import chart_rotation

    return TrajoptPrepared(_grim.motion_robot(robot), collision,
                           World(*_world_arrays(world_geoms)), traced, *chart_rotation(robot))


def _check_endpoints(start, goal, B, n):
    for name, arr in (("start", start), ("goal", goal)):
        if jnp.shape(arr) not in ((n,), (B, n)):
            raise ValueError(f"{name} must be [n_act] or [B, n_act]; got {jnp.shape(arr)} "
                             f"with B={B}, n_act={n}.")


def run(kind: str, prep: TrajoptPrepared, init_trajs, start, goal, opt_cfg, *,
        rng_seed: int = 0, fd_eps: float = 1e-4):
    """One GRiM optimizer (``"sco"``, ``"stomp"``, ``"chomp"`` or ``"ls"``) -> ``(best_traj,
    costs, final_trajs)``. ``start``/``goal`` are ``[n_act]`` or per trajectory ``[B, n_act]``;
    with the latter ``best_traj`` compares different problems and is meaningless."""
    from grim.motion import trajopt

    B, _, n = init_trajs.shape
    _check_endpoints(start, goal, B, n)
    c = opt_cfg
    if kind == "sco":
        fn, kw = trajopt.sco_trajopt, dict(
            n_outer_iters=c.n_outer_iters, n_inner_iters=c.n_inner_iters, m_lbfgs=c.m_lbfgs,
            w_trust=c.w_trust, penalty_scale=c.penalty_scale,
            smooth_min_temperature=c.smooth_min_temperature, fd_eps=fd_eps)
    elif kind == "stomp":
        fn, kw = trajopt.stomp_trajopt, dict(
            n_iters=c.n_iters, n_samples=c.n_samples, noise_scale=c.noise_scale,
            temperature=c.temperature, step_size=c.step_size,
            collision_penalty_scale=c.collision_penalty_scale, rng_seed=rng_seed)
    elif kind == "chomp":
        fn, kw = trajopt.chomp_trajopt, dict(
            n_iters=c.n_iters, step_size=c.step_size,
            collision_penalty_scale=c.collision_penalty_scale,
            use_covariant_update=c.use_covariant_update, smoothness_reg=c.smoothness_reg,
            grad_clip_norm=c.grad_clip_norm, max_delta_per_step=c.max_delta_per_step,
            early_stop_patience=c.early_stop_patience, min_cost_improve=c.min_cost_improve,
            fd_eps=fd_eps)
    else:
        fn, kw = trajopt.ls_trajopt, dict(
            n_outer_iters=c.n_outer_iters, n_ls_iters=c.n_ls_iters, lambda_init=c.lambda_init,
            w_trust=c.w_trust, w_endpoint=c.w_endpoint, penalty_scale=c.penalty_scale,
            smooth_min_temperature=c.smooth_min_temperature,
            max_delta_per_step=c.max_delta_per_step, fd_eps=fd_eps)
    trajs, costs = fn(prep.mrobot, init_trajs, collision=prep.collision, world=prep.world,
                      start=start, goal=goal, w_smooth=c.w_smooth, w_acc=c.w_acc,
                      w_jerk=c.w_jerk, w_limits=c.w_limits, w_collision=c.w_collision,
                      w_collision_max=c.w_collision_max, collision_margin=c.collision_margin,
                      traced=prep.traced, runtime_rot_joint=prep.rot_joint, rot=prep.rot,
                      **kw)
    return trajs[jnp.argmin(costs)], costs, trajs


def trajopt_cuda(kind: str, init_trajs, start, goal, robot, robot_coll, world_geoms, opt_cfg,
                 *, key=None, fd_eps: float = 1e-4, traced: bool = False):
    """:func:`prepare` then :func:`run`. STOMP draws its noise seed from ``key``."""
    key = jax.random.PRNGKey(0) if key is None else key
    rng_seed = int(jax.random.randint(key, (), 0, 2**31 - 1))
    return run(kind, prepare(robot, robot_coll, world_geoms, traced), init_trajs, start, goal,
               opt_cfg, rng_seed=rng_seed, fd_eps=fd_eps)
