"""
Solves the basic IK problem with collision avoidance.

Uses pyroffi's native multi-seed Levenberg-Marquardt IK solver
(:func:`pyroffi.optimization_engines.ls_ik_solve`) rather than an external
least-squares backend.  The end-effector is held at its seed pose (a trivial
task residual) while self- and world-collision penalties push the
configuration out of collision, staying close to the seed via LM continuity.
"""

from typing import Sequence

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import numpy as onp
import pyroffi as pk
from pyroffi.collision import colldist_from_sdf
from pyroffi.optimization_engines import ls_ik_solve


def solve_collision_with_config(
    robot: pk.Robot,
    coll: pk.collision.RobotCollision,
    world_coll_list: Sequence[pk.collision.CollGeom],
    cfg: onp.ndarray,
) -> onp.ndarray:
    """
    Solves the basic IK problem for a robot.

    Args:
        robot: PyRoFFI Robot.
        coll: Robot collision model.
        world_coll_list: World collision geometries to avoid.
        cfg: Seed configuration. Shape: (robot.joints.num_actuated_joints,).

    Returns:
        cfg: ArrayLike. Shape: (robot.joint.actuated_count,).
    """
    assert cfg.shape == (robot.joints.num_actuated_joints,)

    cfg = _solve_collision_with_config_jax(
        robot,
        coll,
        tuple(world_coll_list),
        jnp.asarray(cfg),
    )
    assert cfg.shape == (robot.joints.num_actuated_joints,)

    return onp.array(cfg)


@jdc.jit
def _solve_collision_with_config_jax(
    robot: pk.Robot,
    coll: pk.collision.RobotCollision,
    world_coll_list: tuple[pk.collision.CollGeom, ...],
    cfg: jax.Array,
) -> jax.Array:
    """Solves the basic IK problem with collision avoidance. Returns joint configuration."""
    # Hold the distal link at its seed pose so the IK task is trivially
    # satisfied and the solver only moves to resolve collisions.
    ee_link = robot.links.num_links - 1
    target_pose = jaxlie.SE3(robot.forward_kinematics(cfg)[ee_link])

    # Collision penalties expressed as ls_ik constraints:
    # ``c(cfg, robot, args) -> scalar``, 0 when clear, positive when violated.
    def self_collision_c(q, robot, margin):
        dist = coll.compute_self_collision_distance(robot, q)
        return jnp.sqrt(jnp.sum(colldist_from_sdf(dist, margin) ** 2))

    def world_collision_c(q, robot, args):
        world_geom, margin = args
        dist = coll.compute_world_collision_distance(robot, q, world_geom)
        return jnp.sqrt(jnp.sum(colldist_from_sdf(dist, margin) ** 2))

    constraint_fns = (self_collision_c,) + tuple(
        world_collision_c for _ in world_coll_list
    )
    constraint_args = (0.02,) + tuple(
        (world_coll, 0.05) for world_coll in world_coll_list
    )
    constraint_weights = jnp.array([5.0] + [11.0] * len(world_coll_list))

    return ls_ik_solve(
        robot,
        target_link_indices=(ee_link,),
        target_poses=(target_pose,),
        rng_key=jax.random.PRNGKey(0),
        previous_cfg=cfg,
        num_seeds=16,
        continuity_weight=10.0,
        constraint_fns=constraint_fns,
        constraint_args=constraint_args,
        constraint_weights=constraint_weights,
    )
