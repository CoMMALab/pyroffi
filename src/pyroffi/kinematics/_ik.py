"""Inverse kinematics functional entry point backing ``Robot.inverse_kinematics``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax_dataclasses as jdc
import jaxlie
from jax import Array
from jax import numpy as jnp
from jaxtyping import Float

if TYPE_CHECKING:
    from .._robot import Robot


def inverse_kinematics(
    robot: Robot,
    target_link_name: jdc.Static[str],
    target_pose: jaxlie.SE3,
    rng_key: Array | None = None,
    previous_cfg: Float[Array, "n_actuated_joints"] | None = None,
    num_seeds: jdc.Static[int] = 32,
    coarse_max_iter: jdc.Static[int] = 20,
    lm_max_iter: jdc.Static[int] = 40,
    epsilon: float = 0.02,
    nu: float = float(jnp.pi / 2),
    lambda_init: float = 5e-3,
    continuity_weight: float = 1e-3,
    fixed_joint_mask: Float[Array, "n_actuated_joints"] | None = None,
) -> Float[Array, "n_actuated_joints"]:
    """Solve inverse kinematics using the HJCD-IK two-phase optimizer.

    See ``Robot.inverse_kinematics`` for the full parameter documentation.

    Returns:
        Best joint configuration found, shape ``(n_actuated_joints,)``.
    """
    from ..optimization_engines._hjcd_ik import hjcd_solve

    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    if previous_cfg is None:
        previous_cfg = (robot.joints.lower_limits + robot.joints.upper_limits) / 2

    target_link_index = robot.links.names.index(target_link_name)
    return hjcd_solve(
        robot=robot,
        target_link_indices=(target_link_index,),
        target_poses=(target_pose,),
        rng_key=rng_key,
        previous_cfg=previous_cfg,
        num_seeds=num_seeds,
        coarse_max_iter=coarse_max_iter,
        lm_max_iter=lm_max_iter,
        epsilon=epsilon,
        nu=nu,
        lambda_init=lambda_init,
        continuity_weight=continuity_weight,
        fixed_joint_mask=fixed_joint_mask,
    )
