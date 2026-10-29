"""Residual functions behind the PyRoki-named cost factories.

Signatures and semantics follow PyRoki (upstream ``388e43e``, MIT) so that
``import pyroffi as pk`` runs PyRoki cost code unchanged. Each function takes ``vals``
(a :class:`pyroffi.VarValues`) first and can be wrapped directly with
:meth:`pyroffi.Cost.factory`. Everything here is plain JAX, traced once and compiled to CUDA
by ``LeastSquaresProblem.analyze()``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jaxlie
from jax import Array

from .._floating_base import split_cfg
from .._robot import Robot
from ..collision import CollGeom, colldist_from_sdf
from cudals import Var, VarValues


# --- Pose Residuals ---


def pose_residual(
    vals: VarValues,
    robot: Robot,
    joint_var: Var[Array],
    target_pose: jaxlie.SE3,
    target_link_index: Array,
    pos_weight: Array | float,
    ori_weight: Array | float,
) -> Array:
    """Computes the residual for matching link poses to target poses."""
    joint_cfg = vals[joint_var]
    Ts_link_world = robot.forward_kinematics(joint_cfg)
    pose_actual = jaxlie.SE3(Ts_link_world[..., target_link_index, :])
    residual = (pose_actual.inverse() @ target_pose).log()
    pos_residual = residual[..., :3] * pos_weight
    ori_residual = residual[..., 3:] * ori_weight
    return jnp.concatenate([pos_residual, ori_residual]).flatten()


def pose_masked_residual(
    vals: VarValues,
    robot: Robot,
    joint_var: Var[Array],
    target_pose: jaxlie.SE3,
    target_link_index: Array,
    pos_weight: Array | float,
    ori_weight: Array | float,
    joint_mask: Array | float = 1.0,
) -> Array:
    """``pose_residual`` whose Jacobian columns are scaled by ``joint_mask``.

    PyRoki's ``pose_cost_analytic_jac`` computes this Jacobian by hand; autodiff through a
    masked ``stop_gradient`` gives the same one, and it compiles like any other residual.
    """
    q = vals[joint_var]
    q = joint_mask * q + (1.0 - joint_mask) * jax.lax.stop_gradient(q)
    Ts_link_world = robot.forward_kinematics(q)
    pose_actual = jaxlie.SE3(Ts_link_world[..., target_link_index, :])
    residual = (pose_actual.inverse() @ target_pose).log()
    return jnp.concatenate([residual[..., :3] * pos_weight, residual[..., 3:] * ori_weight]).flatten()


def pose_numerical_residual(
    vals: VarValues,
    robot: Robot,
    joint_var: Var[Array],
    target_pose: jaxlie.SE3,
    target_link_index: Array,
    pos_weight: Array | float,
    ori_weight: Array | float,
    eps: float = 1e-4,
) -> Array:
    """PyRoki's ``pose_cost_numerical_jac`` residual (``eps`` is accepted; the Jacobian is exact)."""
    del eps
    T_world_ee = jaxlie.SE3(robot.forward_kinematics(vals[joint_var])[..., target_link_index, :])
    residual = (target_pose.inverse() @ T_world_ee).log()
    return (residual * jnp.array([pos_weight] * 3 + [ori_weight] * 3)).flatten()


def pose_with_base_residual(
    vals: VarValues,
    robot: Robot,
    joint_var: Var[Array],
    T_world_base_var: Var[jaxlie.SE3],
    target_pose: jaxlie.SE3,
    target_link_indices: Array,
    pos_weight: Array | float,
    ori_weight: Array | float,
) -> Array:
    """Computes the residual for matching link poses relative to a mobile base."""
    joint_cfg = vals[joint_var]
    T_world_base = vals[T_world_base_var]
    Ts_base_link = robot.forward_kinematics(joint_cfg)  # FK is T_base_link
    T_base_target_link = jaxlie.SE3(Ts_base_link[..., target_link_indices, :])
    T_world_target_link_actual = T_world_base @ T_base_target_link

    residual = (T_world_target_link_actual.inverse() @ target_pose).log()
    pos_residual = residual[..., :3] * pos_weight
    ori_residual = residual[..., 3:] * ori_weight
    return jnp.concatenate([pos_residual, ori_residual]).flatten()


# --- Limit Residuals ---


def _joints(robot: Robot, cfg: Array) -> Array:
    """The actuated joints of a configuration (a floating base's pose has no limits)."""
    return split_cfg(robot, cfg)[1]


def limit_residual(
    vals: VarValues,
    robot: Robot,
    joint_var: Var[Array],
    weight: Array | float = 1.0,
) -> Array:
    """Joint limit violation: positive when violated, zero when satisfied (``<= 0`` is feasible)."""
    joint_cfg = _joints(robot, vals[joint_var])
    joint_cfg_eff = robot.joints.get_full_config(joint_cfg)
    upper_violation = jnp.maximum(0.0, joint_cfg_eff - robot.joints.upper_limits_all)
    lower_violation = jnp.maximum(0.0, robot.joints.lower_limits_all - joint_cfg_eff)
    return (jnp.concatenate([upper_violation, lower_violation]) * weight).flatten()


def limit_velocity_residual(
    vals: VarValues,
    robot: Robot,
    joint_var: Var[Array],
    prev_joint_var: Var[Array],
    dt: float,
    weight: Array | float = 1.0,
) -> Array:
    """Joint velocity limit violation: positive when ``|velocity| > limit``, else zero."""
    joint_vel = (_joints(robot, vals[joint_var]) - _joints(robot, vals[prev_joint_var])) / dt
    residual = jnp.maximum(0.0, jnp.abs(joint_vel) - robot.joints.velocity_limits)
    return (residual * weight).flatten()


# --- Regularization Residuals ---


def rest_residual(
    vals: VarValues,
    joint_var: Var[Array],
    rest_pose: Array,
    weight: Array | float = 1.0,
) -> Array:
    """Computes the residual biasing joints towards a rest pose."""
    return ((vals[joint_var] - rest_pose) * weight).flatten()


def rest_with_base_residual(
    vals: VarValues,
    joint_var: Var[Array],
    T_world_base_var: Var[jaxlie.SE3],
    rest_pose: Array,
    weight: Array | float = 1.0,
) -> Array:
    """Computes the residual biasing joints and base towards its default pose."""
    residual_joints = vals[joint_var] - rest_pose
    residual_base = (
        vals[T_world_base_var].inverse() @ T_world_base_var.default_factory()
    ).log()
    return (jnp.concatenate([residual_joints, residual_base]) * weight).flatten()


def smoothness_residual(
    vals: VarValues,
    curr_joint_var: Var[Array],
    past_joint_var: Var[Array],
    weight: Array | float = 1.0,
) -> Array:
    """Computes the residual penalizing joint configuration differences (velocity)."""
    return ((vals[curr_joint_var] - vals[past_joint_var]) * weight).flatten()


# --- Manipulability Residual ---


def _compute_manip_yoshikawa(cfg: Array, robot: Robot, target_link_index: Array) -> Array:
    """Manipulability measure (translation) for a single link."""
    jacobian = jax.jacfwd(
        lambda q: jaxlie.SE3(robot.forward_kinematics(q)).translation()
    )(cfg)[target_link_index]
    JJT = jacobian @ jacobian.T
    assert JJT.shape == (3, 3)
    return jnp.sqrt(jnp.maximum(0.0, jnp.linalg.det(JJT)))


def manipulability_residual(
    vals: VarValues,
    robot: Robot,
    joint_var: Var[Array],
    target_link_indices: Array,
    weight: Array | float = 1.0,
) -> Array:
    """Computes the residual penalizing low manipulability (translation)."""
    cfg = vals[joint_var]
    if jnp.ndim(target_link_indices) == 0:
        vmapped_manip = _compute_manip_yoshikawa(cfg, robot, target_link_indices)
    else:
        vmapped_manip = jax.vmap(_compute_manip_yoshikawa, in_axes=(None, None, 0))(
            cfg, robot, target_link_indices
        )
    residual = 1.0 / (vmapped_manip + 1e-6)
    return (residual * weight).flatten()


# --- Collision Residuals ---


def self_collision_residual(
    vals: VarValues,
    robot: Robot,
    robot_coll,
    joint_var: Var[Array],
    margin: float,
    weight: Array | float = 1.0,
) -> Array:
    """Self-collision violation residual; positive when a collision is detected."""
    cfg = vals[joint_var]
    active_distances = robot_coll.compute_self_collision_distance(robot, cfg)
    return -(colldist_from_sdf(active_distances, margin) * weight).flatten()


def world_collision_residual(
    vals: VarValues,
    robot: Robot,
    robot_coll,
    joint_var: Var[Array],
    world_geom: CollGeom,
    margin: float,
    weight: Array | float = 1.0,
) -> Array:
    """World collision violation residual; positive when a collision is detected."""
    cfg = vals[joint_var]
    dist_matrix = robot_coll.compute_world_collision_distance(robot, cfg, world_geom)
    return -(colldist_from_sdf(dist_matrix, margin) * weight).flatten()


# --- Finite Difference Residuals ---


def five_point_velocity_residual(
    vals: VarValues,
    robot: Robot,
    var_t_plus_2: Var[Array],
    var_t_plus_1: Var[Array],
    var_t_minus_1: Var[Array],
    var_t_minus_2: Var[Array],
    dt: float,
    weight: Array | float = 1.0,
) -> Array:
    """Velocity limit violation using a 5-point stencil (positive when violated)."""
    q_tm2, q_tm1, q_tp1, q_tp2 = (_joints(robot, vals[v]) for v in
                                  (var_t_minus_2, var_t_minus_1, var_t_plus_1, var_t_plus_2))
    velocity = (-q_tp2 + 8 * q_tp1 - 8 * q_tm1 + q_tm2) / (12 * dt)
    residual = jnp.maximum(0.0, jnp.abs(velocity) - robot.joints.velocity_limits)
    return (residual * weight).flatten()


def five_point_acceleration_residual(
    vals: VarValues,
    var_t: Var[Array],
    var_t_plus_2: Var[Array],
    var_t_plus_1: Var[Array],
    var_t_minus_1: Var[Array],
    var_t_minus_2: Var[Array],
    dt: float,
    weight: Array | float = 1.0,
) -> Array:
    """Joint acceleration using a 5-point stencil."""
    q_tm2 = vals[var_t_minus_2]
    q_tm1 = vals[var_t_minus_1]
    q_t = vals[var_t]
    q_tp1 = vals[var_t_plus_1]
    q_tp2 = vals[var_t_plus_2]
    acceleration = (-q_tp2 + 16 * q_tp1 - 30 * q_t + 16 * q_tm1 - q_tm2) / (12 * dt**2)
    return (jnp.abs(acceleration) * weight).flatten()


def five_point_jerk_residual(
    vals: VarValues,
    var_t_plus_3: Var[Array],
    var_t_plus_2: Var[Array],
    var_t_plus_1: Var[Array],
    var_t_minus_1: Var[Array],
    var_t_minus_2: Var[Array],
    var_t_minus_3: Var[Array],
    dt: float,
    weight: Array | float = 1.0,
) -> Array:
    """Joint jerk using a 7-point stencil."""
    q_tm3 = vals[var_t_minus_3]
    q_tm2 = vals[var_t_minus_2]
    q_tm1 = vals[var_t_minus_1]
    q_tp1 = vals[var_t_plus_1]
    q_tp2 = vals[var_t_plus_2]
    q_tp3 = vals[var_t_plus_3]
    jerk = (-q_tp3 + 8 * q_tp2 - 13 * q_tp1 + 13 * q_tm1 - 8 * q_tm2 + q_tm3) / (8 * dt**3)
    return (jnp.abs(jerk) * weight).flatten()
