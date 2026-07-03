"""Robot-level dynamics entry points (free-function style, like ``pyroffi.kinematics``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jax import Array
from jaxtyping import Float

from .._robot_urdf_parser import DynamicsInfo
from ._dynamics_jax import (
    _DEFAULT_GRAVITY,
    forward_dynamics_jax,
    inverse_dynamics_jax,
    mass_matrix_jax,
)

if TYPE_CHECKING:
    from .._robot import Robot


def _require_dynamics(robot: Robot) -> DynamicsInfo:
    if robot.dynamics is None:
        raise ValueError(
            "Robot has no dynamics information. The URDF must provide "
            "<inertial> data (and no mimic joints) for dynamics; see "
            "RobotURDFParser.parse_dynamics."
        )
    return robot.dynamics


def inverse_dynamics(
    robot: Robot,
    q: Float[Array, "*batch n_act_joints"],
    qd: Float[Array, "*batch n_act_joints"],
    qdd: Float[Array, "*batch n_act_joints"],
    gravity: float = _DEFAULT_GRAVITY,
) -> Float[Array, "*batch n_act_joints"]:
    """Joint torques realizing ``qdd`` at state ``(q, qd)`` (RNEA + damping)."""
    return inverse_dynamics_jax(_require_dynamics(robot), q, qd, qdd, gravity)


def forward_dynamics(
    robot: Robot,
    q: Float[Array, "*batch n_act_joints"],
    qd: Float[Array, "*batch n_act_joints"],
    tau: Float[Array, "*batch n_act_joints"],
    gravity: float = _DEFAULT_GRAVITY,
) -> Float[Array, "*batch n_act_joints"]:
    """Joint accelerations produced by torques ``tau`` at state ``(q, qd)``."""
    return forward_dynamics_jax(_require_dynamics(robot), q, qd, tau, gravity)


def mass_matrix(
    robot: Robot,
    q: Float[Array, "*batch n_act_joints"],
) -> Float[Array, "*batch n_act_joints n_act_joints"]:
    """Joint-space mass matrix M(q) (CRBA)."""
    return mass_matrix_jax(_require_dynamics(robot), q)
