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
    jacobian_jax,
    mass_matrix_jax,
)
from ._integrators import StepMethod, step_with_fd

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
    f_ext: Float[Array, "*batch n_act_joints 6"] | None = None,
) -> Float[Array, "*batch n_act_joints"]:
    """Joint torques realizing ``qdd`` at state ``(q, qd)`` (RNEA + damping).

    ``f_ext`` are optional per-body external wrenches ``[torque; force]``
    applied at each body's frame origin, expressed in world axes.
    """
    return inverse_dynamics_jax(_require_dynamics(robot), q, qd, qdd, gravity, f_ext)


def forward_dynamics(
    robot: Robot,
    q: Float[Array, "*batch n_act_joints"],
    qd: Float[Array, "*batch n_act_joints"],
    tau: Float[Array, "*batch n_act_joints"],
    gravity: float = _DEFAULT_GRAVITY,
    f_ext: Float[Array, "*batch n_act_joints 6"] | None = None,
) -> Float[Array, "*batch n_act_joints"]:
    """Joint accelerations produced by torques ``tau`` at state ``(q, qd)``.

    ``f_ext`` follows the same convention as :func:`inverse_dynamics`.
    """
    return forward_dynamics_jax(_require_dynamics(robot), q, qd, tau, gravity, f_ext)


def mass_matrix(
    robot: Robot,
    q: Float[Array, "*batch n_act_joints"],
) -> Float[Array, "*batch n_act_joints n_act_joints"]:
    """Joint-space mass matrix M(q) (CRBA)."""
    return mass_matrix_jax(_require_dynamics(robot), q)


def jacobian(
    robot: Robot,
    q: Float[Array, "*batch n_act_joints"],
) -> tuple[
    Float[Array, "*batch n_body 6 n_act_joints"],
    Float[Array, "*batch n_body 3"],
]:
    """World-frame geometric Jacobians ``(J, r)`` for every dynamic body.

    ``J[..., i, :, :] @ qd`` is the angular-first spatial velocity
    ``[omega; v]`` of body ``i``'s frame, with the linear part measured at the
    frame origin ``r[..., i, :]`` (LOCAL_WORLD_ALIGNED). Bodies are indexed by
    actuated joint, matching ``robot.dynamics.dof_names``.
    """
    return jacobian_jax(_require_dynamics(robot), q)


def step(
    robot: Robot,
    q: Float[Array, "*batch n_act_joints"],
    qd: Float[Array, "*batch n_act_joints"],
    tau: Float[Array, "*batch n_act_joints"],
    dt: float,
    gravity: float = _DEFAULT_GRAVITY,
    f_ext: Float[Array, "*batch n_act_joints 6"] | None = None,
    method: StepMethod = "semi_implicit",
) -> tuple[
    Float[Array, "*batch n_act_joints"], Float[Array, "*batch n_act_joints"]
]:
    """Advance ``(q, qd)`` one timestep under torques ``tau`` (and ``f_ext``).

    Semi-implicit (symplectic) Euler by default; also ``"euler"`` and
    ``"rk4"``. jit/vmap/scan-compatible for trajectory rollouts.
    """
    dyn = _require_dynamics(robot)
    return step_with_fd(
        lambda q_, qd_: forward_dynamics_jax(dyn, q_, qd_, tau, gravity, f_ext),
        q,
        qd,
        dt,
        method,
    )
