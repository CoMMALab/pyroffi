"""Robot-level dynamics entry points (free-function style, like ``pyroffi.kinematics``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
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
    substeps: int = 1,
) -> tuple[
    Float[Array, "*batch n_act_joints"], Float[Array, "*batch n_act_joints"]
]:
    """Advance ``(q, qd)`` one timestep under torques ``tau`` (and ``f_ext``).

    Semi-implicit (symplectic) Euler by default; also ``"euler"``, ``"rk4"``,
    and ``"linearly_implicit"``. jit/vmap/scan-compatible for trajectory
    rollouts.

    The explicit fixed-step methods (``"semi_implicit"``, ``"euler"``, ``"rk4"``)
    are only conditionally stable: they blow up to NaN/Inf once ``dt`` exceeds
    ``~2 / omega_max``. For the Panda this stiffness limit is small enough that
    even physically reasonable ``dt`` diverge (e.g. semi-implicit Euler NaNs at
    ``dt >= 0.05s`` under gravity alone), *regardless of the forward-dynamics
    accuracy* -- the mass matrix is well conditioned; the instability is the
    integrator. Two mitigations: ``substeps > 1`` subdivides ``dt`` (cheaper per
    step, but still only conditionally stable), or ``method="linearly_implicit"``
    (Rosenbrock--Euler), which is A-stable for the linearized dynamics and so
    extends the usable ``dt`` range several-fold (and does not NaN where the
    explicit methods do). The implicit method is most effective when the
    stabilizing forces are inside ``fd`` (joint damping, or a PD law folded into
    the torque); for very coarse ``dt`` on an undamped system combine it with
    ``substeps``. See ``_linearly_implicit_step`` for the precise stability
    caveats.
    """
    dyn = _require_dynamics(robot)

    def _single_step(q_, qd_, tau_, f_):
        return step_with_fd(
            lambda a, b: forward_dynamics_jax(dyn, a, b, tau_, gravity, f_),
            q_,
            qd_,
            dt,
            method,
            substeps,
        )

    # Flatten arbitrary leading batch dims and vmap the per-sample step, so the
    # linearly-implicit method (which builds a per-sample Jacobian) sees the
    # matching per-sample ``tau``/``f_ext`` rather than a batch-shared closure.
    batch_axes = q.shape[:-1]
    if batch_axes == ():
        return _single_step(q, qd, tau, f_ext)

    nb = len(batch_axes)

    def _flat(x):
        return None if x is None else x.reshape(-1, *x.shape[nb:])

    in_axes = (0, 0, 0, None if f_ext is None else 0)
    q_next, qd_next = jax.vmap(_single_step, in_axes=in_axes)(
        _flat(q), _flat(qd), _flat(tau), _flat(f_ext)
    )
    return (
        q_next.reshape(*batch_axes, *q_next.shape[1:]),
        qd_next.reshape(*batch_axes, *qd_next.shape[1:]),
    )
