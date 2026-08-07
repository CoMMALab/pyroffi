"""Inverse kinematics functional entry point backing ``Robot.inverse_kinematics``.

Unifies the four IK solver entry points behind a single dispatcher:

    solver ∈ {"hjcd", "ls"}   ×   use_cuda ∈ {False (JAX), True (CUDA FFI)}

All four share the same core algorithm shape (seeded coarse search + LM/CD
refinement) and the same ``(robot, target_link_indices, target_poses, ...)``
call convention, differing only in constraint-kwarg naming (the JAX solvers
take ``constraint_fns``; the CUDA solvers take ``constraints`` plus optional
collision-kernel extras).  This module normalises single- vs multi-EE targets,
resolves link names to indices, and maps the constraint kwargs accordingly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import jax
import jaxlie
from jax import Array
from jax import numpy as jnp
from jaxtyping import Float

if TYPE_CHECKING:
    from .._robot import Robot


def _resolve_solver(solver: str, use_cuda: bool):
    """Return ``(solve_fn, constraints_are_cuda_style)`` for the requested backend."""
    if solver == "hjcd":
        if use_cuda:
            from ..optimization_engines._hjcd_ik import hjcd_solve_cuda

            return hjcd_solve_cuda, True
        from ..optimization_engines._hjcd_ik import hjcd_solve

        return hjcd_solve, False
    if solver == "ls":
        if use_cuda:
            from ..optimization_engines._ls_ik import ls_ik_solve_cuda

            return ls_ik_solve_cuda, True
        from ..optimization_engines._ls_ik import ls_ik_solve

        return ls_ik_solve, False
    if solver == "quik":
        # QuIK is a CPU C++ (Halley's-method) serial-chain backend; it ignores
        # use_cuda and takes no differentiable constraints.
        from ..optimization_engines._quik_ik import quik_ik_solve

        return quik_ik_solve, False
    if solver == "halley":
        # Pure-JAX reimplementation of QuIK's third-order Halley update; runs on
        # whatever JAX platform is active (use JAX_PLATFORMS=cpu for CPU-only).
        from ..optimization_engines._halley_ik import halley_ik_solve

        return halley_ik_solve, False
    raise ValueError(
        f"Unknown IK solver {solver!r}; expected 'hjcd', 'ls', 'quik' or 'halley'."
    )


def inverse_kinematics(
    robot: Robot,
    target_link_name: str | Sequence[str],
    target_pose: jaxlie.SE3 | Sequence[jaxlie.SE3],
    rng_key: Array | None = None,
    previous_cfg: Float[Array, "n_actuated_joints"] | None = None,
    solver: str = "hjcd",
    num_seeds: int = 32,
    continuity_weight: float = 1e-3,
    fixed_joint_mask: Float[Array, "n_actuated_joints"] | None = None,
    constraints: Sequence = (),
    constraint_args: Sequence = (),
    constraint_weights=None,
    use_cuda: bool = False,
    **solver_kwargs,
) -> Float[Array, "n_actuated_joints"]:
    """Solve inverse kinematics, dispatching over solver and backend.

    See ``Robot.inverse_kinematics`` for the full parameter documentation.

    Args:
        solver:            ``"hjcd"`` (two-phase coordinate-descent + LM),
                           ``"ls"`` (Levenberg-Marquardt least-squares),
                           ``"quik"`` (QuIK C++ Halley's-method CPU backend, a
                           fast serial-chain solver for ``JAX_PLATFORMS=cpu``
                           planning), or ``"halley"`` (a pure-JAX reimplementation
                           of QuIK's third-order update — same algorithm, any JAX
                           platform).  ``quik``/``halley`` are single-end-effector
                           serial-chain solvers and ignore differentiable
                           constraints.
        use_cuda:          Select the CUDA FFI backend for the chosen solver
                           instead of the pure-JAX implementation (ignored by
                           ``quik``/``halley``).
        target_link_name:  A single link name, or a sequence of names for a
                           multi-end-effector (e.g. bimanual) solve.
        target_pose:       A matching single ``SE3`` or sequence of poses.
        constraints:       Optional differentiable penalty callables folded into
                           seed selection / refinement.
        solver_kwargs:     Extra solver-specific options forwarded verbatim
                           (e.g. ``max_iter``/``pos_weight``/``ori_weight`` for
                           ``ls``; ``coarse_max_iter``/``lm_max_iter``/``epsilon``/
                           ``nu``/``lambda_init`` for ``hjcd``; and the CUDA
                           collision-kernel options ``collision_free`` etc.).

    Returns:
        Best joint configuration found, shape ``(n_actuated_joints,)``.
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    if previous_cfg is None:
        previous_cfg = (robot.joints.lower_limits + robot.joints.upper_limits) / 2

    # Normalise single- vs multi-EE targets to tuples.
    names = (
        (target_link_name,)
        if isinstance(target_link_name, str)
        else tuple(target_link_name)
    )
    poses = (
        (target_pose,)
        if isinstance(target_pose, jaxlie.SE3)
        else tuple(target_pose)
    )
    target_link_indices = tuple(robot.links.names.index(n) for n in names)

    solve, cuda_style_constraints = _resolve_solver(solver, use_cuda)

    # The JAX and CUDA solvers name their constraint kwargs differently.
    if cuda_style_constraints:
        constraint_kwargs = dict(
            constraints=tuple(constraints) or None,
            constraint_args=tuple(constraint_args) or None,
            constraint_weights=constraint_weights,
        )
        # Hoist the CUDA solvers' host-side ancestor-mask precompute out of the
        # trace by deriving it from the robot's *concrete* kinematic structure
        # (cached on the backends holder).  Passing these lets the whole CUDA
        # IK call run inside a caller's jax.jit.  Falls back to the solver's own
        # in-trace-incompatible precompute if no backends holder is present.
        if robot._backends is not None:
            target_jnts, ancestor_masks = robot._backends.ik_ancestor_masks(
                target_link_indices
            )
            constraint_kwargs["ancestor_masks"] = ancestor_masks
            constraint_kwargs["target_jnts"] = target_jnts
    else:
        constraint_kwargs = dict(
            constraint_fns=tuple(constraints),
            constraint_args=tuple(constraint_args),
            constraint_weights=constraint_weights,
        )

    return solve(
        robot=robot,
        target_link_indices=target_link_indices,
        target_poses=poses,
        rng_key=rng_key,
        previous_cfg=previous_cfg,
        num_seeds=num_seeds,
        continuity_weight=continuity_weight,
        fixed_joint_mask=fixed_joint_mask,
        **constraint_kwargs,
        **solver_kwargs,
    )
