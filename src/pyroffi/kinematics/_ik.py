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
    if solver == "analytic":
        # Closed-form subproblem decomposition. Unlike the iterative solvers it
        # takes no seeds and no differentiable constraints; `use_cuda` selects
        # the kernel explicitly, while the `analytic_ik_solve_batched` entry
        # point dispatches on batch size instead. See _analytic_ik.py.
        from ._analytic_ik import analytic_ik_solve

        return analytic_ik_solve, False

    raise ValueError(
        f"Unknown IK solver {solver!r}; expected 'hjcd', 'ls' "
        f"or 'analytic'."
    )


_CHART_CONSTRAINTS: dict = {}


def _chart_constraint(fn, q_chart, chart, args):
    """A floating robot's constraint ``fn(cfg, robot, args)`` evaluated at a chart point."""
    from .._floating_base import from_chart

    robot, ref, user_args = args
    return fn(from_chart(robot, q_chart, ref), robot, user_args)


def _solve_on_chart(robot, target_link_name, target_pose, rng_key, previous_cfg, solver,
                    num_seeds, continuity_weight, fixed_joint_mask, constraints,
                    constraint_args, constraint_weights, use_cuda, base_region, solver_kwargs):
    import functools

    from .._floating_base import chart_robot, from_chart, to_chart

    ref = previous_cfg[..., :7]
    if ref.ndim > 1:
        # ponytail: batched warm starts share the first problem's chart centre.
        ref = ref.reshape(-1, 7)[0]
    if solver_kwargs.get("traced"):
        # ponytail: traced kernels bake the chart's reference rotation as the identity, so
        # the chart is world-aligned (singular where the base pitches 90 deg in the world).
        ref = ref.at[:4].set(jnp.array([1.0, 0.0, 0.0, 0.0], ref.dtype))
    chart = chart_robot(robot, ref, base_region)
    if fixed_joint_mask is not None:
        fixed_joint_mask = jnp.concatenate(
            [jnp.asarray(fixed_joint_mask, bool), jnp.zeros(6, bool)])
    wrapped = []
    for fn in constraints:  # cached so a static constraint tuple stays hash-stable
        wrapped.append(_CHART_CONSTRAINTS.setdefault(fn, functools.partial(_chart_constraint, fn)))
    args = tuple((robot, ref, a) for a in constraint_args)
    q = inverse_kinematics(
        chart, target_link_name, target_pose, rng_key=rng_key,
        previous_cfg=to_chart(robot, previous_cfg, ref), solver=solver, num_seeds=num_seeds,
        continuity_weight=continuity_weight, fixed_joint_mask=fixed_joint_mask,
        constraints=tuple(wrapped), constraint_args=args, constraint_weights=constraint_weights,
        use_cuda=use_cuda, **solver_kwargs)
    return from_chart(robot, q, ref)


def _solve_analytic(robot, target_link_name, target_pose, previous_cfg, solver_kwargs):
    """Closed-form 7-DOF arm IK (the first seven actuated joints; the rest ride along from
    ``previous_cfg``). A floating base stays where ``previous_cfg`` puts it and the arm is
    solved for the target expressed in the base frame: exact whenever the base is given."""
    from .._floating_base import fixed_tree
    from ._analytic_ik import analytic_ik_solve

    if not isinstance(target_link_name, str):
        raise ValueError("solver='analytic' takes a single target link.")
    if previous_cfg is None:
        previous_cfg = robot.default_cfg
    base = None
    if robot.floating_base:
        base, previous_cfg = previous_cfg[:7], previous_cfg[7:]
        target_pose = jaxlie.SE3(base).inverse() @ target_pose
        robot = fixed_tree(robot)
    q7, _ = analytic_ik_solve(robot, target_link_name, target_pose,
                              previous_cfg=previous_cfg, **solver_kwargs)
    q = jnp.asarray(previous_cfg).at[:7].set(q7)
    return q if base is None else jnp.concatenate([base, q])


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
    base_region=None,
    **solver_kwargs,
) -> Float[Array, "n_actuated_joints"]:
    """Solve inverse kinematics, dispatching over solver and backend.

    See ``Robot.inverse_kinematics`` for the full parameter documentation.

    Args:
        solver:            ``"hjcd"`` (two-phase coordinate-descent + LM),
                           or ``"ls"`` (Levenberg-Marquardt least-squares).
        use_cuda:          Select the CUDA FFI backend for the chosen solver
                           instead of the pure-JAX implementation.
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
    if solver == "analytic":
        return _solve_analytic(robot, target_link_name, target_pose, previous_cfg, solver_kwargs)
    if robot.floating_base:
        if previous_cfg is None:
            previous_cfg = robot.default_cfg
        if solver != "ls" or use_cuda or base_region is not None:
            # Every other solver runs on the fixed-base chart centred on the warm start's
            # base (JAX LS works in the base's tangent space directly).
            return _solve_on_chart(
                robot, target_link_name, target_pose, rng_key, previous_cfg, solver,
                num_seeds, continuity_weight, fixed_joint_mask, constraints, constraint_args,
                constraint_weights, use_cuda, base_region, solver_kwargs)
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
