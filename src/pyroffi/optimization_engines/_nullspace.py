"""Null-space projection: enforce arbitrary JAX constraints without losing the pose.

This is the third IK path. The first two live in CUDA and answer "give me a valid
configuration for this pose", with self-collision (path 1) and optionally
environment obstacles (path 2). Neither can evaluate a user's constraint: the
kernels are compiled C++ and cannot call back into a Python/JAX callable.

So arbitrary constraints are enforced *after* the solve, by moving only in
directions that leave the end-effector where it is. For a task Jacobian ``J`` at
the current configuration, the projector

    P = I - J⁺J

annihilates any joint motion the task can see, so ``q + P z`` reaches the same
pose to first order for any ``z``. Descending a constraint's violation inside
that subspace trades redundancy for constraint satisfaction and spends nothing
from the pose.

Three properties of this that callers need to know, because they are the
difference between a projector that works and one that quietly lies:

**The null space is usually tiny.** Its dimension is ``n_act - rank(J)``: for a
7-DOF arm on a full 6-DOF pose task, exactly ONE. A single scalar constraint can
generally be satisfied; two generally cannot, at any step size or iteration
count. That is geometry, not a tuning failure, and `NullspaceResult.success`
reports it rather than returning a configuration that silently satisfies neither.

**The projector is only first-order.** ``P`` is exact for infinitesimal motion;
a finite step drifts the pose. Each iteration therefore follows its null-space
step with a task-space correction that pulls the residual back, and the loop
verifies the pose is still within tolerance before accepting. Without that the
"without losing the pose" guarantee decays silently over iterations.

**Projection can walk into an obstacle.** A configuration that arrived here
collision-free (paths 1 and 2 guarantee that) can be projected straight into
one, because the null-space direction knows only about the constraint. When a
collision checker is supplied, a step that breaks collision-freedom is rejected
and the step size backtracks -- so path 3 preserves what paths 1 and 2 bought
rather than undoing it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jaxlie
from jax import Array
from jaxtyping import Float

from .._robot import Robot
from ._ik_primitives import _ik_residual


class NullspaceResult(NamedTuple):
    """Outcome of a projection.

    Attributes:
        cfg: the projected configuration. On failure this is the best iterate
            found, NOT a partially-applied step -- it is always pose-valid and,
            when a checker was supplied, collision-free.
        success: constraints satisfied to ``constraint_tol`` with the pose still
            within ``pose_tol``.
        constraint_violation: final ``max |c_i(q)|``.
        pose_error: final ``max`` per-EE residual norm.
        nullspace_dim: ``n_act - rank(J)`` at the returned configuration. Zero
            means there was no freedom to move in and no constraint could have
            been enforced whatever the settings.
        start_collision_free: whether the INPUT configuration already satisfied
            the collision check. False means nothing could be accepted no matter
            the constraint, because every candidate is compared against a start
            that was already in collision -- the failure is upstream, in the
            solve that produced ``cfg``, not here. Reported separately because
            it is otherwise indistinguishable from an over-constrained problem,
            and the two have completely different fixes. ``True`` when no
            checker was supplied.
    """

    cfg: Float[Array, "n_act"]
    success: Array
    constraint_violation: Array
    pose_error: Array
    nullspace_dim: Array
    start_collision_free: Array


def _task_residual_and_jacobian(cfg, robot, target_link_indices, target_poses):
    """Stacked per-EE SE(3) residual and its Jacobian w.r.t. the configuration."""

    def residual(q):
        return jnp.concatenate([
            _ik_residual(q, robot, int(li), tp)
            for li, tp in zip(target_link_indices, target_poses)
        ])

    return residual(cfg), jax.jacobian(residual)(cfg)


def _constraint_residual_and_jacobian(cfg, robot, constraint_fns, constraint_args):
    """Stacked constraint values and their Jacobian.

    Constraints are treated as equalities driven to zero. An inequality is
    expressed the usual way, as a hinge that is already zero when satisfied, so
    a feasible constraint contributes no gradient and does not fight the others.
    """

    def residual(q):
        return jnp.stack([
            jnp.asarray(fn(q, robot, args)).reshape(())
            for fn, args in zip(constraint_fns, constraint_args)
        ])

    return residual(cfg), jax.jacobian(residual)(cfg)


def project_onto_constraints(
    cfg: Float[Array, "n_act"],
    robot: Robot,
    target_link_indices: Sequence[int],
    target_poses: Sequence[jaxlie.SE3],
    constraint_fns: Sequence[Callable],
    constraint_args: Sequence = (),
    *,
    max_iter: int = 25,
    step_size: float = 1.0,
    max_step_norm: float = 0.2,
    constraint_tol: float = 1e-4,
    pose_tol: float = 1e-4,
    damping: float = 1e-3,
    pose_restore_iters: int = 3,
    collision_checker: Any | None = None,
    collision_world: Any | None = None,
    collision_margin: float = 0.0,
    lower: Float[Array, "n_act"] | None = None,
    upper: Float[Array, "n_act"] | None = None,
) -> NullspaceResult:
    """Move ``cfg`` to satisfy ``constraint_fns`` while holding the pose.

    Args:
        cfg: starting configuration, normally the output of a CUDA IK solve.
        constraint_fns: callables ``f(cfg, robot, args) -> scalar``, driven to
            zero. Use a hinge such as ``relu(d_min - d(q))`` for an inequality.
        step_size: initial null-space step scale. Backtracked on any step that
            worsens the constraint, breaks the pose tolerance, or (with a
            checker) causes a collision.
        collision_checker / collision_world: when given, every accepted step is
            required to stay collision-free, so a path-1 or path-2 guarantee
            survives projection.

    Returns:
        A :class:`NullspaceResult`. Check ``success`` -- an over-constrained
        problem returns the best pose-valid iterate with ``success=False``
        rather than raising or silently drifting off the target.
    """
    if lower is None:
        lower = robot.joints.lower_limits
    if upper is None:
        upper = robot.joints.upper_limits
    n_act = cfg.shape[-1]
    constraint_args = tuple(constraint_args) or tuple(() for _ in constraint_fns)

    def collision_ok(q):
        if collision_checker is None:
            return True
        q_b = jnp.asarray(q, jnp.float32)[None]
        d_self = jnp.min(collision_checker.compute_self_collision_distance(robot, q_b))
        ok = d_self >= collision_margin
        if collision_world is not None:
            d_world = jnp.min(collision_checker.compute_world_collision_distance(
                robot, q_b, collision_world))
            ok = jnp.logical_and(ok, d_world >= collision_margin)
        return bool(ok)

    def violation(q):
        c, _ = _constraint_residual_and_jacobian(q, robot, constraint_fns, constraint_args)
        return float(jnp.max(jnp.abs(c)))

    # Checked once, up front: a start that is already in collision rejects every
    # candidate step and would otherwise look exactly like an infeasible
    # constraint. Path 3 is meant to consume the output of path 1 or 2, both of
    # which guarantee collision-freedom; a False here means that contract was
    # broken before the projector was called.
    start_free = collision_ok(cfg)

    best = cfg
    best_viol = violation(cfg)
    step = step_size

    for _ in range(max_iter):
        if best_viol <= constraint_tol:
            break

        r, J = _task_residual_and_jacobian(best, robot, target_link_indices, target_poses)
        c, Jc = _constraint_residual_and_jacobian(
            best, robot, constraint_fns, constraint_args)

        # P = I - J^+ J. Built from the pseudoinverse so a rank-deficient task
        # Jacobian (a singular configuration) widens the null space rather than
        # blowing up, which is the behaviour that keeps this usable near
        # singularities instead of exactly where it is needed most.
        J_pinv = jnp.linalg.pinv(J)
        P = jnp.eye(n_act) - J_pinv @ J

        # Gauss-Newton step on the constraints, confined to the null space.
        #
        # Damping is RELATIVE to the projected constraint Jacobian, not an
        # absolute floor. A constraint is often only weakly visible inside the
        # null space -- on a 7-DOF arm holding a full pose, ||Jc P|| ~ 0.02 is
        # typical -- so an absolute 1e-6 leaves the solve dividing by ~5e-4 and
        # returns a step of several radians, which then fails every backtrack
        # for overshooting the pose. Scaling with the problem keeps the step
        # sane whatever the conditioning.
        Jc_n = Jc @ P
        scale = jnp.maximum(jnp.sum(Jc_n * Jc_n), 1e-12)
        JJt = Jc_n @ Jc_n.T + damping * scale * jnp.eye(Jc_n.shape[0])
        dq = -P @ (Jc_n.T @ jnp.linalg.solve(JJt, c))

        # Trust region. The pose is restored by a linearised correction below,
        # whose error grows with the square of the step, so an unbounded step
        # cannot be corrected back onto the pose no matter how it is scaled.
        dq_norm = jnp.linalg.norm(dq)
        dq = jnp.where(dq_norm > max_step_norm, dq * (max_step_norm / dq_norm), dq)

        accepted = False
        trial_step = step
        for _ in range(8):                      # backtracking line search
            q_try = jnp.clip(best + trial_step * dq, lower, upper)

            # First-order projection drifts; pull the pose back before judging
            # the step, so the comparison is against a pose-valid candidate.
            # Iterated, because one Newton correction is itself linearised and
            # leaves second-order error -- which is exactly the error that made
            # otherwise-good steps fail the pose test.
            for _ in range(pose_restore_iters):
                r_try, J_try = _task_residual_and_jacobian(
                    q_try, robot, target_link_indices, target_poses)
                if float(jnp.max(jnp.abs(r_try))) <= pose_tol:
                    break
                q_try = jnp.clip(q_try - jnp.linalg.pinv(J_try) @ r_try, lower, upper)

            r_fix, _ = _task_residual_and_jacobian(
                q_try, robot, target_link_indices, target_poses)
            pose_err = float(jnp.max(jnp.abs(r_fix)))
            viol_try = violation(q_try)

            if pose_err <= pose_tol and viol_try < best_viol and collision_ok(q_try):
                best, best_viol = q_try, viol_try
                step = min(step * 1.5, step_size)
                accepted = True
                break
            trial_step *= 0.5

        if not accepted:
            # No admissible step in the null space: either it is exhausted or
            # the constraints conflict with the pose. Stop rather than drift.
            break

    r_fin, J_fin = _task_residual_and_jacobian(
        best, robot, target_link_indices, target_poses)
    pose_error = jnp.max(jnp.abs(r_fin))
    ns_dim = n_act - jnp.linalg.matrix_rank(J_fin)
    return NullspaceResult(
        cfg=best,
        success=jnp.logical_and(
            jnp.logical_and(best_viol <= constraint_tol, pose_error <= pose_tol),
            jnp.asarray(start_free)),
        constraint_violation=jnp.asarray(best_viol),
        pose_error=pose_error,
        nullspace_dim=ns_dim,
        start_collision_free=jnp.asarray(start_free),
    )
