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
count. That is geometry, not a tuning failure, and `NullspaceResult` reports it.

**The projector is only first-order.** ``P`` is exact for infinitesimal motion;
a finite step drifts the pose. Each null-space step is followed by an iterated
task-space correction, and the pose is re-verified before the step is accepted.
Without that the "without losing the pose" guarantee decays over iterations.

**Projection can walk into an obstacle.** The null-space direction knows only
about the constraint, so a collision-free start can be projected straight into
contact. With a checker supplied, such a step is rejected.

Batching
--------
This is BATCH-NATIVE: it operates on ``(B, n_act)`` throughout, and a single
configuration is just ``B == 1``. It is deliberately not a ``vmap`` of a scalar
implementation, because the collision checker it calls every iteration reaches
an FFI declared ``vmap_method="sequential"`` -- vmapping around it would
serialise B kernel launches per iteration, which is exactly the pathology the
CUDA IK solvers were just fixed for. Operating on the batch directly means one
collision call per iteration regardless of B.

Control flow is closed-loop -- fixed trip count, masked updates, a parallel step
ladder instead of a ``break`` -- because unlike the CUDA kernels this has to stay
traceable for ``jit``. Converged elements are masked out rather than exited, so a
batch costs the worst element's iteration count, not the sum.

Performance: HOIST YOUR CONSTRAINT FUNCTIONS
--------------------------------------------
The compiled inner solve is cached on the STATIC configuration, and
``constraint_fns`` is part of that key -- by function IDENTITY, since a closure
is what it is. Building the closure inside the call site therefore produces a
fresh key every call and recompiles every call::

    for batch in batches:                       # 5597 ms per call at B=64
        project_onto_constraints(..., (make_constraint(elbow),), ...)

    con = make_constraint(elbow)                # 585 ms per call at B=64
    for batch in batches:
        project_onto_constraints(..., (con,), ...)

A 9.6x difference, and it looks like the solver being slow rather than like a
cache miss. Define constraint callables once, outside the loop.

Performance: the rest
---------------------
Wrap calls in ``jax.jit``. Measured 6.8x at B=64 (4374 -> 643 ms) with
byte-identical output.

The cost is DISPATCH-BOUND, not compute-bound, which is not the obvious guess.
Runtime is nearly flat in batch size -- 4.4 s at B=16 against 5.2 s at B=256 --
the signature of per-operation overhead rather than per-element work. The loop
is ``max_iter x pose_restore_iters`` deep, roughly 120 sequential steps of small
operations, and paying dispatch on each dominates everything else.

That is also why the obvious optimisations are NOT applied here: replacing
``pinv`` with a damped least-squares solve, or shrinking the step ladder, both
cut per-element FLOPs, which the measurement says is not where the time goes.

Compiling internally was tried and REVERTED: caching the executable across calls
bakes the first call's ``cfg``, targets and ``constraint_args`` in as closure
constants, so later calls with new inputs get the first call's answer. Making
that safe means threading every varying array through as an explicit argument
rather than closing over it -- worth doing, but it is a restructure, not a
wrapper.
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


#: Trial step scales evaluated in PARALLEL each iteration. Sequential
#: backtracking needs data-dependent control flow; evaluating the whole ladder
#: and selecting the best admissible rung is equivalent and traceable. Same
#: idiom as the LM line search in the CUDA kernels.
_STEP_LADDER = jnp.array([1.0, 0.5, 0.25, 0.1, 0.025, 0.005])


class NullspaceResult(NamedTuple):
    """Outcome of a projection. Every field carries a leading batch axis.

    Attributes:
        cfg: ``(B, n_act)`` projected configurations. On failure this is the
            best iterate found, NOT a partially-applied step -- always
            pose-valid and, when a checker was supplied, collision-free.
        success: constraints within ``constraint_tol``, pose within
            ``pose_tol``, and the start was collision-free.
        constraint_violation: final ``max_i |c_i(q)|`` per element.
        pose_error: final ``max`` task residual per element. Compare it against
            the input's -- the projector holds the pose it was GIVEN rather than
            improving on it.
        nullspace_dim: ``n_act - rank(J)`` per element. Zero means there was no
            freedom to move in, so no constraint could have been enforced.
        start_collision_free: whether the INPUT already passed the collision
            check. False means nothing could be accepted whatever the
            constraint, because every candidate is judged against a start
            already in collision -- the failure is upstream, in the solve that
            produced ``cfg``. Reported separately because it is otherwise
            indistinguishable from an over-constrained problem and the two have
            completely different fixes. ``True`` when no checker was supplied.
    """

    cfg: Float[Array, "b n_act"]
    success: Array
    constraint_violation: Array
    pose_error: Array
    nullspace_dim: Array
    start_collision_free: Array


def _task_residual(cfg, robot, target_link_indices, target_poses, b):
    """Stacked per-EE SE(3) residual for element ``b`` of the batch."""
    return jnp.concatenate([
        _ik_residual(cfg, robot, int(li), jax.tree.map(lambda x: x[b], tp))
        for li, tp in zip(target_link_indices, target_poses)
    ])


def _constraint_residual(cfg, constraint_args, robot, constraint_fns):
    """Stacked constraint values for ONE configuration.

    Constraints are equalities driven to zero. An inequality is written as a
    hinge that is already zero when satisfied, so a feasible constraint
    contributes no gradient and does not fight the others.
    """
    return jnp.stack([
        jnp.asarray(fn(cfg, robot, args)).reshape(())
        for fn, args in zip(constraint_fns, constraint_args)
    ])



def _collision_free(q, robot, collision_checker, collision_world, collision_margin):
    """``(B,)`` bool -- ONE checker call for the whole batch, not B calls."""
    if collision_checker is None:
        return jnp.ones(q.shape[0], dtype=bool)
    qf = jnp.asarray(q, jnp.float32)
    ok = jnp.min(collision_checker.compute_self_collision_distance(robot, qf),
                 axis=-1) >= collision_margin
    if collision_world is not None:
        dw = collision_checker.compute_world_collision_distance(
            robot, qf, collision_world).reshape(q.shape[0], -1)
        ok = jnp.logical_and(ok, jnp.min(dw, axis=-1) >= collision_margin)
    return ok


#: Compiled inner solves, keyed on the STATIC parts of a call.
_CORE_CACHE: dict = {}


def _get_core(robot, target_link_indices, constraint_fns, collision_checker,
              collision_world, collision_margin, max_iter, max_step_norm,
              constraint_tol, pose_tol, damping, pose_restore_iters,
              batched_constraint_args):
    """Return the compiled inner solve for this static configuration.

    ONLY statics are captured in the closure. Every array that varies between
    calls -- the configurations, the target poses, the constraint arguments, the
    joint limits, the free-joint mask -- is a PARAMETER of ``core``.
    
    That split is the whole point. An earlier attempt cached a jit wrapper that
    closed over ``cfg``, the targets and ``constraint_args``, which baked the
    FIRST call's values in as compile-time constants: every later call returned
    the first call's answer, silently. It survived benchmarking because each
    batch size produced a distinct cache key and so a fresh compile.

    Keying on ``id()`` for the robot and collision objects follows the same
    assumption ``_ls_ik._COLLISION_BUFFER_CACHE`` already makes -- these are
    configuration objects whose contents do not change between calls. Shape
    changes need no key: ``jax.jit`` retraces on those by itself.

    The cache holds a STRONG REFERENCE to every object it keyed on. Without
    that, an id is only unique while its object is alive: a robot that is freed
    lets CPython hand the same id to the next allocation, and a later call with
    a DIFFERENT robot would hit this cache and silently run the earlier robot's
    compiled kinematics. Pinning them trades a little memory for making the key
    mean what it appears to mean.
    """
    key = (id(robot), target_link_indices, constraint_fns,
           id(collision_checker), id(collision_world), float(collision_margin),
           int(max_iter), float(max_step_norm), float(constraint_tol),
           float(pose_tol), float(damping), int(pose_restore_iters),
           bool(batched_constraint_args))
    entry = _CORE_CACHE.get(key)
    if entry is not None:
        return entry[0]

    def core(cfg, target_poses, constraint_args, lower, upper, free_mask):
        B, n_act = cfg.shape
        idx = jnp.arange(B)

        def task_res(q, b):
            return _task_residual(q, robot, target_link_indices, target_poses, b)

        def con_res(q, args):
            return _constraint_residual(q, args, robot, constraint_fns)

        _arg_ax = 0 if batched_constraint_args else None
        batched_task = jax.vmap(task_res)
        batched_task_jac = jax.vmap(jax.jacobian(task_res, argnums=0))
        batched_con = jax.vmap(con_res, in_axes=(0, _arg_ax))
        batched_con_jac = jax.vmap(jax.jacobian(con_res, argnums=0), in_axes=(0, _arg_ax))

        def collision_free(q):
            """``(B,)`` bool -- ONE checker call for the whole batch, not B calls."""
            if collision_checker is None:
                return jnp.ones(q.shape[0], dtype=bool)
            qf = jnp.asarray(q, jnp.float32)
            ok = jnp.min(collision_checker.compute_self_collision_distance(robot, qf),
                         axis=-1) >= collision_margin
            if collision_world is not None:
                dw = collision_checker.compute_world_collision_distance(
                    robot, qf, collision_world).reshape(q.shape[0], -1)
                ok = jnp.logical_and(ok, jnp.min(dw, axis=-1) >= collision_margin)
            return ok

        def restore_pose(q):
            """Iterated task-space correction.

            One Newton correction is itself linearised and leaves the second-order
            error that rejects otherwise-good null-space steps. Iterating is far
            cheaper than shrinking the step until the linearisation happens to hold.
            """
            for _ in range(pose_restore_iters):
                r = batched_task(q, idx)
                J = batched_task_jac(q, idx)
                corr = jnp.einsum('bij,bj->bi', jnp.linalg.pinv(J), r) * free_mask
                q = jnp.clip(q - corr, lower, upper)
            return q

        start_free = collision_free(cfg)

        start_perr = jnp.max(jnp.abs(batched_task(cfg, idx)), axis=-1)
        pose_budget = jnp.maximum(pose_tol, start_perr)

        def body(_, carry):
            best, best_viol = carry

            J = batched_task_jac(best, idx)
            c = batched_con(best, constraint_args)
            Jc = batched_con_jac(best, constraint_args)

            eye = jnp.broadcast_to(jnp.eye(n_act), (B, n_act, n_act))
            P = eye - jnp.einsum('bij,bjk->bik', jnp.linalg.pinv(J), J)

            # Gauss-Newton on the constraints, confined to the null space. Damping
            # is RELATIVE to the projected constraint Jacobian, not an absolute
            # floor: a constraint is often only weakly visible in the null space
            # (||Jc P|| ~ 0.02 on a 7-DOF arm), so an absolute 1e-6 leaves this
            # dividing by ~5e-4 and returning a multi-radian step that fails every
            # rung of the ladder for overshooting the pose.
            Jc_n = jnp.einsum('bij,bjk->bik', Jc, P)
            scale = jnp.maximum(jnp.sum(Jc_n * Jc_n, axis=(1, 2)), 1e-12)
            JJt = (jnp.einsum('bij,bkj->bik', Jc_n, Jc_n)
                   + damping * scale[:, None, None] * jnp.eye(Jc_n.shape[1]))
            dq = -jnp.einsum('bij,bj->bi', P,
                             jnp.einsum('bji,bj->bi', Jc_n,
                                        jnp.linalg.solve(JJt, c[..., None]).squeeze(-1)))

            # Trust region: the pose is restored by a linearised correction whose
            # error grows with the square of the step, so an unbounded step cannot
            # be corrected back onto the pose at any scale.
            dq = dq * free_mask                      # frozen joints do not move
            nrm = jnp.linalg.norm(dq, axis=-1, keepdims=True)
            dq = jnp.where(nrm > max_step_norm, dq * (max_step_norm / (nrm + 1e-12)), dq)

            def trial(alpha):
                q = restore_pose(jnp.clip(best + alpha * dq, lower, upper))
                viol = jnp.max(jnp.abs(batched_con(q, constraint_args)), axis=-1)
                perr = jnp.max(jnp.abs(batched_task(q, idx)), axis=-1)
                return q, viol, (perr <= pose_budget) & (viol < best_viol) & collision_free(q)

            qs, viols, oks = jax.vmap(trial)(_STEP_LADDER)         # (S, B, ...)

            # First admissible rung wins, which is the largest step that works.
            ranked = jnp.where(oks, jnp.arange(len(_STEP_LADDER))[:, None],
                               len(_STEP_LADDER))
            pick = jnp.argmin(ranked, axis=0)
            any_ok = jnp.any(oks, axis=0)

            # Converged or stuck elements are masked, not exited: a batch costs the
            # worst element's iteration count rather than the sum.
            take = any_ok & (best_viol > constraint_tol)
            return (jnp.where(take[:, None], qs[pick, idx], best),
                    jnp.where(take, viols[pick, idx], best_viol))

        viol0 = jnp.max(jnp.abs(batched_con(cfg, constraint_args)), axis=-1)
        best, best_viol = jax.lax.fori_loop(0, max_iter, body, (cfg, viol0))

        pose_error = jnp.max(jnp.abs(batched_task(best, idx)), axis=-1)
        ns_dim = n_act - jnp.linalg.matrix_rank(batched_task_jac(best, idx))
        return best, best_viol, pose_error, ns_dim, pose_budget, start_free

    compiled = jax.jit(core)
    # Pinned alongside the executable: see the id-reuse note above.
    _CORE_CACHE[key] = (compiled, robot, collision_checker, collision_world)
    return compiled


def project_onto_constraints(
    cfg: Float[Array, "b n_act"],
    robot: Robot,
    target_link_indices: Sequence[int],
    target_poses: Sequence[jaxlie.SE3],
    constraint_fns: Sequence[Callable],
    constraint_args: Sequence = (),
    *,
    max_iter: int = 40,
    max_step_norm: float = 0.2,
    constraint_tol: float = 1e-4,
    pose_tol: float = 1e-3,
    damping: float = 1e-3,
    pose_restore_iters: int = 3,
    batched_constraint_args: bool = False,
    collision_checker: Any | None = None,
    collision_world: Any | None = None,
    collision_margin: float = 0.0,
    lower: Float[Array, "n_act"] | None = None,
    upper: Float[Array, "n_act"] | None = None,
    fixed_joint_mask: Float[Array, "n_act"] | None = None,
) -> NullspaceResult:
    """Move configurations to satisfy ``constraint_fns`` while holding their poses.

    Args:
        cfg: ``(B, n_act)`` or ``(n_act,)`` starting configurations, normally the
            output of a CUDA IK solve. A single configuration is promoted to a
            batch of one; the result keeps the leading axis.
        target_poses: one :class:`jaxlie.SE3` per end-effector, each with a
            leading batch axis matching ``cfg``.
        constraint_fns: callables ``f(cfg, robot, args) -> scalar`` driven to
            zero. Use a hinge such as ``relu(d_min - d(q))`` for an inequality.
        batched_constraint_args: whether each entry of ``constraint_args``
            carries a leading batch axis, i.e. every problem has its OWN
            constraint target (a per-problem elbow height, a per-problem
            clearance). Explicit rather than inferred from shapes, because a
            shared arg that happens to have length B is indistinguishable from a
            per-element one and guessing wrong is silent.
        collision_checker / collision_world: when given, an accepted step must
            remain collision-free, so a path-1 or path-2 guarantee survives.
        fixed_joint_mask: joints marked here are held exactly. Enforced by
            zeroing the null-space step on those columns rather than by clamping
            afterwards: clamping would let the pose-restoration correction move
            a frozen joint and then snap it back, which silently breaks the pose
            it had just restored.

    Returns:
        A batched :class:`NullspaceResult`. On failure check
        ``start_collision_free`` and ``nullspace_dim`` before tuning anything --
        they separate "your input was already invalid" and "this arm has no
        freedom left" from "the constraint is too tight".
    """
    if lower is None:
        lower = robot.joints.lower_limits
    if upper is None:
        upper = robot.joints.upper_limits
    lower, upper = jnp.asarray(lower), jnp.asarray(upper)

    cfg = jnp.atleast_2d(jnp.asarray(cfg))
    B, n_act = cfg.shape

    free_mask = (jnp.ones(n_act) if fixed_joint_mask is None
                 else 1.0 - jnp.asarray(fixed_joint_mask).astype(jnp.float32))
    constraint_args = tuple(constraint_args) or tuple(() for _ in constraint_fns)

    if len(constraint_fns) == 0:
        # Nothing to enforce. Returning the input IS the answer: this is a
        # CONSTRAINT projector, not a pose refiner, and it deliberately does not
        # improve a pose it was given. Callers wanting pose refinement want an
        # LM refiner -- conflating the two is what let the old _ls_ik_single
        # serve both roles and hid which one each call site depended on.
        idx0 = jnp.arange(B)
        perr = jnp.max(jnp.abs(jax.vmap(
            lambda q, b: _task_residual(q, robot, target_link_indices, target_poses, b)
        )(cfg, idx0)), axis=-1)
        free = _collision_free(cfg, robot, collision_checker, collision_world,
                               collision_margin)
        return NullspaceResult(
            cfg=cfg, success=free, constraint_violation=jnp.zeros(B),
            pose_error=perr, nullspace_dim=jnp.zeros(B, jnp.int32),
            start_collision_free=free,
        )

    core = _get_core(robot, tuple(target_link_indices), tuple(constraint_fns),
                     collision_checker, collision_world, collision_margin,
                     max_iter, max_step_norm, constraint_tol, pose_tol, damping,
                     pose_restore_iters, batched_constraint_args)
    best, best_viol, pose_error, ns_dim, pose_budget, start_free = core(
        cfg, tuple(target_poses), constraint_args, lower, upper, free_mask)

    return NullspaceResult(
        cfg=best,
        success=(best_viol <= constraint_tol) & (pose_error <= pose_budget) & start_free,
        constraint_violation=best_viol,
        pose_error=pose_error,
        nullspace_dim=ns_dim,
        start_collision_free=start_free,
    )


def project_single(
    cfg: Float[Array, "n_act"],
    robot: Robot,
    target_link_indices: Sequence[int],
    target_poses: Sequence[jaxlie.SE3],
    constraint_fns: Sequence[Callable],
    constraint_args: Sequence = (),
    **kwargs,
) -> Float[Array, "n_act"]:
    """One configuration in, one configuration out.

    Adapter for the single-problem solvers, whose post-solve refinement handles
    one configuration at a time. Adds and strips the batch axis the batched
    implementation works in; returns the configuration only, so it is a drop-in
    for the LM refiners it replaces. Callers that need to know WHETHER the
    constraints were met should call :func:`project_onto_constraints` directly
    and read ``success`` -- this wrapper deliberately cannot report failure, and
    silently returning the best pose-valid iterate is the right behaviour only
    where the caller already treats refinement as best-effort.
    """
    batched_targets = tuple(jax.tree.map(lambda x: x[None], tp) for tp in target_poses)
    return project_onto_constraints(
        jnp.asarray(cfg)[None], robot, target_link_indices, batched_targets,
        constraint_fns, constraint_args, **kwargs).cfg[0]
