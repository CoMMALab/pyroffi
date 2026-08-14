"""Implicit differentiation for IK solvers.

The IK solvers in this package (HJCD, LS, SQP, learned) are iterative/CUDA
optimizers: differentiating *through* their iterations is unstable (and
impossible for the FFI kernels).  Instead we make the converged solution
``q* = solve(target)`` differentiable w.r.t. the target pose by the **implicit
function theorem** applied to the optimality condition.

At a (near-)reachable solution the SE(3) log-map residual vanishes,
``r(q*, t) = 0``.  Differentiating implicitly:

    J_q dq* + J_t dt = 0   ⇒   dq* = -J_q^+ (J_t dt)

where ``J_q = ∂r/∂q`` is the kinematic Jacobian at ``q*`` and ``J_t = ∂r/∂t`` is
the residual's sensitivity to the target's SE(3) (wxyz_xyz) tangent.  Both come
from autodiff of the differentiable pure-JAX residual ``_ik_residual``.  The
pseudoinverse ``J_q^+`` yields the minimum-norm joint tangent and handles
redundant DOF / non-square Jacobians.

This is exposed as a ``jax.custom_jvp`` whose primal returns ``q*`` unchanged
(the solver is a black box, ``stop_gradient``-ed) and whose tangent rule is the
expression above.  ``custom_jvp`` provides forward mode (``jax.jvp``); JAX
transposes the linear rule for reverse mode (``jax.grad``).

Scope: gradients flow to the target pose(s) only.  Robot parameters, seeds,
cost weights and solver internals are treated as constants.  MPPI-IK (a particle
method) and the region-IK samplers (which return solution *sets*) are not
wrapped.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import jaxlie
from jax import Array
from jaxtyping import Float

from .._robot import Robot
from ._ik_primitives import _ik_residual


def _normalize_targets(
    target_link_indices: int | Sequence[int],
    target_poses,
) -> tuple[tuple[int, ...], Float[Array, "n_ee 7"]]:
    """Accept either a single (idx, SE3) or tuples thereof; return (link_idx_tuple, target_T)."""
    if isinstance(target_link_indices, int):
        target_link_indices = (target_link_indices,)
    if isinstance(target_poses, jaxlie.SE3):
        target_poses = (target_poses,)
    link_idx = tuple(int(i) for i in target_link_indices)
    target_T = jnp.stack([jnp.asarray(tp.wxyz_xyz) for tp in target_poses], axis=0)
    return link_idx, target_T


def differentiable_ik_solution(
    q_star: Float[Array, "n_act"],
    robot: Robot,
    target_link_indices: int | Sequence[int],
    target_poses,
) -> Float[Array, "n_act"]:
    """Attach an implicit-diff gradient w.r.t. the target pose(s) to an IK solution.

    Args:
        q_star:              The solver's joint solution, shape ``(n_act,)``.  Its
                             own dependence on the inputs is cut (``stop_gradient``);
                             the returned value is numerically identical.
        robot:               The robot model (constant w.r.t. differentiation).
        target_link_indices: EE link index/indices the solution targets.
        target_poses:        Desired SE(3) pose(s) — a single ``jaxlie.SE3`` or a
                             tuple, one per EE.  Gradients flow back to these.

    Returns:
        ``q_star`` (unchanged value) carrying a ``custom_jvp`` so that
        ``jax.jvp`` / ``jax.grad`` w.r.t. ``target_poses`` give ``dq*/dt``.
    """
    link_idx, target_T = _normalize_targets(target_link_indices, target_poses)
    # Cut the solver's own dependence on the inputs: the gradient is supplied
    # entirely by the implicit rule below, not by unrolling the solver.
    q_star = jax.lax.stop_gradient(q_star)
    out_dtype = q_star.dtype

    def _residual(q: Array, robot_: Robot, t: Array) -> Array:
        # Stacked SE(3) log-map residual over all EEs: shape (6 * n_ee,).
        # link_idx is static (a Python tuple), so closing over it is safe.
        return jnp.concatenate(
            [
                _ik_residual(q, robot_, link_idx[k], jaxlie.SE3(t[k]))
                for k in range(len(link_idx))
            ]
        )

    # ``q_star`` and ``robot`` are passed as explicit arguments (not closed over)
    # so the rule has no closed-over tracers when this runs inside a solver's
    # jax.jit; JAX can then transpose the linear jvp rule for reverse mode.
    @jax.custom_jvp
    def _ik_layer(t: Array, q_s: Array, robot_: Robot) -> Array:
        return q_s

    @_ik_layer.defjvp
    def _ik_layer_jvp(primals, tangents):
        (t, q_s, robot_) = primals
        (dt, _dq_s, drobot) = tangents
        # J_q and its pseudoinverse depend only on the (constant) solution and
        # robot, so the tangent map (dt, dtheta) -> dq* is linear and
        # JAX-transposable.
        J_q = jax.jacobian(_residual, argnums=0)(q_s, robot_, t)   # (6*n_ee, n_act)
        J_q_pinv = jnp.linalg.pinv(J_q)                            # (n_act, 6*n_ee)

        # Differentiating the optimality condition r(q*, t, theta) = 0 in BOTH
        # its arguments gives J_q dq* + J_t dt + J_theta dtheta = 0. The pose
        # term was already here; the second is the robot's own kinematic
        # parameters -- twists and parent transforms, 156 scalars on a Panda --
        # which is the calibration Jacobian.
        #
        # SOLVER hyperparameters (pose weights, damping, iteration counts, seed
        # counts) are deliberately absent, and that is not an omission: they do
        # not appear in r(q*, t, theta) = 0 at all, so dq*/d(hyperparameter) is
        # EXACTLY ZERO at a converged solution. That is what convergence means.
        # A non-zero value there would have to come from unrolling the solver or
        # from a stochastic estimator, and both answer a different question.
        _, Jt_dt = jax.jvp(lambda tt: _residual(q_s, robot_, tt), (t,), (dt,))
        # A symbolic-zero robot tangent (the common case: nobody is
        # differentiating the model) must not be fed to jax.jvp, which wants a
        # concrete tangent pytree. Materialising zeros costs a wasted FK, so
        # detect it and skip.
        drobot_leaves = [l for l in jax.tree.leaves(drobot)
                         if not isinstance(l, object.__class__)]
        has_robot_tangent = any(
            getattr(l, "shape", None) is not None for l in jax.tree.leaves(drobot))
        rhs = Jt_dt
        if has_robot_tangent:
            _, Jr_dr = jax.jvp(lambda rr: _residual(q_s, rr, t),
                               (robot_,), (drobot,))
            rhs = rhs + Jr_dr
        dq = -(J_q_pinv @ rhs)
        return q_s, dq.astype(out_dtype)

    return _ik_layer(target_T, q_star, robot)


def differentiable_ik_solution_batch(
    q_stars: Float[Array, "n_problems n_act"],
    robot: Robot,
    target_link_indices: int | Sequence[int],
    target_poses,
) -> Float[Array, "n_problems n_act"]:
    """Batched :func:`differentiable_ik_solution`.

    The batch entry points returned their solver output RAW, with no implicit
    rule attached, so ``jax.grad`` through a batched solve did not fall back to
    anything -- it tried to differentiate the FFI call itself and failed with
    "The FFI call to `ls_ik_cuda` cannot be differentiated". Every batched
    solver was undifferentiable, while the single-problem paths worked, which
    is not a difference anyone would predict from the API.

    Applies the same implicit rule per element. The solver output is
    ``stop_gradient``-ed inside, so the kernel is never differentiated through;
    only the optimality condition at the returned configuration is.
    """
    # stop_gradient OUTSIDE the vmap, not inside it. Applied within the mapped
    # function it is too late: JAX must still form the batched INPUT tangent to
    # enter the vmap, and that tangent comes from the FFI, so the kernel's JVP
    # is demanded before the cut is ever reached. Detaching first means the
    # solver output enters as a constant and the FFI is never differentiated --
    # which is why the single-problem path, where the cut sits directly on the
    # solver output, worked all along.
    q_stars = jax.lax.stop_gradient(q_stars)
    return jax.vmap(
        lambda q, t: differentiable_ik_solution(
            q, robot, target_link_indices, jaxlie.SE3(t))
    )(q_stars, target_poses.wxyz_xyz)


def detached_robot(robot):
    """A copy of ``robot`` with every leaf cut from autodiff.

    Pass THIS to the CUDA kernels and keep the live model for the implicit rule.

    The split is required, not stylistic. A robot parameter feeds two places:
    the kernel, and the pure-JAX residual the implicit rule differentiates. If
    the parameter reaches the FFI still carrying a tangent, JAX classifies the
    kernel's output as UNKNOWN during linearisation and the whole thing fails
    with "Linearization failed to produce known values for all output primals" --
    shielding the kernel's OUTPUT with a custom_jvp is not enough, because the
    problem is at its inputs.

    It is also the semantically correct cut: dq*/dtheta comes from the
    optimality condition at the returned configuration, never from how the
    solver happened to use theta while searching.
    """
    return jax.tree.map(jax.lax.stop_gradient, robot)
