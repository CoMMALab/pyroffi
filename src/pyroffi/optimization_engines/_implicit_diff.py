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
from ._ik_primitives import _ik_residual, _ik_residual_kernel_convention


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
    collision_checker=None,
    collision_world=None,
    collision_margin: float = 0.0,
    task_jacobian: Float[Array, "n_res n_act"] | None = None,
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
    constraint_grads = _active_constraint_grads(
        q_star, robot, collision_checker, collision_world, collision_margin)
    # Cut the solver's own dependence on the inputs: the gradient is supplied
    # entirely by the implicit rule below, not by unrolling the solver.
    q_star = jax.lax.stop_gradient(q_star)
    out_dtype = q_star.dtype

    # The residual must match whatever produced J_q, or the tangent is wrong.
    # See `_ik_residual_kernel_convention` for why: the two residuals differ by
    # an invertible A, the gradient is invariant only when EVERY block shares
    # one convention, and a mismatch raises nothing -- it just returns a
    # confidently wrong number. So the choice is made once, here, and the same
    # `_residual` then supplies the J_t and J_theta blocks below.
    _res_fn = (_ik_residual_kernel_convention if task_jacobian is not None
               else _ik_residual)

    def _residual(q: Array, robot_: Robot, t: Array) -> Array:
        # Stacked residual over all EEs: shape (6 * n_ee,).
        # link_idx is static (a Python tuple), so closing over it is safe.
        return jnp.concatenate(
            [
                _res_fn(q, robot_, link_idx[k], jaxlie.SE3(t[k]))
                for k in range(len(link_idx))
            ]
        )

    # ``q_star`` and ``robot`` are passed as explicit arguments (not closed over)
    # so the rule has no closed-over tracers when this runs inside a solver's
    # jax.jit; JAX can then transpose the linear jvp rule for reverse mode.
    # J_q travels as an explicit primal for the same reason q_s and robot_ do:
    # a closed-over tracer inside the rule blocks JAX from transposing it, which
    # would leave forward mode working and reverse mode (jax.grad) broken.
    # `jnp.zeros((0, 0))` stands in for "absent" so the signature is fixed --
    # a None primal is not a valid JAX type.
    _jac_primal = (jnp.zeros((0, 0), q_star.dtype) if task_jacobian is None
                   else jnp.asarray(task_jacobian, q_star.dtype))

    @jax.custom_jvp
    def _ik_layer(t: Array, q_s: Array, robot_: Robot, jac: Array) -> Array:
        return q_s

    @_ik_layer.defjvp
    def _ik_layer_jvp(primals, tangents):
        (t, q_s, robot_, jac) = primals
        (dt, _dq_s, drobot, _djac) = tangents
        # J_q and its pseudoinverse depend only on the (constant) solution and
        # robot, so the tangent map (dt, dtheta) -> dq* is linear and
        # JAX-transposable.
        # J_q from the CUDA task-Jacobian kernel when the caller supplied it,
        # mirroring how GRiD feeds its analytic gradient kernels into a
        # custom_jvp instead of re-differentiating on the host. Falling back to
        # jax.jacobian keeps every existing caller working and keeps the CPU
        # path testable against the GPU one.
        #
        # The kernel returns the GEOMETRIC Jacobian: its position rows are
        # exactly d(p_ee)/dq, but its orientation rows are the angular Jacobian
        # rather than d(log(R_ee R_tgt^-1))/dq. Those coincide only where the
        # orientation error vanishes, because the right-Jacobian of the log map
        # tends to identity there -- which is exactly the converged solution
        # this rule is defined at (it differentiates r = 0). MEASURED on a
        # panda over 64 solved problems: max|J_cuda - J_jax| = 1.5e-3, i.e.
        # float32 agreement, while at random UNCONVERGED configurations the
        # orientation rows differ by O(1). Do not reuse this J away from a
        # converged solution.
        #
        # PRECISION: the kernel is float32 while a JAX-computed J_q follows the
        # ambient x64 setting. That is immaterial for a well-conditioned J, but
        # pinv amplifies it without bound near a kinematic singularity, where
        # dq*/dt is genuinely ill-defined rather than merely hard to compute.
        # MEASURED at B=64 on a panda: J agrees to 5.6e-4 median and pinv to
        # 9.4e-4 median, but the single problem at cond(J) = 4.5e3 differed by
        # 19%. Under x64 with a well-conditioned batch the two agree to ~1e-3
        # relative. Anyone differentiating THROUGH a near-singular solution
        # should distrust the magnitude from either source, not just this one.
        if task_jacobian is not None:
            J_q = jac
        else:
            J_q = jax.jacobian(_residual, argnums=0)(q_s, robot_, t)  # (6*n_ee, n_act)
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

        # A solution held OFF-TARGET by an active collision constraint does not
        # satisfy r(q*, t) = 0, so the rule above -- which differentiates exactly
        # that condition -- is describing a stationarity that does not hold, and
        # returns a confidently wrong gradient. Restricting the tangent to the
        # null space of the ACTIVE constraint gradients is the correction: the
        # perturbed solution has to keep those constraints satisfied, so it can
        # only move in directions they cannot see.
        #
        # This is the same projector as IK path 3, applied to the tangent rather
        # than to a configuration. It is not the full KKT sensitivity (that would
        # solve the bordered system for the multipliers too), but it is right to
        # first order in the constrained directions and is strictly better than
        # ignoring the constraint, which is what happened before.
        if constraint_grads is not None:
            G = constraint_grads                              # (n_active, n_act)
            GGt = G @ G.T + 1e-9 * jnp.eye(G.shape[0])
            dq = dq - G.T @ jnp.linalg.solve(GGt, G @ dq)
        return q_s, dq.astype(out_dtype)

    return _ik_layer(target_T, q_star, robot, _jac_primal)


def _active_constraint_grads(q_star, robot, checker, world, margin):
    """Gradients of the collision constraints ACTIVE at ``q_star``, or None.

    Only constraints at their boundary restrict the tangent; an inactive one
    leaves the solution free in that direction and must not be projected out, or
    the gradient is over-constrained and wrong in the other direction.

    Returns ``None`` when nothing is active, which is the common case and keeps
    the unconstrained rule exactly as it was.
    """
    if checker is None:
        return None
    q = jnp.asarray(q_star, jnp.float32)[None]

    def dists(qq):
        d = [jnp.min(checker.compute_self_collision_distance(robot, qq), axis=-1)]
        if world is not None:
            d.append(jnp.min(
                checker.compute_world_collision_distance(robot, qq, world
                                                         ).reshape(qq.shape[0], -1), axis=-1))
        return jnp.concatenate(d)

    try:
        d0 = dists(q)
        J = jax.jacobian(lambda x: dists(x[None]))(jnp.asarray(q_star, jnp.float32))
    except Exception:
        # A checker that cannot be differentiated cannot constrain the tangent;
        # say so by returning None rather than silently projecting with garbage.
        return None

    active = d0 <= margin + 1e-6
    if not bool(jnp.any(active)):
        return None
    return jnp.where(active[:, None], J, 0.0)


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

    # ONE kernel launch for the whole batch's task Jacobians, instead of a
    # per-element jax.jacobian inside the vmap. This is the GRiD arrangement:
    # the solve kernel and a separate analytic-derivative kernel, with the
    # latter feeding the custom_jvp tangent rule.
    jacs = _batch_task_jacobians(q_stars, robot, target_link_indices,
                                 target_poses.wxyz_xyz)
    if jacs is None:
        return jax.vmap(
            lambda q, t: differentiable_ik_solution(
                q, robot, target_link_indices, jaxlie.SE3(t))
        )(q_stars, target_poses.wxyz_xyz)

    return jax.vmap(
        lambda q, t, j: differentiable_ik_solution(
            q, robot, target_link_indices, jaxlie.SE3(t), task_jacobian=j)
    )(q_stars, target_poses.wxyz_xyz, jacs)


#: Ancestor tables are derived by walking the chain with numpy, so they cannot be
#: built under a trace. Keyed on link NAMES plus the EE tuple -- not id(robot),
#: which would hand back another robot's chain after a garbage collection.
_ANCESTOR_CACHE: dict = {}


def _batch_task_jacobians(q_stars, robot, target_link_indices, target_wxyz_xyz):
    """``(n_problems, 6*n_ee, n_act)`` task Jacobians from CUDA, or None.

    Returns None -- and the caller falls back to ``jax.jacobian`` -- when the
    kernel is unavailable or the robot's arrays are still tracers. The fallback
    is deliberate: this is an optimisation of HOW J_q is obtained, and a missing
    .so or an exotic tracing context should cost speed, never correctness.
    """
    from ..cuda_kernels.ik import _ik_jacobian

    if not _ik_jacobian.library_available():
        return None

    link_idx = ((target_link_indices,) if isinstance(target_link_indices, int)
                else tuple(int(i) for i in target_link_indices))
    try:
        key = (tuple(robot.links.names), link_idx)
        tables = _ANCESTOR_CACHE.get(key)
        if tables is None:
            tables = _ANCESTOR_CACHE[key] = _ik_jacobian.ancestor_tables(
                robot, link_idx)
        target_jnts, ancestor_masks = tables

        # EVERY input is detached. J_q enters the tangent rule as a CONSTANT --
        # differentiating it would be a second-order term the rule does not use
        # -- and an input still carrying a tangent makes JAX try to
        # differentiate the FFI itself, which fails outright with "The FFI call
        # to `ik_task_jacobian` cannot be differentiated". This is the same cut
        # `detached_robot` makes for the solve kernels, and the same treatment
        # GRiD gives its analytic gradient kernels (constants for second order).
        J = jax.tree.map(jax.lax.stop_gradient, robot.joints)
        buffers = (J.twists, J.parent_transforms, J.parent_indices,
                   J.actuated_indices, J.mimic_multiplier, J.mimic_offset,
                   J.mimic_act_indices, J._topo_sort_inv)
        n_problems = q_stars.shape[0]
        targets = jax.lax.stop_gradient(
            jnp.asarray(target_wxyz_xyz).reshape(n_problems, len(link_idx), 7))
        _r, jac = _ik_jacobian.task_jacobian(
            jax.lax.stop_gradient(q_stars), buffers,
            target_jnts, ancestor_masks, targets)
        return jac
    except Exception:
        # Tracer-valued robot arrays, a capacity overflow, an unregistered
        # target -- all recoverable by computing J_q in JAX instead.
        return None


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
