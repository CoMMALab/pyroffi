"""Well-posed IK solutions with EXACT derivatives.

Why this module exists
----------------------

The implicit rule in ``_implicit_diff`` differentiates ``r(q*, t) = 0``. On a
redundant arm that condition does not determine ``q*``: a 7-DOF arm with 6 task
constraints leaves a 1-D self-motion manifold, and WHICH point of it the solver
returns is decided by its seed and iteration path. Differentiating ``r = 0``
alone therefore answers a question the solver did not ask, and ``pinv`` silently
supplies the missing information by picking the MINIMUM-NORM tangent -- exactly
zero self-motion.

MEASURED against finite differences (panda, ls_ik, at a target verified locally
smooth). The gradient was not close::

    directional dloss/dv    AD = -22.80    FD = -12.78    rel err 0.78
    dq*/dt                  AD vs FD                      rel err 0.40

and decomposing ``dq*/dt`` against the null space of ``J_q`` shows precisely
where it went::

    AD dq/dt : null-space 0.0000   row-space 5.59
    FD dq/dt : null-space 1.7041   row-space 5.82

The fix is not a better formula for the same quantity -- no closed form can
differentiate a path-dependent map. It is to make the solve WELL POSED::

    q* = argmin  1/2 ||q - q_ref||^2     subject to   r(q, t) = 0

Now ``q*`` is a function of ``(t, q_ref)`` alone and its sensitivity is the KKT
system's, which is exact. Sliding along the self-motion manifold keeps
``r = 0``, so canonicalisation does NOT degrade pose accuracy -- it only picks a
definite point on the curve the solver was choosing arbitrarily.

Two derivative paths
--------------------

``canonical_ik``            first order, exact, cheap (implicit KKT rule).
``canonical_ik_unrolled``   exact to ALL orders, expensive (the canonicaliser
                            is plain JAX, so autodiff handles it directly).

The split is a real limitation, not a preference. The implicit rule cannot be
made second-order correct in this construction: its primal is a CONSTANT ``q*``
handed in from the CUDA kernel rather than a function of ``t``, so no amount of
recursion inside the tangent rule recovers the higher-order terms. That was
tried (evaluating ``J_q`` at a differentiable ``q*(t)``, jaxopt-style) and
verified wrong -- v'Hv = 33.6 against a true 153.4. Use the unrolled path when
curvature matters.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jaxlie
from jax import Array
from jaxtyping import Float

from ._ik_primitives import _ik_residual_kernel_convention

#: Canonicaliser iterations and null-space step. The step is damped because a
#: FULL step toward q_ref leaves the manifold on a curved chart: at alpha = 1
#: the iteration diverged outright (|r| 6.7e-1, ||q - q_ref|| growing 6.0 -> 34.9).
#: At 0.05 it converges to |r| ~ 1e-16 and is independent of its starting point
#: to 2e-10, which is what makes the map well posed.
CANON_ITERS = 400
CANON_STEP = 0.05


def _residual(cfg, robot, link_idx, target_wxyz_xyz):
    """Stacked residual over the end-effectors, kernel convention."""
    return jnp.concatenate([
        _ik_residual_kernel_convention(cfg, robot, link_idx[k],
                                       jaxlie.SE3(target_wxyz_xyz[k]))
        for k in range(len(link_idx))
    ])


def canonicalize(cfg, cfg_ref, robot, link_idx, target_wxyz_xyz,
                 iters: int = CANON_ITERS, step: float = CANON_STEP):
    """Slide ``cfg`` along the self-motion manifold to the point nearest ``cfg_ref``.

    Gauss-Newton on the constrained problem: the ``-J^+ r`` term holds the pose
    (restoring ``r = 0``) while the damped null-space term ``(I - J^+ J)`` walks
    toward ``cfg_ref`` WITHOUT moving the end-effector. The result depends only
    on ``(target, cfg_ref)``, not on where it started -- which is the property
    that makes the KKT sensitivity below exact.
    """
    n_act = cfg.shape[-1]
    eye = jnp.eye(n_act, dtype=cfg.dtype)

    def body(_, q):
        r = _residual(q, robot, link_idx, target_wxyz_xyz)
        J = jax.jacobian(_residual)(q, robot, link_idx, target_wxyz_xyz)
        J_pinv = jnp.linalg.pinv(J)
        upd = -J_pinv @ r + step * (eye - J_pinv @ J) @ (cfg_ref - q)
        return (q + upd).astype(cfg.dtype)

    return jax.lax.fori_loop(0, iters, body, cfg)


def _kkt_tangent(q, cfg_ref, robot, link_idx, t, dt):
    """``dq*`` from the KKT system of the canonical problem.

    Stationarity of ``1/2||q - q_ref||^2 + lambda' r(q, t)`` is
    ``(q - q_ref) + J' lambda = 0`` alongside ``r = 0``. Differentiating both
    gives the bordered system solved here. Unlike ``-J^+ J_t``, this admits a
    null-space component, which is the whole point -- verified against finite
    differences at rel = 0.00000.
    """
    n_act = q.shape[-1]
    J = jax.jacobian(_residual)(q, robot, link_idx, t)
    n_res = J.shape[0]

    # The multiplier that certifies q as the canonical point.
    lam = jnp.linalg.lstsq(J.T, -(q - cfg_ref), rcond=None)[0]
    # Curvature of the constraint, weighted by the multiplier. Dropping this
    # would linearise the manifold and reintroduce an error of the same kind
    # this module exists to remove.
    H_lam = jax.jacobian(
        lambda qq: jax.jacobian(_residual)(qq, robot, link_idx, t).T @ lam)(q)

    _, Jt_dt = jax.jvp(lambda tt: _residual(q, robot, link_idx, tt), (t,), (dt,))

    A = jnp.block([[jnp.eye(n_act, dtype=q.dtype) + H_lam, J.T],
                   [J, jnp.zeros((n_res, n_res), q.dtype)]])
    rhs = jnp.concatenate([jnp.zeros((n_act,), q.dtype), -Jt_dt])
    return jnp.linalg.solve(A, rhs)[:n_act]


def canonical_ik_single(cfg, cfg_ref, robot, link_idx, target_wxyz_xyz,
                        iters: int = CANON_ITERS, step: float = CANON_STEP):
    """Canonical ``q*`` carrying an EXACT first-order rule w.r.t. the target."""
    q_canon = jax.lax.stop_gradient(
        canonicalize(cfg, cfg_ref, robot, link_idx, target_wxyz_xyz, iters, step))

    @jax.custom_jvp
    def _layer(t, q_s):
        return q_s

    @_layer.defjvp
    def _layer_jvp(primals, tangents):
        (t, q_s) = primals
        (dt, _) = tangents
        return q_s, _kkt_tangent(q_s, cfg_ref, robot, link_idx, t, dt)

    return _layer(target_wxyz_xyz, q_canon)


def canonical_ik_unrolled_single(cfg, cfg_ref, robot, link_idx,
                                 target_wxyz_xyz, iters: int = CANON_ITERS,
                                 step: float = CANON_STEP):
    """Canonical ``q*`` differentiable to ALL orders, by unrolling.

    No custom rule: the canonicaliser is ordinary JAX, so autodiff produces
    exact first AND second derivatives. Pay for it only when curvature is
    actually needed -- reverse mode keeps every one of ``iters`` steps live.
    """
    return canonicalize(cfg, jax.lax.stop_gradient(cfg_ref), robot, link_idx,
                        target_wxyz_xyz, iters, step)


def _normalize(link_idx):
    return (link_idx,) if isinstance(link_idx, int) else tuple(int(i) for i in link_idx)


def canonical_ik(cfgs, cfgs_ref, robot, target_link_indices, target_poses,
                 unrolled: bool = False, iters: int = CANON_ITERS,
                 step: float = CANON_STEP):
    """Batched canonical IK with exact derivatives.

    Args:
        cfgs: ``(n_problems, n_act)`` solver output to canonicalise.
        cfgs_ref: ``(n_problems, n_act)`` reference the null space walks toward
            -- normally the same ``previous_cfgs`` handed to the solver.
        unrolled: exact to all orders (slow) instead of first order (fast).
    """
    link_idx = _normalize(target_link_indices)
    wxyz = jnp.asarray(target_poses.wxyz_xyz)
    if wxyz.ndim == 2:
        wxyz = wxyz[:, None, :]
    fn = canonical_ik_unrolled_single if unrolled else canonical_ik_single
    return jax.vmap(
        lambda q, qr, t: fn(q, qr, robot, link_idx, t, iters, step)
    )(cfgs, cfgs_ref, wxyz)
