"""Contact-free, single-arm L-BFGS trajectory optimization over an arbitrary cost.

Structurally the same skeleton as :func:`~pyroffi.optimization_engines.
contact_rich_trajopt` and :func:`~pyroffi.optimization_engines.
flat_contact_trajopt` -- L-BFGS with a 5-point line search, built from the same
``_lbfgs_two_loop`` / ``_LS_ALPHAS`` primitives in ``_sco_optimization.py`` --
with all contact/grasp/object machinery removed. There is no
``ManipulatorSpec``, no ``ContactSystem`` and no augmented-Lagrangian outer
loop: with no hard grasp/object constraints to relax, a single L-BFGS minimize
of the caller's ``cost_fn`` is the whole solver.

Unlike the two contact engines, the cost is not fixed internal weights -- it is
supplied by the caller as an arbitrary ``cost_fn(x) -> scalar``, so this engine
can serve directly as the forward solver for ``ioc.inner.make_inner_solver``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from ._sco_optimization import _LS_ALPHAS, _lbfgs_two_loop


@dataclass(frozen=True)
class DynamicsTrajOptConfig:
    """Hyper-parameters for the dynamics-aware L-BFGS trajopt engine."""

    n_iters: int = 200
    """Number of L-BFGS steps."""

    m_lbfgs: int = 8
    """L-BFGS history size (number of curvature pairs)."""

    grad_tol: float = 1e-6
    """Inner L-BFGS stops once ``max|grad|`` drops below this. 0 disables (falls
    back to the fixed ``n_iters`` budget). Unlike the AL-penalty engines
    (`flat_contact_trajopt`, `contact_rich_trajopt`), this solver's objective is
    an unconstrained blend supplied directly by the caller, so the gradient norm
    genuinely does approach zero at a local optimum and this check is worth
    having on by default. Only meaningful when ``early_stop=True``."""

    early_stop: bool = True
    """True: run the ``while_loop`` early-stopping form -- cheap, but not
    reverse-mode differentiable (JAX cannot backprop through a data-dependent
    trip count), so this form must only ever be called under `stop_gradient`
    (e.g. as the forward solve inside `ioc.inner.solve_implicit`, whose gradient
    comes from the analytic adjoint, not from differentiating this loop).

    False: run a fixed-length, reverse-mode-differentiable form instead --
    `jax.lax.scan` for exactly ``n_iters`` steps, optionally split so the first
    ``n_iters - unroll_tail`` steps run under `stop_gradient` and only the last
    ``unroll_tail`` are actually unrolled and differentiated (Domke 2012
    truncated unrolling, same head/tail split as `ioc.inner.solve_unrolled`
    used with its retired internal Gauss-Newton loop). This is the form to use
    wherever the *caller* needs to backprop through the solve itself."""

    unroll_tail: int = 0
    """Only used when ``early_stop=False``. Number of trailing steps that carry
    gradients; 0 (or >= n_iters) unrolls and differentiates the whole solve."""


def dynamics_trajopt(
    x0: Float[Array, "n"],
    cost_fn: Callable[[Float[Array, "n"]], Array],
    opt_cfg: DynamicsTrajOptConfig = DynamicsTrajOptConfig(),
) -> Float[Array, "n"]:
    """Minimize ``cost_fn`` over a flat decision vector with L-BFGS.

    Same carry/step/line-search shape as ``_sco_optimization._lbfgs_inner_solve``
    (Nocedal two-loop direction, 5-point backtracking line search, best-iterate
    tracking), closing over an arbitrary ``cost_fn`` rather than a hardwired
    collision-linearization objective.
    """
    m = opt_cfg.m_lbfgs
    n = x0.shape[0]

    cost0, g0 = jax.value_and_grad(cost_fn)(x0)

    init_carry = (
        x0,                # current iterate
        x0,                # best-STATIONARITY iterate seen so far (gradient computed AT this x)
        jnp.max(jnp.abs(g0)),  # best max|grad| seen so far
        x0,                # x_prev  (dummy for iter 0)
        g0,                # g_prev  (dummy for iter 0)
        jnp.zeros((m, n)),  # s_buf
        jnp.zeros((m, n)),  # y_buf
        jnp.zeros(m),       # rho_buf
        jnp.int32(0),        # m_used
        jnp.int32(0),        # newest
        jnp.int32(0),        # iter_count
        jnp.bool_(False),    # converged
    )

    def lbfgs_step(carry):
        (x, best_x, best_gnorm,
         x_prev, g_prev,
         s_buf, y_buf, rho_buf,
         m_used, newest, iter_count, _) = carry

        cost_val, g = jax.value_and_grad(cost_fn)(x)
        gnorm = jnp.max(jnp.abs(g))

        # Track the best point by the gradient actually measured AT it -- not
        # by cost at line-search trial points, which are never re-differentiated
        # and so were previously returned with no gradient guarantee at all
        # (the grad_tol check below ran on `x`, but the function returned
        # `best_x`, a possibly-unrelated line-search trial point; a solve could
        # report converged while handing back a non-stationary point).
        improved = gnorm < best_gnorm
        best_x = jnp.where(improved, x, best_x)
        best_gnorm = jnp.where(improved, gnorm, best_gnorm)

        s_k = x - x_prev
        y_k = g - g_prev
        sy = jnp.dot(s_k, y_k)
        yy = jnp.dot(y_k, y_k)
        valid = (sy > 1e-10 * yy + 1e-30) & (iter_count > 0)

        new_newest = (newest + 1) % m
        actual_newest = jnp.where(valid, new_newest, newest)
        s_buf = s_buf.at[new_newest].set(jnp.where(valid, s_k, s_buf[new_newest]))
        y_buf = y_buf.at[new_newest].set(jnp.where(valid, y_k, y_buf[new_newest]))
        # Guard the reciprocal itself (not just its result) against sy ~ 0 on
        # the invalid branch: `jnp.where` still differentiates both branches,
        # so an unguarded `1/(sy+eps)` blows up at sy=0 (iter 0, s_k=0) and
        # leaks a NaN gradient through the select even though that branch is
        # never selected. Only matters under `early_stop=False`, where this
        # loop is actually differentiated.
        safe_sy = jnp.where(valid, sy, 1.0)
        rho_buf = jnp.where(valid, rho_buf.at[new_newest].set(1.0 / (safe_sy + 1e-30)), rho_buf)
        m_used = jnp.where(valid & (m_used < m), m_used + 1, m_used)
        newest = actual_newest

        dir_lbfgs = _lbfgs_two_loop(g, s_buf, y_buf, rho_buf, m_used, newest, m)
        # `jnp.linalg.norm(g)` has an undefined gradient at g=0 (0/0 inside
        # sqrt's derivative); once `m_used > 0` this branch is unselected but
        # `where`'s backward still differentiates it and 0*nan=nan leaks
        # through, so this smooth epsilon-inside-sqrt form is required (not
        # just cosmetic) whenever `early_stop=False` differentiates this step.
        dir_gd = -g / jnp.sqrt(jnp.sum(g**2) + 1e-18)
        direction = jnp.where(m_used > 0, dir_lbfgs, dir_gd)

        suff_thresh = cost_val * (1.0 - 1e-4)
        trial_costs = jax.vmap(lambda a: cost_fn(x + a * direction))(_LS_ALPHAS)
        has_suff = trial_costs < suff_thresh
        best_idx = jnp.where(
            jnp.any(has_suff),
            jnp.argmax(has_suff),
            jnp.argmin(trial_costs),
        )
        alpha = _LS_ALPHAS[best_idx]
        x_new = x + alpha * direction

        converged = (
            gnorm < opt_cfg.grad_tol
            if opt_cfg.grad_tol > 0.0
            else jnp.bool_(False)
        )

        return (
            x_new, best_x, best_gnorm,
            x, g,
            s_buf, y_buf, rho_buf,
            m_used, newest, iter_count + 1,
            converged,
        )

    if opt_cfg.early_stop:
        # `while_loop` (not a fixed-length `scan`) so a solve that hits
        # `grad_tol` stops paying for the remaining budget. Safe here because
        # this form is only ever called under `stop_gradient` (e.g. inside
        # `ioc.inner.solve_implicit`, whose gradient comes from the analytic
        # implicit-function-theorem `_bwd`, not from differentiating this loop)
        # -- reverse-mode AD through `while_loop` is unsupported by JAX
        # regardless, so this branch must never be differentiated through.
        def cond_fn(carry):
            iter_count, converged = carry[10], carry[-1]
            return jnp.logical_and(iter_count < opt_cfg.n_iters, jnp.logical_not(converged))

        final_carry = jax.lax.while_loop(cond_fn, lbfgs_step, init_carry)
    else:
        # Fixed-length, reverse-mode-differentiable form: `n_iters` L-BFGS
        # steps via `scan`, with the head (`n_iters - unroll_tail` steps)
        # under `stop_gradient` and only the tail actually unrolled --
        # truncated unrolling (Domke 2012), same head/tail split
        # `ioc.inner.solve_unrolled`'s retired Gauss-Newton loop used.
        tail = opt_cfg.unroll_tail if 0 < opt_cfg.unroll_tail < opt_cfg.n_iters else opt_cfg.n_iters
        n_head = opt_cfg.n_iters - tail

        def scan_step(carry, _):
            return lbfgs_step(carry), None

        head_carry, _ = jax.lax.scan(scan_step, init_carry, None, length=n_head)
        carry = jax.lax.stop_gradient(head_carry)
        for _ in range(tail):
            carry = jax.checkpoint(lbfgs_step)(carry)
        # Return the *final* iterate, not best-so-far: `best_x`/`best_gnorm`
        # are updated inside the stop_gradiented head, so once the head
        # converges the tail's differentiable steps almost never improve on
        # an already-tiny gradient norm, and `best_x` silently stays pinned to
        # a value with no gradient path through theta at all (measured: exact
        # zero gradient, cos=0 against FD, no error raised). Domke unrolling
        # differentiates the trajectory the tail actually walks, so the last
        # point of that walk is the only correct thing to return here.
        return carry[0]

    best_x = final_carry[1]

    return best_x
