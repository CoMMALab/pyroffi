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

    soft_line_search: bool = False
    """Opt-in, default False (every existing caller unaffected).

    The line search below picks among ``_LS_ALPHAS`` by
    ``best_idx = argmax(has_suff)`` -- a hard, discrete selection.  When this
    solve sits inside a larger differentiable pipeline whose INPUTS (not its
    own theta) shift the trajectory continuously -- e.g. a boundary condition
    fed in from an upstream differentiable IK stage, as in ``iosp.pickplace``
    -- an infinitesimal shift in that input can flip which alpha first
    satisfies sufficient decrease, and the flip compounds across ``n_iters``
    steps into a genuinely discontinuous final iterate.  This is the same
    failure class as the documented "collision hard-max nonsmooth" bug and
    ``clearance_residual``'s soft-min fix in ``ioc.robot.problem``: a hard
    argmax/max hiding a discontinuity that breaks implicit differentiation's
    precondition that the solution vary smoothly with the inputs.  Setting
    this True replaces the hard argmax with a softmax blend over the trial
    costs (see the call site) at a fixed temperature -- enough to kill the
    discontinuity, not a fancy line search."""

    soft_curvature_gate: bool = False
    """Opt-in, default False (every existing caller unaffected).

    Separate flag from `soft_line_search` -- different mechanism, same failure
    class.  The hard version admits/rejects a curvature pair with
    ``valid = (sy > 1e-10*yy + 1e-30) & (iter_count > 0)``: whether a pair
    crosses that threshold depends on trajectory values that vary continuously
    with an upstream boundary condition (e.g. `iosp.pickplace`'s IK-derived
    `q_pick`/`q_place`), so a pair can flip discretely from
    admitted/full-weight to rejected/zero-weight as that boundary shifts
    infinitesimally, and the flip compounds across ``n_iters`` steps.  It also
    gates a hard ``direction = where(m_used > 0, dir_lbfgs, dir_gd)`` switch.

    Setting this True: (1) always writes the ring buffer (no discrete
    admit/reject on `s_buf`/`y_buf`/`rho_buf`, so their contents are a smooth
    function of `sy`/`yy`); (2) makes `m_used` a deterministic function of the
    iteration index alone (`min(iter_count+1, m)`) instead of a
    trajectory-dependent counter, which makes the old `m_used > 0` switch
    theta-independent and therefore harmless without its own smoothing; (3)
    replaces the curvature *quality* signal with a smooth blend weight
    `w = sigmoid((sy - 1e-10*yy) / tau)` and interpolates
    `direction = w * dir_lbfgs + (1-w) * dir_gd` instead of a hard switch on
    pair validity."""


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

        if opt_cfg.soft_curvature_gate:
            # Always advance the ring buffer and always store the pair -- no
            # discrete admit/reject, so s_buf/y_buf/rho_buf are smooth
            # functions of sy/yy (see the config flag's docstring).  `newest`
            # and `m_used` are iteration-index-only (not trajectory-value
            # dependent), so they stay exact integers safely: nothing here
            # depends on a theta-varying comparison.
            new_newest = (newest + 1) % m
            s_buf = s_buf.at[new_newest].set(s_k)
            y_buf = y_buf.at[new_newest].set(y_k)
            safe_sy = jnp.where(jnp.abs(sy) > 1e-30, sy, 1.0)
            rho_buf = rho_buf.at[new_newest].set(1.0 / (safe_sy + 1e-30))
            m_used = jnp.minimum(iter_count + 1, m)
            newest = new_newest

            dir_lbfgs = _lbfgs_two_loop(g, s_buf, y_buf, rho_buf, m_used, newest, m)
            dir_gd = -g / jnp.sqrt(jnp.sum(g**2) + 1e-18)
            # Smooth curvature-quality weight replacing the hard `valid` gate
            # and the hard `m_used > 0` direction switch.  tau is scaled to
            # THIS pair's own (sy, yy) magnitude -- not a fixed constant --
            # for the same reason `soft_line_search`'s temperature had to be
            # rescaled: a fixed-magnitude tau either saturates back to a hard
            # gate (too small relative to sy) or melts the weight to a
            # constant ~0.5 regardless of curvature quality (too large).
            margin_curv = sy - 1e-10 * yy
            tau_curv = 0.1 * (jnp.abs(sy) + 1e-10 * jnp.abs(yy)) + 1e-12
            w = jax.nn.sigmoid(margin_curv / tau_curv) * (iter_count > 0)
            direction = w * dir_lbfgs + (1.0 - w) * dir_gd
        else:
            valid = (sy > 1e-10 * yy + 1e-30) & (iter_count > 0)

            new_newest = (newest + 1) % m
            actual_newest = jnp.where(valid, new_newest, newest)
            s_buf = s_buf.at[new_newest].set(jnp.where(valid, s_k, s_buf[new_newest]))
            y_buf = y_buf.at[new_newest].set(jnp.where(valid, y_k, y_buf[new_newest]))
            # Guard the reciprocal itself (not just its result) against sy ~ 0
            # on the invalid branch: `jnp.where` still differentiates both
            # branches, so an unguarded `1/(sy+eps)` blows up at sy=0 (iter 0,
            # s_k=0) and leaks a NaN gradient through the select even though
            # that branch is never selected. Only matters under
            # `early_stop=False`, where this loop is actually differentiated.
            safe_sy = jnp.where(valid, sy, 1.0)
            rho_buf = jnp.where(valid, rho_buf.at[new_newest].set(1.0 / (safe_sy + 1e-30)), rho_buf)
            m_used = jnp.where(valid & (m_used < m), m_used + 1, m_used)
            newest = actual_newest

            dir_lbfgs = _lbfgs_two_loop(g, s_buf, y_buf, rho_buf, m_used, newest, m)
            # `jnp.linalg.norm(g)` has an undefined gradient at g=0 (0/0 inside
            # sqrt's derivative); once `m_used > 0` this branch is unselected
            # but `where`'s backward still differentiates it and 0*nan=nan
            # leaks through, so this smooth epsilon-inside-sqrt form is
            # required (not just cosmetic) whenever `early_stop=False`
            # differentiates this step.
            dir_gd = -g / jnp.sqrt(jnp.sum(g**2) + 1e-18)
            direction = jnp.where(m_used > 0, dir_lbfgs, dir_gd)

        suff_thresh = cost_val * (1.0 - 1e-4)
        trial_costs = jax.vmap(lambda a: cost_fn(x + a * direction))(_LS_ALPHAS)

        if opt_cfg.soft_line_search:
            # Softmax over the signed sufficient-decrease margin instead of a
            # hard argmax -- see the config flag's docstring.
            #
            # MEASURED bug in an earlier version of this: scaling the
            # temperature by `|cost_val| * 1e-2` used the wrong reference
            # scale.  `cost_val` (the pre-step cost, e.g. O(1) or O(1e4)) has
            # no fixed relationship to the SPREAD across the 5 trial costs --
            # what actually determines whether softmax saturates.  Measured
            # case: cost_val ~ 5.7, margin spread ~ 4.0 across trials, but
            # tau = cost_val*1e-2 ~ 0.057 -> margin/tau up to 72 ->
            # softmax weights [1.0, 6e-15, ...] -- numerically bit-identical
            # to hard argmax despite the graph genuinely containing a softmax
            # (confirmed via `jax.make_jaxpr`: argmax present in the hard
            # path, absent in this one) -- a "the flag reached the graph but
            # the graph is numerically degenerate" failure, not a wiring bug.
            # Fixed by scaling tau to the trial costs' OWN spread instead, so
            # the blend is always meaningfully soft regardless of the
            # absolute cost magnitude.
            margin = suff_thresh - trial_costs
            spread = jnp.max(trial_costs) - jnp.min(trial_costs)
            tau = 0.1 * spread + 1e-6
            weights = jax.nn.softmax(margin / tau)
            alpha = jnp.dot(weights, _LS_ALPHAS)
        else:
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
