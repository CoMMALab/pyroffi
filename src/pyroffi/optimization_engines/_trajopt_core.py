"""Shared primitives for the dynamics-aware trajopt tiers.

This module holds the machinery that the four trajopt engines
(:mod:`_dynamics_trajopt`, :mod:`_sco_optimization`, :mod:`_flat_contact_trajopt`,
:mod:`_contact_rich_trajopt`) used to each carry a private, near-identical copy
of:

* :data:`_LS_ALPHAS` / :func:`_lbfgs_two_loop` — the line-search step ladder and
  the Nocedal two-loop recursion (canonical home; the other modules re-export
  these for backward-compatible imports).
* :func:`_lbfgs_driver` — the one L-BFGS step loop, generalizing the most
  complete of the four copies (``_dynamics_trajopt``): ``while_loop`` early stop
  vs. fixed-length differentiable ``scan`` with Domke-2012 ``unroll_tail``,
  optional ``soft_line_search`` / ``soft_curvature_gate`` smoothing, an optional
  endpoint mask, and a choice of best-iterate criterion (gradient norm vs. cost).
* :func:`_projected_gd` — SPaSM-style box-projected gradient descent (moved
  verbatim from ``_dynamics_trajopt``).
* :class:`AugmentedLagrangianTerm` / :func:`_al_outer_loop` — a generic
  augmented-Lagrangian outer loop over an arbitrary tuple of constraint-residual
  functions, generalizing the two hardcoded terms in ``_contact_rich_trajopt``
  (grasp closure, object Newton-Euler) to an arbitrary list, and giving SCO a
  real dual-ascent AL loop in place of its penalty-continuation-only outer loop.

The extraction is behavior-preserving: each engine's inner solve reduces to a
thin :func:`_lbfgs_driver` call with just its cost closure, best-criterion and
loop form swapped in, so every existing caller's numbers are unchanged at the
default (off) settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

# ---------------------------------------------------------------------------
# Line-search step sizes (mirrors _ik_primitives._LS_ALPHAS)
# ---------------------------------------------------------------------------

_LS_ALPHAS = jnp.array([1.0, 0.5, 0.25, 0.1, 0.025])


# ---------------------------------------------------------------------------
# L-BFGS two-loop recursion  (Nocedal; canonical home)
# ---------------------------------------------------------------------------

def _lbfgs_two_loop(
    g:       Float[Array, "n"],
    s_buf:   Float[Array, "m n"],
    y_buf:   Float[Array, "m n"],
    rho_buf: Float[Array, "m"],
    m_used:  Array,          # traced int32
    newest:  Array,          # traced int32
    m_lbfgs: int,            # static Python int — loops are unrolled
) -> Float[Array, "n"]:
    """Nocedal two-loop recursion returning the L-BFGS search direction -H*g.

    Inactive history slots are masked to no-ops so the function is safe
    to call before the history is fully populated.  When m_used == 0 the
    result is the zero vector; the caller should fall back to -g/||g||.
    """
    alpha_arr = jnp.zeros(m_lbfgs)
    q = g

    for i in range(m_lbfgs):
        buf_idx = (newest - i + m_lbfgs) % m_lbfgs
        active  = i < m_used
        si      = s_buf[buf_idx]
        yi      = y_buf[buf_idx]
        rho_i   = rho_buf[buf_idx]
        alpha_i = rho_i * jnp.dot(si, q)
        alpha_arr = jnp.where(active, alpha_arr.at[buf_idx].set(alpha_i), alpha_arr)
        q = jnp.where(active, q - alpha_i * yi, q)

    # Shanno-Kettler H₀ scaling from the most recent pair
    sy    = jnp.dot(s_buf[newest], y_buf[newest])
    yy    = jnp.dot(y_buf[newest], y_buf[newest])
    gamma = sy / (yy + 1e-18)
    r     = gamma * q

    for step in range(m_lbfgs):
        buf_idx = (newest - m_used + 1 + step + m_lbfgs) % m_lbfgs
        active  = step < m_used
        si      = s_buf[buf_idx]
        yi      = y_buf[buf_idx]
        rho_i   = rho_buf[buf_idx]
        alpha_i = alpha_arr[buf_idx]
        beta    = rho_i * jnp.dot(yi, r)
        r       = jnp.where(active, r + si * (alpha_i - beta), r)

    return -r


# ---------------------------------------------------------------------------
# Shared L-BFGS driver
# ---------------------------------------------------------------------------

def _gd_direction(g: Array, form: str) -> Array:
    """Normalized steepest-descent fallback direction.

    Two eps-guarded forms, kept distinct because the four engines were calibrated
    against different ones: ``"sqrt"`` (``-g / sqrt(sum g² + eps)``, the tier-1
    ``_dynamics_trajopt`` form — smooth at g=0, safe to differentiate) and
    ``"norm"`` (``-g / (||g|| + eps)``, the SCO / flat / contact-rich form). They
    agree to ~1e-9 but not bit-for-bit, and the contact solvers' pass/fail
    thresholds are tight enough to see the difference.
    """
    if form == "norm":
        return -g / (jnp.linalg.norm(g) + 1e-18)
    return -g / jnp.sqrt(jnp.sum(g**2) + 1e-18)


def _lbfgs_driver(
    x0: Float[Array, "n"],
    cost_fn: Callable[[Float[Array, "n"]], Array],
    *,
    n_iters: int,
    m_lbfgs: int,
    grad_tol: float = 0.0,
    loop: str = "while",              # "while" | "scan" | "unroll"
    unroll_tail: int = 0,
    soft_line_search: bool = False,
    soft_curvature_gate: bool = False,
    endpoint_mask: Float[Array, "n"] | None = None,
    best_by: str = "grad",            # "grad" | "cost"
    gd_dir: str = "sqrt",             # "sqrt" | "norm"
) -> Float[Array, "n"]:
    """One canonical L-BFGS solve, generalizing the four private copies.

    ``loop``:
      * ``"while"`` — ``lax.while_loop`` capped at ``n_iters``, exits early once
        ``max|grad| < grad_tol`` (0 disables). Returns the best iterate. NOT
        reverse-mode differentiable (data-dependent trip count) — only call under
        ``stop_gradient``.
      * ``"scan"`` — fixed-length ``lax.scan`` for exactly ``n_iters`` steps.
        Returns the best iterate. Differentiable, but differentiating the whole
        thing; use ``"unroll"`` for the truncated form.
      * ``"unroll"`` — fixed-length ``scan`` with the first ``n_iters -
        unroll_tail`` steps under ``stop_gradient`` and the last ``unroll_tail``
        actually unrolled and checkpointed (Domke 2012). Returns the FINAL
        iterate (not best-so-far — the tail differentiates the walk it took).

    ``best_by``: ``"grad"`` tracks the iterate with the smallest ``max|grad|``
    measured AT it (before the step); ``"cost"`` tracks the smallest post-step
    line-search cost. Both are ignored for ``loop="unroll"``, which returns the
    final iterate.

    ``endpoint_mask``: multiplies both the gradient and the step direction, to
    pin masked-out coordinates (start/goal waypoints). ``None`` ⇒ no masking.
    """
    m = m_lbfgs
    n = x0.shape[0]
    mask = jnp.ones(n, x0.dtype) if endpoint_mask is None else endpoint_mask

    cost0, g0 = jax.value_and_grad(cost_fn)(x0)
    g0 = g0 * mask

    init_carry = (
        x0,                          # 0 current iterate
        x0,                          # 1 best iterate seen so far
        (jnp.max(jnp.abs(g0)) if best_by == "grad" else cost0),  # 2 best metric
        x0,                          # 3 x_prev  (dummy for iter 0)
        g0,                          # 4 g_prev  (dummy for iter 0)
        jnp.zeros((m, n)),           # 5 s_buf
        jnp.zeros((m, n)),           # 6 y_buf
        jnp.zeros(m),                # 7 rho_buf
        jnp.int32(0),                # 8 m_used
        jnp.int32(0),                # 9 newest
        jnp.int32(0),                # 10 iter_count
        jnp.bool_(False),            # 11 converged
    )

    def lbfgs_step(carry):
        (x, best_x, best_metric,
         x_prev, g_prev,
         s_buf, y_buf, rho_buf,
         m_used, newest, iter_count, _) = carry

        cost_val, g = jax.value_and_grad(cost_fn)(x)
        g = g * mask
        gnorm = jnp.max(jnp.abs(g))

        if best_by == "grad":
            improved = gnorm < best_metric
            best_x = jnp.where(improved, x, best_x)
            best_metric = jnp.where(improved, gnorm, best_metric)

        s_k = x - x_prev
        y_k = g - g_prev
        sy = jnp.dot(s_k, y_k)
        yy = jnp.dot(y_k, y_k)

        if soft_curvature_gate:
            new_newest = (newest + 1) % m
            s_buf = s_buf.at[new_newest].set(s_k)
            y_buf = y_buf.at[new_newest].set(y_k)
            safe_sy = jnp.where(jnp.abs(sy) > 1e-30, sy, 1.0)
            rho_buf = rho_buf.at[new_newest].set(1.0 / (safe_sy + 1e-30))
            m_used = jnp.minimum(iter_count + 1, m)
            newest = new_newest

            dir_lbfgs = _lbfgs_two_loop(g, s_buf, y_buf, rho_buf, m_used, newest, m)
            dir_gd = _gd_direction(g, gd_dir)
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
            safe_sy = jnp.where(valid, sy, 1.0)
            rho_buf = jnp.where(valid, rho_buf.at[new_newest].set(1.0 / (safe_sy + 1e-30)), rho_buf)
            m_used = jnp.where(valid & (m_used < m), m_used + 1, m_used)
            newest = actual_newest

            dir_lbfgs = _lbfgs_two_loop(g, s_buf, y_buf, rho_buf, m_used, newest, m)
            dir_gd = _gd_direction(g, gd_dir)
            direction = jnp.where(m_used > 0, dir_lbfgs, dir_gd)

        direction = direction * mask

        suff_thresh = cost_val * (1.0 - 1e-4)
        trial_costs = jax.vmap(lambda a: cost_fn(x + a * direction))(_LS_ALPHAS)

        if soft_line_search:
            margin = suff_thresh - trial_costs
            spread = jnp.max(trial_costs) - jnp.min(trial_costs)
            tau = 0.1 * spread + 1e-6
            weights = jax.nn.softmax(margin / tau)
            alpha = jnp.dot(weights, _LS_ALPHAS)
            x_new = x + alpha * direction
            new_cost = cost_fn(x_new)
        else:
            has_suff = trial_costs < suff_thresh
            best_idx = jnp.where(
                jnp.any(has_suff),
                jnp.argmax(has_suff),
                jnp.argmin(trial_costs),
            )
            alpha = _LS_ALPHAS[best_idx]
            x_new = x + alpha * direction
            new_cost = trial_costs[best_idx]

        if best_by == "cost":
            improved = new_cost < best_metric
            best_x = jnp.where(improved, x_new, best_x)
            best_metric = jnp.where(improved, new_cost, best_metric)

        converged = (
            gnorm < grad_tol if grad_tol > 0.0 else jnp.bool_(False)
        )

        return (
            x_new, best_x, best_metric,
            x, g,
            s_buf, y_buf, rho_buf,
            m_used, newest, iter_count + 1,
            converged,
        )

    if loop == "while":
        def cond_fn(carry):
            iter_count, converged = carry[10], carry[-1]
            return jnp.logical_and(iter_count < n_iters, jnp.logical_not(converged))

        final_carry = jax.lax.while_loop(cond_fn, lbfgs_step, init_carry)
        return final_carry[1]

    if loop == "scan":
        def scan_step(carry, _):
            return lbfgs_step(carry), None

        final_carry, _ = jax.lax.scan(scan_step, init_carry, None, length=n_iters)
        return final_carry[1]

    # loop == "unroll": Domke truncated unrolling, returns the FINAL iterate.
    tail = unroll_tail if 0 < unroll_tail < n_iters else n_iters
    n_head = n_iters - tail

    def scan_step(carry, _):
        return lbfgs_step(carry), None

    head_carry, _ = jax.lax.scan(scan_step, init_carry, None, length=n_head)
    carry = jax.lax.stop_gradient(head_carry)
    for _ in range(tail):
        carry = jax.checkpoint(lbfgs_step)(carry)
    return carry[0]


# ---------------------------------------------------------------------------
# Projected gradient descent (SPaSM-style)
# ---------------------------------------------------------------------------

def _projected_gd(x0, cost_fn, *, n_iters, gd_lr, q_lo=(), q_hi=(), dof=0):
    """SPaSM-style projected gradient descent: GD from the feasible seed with a
    linearly-decayed step, clipped onto the joint box after each step.

    Fixed-length ``lax.scan``, so it is reverse-mode differentiable like the
    L-BFGS ``loop="unroll"`` path (the box projection is subdifferentiable).
    Endpoints/pinned waypoints are handled by the caller's ``unpack``, so ``x0``
    is the interior waypoints only; the box projection keeps them in-limits.
    """
    project = len(q_lo) > 0 and dof > 0
    if project:
        lo = jnp.asarray(q_lo, dtype=x0.dtype)
        hi = jnp.asarray(q_hi, dtype=x0.dtype)

    grad_fn = jax.grad(cost_fn)

    def step(x, i):
        g = grad_fn(x)
        lr = gd_lr * (1.0 - i / n_iters)
        x = x - lr * g
        if project:
            x = jnp.clip(x.reshape(-1, dof), lo, hi).reshape(-1)
        return x, None

    x_final, _ = jax.lax.scan(step, x0, jnp.arange(n_iters, dtype=x0.dtype))
    return x_final


# ---------------------------------------------------------------------------
# Generic augmented-Lagrangian outer loop
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrustRegionConfig:
    """Adaptive (Schulman et al. 2013) trust-region sizing for :func:`_al_outer_loop`.

    The trust region is a soft quadratic ``coef * ||z - z_k||²`` added to the
    inner subproblem; ``coef`` is adjusted each outer iteration by the standard
    ratio test on *actual vs. predicted* improvement of the AL merit (predicted =
    the linearized-constraint model the subproblem actually optimizes):

        ratio = (merit_true(z_k) - merit_true(z_trial))
                / (merit_model(z_k) - merit_model(z_trial))

    * ``ratio < shrink_ratio`` — the linear model over-promised: **reject** the
      step (keep ``z_k``) and **tighten** the region (``coef *= tighten``).
    * ``ratio > expand_ratio`` — the model was accurate: accept and **loosen**
      (``coef *= loosen``).
    * otherwise — accept, region unchanged.

    A step is accepted iff ``ratio > accept_ratio`` and the model predicted a
    real decrease. ``coef`` is clamped to ``[coef_min, coef_max]``. Larger
    ``coef`` ⇒ smaller effective step, so ``tighten > 1`` and ``loosen < 1``.
    """

    coef0: float = 1.0
    """Initial trust-region penalty coefficient."""
    tighten: float = 4.0
    """Multiplier applied to ``coef`` on a rejected / poor step (``> 1``)."""
    loosen: float = 0.25
    """Multiplier applied to ``coef`` on a good step (``< 1``)."""
    shrink_ratio: float = 0.25
    """Below this ratio the step is rejected and the region tightened."""
    expand_ratio: float = 0.75
    """Above this ratio the region is loosened."""
    accept_ratio: float = 0.1
    """A step is accepted iff its ratio exceeds this (and the model improved)."""
    coef_min: float = 1e-2
    coef_max: float = 1e4


@dataclass(frozen=True)
class AugmentedLagrangianTerm:
    """One constraint folded into :func:`_al_outer_loop` as an AL term.

    ``residual_fn(z) -> Array`` returns the (flat or shaped) constraint residual
    at the current decision vector; the AL loop keeps a matching dual and penalty
    for it.

    * ``kind="eq"``   — equality ``residual = 0``. The AL term is
      ``dual . r + 0.5 * rho * ||r||²`` and the dual ascends by
      ``dual += dual_scale * rho * r``.
    * ``kind="ineq"`` — inequality ``residual <= 0`` (write a ``>= 0`` constraint
      as ``-r``). Uses the standard projected form: the active residual is
      ``max(r, -dual/rho)``, the penalty is ``0.5 * rho * active²`` (equivalently
      the Rockafellar ``(max(0, dual + rho*r)² - dual²)/(2 rho)``), and the dual
      is projected onto ``>= 0`` after ascent.

    ``rho0`` / ``rho_max`` / ``penalty_scale`` control this term's own penalty
    continuation. ``name`` is cosmetic.
    """

    residual_fn: Callable[[Array], Array]
    kind: str = "eq"
    rho0: float = 1.0
    rho_max: float = 1e4
    penalty_scale: float = 2.0
    name: str = ""


def _al_penalty(term: AugmentedLagrangianTerm, r: Array, dual: Array, rho: Array) -> Array:
    """Augmented-Lagrangian contribution of one term to the inner cost."""
    if term.kind == "eq":
        return jnp.sum(dual * r) + 0.5 * rho * jnp.sum(r**2)
    # inequality r <= 0, Rockafellar shifted form.
    active = jnp.maximum(r, -dual / rho)
    return jnp.sum(dual * active) + 0.5 * rho * jnp.sum(active**2)


def _make_trust(cfg) -> "TrustRegionConfig | None":
    """Build a :class:`TrustRegionConfig` from an engine config's ``adaptive_trust``
    + ``tr_*`` fields, or ``None`` when adaptive TR is off. Lets every engine wire
    adaptive trust regions from the same field block."""
    if not getattr(cfg, "adaptive_trust", False):
        return None
    return TrustRegionConfig(
        coef0=cfg.tr_coef0, tighten=cfg.tr_tighten, loosen=cfg.tr_loosen,
        shrink_ratio=cfg.tr_shrink_ratio, expand_ratio=cfg.tr_expand_ratio,
        accept_ratio=cfg.tr_accept_ratio,
        coef_min=cfg.tr_coef_min, coef_max=cfg.tr_coef_max,
    )


def _adaptive_trust_step(z_k, z_trial, m_zk, m_model, m_true, tr_coef, trust):
    """One Schulman trust-region ratio-test update, shared by every engine.

    Given the merit at the outer iterate (``m_zk``), at the trial point under the
    *linearized* model (``m_model``) and under the *true* cost (``m_true``), plus
    the current trust coefficient, returns ``(z_next, tr_coef, accept)``:

    * ``predicted = m_zk - m_model`` (the decrease the convex model promised),
      ``actual = m_zk - m_true`` (what actually happened);
    * reject (keep ``z_k``) unless ``actual/predicted > accept_ratio`` and the
      model predicted a real decrease;
    * tighten ``tr_coef`` (``*= tighten``) when the ratio is below
      ``shrink_ratio``, loosen (``*= loosen``) above ``expand_ratio``, clamp to
      ``[coef_min, coef_max]``.
    """
    predicted = m_zk - m_model
    actual = m_zk - m_true
    ratio = actual / (predicted + 1e-12)
    improved = predicted > 1e-12
    accept = improved & (ratio > trust.accept_ratio)
    z_next = jnp.where(accept, z_trial, z_k)
    tr_coef = jnp.where(
        ratio < trust.shrink_ratio, tr_coef * trust.tighten,
        jnp.where(ratio > trust.expand_ratio, tr_coef * trust.loosen, tr_coef),
    )
    tr_coef = jnp.clip(tr_coef, trust.coef_min, trust.coef_max)
    return z_next, tr_coef, accept


def _al_dual_update(
    term: AugmentedLagrangianTerm, r: Array, dual: Array, rho: Array, dual_scale: float
) -> Array:
    if term.kind == "eq":
        return dual + dual_scale * rho * r
    # inequality: ascend then project onto the non-negative orthant.
    return jnp.maximum(0.0, dual + dual_scale * rho * r)


def _al_outer_loop(
    z0: Float[Array, "nz"],
    inner_solve_fn: Callable[[Array, Callable[[Array], Array]], Array],
    constraints: tuple[AugmentedLagrangianTerm, ...],
    base_cost_fn: Callable[[Array, Array], Array],
    *,
    n_outer_iters: int,
    dual_scale: float = 1.0,
    constraint_tol: float = 0.0,
    repin_fn: Callable[[Array], Array] | None = None,
    sco_linearize: bool = False,
    trust: "TrustRegionConfig | None" = None,
    return_solve_multipliers: bool = False,
) -> tuple[Array, tuple, tuple]:
    """Generic AL outer loop over an arbitrary tuple of constraint terms.

    Each iteration:
      1. Build the AL inner cost = ``base_cost_fn(z, z_k)`` + Σ term penalties,
         solve it with ``inner_solve_fn(z, al_cost_fn) -> z``. ``z_k`` is the
         (``stop_gradient``'d) outer iterate the inner solve starts from, passed
         to the base cost so an SCO trust region ``||z - z_k||²`` can be written
         there. Base costs that don't need it just ignore the second argument.
      2. Optionally re-pin endpoints via ``repin_fn`` (numerical safety).
      3. Dual ascent + penalty continuation per term.
      4. If ``constraint_tol > 0``, stop once every term's ``max|r|`` is below it.

    Mirrors ``_contact_rich_trajopt``'s ``outer_body``/``outer_cond`` exactly,
    generalized from two hardcoded terms to an arbitrary tuple. With
    ``constraints=()`` it runs ``inner_solve_fn`` once on ``base_cost_fn`` per
    outer iteration with no duals — i.e. plain penalty-free minimization (SCO's
    penalty continuation lives in ``base_cost_fn``'s closed-over weights, so an
    empty tuple reproduces the continuation-only behavior).

    Returns ``(z, duals, rhos)`` — the final iterate and each term's final dual
    and penalty, so the caller can report diagnostics.

    ``return_solve_multipliers``: when True, the returned ``(duals, rhos)`` are
    instead the multipliers the returned ``z`` was actually *optimized against*
    (the values used in the inner solve that produced it), NOT the post-update
    continuation values. This matters for a frozen-multiplier implicit adjoint:
    ``z`` is stationary of ``base + penalty(duals_used, rhos_used)``, so an IFT
    that linearizes that stationarity must use exactly those multipliers — the
    post-update ``rho`` (grown by ``penalty_scale``) would put the wrong penalty
    curvature in the Hessian while leaving the gradient screen fooled (the dual
    barely moves once the residual is ~0). Default False preserves the exact
    diagnostic-reporting behavior every existing caller relies on.

    ``trust``: when a :class:`TrustRegionConfig` is passed, a soft trust region
    ``coef * ||z - z_k||²`` is added to each inner subproblem and ``coef`` is
    adjusted every outer iteration by the actual-vs-predicted ratio test
    (Schulman et al. 2013); a step whose ratio is too low is rejected and the
    region tightened. ``None`` (default) reproduces the fixed behavior exactly —
    the caller keeps its own (fixed) trust term in ``base_cost_fn`` and every
    step is accepted.

    Runs as a ``lax.while_loop`` (safe: only ever called under ``stop_gradient``,
    like the engines it backs). ``duals`` / ``rhos`` are homogeneous-shaped only
    per term, so they are carried as Python tuples folded into the carry.
    """
    nc = len(constraints)

    # Seed each dual to a zero of the residual's shape, each rho to rho0.
    duals0 = tuple(jnp.zeros_like(c.residual_fn(z0)) for c in constraints)
    rhos0 = tuple(jnp.asarray(c.rho0, jnp.float32) for c in constraints)
    tr_coef0 = jnp.asarray(trust.coef0 if trust is not None else 0.0, jnp.float32)

    def _term_residual(c, zk):
        """The residual used *inside* one inner solve: exact, or linearized about
        the frozen outer iterate ``zk`` for inequality terms under SCO."""
        if not (sco_linearize and c.kind == "ineq"):
            return c.residual_fn
        d0, jvp_fn = jax.linearize(c.residual_fn, zk)
        return lambda z: d0 + jvp_fn(z - zk)

    def al_cost_fn(z, zk, duals, rhos, tr_coef):
        cost = base_cost_fn(z, zk)
        for c, dual, rho in zip(constraints, duals, rhos):
            r = _term_residual(c, zk)(z)
            cost = cost + _al_penalty(c, r, dual, rho)
        if trust is not None:
            cost = cost + tr_coef * jnp.sum((z - zk) ** 2)
        return cost

    def _merit(z, zk, duals, rhos, linearize):
        """AL merit (no trust term) at ``z``; constraints exact or linearized
        about ``zk``. Used for the trust-region ratio test only."""
        cost = base_cost_fn(z, zk)
        for c, dual, rho in zip(constraints, duals, rhos):
            r = (_term_residual(c, zk) if linearize else c.residual_fn)(z)
            cost = cost + _al_penalty(c, r, dual, rho)
        return cost

    def body(carry):
        z, duals, rhos, _, it, tr_coef, solved_duals, solved_rhos = carry
        zk = jax.lax.stop_gradient(z)
        z_trial = inner_solve_fn(
            zk, lambda zz: al_cost_fn(zz, zk, duals, rhos, tr_coef)
        )
        if repin_fn is not None:
            z_trial = repin_fn(z_trial)

        if trust is not None:
            # Schulman ratio test: actual vs. predicted (linearized) improvement.
            m_zk = _merit(zk, zk, duals, rhos, linearize=True)
            m_model = _merit(z_trial, zk, duals, rhos, linearize=True)
            m_true = _merit(z_trial, zk, duals, rhos, linearize=False)
            z_next, tr_coef, accept = _adaptive_trust_step(
                zk, z_trial, m_zk, m_model, m_true, tr_coef, trust
            )
        else:
            z_next = z_trial
            accept = jnp.bool_(True)

        # Dual ascent + penalty continuation, only on an accepted step.
        new_duals, new_rhos, resid_ok = [], [], []
        for c, dual, rho in zip(constraints, duals, rhos):
            r = c.residual_fn(z_next)
            asc = _al_dual_update(c, r, dual, rho, dual_scale)
            new_duals.append(jnp.where(accept, asc, dual))
            new_rhos.append(
                jnp.where(accept, jnp.minimum(rho * c.penalty_scale, c.rho_max), rho)
            )
            resid_ok.append(jnp.max(jnp.abs(r)))
        new_duals = tuple(new_duals)
        new_rhos = tuple(new_rhos)

        # The multipliers ``z_next`` is stationary against: on an accepted step
        # z_next = z_trial, solved with (duals, rhos); on a rejected step
        # z_next = zk, whose solve multipliers are whatever they were before.
        new_solved_duals = tuple(
            jnp.where(accept, d, sd) for d, sd in zip(duals, solved_duals)
        )
        new_solved_rhos = tuple(
            jnp.where(accept, r, sr) for r, sr in zip(rhos, solved_rhos)
        )

        if constraint_tol > 0.0 and nc > 0:
            worst = jnp.max(jnp.stack(resid_ok))
            converged = worst < constraint_tol
        else:
            converged = jnp.bool_(False)
        return (z_next, new_duals, new_rhos, converged, it + 1, tr_coef,
                new_solved_duals, new_solved_rhos)

    def cond(carry):
        converged, it = carry[3], carry[4]
        return jnp.logical_and(it < n_outer_iters, jnp.logical_not(converged))

    z, duals, rhos, _, _, _, solved_duals, solved_rhos = jax.lax.while_loop(
        cond, body,
        (z0, duals0, rhos0, jnp.bool_(False), jnp.int32(0), tr_coef0, duals0, rhos0),
    )
    if return_solve_multipliers:
        return z, solved_duals, solved_rhos
    return z, duals, rhos
