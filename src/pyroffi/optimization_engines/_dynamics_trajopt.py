"""Tier 1 — dynamics-aware L-BFGS trajopt over an arbitrary cost.

Built on the shared :mod:`_trajopt_core` primitives. In its default
configuration this is exactly the contact-free, single-arm L-BFGS solver that
``ioc`` / ``iosp`` (SPaSM / inverse-SPaSM) call as their forward solver: a single
L-BFGS minimize of the caller's ``cost_fn(x) -> scalar``, tracking the
best-*stationarity* iterate, with the ``early_stop`` / ``unroll_tail`` /
``soft_line_search`` / ``soft_curvature_gate`` semantics the implicit-adjoint IOC
pipeline depends on.

Opt-in, all defaulting to preserve exact current behavior, it also exposes:

* ``constraints`` — an arbitrary tuple of
  :class:`~pyroffi.optimization_engines._trajopt_core.AugmentedLagrangianTerm`,
  folded into a generic AL outer loop. Empty ⇒ the single-solve behavior above.
* ``use_sco`` — Sequential Convex Optimization: each AL outer iteration
  linearizes every supplied *inequality* residual at the current iterate before
  the inner solve (à la Schulman et al. 2013), generalized beyond collision.
* ``robot`` / ``grid`` — when supplied, tier 1 auto-adds kinematic smoothness /
  joint-limit terms and a dynamics-feasibility (torque-limit) AL term, so a
  standalone caller gets a dynamics-feasible solve without hand-writing the cost.
  Existing ``ioc`` / ``iosp`` callers pass neither, so their graph is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from jax import Array
from jaxtyping import Float

from ._trajopt_core import (
    AugmentedLagrangianTerm,
    _al_outer_loop,
    _lbfgs_driver,
    _make_trust,
    _projected_gd,
)


@dataclass(frozen=True)
class DynamicsTrajOptConfig:
    """Hyper-parameters for the dynamics-aware L-BFGS trajopt engine."""

    n_iters: int = 200
    """Number of L-BFGS steps."""

    method: str = "lbfgs"
    """"lbfgs" (default, unchanged) or "projected_gd".  The projected-GD path is
    a SPaSM-style solver: plain gradient descent from the (already valid)
    straight-line seed with a linearly-decayed step, PROJECTED onto the joint
    box after every step.  It stays in the neighborhood of the feasible init and
    cannot diverge out of joint limits the way an unconstrained L-BFGS step can
    on the composed pick-place objective."""

    gd_lr: float = 0.1
    """projected_gd only: initial GD step, decayed linearly to 0 over n_iters."""

    q_lo: tuple = ()
    """projected_gd only: per-DOF lower joint limits (hashable tuple).  Empty
    disables the box projection (pure GD)."""

    q_hi: tuple = ()
    """projected_gd only: per-DOF upper joint limits (hashable tuple)."""

    dof: int = 0
    """projected_gd only: DOF, to reshape the flat iterate to (T_interior, dof)
    for per-waypoint box projection."""

    m_lbfgs: int = 8
    """L-BFGS history size (number of curvature pairs)."""

    grad_tol: float = 1e-6
    """Inner L-BFGS stops once ``max|grad|`` drops below this. 0 disables (falls
    back to the fixed ``n_iters`` budget). Only meaningful when
    ``early_stop=True``."""

    early_stop: bool = True
    """True: run the ``while_loop`` early-stopping form -- cheap, but not
    reverse-mode differentiable (JAX cannot backprop through a data-dependent
    trip count), so this form must only ever be called under `stop_gradient`.

    False: run a fixed-length, reverse-mode-differentiable form instead --
    `jax.lax.scan` for exactly ``n_iters`` steps, optionally split so only the
    last ``unroll_tail`` are actually unrolled and differentiated (Domke 2012
    truncated unrolling)."""

    unroll_tail: int = 0
    """Only used when ``early_stop=False``. Number of trailing steps that carry
    gradients; 0 (or >= n_iters) unrolls and differentiates the whole solve."""

    soft_line_search: bool = False
    """Opt-in, default False. Replaces the hard argmax over ``_LS_ALPHAS`` with a
    softmax blend over the trial costs, to keep the solve continuous in an
    upstream differentiable input (e.g. an IK boundary condition). See the shared
    driver for the mechanism."""

    soft_curvature_gate: bool = False
    """Opt-in, default False. Replaces the discrete curvature-pair admit/reject
    and the hard L-BFGS/GD direction switch with smooth blends, for the same
    reason as ``soft_line_search``. See the shared driver."""

    # --- Opt-in tier-1 SCO / augmented-Lagrangian extensions --------------
    #     All default to preserve the exact single-solve behavior above.

    constraints: tuple = ()
    """Tuple of
    :class:`~pyroffi.optimization_engines._trajopt_core.AugmentedLagrangianTerm`.
    Empty (default) ⇒ a single L-BFGS solve of ``cost_fn``, exactly as before.
    Non-empty ⇒ ``cost_fn`` becomes the AL *base* cost and these terms are folded
    into a generic AL outer loop with their own duals and penalties."""

    use_sco: bool = False
    """When True (and ``constraints`` non-empty), each AL outer iteration
    linearizes every *inequality* residual at the current iterate before the
    inner solve, à la Sequential Convex Optimization. Equality terms are left
    exact. No effect when ``constraints`` is empty."""

    n_outer_iters: int = 10
    """AL / SCO outer iterations (only used when ``constraints`` is non-empty)."""

    constraint_tol: float = 0.0
    """Outer loop stops once every term's ``max|residual|`` is below this. 0
    disables (runs the full ``n_outer_iters`` budget)."""

    dual_scale: float = 1.0
    """Scaling on the AL dual-ascent step."""

    robot: object = None
    """Optional :class:`~pyroffi.Robot`. When supplied together with ``grid`` (and
    ``dof`` for the reshape), tier 1 auto-adds kinematic smoothness / joint-limit
    terms and a dynamics-feasibility AL term for a standalone dynamics-feasible
    solve. Left ``None`` by every ``ioc`` / ``iosp`` caller."""

    grid: object = None
    """Optional :class:`~pyroffi.dynamics.GRiDDynamics`, paired with ``robot``."""

    tau_max: float = 87.0
    """Torque limit for the auto-added dynamics-feasibility term (robot/grid)."""

    auto_dt: float = 0.1
    """Timestep for finite-difference velocities/accelerations in the auto-added
    dynamics-feasibility term."""

    w_smooth: float = 1.0
    w_limits: float = 1.0
    """Weights for the auto-added kinematic terms (robot/grid path only)."""

    # --- Adaptive (Schulman) trust region for the AL/SCO outer loop --------
    #     Only active when ``constraints`` is non-empty (the AL path); best paired
    #     with ``use_sco`` (the ratio test judges the linearized model). Default
    #     off ⇒ no trust region, byte-identical to the plain AL loop.
    adaptive_trust: bool = False
    """Enable Schulman actual-vs-predicted trust-region resizing in the AL loop."""
    tr_coef0: float = 1.0
    """Initial trust-region coefficient."""
    tr_tighten: float = 4.0
    tr_loosen: float = 0.25
    tr_shrink_ratio: float = 0.25
    tr_expand_ratio: float = 0.75
    tr_accept_ratio: float = 0.1
    tr_coef_min: float = 1e-2
    tr_coef_max: float = 1e4


def dynamics_trajopt(
    x0: Float[Array, "n"],
    cost_fn: Callable[[Float[Array, "n"]], Array],
    opt_cfg: DynamicsTrajOptConfig = DynamicsTrajOptConfig(),
) -> Float[Array, "n"]:
    """Minimize ``cost_fn`` over a flat decision vector with L-BFGS.

    Default (``method="lbfgs"``, ``constraints=()``, ``robot=None``): a single
    L-BFGS solve tracking the best-stationarity iterate — the exact forward solver
    ``ioc.inner.make_inner_solver`` wires in. ``method="projected_gd"`` runs the
    SPaSM box-projected GD fallback instead.

    With ``constraints`` (and/or ``robot``/``grid``) supplied, the solve becomes
    a generic augmented-Lagrangian outer loop over those terms, optionally with
    SCO linearization of inequality residuals (``use_sco``).
    """
    if opt_cfg.method == "projected_gd":
        return _projected_gd(
            x0, cost_fn,
            n_iters=opt_cfg.n_iters, gd_lr=opt_cfg.gd_lr,
            q_lo=opt_cfg.q_lo, q_hi=opt_cfg.q_hi, dof=opt_cfg.dof,
        )

    loop = "while" if opt_cfg.early_stop else "unroll"

    def inner_solve(z0, cf):
        return _lbfgs_driver(
            z0, cf,
            n_iters=opt_cfg.n_iters, m_lbfgs=opt_cfg.m_lbfgs,
            grad_tol=opt_cfg.grad_tol, loop=loop,
            unroll_tail=opt_cfg.unroll_tail,
            soft_line_search=opt_cfg.soft_line_search,
            soft_curvature_gate=opt_cfg.soft_curvature_gate,
            best_by="grad",
        )

    constraints = _build_constraints(x0, opt_cfg)
    base_cost, extra = _augmented_base_cost(cost_fn, opt_cfg)

    if not constraints:
        # Exact legacy path: a single inner solve of the (possibly kinematic-
        # augmented) cost. With robot=None this is byte-identical to before.
        return inner_solve(x0, base_cost if extra else cost_fn)

    z, _, _ = _al_outer_loop(
        x0, inner_solve, constraints, lambda z, zk: base_cost(z),
        n_outer_iters=opt_cfg.n_outer_iters,
        dual_scale=opt_cfg.dual_scale,
        constraint_tol=opt_cfg.constraint_tol,
        sco_linearize=opt_cfg.use_sco,
        trust=_make_trust(opt_cfg),
    )
    return z


# ---------------------------------------------------------------------------
# Opt-in robot/grid auto-terms
# ---------------------------------------------------------------------------

def _augmented_base_cost(cost_fn, opt_cfg):
    """``(base_cost_fn, added_anything)``.

    When ``robot``/``grid`` are supplied, add kinematic smoothness + joint-limit
    penalties to the caller's cost (the AL base). Otherwise return ``cost_fn``
    unchanged and ``False`` (the legacy graph).
    """
    if opt_cfg.robot is None or opt_cfg.dof <= 0:
        return cost_fn, False

    from ._sco_optimization import _limits_cost, _smoothness_cost

    dof = opt_cfg.dof
    robot = opt_cfg.robot
    lower = robot.joints.lower_limits
    upper = robot.joints.upper_limits

    def base(x):
        t = x.reshape(-1, dof)
        c = cost_fn(x)
        c = c + opt_cfg.w_smooth * _smoothness_cost(t, 1.0, 0.5, 0.1)
        c = c + opt_cfg.w_limits * _limits_cost(t, lower, upper)
        return c

    return base, True


def _build_constraints(x0, opt_cfg) -> tuple:
    """Assemble the AL constraint tuple: caller-supplied plus, if robot/grid are
    given, an auto dynamics-feasibility (torque-limit) inequality term."""
    constraints = tuple(opt_cfg.constraints)
    if opt_cfg.robot is not None and opt_cfg.grid is not None and opt_cfg.dof > 0:
        from ..dynamics._contact import dynamics_feasibility_residual

        dof = opt_cfg.dof
        grid = opt_cfg.grid

        def residual(x):
            t = x.reshape(-1, dof)
            return dynamics_feasibility_residual(
                grid, t, opt_cfg.auto_dt, opt_cfg.tau_max
            )

        constraints = constraints + (
            AugmentedLagrangianTerm(
                residual_fn=residual, kind="ineq",
                rho0=1.0, rho_max=1e3, penalty_scale=2.0,
                name="torque_limit",
            ),
        )
    return constraints
