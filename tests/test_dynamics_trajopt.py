"""Tier-1 ``dynamics_trajopt`` + shared ``_trajopt_core`` correctness.

The default (``constraints=()``, ``robot=None``, ``method="lbfgs"``) path is the
forward solver ``ioc`` / ``iosp`` depend on, so the first tests lock in that it
still minimizes an arbitrary ``cost_fn`` and is reverse-mode differentiable in
its ``early_stop=False`` / ``unroll_tail`` form. The rest exercise the opt-in
augmented-Lagrangian interface (arbitrary equality / inequality terms, SCO
linearization) that the refactor adds.

CPU-only; no GPU or GRiD needed. The GRiD-backed dynamics-feasibility AL term is
covered (GPU-gated) in ``test_contact_rich_trajopt.py`` via the shared
``dynamics_feasibility_residual``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyroffi.optimization_engines import dynamics_trajopt, DynamicsTrajOptConfig
from pyroffi.optimization_engines._trajopt_core import (
    AugmentedLagrangianTerm,
    _al_outer_loop,
    _lbfgs_driver,
)


def _quadratic(n=20, seed=0):
    key = jax.random.PRNGKey(seed)
    A = jax.random.normal(key, (n, n))
    A = A @ A.T + jnp.eye(n)
    b = jax.random.normal(jax.random.PRNGKey(seed + 1), (n,))
    x_star = jnp.linalg.solve(A, b)
    return (lambda x: 0.5 * x @ A @ x - b @ x), x_star


# --------------------------------------------------------------------------- #
# Default (legacy) path                                                        #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("early_stop", [True, False])
def test_default_minimizes_quadratic(early_stop):
    cost, x_star = _quadratic()
    x0 = jnp.zeros_like(x_star)
    cfg = DynamicsTrajOptConfig(n_iters=200, early_stop=early_stop, unroll_tail=0)
    x = dynamics_trajopt(x0, cost, cfg)
    assert float(jnp.max(jnp.abs(x - x_star))) < 1e-3


def test_unrolled_form_is_differentiable():
    """``early_stop=False`` with a finite ``unroll_tail`` must backprop through
    the solve wrt an upstream scalar (the IOC pipeline's precondition)."""
    cost, _ = _quadratic()
    x0 = jnp.zeros(20)

    def outer_loss(s, tail):
        cf = lambda x: cost(x) + s * jnp.sum(x**2)
        cfg = DynamicsTrajOptConfig(n_iters=40, early_stop=False, unroll_tail=tail)
        return jnp.sum(dynamics_trajopt(x0, cf, cfg) ** 2)

    # Full unrolling differentiates the whole (converged) solve, so its gradient
    # matches finite differences; a finite tail is a truncated (Domke) estimate,
    # asserted only to be finite and correctly signed.
    g_full = jax.grad(lambda s: outer_loss(s, 40))(0.5)
    fd = (outer_loss(0.5 + 1e-3, 40) - outer_loss(0.5 - 1e-3, 40)) / 2e-3
    assert abs(float(g_full) - float(fd)) < 1e-2 * (abs(float(fd)) + 1e-6)

    g_trunc = jax.grad(lambda s: outer_loss(s, 4))(0.5)
    assert np.isfinite(float(g_trunc))
    assert float(g_trunc) * float(fd) > 0.0


def test_projected_gd_respects_box():
    """``method="projected_gd"`` stays inside the joint box."""
    n, dof = 12, 3
    lo = (-0.5,) * dof
    hi = (0.5,) * dof
    cost = lambda x: jnp.sum((x - 5.0) ** 2)  # pulls far outside the box
    cfg = DynamicsTrajOptConfig(
        method="projected_gd", n_iters=100, gd_lr=0.2,
        q_lo=lo, q_hi=hi, dof=dof,
    )
    x = dynamics_trajopt(jnp.zeros(n), cost, cfg).reshape(-1, dof)
    assert float(jnp.max(x)) <= 0.5 + 1e-6
    assert float(jnp.min(x)) >= -0.5 - 1e-6


# --------------------------------------------------------------------------- #
# Generic augmented-Lagrangian interface                                       #
# --------------------------------------------------------------------------- #

def test_equality_constraint_via_al():
    """min 0.5||x||^2 s.t. sum(x) = c  ->  x_i = c / n."""
    n, c = 6, 3.0
    base = lambda x: 0.5 * jnp.sum(x**2)
    eq = AugmentedLagrangianTerm(
        residual_fn=lambda x: jnp.array([jnp.sum(x) - c]),
        kind="eq", rho0=1.0, penalty_scale=3.0,
    )
    cfg = DynamicsTrajOptConfig(n_iters=60, constraints=(eq,), n_outer_iters=12)
    x = dynamics_trajopt(jnp.zeros(n), base, cfg)
    assert abs(float(jnp.sum(x)) - c) < 1e-3
    assert float(jnp.std(x)) < 1e-3  # all components equal


def test_arbitrary_synthetic_inequality_constraint():
    """A caller-supplied inequality unrelated to collision/grasp/dynamics:
    min 0.5||x||^2 s.t. x_i >= 1 (written as 1 - x <= 0). Proves the generic
    AL interface works for arbitrary residuals, not just the built-in terms."""
    n = 5
    base = lambda x: 0.5 * jnp.sum(x**2)
    ineq = AugmentedLagrangianTerm(
        residual_fn=lambda x: 1.0 - x, kind="ineq", rho0=1.0, penalty_scale=3.0,
    )
    cfg = DynamicsTrajOptConfig(n_iters=60, constraints=(ineq,), n_outer_iters=15)
    x = dynamics_trajopt(jnp.zeros(n), base, cfg)
    assert float(jnp.min(x)) > 1.0 - 1e-2
    assert float(jnp.max(x)) < 1.0 + 1e-2  # tight: pulled to the constraint


def test_constraints_empty_matches_single_solve():
    """``constraints=()`` must reproduce the plain single L-BFGS solve exactly."""
    cost, _ = _quadratic()
    x0 = jnp.zeros(20)
    cfg_plain = DynamicsTrajOptConfig(n_iters=100)
    cfg_empty = DynamicsTrajOptConfig(n_iters=100, constraints=(), n_outer_iters=5)
    x_plain = dynamics_trajopt(x0, cost, cfg_plain)
    x_empty = dynamics_trajopt(x0, cost, cfg_empty)
    assert float(jnp.max(jnp.abs(x_plain - x_empty))) == 0.0


def test_sco_linearization_matches_exact_for_ineq():
    """SCO-linearized inequality reaches the same optimum as the exact AL term
    on a problem whose constraint is already affine (linearization is exact)."""
    n = 5
    base = lambda x: 0.5 * jnp.sum(x**2)
    ineq = AugmentedLagrangianTerm(
        residual_fn=lambda x: 1.0 - x, kind="ineq", rho0=1.0, penalty_scale=3.0,
    )
    x_exact = dynamics_trajopt(
        jnp.zeros(n), base,
        DynamicsTrajOptConfig(n_iters=60, constraints=(ineq,), n_outer_iters=15),
    )
    x_sco = dynamics_trajopt(
        jnp.zeros(n), base,
        DynamicsTrajOptConfig(n_iters=60, constraints=(ineq,), n_outer_iters=15,
                              use_sco=True),
    )
    assert float(jnp.max(jnp.abs(x_exact - x_sco))) < 1e-2


def test_al_outer_loop_reports_duals_and_rhos():
    """The AL driver hands back per-term duals and penalties for diagnostics."""
    n = 4
    eq = AugmentedLagrangianTerm(
        residual_fn=lambda x: jnp.array([jnp.sum(x) - 2.0]), kind="eq",
        rho0=1.0, rho_max=100.0, penalty_scale=2.0,
    )
    inner = lambda z, cf: _lbfgs_driver(z, cf, n_iters=40, m_lbfgs=6, loop="while")
    z, duals, rhos = _al_outer_loop(
        jnp.zeros(n), inner, (eq,), lambda x, xk: 0.5 * jnp.sum(x**2),
        n_outer_iters=8,
    )
    assert len(duals) == 1 and len(rhos) == 1
    assert float(rhos[0]) <= 100.0
    assert abs(float(jnp.sum(z)) - 2.0) < 1e-3


def test_adaptive_trust_region_on_nonlinear_constraint():
    """Adaptive (Schulman) trust-region sizing converges on a constraint whose
    linearization is inaccurate far from the iterate.

    min 0.5||x - 5||^2  s.t.  exp(x) - e <= 0  (i.e. x <= 1). The objective pulls
    hard toward x = 5, so the collision-style linearized inequality must hold the
    solution at the boundary x = 1; the exponential makes the linear model
    over-promise on big steps, which is exactly what the ratio test must catch by
    shrinking the region."""
    from pyroffi.optimization_engines._trajopt_core import (
        AugmentedLagrangianTerm, TrustRegionConfig, _al_outer_loop, _lbfgs_driver,
    )

    n = 4
    base = lambda z, zk: 0.5 * jnp.sum((z - 5.0) ** 2)
    term = AugmentedLagrangianTerm(
        residual_fn=lambda x: jnp.exp(x) - jnp.e, kind="ineq",
        rho0=1.0, rho_max=1e4, penalty_scale=2.0,
    )
    inner = lambda z, cf: _lbfgs_driver(z, cf, n_iters=50, m_lbfgs=6, loop="while")

    z, _, _ = _al_outer_loop(
        jnp.zeros(n), inner, (term,), base,
        n_outer_iters=25, sco_linearize=True,
        trust=TrustRegionConfig(coef0=1.0),
    )
    assert np.isfinite(np.array(z)).all()
    assert float(jnp.max(z)) < 1.0 + 5e-2       # feasible: x <= 1
    assert float(jnp.min(z)) > 1.0 - 2e-1       # tight at the boundary, not slack
