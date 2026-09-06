"""The inner trajectory optimizer and the two ways to differentiate through it.

Theory
------
The inner problem is a weighted nonlinear least-squares problem in the interior
waypoints x, parameterized by the cost weights theta and a context c:

    C_theta(x, c) = sum_k (theta_k / s_k) * || r_k(x, c) ||^2,
    x*(theta, c)  = argmin_x C_theta(x, c).

It is solved by damped Gauss-Newton (Levenberg-Marquardt): each iteration builds
g = 2 J^T r and H_gn = 2 J^T J from the whitened, theta-weighted residual stack,
adds lambda * (tr H / n + 1) * I, and takes x <- x - H^-1 g, accepting the step
only if the cost decreased and adapting lambda accordingly.  H_gn is PSD by
construction, so the direction is always a descent direction and the iteration
converges to a genuine stationary point -- which is the precondition for *both*
finite differences and the implicit function theorem to mean anything.

What the outer loop needs is dx*/dtheta.  There are three ways to get it, and
this module implements the two that go through the solver:

**Implicit adjoint** (`solve_implicit`).  At a stationary point the solution is
characterized by

    F(x*, theta) = grad_x C_theta(x*, c) = 0.

Differentiating that identity in theta gives the implicit function theorem,

    H dx*/dtheta + B = 0,   H = grad^2_xx C,  B = grad^2_x,theta C
    =>  dx*/dtheta = -H^-1 B,

so the reverse-mode cotangent of a downstream loss is  -B^T H^-T g_out.  The
cost is one linear solve with the *final* iterate: nothing about the path taken
to reach it is needed, so memory is independent of the iteration count and the
gradient is exact regardless of how many iterations ran -- provided the solve
actually converged.  That proviso is not decoration; `stationarity` exists so
every experiment can screen contexts on it, and contexts that plateau are
discarded (on them the returned x still depends on the solver path, FD picks up
sensitivity the adjoint cannot see, and agreement falls from cos 0.9999 to 0.59).

**Truncated unrolling** (`solve_unrolled`, Domke 2012).  Treat the solver as a
finite computation graph and backpropagate through the last `unroll_tail`
iterations, with the head under `stop_gradient`.  It needs no Hessian solve and
converges to the implicit gradient as the tail grows, but its memory scales with
the tail length: each retained step holds a dense inner Jacobian built from a
full FK graph over the horizon, and backpropagating through all iterations
exceeds device memory even with rematerialization.  That limit is itself a
result -- it is exactly the cost the implicit adjoint avoids.

The third way, finite differences on L, needs no differentiable solver at all
and lives in `ioc.outer`; it costs K+1 solves per outer step instead of 1.

Curvature
---------
The implicit adjoint always uses the exact Hessian grad^2_xx C
(`adjoint_hessian="jax"`, `hess_x = jax.hessian(cost, ...)`). This used to
need a first-derivatives-only Gauss-Newton fallback ("gn") or a float64-JAX
curvature computed alongside a float32 GRiD forward solve ("hybrid"), because
the GRiD CUDA inverse-dynamics FFI supported only one level of
differentiation -- `jax.hessian` raised straight through it. That limit is
fixed at the source now: `pyroffi.dynamics.GRiDDynamics`'s analytic-gradient
kernels (`inverse_dynamics_gradient`, `crba`) carry their own `custom_jvp`
built from GRiD's `idsva_so` second-order kernel, so `jax.hessian` works
straight through `inverse_dynamics` on the GRiD backend exactly as it always
did on the pure-JAX one. "jax" names the single remaining mode to flag that
it is backend-agnostic -- a plain `jax.hessian` of whatever `cost`/
`adjoint_cost_fn` computes, GRiD-backed or not.

`forward_solver` selects which optimizer produces x*; the implicit adjoint only
needs x* to be a stationary point, not to know how it was found.

`adjoint_cost_fn` overrides the objective used to build curvature while leaving
the forward solve alone, for the rare case a caller wants a different (but
still twice-differentiable) objective driving curvature than the one driving
the forward solve.
"""

import dataclasses
from typing import Callable

import jax
import jax.numpy as jnp

from pyroffi.optimization_engines._trajopt_core import (
    _al_outer_loop,
    _al_penalty,
)


@dataclasses.dataclass(frozen=True)
class InnerSolver:
    """The inner problem, packaged with everything the outer loop needs.

    Attributes
    ----------
    cost           C_theta(x, theta, ctx), whitened by the calibrated scales.
    grad_x         grad_x C; the object Inverse KKT asserts is zero at a demo.
    gn_system      (g, H_gn) at a point; the PSD curvature CIOC integrates over.
    features       phi(x, ctx) in R^K, unweighted, whitened.
    solve          the raw forward solve (no custom differentiation rule).
    solve_implicit forward solve with the implicit-adjoint VJP attached.
    solve_unrolled forward solve differentiated by truncated unrolling.
    stationarity   ||grad_x C|| at the returned solution; the screening statistic.
    """

    cost: Callable
    grad_x: Callable
    gn_system: Callable
    features: Callable
    solve: Callable
    solve_implicit: Callable
    solve_unrolled: Callable
    stationarity: Callable


def make_inner_solver(
    residual_fn,
    scales,
    *,
    adjoint_ridge=1e-9,
    adjoint_hessian="jax",
    adjoint_cost_fn=None,
    n_restarts=1,
    restart_jitter=0.35,
    restart_seed_fn=None,
    forward_solver=None,
    unrolled_forward_solver=None,
    constraints_fn=None,
    use_sco=False,
    n_outer_iters=10,
    al_dual_scale=1.0,
    trust=None,
):
    """Build the inner solver for `residual_fn(x, ctx) -> tuple of residuals`.

    `scales` are the fixed nominal feature magnitudes from the calibration step;
    dividing by them makes weight recovery measure recovery rather than feature
    scaling.  See the module docstring for the theory and for what
    `adjoint_hessian` / `adjoint_cost_fn` select.

    `forward_solver` is required: `Callable[[x0, cost_fn], x]`, the optimizer
    that finds x*, used by `solve`/`solve_implicit`/`stationarity`.  This is a
    third, orthogonal axis alongside `adjoint_hessian`/`adjoint_cost_fn`: which
    optimizer finds x*, independent of which curvature the implicit adjoint
    uses to differentiate through it.  Must be the early-stopping form (only
    ever called under `stop_gradient` here, so reverse-mode differentiability
    doesn't matter).

    `unrolled_forward_solver` backs `solve_unrolled` and must instead be a
    reverse-mode-differentiable, fixed-iteration solver -- in practice
    `pyroffi.optimization_engines.dynamics_trajopt` with
    `DynamicsTrajOptConfig(early_stop=False, unroll_tail=...)`.  Defaults to
    `forward_solver` if omitted (fine as long as that solver is itself built
    with `early_stop=False`).

    `constraints_fn` (optional) `Callable[[ctx], tuple[AugmentedLagrangianTerm]]`
    makes the forward solve *constrained*: each particle settles in a feasible
    local optimum via an augmented-Lagrangian outer loop around `forward_solver`
    (`use_sco` linearizes inequality residuals à la Schulman; `trust` is an
    optional `TrustRegionConfig` for adaptive sizing; `n_outer_iters` /
    `al_dual_scale` control the AL loop). The constraints must be
    theta-INDEPENDENT. The implicit adjoint then linearizes the *augmented*
    stationarity `grad_x[C + penalties] = 0` with the converged multipliers
    frozen -- a single-gradient IFT, not the KKT system. `constraints_fn=None`
    (default) is byte-identical to the unconstrained solver above.
    """
    if forward_solver is None:
        raise ValueError(
            "make_inner_solver requires a forward_solver "
            "(e.g. pyroffi.optimization_engines.dynamics_trajopt wrapped as "
            "Callable[[x0, cost_fn], x]); the internal Gauss-Newton fallback "
            "has been removed."
        )
    if unrolled_forward_solver is None:
        unrolled_forward_solver = forward_solver

    if adjoint_hessian != "jax":
        raise ValueError(
            f"adjoint_hessian must be 'jax' (the only remaining mode -- see "
            f"the module docstring), got {adjoint_hessian!r}."
        )

    def features(x, ctx):
        rs = residual_fn(x, ctx)
        return jnp.stack([jnp.sum(r**2) for r in rs]) / scales

    def cost(x, theta, ctx):
        return jnp.dot(theta, features(x, ctx))

    grad_x = jax.grad(cost, argnums=0)
    hess_x = jax.hessian(cost if adjoint_cost_fn is None else adjoint_cost_fn,
                         argnums=0)

    def gn_system(x, theta, ctx):
        """Gradient and PSD Gauss-Newton Hessian of sum_k theta_k ||r_k||^2 /
        s_k. Used by CIOC (`ioc.analytic.cioc_fit`), which integrates this
        curvature directly -- unrelated to the implicit adjoint's curvature
        (see the module docstring), which always uses the exact Hessian."""

        def res_cat(xx):
            rs = residual_fn(xx, ctx)
            w = jnp.sqrt(theta / scales)
            return jnp.concatenate([wk * r for wk, r in zip(w, rs)])

        r = res_cat(x)
        J = jax.jacobian(res_cat)(x)
        return 2.0 * J.T @ r, 2.0 * J.T @ J

    def _damped(H, lam):
        n = H.shape[0]
        return H + lam * (jnp.trace(H) / n + 1.0) * jnp.eye(n, dtype=H.dtype)

    # --- Constrained (augmented-Lagrangian) forward + augmented stationarity --
    #
    # When `constraints_fn(ctx) -> tuple[AugmentedLagrangianTerm]` is supplied,
    # each particle settles in a *feasible* local optimum: the base cost
    # C_theta is minimized subject to the (theta-INDEPENDENT) constraint terms
    # via a dual-ascent AL outer loop (`_al_outer_loop`), optionally with SCO
    # linearization / adaptive trust region.  The implicit adjoint then
    # differentiates the stationarity of the AUGMENTED objective
    #
    #     grad_x [ C_theta(x) + sum_c penalty_c(x; lambda*, rho*) ] = 0
    #
    # with the converged multipliers (lambda*, rho*) frozen (stop_gradient) --
    # a single-gradient IFT, NOT the full KKT system (no complementarity is
    # differentiated).  Because the constraints carry no theta, the mixed term
    # B = grad^2_{x,theta} is unchanged; only the curvature H picks up the
    # penalty Hessian.  See the module docstring / `stationarity`.

    def _aug_cost(x, theta, ctx, terms, duals, rhos):
        c = cost(x, theta, ctx)
        for term, d, rho in zip(terms, duals, rhos):
            c = c + _al_penalty(term, term.residual_fn(x), d, rho)
        return c

    _grad_x_aug = jax.grad(_aug_cost, argnums=0)
    _hess_x_aug = jax.hessian(_aug_cost, argnums=0)

    def _solve_one(x0, theta, ctx):
        return forward_solver(x0, lambda x: cost(x, theta, ctx))

    def _solve_one_constrained(x0, theta, ctx):
        """Constrained local solve -> (x*, duals*, rhos*).  `terms` is rebuilt
        from `ctx` by the caller (constraints depend on scene, not on x)."""
        terms = constraints_fn(ctx)
        z, duals, rhos = _al_outer_loop(
            x0,
            lambda z0_, cf: forward_solver(z0_, cf),
            terms,
            lambda z, zk: cost(z, theta, ctx),
            n_outer_iters=n_outer_iters,
            dual_scale=al_dual_scale,
            sco_linearize=use_sco,
            trust=trust,
            return_solve_multipliers=True,
        )
        return z, duals, rhos

    def _solve_full(x0, theta, ctx):
        """Multistart forward solve returning (x*, duals*, rhos*).  With
        `constraints_fn=None`, duals/rhos are empty tuples and the selection is
        the plain best-cost basin (byte-identical to the unconstrained `solve`).
        The constrained path selects the basin by AUGMENTED cost, so an
        infeasible-but-cheap basin does not win over a feasible one."""
        terms_probe = None if constraints_fn is None else constraints_fn(ctx)

        def one(s0):
            if constraints_fn is None:
                return _solve_one(s0, theta, ctx), (), ()
            return _solve_one_constrained(s0, theta, ctx)

        def sel_cost(x, duals, rhos):
            if constraints_fn is None:
                return cost(x, theta, ctx)
            return _aug_cost(x, theta, ctx, terms_probe, duals, rhos)

        if n_restarts <= 1:
            starts = x0[None]
        elif restart_seed_fn is not None:
            starts = restart_seed_fn(x0, ctx, n_restarts)
        else:
            keys = jax.random.split(jax.random.key(0), n_restarts - 1)
            starts = jnp.stack(
                [x0] + [x0 + restart_jitter * jax.random.normal(k, x0.shape)
                       for k in keys]
            )

        xs, duals, rhos = jax.vmap(one)(starts)
        costs = jax.vmap(sel_cost)(xs, duals, rhos)
        i = jnp.argmin(costs)
        pick = lambda t: jax.tree.map(lambda a: a[i], t)
        return xs[i], pick(duals), pick(rhos)

    def solve(x0, theta, ctx):
        """Forward solve; the best of `n_restarts` local solves when > 1.

        With a single start, a multimodal inner problem makes x*(theta) jump as
        theta moves the solver between basins -- the outer loss is then
        discontinuous and *every* gradient method fails on it, while
        derivative-free search is unaffected (measured on the Gaussian-field
        benchmark: FD gradients 10-23x larger than the adjoint on exactly the
        contexts that flip basins).  Approximating the global inner solution
        makes x*(theta) a far better-behaved function of theta.  Restarts are
        charged to every method's solve budget, so the comparison stays fair.

        By default the extra starts are i.i.d. per-waypoint Gaussian jitter
        around x0.  That is a *local* perturbation: for anything past a few
        waypoints of amplitude it still lands in the basin nearest x0 after a
        few Newton steps, so it resamples the same basin instead of covering
        the distinct basins ("topological convexities" -- different routes
        relative to the obstacle/bump field) that a multimodal context
        actually has.  `restart_seed_fn(x0, ctx, n_restarts) -> (n_restarts, d)`
        lets a benchmark supply *structured* candidates instead -- e.g.
        coherent low-frequency lateral detours that are guaranteed to cross to
        the other side of a bump/obstacle -- so the population that gets
        vmapped through `_solve_one` actually characterizes the field rather
        than resampling one basin n_restarts times.

        With `constraints_fn` supplied the whole selection is delegated to
        `_solve_full` (feasible basins, augmented-cost argmin); the code below is
        the exact unconstrained path every existing caller still hits.
        """
        if constraints_fn is not None:
            x, _, _ = _solve_full(x0, theta, ctx)
            return x
        if n_restarts <= 1:
            return _solve_one(x0, theta, ctx)
        if restart_seed_fn is not None:
            starts = restart_seed_fn(x0, ctx, n_restarts)
        else:
            keys = jax.random.split(jax.random.key(0), n_restarts - 1)
            starts = jnp.stack(
                [x0] + [x0 + restart_jitter * jax.random.normal(k, x0.shape)
                       for k in keys]
            )
        xs = jax.vmap(lambda s0: _solve_one(s0, theta, ctx))(starts)
        costs = jax.vmap(lambda xx: cost(xx, theta, ctx))(xs)
        return xs[jnp.argmin(costs)]

    def solve_unrolled(x0, theta, ctx):
        """Truncated unrolling (Domke 2012): reverse-mode AD through the last
        `unroll_tail` steps of `unrolled_forward_solver`'s solve, head under
        `stop_gradient`.  The head/tail split itself is owned by that solver's
        own config (`DynamicsTrajOptConfig(early_stop=False, unroll_tail=...)`);
        this just points it at the current `cost_fn`."""
        return unrolled_forward_solver(x0, lambda x: cost(x, theta, ctx))

    def stationarity(x0, theta, ctx):
        """Stationarity residual at the solution -- implicit diff assumes ~0.

        Unconstrained: ``||grad_x C||``.  Constrained: ``||grad_x [C + AL
        penalties]||`` at the converged multipliers -- the AUGMENTED
        stationarity the constrained adjoint actually linearizes, so the
        CONFOUND-4 screen checks the right residual (bare ``grad_x C`` is
        nonzero at a constrained optimum and would false-alarm)."""
        if constraints_fn is None:
            return jnp.linalg.norm(grad_x(solve(x0, theta, ctx), theta, ctx))
        x, duals, rhos = _solve_full(x0, theta, ctx)
        terms = constraints_fn(ctx)
        return jnp.linalg.norm(_grad_x_aug(x, theta, ctx, terms, duals, rhos))

    @jax.custom_vjp
    def solve_implicit(x0, theta, ctx):
        return solve(x0, theta, ctx)

    def _fwd(x0, theta, ctx):
        if constraints_fn is None:
            xs = jax.lax.stop_gradient(solve(x0, theta, ctx))
            return xs, (xs, theta, ctx, None, None)
        xs, duals, rhos = _solve_full(x0, theta, ctx)
        xs = jax.lax.stop_gradient(xs)
        duals = jax.lax.stop_gradient(duals)
        rhos = jax.lax.stop_gradient(rhos)
        return xs, (xs, theta, ctx, duals, rhos)

    def _bwd(res, g_out):
        xs, theta, ctx, duals, rhos = res
        if duals is None:
            # Unconstrained: differentiate stationarity of the bare cost.
            H = hess_x(xs, theta, ctx)
            gfn = lambda th: grad_x(xs, th, ctx)
        else:
            # Constrained: differentiate stationarity of the AUGMENTED
            # objective with the multipliers frozen (single-gradient IFT, not
            # KKT).  The penalty carries no theta, so its theta-VJP is zero and
            # B = grad^2_{x,theta} is unchanged; only H gains the penalty
            # Hessian.
            terms = constraints_fn(ctx)
            H = _hess_x_aug(xs, theta, ctx, terms, duals, rhos)
            gfn = lambda th: _grad_x_aug(xs, th, ctx, terms, duals, rhos)
        # Ridge only for numerical invertibility.  Reusing the LM damping here
        # would be a bug: that damping exists to make the *forward* step safe,
        # and it is many orders of magnitude larger than what conditioning
        # needs, so it biases H^-1 and corrupts the adjoint.
        H = _damped(H, adjoint_ridge)
        # dx*/dtheta = -H^-1 B, so the cotangent is -B^T H^-T g_out, and B^T u
        # is one VJP of grad_x (C or augmented) with respect to theta.
        u = jnp.linalg.solve(H.T, g_out)
        _, vjp_theta = jax.vjp(gfn, theta)
        (g_theta,) = vjp_theta(u)
        return jnp.zeros_like(xs), -g_theta, jax.tree.map(jnp.zeros_like, ctx)

    solve_implicit.defvjp(_fwd, _bwd)

    return InnerSolver(
        cost=cost,
        grad_x=grad_x,
        gn_system=gn_system,
        features=features,
        solve=solve,
        solve_implicit=solve_implicit,
        solve_unrolled=solve_unrolled,
        stationarity=stationarity,
    )
