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

Curvature choices
-----------------
`adjoint_hessian` selects the H used by the adjoint:

  "true"  exact grad^2_xx C.  Most accurate, but needs *second* derivatives of
          every feature.
  "gn"    the Gauss-Newton J^T J.  First derivatives only, so it is the only
          option when a feature comes from an FFI kernel that supports a single
          level of differentiation (the GRiD CUDA inverse-dynamics call, whose
          FFI raises on `jax.hessian`).  At a converged small-residual solution
          the two nearly agree; the experiments measure the gap rather than
          assuming it (measured: ~14% magnitude bias on E3).

`forward_solver` selects which optimizer produces x* in the first place --
orthogonal to `adjoint_hessian`: the implicit adjoint only needs x* to be a
genuine stationary point, not to know how it was found, so any solver can be
swapped in as `_solve_one` (required; see `make_inner_solver`).  In practice
this is always `pyroffi.optimization_engines.dynamics_trajopt`, the generic
L-BFGS engine this codebase ships, wrapped as `Callable[[x0, cost_fn], x]`.

`adjoint_cost_fn` overrides the objective used to *build* that curvature while
leaving the forward solve alone.  This is the hybrid path: run the forward solve
on a fast, reduced-precision, once-differentiable backend (GRiD CUDA) and build
the adjoint -- evaluated once per outer step, against ~n_iter solver iterations
-- from an exact float64 model.  Measured to reproduce the pure-float64 gradient
at cos = 1.000000.
"""

import dataclasses
from typing import Callable

import jax
import jax.numpy as jnp


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
    adjoint_hessian="true",
    adjoint_cost_fn=None,
    n_restarts=1,
    restart_jitter=0.35,
    restart_seed_fn=None,
    forward_solver=None,
    unrolled_forward_solver=None,
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

    def features(x, ctx):
        rs = residual_fn(x, ctx)
        return jnp.stack([jnp.sum(r**2) for r in rs]) / scales

    def cost(x, theta, ctx):
        return jnp.dot(theta, features(x, ctx))

    grad_x = jax.grad(cost, argnums=0)
    hess_x = jax.hessian(cost if adjoint_cost_fn is None else adjoint_cost_fn,
                         argnums=0)

    def gn_system(x, theta, ctx):
        """Gradient and PSD Gauss-Newton Hessian of sum_k theta_k ||r_k||^2 / s_k."""

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

    def _solve_one(x0, theta, ctx):
        return forward_solver(x0, lambda x: cost(x, theta, ctx))

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
        """
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
        """||grad_x C|| at the solution -- implicit diff assumes this is ~0."""
        return jnp.linalg.norm(grad_x(solve(x0, theta, ctx), theta, ctx))

    @jax.custom_vjp
    def solve_implicit(x0, theta, ctx):
        return solve(x0, theta, ctx)

    def _fwd(x0, theta, ctx):
        xs = jax.lax.stop_gradient(solve(x0, theta, ctx))
        return xs, (xs, theta, ctx)

    def _bwd(res, g_out):
        xs, theta, ctx = res
        if adjoint_hessian == "gn":
            _, H = gn_system(xs, theta, ctx)
        else:
            H = hess_x(xs, theta, ctx)
        # Ridge only for numerical invertibility.  Reusing the LM damping here
        # would be a bug: that damping exists to make the *forward* step safe,
        # and it is many orders of magnitude larger than what conditioning
        # needs, so it biases H^-1 and corrupts the adjoint.
        H = _damped(H, adjoint_ridge)
        # dx*/dtheta = -H^-1 B, so the cotangent is -B^T H^-T g_out, and B^T u
        # is one VJP of grad_x C with respect to theta.
        u = jnp.linalg.solve(H.T, g_out)
        _, vjp_theta = jax.vjp(lambda th: grad_x(xs, th, ctx), theta)
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
