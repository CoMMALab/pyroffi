"""The inner trajectory optimizer and how gradients pass through it.

DiffTORI's policy is the *solution* of a trajectory optimization problem whose
cost is a neural network, so training needs ``dx*/dtheta``.  This module is the
same construction as ``ioc.inner.make_inner_solver`` in this repo, specialised
to DiffTORI's inner problem:

    C(x; theta, z) = -sum_l gamma^l f_theta(z_l, a_l) + penalties,   x = actions
    x*(theta, z)   = argmin_x C(x; theta, z)

with the latent dynamics ``z_{l+1} = d_theta(z_l, a_l)`` substituted into the
objective rather than imposed as a constraint (Eq. 5 of the paper).

**Forward solve.**  ``forward_solver`` is required, exactly as in ``ioc.inner``:
a ``Callable[[x0, cost_fn], x]``.  In practice it is
``pyroffi.optimization_engines.dynamics_trajopt`` -- this repo's dynamics-aware
L-BFGS engine -- wrapped by ``difftori.pyroffi_trajopt``.  The paper uses
Theseus' Levenberg--Marquardt, which only accepts *nonlinear least squares* and
forced it to encode a maximisation as a residual; a general nonlinear minimizer
does not need that reduction, so we optimize the discounted return directly.

**Gradient: implicit adjoint.**  At a stationary point
``F(x*, theta) = grad_x C = 0``; differentiating that identity gives

    dx*/dtheta = -H^-1 B,    H = grad^2_xx C,   B = grad^2_{x,theta} C

so the reverse-mode cotangent is ``-B^T H^-T g_out`` -- one linear solve with
the final iterate, memory independent of the iteration count.  Both ``theta``
(network parameters) and ``z`` (the latent, hence the encoder and the
observation) receive gradients this way; the initialisation does not, because at
a stationary point ``x*`` is locally independent of where the solve started.

**The convergence proviso is not decoration.**  The adjoint is exact only if
the forward solve actually reached stationarity.  ``stationarity()`` returns
``||grad_x C||`` at the returned solution; ``ioc`` screens on it and discards
contexts that plateau (there, agreement with finite differences falls from
cos 0.9999 to 0.59 on non-stationary ones).  Monitor it during training.

**Truncated unrolling** (``solve_unrolled``, Domke 2012) is provided for the
same reason ``ioc.inner`` provides it: as an independent check on the adjoint.
It needs ``dynamics_trajopt``'s fixed-iteration form
(``early_stop=False, unroll_tail=k``), and its memory scales with the tail.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import Array

__all__ = ["DiffTORISolver", "make_difftori_solver"]

# cost_fn(x, params, aux) -> scalar, for ONE problem.  Batching is by vmap.
CostFn = Callable[[Array, Any, Any], Array]


@dataclasses.dataclass(frozen=True)
class DiffTORISolver:
    """The inner problem packaged with everything the outer loop needs.

    Attributes
    ----------
    cost            ``C(x; params, aux)``, the objective being minimized.
    grad_x          ``grad_x C``; zero at a converged solve.
    solve           raw batched forward solve, no differentiation rule.
    solve_implicit  forward solve with the implicit-adjoint VJP attached.
    solve_unrolled  forward solve differentiated by truncated unrolling.
    stationarity    ``||grad_x C||`` per problem -- the screening statistic.
    """

    cost: Callable
    grad_x: Callable
    solve: Callable
    solve_implicit: Callable
    solve_unrolled: Callable
    stationarity: Callable


def _damped(H: Array, lam: float) -> Array:
    n = H.shape[0]
    return H + lam * (jnp.trace(H) / n + 1.0) * jnp.eye(n, dtype=H.dtype)


def make_difftori_solver(
    cost_fn: CostFn,
    *,
    forward_solver: Callable[[Array, Callable], Array] | None = None,
    unrolled_forward_solver: Callable[[Array, Callable], Array] | None = None,
    adjoint_ridge: float = 1e-9,
) -> DiffTORISolver:
    """Build the inner solver for a scalar ``cost_fn(x, params, aux)``.

    Args:
        forward_solver: required, ``Callable[[x0, cost_fn], x]``.  Use
            ``difftori.pyroffi_trajopt.make_dynamics_forward_solver()``.  Only
            ever called under ``stop_gradient``, so the cheap early-stopping
            (``while_loop``) form is fine and preferred.
        unrolled_forward_solver: backs ``solve_unrolled``; must be the
            fixed-iteration reverse-mode-differentiable form
            (``early_stop=False``).  Defaults to ``forward_solver``, which is
            only correct if that solver was itself built with
            ``early_stop=False``.
        adjoint_ridge: conditioning only.  Do NOT reuse the solver's line-search
            or damping constants here: those exist to make the *forward* step
            safe and are orders of magnitude larger than conditioning needs, so
            they bias ``H^-1`` and corrupt the adjoint.
    """
    if forward_solver is None:
        raise ValueError(
            "make_difftori_solver requires a forward_solver, e.g. "
            "difftori.pyroffi_trajopt.make_dynamics_forward_solver() which "
            "wraps pyroffi.optimization_engines.dynamics_trajopt."
        )
    if unrolled_forward_solver is None:
        unrolled_forward_solver = forward_solver

    cost = cost_fn
    grad_x = jax.grad(cost, argnums=0)
    hess_x = jax.hessian(cost, argnums=0)

    def _promote(x0, params, aux):
        """Cast x0 to the dtype the cost (and so its gradient) comes back in.

        The engine's iteration carries x through a `while_loop`, and JAX
        requires a carry to keep its dtype.  Two things promote it: a cost built
        from higher-precision parameters, and -- less obviously -- the engine's
        own line-search constants, which are float64 under JAX_ENABLE_X64=1.  So
        a float32 x0 (Flax defaults its parameters to float32, and datasets are
        stored float32) enters the loop and a float64 step comes out, raising a
        type error from inside the engine, several frames from its cause.

        The working dtype is therefore the graph's default float, exactly the
        convention `ioc.robot.bases.x_dtype` follows, promoted against the cost
        in case the caller's parameters are wider still.  Fixing it here settles
        it once for every caller rather than imposing a dtype discipline on the
        data pipeline.
        """
        one_aux = jax.tree.map(lambda a: a[0], aux)
        cost_dtype = jax.eval_shape(cost, x0[0], params, one_aux).dtype
        engine_dtype = jnp.zeros((), float).dtype
        return x0.astype(jnp.promote_types(
            jnp.promote_types(x0.dtype, cost_dtype), engine_dtype))

    def _solve_one(x0, params, aux):
        return forward_solver(x0, lambda x: cost(x, params, aux))

    def solve(x0, params, aux):
        """Batched forward solve: ``x0 (B, n)``, ``aux`` batched on axis 0."""
        x0 = _promote(x0, params, aux)
        return jax.vmap(_solve_one, in_axes=(0, None, 0))(x0, params, aux)

    def solve_unrolled(x0, params, aux):
        """Truncated unrolling: the head runs under ``stop_gradient`` and only
        the trailing ``unroll_tail`` steps are differentiated.  The head/tail
        split is owned by the solver's own config."""
        x0 = _promote(x0, params, aux)
        return jax.vmap(
            lambda s0, a: unrolled_forward_solver(s0, lambda x: cost(x, params, a)),
            in_axes=(0, 0),
        )(x0, aux)

    def stationarity(x0, params, aux):
        xs = solve(x0, params, aux)
        return jax.vmap(lambda x, a: jnp.linalg.norm(grad_x(x, params, a)))(xs, aux)

    @jax.custom_vjp
    def solve_implicit(x0, params, aux):
        return solve(x0, params, aux)

    def _fwd(x0, params, aux):
        xs = jax.lax.stop_gradient(solve(x0, params, aux))
        return xs, (xs, params, aux)

    def _bwd(res, g_out):
        xs, params, aux = res

        def single(x, a, g):
            H = _damped(hess_x(x, params, a), adjoint_ridge)
            # dx*/dtheta = -H^-1 B, so the cotangent is -B^T H^-T g_out, and
            # B^T u is one VJP of grad_x C with respect to (params, aux).
            u = jnp.linalg.solve(H.T, g)
            _, vjp = jax.vjp(lambda p, aa: grad_x(x, p, aa), params, a)
            d_params, d_aux = vjp(u)
            return jax.tree.map(jnp.negative, (d_params, d_aux))

        d_params, d_aux = jax.vmap(single, in_axes=(0, 0, 0))(xs, aux, g_out)
        # params is shared across the batch; aux is per-problem.
        d_params = jax.tree.map(lambda g: jnp.sum(g, axis=0), d_params)
        return jnp.zeros_like(xs), d_params, d_aux

    solve_implicit.defvjp(_fwd, _bwd)

    return DiffTORISolver(
        cost=cost,
        grad_x=grad_x,
        solve=solve,
        solve_implicit=solve_implicit,
        solve_unrolled=solve_unrolled,
        stationarity=stationarity,
    )
