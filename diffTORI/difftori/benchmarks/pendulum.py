"""Amos et al.'s pendulum swing-up -- the paper's Appendix A.3 / Table 8 benchmark.

DiffTORI compares against Amos et al. [3] ("Differentiable MPC for End-to-end
Planning and Control") on that paper's pendulum imitation task, in two settings:
without damping, and with.  Table 8 of DiffTORI reports the *cost of the learned
policy*::

                            Expert     Amos et al.      LSTM policy    DiffTORI
    Pendulum w/o damping    13.126   13.576 +- 0.012  15.962 +- 0.164  14.603 +- 0.190
    Pendulum with damping   10.132   14.874 +- 0.600  12.098 +- 0.031  10.644 +- 0.029

Every constant here is taken from Amos' released code, not inferred:

``mpc.pytorch/mpc/env_dx/pendulum.py`` (``PendulumDx``)
    ``dt = 0.05``, ``max_torque = 2.0``, state ``(cos th, sin th, dth)``,
    ``goal_state = (1, 0, 0)``, ``goal_weights = (1, 1, 0.1)``,
    ``ctrl_penalty = 0.001``, and the two dynamics branches below.

``differentiable-mpc/imitation_nonconvex/il_env.py`` (``IL_Env``)
    ``mpc_T = 20``, ``lqr_iter = 500``; ``pendulum`` uses params
    ``(g, m, l) = (10, 1, 1)`` and ``pendulum-complex`` uses
    ``(g, m, l, d, b) = (10, 1, 1, 1.0, 0.1)`` -- the latter is DiffTORI's
    "with damping" row.  Initial states are
    ``th ~ U(-pi/2, pi/2)``, ``dth ~ U(-1, 1)``, and the split is
    ``n_train, n_val, n_test = 100, 10, 10`` at ``seed = 0``.

``differentiable-mpc/imitation_nonconvex/make_dataset.py``
    one MPC solve per initial state; the demonstration is the resulting
    **open-loop nominal action sequence** ``u_{1:T}``, not a receding-horizon
    rollout.

Two things are ours rather than Amos':

* **The solver.**  Amos uses box-constrained iLQR.  We use
  ``pyroffi.optimization_engines.dynamics_trajopt`` (L-BFGS on a flat decision
  vector) with a one-sided quadratic barrier standing in for the box, which is
  the same substitution ``policy_il.plan_cost`` makes for the action clamp.
  The expert's achieved cost is the check on whether that substitution is
  faithful -- see ``expert_cost_report``.
* **float64.**  The implicit adjoint inverts the inner Hessian.  Run under
  ``JAX_ENABLE_X64=1``.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

__all__ = ["PendulumParams", "SIMPLE", "COMPLEX", "step", "rollout",
           "trajectory_cost", "sample_xinit", "solve_expert", "make_dataset",
           "policy_cost", "T_HORIZON", "MAX_TORQUE"]

DT = 0.05
MAX_TORQUE = 2.0
T_HORIZON = 20              # IL_Env.mpc_T
GOAL_STATE = jnp.array([1.0, 0.0, 0.0])
GOAL_WEIGHTS = jnp.array([1.0, 1.0, 0.1])
CTRL_PENALTY = 0.001


@dataclasses.dataclass(frozen=True)
class PendulumParams:
    """``PendulumDx`` parameters.  ``simple=True`` ignores ``d`` and ``b``."""

    g: float = 10.0
    m: float = 1.0
    l: float = 1.0
    d: float = 0.0          # damping
    b: float = 0.0          # gravity bias ("wind")
    simple: bool = True


SIMPLE = PendulumParams()                                   # 'pendulum'
COMPLEX = PendulumParams(d=1.0, b=0.1, simple=False)        # 'pendulum-complex'


def step(x: Array, u: Array, p: PendulumParams) -> Array:
    """One ``PendulumDx.forward`` step.  ``x = (cos th, sin th, dth)``.

    Transcribed from Amos' code, including the sign convention ``-sin_th``
    inside the gravity term and the fact that the damping branch damps ``th``
    (the angle) rather than ``dth`` -- that is what their code does, and
    changing it would change the expert.
    """
    u = jnp.clip(u, -MAX_TORQUE, MAX_TORQUE)[0]
    cos_th, sin_th, dth = x[0], x[1], x[2]
    th = jnp.arctan2(sin_th, cos_th)
    if p.simple:
        newdth = dth + DT * (-3.0 * p.g / (2.0 * p.l) * (-sin_th)
                             + 3.0 * u / (p.m * p.l ** 2))
    else:
        sin_th_bias = jnp.sin(th + p.b)
        newdth = dth + DT * (-3.0 * p.g / (2.0 * p.l) * (-sin_th_bias)
                             + 3.0 * u / (p.m * p.l ** 2) - p.d * th)
    newth = th + newdth * DT
    return jnp.stack([jnp.cos(newth), jnp.sin(newth), newdth])


def rollout(x0: Array, us: Array, p: PendulumParams) -> Array:
    """Open-loop rollout.  ``us (T, 1)`` -> states ``(T, 3)`` after each step."""

    def body(x, u):
        nx = step(x, u, p)
        return nx, nx

    _, xs = jax.lax.scan(body, x0, us)
    return xs


def trajectory_cost(xs: Array, us: Array) -> Array:
    """``sum_t [ 0.5 tau_t^T diag(q) tau_t + p^T tau_t ]`` -- Amos' ``QuadCost``.

    ``q`` and ``p`` are exactly ``PendulumDx.get_true_obj()``::

        q = (goal_weights, ctrl_penalty) = (1, 1, 0.1, 0.001)
        p = (-sqrt(goal_weights) * goal_state, 0) = (-1, 0, 0, 0)

    and ``mpc.pytorch``'s ``QuadCost`` is ``0.5 tau^T C tau + c^T tau``.  This
    is the metric Table 8 reports.

    It is affinely related to the "weighted distance to a goal state" form
    ``||sqrt(q) o (tau - tau_g)||^2`` that Amos' Sec. 5.3 text describes --
    per step the two differ by exactly ``weighted = 2 * quad + 1`` -- so they
    induce the same optimal controls and differ only in reported magnitude.
    The quadratic form is the one whose numbers match Table 8; the weighted
    form runs about ``2 * cost + 20`` over ``T = 20``.
    """
    tau = jnp.concatenate([xs, us], axis=-1)
    q = jnp.concatenate([GOAL_WEIGHTS, jnp.array([CTRL_PENALTY])])
    p_lin = jnp.concatenate([-jnp.sqrt(GOAL_WEIGHTS) * GOAL_STATE,
                             jnp.zeros(1)])
    return jnp.sum(0.5 * jnp.sum(tau ** 2 * q, axis=-1) + tau @ p_lin)


def sample_xinit(n: int, seed: int = 0) -> np.ndarray:
    """``IL_Env.sample_xinit``: ``th ~ U(-pi/2, pi/2)``, ``dth ~ U(-1, 1)``.

    Amos draws these with ``torch.manual_seed(seed)``; we cannot reproduce
    Torch's RNG stream from JAX, so the *distribution* matches and the exact
    sample does not.  With 120 initial states the reported means are stable to
    well under the +-0.19 spread Table 8 quotes, but this is the one place a
    small discrepancy against their numbers can come from.
    """
    rng = np.random.default_rng(seed)
    th = rng.uniform(-np.pi / 2, np.pi / 2, size=n)
    thdot = rng.uniform(-1.0, 1.0, size=n)
    return np.stack([np.cos(th), np.sin(th), thdot], axis=1)


def _expert_objective(x0: Array, p: PendulumParams,
                      barrier: float) -> Callable[[Array], Array]:
    """Cost over a flat ``u_{1:T}``, with the torque box as a soft barrier."""

    def cost(u_flat: Array) -> Array:
        us = u_flat.reshape(T_HORIZON, 1)
        xs = rollout(x0, us, p)
        box = jnp.sum(jnp.maximum(jnp.abs(us) - MAX_TORQUE, 0.0) ** 2)
        return trajectory_cost(xs, us) + barrier * box

    return cost


def solve_expert(xinit: Array, p: PendulumParams, n_iters: int = 500,
                 barrier: float = 100.0, restarts: int = 4,
                 seed: int = 0) -> Array:
    """Expert nominal action sequences ``(B, T, 1)`` for a batch of states.

    Multi-start because the swing-up objective is non-convex and L-BFGS from a
    single zero initialisation drops into a "hang at the bottom" local minimum
    on a minority of initial states -- the same failure Amos avoids by using
    iLQR with a line search over a box-constrained subproblem.  The best of
    ``restarts`` is kept, scored on the true cost with the barrier excluded.
    """
    from pyroffi.optimization_engines import (DynamicsTrajOptConfig,
                                              dynamics_trajopt)

    cfg = DynamicsTrajOptConfig(n_iters=n_iters, grad_tol=1e-10,
                                early_stop=True, soft_line_search=False,
                                soft_curvature_gate=False)
    key = jax.random.PRNGKey(seed)
    n = T_HORIZON

    def solve_one(x0, u0):
        return dynamics_trajopt(u0, _expert_objective(x0, p, barrier), cfg)

    xinit = jnp.asarray(xinit)
    B = xinit.shape[0]
    inits = [jnp.zeros((B, n))]
    for i in range(restarts - 1):
        inits.append(MAX_TORQUE * jax.random.uniform(
            jax.random.fold_in(key, i), (B, n), minval=-1.0, maxval=1.0))

    best_u, best_c = None, jnp.full((B,), jnp.inf)
    score = jax.vmap(lambda x0, uf: trajectory_cost(
        rollout(x0, uf.reshape(n, 1), p), uf.reshape(n, 1)))
    for u0 in inits:
        u = jax.vmap(solve_one)(xinit, u0.astype(xinit.dtype))
        c = score(xinit, u)
        better = c < best_c
        best_u = u if best_u is None else jnp.where(better[:, None], u, best_u)
        best_c = jnp.where(better, c, best_c)
    return best_u.reshape(B, n, 1)


def make_dataset(p: PendulumParams, n_train: int = 100, n_val: int = 10,
                 n_test: int = 10, seed: int = 0, **solve_kw):
    """``make_dataset.py``: 100/10/10 split of (x_init, expert u_{1:T}) pairs."""
    n = n_train + n_val + n_test
    x = jnp.asarray(sample_xinit(n, seed))
    u = solve_expert(x, p, seed=seed, **solve_kw)
    cut = (n_train, n_train + n_val)
    return {"train": (x[:cut[0]], u[:cut[0]]),
            "val": (x[cut[0]:cut[1]], u[cut[0]:cut[1]]),
            "test": (x[cut[1]:], u[cut[1]:])}


def policy_cost(xinit: Array, us: Array, p: PendulumParams) -> Array:
    """Table 8's metric: roll the policy's ``u_{1:T}`` out on the TRUE dynamics
    and accumulate the true cost.  Returns per-initial-state costs ``(B,)``.

    Note this scores the policy's actions under the real system, so a learned
    dynamics model that is wrong shows up here even when the imitation MSE is
    small -- which is the whole point of the damped row, where Amos' assumed
    dynamics cannot represent the expert.
    """
    return jax.vmap(lambda x0, u: trajectory_cost(rollout(x0, u, p), u))(
        jnp.asarray(xinit), jnp.asarray(us))
