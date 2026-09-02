"""DiffTORI for model-based RL (Sec. 4.2), built on TD-MPC.

Everything TD-MPC has is kept -- encoder ``h``, latent dynamics ``d``, reward
``R``, latent value ``Q``, latent policy ``pi`` and the surrogate loss of
Eq. 3.  The two changes DiffTORI makes are:

1. MPPI planning is replaced by *differentiable* trajectory optimisation of
   Eq. 4, with the dynamics substituted into the objective (Eq. 5) and the
   actions initialised from ``pi(z)``.  Unlike the imitation-learning policy,
   the RL agent really does roll out the latent dynamics -- this matches both
   the paper and ``mbrl/src/algorithm/tdmpc.py``;
2. a deterministic policy-gradient loss is put on the optimised actions and
   back-propagated *through the optimiser* (Eq. 6), so the encoder, dynamics
   and reward models are trained for task performance rather than only for
   their surrogate losses -- this is the fix for objective mismatch.

The Q used for the policy gradient (``Q_tilde_phi``) lives in the original
state space, not the latent space (Sec. 4.2), and is trained by ordinary
Bellman backups; its gradient must not be allowed to reach ``theta`` except
through ``a(theta)``.
"""

from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import Array

from .config import RLConfig
from .networks import (LatentDynamics, LatentPolicy, LatentQ, LatentReward,
                       StateEncoder, TwinStateQ)
from .solver import DiffTORISolver, make_difftori_solver

__all__ = ["DiffTORIAgent", "StateCritic", "make_solver", "planning_horizon",
           "tdmpc_loss", "difftori_loss", "critic_loss", "plan"]


def planning_horizon(step: int, cfg: RLConfig) -> int:
    """Linear 1 -> 5 schedule over 25k steps (Table 9). Static per-step value."""
    frac = min(max(step / cfg.horizon_anneal_steps, 0.0), 1.0)
    return int(round(cfg.horizon_start
                     + frac * (cfg.horizon_end - cfg.horizon_start)))


class DiffTORIAgent(nn.Module):
    """Latent-space world model + policy (the TD-MPC components)."""

    cfg: RLConfig
    horizon: int = 1          # static; rebuild/re-jit when the schedule steps
    obs_encoder: nn.Module | None = None

    def setup(self):
        c = self.cfg
        self.h = self.obs_encoder or StateEncoder(c.latent_dim, c.enc_hidden)
        self.d = LatentDynamics(c.latent_dim, c.mlp_hidden)
        self.R = LatentReward(c.mlp_hidden)
        # Twin latent Q, clipped-double-Q as in TD-MPC.
        self.Q1 = LatentQ(c.mlp_hidden)
        self.Q2 = LatentQ(c.mlp_hidden)
        self.pi = LatentPolicy(c.action_dim, c.mlp_hidden)

    def encode(self, obs: Array) -> Array:
        return self.h(obs)

    def next_latent(self, z: Array, action: Array) -> Array:
        return self.d(z, action)

    def reward(self, z: Array, action: Array) -> Array:
        return self.R(z, action)

    def value(self, z: Array, action: Array) -> tuple[Array, Array]:
        return self.Q1(z, action), self.Q2(z, action)

    def min_value(self, z: Array, action: Array) -> Array:
        q1, q2 = self.value(z, action)
        return jnp.minimum(q1, q2)

    def policy(self, z: Array) -> Array:
        return self.pi(z)

    def plan_cost(self, z: Array, actions_flat: Array) -> Array:
        """Eq. 5, negated: ``-[sum_t gamma^t R(z_t,a_t) + gamma^H min Q(z_H, pi(z_H))]``.

        The decision variables are the ``H`` actions of the horizon.  The
        terminal value is evaluated at the *policy's* action, not at a free
        variable -- that is what ``mbrl/src/algorithm/tdmpc.py`` does, and it
        differs from Eq. 4 of the paper, which writes ``Q(z_H, a_H)`` with
        ``a_H`` optimised.

        The barrier replaces the released code's ``torch.clamp(a, -1, 1)``; see
        ``policy_il.plan_cost`` for why a clamp is the wrong bound here.
        """
        c = self.cfg
        actions = actions_flat.reshape(self.horizon, c.action_dim)
        z_t, total, discount = z, jnp.zeros((), actions.dtype), 1.0
        for t in range(self.horizon):
            total = total - discount * self.R(z_t, actions[t])
            z_t = self.d(z_t, actions[t])
            discount = discount * c.discount
        total = total - discount * self.min_value(z_t, self.pi(z_t))
        barrier = jnp.sum(jnp.maximum(jnp.abs(actions) - 1.0, 0.0) ** 2)
        return total + c.solver.action_penalty * barrier

    def __call__(self, obs: Array, action: Array) -> Array:
        """Trace every submodule once so ``init`` sees all parameters."""
        z = self.encode(obs)
        self.reward(z, action)
        self.value(z, action)
        self.policy(z)
        self.next_latent(z, action)
        n = self.horizon * self.cfg.action_dim
        return self.plan_cost(z[0], jnp.zeros(n, z.dtype))


class StateCritic(nn.Module):
    """``Q_tilde_phi``: twin critic over raw states, for the policy gradient."""

    cfg: RLConfig

    @nn.compact
    def __call__(self, obs: Array, action: Array) -> tuple[Array, Array]:
        return TwinStateQ(self.cfg.latent_dim, self.cfg.enc_hidden,
                          self.cfg.mlp_hidden)(obs, action)


def make_solver(agent: DiffTORIAgent, forward_solver=None,
                unrolled_forward_solver=None) -> DiffTORISolver:
    """Inner solver for the RL agent; defaults to ``pyroffi``'s L-BFGS engine.

    Rebuild it whenever ``agent.horizon`` changes: the horizon sets the number
    of decision variables.
    """
    cfg = agent.cfg
    if forward_solver is None:
        from .pyroffi_trajopt import make_dynamics_forward_solver

        forward_solver = make_dynamics_forward_solver(
            n_iters=cfg.solver.n_iters, grad_tol=cfg.solver.grad_tol,
            m_lbfgs=cfg.solver.m_lbfgs, smooth=cfg.solver.smooth)

    def cost(x, params, z):
        return agent.apply({"params": params}, z, x,
                           method=DiffTORIAgent.plan_cost)

    return make_difftori_solver(
        cost, forward_solver=forward_solver,
        unrolled_forward_solver=unrolled_forward_solver,
        adjoint_ridge=cfg.solver.adjoint_ridge)


def plan(agent: DiffTORIAgent, solver: DiffTORISolver, params: Any,
         obs: Array) -> Array:
    """Differentiable trajectory optimisation of Eq. 4; ``(B, H, A)``.

    Initialised from ``pi(z)`` as in Sec. 4.2.  That initialisation carries no
    gradient -- at a stationary point the solution does not depend on it.
    """
    cfg = agent.cfg
    z = agent.apply({"params": params}, obs, method=DiffTORIAgent.encode)
    a_pi = jax.lax.stop_gradient(
        agent.apply({"params": params}, z, method=DiffTORIAgent.policy))
    x0 = jnp.tile(a_pi[:, None, :], (1, agent.horizon, 1)).reshape(
        z.shape[0], -1)
    x = solver.solve_implicit(x0, params, z)
    return x.reshape(z.shape[0], agent.horizon, cfg.action_dim)


def tdmpc_loss(
    agent: DiffTORIAgent,
    params: Any,
    target_params: Any,
    batch: dict[str, Array],
) -> tuple[Array, dict[str, Array]]:
    """Eq. 2/3: reward, value and latent-consistency terms, ``rho``-weighted.

    ``batch``: ``obs`` ``(B, H+1, S)``, ``action`` ``(B, H, A)``,
    ``reward`` ``(B, H)``, ``td_target`` ``(B, H)`` (computed by the caller
    from the target networks, as in TD-MPC).
    """
    cfg = agent.cfg
    ap = lambda p, *a, **kw: agent.apply({"params": p}, *a, **kw)
    z = ap(params, batch["obs"][:, 0], method=DiffTORIAgent.encode)

    total = jnp.zeros(())
    parts = {"reward": jnp.zeros(()), "value": jnp.zeros(()),
             "consistency": jnp.zeros(())}
    horizon = batch["action"].shape[1]
    for i in range(horizon):
        a_i = batch["action"][:, i]
        r_hat = ap(params, z, a_i, method=DiffTORIAgent.reward)
        q1, q2 = ap(params, z, a_i, method=DiffTORIAgent.value)
        z_next = ap(params, z, a_i, method=DiffTORIAgent.next_latent)
        z_target = jax.lax.stop_gradient(
            ap(target_params, batch["obs"][:, i + 1], method=DiffTORIAgent.encode))

        w = cfg.rho ** i
        reward = jnp.mean((r_hat - batch["reward"][:, i]) ** 2)
        value = (jnp.mean((q1 - batch["td_target"][:, i]) ** 2)
                 + jnp.mean((q2 - batch["td_target"][:, i]) ** 2))
        consistency = jnp.mean(jnp.sum((z_next - z_target) ** 2, axis=-1))
        total = total + w * (cfg.reward_coef * reward
                             + cfg.value_coef * value
                             + cfg.consistency_coef * consistency)
        parts = {"reward": parts["reward"] + w * reward,
                 "value": parts["value"] + w * value,
                 "consistency": parts["consistency"] + w * consistency}
        z = z_next
    return total, parts


def difftori_loss(
    agent: DiffTORIAgent,
    solver: DiffTORISolver,
    critic: StateCritic,
    params: Any,
    target_params: Any,
    critic_params: Any,
    batch: dict[str, Array],
) -> tuple[Array, dict[str, Array]]:
    """Eq. 6: TD-MPC surrogate losses + ``c0 * L_PG`` on the planned action."""
    cfg = agent.cfg
    td_loss, parts = tdmpc_loss(agent, params, target_params, batch)

    actions = plan(agent, solver, params, batch["obs"][:, 0])
    q1, q2 = critic.apply({"params": jax.lax.stop_gradient(critic_params)},
                          batch["obs"][:, 0], actions[:, 0])
    pg = -jnp.mean(jnp.minimum(q1, q2))

    loss = td_loss + cfg.action_loss_coefficient * pg
    return loss, {**parts, "policy_gradient": pg, "loss": loss}


def critic_loss(
    critic: StateCritic,
    critic_params: Any,
    batch: dict[str, Array],
) -> Array:
    """Bellman update for ``Q_tilde_phi`` (targets supplied by the caller)."""
    q1, q2 = critic.apply({"params": critic_params},
                          batch["obs"][:, 0], batch["action"][:, 0])
    y = jax.lax.stop_gradient(batch["state_td_target"])
    return jnp.mean((q1 - y) ** 2) + jnp.mean((q2 - y) ** 2)
