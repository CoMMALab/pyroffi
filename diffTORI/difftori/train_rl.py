"""Model-based RL training step for DiffTORI (Eq. 6).

The environment loop is left to the caller (the paper uses the TD-MPC codebase
on DMControl).  This module supplies the two gradient steps per update:
the DiffTORI world-model/policy-gradient step and the state-space critic step,
plus the target-network EMA.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from jax import Array

from .agent_rl import (DiffTORIAgent, StateCritic, critic_loss, difftori_loss,
                       make_solver, planning_horizon)
from .solver import DiffTORISolver
from .config import RLConfig

__all__ = ["create_train_states", "make_update", "soft_update", "make_logger"]


def make_logger(cfg: RLConfig, run_name: str = "difftori_rl",
                log_dir: str = "runs", enabled: bool = True):
    """A `tblog.Logger` for the RL loop.

    The environment loop lives with the caller, so logging cannot be wired in
    here the way `train_il.train` does it.  Call `logger.log(step, metrics)`
    with whatever `make_update` returns, and add the episode return alongside
    it -- the surrogate losses of Eq. 3 say nothing about task performance,
    which is the entire point DiffTORI is making.
    """
    from .tblog import Logger

    return Logger(run_name, log_dir, config=cfg, enabled=enabled)


def create_train_states(cfg: RLConfig, rng: Array, horizon: int = 1):
    agent = DiffTORIAgent(cfg=cfg, horizon=horizon)
    critic = StateCritic(cfg=cfg)
    k1, k2 = jax.random.split(rng)
    obs = jnp.zeros((1, cfg.obs_dim))
    act = jnp.zeros((1, cfg.action_dim))
    agent_params = agent.init(k1, obs, act)["params"]
    critic_params = critic.init(k2, obs, act)["params"]

    def tx(lr):
        return optax.chain(optax.clip_by_global_norm(cfg.grad_norm),
                           optax.adam(lr, b1=0.9, b2=0.999))

    agent_state = TrainState.create(
        apply_fn=agent.apply, params=agent_params, tx=tx(cfg.learning_rate))
    critic_state = TrainState.create(
        apply_fn=critic.apply, params=critic_params, tx=tx(cfg.learning_rate))
    return agent, critic, agent_state, critic_state, agent_params


def soft_update(target: Any, online: Any, tau: float) -> Any:
    return jax.tree.map(lambda t, o: (1 - tau) * t + tau * o, target, online)


def make_update(agent: DiffTORIAgent, critic: StateCritic,
                solver: DiffTORISolver | None = None) -> Callable:
    """One DiffTORI update: Eq. 6 for the model, Bellman for ``Q_tilde_phi``.

    ``batch`` must carry ``obs (B,H+1,S)``, ``action (B,H,A)``,
    ``reward (B,H)``, ``td_target (B,H)`` and ``state_td_target (B,)``; the
    caller computes both TD targets with the target networks (TD-MPC does the
    same), which keeps this function free of stale-parameter bookkeeping.

    ``solver`` defaults to ``pyroffi``'s dynamics-aware L-BFGS engine.  It
    depends on ``agent.horizon``, so rebuild both when the 1->5 planning-horizon
    schedule steps.
    """
    if solver is None:
        solver = make_solver(agent)

    @jax.jit
    def update(agent_state: TrainState, critic_state: TrainState,
               target_params: Any, batch: dict[str, Array]):
        (_, metrics), grads = jax.value_and_grad(
            lambda p: difftori_loss(agent, solver, critic, p, target_params,
                                    critic_state.params, batch),
            has_aux=True)(agent_state.params)
        agent_state = agent_state.apply_gradients(grads=grads)

        c_loss, c_grads = jax.value_and_grad(
            lambda p: critic_loss(critic, p, batch))(critic_state.params)
        critic_state = critic_state.apply_gradients(grads=c_grads)

        target_params = soft_update(target_params, agent_state.params,
                                    agent.cfg.tau)
        return agent_state, critic_state, target_params, {
            **metrics, "critic_loss": c_loss}

    return update
