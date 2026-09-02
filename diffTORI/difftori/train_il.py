"""Imitation-learning training loop for DiffTORI, with TensorBoard logging.

Data-source agnostic: pass any iterator of
``(obs (B, n_obs_steps, obs_dim), expert_chunk (B, horizon, action_dim))``
batches -- e.g. ``difftori.data.dataset.batches``.  Nothing runs at import time.

Logged per step: total loss, the reconstruction and KL terms separately (they
are weighted 3000 and 10, so the raw terms are what tell you which one is
actually driving the gradient), the learning rate, and the gradient norm.

Logged periodically: **inner-solve stationarity**, ``||grad_x C||`` at the
returned actions.  This is the one diagnostic that decides whether the run means
anything -- the implicit adjoint is exact only at a stationary point, and in the
IOC experiments non-stationary contexts dropped gradient agreement from
cos 0.9999 to 0.59.  If ``diag/stationarity_max`` climbs, the gradients are
quietly wrong and no amount of loss curve will show it.  It costs an extra
forward solve, hence ``diag_every``.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from jax import Array

from .checkpoint import save_params
from .config import ILConfig
from .policy_il import DiffTORIPolicy, il_loss, make_solver
from .solver import DiffTORISolver
from .tblog import Logger

__all__ = ["create_train_state", "make_train_step", "make_eval_step", "train"]


def create_train_state(
    cfg: ILConfig, rng: Array, obs_encoder=None
) -> tuple[DiffTORIPolicy, TrainState]:
    module = DiffTORIPolicy(cfg=cfg, obs_encoder=obs_encoder)
    dummy_obs = jnp.zeros((1, cfg.n_obs_steps, cfg.obs_dim))
    dummy_act = jnp.zeros((1, cfg.chunk_len, cfg.action_dim))
    params = module.init(rng, dummy_obs, dummy_act)["params"]
    # Adam with cosine decay, as in the released code's optimizer + scheduler.
    schedule = optax.cosine_decay_schedule(
        cfg.learning_rate, cfg.lr_schedule_steps,
        alpha=cfg.lr_min / cfg.learning_rate)
    steps = [optax.adam(schedule)]
    if cfg.grad_norm > 0:   # released code leaves clipping commented out
        steps.insert(0, optax.clip_by_global_norm(cfg.grad_norm))
    state = TrainState.create(apply_fn=module.apply, params=params,
                              tx=optax.chain(*steps))
    return module, state


def make_train_step(module: DiffTORIPolicy,
                    solver: DiffTORISolver) -> Callable:
    schedule = optax.cosine_decay_schedule(
        module.cfg.learning_rate, module.cfg.lr_schedule_steps,
        alpha=module.cfg.lr_min / module.cfg.learning_rate)

    @jax.jit
    def step(state: TrainState, obs: Array, expert_chunk: Array, rng: Array):
        grad_fn = jax.value_and_grad(
            lambda p: il_loss(module, solver, p, obs, expert_chunk, rng),
            has_aux=True)
        (_, metrics), grads = grad_fn(state.params)
        metrics = {**metrics,
                   "grad_norm": optax.global_norm(grads),
                   "lr": schedule(state.step)}
        return state.apply_gradients(grads=grads), metrics

    return step


def make_eval_step(module: DiffTORIPolicy,
                   solver: DiffTORISolver) -> Callable:
    """Validation loss and the inner-solve stationarity diagnostic."""

    @jax.jit
    def evaluate(state: TrainState, obs: Array, expert_chunk: Array,
                 rng: Array):
        _, metrics = il_loss(module, solver, state.params, obs, expert_chunk,
                             rng)
        # Re-derive the latent the same way il_loss does, then measure how
        # close the returned actions are to a stationary point of the cost.
        from .policy_il import _latent

        z, _, _ = _latent(module, state.params, obs, expert_chunk, rng,
                          sample=True)
        n = module.cfg.chunk_len * module.cfg.action_dim
        stat = solver.stationarity(jnp.zeros((z.shape[0], n), z.dtype),
                                   state.params, z)
        return {**metrics, "stationarity_max": jnp.max(stat),
                "stationarity_mean": jnp.mean(stat)}

    return evaluate


def train(
    cfg: ILConfig,
    batches: Iterator[tuple[Array, Array]],
    rng: Array,
    steps: int,
    obs_encoder=None,
    val_batches: Iterator[tuple[Array, Array]] | None = None,
    log_every: int = 10,
    print_every: int = 100,
    diag_every: int = 250,
    run_name: str = "difftori_il",
    log_dir: str = "runs",
    logging: bool = True,
    ckpt_every: int = 1000,
) -> tuple[DiffTORIPolicy, TrainState]:
    """Train, log and checkpoint into ``<log_dir>/<run_name>-<timestamp>/``.

    Checkpoints are written on the same schedule as the diagnostics: the best
    validation reconstruction seen so far, plus a periodic snapshot every
    ``ckpt_every`` steps so a killed run still leaves something usable.
    """
    module, state = create_train_state(cfg, rng, obs_encoder)
    solver = make_solver(module)
    step_fn = make_train_step(module, solver)
    eval_fn = make_eval_step(module, solver)

    with Logger(run_name, log_dir, config=cfg, enabled=logging) as logger:
        if logger.dir is not None:
            print(f"logging to {logger.dir}")
        best_val = float("inf")
        for i in range(steps):
            obs, expert_chunk = next(batches)
            rng, sub = jax.random.split(rng)
            state, metrics = step_fn(state, jnp.asarray(obs),
                                     jnp.asarray(expert_chunk), sub)
            metrics = {k: float(v) for k, v in metrics.items()}

            if i % log_every == 0:
                logger.log(i, metrics)
            if i % print_every == 0:
                print(f"[{i}] " + " ".join(
                    f"{k}={v:.4g}" for k, v in metrics.items()))
            if diag_every and i % diag_every == 0:
                rng, sub = jax.random.split(rng)
                src = val_batches if val_batches is not None else batches
                v_obs, v_act = next(src)
                diag = {k: float(v) for k, v in eval_fn(
                    state, jnp.asarray(v_obs), jnp.asarray(v_act), sub).items()}
                logger.log(i, diag, prefix="val" if val_batches else "diag")
                print(f"[{i}] val/diag " + " ".join(
                    f"{k}={v:.4g}" for k, v in diag.items()))
                if logger.dir is not None and diag["recon"] < best_val:
                    best_val = diag["recon"]
                    save_params(logger.dir / "params_best.msgpack", state.params)

            if (logger.dir is not None and ckpt_every
                    and i and i % ckpt_every == 0):
                save_params(logger.dir / f"params_step_{i}.msgpack", state.params)

        if logger.dir is not None:
            save_params(logger.dir / "params_final.msgpack", state.params)
            print(f"checkpoints in {logger.dir}")
    return module, state
