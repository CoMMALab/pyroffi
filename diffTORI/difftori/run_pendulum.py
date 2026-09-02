"""Reproduce Table 8: DiffTORI vs the LSTM baseline on Amos' pendulum swing-up.

    conda activate pyroffi
    JAX_ENABLE_X64=1 PYTHONPATH=diffTORI python -m difftori.run_pendulum \\
        --setting both --seeds 3

Every solve -- the expert's, and DiffTORI's inner trajectory optimisation --
runs on ``pyroffi.optimization_engines.dynamics_trajopt``.  The paper uses
Theseus, whose Levenberg--Marquardt accepts only nonlinear least squares; the
scalar objective is optimised here directly.

The policy follows **Eq. 7 and Appendix D**, not the released code:
``ILConfig.paper()`` turns on the latent dynamics rollout ``d_theta``, the
three-network CVAE encoder, and Table 9's hyperparameters.  ``planning_horizon``
is set to ``T - 1`` so the decoder decides the full 20-step nominal action
sequence the expert demonstrates, with ``d_theta`` applied 19 times.

Actions are divided by ``MAX_TORQUE`` so they live in the unit box the inner
problem's barrier assumes, and multiplied back before scoring.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from difftori.benchmarks.lstm_baseline import train_lstm
from difftori.benchmarks.pendulum import (COMPLEX, MAX_TORQUE, SIMPLE,
                                          T_HORIZON, make_dataset, policy_cost)
from difftori.config import ILConfig, SolverConfig
from difftori.policy_il import (DiffTORIPolicy, _latent, _plan, il_loss,
                                make_solver)

# Table 8, for reporting alongside our numbers.
TABLE8 = {
    "simple":  {"expert": 13.126, "amos": 13.576, "lstm": 15.962, "difftori": 14.603},
    "complex": {"expert": 10.132, "amos": 14.874, "lstm": 12.098, "difftori": 10.644},
}


def train_difftori(data, cfg: ILConfig, steps: int, seed: int, lr: float):
    import optax
    from flax.training.train_state import TrainState

    x_tr, u_tr = data["train"]
    x_va, u_va = data["val"]
    obs_tr = x_tr[:, None, :]                     # (B, n_obs_steps=1, 3)
    act_tr = u_tr / MAX_TORQUE                    # into the unit box
    obs_va, act_va = x_va[:, None, :], u_va / MAX_TORQUE

    module = DiffTORIPolicy(cfg=cfg)
    rng = jax.random.PRNGKey(seed)
    params = module.init(rng, obs_tr[:1], act_tr[:1])["params"]
    solver = make_solver(module)
    state = TrainState.create(apply_fn=module.apply, params=params,
                              tx=optax.adam(lr))

    @jax.jit
    def step(state, obs, act, key):
        (loss, mets), g = jax.value_and_grad(
            lambda p: il_loss(module, solver, p, obs, act, key), has_aux=True)(
                state.params)
        return state.apply_gradients(grads=g), mets

    n = obs_tr.shape[0]
    rs = np.random.default_rng(seed)
    bs = min(cfg.batch_size, n)
    best, best_params = np.inf, state.params
    for i in range(steps):
        idx = rs.choice(n, size=bs, replace=False)
        rng, sub = jax.random.split(rng)
        state, mets = step(state, obs_tr[idx], act_tr[idx], sub)
        if i % 25 == 0 or i == steps - 1:
            rng, sub = jax.random.split(rng)
            _, vm = il_loss(module, solver, state.params, obs_va, act_va, sub)
            v = float(vm["recon"])
            if v < best:
                best, best_params = v, state.params
            if i % 200 == 0:
                print(f"    [{i:5d}] recon={float(mets['recon']):.5f} "
                      f"kl={float(mets['kl']):.4f} val_recon={v:.5f}")
    return module, solver, best_params


def difftori_actions(module, solver, params, x, cfg, seed=0):
    """Test-time: prior sample -> trajectory optimisation -> torques."""
    obs = jnp.asarray(x)[:, None, :]
    dummy = jnp.zeros((obs.shape[0], cfg.chunk_len, cfg.action_dim), obs.dtype)
    rng = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(rng)
    z, _, _ = _latent(module, params, obs, dummy, k1, sample=False)
    return _plan(solver, cfg, params, z, None, k2) * MAX_TORQUE


def run_setting(name: str, p, seeds: int, steps: int, n_iters: int,
                lstm_steps: int):
    ref = TABLE8[name]
    print(f"\n=== Pendulum {'with' if name == 'complex' else 'w/o'} damping ===")
    dt_costs, lstm_costs, exp_costs = [], [], []

    for sd in range(seeds):
        print(f"  seed {sd}")
        data = make_dataset(p, seed=sd, restarts=8)
        x_te, u_te = data["test"]
        exp_costs.append(float(np.mean(np.asarray(policy_cost(x_te, u_te, p)))))

        cfg = ILConfig.paper(
            action_dim=1, obs_dim=3, n_obs_steps=1,
            planning_horizon=T_HORIZON - 1,   # decide the full nominal sequence
            discount=1.0,                     # finite-horizon expert: no discount
            batch_size=32,
            solver=SolverConfig(n_iters=n_iters))
        module, solver, params = train_difftori(data, cfg, steps, sd,
                                                cfg.learning_rate)
        u_dt = difftori_actions(module, solver, params, x_te, cfg, seed=sd)
        dt_costs.append(float(np.mean(np.asarray(policy_cost(x_te, u_dt, p)))))

        lm, lp = train_lstm(data["train"][0], data["train"][1] / MAX_TORQUE,
                            data["val"][0], data["val"][1] / MAX_TORQUE,
                            horizon=T_HORIZON, steps=lstm_steps, seed=sd)
        u_ls = lm.apply({"params": lp}, jnp.asarray(x_te)) * MAX_TORQUE
        lstm_costs.append(float(np.mean(np.asarray(policy_cost(x_te, u_ls, p)))))
        print(f"    expert={exp_costs[-1]:.3f}  difftori={dt_costs[-1]:.3f}  "
              f"lstm={lstm_costs[-1]:.3f}")

    def fmt(v, target):
        a = np.array(v)
        return f"{a.mean():7.3f} +- {a.std():.3f}   (Table 8: {target})"

    print(f"\n  {'expert':10s} {fmt(exp_costs, ref['expert'])}")
    print(f"  {'DiffTORI':10s} {fmt(dt_costs, ref['difftori'])}")
    print(f"  {'LSTM':10s} {fmt(lstm_costs, ref['lstm'])}")
    print(f"  {'Amos et al.':10s} {'':7s}        (Table 8: {ref['amos']}) "
          f"-- not reimplemented")
    return {"expert": exp_costs, "difftori": dt_costs, "lstm": lstm_costs}


def main(setting: str = "both", seeds: int = 3, steps: int = 2000,
         n_iters: int = 100, lstm_steps: int = 4000):
    if not jax.config.jax_enable_x64:
        print("WARNING: x64 is OFF; the implicit adjoint inverts the inner "
              "Hessian and wants float64.  Re-run with JAX_ENABLE_X64=1.")
    print(f"jax devices: {jax.devices()}")
    out = {}
    for name, p in [("simple", SIMPLE), ("complex", COMPLEX)]:
        if setting in (name, "both"):
            out[name] = run_setting(name, p, seeds, steps, n_iters, lstm_steps)
    return out


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
