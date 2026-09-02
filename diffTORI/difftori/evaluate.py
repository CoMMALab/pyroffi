"""Probe a trained policy: is it actually using its inputs, and its optimizer?

Loss curves say a number went down.  They do not say whether the policy uses
the observation, whether the inner trajectory optimization contributes anything,
or whether the CVAE latent does.  Each check below isolates one of those.

    PYTHONPATH=diffTORI python -m difftori.evaluate --run <run_dir>

Checks
------
``val_mse``          reconstruction on the held-out split (should match the
                     run's logged ``val/recon``; a mismatch means the checkpoint
                     and the config disagree).
``shuffled_obs``     the same batch with observations permuted across samples.
                     If this is no worse than ``val_mse``, the policy is
                     ignoring the observation and has memorised the marginal.
``init_movement``    how far the solve moves the actions from ``a_init``.  Near
                     zero means the inner optimization is decorative and the
                     barrier or the initialisation is deciding the output.
``latent_spread``    action variation across prior samples.  Near zero is
                     posterior collapse: the decoder ignores z.
``per_joint``        which joints carry the error.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from difftori.checkpoint import latest_checkpoint, load_params
from difftori.config import ILConfig, SolverConfig
from difftori.data.dataset import ReplayBuffer, SequenceDataset, batches
from difftori.policy_il import DiffTORIPolicy, _latent, _plan, make_solver


def evaluate(run: str, data: str | None = None, n_batches: int = 2,
             batch_size: int = 128, n_iters: int | None = None,
             latent_samples: int = 8, seed: int = 0):
    run_dir = Path(run if Path(run).is_absolute() else _ROOT / run)
    meta = json.loads((run_dir / "config.json").read_text())
    c = meta["config"]
    data = data or "diffTORI/data/panda_reach_expert_v2.zarr"
    buf = ReplayBuffer.load(str(_ROOT / data) if not Path(data).is_absolute() else data)

    cfg = ILConfig(
        action_dim=int(c["action_dim"]), obs_dim=int(c["obs_dim"]),
        horizon=int(c["horizon"]), n_obs_steps=int(c["n_obs_steps"]),
        obs_feature_dim=int(c["obs_feature_dim"]),
        posterior_dim=int(c["posterior_dim"]), mlp_hidden=int(c["mlp_hidden"]),
        solver=SolverConfig(n_iters=n_iters or int(c["solver.n_iters"])))

    module = DiffTORIPolicy(cfg=cfg)
    solver = make_solver(module)
    rng = jax.random.PRNGKey(seed)
    template = module.init(
        rng, jnp.zeros((1, cfg.n_obs_steps, cfg.obs_dim)),
        jnp.zeros((1, cfg.chunk_len, cfg.action_dim)))["params"]
    ckpt = latest_checkpoint(run_dir)
    params = load_params(ckpt, template)
    print(f"run {run_dir.name}  checkpoint {ckpt.name}")

    tr = SequenceDataset(buf, cfg.n_obs_steps, cfg.horizon)
    va = SequenceDataset(buf, cfg.n_obs_steps, cfg.horizon,
                         validation=True).share_normalizers(tr)
    it = batches(va, min(batch_size, len(va)), seed=seed)

    mse, mse_shuf, moved, spread, per_joint = [], [], [], [], []
    for _ in range(n_batches):
        obs, act = next(it)
        obs, act = jnp.asarray(obs), jnp.asarray(act)
        rng, k1, k2 = jax.random.split(rng, 3)

        z, _, _ = _latent(module, params, obs, act, k1, sample=False)
        pred = _plan(solver, cfg, params, z, None, k2)
        mse.append(float(jnp.mean((pred - act) ** 2)))
        per_joint.append(np.asarray(jnp.mean((pred - act) ** 2, axis=(0, 1))))
        # a_init is zeros here, so this is just the norm of the solution.
        moved.append(float(jnp.mean(jnp.linalg.norm(
            pred.reshape(pred.shape[0], -1), axis=-1))))

        # Observation ablation: same targets, mismatched observations.
        perm = jax.random.permutation(k1, obs.shape[0])
        z_s, _, _ = _latent(module, params, obs[perm], act, k1, sample=False)
        pred_s = _plan(solver, cfg, params, z_s, None, k2)
        mse_shuf.append(float(jnp.mean((pred_s - act) ** 2)))

        # Latent ablation: same observation, different prior draws.
        preds = []
        for j in range(latent_samples):
            zj, _, _ = _latent(module, params, obs, act,
                               jax.random.fold_in(k2, j), sample=False)
            preds.append(_plan(solver, cfg, params, zj, None, k2))
        spread.append(float(jnp.mean(jnp.std(jnp.stack(preds), axis=0))))

    out = {
        "val_mse": float(np.mean(mse)),
        "shuffled_obs_mse": float(np.mean(mse_shuf)),
        "action_norm": float(np.mean(moved)),
        "latent_spread": float(np.mean(spread)),
        "per_joint_mse": np.mean(per_joint, axis=0).tolist(),
    }
    # Two normalisations sit between a stored action and radians: the
    # generator's global `action_scale`, and the loader's per-joint limits map.
    # Applying only the first understates the error by the second's factor.
    scale = buf.meta.get("action_scale", 1.0) * (tr.act_norm.scale / 2.0)
    print(f"  val_mse            {out['val_mse']:.6f}")
    print(f"  shuffled_obs_mse   {out['shuffled_obs_mse']:.6f}"
          f"   ({out['shuffled_obs_mse'] / max(out['val_mse'], 1e-12):.1f}x worse)")
    print(f"  action_norm        {out['action_norm']:.4f}  "
          f"(solution distance from a_init=0)")
    print(f"  latent_spread      {out['latent_spread']:.6f}  "
          f"(action std across {latent_samples} prior samples)")
    print(f"  per-joint RMSE(rad) "
          + " ".join(f"{(m ** 0.5) * sj:.4f}"
                     for m, sj in zip(out["per_joint_mse"], np.atleast_1d(scale))))
    return out


def main(run: str, data: str | None = None, n_batches: int = 2,
         batch_size: int = 128, n_iters: int | None = None,
         latent_samples: int = 8, seed: int = 0):
    evaluate(run, data, n_batches, batch_size, n_iters, latent_samples, seed)


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
