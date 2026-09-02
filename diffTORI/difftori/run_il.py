"""CLI: train the DiffTORI imitation policy on a generated zarr dataset.

    XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=2 \\
        PYTHONPATH=diffTORI python -m difftori.run_il --steps 15000

Run from the repository root.  ``--steps 0`` builds everything and exits, which
is the cheapest way to check shapes before committing a GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax

from .config import ILConfig, SolverConfig
from .data.dataset import ReplayBuffer, SequenceDataset, batches
from .train_il import train


def main(
    data: str = "diffTORI/data/panda_reach_expert.zarr",
    steps: int = 15_000,
    seed: int = 0,
    horizon: int = 4,
    n_obs_steps: int = 2,
    batch_size: int = 128,
    n_iters: int = 100,
    learning_rate: float = 1e-4,
    kl_coefficient: float = 10.0,
    action_loss_weight: float = 3000.0,
    val_ratio: float = 0.02,
    log_every: int = 10,
    print_every: int = 100,
    diag_every: int = 250,
    run_name: str = "difftori_il_panda",
    log_dir: str = "diffTORI/runs",
    logging: bool = True,
    ckpt_every: int = 1000,
):
    # Resolve repo-relative paths against the repo root, not the cwd.  Leaving
    # them cwd-relative means the same command run from `diffTORI/` silently
    # writes its run somewhere else -- already cost one debugging detour in
    # `data/visualize.py`.
    data = str(_ROOT / data) if not Path(data).is_absolute() else data
    log_dir = str(_ROOT / log_dir) if not Path(log_dir).is_absolute() else log_dir
    buf = ReplayBuffer.load(data)
    train_ds = SequenceDataset(buf, n_obs_steps=n_obs_steps, horizon=horizon,
                               val_ratio=val_ratio, seed=seed)
    val_ds = SequenceDataset(buf, n_obs_steps=n_obs_steps, horizon=horizon,
                             val_ratio=val_ratio, seed=seed,
                             validation=True).share_normalizers(train_ds)

    cfg = ILConfig(
        action_dim=buf.action.shape[-1],
        obs_dim=buf.state.shape[-1],
        horizon=horizon,
        n_obs_steps=n_obs_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        kl_coefficient=kl_coefficient,
        action_loss_weight=action_loss_weight,
        lr_schedule_steps=max(steps, 1),
        solver=SolverConfig(n_iters=n_iters),
    )
    print(f"task={buf.meta.get('task')} episodes={buf.n_episodes} "
          f"train/val={len(train_ds)}/{len(val_ds)} "
          f"obs_dim={cfg.obs_dim} action_dim={cfg.action_dim}")
    print(f"jax devices: {jax.devices()}")
    if not jax.config.jax_enable_x64:
        print("WARNING: x64 is OFF; the implicit adjoint inverts the inner "
              "Hessian and wants float64.")
    if steps == 0:
        return cfg

    return train(
        cfg,
        batches(train_ds, batch_size, seed=seed),
        jax.random.PRNGKey(seed),
        steps=steps,
        val_batches=batches(val_ds, min(batch_size, len(val_ds)), seed=seed + 1),
        log_every=log_every, print_every=print_every, diag_every=diag_every,
        run_name=run_name, log_dir=log_dir, logging=logging,
        ckpt_every=ckpt_every,
    )


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
