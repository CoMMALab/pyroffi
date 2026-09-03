"""E3 — Fit wide → Gram → select r → refit on the identifiable subspace.

Does restricting to U_r turn rank deficiency into a generalization cost (bad
transfer) rather than a reconstruction cost (bad fit)?  Runs on the UNANCHORED
scene deliberately — the point is not having to curate the demonstration.

Usage:
    CUDA_VISIBLE_DEVICES=<idx> XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        python -m iosp.experiments.e3_identifiable_refit
"""

import argparse

from iosp import config

config.enable_compilation_cache()

from iosp.fit.parametric import build_parametric
from iosp.fit.procedure import _report, run_procedure


def main(seed=0, n_iters=config.N_ITERS, n_steps=config.N_STEPS, lr=config.LR,
         space="ee", scene_b_scale=1.0):
    print("Path A (known cost library, K=9) on the UNANCHORED scene...", flush=True)
    built = build_parametric(seed=seed, n_iters=n_iters, space=space,
                             scene_b_scale=scene_b_scale)
    res = run_procedure(built, "path A (parametric)", n_steps=n_steps, lr=lr)
    _report(res)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-iters", type=int, default=config.N_ITERS)
    ap.add_argument("--steps", type=int, default=config.N_STEPS)
    ap.add_argument("--lr", type=float, default=config.LR)
    ap.add_argument("--space", default="ee", choices=("ee", "joint"))
    ap.add_argument("--scene-b-scale", type=float, default=1.0)
    a = ap.parse_args()
    main(seed=a.seed, n_iters=a.n_iters, n_steps=a.steps, lr=a.lr,
         space=a.space, scene_b_scale=a.scene_b_scale)
