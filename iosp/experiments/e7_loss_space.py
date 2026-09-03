"""E7 — Joint-space vs EE-space outer loss comparison.

Both runs are scored on BOTH criteria every step.  Also computes the sensitivity
spectrum at the fitted point to check whether joint coordinates improve
identifiability.
"""

import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ioc import identifiability as ident
from iosp.fit import parametric as s3
from iosp import config


def rmse(P, D, i):
    return float(np.sqrt(np.mean(np.sum((np.asarray(P)[i] - np.asarray(D)[i]) ** 2, -1))))


def run(space="joint", n_steps=40, lr=config.LR, n_iters=60, seed=0, n_restarts=1,
        out=None, spectrum=True):
    out = out or f"iosp/data/viz/loss_space_{space}.npz"
    t0 = time.perf_counter()
    b = s3.build_parametric(seed=seed, n_iters=n_iters, n_restarts=n_restarts,
                            space=space)
    print(f"[build] {time.perf_counter()-t0:.0f}s  K={b['K']}  space={space}",
          flush=True)

    gf, paths_j, demo = b["gf"], b["paths_fn"], b["demo_paths"]
    ee_j, ee_demo = b["ee_paths_fn"], b["ee_demo_paths"]

    opt = optax.adamw(lr, weight_decay=0.0)
    u = jnp.zeros(b["K"], dtype=jnp.float32)
    st = opt.init(u)
    hist = {k: [] for k in ("loss", "fit", "held", "ee_fit", "ee_held")}
    u_hist = []
    for t in range(n_steps + 1):
        val, g = gf(u)
        P, E = paths_j(u), ee_j(u)
        hist["loss"].append(float(val))
        hist["fit"].append(rmse(P, demo, 0));   hist["held"].append(rmse(P, demo, 1))
        hist["ee_fit"].append(rmse(E, ee_demo, 0))
        hist["ee_held"].append(rmse(E, ee_demo, 1))
        u_hist.append(np.asarray(u))
        print(f"[{space}] step {t:3d}/{n_steps}  loss={float(val):.6f}  "
              f"fit={hist['fit'][-1]:.5f} held={hist['held'][-1]:.5f}  |  "
              f"ee_fit={hist['ee_fit'][-1]:.5f} ee_held={hist['ee_held'][-1]:.5f}",
              flush=True)
        if t == n_steps:
            break
        upd, st = opt.update(g, st, u)
        u = optax.apply_updates(u, upd)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(out, space=space, u_hist=np.stack(u_hist),
                        theta_star=np.asarray(b["theta_star"]),
                        names=np.array(b["names"], dtype=object),
                        **{k: np.asarray(v) for k, v in hist.items()})
    print(f"[{space}] wrote {out}", flush=True)

    # Best-by-training-loss iterate, not the last one: the EE baseline's held-out
    # RMSE peaked at step 35 and rose after, so "final" and "best" are different
    # numbers and only one of them is a fair summary.
    i_best = int(np.argmin(hist["loss"]))
    print(f"\n[{space}] SUMMARY over {n_steps} steps"
          f"\n  final : loss={hist['loss'][-1]:.6f} fit={hist['fit'][-1]:.5f} "
          f"held={hist['held'][-1]:.5f} ee_fit={hist['ee_fit'][-1]:.5f} "
          f"ee_held={hist['ee_held'][-1]:.5f}"
          f"\n  best-train-loss iterate (step {i_best}): "
          f"fit={hist['fit'][i_best]:.5f} held={hist['held'][i_best]:.5f} "
          f"ee_fit={hist['ee_fit'][i_best]:.5f} ee_held={hist['ee_held'][i_best]:.5f}"
          f"\n  min ee_held over trajectory = {min(hist['ee_held']):.5f} "
          f"(step {int(np.argmin(hist['ee_held']))})", flush=True)

    if spectrum:
        t1 = time.perf_counter()
        lam, U = ident.sensitivity_spectrum(b["jac_fn"], u)
        lam = np.asarray(lam)
        # Both rules: `gap` is `select_rank`'s default and the one its docstring
        # argues for, `trace` is what every recorded result used, so reporting
        # only one would make this run incomparable to one of them.
        _, _, r_gap = ident.select_rank(lam, rule="gap")
        _, _, r_tr = ident.select_rank(lam, config.TRACE_FRAC, rule="trace")
        print(f"\n[{space}] jac+svd {time.perf_counter()-t1:.0f}s"
              f"\n  eigenvalues (desc) = {np.array2string(lam, precision=4)}"
              f"\n  r = {r_gap} (gap rule) / {r_tr} (95% trace) of K = {b['K']}",
              flush=True)
        r = r_gap
        np.savez_compressed(out.replace(".npz", "_spectrum.npz"),
                            eigvals=lam, U=np.asarray(U), r=r)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default="joint", choices=("joint", "ee"))
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--n-iters", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-restarts", type=int, default=1)
    ap.add_argument("--no-spectrum", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(space=a.space, n_steps=a.steps, n_iters=a.n_iters, seed=a.seed,
        n_restarts=a.n_restarts, out=a.out, spectrum=not a.no_spectrum)
