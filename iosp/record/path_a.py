"""Reconstruction data for the 7-DOF Panda: the trajectory converging to the demo.

This records the NAMED-cost-library fit ("path A" in
`iosp.fit.parametric`) -- the known-cost-function reconstruction.
No RKHS: the unknown-cost path is out of scope.

What it saves, and why it is not what the earlier animation saved
-----------------------------------------------------------------
`iosp.viz.fit_animation` recorded cost WEIGHTS against ground truth.  On this
problem that panel is close to uninterpretable: the Gram spectrum is
rank-deficient, so the fit is expected to reproduce behaviour while drifting
along null directions, and the bars miss their targets even on a good fit
(measured: the two IK standoffs land on ground truth, the seven trajopt
weights scramble).  This records BEHAVIOUR instead -- the end-effector path at
every outer step on both the fit scene and the held-out scene, plus the
reconstruction loss -- which `iosp.viz.behavior` animates as a trajectory
walking onto the demonstration with the loss curve beside it.

`build_parametric` supplies `paths(u) -> (2, T, 3)` (row 0 = fit scene A, row
1 = held-out scene B) and the theta*-rollout demos, so nothing about the
forward model is redefined here.  Note this is the TWO-stage model that every
recorded path-A result uses (IK -> per-segment trajopt); the three-stage
refine pass is `iosp.experiments.e4_three_stage` and is not yet validated,
so it is deliberately not the thing being drawn for a figure.

Recording starts from u = 0 (uniform weights), not from a multistart winner:
an animation started at an already-good iterate shows no motion, which is
exactly what made the first path-A GIF look dead.
"""

import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from iosp.fit import parametric as s3
from iosp.config import OBS_CENTER, OBS_RADIUS, PICK_POS, PLACE_POS
from iosp import config


def record(n_steps=40, lr=config.LR, n_iters=60, seed=0, n_restarts=1,
           out="iosp/data/viz/pathA_behavior.npz"):
    t0 = time.perf_counter()
    built = s3.build_parametric(seed=seed, n_iters=n_iters, n_restarts=n_restarts)
    K = built["K"]
    print(f"[build] {time.perf_counter()-t0:.0f}s  K={K}  n_restarts={n_restarts}",
          flush=True)

    # Both the rollout and the gradient come from `build_parametric`, so the
    # plotted paths and the steps that produced them are the same solver.
    # Rebuilding `paths` locally (the previous version) would have paired a
    # restart-enabled rollout with a no-restart gradient once n_restarts > 1.
    paths_j, demo, gf = built["paths_fn"], built["demo_paths"], built["gf"]

    opt = optax.adamw(lr, weight_decay=0.0)
    u = jnp.zeros(K, dtype=jnp.float32)
    st = opt.init(u)
    th_h, loss_h, rmse_a_h, rmse_b_h, path_h = [], [], [], [], []
    for t in range(n_steps + 1):
        val, g = gf(u)
        P = np.asarray(paths_j(u))
        th_h.append(built["theta_of"](u))
        loss_h.append(float(val))
        rmse_a_h.append(float(np.sqrt(np.mean(np.sum((P[0] - np.asarray(demo)[0]) ** 2, -1)))))
        rmse_b_h.append(float(np.sqrt(np.mean(np.sum((P[1] - np.asarray(demo)[1]) ** 2, -1)))))
        path_h.append(P)
        print(f"[record] step {t:3d}/{n_steps}  loss={float(val):.6f}  "
              f"rmse_fit={rmse_a_h[-1]:.5f}  rmse_heldout={rmse_b_h[-1]:.5f}",
              flush=True)
        if t == n_steps:
            break
        upd, st = opt.update(g, st, u)
        u = optax.apply_updates(u, upd)

    # Obstacle and skeleton anchors, for the geometric panel.  Scene B's
    # offsets come from `study3`'s own constants so the two panels are drawn
    # against the scenes they were actually solved in.
    # (x, y, radius) -- the renderer projects the 3D EE path onto x-y, and the
    # x-y projection of a sphere is a disc of the same radius, so this is the
    # correct silhouette rather than an approximation.  Both scenes share the
    # obstacle: `study3.scene_b` offsets q_start/pick/place, not the obstacle.
    _o = np.array([float(OBS_CENTER[0]), float(OBS_CENTER[1]), float(OBS_RADIUS[0])])
    obs = np.stack([_o, _o])[:, None, :]  # (2, 1, 3)
    way = np.stack([
        np.stack([np.asarray(PICK_POS), np.asarray(PLACE_POS)]),
        np.stack([np.asarray(PICK_POS) + np.asarray(config.SCENE_B_PICK_OFFSET),
                  np.asarray(PLACE_POS) + np.asarray(config.SCENE_B_PLACE_OFFSET)]),
    ])

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(
        out,
        path_hist=np.stack(path_h),            # (F, 2, T, 3)
        demo=np.asarray(demo),                 # (2, T, 3)
        rmse_hist=np.asarray(rmse_b_h),        # held-out: the criterion
        rmse_fit_hist=np.asarray(rmse_a_h),
        loss_hist=np.asarray(loss_h),
        theta_hist=np.stack(th_h),
        obstacles=obs,                         # (2, 1, 3) x, y, radius
        waypoints=way,                         # (2, 2, 3)
        names=np.array(built["names"], dtype=object),
        theta_star=np.asarray(built["theta_star"]),
        label="7-DOF Panda, named cost library (path A)")
    print(f"[record] wrote {out}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=config.LR)
    ap.add_argument("--n-iters", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-restarts", type=int, default=1,
                    help="inner-solve multistart; >1 suppresses basin flips")
    ap.add_argument("--out", default="iosp/data/viz/pathA_behavior.npz")
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    npz = record(n_steps=a.steps, lr=a.lr, n_iters=a.n_iters, seed=a.seed,
                 n_restarts=a.n_restarts, out=a.out)
    if not a.no_render:
        from iosp.viz import behavior as viz_behavior
        viz_behavior.render(npz, npz.replace(".npz", ".gif"), max_ctx=2)
