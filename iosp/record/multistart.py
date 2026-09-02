"""Reconstruction data for the MULTISTART path-A fit: every candidate's
trajectory, recorded at every outer step, with one selection at the end.

How this differs from `record_pathA_behavior`
---------------------------------------------
That module records a single bilevel fit -- one IK branch, one cost-parameter
start -- and animates it converging.  It is the honest picture of the method
only if the basin it happens to land in is the right one.  `study6_batched_
spasm` measured that this is not a safe assumption: at B=2 branches x S=2
starts the held-out RMSE spread across candidates was 81.5x (0.00773 ..
0.62977), and it was the BRANCH, not the cost start, that separated them
(0.0077/0.0327 on branch 0 against 0.362/0.630 on branch 1).

So this records all C = B*S candidates as one batched program, exactly as
`study6.build` runs them: each candidate's branch is fixed for the whole fit,
nothing is selected inside the differentiated map, and the single argmin --
over TRAINING loss, never held-out -- happens after every candidate has
converged.

What it saves
-------------
Every candidate's EE path at every step (`cand_hist`), so the animation shows
the population converging and the losers staying lost, plus `path_hist` set to
the WINNER's path so the existing single-fit renderers read the file unchanged.

`--space joint` scores the outer loss in configuration space instead of EE.
Note what that does to the candidate population: the demo is rolled out on
`refs[0]`, so a candidate on any other branch carries an irreducible joint-space
floor.  That is the intended behaviour -- it is what makes the joint loss a
decisive branch classifier rather than an ambiguous one -- but it means the
off-branch candidates are expected to look far worse under `--space joint` than
under `--space ee`, and that is a property of the criterion, not a failure of
the fit.

`--scene-b-scale` pushes the held-out scene further from the fit scene (1.0 is
every recorded result).  Read the `screen_stationarity` output when raising it:
past some scale the pick/place targets leave the reachable set and the held-out
number stops measuring generalization.
"""

import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from iosp import config
config.enable_compilation_cache()

from iosp.fit import parametric as s3
from iosp.fit import multistart as s6
from iosp.config import OBS_CENTER, OBS_RADIUS, PICK_POS, PLACE_POS


def _anchors(scale):
    """Obstacle discs and skeleton waypoints for the two scenes, for the plot.

    Scene B's waypoints must use the SAME scaled offsets the scenes were built
    with, or the figure would draw the fit against a skeleton it was never
    solved for.
    """
    o = np.array([float(OBS_CENTER[0]), float(OBS_CENTER[1]), float(OBS_RADIUS[0])])
    obs = np.stack([o, o])[:, None, :]                       # (2, 1, 3)
    way = np.stack([
        np.stack([np.asarray(PICK_POS), np.asarray(PLACE_POS)]),
        np.stack([np.asarray(PICK_POS) + scale * np.asarray(config.SCENE_B_PICK_OFFSET),
                  np.asarray(PLACE_POS) + scale * np.asarray(config.SCENE_B_PLACE_OFFSET)]),
    ])
    return obs, way


def record(n_steps=40, lr=config.LR, n_iters=60, seed=0, n_branches=3, n_starts=3,
           space="joint", scene_b_scale=1.0,
           out="iosp/data/viz/multistart_behavior.npz"):
    t0 = time.perf_counter()
    built = s6.build(seed=seed, n_iters=n_iters, n_branches=n_branches,
                     scene_b_scale=scene_b_scale)
    bp = built["batched_paths"]
    demo_ee, demo_q = built["demo"], built["demo_joint"]
    demo_loss = demo_q if space == "joint" else demo_ee
    U0, refs_c, B, S = s6.make_candidates(built, n_starts, seed)
    C = U0.shape[0]
    print(f"[build] {time.perf_counter()-t0:.0f}s  B={B} branches x S={S} starts "
          f"= {C} candidates  space={space}  scene_b_scale={scene_b_scale}",
          flush=True)

    def per_cand(U, i):
        """(C,) mean squared error against demo row `i`, in the LOSS space."""
        return jnp.mean(jnp.sum((bp(U, refs_c, space)[:, i] - demo_loss[i]) ** 2, -1), -1)

    # Sum over candidates: each row of U affects only its own term, so ONE
    # gradient call returns every candidate's own gradient exactly.
    gf = jax.jit(jax.value_and_grad(lambda U: jnp.sum(per_cand(U, 0))))
    train_j = jax.jit(lambda U: per_cand(U, 0))
    held_j = jax.jit(lambda U: per_cand(U, 1))
    ee_j = jax.jit(lambda U: bp(U, refs_c, "ee"))
    # Joint paths too: the EE history alone cannot drive a robot viewer (the
    # arm is redundant, so q is not recoverable from the EE path), and
    # re-deriving q post hoc would mean re-running this whole fit.
    q_j = jax.jit(lambda U: bp(U, refs_c, "joint"))

    opt = optax.adamw(lr, weight_decay=0.0)
    U, st = U0, opt.init(U0)
    cand_h, tr_h, he_h, ee_tr_h, ee_he_h = [], [], [], [], []
    q_h, u_h = [], []
    for t in range(n_steps + 1):
        _, g = gf(U)
        E = np.asarray(ee_j(U))                              # (C, M, T, 3)
        cand_h.append(E)
        q_h.append(np.asarray(q_j(U)))                       # (C, M, T, dof)
        u_h.append(np.asarray(U))                            # (C, K)
        tr_h.append(np.sqrt(np.asarray(train_j(U))))
        he_h.append(np.sqrt(np.asarray(held_j(U))))
        # The EE criterion, always, whatever space the loss is in: a joint-space
        # fit still has to be scored on the number the paper reports.
        ee_tr_h.append(np.sqrt(np.mean(np.sum((E[:, 0] - np.asarray(demo_ee)[0]) ** 2, -1), -1)))
        ee_he_h.append(np.sqrt(np.mean(np.sum((E[:, 1] - np.asarray(demo_ee)[1]) ** 2, -1), -1)))
        w = int(np.argmin(tr_h[-1]))
        print(f"[rec] step {t:3d}/{n_steps}  train(min)={tr_h[-1].min():.5f} "
              f"held@trainargmin={he_h[-1][w]:.5f}  |  ee_fit={ee_tr_h[-1][w]:.5f} "
              f"ee_held={ee_he_h[-1][w]:.5f}  spread={he_h[-1].max()/he_h[-1].min():.1f}x",
              flush=True)
        if t == n_steps:
            break
        upd, st = opt.update(g, st, U)
        U = optax.apply_updates(U, upd)

    tr_h, he_h = np.stack(tr_h), np.stack(he_h)
    ee_tr_h, ee_he_h = np.stack(ee_tr_h), np.stack(ee_he_h)
    cand = np.stack(cand_h)                                  # (F, C, M, T, 3)
    qh, uh = np.stack(q_h), np.stack(u_h)                    # (F,C,M,T,dof), (F,C,K)
    winner = int(np.argmin(tr_h[-1]))                        # TRAINING loss, at the end
    leak = int(np.argmin(he_h[-1]))

    obs, way = _anchors(scene_b_scale)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(
        out,
        cand_hist=cand,                                      # (F, C, M, T, 3)
        path_hist=cand[:, winner],                           # winner, for the 1-fit renderers
        demo=np.asarray(demo_ee),                            # (M, T, 3)
        rmse_hist=ee_he_h[:, winner],                        # the reported criterion
        rmse_fit_hist=ee_tr_h[:, winner],
        # The EE criterion, NOT the loss-space one: `viz_behavior` draws
        # sqrt(loss_hist) on the same log axis as `rmse_hist`, so a joint-space
        # train curve there would put two different units on one axis and read
        # as a train/held-out gap that is really a change of coordinates.
        loss_hist=ee_tr_h[:, winner] ** 2,
        loss_space_hist=tr_h[:, winner] ** 2,
        train_hist=tr_h, held_hist=he_h,                     # (F, C), loss space
        ee_train_hist=ee_tr_h, ee_held_hist=ee_he_h,         # (F, C), EE space
        winner=winner, leak_winner=leak, B=B, S=S, space=space,
        scene_b_scale=scene_b_scale,
        obstacles=obs, waypoints=way,
        # For the viser replay (`iosp.viz.multistart_viser --joints`): the
        # configuration behind every EE path, and the cost params that produced
        # it, so a step can be inspected without re-running the fit.
        q_hist=qh, q_path_hist=qh[:, winner], u_hist=uh,
        demo_joint=np.asarray(demo_q), q_start=np.asarray(config.Q_START),
        label=f"7-DOF Panda, path A, {C} candidates ({B} branches x {S} starts), "
              f"{space}-space loss")
    print(f"[rec] wrote {out}", flush=True)

    print(f"\n{'cand':>5} {'branch':>7} {'start':>6} {'train':>9} {'held':>9} "
          f"{'ee_fit':>9} {'ee_held':>9}")
    for i in range(C):
        mark = "  <- winner (train argmin)" if i == winner else ""
        mark += "  [held argmin: NOT used]" if i == leak and i != winner else ""
        print(f"{i:5d} {i//S:7d} {i%S:6d} {tr_h[-1, i]:9.5f} {he_h[-1, i]:9.5f} "
              f"{ee_tr_h[-1, i]:9.5f} {ee_he_h[-1, i]:9.5f}{mark}")
    print(f"\nselected on training loss: candidate {winner} "
          f"(branch {winner//S}, start {winner%S})")
    print(f"  EE held-out RMSE = {ee_he_h[-1, winner]:.5f}")
    if leak != winner:
        print(f"  had we (wrongly) selected on held-out: {he_h[-1, leak]:.5f} "
              f"-- the size of the leakage this guards against")
    print(f"held-out spread across candidates: "
          f"{he_h[-1].max()/he_h[-1].min():.1f}x "
          f"({he_h[-1].min():.5f} .. {he_h[-1].max():.5f})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=config.LR)
    ap.add_argument("--n-iters", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-branches", type=int, default=3)
    ap.add_argument("--n-starts", type=int, default=3)
    ap.add_argument("--space", default="joint", choices=("joint", "ee"))
    ap.add_argument("--scene-b-scale", type=float, default=1.0)
    ap.add_argument("--out", default="iosp/data/viz/multistart_behavior.npz")
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    npz = record(n_steps=a.steps, lr=a.lr, n_iters=a.n_iters, seed=a.seed,
                 n_branches=a.n_branches, n_starts=a.n_starts, space=a.space,
                 scene_b_scale=a.scene_b_scale, out=a.out)
    if not a.no_render:
        from iosp.viz import behavior as viz_behavior
        from iosp.viz import behavior3d as viz_behavior3d
        viz_behavior.render(npz, npz.replace(".npz", ".gif"), max_ctx=2)
        viz_behavior3d.render(npz, npz.replace(".npz", "_3d.gif"), max_ctx=2)
