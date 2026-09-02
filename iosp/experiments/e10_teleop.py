"""E10 -- can the sequential planner be recovered from HUMAN demonstrations?

Question
--------
Every iosp result so far fits a demonstration this model generated, at a known
`theta_star`.  That makes "did recovery work" exact, and it makes the loss have
a zero.  Both properties are artefacts of self-demonstration.  E10 removes them:
the demonstrations are teleoperated pick-and-place episodes recorded through a
GELLO leader into the MuJoCo scene in `sim_teleop`, and there is no vector of
weights that generated them.

So the claim under test is NOT "the weights are recovered" -- that question is
unanswerable here -- but the behavioural one, which is the claim worth making:
FITTED ON SOME EPISODES, DOES THE COMPOSED PLANNER REPRODUCE HELD-OUT ONES?
Three numbers decide it, all printed by `iosp.fit.procedure._report`:

  * `fit RMSE` at `init (u=0)` vs at the wide fit -- did fitting do anything;
  * `gen RMSE` at the fit -- does it carry to episodes the loss never saw;
  * `r` of `K` -- how many cost directions ten human demonstrations resolve.

The floor is not zero
---------------------
Model misspecification is now a real term: whatever the demonstrator did that a
four-phase IK-seeded trajopt cannot express is an irreducible residual.  A
`fit RMSE` that stalls well above 0 is therefore the EXPECTED outcome, and the
result is in the drop from init and in the fit/gen gap -- not in the absolute
value.  Read it against E3, which is the same procedure on synthetic demos.

Same scenes as the demonstrations
---------------------------------
Each episode's own randomisation record supplies its scene: the cube where it
actually was, the bucket where it actually was, `q_start` the demo's own first
waypoint.  Nothing is rolled out on `iosp.config`'s canonical task.  The robot
is the FR3 the episodes were recorded on (`iosp.model.fr3`), NOT the Panda every
earlier iosp experiment used -- pyroffi's FR3 FK was checked against the
recorded MuJoCo `ee_pos` and agrees to 1.3e-4 m.

Two things this data cannot answer, by construction
---------------------------------------------------
  * `clearance` is UNIDENTIFIABLE: the recording scene has no obstacle, and the
    exporter emits a constant placeholder in every episode.  Expect it in the
    null space; if it appears in `U_r`, distrust the spectrum.
  * `upright` is weakly excited: a human doing pick-and-place keeps the gripper
    roughly down throughout, so the feature barely varies over the demo set.

Two modes, and why the default is multistart
--------------------------------------------
`--mode multistart` (default) is `iosp.fit.multistart`: B IK branches x S cost
seeds, every candidate a complete independent bilevel fit on its own FIXED
branch, with ONE selection at the very end on TRAINING loss.  That is the right
default here for the reason the module documents -- a hard selection anywhere
inside the differentiated forward map makes x*(theta) discontinuous and breaks
the implicit adjoint -- and because on this problem a single start has been
measured to land in a basin rather than at the floor.  It also runs the full
THREE-stage forward model (segments, then a global refine), which is the planner
the demonstrator is being compared against.

`--mode procedure` is E3's five-stage identifiable refit, which answers the
other question: how many cost directions do ten human demonstrations actually
resolve (`r` of `K`)?  It is single-start and two-stage.  `--mode both` runs
them in that order.

Reproduce
---------
    CUDA_VISIBLE_DEVICES=<idx> XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        python -m iosp.experiments.e10_teleop --out iosp/data/e10_teleop.json

    --demo-dir DIR   episode directories (default: ../sim_teleop/data/demos)
    --n-fit N        first N episodes fit, the rest held out (default: 3/4)
    --n-branches/--n-starts   the multistart candidate grid (default 4 x 3)
    --mode           multistart (default) | procedure | both

Cold-compile is the same composed chain as E3 and just as slow.  The batch is
one solve over (candidates x episodes), so episode count costs solve time, not
compilations -- but it is a 12x10 = 120-row batch at the defaults against the
synthetic study's 12x2.
"""

import argparse
import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from iosp import config

config.enable_compilation_cache()

from iosp.fit.procedure import _report, run_procedure
from iosp.fit.teleop import DEFAULT_DEMO_DIR, build_teleop
from iosp.model import pickplace as pp


def run_multistart(demo_dir, n_fit, seed, n_iters, n_steps, lr,
                   n_branches, n_starts, chunk=4, out=None):
    """The default mode: B branches x S starts, one selection at the end.

    `chunk` is a memory knob only -- see `iosp.fit.multistart.run`.  It defaults
    to 4 candidates at a time because the whole 12x10 batch does not fit: the
    refine stage's collision Jacobian is dense in (candidates x episodes) and
    XLA's autotuner OOMs on a 24 GiB card trying to profile it."""
    from iosp.fit import multistart as ms

    built = ms.build_from_demos(demo_dir=demo_dir, n_fit=n_fit, seed=seed,
                                n_iters=n_iters, n_branches=n_branches)
    for i, name in enumerate(built["episodes"]):
        print(f"    {'fit ' if i < built['n_fit'] else 'held'}  {name}", flush=True)

    res = ms.run(seed=seed, n_iters=n_iters, n_branches=n_branches,
                 n_starts=n_starts, n_steps=n_steps, lr=lr, built=built,
                 chunk=chunk)
    ms.report(res)

    w = res["winner"]
    u_w = jnp.asarray(res["u"][w])[None]
    refs_w = built["refs"][w // res["S"]][None]

    # The winner scored on the EE criterion too, from the SAME rollout -- so a
    # joint-space fit is still comparable with every EE-scored iosp result.
    ee = built["batched_paths"](u_w, refs_w, "ee")[0]
    d_ee = built["demo_ee"]
    ee_rmse = lambda idx: float(jnp.sqrt(jnp.mean(
        jnp.sum((ee[idx] - d_ee[idx]) ** 2, -1))))
    fit_ee, gen_ee = ee_rmse(built["fit_idx"]), ee_rmse(built["gen_idx"])
    print(f"\nwinner in EE space:  fit {fit_ee:.4f} m   held-out {gen_ee:.4f} m")
    print("  (the release-row lateral offset alone floors this at ~0.015 m; "
          "see iosp.fit.teleop)")

    z_w = np.asarray(built["P"]) + np.asarray(built["S"]) * res["u"][w]
    theta = np.concatenate([z_w[:pp.K_IK],
                            np.asarray(jax.nn.softmax(jnp.asarray(
                                z_w[pp.K_IK:pp.K_IK + pp.K_TRAJOPT]))),
                            np.asarray(jax.nn.softmax(jnp.asarray(
                                z_w[pp.K_IK + pp.K_TRAJOPT:])))])
    print("\nwinning cost:")
    for name, val in zip(built["names"], theta):
        print(f"    {name:22s} {val: .4f}")

    payload = dict(
        mode="multistart", episodes=built["episodes"], n_fit=built["n_fit"],
        names=built["names"], theta_winner=theta.tolist(),
        u_winner=res["u"][w].tolist(), winner=int(w),
        standoff_prior=np.asarray(built["standoff_prior"]).tolist(),
        train_rmse=np.sqrt(res["train"]).tolist(),
        held_rmse=np.sqrt(res["held"]).tolist(),
        loss_history=res["losses"].tolist(),
        fit_ee_rmse=fit_ee, gen_ee_rmse=gen_ee,
        n_branches=n_branches, n_starts=n_starts, n_steps=n_steps, lr=lr,
        chunk=chunk,
        n_iters=n_iters, seed=seed, demo_dir=str(demo_dir), space="joint")
    _write(out, payload)
    return res


def run_procedure_mode(demo_dir, n_fit, seed, n_iters, n_steps, lr, space,
                       n_restarts=1, out=None):
    """E3's five-stage identifiable refit, on the same demonstrations."""
    built = build_teleop(demo_dir=demo_dir, n_fit=n_fit, seed=seed,
                         n_iters=n_iters, n_restarts=n_restarts, space=space)
    print(f"E10: {len(built['episodes'])} teleop episodes, "
          f"{built['n_fit']} fit / {len(built['gen_idx'])} held out, "
          f"loss in {space} space", flush=True)
    for i, name in enumerate(built["episodes"]):
        print(f"    {'fit ' if i < built['n_fit'] else 'held'}  {name}", flush=True)

    res = run_procedure(built, "teleop (path A, human demos)",
                        n_steps=n_steps, lr=lr)
    _report(res)

    # The EE criterion, from the same rollouts -- reported alongside a
    # joint-space fit so E10 is comparable with every EE-scored iosp result.
    u0 = jnp.zeros(built["K"], dtype=jnp.float32)
    print(f"\n{'':22s} {'fit EE RMSE':>13s} {'gen EE RMSE':>13s}   [m]")
    for label, u in (("init (u=0)", u0), ("wide fit (all K)", res["u_wide"]),
                     ("refit on U_r", res["u_refit"])):
        a, b = built["ee_rmse_a"](u), built["ee_rmse_b"](u)
        res[f"ee_fit_rmse_{label.split()[0]}"] = a
        res[f"ee_gen_rmse_{label.split()[0]}"] = b
        print(f"{label:22s} {a:13.4f} {b:13.4f}")

    print("\nfitted weights:")
    for name, val in zip(built["names"], res["theta_wide"]):
        print(f"    {name:22s} {val: .4f}")

    payload = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
               for k, v in res.items()}
    payload.update(mode="procedure", episodes=built["episodes"],
                   n_fit=built["n_fit"], space=space, names=built["names"],
                   standoff_prior=np.asarray(built["standoff_prior"]).tolist(),
                   demo_dir=str(demo_dir), n_steps=n_steps, lr=lr,
                   n_iters=n_iters, seed=seed)
    _write(out, payload)
    return res


def _write(out, payload):
    if not out:
        return
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out).write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nwrote {out}")


def main(demo_dir=DEFAULT_DEMO_DIR, n_fit=None, seed=0, n_iters=config.N_ITERS,
         n_steps=config.N_STEPS, lr=config.LR, space="joint", n_restarts=1,
         mode="multistart", n_branches=4, n_starts=3, chunk=4, out=None):
    stem = (out or "").rsplit(".", 1)[0]
    res = {}
    if mode in ("multistart", "both"):
        res["multistart"] = run_multistart(
            demo_dir, n_fit, seed, n_iters, n_steps, lr, n_branches, n_starts,
            chunk=chunk, out=(f"{stem}_multistart.json" if mode == "both" else out))
    if mode in ("procedure", "both"):
        res["procedure"] = run_procedure_mode(
            demo_dir, n_fit, seed, n_iters, n_steps, lr, space, n_restarts,
            out=(f"{stem}_procedure.json" if mode == "both" else out))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--demo-dir", default=str(DEFAULT_DEMO_DIR))
    ap.add_argument("--n-fit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-iters", type=int, default=config.N_ITERS)
    ap.add_argument("--steps", type=int, default=config.N_STEPS)
    ap.add_argument("--lr", type=float, default=config.LR)
    ap.add_argument("--n-restarts", type=int, default=1)
    ap.add_argument("--space", default="joint", choices=("ee", "joint"))
    ap.add_argument("--mode", default="multistart",
                    choices=("multistart", "procedure", "both"))
    ap.add_argument("--n-branches", type=int, default=4)
    ap.add_argument("--n-starts", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=4,
                    help="candidates evaluated at once (memory only; exact)")
    ap.add_argument("--out", default="iosp/data/e10_teleop.json")
    a = ap.parse_args()
    main(demo_dir=a.demo_dir, n_fit=a.n_fit, seed=a.seed, n_iters=a.n_iters,
         n_steps=a.steps, lr=a.lr, space=a.space, n_restarts=a.n_restarts,
         mode=a.mode, n_branches=a.n_branches, n_starts=a.n_starts,
         chunk=a.chunk, out=a.out)
