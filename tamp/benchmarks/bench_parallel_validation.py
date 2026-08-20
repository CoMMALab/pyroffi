"""Parallel motion validation throughput across geometric backends.

The claim this is built to test:

    In parallel TAMP, where many candidate plan skeletons are validated at
    once, pyroffi has the highest motion-validation throughput -- and is the
    only backend that can screen actuator-torque feasibility in the same pass.

Why this design rather than "run PDDLStream and time it"
--------------------------------------------------------
Earlier experiments here varied the *search* (PDDLStream vs cuTAMP) and left
the validator as a null variable, or ran the validator at batch 1, where every
GPU backend is dominated by dispatch overhead and a mature CPU engine wins.
Neither measures the property pyroffi actually has.

So: **the skeletons are fixed**. They are generated once, saved, and replayed
identically to every backend. That removes symbolic search, sampling luck and
plan-length variation from the comparison entirely -- every backend validates
byte-identical work, and the only free variable is how many plans are checked
at once. Any difference is throughput, not luck.

What is measured
----------------
* **Throughput vs batch size.** Plans validated per second as the batch grows.
  This is the whole experiment: serial backends are flat, batched backends
  scale.
* **Agreement.** Every backend's verdict on every plan. Throughput is
  meaningless if the answers differ, and a disagreement is a finding in itself
  (the backends do not share collision geometry exactly -- see below).
* **Torque feasibility.** Of the plans all backends call kinematically valid,
  how many exceed the Franka's actuator limits. This is the axis where a
  kinematic validator cannot answer at all.
* **Best plan found under a fixed time budget.** Checking more candidates per
  second means choosing from a larger pool, so throughput should translate into
  solution quality rather than only speed.

Hypotheses (stated before measuring, so the result can contradict them)
-----------------------------------------------------------------------
1. pybullet's per-plan cost is flat in batch size -- it steps one configuration
   at a time -- so its throughput plateaus almost immediately.
2. cuRobo and pyroffi both scale with batch size and land in the same
   neighbourhood on kinematic checks.
3. Only pyroffi reports torque feasibility. pybullet *can* compute inverse
   dynamics (``calculateInverseDynamics``) but serially and non-differentiably;
   cuRobo bounds velocity/acceleration/jerk, which are kinematic limits, not
   actuator torque. If cuRobo turns out to expose a torque path, hypothesis 3
   narrows and should be restated rather than defended.

A caveat that constrains every number here
------------------------------------------
The backends do **not** share collision geometry exactly. pyroffi and pybullet
run the same spherized Panda URDF; cuRobo uses its own shipped Franka config.
So cross-backend verdicts can legitimately differ, and throughput is not a
comparison of identical computations. The agreement column exists to quantify
that rather than assume it away.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

TAMP_ROOT = Path(__file__).resolve().parents[1]

#: Waypoints per motion segment, matching the PDDLStream streams.
N_WAYPOINTS = 20


# --------------------------------------------------------------------------- #
# Skeleton generation -- run once, shared by every backend
# --------------------------------------------------------------------------- #

def generate_skeletons(n_plans, n_objects=3, seed=0, out=None):
    """Generate candidate plan skeletons and freeze them to disk.

    A skeleton is the motion content of a pick-and-place plan: for each object,
    a transit to its grasp and a transfer to its placement. Grasps and
    placements are sampled the way PDDLStream's streams would sample them, so
    the workload is realistic rather than random joint noise -- but they are
    *not* filtered for validity, because validity is exactly what the backends
    are being asked to decide.

    Returns a dict with:
        paths    [B, S, T, 7]  joint paths, S segments per plan
        holding  [B, S]        bool, whether a payload is carried on that
                               segment (transfers carry, transits do not)
    """
    from spasm.tamp import _setup  # noqa: F401
    from spasm.tamp import geometry as g
    from spasm.tamp.problems import make_rearrange_world

    rng = np.random.default_rng(seed)
    world = make_rearrange_world(n_objects, seed=seed)
    q_home = np.asarray(world.conf0)[:7]
    region = world.regions["goal"]

    paths, holding = [], []
    for _ in range(n_plans):
        segs, hold = [], []
        q_cur = q_home
        for name in world.initial_poses:
            pick = np.asarray(world.initial_poses[name], dtype=float)
            place = np.array([
                rng.uniform(region["cx"] - region["hx"], region["cx"] + region["hx"]),
                rng.uniform(region["cy"] - region["hy"], region["cy"] + region["hy"]),
                region["z"] + world.block_half_height(name),
                rng.uniform(-np.pi, np.pi),
            ])
            q_pick, _ = g.ik_topdown(pick, grasp_yaw=float(pick[3]))
            q_place, _ = g.ik_topdown(place, grasp_yaw=float(place[3]))

            segs.append(g.interpolate(q_cur, q_pick, N_WAYPOINTS))   # transit
            hold.append(False)
            segs.append(g.interpolate(q_pick, q_place, N_WAYPOINTS))  # transfer
            hold.append(True)
            q_cur = q_place

        paths.append(np.stack(segs))
        holding.append(np.asarray(hold))

    data = {"paths": np.stack(paths).astype(np.float32),
            "holding": np.stack(holding)}
    if out:
        np.savez_compressed(out, **data)
    return data


# --------------------------------------------------------------------------- #
# Backends: each takes [B, S, T, 7] and returns [B] validity
# --------------------------------------------------------------------------- #

def validate_pyroffi(paths, batched=True):
    """pyroffi: one fused jitted call over the whole batch.

    The batched path goes through ``geometry.arm_paths_valid``, which is the
    public batched entry point and routes to the fused FK+collision kernel.
    Calling ``_path_validator()`` under ``vmap`` instead -- as this harness once
    did -- benchmarks the per-path predicate, which re-enters the collision
    model per waypoint and materialises an intermediate sphere tensor. cuRobo is
    measured through its own batched entry point, so pyroffi must be too.
    """
    import jax
    import jax.numpy as jnp

    from spasm.tamp import _setup  # noqa: F401
    from spasm.tamp import geometry as g

    B, S = paths.shape[:2]
    flat = jnp.asarray(paths.reshape(B * S, *paths.shape[2:]))

    if not batched:
        # Serial reference, to isolate how much of the win is batching.
        validator = g._path_validator()
        out = np.array([bool(validator(p)) for p in flat])
    else:
        out = np.asarray(g.arm_paths_valid(flat))
    return out.reshape(B, S).all(axis=1)


def validate_pybullet(paths):
    """pybullet: inherently serial -- one configuration reset at a time."""
    from spasm.tamp import pybullet_backend as pbb

    B, S = paths.shape[:2]
    out = np.empty((B, S), dtype=bool)
    for b in range(B):
        for s in range(S):
            out[b, s] = pbb.arm_path_valid(paths[b, s])
    return out.all(axis=1)


def validate_curobo(paths):
    """cuRobo: batched over the whole set of segments in one call.

    Runs in cuRobo's own environment -- it cannot import ``spasm.tamp``, whose
    package import pulls JAX and enforces numpy>=2. That is why the skeletons
    are exchanged as a plain ``.npz`` rather than regenerated per backend: the
    file is just arrays, readable anywhere, and it guarantees every backend
    validates byte-identical work.
    """
    import sys
    sys.path.insert(0, str(TAMP_ROOT))
    from backends import curobo_backend as cb

    B, S = paths.shape[:2]
    flat = paths.reshape(B * S, *paths.shape[2:])
    return cb.arm_paths_valid(flat).reshape(B, S).all(axis=1)


def torque_feasible(paths, holding, dt=0.15):
    """Fraction of segments within the Franka's actuator limits (pyroffi only).

    This is the column no kinematic validator can fill. Batched the same way as
    the validity check, so it is a like-for-like throughput measurement rather
    than an expensive add-on.
    """
    import jax
    import jax.numpy as jnp

    from spasm.tamp import _setup  # noqa: F401
    from spasm.tamp import motion as M
    from spasm.extensions import dynamics as dyn

    B, S = paths.shape[:2]
    flat = jnp.asarray(paths.reshape(B * S, *paths.shape[2:]))
    mass = jnp.where(jnp.asarray(holding.reshape(-1)), M.CUBE_MASS, 0.0)

    def peak_tau(q_traj, m):
        qd, qdd = dyn._finite_diff_qd_qdd(q_traj, dt)
        tau = jax.vmap(dyn.inverse_dynamics)(q_traj, qd, qdd)
        # Payload is folded in as an end-effector wrench; see spasm.tamp.motion.
        tau = tau + jax.vmap(M._payload_torque_state, in_axes=(0, 0, 0, None))(
            q_traj, qd, qdd, 1.0) * m
        return jnp.max(jnp.abs(tau) / dyn.TORQUE_LIMITS[None, :])

    util = np.asarray(jax.jit(jax.vmap(peak_tau))(flat, mass))
    return util.reshape(B, S)


# --------------------------------------------------------------------------- #

def bench(fn, *args, reps=3):
    fn(*args)                                   # warm compile / caches
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn(*args)
    return (time.perf_counter() - t0) / reps, out


BACKENDS = {
    "pyroffi": lambda p: validate_pyroffi(p, True),
    "pyroffi-serial": lambda p: validate_pyroffi(p, False),
    "pybullet": validate_pybullet,
    "curobo": validate_curobo,
}

CSV_HEADER = ("backend,batch,rep,n_plans,n_segments,wall_s,plans_per_s,"
              "segments_per_s,n_valid")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", required=True, choices=sorted(BACKENDS),
                    help="one backend per process; each needs its own env")
    ap.add_argument("--batch-sizes", type=int, nargs="+",
                    default=[1, 8, 32, 128, 512])
    ap.add_argument("--reps", type=int, default=3,
                    help="timed repetitions per batch size; every rep is a CSV "
                         "row so the aggregator can compute spread rather than "
                         "trusting a single measurement")
    ap.add_argument("--skeletons", required=True,
                    help="npz of skeletons, shared across backends. Generated "
                         "by --generate; never regenerated per backend, since "
                         "identical work is the whole basis of the comparison")
    ap.add_argument("--generate", type=int, default=0,
                    help="generate this many skeletons and exit")
    ap.add_argument("--objects", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default=None, help="append rows here")
    ap.add_argument("--torque", action="store_true",
                    help="also report torque feasibility (pyroffi env only)")
    args = ap.parse_args()

    if args.generate:
        print(f"generating {args.generate} skeletons -> {args.skeletons}")
        generate_skeletons(args.generate, args.objects, args.seed,
                           out=args.skeletons)
        return

    d = np.load(args.skeletons)
    paths_all, hold_all = d["paths"], d["holding"]
    S = paths_all.shape[1]

    rows = []
    fn = BACKENDS[args.backend]
    for B in args.batch_sizes:
        if B > paths_all.shape[0]:
            continue
        p = paths_all[:B]
        fn(p)                                    # warm compile / caches
        for rep in range(args.reps):
            t0 = time.perf_counter()
            out = fn(p)
            dt = time.perf_counter() - t0
            rows.append((args.backend, B, rep, B, B * S, dt,
                         B / dt, B * S / dt, int(out.sum())))
            print(f"{args.backend:16s} B={B:<5} rep={rep} {dt:.4f}s "
                  f"{B/dt:9.1f} plans/s  valid={int(out.sum())}/{B}", flush=True)

    if args.csv:
        path = Path(args.csv)
        new = not path.exists()
        with path.open("a") as f:
            if new:
                f.write(CSV_HEADER + "\n")
            for r in rows:
                f.write(",".join(str(x) for x in r) + "\n")

    if args.torque:
        n = min(128, paths_all.shape[0])
        util = torque_feasible(paths_all[:n], hold_all[:n])
        feas = (util <= 1.0).all(axis=1)
        print(f"\ntorque: {int(feas.sum())}/{len(feas)} plans within actuator "
              f"limits; median peak {np.median(util):.2f}x, "
              f"worst {np.max(util):.2f}x")


if __name__ == "__main__":
    main()
