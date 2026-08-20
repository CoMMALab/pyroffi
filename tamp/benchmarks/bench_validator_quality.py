"""Motion-validator QUALITY: how often does a backend accept an invalid path?

The throughput benchmark (``bench_parallel_validation.py``) asks how fast each
backend answers. This asks whether the answer is right, which is the half that
decides whether a TAMP plan is executable.

Ground truth
------------
Every backend here validates a path by sampling it at its waypoints and
checking each one. That misses collisions that happen *between* waypoints --
the arm can sweep through an obstacle and come out the other side with all 20
sampled configurations clear. This is the dominant failure mode of discrete
validation and it is a pure false ACCEPT: it never rejects a good path, it
silently passes a bad one.

So the reference for each backend is *that same backend* run on a densely
resampled path (``--dense`` times more waypoints). Using each backend as its own
reference is deliberate: it isolates discretization error from representation
differences, and keeps the comparison fair to cuRobo, whose 61-sphere +
``self_collision_buffer`` model is not the 59-sphere model pyroffi and pybullet
share. Cross-backend disagreement is reported separately, and is a different
quantity -- it says the backends model different robots, not that one is wrong.

The dense verdict is a strict refinement: the resampled path contains the
original waypoints, so anything the native check rejects the dense check also
rejects. False rejects are therefore zero by construction, and the only error
this can find is a false accept.

Iso-time comparison
-------------------
Resolution is the fix for tunneling, and resolution costs throughput, so the
honest question is not "who is more accurate at 20 waypoints" but "who is more
accurate per unit of wall-clock". ``--iso-time`` sweeps waypoint density and
reports the residual false-accept rate each backend reaches inside the same
time budget.

Usage::

    python benchmarks/bench_validator_quality.py --backend pyroffi \\
        --skeletons benchmarks/results/skeletons_3obj_seed0.npz --dense 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

TAMP_ROOT = Path(__file__).resolve().parents[1]


def generate_stress(n_paths, seed=0, T=20, out=None):
    """Long joint-space sweeps with clear endpoints -- the tunneling stress set.

    The rearrangement skeletons are the wrong instrument for measuring validator
    quality: they are short, nearly straight connections in a scene whose only
    obstacles are the robot itself and the table, so almost nothing is invalid
    at any resolution (measured: 4 bad segments in 3072, all caught natively).
    A validator can be arbitrarily sloppy and still score perfectly there.

    This instead samples pairs of *valid* configurations across the whole joint
    range and connects them by a straight joint-space line. Both endpoints pass
    by construction, so every rejection comes from the interior -- exactly the
    case discrete waypoint sampling can miss. Long sweeps through the workspace
    also make large excursions between consecutive waypoints, which is what
    tunneling needs.

    Endpoint validity is decided by pyroffi, so the set is not tuned to any
    backend's idea of what is in collision -- it is only used to guarantee the
    endpoints are not the reason a path fails.
    """
    import jax.numpy as jnp
    from spasm.tamp import _setup  # noqa: F401
    from spasm.tamp import geometry as g
    from spasm import backend

    rng = np.random.default_rng(seed)
    lo, hi = (np.asarray(x) for x in backend.get_joint_limits())

    # Oversample, keep the configurations that are collision-free on their own.
    pool = []
    while len(pool) < 2 * n_paths:
        cand = rng.uniform(lo, hi, size=(8 * n_paths, 7)).astype(np.float32)
        ok = np.asarray(g.arm_paths_valid(jnp.asarray(cand[:, None, :])))
        pool.extend(cand[ok])
    pool = np.asarray(pool[:2 * n_paths])

    a, b = pool[:n_paths], pool[n_paths:]
    w = np.linspace(0.0, 1.0, T)[None, :, None]
    paths = (a[:, None, :] * (1.0 - w) + b[:, None, :] * w).astype(np.float32)

    data = {"paths": paths[:, None, :, :],     # [N, 1, T, 7], one segment/plan
            "holding": np.zeros((n_paths, 1), dtype=bool)}
    if out:
        np.savez_compressed(out, **data)
    return data


def resample(paths, factor):
    """``[N, T, 7]`` -> ``[N, (T-1)*factor + 1, 7]`` by linear interpolation.

    The original waypoints are preserved exactly (they are the segment
    endpoints), which is what makes the dense verdict a strict refinement of
    the native one.
    """
    paths = np.asarray(paths)
    if factor <= 1:
        return paths
    N, T, D = paths.shape
    a = paths[:, :-1, :]                       # [N, T-1, D]
    b = paths[:, 1:, :]
    w = np.linspace(0.0, 1.0, factor, endpoint=False)[None, None, :, None]
    seg = a[:, :, None, :] * (1.0 - w) + b[:, :, None, :] * w   # [N,T-1,factor,D]
    out = seg.reshape(N, (T - 1) * factor, D)
    return np.concatenate([out, paths[:, -1:, :]], axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #
# Backends: [N, T, 7] -> [N] bool, one call, batched
# --------------------------------------------------------------------------- #

def _pyroffi_validate(paths):
    import jax.numpy as jnp
    from spasm.tamp import _setup  # noqa: F401
    from spasm.tamp import geometry as g
    return np.asarray(g.arm_paths_valid(jnp.asarray(paths)))


def _curobo_validate(paths):
    import sys
    sys.path.insert(0, str(TAMP_ROOT))
    from backends import curobo_backend as cb
    return np.asarray(cb.arm_paths_valid(paths))


def _pybullet_validate(paths):
    from spasm.tamp import pybullet_backend as pbb
    return np.array([pbb.arm_path_valid(p) for p in paths])


VALIDATORS = {
    "pyroffi": _pyroffi_validate,
    "curobo": _curobo_validate,
    "pybullet": _pybullet_validate,
}


def timed(fn, paths, reps=3):
    fn(paths)                                  # warm compile / caches
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn(paths)
    return (time.perf_counter() - t0) / reps, out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", required=True, choices=sorted(VALIDATORS))
    ap.add_argument("--skeletons", required=True)
    ap.add_argument("--generate-stress", type=int, default=0,
                    help="build the tunneling stress set here and exit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emit-truth", default=None,
                    help="compute mesh-level ground truth here and exit")
    ap.add_argument("--truth-file", default=None,
                    help="compare against mesh truth instead of self-dense; "
                         "the reference is shared across backends, so this "
                         "measures geometric fidelity rather than resolution")
    ap.add_argument("--plans", type=int, default=512)
    ap.add_argument("--dense", type=int, default=20,
                    help="resampling factor defining ground truth")
    ap.add_argument("--iso-time", type=int, nargs="*", default=[1, 2, 4, 8],
                    help="densities to time for the accuracy-per-second curve")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if args.generate_stress:
        generate_stress(args.generate_stress, seed=args.seed, out=args.skeletons)
        print(f"wrote {args.generate_stress} stress paths to {args.skeletons}")
        return

    d = np.load(args.skeletons)
    paths = d["paths"][:args.plans]
    B, S, T = paths.shape[:3]
    flat = paths.reshape(B * S, T, 7).astype(np.float32)
    fn = VALIDATORS[args.backend]

    if args.emit_truth:
        import _mesh_truth as mt
        t0 = time.perf_counter()
        truth = mt.paths_valid(flat)
        np.savez_compressed(args.emit_truth, valid=truth)
        print(f"mesh truth: {int(truth.sum())}/{len(truth)} paths clear "
              f"({time.perf_counter()-t0:.1f}s) -> {args.emit_truth}")
        return

    # --- ground truth ------------------------------------------------------ #
    if args.truth_file:
        # Mesh reference, evaluated at the SAME waypoints, so the only thing
        # separating a backend from the truth is its geometric model.
        truth = np.load(args.truth_file)["valid"][:B * S]
        t_dense = float("nan")
        dense = flat
    else:
        dense = resample(flat, args.dense)
        t_dense, truth = timed(fn, dense, reps=1)
    t_native, native = timed(fn, flat, reps=args.reps)

    fa = int((native & ~truth).sum())
    fr = int((~native & truth).sum())
    n_bad = int((~truth).sum())
    print(f"\n{args.backend}: {B*S} segments, native T={T}, "
          + (f"reference = MESH ({args.truth_file})"
             if args.truth_file else
             f"dense T={dense.shape[1]} ({args.dense}x)"))
    print(f"  truly invalid (dense)      {n_bad:6d} / {B*S}")
    print(f"  caught at native T         {int((~native).sum()):6d}")
    print(f"  FALSE ACCEPTS              {fa:6d}"
          + (f"  ({100.0*fa/n_bad:.1f}% of invalid segments missed)" if n_bad else ""))
    n_good = int(truth.sum())
    print(f"  false rejects              {fr:6d}"
          + (f"  ({100.0*fr/n_good:.1f}% of clear segments refused)"
             if args.truth_file and n_good else "   (0 by construction)"))

    # Plan-level: a plan is usable only if all its segments are.
    pn = native.reshape(B, S).all(axis=1)
    pt = truth.reshape(B, S).all(axis=1)
    pfa = int((pn & ~pt).sum())
    print(f"  plan-level false accepts   {pfa:6d} / {B}"
          f"  ({100.0*pfa/B:.1f}% of plans returned as valid but are not)")

    # --- accuracy per unit time -------------------------------------------- #
    rows = []
    print(f"\n  density   T     wall_ms   segs/s     false accepts")
    for k in args.iso_time:
        pk_ = resample(flat, k)
        t, v = timed(fn, pk_, reps=args.reps)
        f = int((v & ~truth).sum())
        rows.append(dict(density=k, T=pk_.shape[1], wall_s=t,
                         segs_per_s=(B * S) / t, false_accepts=f))
        print(f"  {k:5d}x  {pk_.shape[1]:4d}  {t*1e3:9.2f}  {(B*S)/t:9.0f}  {f:6d}")

    out = dict(backend=args.backend, n_segments=B * S, n_plans=B, native_T=T,
               dense_factor=args.dense, dense_T=int(dense.shape[1]),
               truly_invalid=n_bad, false_accepts=fa, false_rejects=fr,
               plan_false_accepts=pfa, t_native=t_native, t_dense=t_dense,
               native_valid=native.tolist(), truth_valid=truth.tolist(),
               curve=rows)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\n  wrote {args.json}")


if __name__ == "__main__":
    main()
