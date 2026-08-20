"""Three-way TAMP benchmark: cuTAMP vs PDDLStream on two geometry backends.

Compares, end to end on the same task:

    cutamp            cuTAMP (NVlabs, RSS 2025) — own skeleton search plus
                      GPU-parallel differentiable particle optimisation
    pddlstream-stock  PDDLStream + FastDownward, stock SPaSM kinematics
    pddlstream-pyroffi  the same, with pyroffi as the geometric oracle

These are three *systems*, not three backends under one planner. cuTAMP brings
its own symbolic search (``cutamp/task_planning/``) and imports pddlstream
nowhere, so there is no oracle to swap: its constraint satisfaction *is* the
optimisation. Comparing them end to end is the honest framing; see the module
notes below for what that costs in interpretability.

Protocol follows the cuTAMP paper (arXiv 2411.11833, Sec. VIII) so numbers are
comparable to their Table IV:

* coverage (success rate) over N trials, 50 for Tetris in the paper
* solution time as mean +/- 95% confidence interval
* particle batch size N_b swept for cuTAMP; irrelevant to PDDLStream

Two fairness hazards, both handled explicitly rather than papered over
-------------------------------------------------------------------
**1. cuTAMP's published times exclude motion generation.** The paper states it
defers motion planning until after solving placements and configurations, and
that "the timing information we present does not include this motion planning
time". PDDLStream's ``s-motion`` is inside its search loop and cannot be
excluded the same way. So this harness reports *two* times per run:

    t_placements   time to satisfying placements + configurations
                   (the quantity cuTAMP reports)
    t_full         time to a complete plan including motions
                   (what a robot actually needs)

Quoting cuTAMP's ``t_placements`` against PDDLStream's ``t_full`` would flatter
cuTAMP by however long motion generation takes. Both columns are always shown.

**2. cuTAMP's paper does not benchmark PDDLStream.** Its baselines are internal
ablations — SAMPLING (resample, no optimisation) and OPTIMIZATION (uniform
init, no sampling) — described as emulating serial sampling-based planners when
run with one particle. That emulation is not PDDLStream: no FastDownward, no
stream algebra, no incremental fact generation. So the ``pddlstream-*``
configurations here are a genuinely new baseline rather than a reproduction,
and should be described that way. Their SAMPLING row is still worth reporting
alongside as the closest published proxy (via ``--approach sampling``).

Environments
------------
cuTAMP needs its own environment (Python 3.10, PyTorch, cuRobo v0.7.8 built
from source); the pyroffi stack needs ``pyroffi-tamp``. They cannot coexist, so
each configuration runs in its own subprocess under its own interpreter — which
is also the only honest way to time them, since neither can warm the other's
caches.

Run::

    python tamp/benchmarks/bench_cutamp.py --problem tetris_5 --trials 10
    python tamp/benchmarks/bench_cutamp.py --problem tetris_3 --trials 50 \\
        --batch-sizes 512 1024 4096
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TAMP_ROOT = Path(__file__).resolve().parents[1]
CUTAMP_ROOT = TAMP_ROOT / "external" / "cutamp"

#: Conda environments per configuration. cuTAMP's is separate by necessity.
ENVS = {
    "cutamp": os.environ.get("CUTAMP_ENV", "cutamp"),
    "pddlstream-stock": os.environ.get("PYROFFI_TAMP_ENV", "pyroffi-tamp"),
    "pddlstream-pyroffi": os.environ.get("PYROFFI_TAMP_ENV", "pyroffi-tamp"),
}

#: Particle batch sizes from the paper's Tetris table. Only cuTAMP uses these.
DEFAULT_BATCH_SIZES = (512, 1024, 2048, 4096)


def ci95(xs):
    """Mean and 95% confidence interval half-width, as the paper reports."""
    xs = np.asarray([x for x in xs if x is not None and math.isfinite(x)])
    if xs.size == 0:
        return float("nan"), float("nan")
    if xs.size == 1:
        return float(xs[0]), 0.0
    return float(xs.mean()), float(1.96 * xs.std(ddof=1) / math.sqrt(xs.size))


def _conda_run(env, args, cwd, timeout, extra_env=None):
    e = dict(os.environ)
    e.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if extra_env:
        e.update(extra_env)
    cmd = ["conda", "run", "--no-capture-output", "-n", env] + args
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                          env=e, timeout=timeout)
    return proc, time.perf_counter() - t0


# --------------------------------------------------------------------------- #
# cuTAMP
# --------------------------------------------------------------------------- #

def run_cutamp(problem, seed, batch_size, timeout, approach="optimization",
               tuned=False, exp_root=None, motion_plan=False):
    """One cuTAMP trial via its own entry point.

    Results are read from the JSON that cuTAMP's ``ExperimentLogger`` writes
    (``overall_metrics.json`` / ``timer_metrics.json``), not scraped from
    stdout. Its logging goes through loguru at debug level and its stdout
    carries no stable result line, so grepping it would silently invent
    numbers.
    """
    exp_root = Path(exp_root or (TAMP_ROOT / "benchmarks" / "results" / "cutamp"))
    tag = "mp" if motion_plan else "nomp"
    name = f"{problem}_{approach}_{tag}_nb{batch_size}_t{seed}" + ("_tuned" if tuned else "")
    run_dir = exp_root / name
    if run_dir.exists():
        import shutil
        shutil.rmtree(run_dir)          # ExperimentLogger warns and reuses otherwise

    # NOTE: run_cutamp.py exposes no --seed. Trials differ only through internal
    # sampling randomness, so `seed` here is a trial *label*, not a seed. That
    # rules out seed-paired comparisons against the PDDLStream configurations —
    # coverage and timing distributions are comparable, individual runs are not.
    args = [
        "python", "-m", "cutamp.scripts.run_cutamp",
        "--env", problem,
        "-n", str(batch_size),
        "--approach", approach,
        "--disable_visualizer",
        "--max_duration", str(timeout),
        "--experiment_root", str(exp_root),
        "--experiment_id", name,
    ]
    if motion_plan:
        # cuTAMP CAN generate motions; its published timings simply exclude
        # them. Running both ways gives t_placements and t_full from the same
        # system measured identically, rather than reconstructing one of them.
        args.append("--motion_plan")
    if tuned:
        args.append("--tuned_tetris_weights")

    try:
        proc, wall = _conda_run(ENVS["cutamp"], args, CUTAMP_ROOT, timeout * 3 + 120)
    except subprocess.TimeoutExpired:
        return {"solved": False, "t_placements": None, "t_full": None,
                "wall": timeout, "note": "timeout"}

    overall = _read_json(run_dir / "overall_metrics.json")
    timers = _read_json(run_dir / "timer_metrics.json")
    if overall is None:
        return {"solved": False, "t_placements": None, "t_full": None,
                "wall": wall, "note": (proc.stderr or proc.stdout)[-400:]}

    # cuTAMP's reported time is the sampling + optimisation loop; motion
    # generation is deliberately excluded from it (paper, Sec. VIII). `wall` is
    # the honest end-to-end number for the same process.
    t_place = _timer_total(timers)
    return {
        "solved": bool(overall.get("found_solution")),
        "t_placements": t_place,
        "t_full": wall,
        "wall": wall,
        "best_cost": overall.get("best_cost"),
        "num_satisfying": overall.get("num_satisfying_final"),
        "returncode": proc.returncode,
    }


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _timer_total(timers):
    """Sampling + optimisation time, the quantity the paper's tables report."""
    if not isinstance(timers, dict):
        return None
    for key in ("start_optimization", "optimization", "total"):
        v = timers.get(key)
        if isinstance(v, dict):
            v = v.get("total", v.get("sum", v.get("mean")))
        if isinstance(v, (int, float)):
            return float(v)
    return None


# --------------------------------------------------------------------------- #
# PDDLStream (both geometry backends)
# --------------------------------------------------------------------------- #

_PDDL_TRIAL = r'''
import contextlib, io, json, os, sys, time
sys.path.insert(0, {root!r})
from spasm.tamp import _setup
from spasm.tamp.tetris_problem import make_tetris_world, pddlstream_from_tetris_world
from spasm.tamp.geometry import IK_BACKEND
from pddlstream.algorithms.meta import solve

# Same tetris instance cuTAMP solves, built from SPaSM's own geometry. The
# Simulation is deterministic, so `seed` varies only the placement sampler's
# RNG -- which is where all the variance in a rejection-based planner lives.
world = make_tetris_world({n}, seed={seed})
problem = pddlstream_from_tetris_world(world, motion_backend={motion!r})
t0 = time.perf_counter()
with contextlib.redirect_stdout(io.StringIO()):
    plan, cost, _ = solve(problem, algorithm="adaptive", unit_costs=False,
                          max_time={budget}, verbose=False)
wall = time.perf_counter() - t0
print("RESULT " + json.dumps({{
    "solved": plan is not None,
    "t_full": wall,
    "cost": None if plan is None else float(cost),
    "plan_len": None if plan is None else len(plan),
    "ik_backend": IK_BACKEND,
}}))
'''


def run_pddlstream(n_objects, seed, budget, ik_backend, motion_backend="linear"):
    """One PDDLStream trial with the given geometric oracle."""
    src = _PDDL_TRIAL.format(root=str(TAMP_ROOT), n=n_objects, seed=seed,
                             budget=budget, motion=motion_backend)
    env = {"PYROFFI_ANALYTIC_IK": ik_backend, "PYTHONPATH": str(TAMP_ROOT)}
    try:
        proc, wall = _conda_run(ENVS["pddlstream-stock"], ["python", "-c", src],
                                TAMP_ROOT, budget * 3, extra_env=env)
    except subprocess.TimeoutExpired:
        return {"solved": False, "t_placements": None, "t_full": budget,
                "note": "timeout"}

    line = next((l for l in proc.stdout.splitlines()
                 if l.startswith("RESULT ")), None)
    if line is None:
        return {"solved": False, "t_placements": None, "t_full": None,
                "note": (proc.stderr or proc.stdout)[-400:]}
    r = json.loads(line[len("RESULT "):])
    # PDDLStream interleaves motion generation with search, so there is no
    # placements-only checkpoint to report. Left as None rather than guessed:
    # inventing one would be the exact unfairness this harness exists to avoid.
    r["t_placements"] = None
    return r


# --------------------------------------------------------------------------- #

def summarise(rows, label):
    solved = [r for r in rows if r.get("solved")]
    m_full, c_full = ci95([r.get("t_full") for r in solved])
    m_pl, c_pl = ci95([r.get("t_placements") for r in solved])
    pl = "—" if math.isnan(m_pl) else f"{m_pl:.2f} ± {c_pl:.2f}"
    fu = "—" if math.isnan(m_full) else f"{m_full:.2f} ± {c_full:.2f}"
    return f"| {label:<28} | {len(solved)}/{len(rows)} | {pl:>16} | {fu:>16} |"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problem", default="tetris_5",
                    help="cuTAMP problem name; the PDDLStream side uses the "
                         "matching object count")
    ap.add_argument("--objects", type=int, default=None,
                    help="pieces for the PDDLStream configurations; defaults to "
                         "the count parsed from --problem (tetris_N)")
    ap.add_argument("--trials", type=int, default=10,
                    help="the paper uses >=10, and 50 for Tetris")
    ap.add_argument("--budget", type=float, default=60.0,
                    help="per-trial time budget, seconds")
    ap.add_argument("--batch-sizes", type=int, nargs="+",
                    default=list(DEFAULT_BATCH_SIZES))
    ap.add_argument("--skip-cutamp", action="store_true")
    ap.add_argument("--skip-pddlstream", action="store_true")
    ap.add_argument("--motion-plan", action="store_true",
                    help="also run cuTAMP with --motion_plan, so t_full is "
                         "measured rather than reconstructed")
    ap.add_argument("--out", default=None, help="write raw results as JSON")
    args = ap.parse_args()
    if args.objects is None:
        # Keep the two sides on the same instance by construction rather than
        # relying on the caller to pass matching numbers.
        args.objects = int(args.problem.rsplit("_", 1)[-1])

    if not args.skip_cutamp and not CUTAMP_ROOT.exists():
        sys.exit(f"missing {CUTAMP_ROOT} — run tamp/setup_externals.sh")

    results = {}

    if not args.skip_cutamp:
        for nb in args.batch_sizes:
            for mp in ((False, True) if args.motion_plan else (False,)):
                label = f"cutamp (N_b={nb})" + (" +motion" if mp else "")
                rows = [run_cutamp(args.problem, s, nb, args.budget,
                                   motion_plan=mp)
                        for s in range(args.trials)]
                results[label] = rows
                print(summarise(rows, label), flush=True)

        label = "cutamp SAMPLING (N_b=4096)"
        rows = [run_cutamp(args.problem, s, 4096, args.budget,
                           approach="sampling") for s in range(args.trials)]
        results[label] = rows
        print(summarise(rows, label), flush=True)

    if args.skip_pddlstream:
        return _finish(results, args)

    for ik in ("spasm", "pyroffi"):
        label = f"pddlstream-{ik}"
        rows = [run_pddlstream(args.objects, s, args.budget, ik)
                for s in range(args.trials)]
        results[label] = rows
        print(summarise(rows, label), flush=True)

    _finish(results, args)


def _finish(results, args):
    print(f"\n## {args.problem} — {args.trials} trials, {args.budget:.0f}s budget\n")
    print(f"| configuration | coverage | t_placements (s) | t_full (s) |")
    print(f"|---|--:|--:|--:|")
    for label, rows in results.items():
        print(summarise(rows, label))
    print("\nt_placements is the quantity the cuTAMP paper reports (motion "
          "generation excluded).\nt_full includes motion. PDDLStream has no "
          "placements-only checkpoint — it\ninterleaves motion with search — so "
          "that cell is empty rather than guessed.")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
