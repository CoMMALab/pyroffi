"""TAMP with PDDLStream and pyroffi as the geometric oracle

A task-and-motion planning problem has two halves that have to agree: a
symbolic planner decides *what* to do (pick b1, place it in the goal box) and a
geometric oracle decides whether any of that is physically possible — is there
a grasp, does the IK solve, does the arm hit anything on the way. The usual
arrangement bolts a task planner onto a simulator and pays for every geometric
query through it.

Here the entire oracle is pyroffi. Every geometric primitive PDDLStream calls
routes through pyroffi:

* ``s-ik``     — analytic Franka IK for a top-down grasp
* ``s-motion`` — a collision-checked joint path (see 20_01 for the backends)
* ``t-cfree``  — sphere-sphere penetration between two placements
* arm collision — FK over the robot's 59 collision spheres

There is no pybullet and no simulator in the loop. The point of this example is
that pyroffi is a viable motion-generation backend for TAMP: it answers the
oracle queries a real task planner makes, fast enough that the planner is not
waiting on it.

The task is tabletop rearrangement: N cubes scattered in a start region must be
packed collision-free into a goal box on a Franka Panda.

Run::

    python examples/20_00_tamp_pddlstream.py --blocks 3
    python examples/20_00_tamp_pddlstream.py --blocks 2 4 6 --seeds 3
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time
from pathlib import Path

import numpy as np

# The TAMP experiments live under `tamp/` with their own vendored PDDLStream.
TAMP_ROOT = Path(__file__).resolve().parents[1] / "tamp"
sys.path.insert(0, str(TAMP_ROOT))

# The IK backend is selected at import time by spasm.tamp.geometry, so it has
# to be in the environment before that import happens.
if "--ik-backend" in sys.argv:
    os.environ["PYROFFI_ANALYTIC_IK"] = sys.argv[sys.argv.index("--ik-backend") + 1]

from spasm.tamp import _setup  # noqa: E402  (path shim; must precede the rest)
from spasm.tamp.problems import make_rearrange_world, pddlstream_from_world  # noqa: E402
from spasm.tamp import geometry as g  # noqa: E402


def solve_once(num_blocks, seed, max_time, algorithm, motion_backend):
    from pddlstream.algorithms.meta import solve

    world = make_rearrange_world(num_blocks, seed=seed)
    problem = pddlstream_from_world(world, motion_backend=motion_backend)

    # PDDLStream's search prints regardless of verbose=; swallow it so the
    # example's own table is readable. Timing brackets only the solve.
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        plan, cost, _ = solve(problem, algorithm=algorithm, unit_costs=False,
                              max_time=max_time, verbose=False)
    wall = time.perf_counter() - t0

    return {
        "num_blocks": num_blocks,
        "seed": seed,
        "solved": plan is not None,
        "wall_s": wall,
        "cost": None if plan is None else float(cost),
        "plan_len": None if plan is None else len(plan),
        "plan": plan,
    }


def oracle_timing(num_blocks=3, seed=0, n=200):
    """Time the pyroffi primitives PDDLStream actually calls.

    This is the number that decides whether pyroffi is a viable backend: the
    task planner issues thousands of these, so their cost is the planner's
    cost.
    """
    world = make_rearrange_world(num_blocks, seed=seed)
    poses = list(world.initial_poses.values())
    q0 = np.asarray(world.conf0)

    # Warm the JIT before timing anything.
    g.ik_topdown(poses[0])
    g.arm_path_valid(g.interpolate(q0, q0, 20))

    t0 = time.perf_counter()
    for i in range(n):
        g.ik_topdown(poses[i % len(poses)])
    ik_ms = (time.perf_counter() - t0) / n * 1e3

    q1, _ = g.ik_topdown(poses[0])
    t0 = time.perf_counter()
    for _ in range(n):
        g.arm_path_valid(g.interpolate(q0, q1, 20))
    motion_ms = (time.perf_counter() - t0) / n * 1e3

    return {"ik_ms": ik_ms, "motion_check_ms": motion_ms}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--blocks", type=int, nargs="+", default=[3])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--max-time", type=float, default=120.0)
    ap.add_argument("--algorithm", default="adaptive",
                    choices=["adaptive", "focused", "binding", "incremental"])
    ap.add_argument("--ik-backend", default="spasm",
                    choices=["spasm", "pyroffi", "pyroffi-cfree"],
                    help="geometric oracle's IK: SPaSM's hand-rolled Franka "
                         "closed form, pyroffi's model-derived analytic solver, "
                         "or the latter restricted to self-collision-free "
                         "branches. Must be set before spasm.tamp is imported.")
    ap.add_argument("--motion-backend", default="linear",
                    choices=["linear", "spasm", "dynamics"],
                    help="see 20_01 for what these mean")
    args = ap.parse_args()

    rows = []
    for nb in args.blocks:
        for seed in range(args.seeds):
            r = solve_once(nb, seed, args.max_time, args.algorithm,
                           args.motion_backend)
            rows.append(r)
            status = "solved" if r["solved"] else "FAILED"
            print(f"  blocks={nb} seed={seed}: {status} "
                  f"wall={r['wall_s']:.2f}s len={r['plan_len']}", flush=True)

    from spasm.tamp.geometry import IK_BACKEND
    print(f"\n=== PDDLStream + pyroffi oracle (IK backend: {IK_BACKEND}) ===")
    print(f"{'blocks':>7} {'solved':>8} {'median wall (s)':>16} {'median len':>11}")
    for nb in sorted({r['num_blocks'] for r in rows}):
        rs = [r for r in rows if r["num_blocks"] == nb]
        ok = [r for r in rs if r["solved"]]
        med = np.median([r["wall_s"] for r in ok]) if ok else float("nan")
        ln = int(np.median([r["plan_len"] for r in ok])) if ok else 0
        print(f"{nb:>7} {len(ok)}/{len(rs):>6} {med:>16.2f} {ln:>11}")

    prim = oracle_timing()
    print("\n=== pyroffi oracle primitive cost ===")
    print(f"  analytic IK (top-down grasp)        {prim['ik_ms']:.3f} ms/call")
    print(f"  motion validity check (20 waypoints) {prim['motion_check_ms']:.3f} ms/call")
    print("\nThese are the queries the task planner issues thousands of times;"
          "\ntheir cost is what makes pyroffi viable as a TAMP backend.")


if __name__ == "__main__":
    main()
