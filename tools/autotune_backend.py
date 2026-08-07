#!/usr/bin/env python
"""Pick the fastest pyroffi::Tier for each (robot, batch) cell, empirically.

WHY THIS SWEEPS TWO AXES
------------------------
The thread/warp/block choice is not a property of the robot. It depends on DOF and
batch size *jointly*: the thread tier amortises 32 seeds across a warp and wins once
there is enough work to fill the GPU, while the block tier wins the small-batch corner
where the thread tier is latency-bound and leaves SMs idle. A sweep over DOF alone
would report whichever tier happened to win at the batch size it was run at, and that
answer flips underneath you. So the unit of measurement here is the (robot, batch)
cell, and the output is a table, not a single verdict.

WHY EACH TIER GETS ITS OWN PROCESS
----------------------------------
`pyroffi_tier_from_env()` in _tier_kernel.cuh caches PYROFFI_IK_TIER in a
function-local static, deliberately: re-reading the environment per launch would land
in the very timings an autotune collects. That means a tier cannot be changed inside a
live process — os.environ writes after the first kernel launch do nothing. The parent
therefore forks one child per (robot, tier) and the child reports JSON on stdout. One
child covers every batch for its tier, so the robot build and JIT warm-up amortise.

SCOPE
-----
Only ls / hjcd / sqp are tiered (they include _tier_kernel.cuh). MPPI is not — it has
no dense solve to tier — so it is not swept.

G1 (43 DOF) is swept, but BLOCK TIER ONLY and not for a tier verdict. Above 32 DOF the
kernels are locked to the block tier (_glass_solve.cuh's TIER_CHOICE_MAX_N: the thread
and warp tiers do not fit), so PYROFFI_IK_TIER is ignored and all three tiers would run
identical code — a three-way sweep would report the noise between them as a winner.
What the G1 row measures is block-tier scaling across batch, which is real. Its other
two columns print `locked`.

G1 also needs a `--max-act 48` build; the default MAX_ACT=16 .so set will be skipped
with a rebuild hint (see _preflight). NOTE that rebuilding at 48 to include G1 makes
the OTHER robots' numbers pessimistic — MAX_ACT sizes per-thread arrays in every
kernel, so Panda/Fetch/Baxter measured on a --max-act 48 build are not measuring the
build you would deploy for them. Sweep G1 separately from the small arms.

USAGE
    python tools/autotune_backend.py                        # sweep all, print tables
    python tools/autotune_backend.py --robots panda --solvers ls
    python tools/autotune_backend.py --json out.json        # also dump raw timings
"""

from __future__ import annotations

import argparse
import ctypes
import functools
import json
import os
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"
KERNELS_DIR = REPO_ROOT / "src" / "pyroffi" / "cuda_kernels"

TIERS = ("thread", "warp", "block")

# Above this DOF the kernels are LOCKED to the block tier and PYROFFI_IK_TIER is
# ignored — the thread/warp tiers do not fit (see _glass_solve.cuh's TIER_CHOICE_MAX_N,
# which this mirrors). Such a robot has no tier choice to autotune, so sweeping all
# three would run identical code three times and report the noise between them as a
# winner. _tiers_for() collapses those robots to a single block-tier run instead.
TIER_CHOICE_MAX_ACT = 32

DEFAULT_ROBOTS = ("panda", "fetch", "baxter", "g1")
DEFAULT_SOLVERS = ("ls", "hjcd", "sqp")

# The .so whose compiled MAX_ACT gates each solver. Read in the PARENT so an oversized
# robot is reported once, actionably, instead of as N identical child crashes.
SOLVER_LIBS = {
    "ls":   KERNELS_DIR / "ik" / "_ls_ik_cuda_lib.so",
    "hjcd": KERNELS_DIR / "ik" / "_hjcd_ik_cuda_lib.so",
    "sqp":  KERNELS_DIR / "ik" / "_sqp_ik_cuda_lib.so",
}

# Batch = number of IK targets solved in one call. Total seeds = batch * num_seeds(32),
# so this spans ~32 to ~131k seeds -- across the point where the tiers trade places.
DEFAULT_BATCHES = (1, 4, 16, 64, 256, 1024, 4096)


# ---------------------------------------------------------------------------
# Worker: runs inside a child process, with PYROFFI_IK_TIER already fixed.
# ---------------------------------------------------------------------------

def _worker(robot_name: str, solver: str, batches: list[int]) -> int:
    sys.path.insert(0, str(TESTS_DIR))

    import jax
    import jax.numpy as jnp
    import jaxlie
    import numpy as np
    import pyroffi as pk
    import yourdfpy

    # Reuse the benchmark's robot tables and batch timer rather than restating them:
    # a second copy of ROBOT_URDFS would drift from the one bench_ik actually uses.
    import bench_ik as B

    solver_fns = {
        "ls":   (B.ls_ik_solve_cuda_batch,   B.IK_KWARGS_LS_CUDA),
        "hjcd": (B.hjcd_solve_cuda_batch,    B.IK_KWARGS_HJCD_CUDA),
        "sqp":  (B.sqp_ik_solve_cuda_batch,  B.IK_KWARGS_SQP_CUDA),
    }
    fn, kwargs = solver_fns[solver]

    urdf_path = B.ROBOT_URDFS[robot_name]
    mesh_dir = urdf_path.parent / "meshes"
    urdf = (
        yourdfpy.URDF.load(str(urdf_path), mesh_dir=str(mesh_dir))
        if mesh_dir.exists()
        else yourdfpy.URDF.load(str(urdf_path))
    )
    robot = pk.Robot.from_urdf(urdf)
    n_act = robot.joints.num_actuated_joints

    target_link_name = B._resolve_target_link_name(robot_name, robot)
    tli = robot.links.names.index(target_link_name)
    fixed_joint_names = B.ROBOT_FIXED_JOINT_NAMES.get(robot_name, ())
    fixed_joint_mask = jnp.array(
        [n in fixed_joint_names for n in robot.joints.actuated_names], dtype=jnp.int32
    )

    lo = np.array(robot.joints.lower_limits)
    hi = np.array(robot.joints.upper_limits)
    mid_cfg = jnp.array((lo + hi) / 2, dtype=jnp.float32)

    # Targets are FK of random configs, so every target is reachable by construction --
    # an unreachable target would make the solver burn its full iteration budget and
    # time the failure path instead of the solve.
    rng_np = np.random.default_rng(0)
    max_batch = max(batches)
    cfgs = rng_np.uniform(lo, hi, size=(max_batch, n_act)).astype(np.float32)
    poses = jnp.stack(
        [robot.forward_kinematics(jnp.array(cfgs[i]))[tli] for i in range(max_batch)]
    )  # (max_batch, 7)

    rng_keys_seq = jnp.stack(
        [jax.random.PRNGKey(i) for i in range(B.N_DEVICE_REPEATS)]
    )  # (N_DEVICE_REPEATS, 2) -- the CUDA batch timer's expected shape

    results: dict[str, float] = {}
    for bs in batches:
        tp = poses[:bs]
        prev = jnp.tile(mid_cfg[None], (bs, 1))
        timer = B._build_batch_ik_timer(fn, robot, tli, fixed_joint_mask, kwargs,
                                        is_jax_batch=False)
        # Warm up with the exact shapes about to be timed. A warm-up on a different
        # batch size just recompiles inside the first timed call and reports the
        # compile as solve time.
        for _ in range(B.N_WARMUP):
            jax.block_until_ready(timer(tp, prev, rng_keys_seq))
        results[str(bs)] = B._time_scan(timer, tp, prev, rng_keys_seq) * 1e3  # ms

    print("PYROFFI_AUTOTUNE_JSON " + json.dumps({
        "robot": robot_name,
        "solver": solver,
        "dof": int(n_act),
        "tier": os.environ.get("PYROFFI_IK_TIER", "thread"),
        "ms": results,
    }))
    return 0


# ---------------------------------------------------------------------------
# Parent: preflight, fork a child per (robot, solver, tier), collect, tabulate.
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _robot_dof(robot_name: str) -> int:
    """Actuated DOF, parsed from the same URDF bench_ik uses.

    Read in the parent so it can decide which tiers are even applicable before forking.
    Meshes and the scene graph are skipped — only the joint list matters here, and
    loading G1's meshes to count joints would dominate the tool's startup.
    """
    sys.path.insert(0, str(TESTS_DIR))
    import bench_ik as B
    import yourdfpy

    urdf = yourdfpy.URDF.load(
        str(B.ROBOT_URDFS[robot_name]), load_meshes=False, build_scene_graph=False
    )
    return sum(
        j.type in ("revolute", "prismatic", "continuous") for j in urdf.robot.joints
    )


def _lib_max_act(solver: str) -> int | None:
    """MAX_ACT the solver's .so was built with, or None if unbuilt/unreadable."""
    so = SOLVER_LIBS[solver]
    if not so.exists():
        return None
    try:
        fn = ctypes.CDLL(str(so)).pyroffi_max_act
    except (OSError, AttributeError):
        return None
    fn.restype = ctypes.c_int
    return int(fn())


def _tiers_for(robot_name: str, requested: list[str]) -> list[str]:
    """Tiers worth measuring for this robot — block only, past TIER_CHOICE_MAX_ACT."""
    if _robot_dof(robot_name) > TIER_CHOICE_MAX_ACT:
        return ["block"]
    return requested


def _preflight(robot_name: str, solver: str) -> str | None:
    """Reason this (robot, solver) cannot be measured, or None if it can.

    The kernels refuse a robot past their compiled MAX_ACT (they would otherwise run off
    the end of their per-thread arrays), so a too-small build is a property of the .so,
    not of the run. Catch it here rather than letting every child die the same way.
    """
    dof = _robot_dof(robot_name)
    max_act = _lib_max_act(solver)
    if max_act is None:
        return None  # unbuilt or pre-accessors; let the child produce the real error
    if dof > max_act:
        need = min(((dof + 7) // 8) * 8, 64)
        return (
            f"{dof} DOF > {SOLVER_LIBS[solver].name}'s MAX_ACT={max_act}. "
            f"Rebuild: bash build_kernels/build_all.sh --max-act {need}"
        )
    return None

def _run_cell(robot: str, solver: str, tier: str, batches: list[int],
              timeout: int) -> dict | None:
    env = {**os.environ, "PYROFFI_IK_TIER": tier}
    cmd = [
        sys.executable, str(pathlib.Path(__file__).resolve()),
        "--_worker", "--robot", robot, "--solver", solver,
        "--batches", ",".join(str(b) for b in batches),
    ]
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  {solver}/{robot}/{tier}: TIMEOUT after {timeout}s", file=sys.stderr)
        return None
    if p.returncode != 0:
        tail = (p.stderr or "").strip().splitlines()[-3:]
        print(f"  {solver}/{robot}/{tier}: exit {p.returncode}", file=sys.stderr)
        for line in tail:
            print(f"      {line}", file=sys.stderr)
        return None
    for line in p.stdout.splitlines():
        if line.startswith("PYROFFI_AUTOTUNE_JSON "):
            return json.loads(line[len("PYROFFI_AUTOTUNE_JSON "):])
    print(f"  {solver}/{robot}/{tier}: no result line", file=sys.stderr)
    return None


def _print_table(solver: str, robot: str, dof: int, by_tier: dict[str, dict],
                 batches: list[int], swept: list[str]) -> list[tuple[int, str]]:
    """Print one (solver, robot) table. Returns [(batch, winning_tier)].

    `swept` is the tiers actually measured. A tier outside it was never a candidate
    (block-locked robot) and prints as `locked` — distinct from `--`, which means it was
    swept and failed. When only one tier is a candidate there is no contest, so no
    winner is starred and no speedup is claimed: the column is the answer, not a verdict.
    """
    locked = dof > TIER_CHOICE_MAX_ACT
    note = f"  [>{TIER_CHOICE_MAX_ACT} DOF: block-tier only]" if locked else ""
    print(f"\n{solver} / {robot} ({dof} DOF)   [ms per call, lower is better]{note}")
    print(f"  {'batch':>7} | " + " | ".join(f"{t:>9}" for t in TIERS) + " |  winner")
    print("  " + "-" * (9 + 12 * len(TIERS) + 12))

    contest = len(swept) > 1
    winners: list[tuple[int, str]] = []
    for bs in batches:
        cells = {t: by_tier.get(t, {}).get("ms", {}).get(str(bs)) for t in swept}
        live = {t: v for t, v in cells.items() if v is not None}
        if not live:
            continue
        best = min(live, key=live.get)
        winners.append((bs, best))

        def fmt(t: str) -> str:
            if t not in swept:
                return f"{'locked':>9}"
            v = cells.get(t)
            if v is None:
                return f"{'--':>9}"
            return f"{v:>8.3f}{'*' if (contest and t == best) else ' '}"

        speedup = ""
        if contest:
            others = [v for t, v in live.items() if t != best]
            if others:
                speedup = f"  ({min(others) / live[best]:.2f}x)"
        verdict = f"{best}{speedup}" if contest else f"{best} (locked)"
        print(f"  {bs:>7} | " + " | ".join(fmt(t) for t in TIERS) + f" |  {verdict}")
    return winners


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robots", default=",".join(DEFAULT_ROBOTS))
    ap.add_argument("--solvers", default=",".join(DEFAULT_SOLVERS))
    ap.add_argument("--tiers", default=",".join(TIERS))
    ap.add_argument("--batches", default=",".join(str(b) for b in DEFAULT_BATCHES))
    ap.add_argument("--timeout", type=int, default=1800,
                    help="per-child wall-clock budget (s)")
    ap.add_argument("--json", type=pathlib.Path, default=None,
                    help="also write raw timings here")
    # Internal: marks the child process. Not for interactive use.
    ap.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--robot", help=argparse.SUPPRESS)
    ap.add_argument("--solver", help=argparse.SUPPRESS)
    args = ap.parse_args()

    batches = [int(b) for b in args.batches.split(",") if b]

    if args._worker:
        return _worker(args.robot, args.solver, batches)

    robots = [r for r in args.robots.split(",") if r]
    solvers = [s for s in args.solvers.split(",") if s]
    tiers = [t for t in args.tiers.split(",") if t]

    print("=" * 78)
    print("pyroffi tier autotune")
    print(f"  robots={robots}  solvers={solvers}  tiers={tiers}")
    print(f"  batches={batches}   (total seeds = batch x num_seeds)")
    print("=" * 78)

    raw: list[dict] = []
    summary: dict[str, list] = {}

    for solver in solvers:
        for robot in robots:
            skip = _preflight(robot, solver)
            if skip is not None:
                print(f"\nskip {solver}/{robot}: {skip}", file=sys.stderr)
                continue

            robot_tiers = _tiers_for(robot, tiers)
            if robot_tiers != tiers:
                print(f"note: {robot} is {_robot_dof(robot)} DOF (> {TIER_CHOICE_MAX_ACT}) "
                      f"— locked to the block tier; sweeping {robot_tiers} only.",
                      file=sys.stderr)

            by_tier: dict[str, dict] = {}
            for tier in robot_tiers:
                print(f"running {solver}/{robot}/{tier} ...", flush=True)
                r = _run_cell(robot, solver, tier, batches, args.timeout)
                if r is not None:
                    by_tier[tier] = r
                    raw.append(r)
            if not by_tier:
                print(f"\n{solver} / {robot}: no tier produced a result", file=sys.stderr)
                continue
            dof = next(iter(by_tier.values()))["dof"]
            winners = _print_table(solver, robot, dof, by_tier, batches, robot_tiers)
            summary[f"{solver}/{robot}"] = [{"batch": b, "tier": t} for b, t in winners]

    if summary:
        print("\n" + "=" * 78)
        print("recommended tier by (solver, robot, batch)")
        print("=" * 78)
        for key, wins in summary.items():
            runs = []  # collapse consecutive batches that pick the same tier
            for w in wins:
                if runs and runs[-1][0] == w["tier"]:
                    runs[-1][2] = w["batch"]
                else:
                    runs.append([w["tier"], w["batch"], w["batch"]])
            parts = [
                f"{t} @ batch {lo}" if lo == hi else f"{t} @ batch {lo}-{hi}"
                for t, lo, hi in runs
            ]
            print(f"  {key:<16} " + ";  ".join(parts))

    if args.json:
        args.json.write_text(json.dumps({"raw": raw, "summary": summary}, indent=2))
        print(f"\nwrote {args.json}")

    return 0 if raw else 1


if __name__ == "__main__":
    sys.exit(main())
