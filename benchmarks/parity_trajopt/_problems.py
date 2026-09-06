"""Shared problem-set for the pyroffi-vs-cuRobo TRAJOPT parity benchmark.

Config->config motion: each problem is (q_start, q_goal), both collision-free,
sharing ONE cuboid obstacle. Plain .npz because the two solvers live in
different conda envs (pyroffi / curobo) and cannot share a pickle/tensor.

Fair-comparison choices (mirroring benchmarks/parity/README.md):
- config->config goals (no IK many-to-one confound; both plan joint->joint).
- Cuboid obstacle (cuRobo's well-trodden primitive; pyroffi handles both).
- Endpoints are collision-free by construction, so neither solver can score by
  failing fast on impossible problems.
- Both LOCAL trajectory optimizers are seeded from the SAME straight line, so
  this measures local collision-trajopt (cuRobo's lbfgs_bspline_trajopt vs
  pyroffi's L-BFGS dynamics_trajopt), NOT global graph planning.
- Success/collision recomputed by ONE shared metric in compare.py.
"""
from __future__ import annotations
import pathlib
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PROBLEM_FILE = HERE / "problems.npz"
RESULT_DIR = HERE / "results"

OBSTACLE_CENTER = np.array([0.45, 0.0, 0.45], dtype=np.float64)
OBSTACLE_DIMS = np.array([0.24, 0.24, 0.24], dtype=np.float64)   # full extents
T_WAYPOINTS = 32          # trajectory length both sides are scored on
CLEARANCE_MARGIN = 0.02   # m; shared safety margin for the collision metric


def load():
    if not PROBLEM_FILE.exists():
        raise SystemExit(f"{PROBLEM_FILE} missing -- run make_problems.py first")
    d = np.load(PROBLEM_FILE)
    return d["q_start"], d["q_goal"], d["joint_lower"], d["joint_upper"]


def save_result(name: str, **arrays) -> None:
    RESULT_DIR.mkdir(exist_ok=True)
    np.savez(RESULT_DIR / f"{name}.npz", **arrays)
    print(f"wrote {RESULT_DIR / (name + '.npz')}")
