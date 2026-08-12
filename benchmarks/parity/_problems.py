"""Shared problem-set format for the parity benchmark.

Deliberately a plain .npz of numpy arrays: the two solvers run under different
conda envs and cannot share a pickle, a torch tensor, or a JAX array.
"""
from __future__ import annotations

import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PROBLEM_FILE = HERE / "problems.npz"
RESULT_DIR = HERE / "results"

# Obstacle placed in the reachable volume. Kept here rather than in either
# runner so both stacks are guaranteed to build the SAME world.
#
# A CUBOID rather than a sphere: cuRobo's primitive collision path raises
# "Primitive Collision has no obstacles" for a sphere-only world, while cuboids
# are its well-trodden case (see curobo examples/motion_gen_api_example.py).
# pyroffi supports both, so the shared shape is chosen to be the one BOTH
# stacks handle natively -- converting on one side would compare different
# geometry.
OBSTACLE_CENTER = np.array([0.45, 0.0, 0.45], dtype=np.float64)
OBSTACLE_DIMS = np.array([0.24, 0.24, 0.24], dtype=np.float64)   # full extents


def load():
    if not PROBLEM_FILE.exists():
        raise SystemExit(f"{PROBLEM_FILE} missing — run make_problems.py first")
    d = np.load(PROBLEM_FILE)
    return d["q_ref"], d["target_wxyz_xyz"], d["ee_link_index"].item()


def save_result(name: str, **arrays) -> None:
    RESULT_DIR.mkdir(exist_ok=True)
    np.savez(RESULT_DIR / f"{name}.npz", **arrays)
    print(f"wrote {RESULT_DIR / (name + '.npz')}")
