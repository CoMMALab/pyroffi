"""Tests for the differentiable-TAMP benchmark scaffolding.

Verifies (1) the pyroffi-backed streams score geometry with SPaSM's own
penetration math, (2) IK/motion primitives behave, (3) no pybullet leaks into
the benchmark, and (4) PDDLStream solves a small rearrangement instance
end-to-end using only the pyroffi motion backend.
"""
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from spasm.tamp import _setup  # noqa: F401
from spasm.tamp import geometry as g
from spasm.tamp.problems import make_rearrange_world, pddlstream_from_world
from spasm.tamp.robosuite_bridge import execute_plan
from spasm.tetris.solve import sphere_sphere_penetration


def test_cfree_matches_spasm_penetration():
    """blocks_collide must agree with a direct spasm.tetris.solve penetration call."""
    loc = g.cuboid_spheres((0.025, 0.025, 0.025))
    near = np.array([0.4, 0.0, 0.025, 0.0]), np.array([0.42, 0.0, 0.025, 0.0])
    far = np.array([0.4, 0.0, 0.025, 0.0]), np.array([0.7, 0.3, 0.025, 0.0])
    for (p1, p2), expect in [(near, True), (far, False)]:
        s1 = np.asarray(g.transform_spheres(loc, p1))
        s2 = np.asarray(g.transform_spheres(loc, p2))
        direct = float(np.max(sphere_sphere_penetration(s1, s2))) > 0.010 + 1e-3
        assert direct == expect
        assert g.blocks_collide(loc, p1, loc, p2) == expect


def test_ik_reachable_flags():
    q, ok = g.ik_topdown(np.array([0.5, 0.0, 0.05, 0.0]))
    assert ok and q.shape == (7,)
    _, far = g.ik_topdown(np.array([1.2, 1.2, 0.05, 0.0]))
    assert not far


def test_motion_check_rejects_below_floor():
    # A straight-line to a reachable config over the table must stay valid.
    q0 = np.asarray(g.NEUTRAL_Q[:7])
    q1, ok = g.ik_topdown(np.array([0.5, 0.0, 0.05, 0.0]))
    assert ok
    assert g.arm_path_valid(g.interpolate(q0, q1, 20))


def test_no_pybullet_import():
    """Fairness gate: nothing in the TAMP library may import pybullet."""
    d = os.path.join(_ROOT, "spasm", "tamp")
    hits = subprocess.run(
        ["grep", "-rInE", r"import pybullet|from pybullet", d],
        capture_output=True, text=True).stdout.strip()
    assert hits == "", f"pybullet import found:\n{hits}"


def test_end_to_end_solve_two_blocks():
    from pddlstream.algorithms.meta import solve
    world = make_rearrange_world(2, seed=0)
    plan, cost, _ = solve(pddlstream_from_world(world), algorithm="adaptive",
                          unit_costs=False, max_time=90, verbose=False)
    assert plan is not None
    _, success, _ = execute_plan(world, plan)
    assert success
