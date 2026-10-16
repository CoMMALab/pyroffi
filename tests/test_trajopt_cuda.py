"""CUDA trajopt (GRiM kernels): table and traced builds agree, and high-DOF robots run."""

import shutil

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yourdfpy

pytest.importorskip("cricket")
if shutil.which("nvcc") is None or jax.default_backend() != "gpu":
    pytest.skip("traced kernels need nvcc and a GPU", allow_module_level=True)

from pyroffi import Robot
from pyroffi._robot_srdf_parser import read_disabled_collisions_from_srdf
from pyroffi.collision import RobotCollisionSpherized, Sphere
from pyroffi.optimization_engines._trajopt_cuda import prepare, run
from pyroffi.optimization_engines._sco_optimization import ScoTrajOptConfig

CFG = ScoTrajOptConfig(n_outer_iters=10, n_inner_iters=30, m_lbfgs=6, w_smooth=1.0, w_vel=1.0, w_acc=0.5,
                       w_jerk=0.1, w_collision=10.0, w_collision_max=100.0, penalty_scale=3.0,
                       collision_margin=0.02, w_trust=0.5, w_limits=1.0)
WORLD = (Sphere.from_center_and_radius(jnp.array([[0.45, 0.0, 0.4], [0.3, 0.3, 0.2]]), jnp.array([0.1, 0.08])),)


def _problem(urdf_path, srdf_path, batch=16, T=32):
    urdf = yourdfpy.URDF.load(urdf_path)
    robot = Robot.from_urdf(urdf)
    ignore = tuple((p["link1"], p["link2"]) for p in read_disabled_collisions_from_srdf(srdf_path))
    coll = RobotCollisionSpherized.from_urdf(urdf, user_ignore_pairs=ignore)
    lo, hi = robot.joints.lower_limits, robot.joints.upper_limits
    start = jnp.clip(robot.default_cfg, lo, hi)
    goal = jnp.clip(start + 0.3 * (hi - lo) * jnp.sin(jnp.arange(lo.shape[0])), lo, hi)
    alpha = jnp.linspace(0, 1, T)[None, :, None]
    init = start + alpha * (goal - start) + 0.05 * jnp.sin(jnp.pi * alpha) * jax.random.normal(
        jax.random.PRNGKey(0), (batch, T, lo.shape[0]))
    return robot, coll, start, goal, init


def test_traced_matches_table_cost():
    robot, coll, start, goal, init = _problem("resources/panda/panda_spherized.urdf", "resources/panda/panda.srdf")
    costs = {}
    for traced in (False, True):
        prep = prepare(robot, coll, WORLD, traced=traced)
        costs[traced] = np.asarray(run("sco", prep, init, start, goal, CFG)[1])
    # Float-level differences steer individual runs to different (equally good) optima.
    np.testing.assert_allclose(np.median(costs[True]), np.median(costs[False]), rtol=0.05)


@pytest.mark.parametrize("traced", [False, True])
def test_high_dof_robot_runs(traced):
    """G1 with hands (43 DOF, 1214 self pairs, 10 spheres per link) overflowed the old
    runtime-table build's arrays; per-robot builds are sized to the robot."""
    robot, coll, start, goal, init = _problem(
        "resources/g1_description/g1_29dof_with_hand_rev_1_0_spherized.urdf",
        "resources/g1_description/g1_29dof_with_hand.srdf", batch=4)
    trajs = np.asarray(run("sco", prepare(robot, coll, WORLD, traced=traced), init, start, goal,
                           CFG)[2])
    assert np.isfinite(trajs).all()


def test_collision_cost_matches_jax():
    """Zero iterations report the cost of the initial trajectories: the kernel's collision
    geometry (sphere rows posed by their links' parent joints) must match the JAX model's."""
    import dataclasses

    from pyroffi.optimization_engines._sco_optimization import sco_trajopt

    robot, coll, start, goal, init = _problem("resources/panda/panda_spherized.urdf", "resources/panda/panda.srdf", batch=8)
    mid = robot.forward_kinematics(init[0, 16])[robot.links.names.index("panda_link4")][4:]
    world = (Sphere.from_center_and_radius(mid[None], jnp.array([0.12])),)
    cfg0 = dataclasses.replace(CFG, n_outer_iters=0, n_inner_iters=0)
    jax_cost = np.asarray(sco_trajopt(init, start, goal, robot, coll, world, cfg0, use_cuda=False)[1])
    for traced in (False, True):
        prep = prepare(robot, coll, world, traced=traced)
        np.testing.assert_allclose(np.asarray(run("sco", prep, init, start, goal, cfg0)[1]),
                                   jax_cost, rtol=5e-3)
