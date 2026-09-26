"""SCO trajopt: the traced build optimizes as well as stock, and runs robots stock cannot."""

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
from pyroffi.cuda_kernels.trajopt._sco_trajopt_cuda import prepare_sco_trajopt_cuda, sco_trajopt_cuda_prepared
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


def test_traced_matches_stock_cost():
    robot, coll, start, goal, init = _problem("resources/panda/panda_spherized.urdf", "resources/panda/panda.srdf")
    costs = {}
    for traced in (False, True):
        prep = prepare_sco_trajopt_cuda(robot, coll, WORLD, traced=traced)
        costs[traced] = np.asarray(sco_trajopt_cuda_prepared(init, start, goal, prep, CFG)[1])
    # Float-level differences steer individual runs to different (equally good) optima.
    np.testing.assert_allclose(np.median(costs[True]), np.median(costs[False]), rtol=0.05)


def test_stock_refuses_over_capacity_and_traced_runs():
    """G1 with hands: 1214 self pairs and 10 spheres per link exceed the stock build's arrays."""
    robot, coll, start, goal, init = _problem(
        "resources/g1_description/g1_29dof_with_hand_rev_1_0_spherized.urdf",
        "resources/g1_description/g1_29dof_with_hand.srdf", batch=4)
    stock = prepare_sco_trajopt_cuda(robot, coll, WORLD, traced=False)
    with pytest.raises(Exception, match="exceeds this build's capacity"):
        jax.block_until_ready(sco_trajopt_cuda_prepared(init, start, goal, stock, CFG))
    traced = prepare_sco_trajopt_cuda(robot, coll, WORLD, traced=True)
    trajs = np.asarray(sco_trajopt_cuda_prepared(init, start, goal, traced, CFG)[2])
    assert np.isfinite(trajs).all()
