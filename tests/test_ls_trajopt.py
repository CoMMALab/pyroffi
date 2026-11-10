"""LS trajopt: the cudals path (``use_cuda=True``) against the JAX solver on the same scene."""

import shutil

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yourdfpy

if shutil.which("nvcc") is None or jax.default_backend() != "gpu":
    pytest.skip("the cudals path needs nvcc and a GPU", allow_module_level=True)

import pyroffi as pk
from pyroffi._robot_srdf_parser import read_disabled_collisions_from_srdf
from pyroffi.collision import Sphere
from pyroffi.optimization_engines import LsTrajOptConfig, ls_trajopt
from pyroffi.optimization_engines._ls_trajopt_optimization import _eval_cost


@pytest.fixture(scope="module")
def scene():
    urdf = yourdfpy.URDF.load("resources/panda/panda_spherized.urdf", mesh_dir="resources/panda/meshes")
    robot = pk.Robot.from_urdf(urdf)
    ignore = tuple((p["link1"], p["link2"]) for p in read_disabled_collisions_from_srdf("resources/panda/panda.srdf"))
    coll = pk.collision.RobotCollisionSpherized.from_urdf(urdf, user_ignore_pairs=ignore)
    lo, hi = robot.joints.lower_limits, robot.joints.upper_limits
    T, B = 32, 16
    start = jnp.clip(robot.default_cfg, lo, hi)
    goal = jnp.clip(start + 0.3 * (hi - lo) * jnp.sin(jnp.arange(lo.shape[0])), lo, hi)
    alpha = jnp.linspace(0, 1, T)[None, :, None]
    init = start + alpha * (goal - start) + 0.05 * jnp.sin(jnp.pi * alpha) * jax.random.normal(
        jax.random.PRNGKey(0), (B, T, lo.shape[0]))
    mid = robot.forward_kinematics(init[0, T // 2])[robot.links.names.index("panda_link4")][4:]
    world = (Sphere.from_center_and_radius(mid[None], jnp.array([0.12])),)
    return robot, coll, world, init, start, goal


def test_cudals_matches_jax(scene):
    robot, coll, world, init, start, goal = scene
    cfg = LsTrajOptConfig()
    _, c_cuda, t_cuda = ls_trajopt(init, start, goal, robot, coll, world, cfg, use_cuda=True)
    _, c_jax, _ = ls_trajopt(init, start, goal, robot, coll, world, cfg, use_cuda=False)
    t_cuda = np.asarray(t_cuda)
    assert np.array_equal(t_cuda[:, 0], np.broadcast_to(np.asarray(start), t_cuda[:, 0].shape))
    assert np.array_equal(t_cuda[:, -1], np.broadcast_to(np.asarray(goal), t_cuda[:, -1].shape))
    lo, hi = robot.joints.lower_limits, robot.joints.upper_limits
    c0 = jax.vmap(lambda t: _eval_cost(t, lo, hi, robot, coll, world, cfg))(
        init.at[:, 0].set(start).at[:, -1].set(goal))
    assert np.all(np.asarray(c_cuda) < np.asarray(c0)), "cudals did not lower the cost"
    assert float(np.mean(c_cuda)) < 1.1 * float(np.mean(c_jax)), (np.mean(c_cuda), np.mean(c_jax))


def test_cudals_deterministic(scene):
    robot, coll, world, init, start, goal = scene
    a = ls_trajopt(init, start, goal, robot, coll, world, use_cuda=True)
    b = ls_trajopt(init, start, goal, robot, coll, world, use_cuda=True)
    assert all(np.array_equal(np.asarray(u), np.asarray(v)) for u, v in zip(a, b))
