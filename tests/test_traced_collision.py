"""Traced collision checkers (compiled against cricket FK) must agree with the stock kernels."""

import shutil

import jax
import numpy as np
import pytest
import yourdfpy

pytest.importorskip("cricket")
if shutil.which("nvcc") is None or jax.default_backend() != "gpu":
    pytest.skip("traced kernels need nvcc and a GPU", allow_module_level=True)

from pyroffi import Robot
from pyroffi._robot_srdf_parser import read_disabled_collisions_from_srdf
from pyroffi.collision import RobotCollisionSpherized
from pyroffi.collision._cuda_collision import FusedCUDACollisionChecker


@pytest.fixture(scope="module")
def panda():
    urdf = yourdfpy.URDF.load("resources/panda/panda_spherized.urdf")
    robot = Robot.from_urdf(urdf)
    ignore = tuple((p["link1"], p["link2"])
                   for p in read_disabled_collisions_from_srdf("resources/panda/panda.srdf"))
    model = RobotCollisionSpherized.from_urdf(urdf, user_ignore_pairs=ignore)
    cfg = jax.random.uniform(jax.random.PRNGKey(0), (2048, robot.joints.num_actuated_joints),
                             minval=robot.joints.lower_limits, maxval=robot.joints.upper_limits)
    return robot, model, cfg


def test_fused_self_collision_matches_stock(panda):
    robot, model, cfg = panda
    stock, traced = (FusedCUDACollisionChecker(robot, model, traced=t)
                     .compute_self_collision_and_floor(robot, cfg) for t in (False, True))
    for a, b in zip(stock, traced):  # (pair distances, lowest sphere point)
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-5)



def test_fused_world_collision_matches_stock(panda):
    import jax.numpy as jnp

    from pyroffi.collision import Sphere

    robot, model, cfg = panda
    world = Sphere.from_center_and_radius(
        jnp.array([[0.45, 0.1 * k, 0.35] for k in range(-4, 5)]), jnp.full((9,), 0.08))
    stock, traced = (np.asarray(FusedCUDACollisionChecker(robot, model, traced=t)
                                .compute_world_collision_distance(robot, cfg, world))
                     for t in (False, True))
    np.testing.assert_allclose(stock, traced, atol=1e-5)


def test_robogpu_verdicts_match_stock(panda):
    import jax.numpy as jnp

    from pyroffi.collision import RoboGPUCollisionChecker, Sphere

    robot, model, cfg = panda
    world = Sphere.from_center_and_radius(
        jnp.array([[0.45, 0.1 * k, 0.35] for k in range(-4, 5)]), jnp.full((9,), 0.08))
    rng = np.random.default_rng(0)
    cloud = jnp.array(np.c_[rng.uniform(0.3, 0.9, 5000), rng.uniform(-0.5, 0.5, 5000),
                            np.full(5000, 0.05)].astype(np.float32))
    verdicts = []
    for traced in (False, True):
        try:
            checker = RoboGPUCollisionChecker(model, traced=traced)
        except RuntimeError as exc:  # library or OptiX SDK missing
            pytest.skip(str(exc))
        checker.set_world(world, point_cloud=cloud, r_env=0.01)
        verdicts.append(np.asarray(checker.check_collision_free(robot, cfg)))
    assert 0.0 < verdicts[0].mean() < 1.0  # both outcomes exercised
    np.testing.assert_array_equal(verdicts[0], verdicts[1])
