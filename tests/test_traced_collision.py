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

