"""Traced IK kernels (compiled against cricket kinematics) must solve what the stock kernels solve."""

import shutil

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pytest
import yourdfpy

pytest.importorskip("cricket")
if shutil.which("nvcc") is None or jax.default_backend() != "gpu":
    pytest.skip("traced kernels need nvcc and a GPU", allow_module_level=True)

from pyroffi import Robot
from pyroffi.optimization_engines._ls_ik import ls_ik_solve_cuda_batch
from pyroffi.optimization_engines._sqp_ik import sqp_ik_solve_cuda, sqp_ik_solve_cuda_batch

BATCH_SOLVERS = [sqp_ik_solve_cuda_batch, ls_ik_solve_cuda_batch]


@pytest.fixture(scope="module")
def panda():
    robot = Robot.from_urdf(yourdfpy.URDF.load("resources/panda/panda_spherized.urdf"))
    return robot, robot.links.names.index("panda_hand")


def _pose_errors(robot, link, q, targets):
    delta = targets.inverse() @ jaxlie.SE3(robot.forward_kinematics(q)[..., link, :])
    return (np.asarray(jnp.linalg.norm(delta.translation(), axis=-1)),
            np.asarray(jnp.linalg.norm(delta.rotation().log(), axis=-1)))


@pytest.mark.parametrize("solve", BATCH_SOLVERS, ids=["sqp", "ls"])
def test_batch_matches_stock(panda, solve):
    robot, link = panda
    key = jax.random.PRNGKey(0)
    q_true = jax.random.uniform(key, (64, robot.joints.num_actuated_joints),
                                minval=robot.joints.lower_limits, maxval=robot.joints.upper_limits)
    targets = jaxlie.SE3(robot.forward_kinematics(q_true)[:, link])
    prev = jnp.broadcast_to(robot.default_cfg, q_true.shape)

    for traced in (False, True):
        q = solve(robot, (link,), targets, key, prev, traced=traced)
        pos, rot = _pose_errors(robot, link, q, targets)
        assert np.mean((pos < 1e-3) & (rot < 1e-2)) == 1.0, f"traced={traced}"


def test_single_and_multi_ee(panda):
    robot, link = panda
    target = jaxlie.SE3(robot.forward_kinematics(robot.default_cfg)[link])
    q = sqp_ik_solve_cuda(robot, (link,), target, jax.random.PRNGKey(1), robot.default_cfg, traced=True)
    pos, rot = _pose_errors(robot, link, q, target)
    assert pos < 1e-3 and rot < 1e-2

    with pytest.raises(NotImplementedError):
        sqp_ik_solve_cuda(robot, (link, link), (target, target), jax.random.PRNGKey(1),
                          robot.default_cfg, traced=True)


@pytest.mark.parametrize("solve", BATCH_SOLVERS, ids=["sqp", "ls"])
def test_in_kernel_collision_matches_stock(panda, solve):
    """The traced build bakes the collision tables in; it must stay as collision-free as stock."""
    from pyroffi._robot_srdf_parser import read_disabled_collisions_from_srdf
    from pyroffi.collision import RobotCollisionSpherized, Sphere, collide

    robot, link = panda
    urdf = yourdfpy.URDF.load("resources/panda/panda_spherized.urdf")
    ignore = tuple((p["link1"], p["link2"])
                   for p in read_disabled_collisions_from_srdf("resources/panda/panda.srdf"))
    coll = RobotCollisionSpherized.from_urdf(urdf, user_ignore_pairs=ignore)
    obstacle = Sphere.from_center_and_radius(jnp.array([0.45, 0.0, 0.35]), jnp.array(0.12))

    key = jax.random.PRNGKey(3)
    q_true = jax.random.uniform(key, (64, robot.joints.num_actuated_joints),
                                minval=robot.joints.lower_limits, maxval=robot.joints.upper_limits)
    targets = jaxlie.SE3(robot.forward_kinematics(q_true)[:, link])
    prev = jnp.broadcast_to(robot.default_cfg, q_true.shape)

    def clear_fraction(q):
        d = jax.vmap(lambda c: jnp.min(collide(coll.at_config(robot, c), obstacle.broadcast_to((1,)))))(q)
        return float(jnp.mean(d > 0))

    fractions = [
        clear_fraction(solve(
            robot, (link,), targets, key, prev, traced=traced, collision_free=True,
            collision_checker=coll, collision_world=[obstacle], constraint_refine_iters=0))
        for traced in (False, True)
    ]
    assert fractions[1] >= fractions[0] - 0.05, fractions
