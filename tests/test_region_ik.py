"""Box-region IK samplers (GRiM kernels) put the end-effector in the box."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yourdfpy

if jax.default_backend() != "gpu":
    pytest.skip("region IK kernels need a GPU", allow_module_level=True)

from pyroffi import Robot
from pyroffi.optimization_engines._region_ik import (
    brownian_motion_sample_box_region_cuda,
    direct_sample_box_region_cuda,
    hit_and_run_sample_box_region_cuda,
    svgd_sample_box_region_cuda,
)

SAMPLERS = {
    "brownian": brownian_motion_sample_box_region_cuda,
    "hit_and_run": hit_and_run_sample_box_region_cuda,
    "svgd": svgd_sample_box_region_cuda,
    "direct": direct_sample_box_region_cuda,
}


@pytest.fixture(scope="module")
def panda():
    robot = Robot.from_urdf(yourdfpy.URDF.load("resources/panda/panda_spherized.urdf"))
    return robot, robot.links.names.index("panda_hand")


@pytest.mark.parametrize("name", list(SAMPLERS))
def test_samples_land_in_box(panda, name):
    robot, ee = panda
    lo, hi = jnp.array([0.35, -0.15, 0.3]), jnp.array([0.55, 0.15, 0.5])
    cfgs, ee_points, _targets, _errs = SAMPLERS[name](
        robot, ee, jax.random.PRNGKey(0), robot.default_cfg, lo, hi, num_samples=256)
    fk = np.asarray(jax.vmap(lambda q: robot.forward_kinematics(q)[ee, 4:])(cfgs))
    np.testing.assert_allclose(fk, np.asarray(ee_points), atol=1e-3)
    inside = np.all((fk >= np.asarray(lo) - 1e-3) & (fk <= np.asarray(hi) + 1e-3), axis=1)
    # Measured 1.0 for all four (the restarts' winner selection absorbs per-seed misses).
    assert inside.mean() >= 0.95, inside.mean()
