"""Fused FK + ESDF world collision (stock and traced) against the pure-JAX ESDF query."""

import shutil

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yourdfpy

if shutil.which("nvcc") is None or jax.default_backend() != "gpu":
    pytest.skip("fused kernels need a GPU", allow_module_level=True)

from pyroffi import Robot
from pyroffi._robot_srdf_parser import read_disabled_collisions_from_srdf
from pyroffi.collision import RobotCollisionSpherized
from pyroffi.collision._cuda_collision import FusedCUDACollisionChecker
from pyroffi.collision._esdf import ESDFWorldGeom, esdf_query_jax


def _box_grid(origin, voxel, shape, centers, halves):
    """Analytic signed distance to the nearest axis-aligned box at every voxel centre."""
    ii, jj, kk = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
    pts = origin + np.stack([ii, jj, kk], axis=-1) * voxel
    best = np.full(shape, np.inf)
    for c, h in zip(centers, halves):
        d = np.abs(pts - c) - h
        best = np.minimum(best, np.linalg.norm(np.maximum(d, 0.0), axis=-1) + np.minimum(d.max(-1), 0.0))
    return best.astype(np.float32)


@pytest.fixture(scope="module")
def scene():
    urdf = yourdfpy.URDF.load("resources/panda/panda_spherized.urdf")
    robot = Robot.from_urdf(urdf)
    ignore = tuple((p["link1"], p["link2"])
                   for p in read_disabled_collisions_from_srdf("resources/panda/panda.srdf"))
    model = RobotCollisionSpherized.from_urdf(urdf, user_ignore_pairs=ignore)
    origin = np.array([-0.6, -0.8, -0.1], np.float32)
    voxel, shape = 0.02, (70, 80, 70)
    grid = _box_grid(origin, voxel, shape, np.array([[0.5, 0.0, 0.3], [0.2, 0.45, 0.2]]),
                     np.array([[0.1, 0.25, 0.3], [0.15, 0.08, 0.2]]))
    esdf = ESDFWorldGeom(grid=jnp.asarray(grid), origin=jnp.asarray(origin), voxel_size=voxel)
    lo, hi = np.asarray(robot.joints.lower_limits), np.asarray(robot.joints.upper_limits)
    q = np.random.default_rng(0).uniform(lo, hi, (256, lo.size)).astype(np.float32)
    return robot, model, esdf, q


def _reference(robot, model, esdf, q):
    def one(c):
        coll = model.at_config(robot, c)  # [S, N]
        d = esdf_query_jax(coll.pose.translation(), esdf.grid, esdf.origin, esdf.voxel_size) - coll.radius
        return jnp.min(jnp.where(coll.radius > 0.0, d, jnp.inf), axis=0)
    return jax.vmap(one)(q)


@pytest.mark.parametrize("traced", [False, True])
def test_matches_jax_and_differentiates(scene, traced):
    robot, model, esdf, q = scene
    checker = FusedCUDACollisionChecker(robot, model, traced=traced)
    out = np.asarray(checker.compute_world_collision_distance(robot, q, esdf))
    ref = np.asarray(_reference(robot, model, esdf, q))
    assert out.shape == (len(q), ref.shape[1], 1)
    finite = np.isfinite(ref)
    np.testing.assert_allclose(out[..., 0][finite], ref[finite], atol=1e-4)
    assert (out[..., 0] < 0).any() and (out[..., 0] > 0).any()  # the scene has hits and misses

    loss = lambda c: jnp.sum(jnp.where(jnp.isfinite(o := checker.compute_world_collision_distance(robot, c, esdf)), o, 0.0))
    ref_loss = lambda c: jnp.sum(jnp.where(jnp.isfinite(r := _reference(robot, model, esdf, c)), r, 0.0))
    np.testing.assert_allclose(np.asarray(jax.grad(loss)(q[:16])), np.asarray(jax.grad(ref_loss)(q[:16])),
                               rtol=1e-4, atol=1e-5)


def test_traced_rejects_other_grid_shape(scene):
    robot, model, esdf, q = scene
    checker = FusedCUDACollisionChecker(robot, model, traced=True)
    checker.compute_world_collision_distance(robot, q[:4], esdf)
    target = checker._traced_targets((0, 0, 0, 0), (*esdf.grid.shape, esdf.voxel_size))[2]
    from pyroffi.cuda_kernels.collision._fused_self_collision_ffi import fused_world_esdf

    with pytest.raises(Exception, match="grid shape or voxel size"):
        jax.block_until_ready(fused_world_esdf(q[:4], checker._robot_buffers, checker._static,
                                               esdf.grid[:-1], esdf.origin, esdf.voxel_size, target))
