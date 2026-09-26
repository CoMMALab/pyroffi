"""Fused collision gradients come from the GPU Jacobian kernels and must match pure-JAX autodiff."""

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
from pyroffi.collision import RobotCollisionSpherized, Sphere
from pyroffi.collision._cuda_collision import FusedCUDACollisionChecker


@pytest.fixture(scope="module")
def panda():
    urdf = yourdfpy.URDF.load("resources/panda/panda_spherized.urdf")
    robot = Robot.from_urdf(urdf)
    ignore = tuple((p["link1"], p["link2"])
                   for p in read_disabled_collisions_from_srdf("resources/panda/panda.srdf"))
    model = RobotCollisionSpherized.from_urdf(urdf, user_ignore_pairs=ignore)
    lo, hi = np.asarray(robot.joints.lower_limits), np.asarray(robot.joints.upper_limits)
    q = np.random.default_rng(3).uniform(lo, hi, (32, lo.size)).astype(np.float32)
    return robot, model, q


def _finite_sum(x):
    return jnp.sum(jnp.where(jnp.isfinite(x) & (jnp.abs(x) < 1e8), x, 0.0))


# Reference distances from live spheres only, with finite sentinels: the model's own
# distance functions return +inf for pairs without spheres, and masking an inf output with
# jnp.where after the fact gives NaN gradients (0 * inf).
def _spheres(robot, model, q):
    coll = model.at_config(robot, q)  # [S, N]
    return coll.pose.translation(), coll.radius


def _self_ref(robot, model, q):
    c, r = _spheres(robot, model, q)
    ci, cj = c[:, model.active_idx_i], c[:, model.active_idx_j]            # [S, P, 3]
    ri, rj = r[:, model.active_idx_i], r[:, model.active_idx_j]            # [S, P]
    d = jnp.linalg.norm(ci[:, None] - cj[None], axis=-1) - ri[:, None] - rj[None]  # [S, S, P]
    live = (ri[:, None] > 0) & (rj[None] > 0)
    return jnp.min(jnp.where(live, d, 1e9), axis=(0, 1))


def _world_ref(robot, model, q, centers, radii):
    c, r = _spheres(robot, model, q)
    d = jnp.linalg.norm(c[..., None, :] - centers, axis=-1) - r[..., None] - radii   # [S, N, M]
    return jnp.min(jnp.where((r > 0)[..., None], d, 1e9), axis=0)


@pytest.mark.parametrize("traced", [False, True])
def test_self_collision_gradient(panda, traced):
    robot, model, q = panda
    checker = FusedCUDACollisionChecker(robot, model, traced=traced)
    ref = jax.vmap(lambda x: _self_ref(robot, model, x))
    got = jax.grad(lambda c: _finite_sum(checker.compute_self_collision_distance(robot, c)))(q)
    want = jax.grad(lambda c: _finite_sum(ref(c)))(q)
    # Summing ~40 pair gradients in float32 (and a few near-tied witness pairs that GPU and
    # JAX FK resolve differently) leaves ~1e-3 on entries of magnitude 2-4.
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=2e-3, atol=2e-3)

    # Forward mode agrees with reverse mode (the tangent map is J @ dq).
    v = jnp.asarray(np.random.default_rng(4).normal(size=q.shape), jnp.float32)
    loss = lambda c: _finite_sum(checker.compute_self_collision_distance(robot, c))
    _, tangent = jax.jvp(loss, (q,), (v,))
    np.testing.assert_allclose(float(tangent), float(jnp.sum(got * v)), rtol=1e-4)


@pytest.mark.parametrize("traced", [False, True])
def test_world_collision_gradient(panda, traced):
    robot, model, q = panda
    centers, radii = jnp.array([[0.45, 0.0, 0.4], [0.2, 0.35, 0.3]]), jnp.array([0.12, 0.1])
    world = Sphere.from_center_and_radius(centers, radii)
    checker = FusedCUDACollisionChecker(robot, model, traced=traced)
    ref = jax.vmap(lambda x: _world_ref(robot, model, x, centers, radii))
    got = jax.grad(lambda c: _finite_sum(checker.compute_world_collision_distance(robot, c, world)))(q)
    want = jax.grad(lambda c: _finite_sum(ref(c)))(q)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=2e-3, atol=2e-4)
