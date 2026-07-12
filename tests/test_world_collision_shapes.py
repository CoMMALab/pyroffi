"""Regression test for the vmap-axis bug in compute_world_collision_distance.

Both ``RobotCollision`` (capsule model) and ``RobotCollisionSpherized`` used
to ``vmap`` the robot-links axis with a back-relative ``in_axes=-2``. That is
wrong whenever a leaf's trailing feature-dim count differs from 2 (e.g. a
Sphere's ``radius`` leaf has 0 trailing feature dims, so -2 pointed at the
wrong axis), which raised a ``ValueError`` or silently produced wrong-shaped
output. The fix maps over the link axis counted from the *front* of the
batch axes (batch axes on CollGeom are always leading, feature dims trail),
which is leaf-independent.

Run:
    python tests/test_world_collision_shapes.py
"""

from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import yourdfpy

import pyroffi as pk
from pyroffi.collision import RobotCollision, RobotCollisionSpherized, Sphere

RESOURCE_ROOT = pathlib.Path(__file__).resolve().parent.parent / "resources"
SPHERIZED_URDF = RESOURCE_ROOT / "panda" / "panda_spherized.urdf"
CAPSULE_URDF = RESOURCE_ROOT / "panda" / "panda.urdf"


def _load_spherized():
    urdf = yourdfpy.URDF.load(str(SPHERIZED_URDF))
    robot = pk.Robot.from_urdf(urdf)
    srdf_path = RESOURCE_ROOT / "panda" / "panda.srdf"
    coll = RobotCollisionSpherized.from_urdf(urdf, srdf_path=str(srdf_path))
    return robot, coll


def _load_capsule():
    urdf = yourdfpy.URDF.load(str(CAPSULE_URDF))
    robot = pk.Robot.from_urdf(urdf)
    coll = RobotCollision.from_urdf(urdf)
    return robot, coll


def _mid_cfg(robot):
    return (robot.joints.lower_limits + robot.joints.upper_limits) / 2.0


def test_spherized_single_config_single_world_obj():
    robot, coll = _load_spherized()
    q = _mid_cfg(robot)
    s = Sphere.from_center_and_radius(
        center=jnp.array([[0.5, 0.0, 0.3]]), radius=jnp.array([0.1])
    )
    out = coll.compute_world_collision_distance(robot, q, s)
    assert out.shape == (coll.num_links, 1)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_spherized_single_config_multi_world_obj():
    robot, coll = _load_spherized()
    q = _mid_cfg(robot)
    s = Sphere.from_center_and_radius(
        center=jnp.array([[0.5, 0.0, 0.3], [2.0, 2.0, 2.0], [-1.0, 0.0, 0.5]]),
        radius=jnp.array([0.1, 0.05, 0.2]),
    )
    out = coll.compute_world_collision_distance(robot, q, s)
    assert out.shape == (coll.num_links, 3)


def test_spherized_far_sphere_matches_hand_computed_separation():
    """A sphere far from the robot should report ~separation distance for
    every link (center distance minus the world sphere's own radius minus
    each link's largest primitive radius -- so we just sanity check the
    reported distance is close to the raw center-to-origin distance, since
    link/primitive radii are small relative to the placement distance)."""
    robot, coll = _load_spherized()
    q = _mid_cfg(robot)
    far_center = jnp.array([[10.0, 0.0, 0.0]])
    world_radius = 0.1
    s = Sphere.from_center_and_radius(
        center=far_center, radius=jnp.array([world_radius])
    )
    out = coll.compute_world_collision_distance(robot, q, s)
    assert out.shape == (coll.num_links, 1)
    # Links with no real collision spheres (e.g. some fixed/virtual frames)
    # report the padding sentinel (1e9); exclude those from the range check.
    real = out[out < 1e8]
    assert real.size > 0
    # Every real link is within ~1.5m of the robot base (panda reach), so the
    # closest link should read a distance within a couple meters of the
    # nominal 10m placement distance, and definitely not tiny/negative.
    assert bool(jnp.all(real > 5.0))
    assert bool(jnp.all(real < 15.0))


def test_spherized_vmap_over_config_from_outside():
    robot, coll = _load_spherized()
    q = _mid_cfg(robot)
    qs = jnp.stack([q, q, q])
    s = Sphere.from_center_and_radius(
        center=jnp.array([[0.5, 0.0, 0.3]]), radius=jnp.array([0.1])
    )
    out = jax.vmap(lambda qq: coll.compute_world_collision_distance(robot, qq, s))(qs)
    assert out.shape == (3, coll.num_links, 1)
    # Same cfg repeated -> identical results.
    np.testing.assert_allclose(np.asarray(out[0]), np.asarray(out[1]))


def test_capsule_single_config_single_world_obj():
    robot, coll = _load_capsule()
    q = _mid_cfg(robot)
    s = Sphere.from_center_and_radius(
        center=jnp.array([[0.5, 0.0, 0.3]]), radius=jnp.array([0.1])
    )
    out = coll.compute_world_collision_distance(robot, q, s)
    assert out.shape == (coll.num_links, 1)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_capsule_multi_world_obj():
    robot, coll = _load_capsule()
    q = _mid_cfg(robot)
    s = Sphere.from_center_and_radius(
        center=jnp.array([[0.5, 0.0, 0.3], [2.0, 2.0, 2.0]]),
        radius=jnp.array([0.1, 0.05]),
    )
    out = coll.compute_world_collision_distance(robot, q, s)
    assert out.shape == (coll.num_links, 2)


if __name__ == "__main__":
    test_spherized_single_config_single_world_obj()
    test_spherized_single_config_multi_world_obj()
    test_spherized_far_sphere_matches_hand_computed_separation()
    test_spherized_vmap_over_config_from_outside()
    test_capsule_single_config_single_world_obj()
    test_capsule_multi_world_obj()
    print("All world-collision shape tests passed.")
