"""Point signed-distance queries on collision primitives (``CollGeom.sdf``).

Exercises the differentiable point-SDF added to ``Sphere``/``Box``/``Capsule``/
``HalfSpace``: known-value checks, batched query points, ``vmap`` over batched
geometries, gradient finiteness on the surface, and consistency with the
existing sphere-vs-geometry pair distances (a query point is the ``radius -> 0``
limit of a sphere).

Run:
    pytest tests/test_geometry_sdf.py -q
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyroffi.collision import Box, Capsule, HalfSpace, Sphere
from pyroffi.collision import _geometry_pairs as gp


def test_sphere_sdf_known_values():
    s = Sphere.from_center_and_radius([1.0, 0.0, 0.0], 1.0)
    # 1 unit outside, on the surface, at the center.
    assert np.isclose(float(s.sdf(jnp.array([3.0, 0.0, 0.0]))), 1.0, atol=1e-5)
    assert np.isclose(float(s.sdf(jnp.array([2.0, 0.0, 0.0]))), 0.0, atol=1e-5)
    assert np.isclose(float(s.sdf(jnp.array([1.0, 0.0, 0.0]))), -1.0, atol=1e-5)


def test_box_sdf_known_values():
    b = Box.from_center_and_half_lengths([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    assert np.isclose(float(b.sdf(jnp.array([0.0, 0.0, 0.0]))), -1.0, atol=1e-5)
    assert np.isclose(float(b.sdf(jnp.array([2.0, 0.0, 0.0]))), 1.0, atol=1e-5)
    # Outside a corner: distance to (1,1,1) vertex.
    corner = jnp.array([2.0, 2.0, 2.0])
    assert np.isclose(float(b.sdf(corner)), float(jnp.sqrt(3.0)), atol=1e-5)


def test_capsule_sdf_known_values():
    # radius 0.5, full height 2.0 along +Z, at the origin.
    c = Capsule.from_radius_height(0.5, 2.0)
    # Past the +Z cap: 1 unit above the top hemisphere center (0,0,1), minus r.
    assert np.isclose(float(c.sdf(jnp.array([0.0, 0.0, 2.0]))), 0.5, atol=1e-5)
    # On the axis at the center -> inside by the radius.
    assert np.isclose(float(c.sdf(jnp.array([0.0, 0.0, 0.0]))), -0.5, atol=1e-5)
    # Radially out from the cylinder wall.
    assert np.isclose(float(c.sdf(jnp.array([1.0, 0.0, 0.0]))), 0.5, atol=1e-5)


def test_halfspace_sdf_known_values():
    p = HalfSpace.from_point_and_normal([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    assert np.isclose(float(p.sdf(jnp.array([0.0, 0.0, 3.0]))), 3.0, atol=1e-5)
    assert np.isclose(float(p.sdf(jnp.array([5.0, -2.0, -1.0]))), -1.0, atol=1e-5)


def test_sdf_batched_points():
    b = Box.from_center_and_half_lengths([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    pts = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    out = b.sdf(pts)
    assert out.shape == (3,)
    np.testing.assert_allclose(np.asarray(out), [-1.0, 1.0, 2.0], atol=1e-5)


def test_sdf_vmap_over_batched_geoms():
    # A batch of spheres, one query point, vmapped per-geom (the reconstruction
    # union-SDF access pattern).
    centers = jnp.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    radii = jnp.array([1.0, 2.0])
    spheres = Sphere.from_center_and_radius(centers, radii)
    x = jnp.array([2.0, 0.0, 0.0])
    dists = jax.vmap(lambda s: s.sdf(x))(spheres)
    np.testing.assert_allclose(np.asarray(dists), [1.0, 1.0], atol=1e-5)


def test_sdf_gradient_is_finite_on_surface():
    # The +eps under the sqrt must keep gradients finite at the degenerate
    # points (box interior, sphere center, capsule axis).
    b = Box.from_center_and_half_lengths([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    g = jax.grad(lambda x: b.sdf(x))(jnp.zeros(3))
    assert jnp.all(jnp.isfinite(g))
    s = Sphere.from_center_and_radius([0.0, 0.0, 0.0], 1.0)
    gs = jax.grad(lambda x: s.sdf(x))(jnp.zeros(3))
    assert jnp.all(jnp.isfinite(gs))


@pytest.mark.parametrize("radius", [0.0, 0.3, 1.0])
def test_sdf_matches_pair_distance_zero_radius_limit(radius):
    # A sphere-vs-geometry distance is exactly the geometry's point-SDF at the
    # sphere center, minus the sphere radius. Validates sdf() against the
    # existing pair implementations.
    box = Box.from_center_and_half_lengths([0.2, -0.1, 0.4], [0.5, 0.3, 0.7])
    hs = HalfSpace.from_point_and_normal([0.0, 0.0, 0.1], [0.1, 0.2, 1.0])
    probe = jnp.array([0.6, 0.4, 1.3])
    s = Sphere.from_center_and_radius(probe, radius)

    np.testing.assert_allclose(
        float(gp.box_sphere(box, s)), float(box.sdf(probe) - radius), atol=1e-5
    )
    np.testing.assert_allclose(
        float(gp.halfspace_sphere(hs, s)), float(hs.sdf(probe) - radius), atol=1e-5
    )
