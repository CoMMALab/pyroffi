"""Baseline (v1) pure-JAX ESDF query -- the differentiable Query-stage
building block for the voxel-grid world-collision backend.

Ported verbatim from the validated prototype in ``scratch/esdf_prototype.py``
(see that file's ARCHITECTURE NOTE for the full baseline-vs-faithful-cuRobo-V2
port plan: block-sparse TSDF + PBA + depth-camera fusion is the confirmed
endpoint; this is deliberately just the v1 baseline slice). Only the Query
stage lives here so far -- building the grid itself (seed -> JFA-propagate ->
distance, or eventually a depth-camera TSDF fusion pipeline) is still
scratch-only: callers construct an :class:`ESDFWorldGeom` from whatever
grid-building path they have (see ``scratch/esdf_prototype.py``'s
``build_esdf_from_boxes`` / ``compute_esdf_pipeline_jax`` for the validated
reference) and hand it to
``CUDADifferentiableSDFCollisionChecker.compute_world_collision_distance``.

Distance convention (matches the rest of pyroffi):
    positive  ->  separated
    negative  ->  penetration
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp


@dataclasses.dataclass
class ESDFWorldGeom:
    """Dense ESDF field, used as the ``world_geom`` argument to
    :meth:`CUDADifferentiableSDFCollisionChecker.compute_world_collision_distance`
    in place of a CollGeom batch of M discrete primitives.

    Deliberately NOT a CollGeom subclass: a dense field is one continuous
    scalar field, not a batch of M discrete obstacles, so it is special-cased
    in ``_cuda_collision.py``'s world-extraction ladders before the
    CollGeom-specific batch-axis normalization runs -- this type doesn't
    implement that interface (no ``get_batch_axes``/``transform``/etc).

    Grid layout matches :func:`esdf_query_jax`: ``grid[i, j, k]`` holds the
    signed distance at world point ``origin + [i, j, k] * voxel_size``.
    """

    grid: jax.Array
    """float32 [nx, ny, nz] signed distance at each voxel center."""
    origin: jax.Array
    """float32 [3] world position of voxel (0, 0, 0)'s center."""
    voxel_size: float
    """Scalar edge length of a (cubic) voxel."""


def esdf_query_jax(
    points: jax.Array,
    esdf_grid: jax.Array,
    origin: jax.Array,
    voxel_size: float,
) -> jax.Array:
    """Trilinearly sample a signed-distance grid at world-space points.

    Args:
        points:     float [*batch, 3] world-space query points.
        esdf_grid:  float [nx, ny, nz] signed distance at each voxel center
                    (negative = inside an obstacle).
        origin:     float [3] world position of voxel (0, 0, 0)'s center.
        voxel_size: scalar edge length of a (cubic) voxel.

    Returns:
        float [*batch] trilinearly interpolated signed distance.

    Differentiability: points outside the grid are clamped to the boundary
    voxel (edge-clamped extrapolation) so the query never index-errors and
    the gradient stays finite everywhere, including strictly inside a voxel
    cell where the interpolation is smooth. ``jnp.floor`` has zero gradient,
    so d(frac)/d(idx) = 1 exactly -- the standard trick that makes trilinear
    interpolation differentiable in the query point without any custom rule.

    This is the baseline (v1) Query-stage dispatch used by
    ``CUDADifferentiableSDFCollisionChecker`` when ``world_geom`` is an
    :class:`ESDFWorldGeom`: being pure JAX, ordinary autodiff flows through
    it directly, so no ``jax.custom_jvp``/forward-CUDA-kernel wrapping is
    needed the way it is for the Sphere/Capsule/Box/HalfSpace world types
    (see ``_cuda_collision.py::compute_world_collision_distance``). A
    forward-only CUDA/FFI trilinear kernel with a pure-JAX ``custom_jvp``
    tangent (mirroring that same pattern) is the faithful-V2 swap target for
    this function, not built here.
    """
    points = jnp.asarray(points, dtype=jnp.float32)
    origin = jnp.asarray(origin, dtype=jnp.float32)
    batch_shape = points.shape[:-1]
    pts = points.reshape(-1, 3)

    nx, ny, nz = esdf_grid.shape
    dims = jnp.array([nx, ny, nz], dtype=jnp.float32)

    # Continuous grid-index coordinates; clamp so the top corner (i0 + 1)
    # never runs off the grid.
    idx = (pts - origin[None, :]) / voxel_size
    idx = jnp.clip(idx, 0.0, dims - 1.0 - 1e-6)

    i0 = jnp.floor(idx).astype(jnp.int32)                    # (P, 3)
    frac = idx - i0.astype(jnp.float32)                      # (P, 3), in [0, 1)
    i1 = jnp.minimum(i0 + 1, jnp.array([nx - 1, ny - 1, nz - 1], dtype=jnp.int32))

    x0, y0, z0 = i0[:, 0], i0[:, 1], i0[:, 2]
    x1, y1, z1 = i1[:, 0], i1[:, 1], i1[:, 2]
    fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]

    def g(xi: jax.Array, yi: jax.Array, zi: jax.Array) -> jax.Array:
        return esdf_grid[xi, yi, zi]

    c000, c001 = g(x0, y0, z0), g(x0, y0, z1)
    c010, c011 = g(x0, y1, z0), g(x0, y1, z1)
    c100, c101 = g(x1, y0, z0), g(x1, y0, z1)
    c110, c111 = g(x1, y1, z0), g(x1, y1, z1)

    c00 = c000 * (1 - fx) + c100 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c11 = c011 * (1 - fx) + c111 * fx

    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy

    c = c0 * (1 - fz) + c1 * fz

    return c.reshape(batch_shape)
