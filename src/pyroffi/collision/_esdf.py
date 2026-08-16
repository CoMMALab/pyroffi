"""Pure-JAX + CUDA voxel-grid ESDF pipeline -- the world-collision backend
for dense signed-distance fields, as an alternative to the discrete
Sphere/Capsule/Box/HalfSpace primitive path in ``_cuda_collision.py``.

Ported from the validated prototype in ``scratch/esdf_prototype.py`` (see
that file's ARCHITECTURE NOTE for the full baseline-vs-faithful-cuRobo-V2
port plan: block-sparse TSDF + PBA + depth-camera fusion is the confirmed
endpoint). This module currently covers:

  Query stage      -- esdf_query_jax: pure-JAX trilinear sample, unchanged
                       since the baseline (v1) pass. Differentiable w.r.t.
                       the query POINT with no FFI/custom_jvp involved.
  Seed stage       -- seed_sites_from_sdf_jax: surface-band threshold.
  Propagate stage  -- SELECTABLE via edt_solver: "jfa" (pure-JAX Jump
                       Flooding, O(log N) approximate passes) or "pba"
                       (CUDA Parallel Banding Algorithm, exact, 5 launches).
                       This is the ONE seam Piece 2 of the faithful-V2 port
                       touches -- seed/distance/query are all shared,
                       solver-agnostic code, exactly matching cuRobo V2's
                       own design (it also picks edt_solver via a config
                       string, both solvers sharing identical seed/distance
                       kernels).
  Distance stage   -- signed_distance_from_sites_jax: geometric distance to
                       the nearest propagated site, sign read from sdf_grid.

Grid CONSTRUCTION (seed -> propagate -> distance) is treated as
NONDIFFERENTIABLE throughout -- deliberately, not as an oversight. See
pba_propagate_jax's docstring below for the specific reasoning and how it
was verified, not just assumed.

Distance convention (matches the rest of pyroffi):
    positive  ->  separated
    negative  ->  penetration
"""

from __future__ import annotations

import dataclasses
import math
from typing import Callable, Optional

import jax
import jax.numpy as jnp

from ..cuda_kernels.esdf._pba_cuda_ffi import pba_propagate_sites_raw


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


# ============================================================================
# Grid CONSTRUCTION: seed -> propagate -> distance.
#
# Each stage below is a standalone function, matching the seams cuRobo V2
# itself treats as swappable (it picks seeding_method / edt_solver via
# config strings; build_esdf_pipeline below picks edt_solver the same way).
# Piece 1 (block-sparse TSDF storage, not yet built) will eventually replace
# seed_sites_from_sdf_jax with a hash-table-backed seed kernel without
# touching the propagate or distance stages.
# ============================================================================


def seed_sites_from_sdf_jax(
    sdf_grid: jax.Array,
    voxel_size: float,
    surface_band_voxels: float = 0.9,
) -> jax.Array:
    """Mark voxels within ``surface_band_voxels`` of the zero-crossing as sites.

    SIMPLIFIED vs. cuRobo V2's ``seed_esdf_sites_*`` kernels: only the
    surface-band criterion (``|sdf| <= surface_band_voxels * voxel_size``,
    matching cuRobo's own ``0.9`` default) is applied. V2 additionally seeds
    voxels past the truncation boundary because its TSDF source is
    truncated -- "unknown beyond truncation" needs its own anchor so the
    interior stays consistently signed. Our SDF source is exact and
    untruncated (until Piece 1 lands), so that second rule has no analogue
    here yet.

    Args:
        sdf_grid: float [nx, ny, nz] exact (or otherwise known) signed
            distance at each voxel center.
        voxel_size: scalar cubic voxel edge length.
        surface_band_voxels: seed threshold in voxels either side of the
            surface. cuRobo's kernels hardcode 0.9.

    Returns:
        bool [nx, ny, nz] seed mask.
    """
    return jnp.abs(sdf_grid) <= surface_band_voxels * voxel_size


def jfa_propagate_jax(
    seed_mask: jax.Array,
    n_passes: Optional[int] = None,
    extra_refinement_passes: int = 2,
) -> tuple[jax.Array, jax.Array]:
    """Jump Flooding Algorithm: propagate nearest-seed coordinates over a grid.

    Pure-JAX ``edt_solver="jfa"`` propagate stage -- the pre-existing
    baseline, kept as a selectable alternative to ``pba_propagate_jax``
    (both share this exact ``(seed_mask) -> (site_xyz, has_site)``
    contract, and both plug into the same seed/distance stages).

    SIMPLIFIED vs. cuRobo V2's ``wp_jfa.py``: 26-connected (all face + edge +
    corner neighbors) rather than cuRobo's 18-connected default. cuRobo skips
    the 8 corner neighbors as a CUDA memory-bandwidth optimization (31% fewer
    reads per pass); that motivation doesn't apply to a vectorized JAX
    implementation, so this uses the more accurate (and, in JAX, no more
    expensive to express) full 26-connected neighborhood instead. This is a
    deliberate deviation, not an oversight.

    Each pass offset s: every voxel looks at its 26 neighbors at coordinate
    offset (di*s, dj*s, dk*s) for di,dj,dk in {-1,0,1}, and keeps whichever
    of {self, 26 neighbors}'s site is nearest in squared Euclidean distance.
    Offsets halve geometrically (standard JFA: passes = ceil(log2(max
    dim))), then ``extra_refinement_passes`` additional offset-1 passes run
    at the end -- cuRobo's own comments describe a "1+JFA+2" scheme for the
    same reason (JFA's geometric-offset schedule alone is provably imperfect
    near Voronoi-cell boundaries; a few offset-1 sweeps at the end mop up
    most of that residual error cheaply).

    Implemented via ``jnp.pad`` + static slicing per neighbor direction
    (27 candidates including self) rather than per-voxel Python logic, so it
    vectorizes over the whole grid; the per-pass Python loop over directions
    is unrolled at trace time (offsets are Python ints, not traced values).

    Args:
        seed_mask: bool [nx, ny, nz]. True at seed (surface) voxels.
        n_passes: number of geometric-offset passes. Defaults to
            ``ceil(log2(max(nx, ny, nz)))``, the standard JFA schedule that
            guarantees every voxel can theoretically reach any site.
        extra_refinement_passes: additional offset-1 passes appended after
            the geometric schedule (see above).

    Returns:
        site_xyz: int32 [nx, ny, nz, 3] -- grid-index coordinates of the
            nearest seed voxel found for each cell.
        has_site: bool [nx, ny, nz] -- False only if propagation never
            reached a seed (e.g. an empty seed_mask, or a grid too small
            for the requested passes to cover).
    """
    nx, ny, nz = seed_mask.shape
    if n_passes is None:
        n_passes = int(math.ceil(math.log2(max(nx, ny, nz, 2))))

    ii, jj, kk = jnp.meshgrid(
        jnp.arange(nx), jnp.arange(ny), jnp.arange(nz), indexing="ij"
    )
    coords = jnp.stack([ii, jj, kk], axis=-1).astype(jnp.int32)  # (nx,ny,nz,3)

    site_xyz = jnp.where(seed_mask[..., None], coords, jnp.int32(-1))
    has_site = seed_mask

    offsets = [2 ** p for p in range(n_passes - 1, -1, -1)]
    offsets += [1] * extra_refinement_passes

    directions = [
        (di, dj, dk)
        for di in (-1, 0, 1)
        for dj in (-1, 0, 1)
        for dk in (-1, 0, 1)
        if not (di == 0 and dj == 0 and dk == 0)
    ]

    for s in offsets:
        cand_xyz = [site_xyz]
        cand_valid = [has_site]
        pad_width_xyz = ((s, s), (s, s), (s, s), (0, 0))
        pad_width_valid = ((s, s), (s, s), (s, s))
        padded_xyz = jnp.pad(site_xyz, pad_width_xyz, constant_values=-1)
        padded_valid = jnp.pad(has_site, pad_width_valid, constant_values=False)

        for (di, dj, dk) in directions:
            sl_i = slice(s + di * s, s + di * s + nx)
            sl_j = slice(s + dj * s, s + dj * s + ny)
            sl_k = slice(s + dk * s, s + dk * s + nz)
            cand_xyz.append(padded_xyz[sl_i, sl_j, sl_k])
            cand_valid.append(padded_valid[sl_i, sl_j, sl_k])

        stacked_xyz = jnp.stack(cand_xyz, axis=0)      # (27, nx,ny,nz,3)
        stacked_valid = jnp.stack(cand_valid, axis=0)  # (27, nx,ny,nz)

        diff = stacked_xyz - coords[None]
        dist2 = jnp.sum(diff * diff, axis=-1).astype(jnp.float32)
        dist2 = jnp.where(stacked_valid, dist2, jnp.inf)

        best = jnp.argmin(dist2, axis=0)  # (nx,ny,nz)
        site_xyz = jnp.take_along_axis(
            stacked_xyz, best[None, ..., None], axis=0
        )[0]
        has_site = jnp.any(stacked_valid, axis=0)

    return site_xyz, has_site


_EMPTY_VOXEL = jnp.int32(-2147483648)  # bit 31 set, matches site_encoding.cuh


def _pack_seed_grid(seed_mask: jax.Array) -> jax.Array:
    """seed_mask [nx,ny,nz] bool -> packed int32 [nx,ny,nz] site grid, in
    the packing convention PBA's CUDA kernels expect (site_encoding.cuh).

    Our own grids are stored [nx,ny,nz] row-major (nz fastest) -- EXACTLY
    cuRobo's own convention, so no data transpose is needed, only
    relabeling which axis the kernel calls "x"/"y"/"z": the kernel's
    (x,y,z) = (our k, our j, our i). Seed at our index (i,j,k) ->
    encode_site(x=k, y=j, z=i, empty=0). Non-seed -> _EMPTY_VOXEL
    (encode_site(0,0,0,empty=1) == 0x80000000). Verified against an
    asymmetric-grid stress test (a cubic grid can't distinguish a correct
    mapping from a consistently-swapped one) -- see
    scratch/pba_validate.py.
    """
    nx, ny, nz = seed_mask.shape
    ii, jj, kk = jnp.meshgrid(
        jnp.arange(nx), jnp.arange(ny), jnp.arange(nz), indexing="ij"
    )
    packed_coords = (
        (kk.astype(jnp.int32) << 20)
        | (jj.astype(jnp.int32) << 10)
        | ii.astype(jnp.int32)
    )
    return jnp.where(seed_mask, packed_coords, _EMPTY_VOXEL).astype(jnp.int32)


def _unpack_site_grid(packed: jax.Array) -> tuple[jax.Array, jax.Array]:
    """packed int32 [nx,ny,nz] -> (site_xyz [nx,ny,nz,3] int32, has_site bool).

    Inverse of _pack_seed_grid's coordinate convention: decoded kernel field
    x = our k, y = our j, z = our i, so site_xyz[...,:] = (z_field, y_field,
    x_field) reordered back into (i,j,k).
    """
    x_field = (packed >> 20) & 0x3FF
    y_field = (packed >> 10) & 0x3FF
    z_field = packed & 0x3FF
    has_site = packed >= 0  # bit 31 clear <=> non-negative int32
    site_xyz = jnp.stack([z_field, y_field, x_field], axis=-1).astype(jnp.int32)
    return site_xyz, has_site


def pba_propagate_jax(
    seed_mask: jax.Array,
    m3: int = 2,
) -> tuple[jax.Array, jax.Array]:
    """CUDA Parallel Banding Algorithm: exact propagate stage,
    ``edt_solver="pba"``.

    Same ``(seed_mask) -> (site_xyz, has_site)`` contract as
    :func:`jfa_propagate_jax`, so it's a drop-in ``propagate_fn`` for
    :func:`compute_esdf_pipeline_jax` / ``edt_solver="pba"`` in
    :func:`build_esdf_pipeline`. Unlike JFA's O(log N) approximate passes,
    this is PBA+'s exact 5-launch Voronoi computation (CUDA, see
    ``cuda_kernels/esdf/pba3d_kernel.cuh``).

    Requires the compiled ``_pba_cuda_lib.so`` (``bash
    build_kernels/build_pba_cuda.sh``) and a CUDA JAX device.

    ARCHITECTURAL INVARIANT -- deliberately NOT wrapped in jax.custom_jvp,
    and no future change should re-add one without re-deriving this:

    ``CUDADifferentiableSDFCollisionChecker``'s CUDA kernels (the discrete
    Sphere/Capsule/Box/HalfSpace world-collision path, ``_cuda_collision.py``)
    DO need a custom_jvp, because two things are both true there: (a) the
    CUDA/FFI kernel is opaque to autodiff (XLA custom calls have no built-in
    JVP rule), AND (b) a REAL, nonzero, useful gradient needs to cross that
    boundary -- distance from a robot sphere to a primitive, differentiated
    w.r.t. robot configuration through forward kinematics, which is exactly
    what CHOMP/SCO trajopt need. The custom_jvp bridges opaque-but-fast CUDA
    to differentiable-but-slow pure-JAX specifically because that nonzero
    tangent has nowhere else to go.

    Neither condition holds here. Grid construction (seed -> propagate ->
    distance) has ZERO gradient w.r.t. sdf_grid BY CONSTRUCTION: distance
    MAGNITUDE comes from an integer nearest-site index (no local derivative
    -- it's a discrete argmin/propagation result, not a continuous function
    of sdf_grid's values), and SIGN comes from a ``sdf_grid < 0`` step
    function (no local derivative almost everywhere). This was verified two
    ways, not just reasoned: (1) jax.grad through the pre-existing pure-JAX
    baseline (seed_sites_from_sdf_jax -> jfa_propagate_jax ->
    signed_distance_from_sites_jax) w.r.t. sdf_grid is identically zero at
    every point tested -- so this is a property of the PIPELINE, not
    something PBA introduced; (2) jax.grad through THIS function (the raw
    CUDA call included, no custom_jvp anywhere in the call graph) succeeds
    without error and returns the correct all-zero gradient, because
    sdf_grid only ever reaches the FFI call through the already-discrete
    seed_mask (bool) -- JAX's AD machinery only asks a primitive for a JVP
    rule when a nonzero tangent actually needs to flow through it, and none
    does here.

    The differentiable seam that matters for trajopt is, and remains,
    esdf_query_jax's trilinear interpolation w.r.t. the CONTINUOUS QUERY
    POINT (robot sphere centers via FK) -- pure JAX, no FFI, needs no
    bridging, and never touches grid construction at all. Grid construction
    runs once per world/scene as a preprocessing step; nothing in this
    codebase differentiates through it, and this function's contract is
    that nothing should have to.

    (When Piece 3, depth-camera fusion, is built: do NOT assume the same
    conclusion applies there without re-checking. Unlike PBA's discrete
    propagation, differentiating fused TSDF/mapping output w.r.t. camera
    pose or depth IS mathematically meaningful -- whether it's actually
    needed depends on whether any intended caller wants those gradients,
    which has to be answered on its own, not inherited from this function's
    reasoning.)
    """
    packed_in = _pack_seed_grid(seed_mask)
    packed_out = pba_propagate_sites_raw(packed_in, m3=m3)
    return _unpack_site_grid(packed_out)


def signed_distance_from_sites_jax(
    site_xyz: jax.Array,
    has_site: jax.Array,
    sdf_grid: jax.Array,
    voxel_size: float,
    unobserved_fill: float = 1e4,
) -> jax.Array:
    """Turn propagated nearest-site coordinates into a signed distance field.

    SIMPLIFIED vs. cuRobo V2's ``compute_esdf_from_min_tsdf_kernel``: sign is
    read directly from ``sdf_grid`` at the QUERY voxel -- appropriate while
    the SDF source is exact and dense (until Piece 1 lands). cuRobo instead
    re-samples sign at a point offset ``adjacent_skip_steps`` voxels toward
    the site (falling back to the query voxel) specifically to dodge sign
    noise from a truncated, weighted-average TSDF -- a problem that doesn't
    exist for an exact analytic source, so that machinery isn't reproduced
    here yet.

    Magnitude is geometric distance-in-voxels to the nearest site (from
    ``site_xyz``) times ``voxel_size`` -- identical in spirit to cuRobo's
    ``edt_dist``, and NOT a re-interpolation of the SDF.

    Args:
        site_xyz: int32 [nx, ny, nz, 3] from a propagate_fn.
        has_site: bool [nx, ny, nz] from a propagate_fn.
        sdf_grid: float [nx, ny, nz], the same array seeding was run on --
            doubles as the sign source (see above).
        voxel_size: scalar cubic voxel edge length.
        unobserved_fill: value written where propagation never found a site
            (mirrors cuRobo's ``1e4`` sentinel for "far/unknown, assume free").

    Returns:
        float32 [nx, ny, nz] signed distance field. Positive = separated,
        negative = penetration (same convention as esdf_query_jax/pyroffi).
    """
    nx, ny, nz = has_site.shape
    ii, jj, kk = jnp.meshgrid(
        jnp.arange(nx), jnp.arange(ny), jnp.arange(nz), indexing="ij"
    )
    coords = jnp.stack([ii, jj, kk], axis=-1).astype(jnp.int32)

    diff = site_xyz - coords
    dist_voxels = jnp.sqrt(jnp.sum((diff * diff).astype(jnp.float32), axis=-1))
    edt_dist = dist_voxels * voxel_size

    signed = jnp.where(sdf_grid < 0.0, -edt_dist, edt_dist)
    return jnp.where(has_site, signed, jnp.float32(unobserved_fill))


_PROPAGATE_FNS: dict[str, Callable[[jax.Array], tuple[jax.Array, jax.Array]]] = {
    "jfa": jfa_propagate_jax,
    "pba": pba_propagate_jax,
}


def compute_esdf_pipeline_jax(
    sdf_grid: jax.Array,
    voxel_size: float,
    *,
    seed_fn=seed_sites_from_sdf_jax,
    propagate_fn=jfa_propagate_jax,
    distance_fn=signed_distance_from_sites_jax,
) -> jax.Array:
    """Seed -> propagate -> distance, composed from swappable stage functions.

    Most callers should prefer :func:`build_esdf_pipeline`, which selects
    ``propagate_fn`` via the ``edt_solver`` string; this lower-level function
    exists for swapping other stages too (e.g. a future Piece-1 seed_fn
    reading from block-sparse TSDF storage instead of a dense sdf_grid).

    Args:
        sdf_grid: float [nx, ny, nz] analytic/known signed distance source.
        voxel_size: scalar cubic voxel edge length.
        seed_fn: ``(sdf_grid, voxel_size) -> seed_mask``.
        propagate_fn: ``(seed_mask) -> (site_xyz, has_site)``.
        distance_fn: ``(site_xyz, has_site, sdf_grid, voxel_size) -> dist``.

    Returns:
        float32 [nx, ny, nz] signed distance field.
    """
    seed_mask = seed_fn(sdf_grid, voxel_size)
    site_xyz, has_site = propagate_fn(seed_mask)
    return distance_fn(site_xyz, has_site, sdf_grid, voxel_size)


def build_esdf_pipeline(
    sdf_grid: jax.Array,
    voxel_size: float,
    edt_solver: str = "pba",
) -> jax.Array:
    """Faithful-V2-Piece-2 ESDF builder: seed -> propagate -> distance, with
    a selectable exact-distance-transform solver.

    This is the main entry point for building an :class:`ESDFWorldGeom`'s
    ``grid`` from a known signed-distance source. Grid CONSTRUCTION is
    nondifferentiable throughout (see :func:`pba_propagate_jax`'s docstring
    for why, and how that was verified rather than assumed) -- the
    resulting grid is meant to be treated as a constant and queried via
    :func:`esdf_query_jax`, which IS differentiable w.r.t. the query point.

    Args:
        sdf_grid: float [nx, ny, nz] analytic/known signed distance source.
        voxel_size: scalar cubic voxel edge length.
        edt_solver: ``"pba"`` (default -- exact, CUDA, requires
            ``_pba_cuda_lib.so`` and a CUDA JAX device; matches cuRobo V2's
            own default) or ``"jfa"`` (pure JAX, approximate, no CUDA
            dependency -- useful for CPU-only development/testing, or as a
            fallback where no GPU is available).

    Returns:
        float32 [nx, ny, nz] signed distance field, ready to wrap in an
        :class:`ESDFWorldGeom`.
    """
    if edt_solver not in _PROPAGATE_FNS:
        raise ValueError(
            f"Unknown edt_solver {edt_solver!r} -- expected one of "
            f"{sorted(_PROPAGATE_FNS)}."
        )
    return compute_esdf_pipeline_jax(
        sdf_grid, voxel_size, propagate_fn=_PROPAGATE_FNS[edt_solver]
    )
