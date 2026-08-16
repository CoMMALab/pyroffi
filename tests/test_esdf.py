"""Correctness tests for pyroffi.collision._esdf's grid-construction pipeline
(seed -> propagate -> distance), covering both edt_solver options.

edt_solver="jfa" is pure JAX and always runs. edt_solver="pba" requires a
CUDA JAX device and the compiled _pba_cuda_lib.so (bash
build_kernels/build_pba_cuda.sh) -- skipped, not failed, when unavailable,
matching test_robogpu_collision.py's convention.

Independent oracles, not shared code with the module under test:
  - A small inline analytic box-SDF builder (brute-force distance to the
    nearest of several axis-aligned boxes) -- same role as
    scratch/esdf_prototype.py's build_esdf_from_boxes, reimplemented here so
    this test suite doesn't depend on scratch/.
  - brute_force_edt: plain nested-loop numpy nearest-seed exact Euclidean
    distance transform, O(N*M), shares no algorithm with JFA or PBA.
"""
from __future__ import annotations

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
import jax  # noqa: E402

from pyroffi.collision import (  # noqa: E402
    build_esdf_pipeline,
    compute_esdf_pipeline_jax,
    jfa_propagate_jax,
    pba_propagate_jax,
    seed_sites_from_sdf_jax,
    signed_distance_from_sites_jax,
)


def _pba_available() -> bool:
    try:
        seed = jnp.zeros((2, 2, 2), dtype=bool).at[0, 0, 0].set(True)
        pba_propagate_jax(seed)
        return True
    except Exception:
        return False


PBA_AVAILABLE = _pba_available()
requires_pba = pytest.mark.skipif(
    not PBA_AVAILABLE,
    reason="PBA CUDA library unavailable (build with "
    "build_kernels/build_pba_cuda.sh and run with a CUDA JAX device)",
)


def _box_sdf_grid(
    centers: np.ndarray, halves: np.ndarray, origin: np.ndarray,
    voxel_size: float, shape: tuple[int, int, int],
) -> jnp.ndarray:
    """Brute-force analytic signed distance to the nearest of several
    axis-aligned boxes, evaluated at every voxel center. Independent
    reimplementation of scratch/esdf_prototype.py's build_esdf_from_boxes.
    """
    nx, ny, nz = shape
    ii, jj, kk = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
    )
    pts = origin[None, None, None, :] + np.stack([ii, jj, kk], axis=-1) * voxel_size

    best = np.full(shape, np.inf, dtype=np.float64)
    for c, h in zip(centers, halves):
        d = np.abs(pts - c[None, None, None, :]) - h[None, None, None, :]
        outside = np.linalg.norm(np.maximum(d, 0.0), axis=-1)
        inside = np.minimum(np.max(d, axis=-1), 0.0)
        sdf = outside + inside
        best = np.minimum(best, sdf)
    return jnp.asarray(best.astype(np.float32))


def _two_box_scene():
    origin = jnp.array([-1.0, -1.0, -1.0], dtype=jnp.float32)
    voxel_size = 0.04
    shape = (50, 50, 50)
    centers = np.array([[-0.4, 0.0, 0.0], [0.4, 0.1, -0.1]], dtype=np.float32)
    halves = np.array([[0.15, 0.15, 0.15], [0.2, 0.1, 0.25]], dtype=np.float32)
    grid = _box_sdf_grid(centers, halves, np.asarray(origin), voxel_size, shape)
    return grid, voxel_size, shape


def brute_force_edt(seed_mask: np.ndarray) -> np.ndarray:
    """O(N*M) independent ground truth: true min squared Euclidean distance
    (voxel units) from every voxel to the nearest seed. Loops over seeds in
    plain Python -- shares no code/algorithm with JFA or PBA.
    """
    nx, ny, nz = seed_mask.shape
    seed_coords = np.argwhere(seed_mask).astype(np.int64)
    assert seed_coords.shape[0] > 0
    ii, jj, kk = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
    )
    all_coords = np.stack([ii, jj, kk], axis=-1).astype(np.int64)
    dist2 = np.full((nx, ny, nz), np.iinfo(np.int64).max, dtype=np.int64)
    for s in seed_coords:
        d2 = np.sum((all_coords - s[None, None, None, :]) ** 2, axis=-1)
        dist2 = np.minimum(dist2, d2)
    return dist2


def _dist2_from_sites(site_xyz: jnp.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    nx, ny, nz = shape
    ii, jj, kk = jnp.meshgrid(
        jnp.arange(nx), jnp.arange(ny), jnp.arange(nz), indexing="ij"
    )
    coords = jnp.stack([ii, jj, kk], axis=-1).astype(jnp.int32)
    diff = site_xyz.astype(jnp.int32) - coords
    return np.asarray(jnp.sum(diff * diff, axis=-1)).astype(np.int64)


# ── edt_solver="jfa": pure JAX, always runs ─────────────────────────────────


def test_jfa_pipeline_matches_analytic_oracle():
    grid, voxel_size, shape = _two_box_scene()
    dist = build_esdf_pipeline(grid, voxel_size, edt_solver="jfa")
    seed_mask = np.asarray(seed_sites_from_sdf_jax(grid, voxel_size))

    err = np.asarray(dist - grid)[~seed_mask]
    assert np.mean(np.abs(err)) < 0.5 * voxel_size
    assert np.max(np.abs(err)) < 2.0 * voxel_size

    sign_match = (np.asarray(dist)[~seed_mask] < 0.0) == (
        np.asarray(grid)[~seed_mask] < 0.0
    )
    assert np.all(sign_match)


def test_build_esdf_pipeline_rejects_unknown_solver():
    grid = jnp.zeros((4, 4, 4), dtype=jnp.float32)
    with pytest.raises(ValueError):
        build_esdf_pipeline(grid, 0.1, edt_solver="not-a-solver")


# ── edt_solver="pba": requires CUDA + compiled library ─────────────────────


@requires_pba
def test_pba_matches_independent_brute_force_edt():
    rng = np.random.default_rng(seed=1234)
    for shape, n_seeds in [((8, 11, 13), 6), ((13, 17, 9), 10)]:
        nx, ny, nz = shape
        seed_mask_np = np.zeros(shape, dtype=bool)
        flat_idx = rng.choice(nx * ny * nz, size=n_seeds, replace=False)
        seed_mask_np.flat[flat_idx] = True

        oracle_dist2 = brute_force_edt(seed_mask_np)
        site_xyz, has_site = pba_propagate_jax(jnp.asarray(seed_mask_np))
        assert bool(jnp.all(has_site))
        pba_dist2 = _dist2_from_sites(site_xyz, shape)

        assert np.array_equal(oracle_dist2, pba_dist2), (
            f"PBA disagrees with the independent brute-force EDT oracle on grid {shape}"
        )


@requires_pba
def test_pba_pipeline_matches_analytic_oracle_tighter_than_jfa():
    grid, voxel_size, shape = _two_box_scene()
    seed_mask = seed_sites_from_sdf_jax(grid, voxel_size)
    seed_np = np.asarray(seed_mask)

    pba_dist = build_esdf_pipeline(grid, voxel_size, edt_solver="pba")
    jfa_dist = build_esdf_pipeline(grid, voxel_size, edt_solver="jfa")

    pba_err = np.asarray(pba_dist - grid)[~seed_np]
    jfa_err = np.asarray(jfa_dist - grid)[~seed_np]

    # Tighter than JFA's own bound (test_jfa_pipeline_matches_analytic_oracle's
    # 0.5/2.0 voxel bounds) -- PBA is exact, only remaining error is
    # voxel/seed quantization, not propagation approximation.
    assert np.mean(np.abs(pba_err)) < 0.35 * voxel_size
    assert np.max(np.abs(pba_err)) < 1.5 * voxel_size
    assert np.mean(np.abs(pba_err)) <= np.mean(np.abs(jfa_err)) + 1e-6

    sign_match = (np.asarray(pba_dist)[~seed_np] < 0.0) == (
        np.asarray(grid)[~seed_np] < 0.0
    )
    assert np.all(sign_match)


@requires_pba
def test_pba_grid_construction_needs_no_custom_jvp():
    """jax.grad through the full pipeline (edt_solver="pba") succeeds with
    NO jax.custom_jvp anywhere in the call graph, and correctly returns an
    all-zero gradient -- see pba_propagate_jax's docstring for why this is
    the expected, architecturally-invariant result, not a bug.
    """
    grid, voxel_size, shape = _two_box_scene()

    def probe(sdf_grid):
        return compute_esdf_pipeline_jax(
            sdf_grid, voxel_size, propagate_fn=pba_propagate_jax
        )[25, 25, 25]

    g = jax.grad(probe)(grid)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert bool(jnp.all(g == 0.0))
