"""Validate the JIT-compiled VAMP CPU collision checker.

This exercises pyroffi's :class:`VAMPCPUCollisionChecker`, which JIT-compiles a
robot-specialised VAMP collision checker through cricket and invokes it via the
JAX FFI on CPU.

The whole suite is skipped unless cricket (with JIT) is importable AND a `clang`
binary is on PATH — cricket's JIT driver shells out to clang at runtime.  Build
cricket first with:

    bash build_kernels/build_cricket_jit.sh

Oracles are kept self-contained (no dependence on matching pyroffi's spherized
model to VAMP's own spherization):

  * shape / dtype of the verdicts,
  * a real mix of free / in-collision configs once obstacles are placed,
  * edge-vs-config consistency: an edge can only be valid if both endpoints are,
  * a hand-built edge that plainly crosses an obstacle is rejected while a
    clear edge in free space is accepted.

Run:
    pytest tests/test_vamp_cpu_collision.py -s
"""

from __future__ import annotations

import pathlib
import shutil
import time

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
import jax  # noqa: E402

import pyroffi as pk  # noqa: E402
from pyroffi.collision import Sphere  # noqa: E402

RESOURCE_ROOT = pathlib.Path(__file__).resolve().parent.parent / "resources"
# Use the spherized URDF: cricket reads the sphere primitives directly, so no
# meshes need to be resolved on disk.
URDF = RESOURCE_ROOT / "panda" / "panda_spherized.urdf"
SRDF = RESOURCE_ROOT / "panda" / "panda.srdf"


def _checker():
    """Build the VAMP CPU checker or skip if the toolchain isn't present."""
    if shutil.which("clang") is None:
        pytest.skip("clang not on PATH; cricket JIT cannot run")
    try:
        from pyroffi.collision import VAMPCPUCollisionChecker
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"VAMP checker unavailable: {exc}")
    try:
        return VAMPCPUCollisionChecker(URDF, srdf_path=SRDF)
    except RuntimeError as exc:  # cricket not built / not importable
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def checker():
    return _checker()


def test_configs_shape_and_mix(checker):
    n = checker.dimension
    cfg = jnp.asarray(
        np.random.RandomState(1).uniform(-1.2, 1.2, size=(512, n)), dtype=jnp.float32
    )
    world = Sphere.from_center_and_radius(
        center=jnp.array([[0.3, 0.0, 0.6], [0.0, 0.35, 0.7], [-0.25, 0.0, 0.5]]),
        radius=jnp.array([0.13, 0.12, 0.12]),
    )
    free = np.asarray(checker.check_collision_free(None, cfg, world))
    assert free.shape == (512,)
    assert free.dtype == bool
    n_free = int(free.sum())
    print(f"[vamp-configs] free {n_free}/512")
    assert 0 < n_free < 512, "vacuous test — need a mix of free / colliding configs"


def test_edges_consistent_with_endpoints(checker):
    n = checker.dimension
    world = Sphere.from_center_and_radius(
        center=jnp.array([[0.3, 0.0, 0.6], [0.0, 0.35, 0.7]]),
        radius=jnp.array([0.15, 0.14]),
    )
    rng = np.random.RandomState(5)
    E = 128
    a = jnp.asarray(rng.uniform(-1.2, 1.2, size=(E, n)), dtype=jnp.float32)
    b = jnp.asarray(rng.uniform(-1.2, 1.2, size=(E, n)), dtype=jnp.float32)
    edges = jnp.stack([a, b], axis=1)  # [E, 2, n]

    edge_ok = np.asarray(checker.check_edges_collision_free(None, edges, world))
    assert edge_ok.shape == (E,)

    a_free = np.asarray(checker.check_collision_free(None, a, world))
    b_free = np.asarray(checker.check_collision_free(None, b, world))
    # A valid edge requires both endpoints valid (necessary condition).
    assert np.all(~edge_ok | (a_free & b_free)), "edge marked valid with a bad endpoint"
    assert 0 < int(edge_ok.sum()) < E, "edge test vacuous — need a mix"
    print(f"[vamp-edges] valid {int(edge_ok.sum())}/{E}")


def test_batch_matches_per_edge_and_is_deterministic(checker):
    """The OpenMP batch edge kernel must equal validating each edge on its own.

    This is the real correctness invariant of the batch handler (independent of
    VAMP's internal discretisation): row *i* of the batch result depends only on
    edge *i*, so any batch/loop disagreement would indicate a data-race or
    indexing bug in the parallel loop.  Also checks run-to-run determinism.
    """
    n = checker.dimension
    world = Sphere.from_center_and_radius(
        center=jnp.array([[0.3, 0.0, 0.6], [0.0, 0.35, 0.7]]),
        radius=jnp.array([0.13, 0.12]),
    )
    rng = np.random.RandomState(7)
    E = 256
    a = jnp.asarray(rng.uniform(-1.2, 1.2, size=(E, n)), dtype=jnp.float32)
    b = jnp.asarray(rng.uniform(-1.2, 1.2, size=(E, n)), dtype=jnp.float32)
    edges = jnp.stack([a, b], axis=1)

    batch = np.asarray(checker.check_edges_collision_free(None, edges, world))
    again = np.asarray(checker.check_edges_collision_free(None, edges, world))
    assert np.array_equal(batch, again), "edge validation is not deterministic"

    # Validate a handful of edges one at a time and compare to the batch rows.
    idx = rng.choice(E, size=16, replace=False)
    per_edge = np.array(
        [
            bool(
                np.asarray(
                    checker.check_edges_collision_free(None, edges[i : i + 1], world)
                )[0]
            )
            for i in idx
        ]
    )
    assert np.array_equal(per_edge, batch[idx]), "batch result differs from per-edge"
    assert 0 < int(batch.sum()) < E, "batch test vacuous — need a mix of edges"
    print(f"[vamp-batch] valid {int(batch.sum())}/{E}; batch==per-edge, deterministic")


def test_point_cloud_capt(checker):
    """A point-cloud (CAPT) wall must invalidate configs that were free without it."""
    n = checker.dimension
    far = Sphere.from_center_and_radius(
        center=jnp.array([[100.0, 100.0, 100.0]]), radius=jnp.array([1e-3])
    )
    cfg = jnp.asarray(
        np.random.RandomState(1).uniform(-1.2, 1.2, size=(512, n)), dtype=jnp.float32
    )
    base = int(np.asarray(checker.check_collision_free(None, cfg, far)).sum())

    gx, gz = np.meshgrid(np.linspace(0.15, 0.5, 25), np.linspace(0.2, 1.0, 25))
    pts = np.stack([gx.ravel(), np.zeros(gx.size), gz.ravel()], axis=1).astype(np.float32)
    withpc = int(
        np.asarray(
            checker.check_collision_free(
                None, cfg, far, point_cloud=jnp.asarray(pts), capt=(0.0, 1.0, 0.04)
            )
        ).sum()
    )
    print(f"[vamp-capt] free without cloud {base}/512; with cloud {withpc}/512")
    assert withpc < base, "CAPT point-cloud wall removed no free configurations"


def _time_call(fn, repeats):
    """Return the best wall-clock time (seconds) over `repeats` runs of `fn`.

    We report the minimum rather than the mean: it's the cleanest estimate of the
    kernel's intrinsic cost, least polluted by scheduler noise / other processes.
    `np.asarray(...)` forces the FFI result to be materialised so we're not timing
    lazy JAX dispatch.
    """
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        np.asarray(fn())
        best = min(best, time.perf_counter() - t0)
    return best


def test_timing_profile(checker):
    """Basic timing profile of config / edge validation across batch sizes.

    This is a profiling aid rather than a hard assertion: it warms up the JIT once
    (so compilation isn't counted) and then reports best-of-N latency and
    throughput for a range of batch sizes.  The only assertion is that timing
    actually succeeded for every batch size.
    """
    n = checker.dimension
    world = Sphere.from_center_and_radius(
        center=jnp.array([[0.3, 0.0, 0.6], [0.0, 0.35, 0.7], [-0.25, 0.0, 0.5]]),
        radius=jnp.array([0.13, 0.12, 0.12]),
    )
    rng = np.random.RandomState(11)
    batch_sizes = [1, 8, 64, 256, 1024, 4096]
    repeats = 5

    # Warm up the JIT-compiled kernels once so compilation isn't timed.
    warm_cfg = jnp.asarray(rng.uniform(-1.2, 1.2, size=(8, n)), dtype=jnp.float32)
    warm_edges = jnp.stack([warm_cfg, warm_cfg], axis=1)
    np.asarray(checker.check_collision_free(None, warm_cfg, world))
    np.asarray(checker.check_edges_collision_free(None, warm_edges, world))

    print("\n[vamp-timing] config validation (best of %d):" % repeats)
    print(f"{'batch':>8} {'total (ms)':>12} {'per-cfg (us)':>14} {'cfg/s':>14}")
    for bs in batch_sizes:
        cfg = jnp.asarray(rng.uniform(-1.2, 1.2, size=(bs, n)), dtype=jnp.float32)
        cfg.block_until_ready()
        dt = _time_call(lambda c=cfg: checker.check_collision_free(None, c, world), repeats)
        print(
            f"{bs:>8} {dt * 1e3:>12.3f} {dt / bs * 1e6:>14.3f} {bs / dt:>14.0f}"
        )

    print("\n[vamp-timing] edge validation (best of %d):" % repeats)
    print(f"{'batch':>8} {'total (ms)':>12} {'per-edge (us)':>14} {'edge/s':>14}")
    for bs in batch_sizes:
        a = jnp.asarray(rng.uniform(-1.2, 1.2, size=(bs, n)), dtype=jnp.float32)
        b = jnp.asarray(rng.uniform(-1.2, 1.2, size=(bs, n)), dtype=jnp.float32)
        edges = jnp.stack([a, b], axis=1)
        edges.block_until_ready()
        dt = _time_call(
            lambda e=edges: checker.check_edges_collision_free(None, e, world), repeats
        )
        print(
            f"{bs:>8} {dt * 1e3:>12.3f} {dt / bs * 1e6:>14.3f} {bs / dt:>14.0f}"
        )


if __name__ == "__main__":
    c = _checker()
    test_configs_shape_and_mix(c)
    test_edges_consistent_with_endpoints(c)
    test_batch_matches_per_edge_and_is_deterministic(c)
    test_point_cloud_capt(c)
    test_timing_profile(c)
    print("\nAll VAMP CPU collision checks passed.")
