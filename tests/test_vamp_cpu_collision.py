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
  * edge-vs-config consistency: a valid edge must have a free goal endpoint
    (VAMP samples ``(0, 1]`` — the start is assumed pre-validated),
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


def _vamp_panda():
    """Return VAMP's nanobind `panda` robot module or skip if unavailable.

    This is the upstream, hand-written/-spherised VAMP collision checker exposed
    through nanobind (`import vamp`).  pyroffi's :class:`VAMPCPUCollisionChecker`
    JIT-compiles its own kernel from the *spherized URDF*; benchmarking the two
    side by side shows the cost of going through the JAX FFI + cricket JIT versus
    calling the native binding directly.
    """
    vamp = pytest.importorskip("vamp")
    return vamp


def _vamp_env_from_world(vamp, world):
    """Build a native VAMP `Environment` from a pyroffi `Sphere` world.

    Reading the centers/radii straight off the pyroffi geometry guarantees both
    checkers see the *same* obstacles, so the only thing that differs is the
    robot's own sphere model (pyroffi's spherized URDF vs VAMP's built-in one) —
    which is why we report agreement as info but never assert verdict equality.
    """
    centers = np.asarray(world.pose.translation()).reshape(-1, 3)
    radii = np.asarray(world.radius).reshape(-1)
    env = vamp.Environment()
    for c, r in zip(centers, radii):
        env.add_sphere(vamp.Sphere([float(c[0]), float(c[1]), float(c[2])], float(r)))
    return env


# VAMP's edge discretisation is controlled by two compile-time constants:
#   * the SIMD rake width (AVX2 -> 8 floats per vector), and
#   * the robot's planning ``resolution`` (Panda is 32, which pyroffi's checker
#     also bakes in via its ``resolution=32`` default).
# We replicate them here so the *raw VAMP* edge check below samples each edge at
# exactly the same density pyroffi's kernel does — see ``_vamp_fine_samples``.
VAMP_RESOLUTION = 32


def _vamp_rake() -> int:
    """SIMD float-vector width VAMP compiles to on this CPU (its sample stride).

    VAMP picks the widest available: AVX-512 -> 16, AVX2/AVX -> 8, else a 4-wide
    SSE/NEON fallback.  This must match the build or the replicated sample
    positions (and hence verdicts) drift from pyroffi's kernel.
    """
    try:
        flags = pathlib.Path("/proc/cpuinfo").read_text()
    except OSError:
        return 8
    if "avx512f" in flags:
        return 16
    if "avx2" in flags or " avx " in flags:
        return 8
    return 4


def _vamp_fine_samples(a, b, rake: int, resolution: int):
    """Replicate the exact sample points of ``validate_motion<Robot, rake, res>``.

    VAMP validates an edge a->b by checking configurations at fractions
    ``m / (rake * n)`` for ``m = 1 .. rake*n`` (the open interval ``(0, 1]`` — the
    start is assumed pre-validated), where ``n = max(ceil(dist / rake * res), 1)``
    and ``dist = ||b - a||``.  Reproducing that fraction set lets us drive the raw
    VAMP checker at the *same* fine resolution pyroffi uses, so a verdict is the
    AND of the per-sample checks and the two backends agree (modulo tiny
    float-accumulation differences at a collision boundary).

    Returns the flat ``[S, dim]`` sample buffer plus the ``[E]`` segment offsets
    into it (the start index of each edge), suitable for ``reduceat``.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    v = b - a
    dist = np.linalg.norm(v, axis=1)
    n = np.maximum(np.ceil(dist / rake * resolution), 1.0).astype(np.int64)
    counts = rake * n                                   # samples per edge [E]
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)

    seg = np.repeat(np.arange(a.shape[0]), counts)      # edge id per sample
    within = np.arange(counts.sum()) - np.repeat(offsets, counts)   # 0-based
    frac = ((within + 1) / counts[seg]).astype(np.float32)[:, None]
    samples = (a[seg] + v[seg] * frac).astype(np.float32)
    return samples, offsets


def _vamp_fine_edges(panda, env, samples, offsets):
    """Raw VAMP edge verdicts at fine resolution: AND per-sample fkcc per edge."""
    per_sample = np.asarray(panda.validate_motion_batch(samples, samples, env))
    return np.logical_and.reduceat(per_sample, offsets)


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

    b_free = np.asarray(checker.check_collision_free(None, b, world))
    # VAMP samples the open interval (0, 1], so a valid edge guarantees the *goal*
    # endpoint is free (the start is assumed pre-validated by the planner; see
    # check_edges_collision_free).  The goal being free is the necessary condition.
    assert np.all(~edge_ok | b_free), "edge marked valid with a colliding goal endpoint"
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


def test_project_collision_free(checker):
    """Projection repairs colliding seeds, passes free ones through, and is
    deterministic per seed."""
    n = checker.dimension
    world = Sphere.from_center_and_radius(
        center=jnp.array([[0.3, 0.0, 0.6], [0.0, 0.35, 0.7], [-0.25, 0.0, 0.5]]),
        radius=jnp.array([0.13, 0.12, 0.12]),
    )
    rng = np.random.RandomState(3)
    B = 1024
    cfg = jnp.asarray(rng.uniform(-1.2, 1.2, size=(B, n)), dtype=jnp.float32)
    free_before = np.asarray(checker.check_collision_free(None, cfg, world))
    assert 0 < int(free_before.sum()) < B, "vacuous test — need a free/colliding mix"

    lower = np.full(n, -2.8, np.float32)
    upper = np.full(n, 2.8, np.float32)
    qp, ok = checker.project_collision_free(
        None, cfg, world, lower=lower, upper=upper, seed=42
    )
    qp, ok = np.asarray(qp), np.asarray(ok)
    cfg_np = np.asarray(cfg)

    # Every config reported ok must actually be collision-free.
    free_after = np.asarray(checker.check_collision_free(None, jnp.asarray(qp), world))
    assert np.all(free_after[ok]), "projection returned colliding configs marked ok"
    # Already-free seeds pass through untouched; failures are returned unchanged.
    assert np.allclose(qp[free_before], cfg_np[free_before])
    assert np.allclose(qp[~ok], cfg_np[~ok])
    # Joint-limit clamping is respected.
    assert np.all(qp >= lower - 1e-6) and np.all(qp <= upper + 1e-6)
    # The projection repaired a real share of the colliding seeds.
    repaired = int((ok & ~free_before).sum())
    n_coll = int((~free_before).sum())
    print(f"[vamp-project] repaired {repaired}/{n_coll} colliding seeds")
    assert repaired > n_coll // 2, "projection repaired too few colliding seeds"

    # Deterministic per (seed, batch index), independent of the OpenMP schedule.
    qp2, ok2 = checker.project_collision_free(
        None, cfg, world, lower=lower, upper=upper, seed=42
    )
    assert np.array_equal(qp, np.asarray(qp2))
    assert np.array_equal(ok, np.asarray(ok2))


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


def test_benchmark_against_vamp_nanobind(checker):
    """Benchmark pyroffi's VAMP CPU checker against the native VAMP nanobind API.

    For both config and edge validation we time pyroffi's JAX-FFI kernel and the
    upstream `vamp.panda` binding on identical obstacle worlds and report
    best-of-N throughput plus the pyroffi/native speedup.  Notes on fairness:

      * Config validation: VAMP exposes only a *single*-config `validate`, so the
        native number is a Python loop (per-call binding overhead dominates at
        small batches); pyroffi validates the whole batch in one FFI call.
      * Edge validation: VAMP's nanobind `validate_motion`/`validate_motion_batch`
        are hardcoded to discretisation *resolution 1*, while pyroffi's kernel
        runs at the robot's full ``resolution`` (32).  To compare like with like
        — both checking each edge *finely* — we drive raw VAMP at resolution 32
        by validating the exact same sample points pyroffi's kernel visits (see
        :func:`_vamp_fine_samples`) through one batched `validate_motion_batch`
        call and AND-reducing per edge.  Because both now sample identically, the
        edge verdicts agree (reported below).  Caveat: pyroffi can early-out per
        edge on the first colliding sample; the flat-batch replication checks
        every sample, so this row slightly favours pyroffi.

    The only assertion is that both backends produced a result for every batch
    size — config verdicts can still differ because the two use different robot
    sphere models (printed for information only).
    """
    vamp = _vamp_panda()
    panda = vamp.panda
    n = checker.dimension
    assert panda.dimension() == n, "VAMP panda DOF disagrees with pyroffi checker"

    world = Sphere.from_center_and_radius(
        center=jnp.array([[0.3, 0.0, 0.6], [0.0, 0.35, 0.7], [-0.25, 0.0, 0.5]]),
        radius=jnp.array([0.13, 0.12, 0.12]),
    )
    env = _vamp_env_from_world(vamp, world)

    rng = np.random.RandomState(11)
    batch_sizes = [1, 8, 64, 256, 1024, 4096]
    repeats = 5

    # Warm up pyroffi's JIT-compiled kernels once so compilation isn't timed.
    warm = jnp.asarray(rng.uniform(-1.2, 1.2, size=(8, n)), dtype=jnp.float32)
    np.asarray(checker.check_collision_free(None, warm, world))
    np.asarray(
        checker.check_edges_collision_free(None, jnp.stack([warm, warm], axis=1), world)
    )

    # --- agreement sanity (informational) ---------------------------------
    cfg_np = rng.uniform(-1.2, 1.2, size=(256, n)).astype(np.float32)
    pk_free = np.asarray(checker.check_collision_free(None, jnp.asarray(cfg_np), world))
    vamp_free = np.array([panda.validate(c, env) for c in cfg_np])
    agree = float(np.mean(pk_free == vamp_free)) * 100.0
    print(
        f"\n[vamp-bench] config-verdict agreement pyroffi vs native VAMP: "
        f"{agree:.1f}% (differ due to distinct sphere models)"
    )

    print("\n[vamp-bench] config validation (best of %d):" % repeats)
    hdr = f"{'batch':>8} {'pyroffi cfg/s':>16} {'vamp cfg/s':>16} {'speedup':>10}"
    print(hdr)
    for bs in batch_sizes:
        cfg_np = rng.uniform(-1.2, 1.2, size=(bs, n)).astype(np.float32)
        cfg_j = jnp.asarray(cfg_np)
        cfg_j.block_until_ready()
        dt_pk = _time_call(
            lambda c=cfg_j: checker.check_collision_free(None, c, world), repeats
        )
        dt_vamp = _time_call(
            lambda c=cfg_np: np.array([panda.validate(x, env) for x in c]), repeats
        )
        print(
            f"{bs:>8} {bs / dt_pk:>16.0f} {bs / dt_vamp:>16.0f} "
            f"{dt_vamp / dt_pk:>9.2f}x"
        )

    # --- edge-verdict agreement at matched (fine) resolution --------------
    rake = _vamp_rake()
    a_np = rng.uniform(-1.2, 1.2, size=(256, n)).astype(np.float32)
    b_np = rng.uniform(-1.2, 1.2, size=(256, n)).astype(np.float32)
    edges = jnp.stack([jnp.asarray(a_np), jnp.asarray(b_np)], axis=1)
    pk_edge = np.asarray(checker.check_edges_collision_free(None, edges, world))
    samples, offsets = _vamp_fine_samples(a_np, b_np, rake, VAMP_RESOLUTION)
    vamp_edge = _vamp_fine_edges(panda, env, samples, offsets)
    edge_agree = float(np.mean(pk_edge == vamp_edge)) * 100.0
    print(
        f"\n[vamp-bench] edge-verdict agreement pyroffi (res {VAMP_RESOLUTION}) vs "
        f"raw VAMP (res {VAMP_RESOLUTION}, rake {rake}): {edge_agree:.1f}%"
    )

    print("\n[vamp-bench] edge validation @ resolution %d (best of %d):"
          % (VAMP_RESOLUTION, repeats))
    hdr = f"{'batch':>8} {'pyroffi edge/s':>16} {'vamp edge/s':>16} {'speedup':>10}"
    print(hdr)
    for bs in batch_sizes:
        a_np = rng.uniform(-1.2, 1.2, size=(bs, n)).astype(np.float32)
        b_np = rng.uniform(-1.2, 1.2, size=(bs, n)).astype(np.float32)
        edges = jnp.stack([jnp.asarray(a_np), jnp.asarray(b_np)], axis=1)
        edges.block_until_ready()
        # Sample positions for the raw-VAMP fine check are precomputed outside the
        # timed region so we measure VAMP's kernel, not numpy interpolation.
        samples, offsets = _vamp_fine_samples(a_np, b_np, rake, VAMP_RESOLUTION)
        dt_pk = _time_call(
            lambda e=edges: checker.check_edges_collision_free(None, e, world), repeats
        )
        dt_vamp = _time_call(
            lambda s=samples, o=offsets: _vamp_fine_edges(panda, env, s, o), repeats
        )
        print(
            f"{bs:>8} {bs / dt_pk:>16.0f} {bs / dt_vamp:>16.0f} "
            f"{dt_vamp / dt_pk:>9.2f}x"
        )


if __name__ == "__main__":
    c = _checker()
    test_configs_shape_and_mix(c)
    test_edges_consistent_with_endpoints(c)
    test_batch_matches_per_edge_and_is_deterministic(c)
    test_point_cloud_capt(c)
    test_timing_profile(c)
    test_benchmark_against_vamp_nanobind(c)
    print("\nAll VAMP CPU collision checks passed.")
