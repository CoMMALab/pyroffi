"""Cross-validate pyroffi's SDF and binary collision checkers against VAMP.

VAMP (via :class:`VAMPCPUCollisionChecker`, JIT-compiled by cricket from the
*same* spherized URDF + SRDF) is treated here as the canonical configuration
validator.  For a batch of configurations we compare three verdicts:

  * ``vamp``   — canonical world+self check (cricket-compiled VAMP kernel).
  * ``binary`` — :class:`CUDABinaryCollisionChecker` (fused FK, full Sᵢ×Sⱼ
    sphere-pair self check, like pRRTC).
  * ``sdf``    — the library's signed-distance reductions as shipped:
    ``compute_world_collision_distance`` + ``compute_self_collision_distance``.

The headline measurement is *agreement with VAMP*.  Because all three consume the
identical sphere model (the URDF primitives) and SRDF pairs, the binary kernel is
expected to track VAMP almost exactly away from the float rounding boundary.

The library SDF path is known to under-report self-collision: its
``compute_self_collision_distance`` only compares *same-index* spheres across a
link pair (the diagonal in S), whereas VAMP — and the binary kernel — check every
Sᵢ×Sⱼ pair (see the note in ``test_binary_collision.py``).  This suite therefore
asserts the binary kernel agrees with VAMP at least as well as the SDF path does,
and reports the SDF gap so the diagonal-S artifact is visible and tracked.

Skipped unless the cricket JIT toolchain is present (``clang`` on PATH +
importable ``VAMPCPUCollisionChecker`` that builds).  Build cricket first:

    bash build_kernels/build_cricket_jit.sh

Run:
    pytest tests/test_vamp_agreement.py -s
"""

from __future__ import annotations

import pathlib
import shutil

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
import jax  # noqa: E402

import pyroffi as pk  # noqa: E402
from pyroffi.collision import (  # noqa: E402
    CUDABinaryCollisionChecker,
    RobotCollisionSpherized,
    Sphere,
)

import yourdfpy  # noqa: E402

RESOURCE_ROOT = pathlib.Path(__file__).resolve().parent.parent / "resources"
# Spherized URDF: cricket reads the sphere primitives directly (no meshes), and
# pyroffi's RobotCollisionSpherized reads the same primitives, so VAMP and
# pyroffi share one sphere model — the only sources of disagreement left are the
# reduction logic (diagonal-S SDF bug) and float rounding at the boundary.
URDF = RESOURCE_ROOT / "panda" / "panda_spherized.urdf"
SRDF = RESOURCE_ROOT / "panda" / "panda.srdf"

# Configs whose closest distance is within this band straddle the collision
# boundary; float32-vs-x64 rounding (and VAMP's own resolution) can flip their
# verdict, so they are excluded from the strict-agreement assertions.
BOUNDARY = 3e-3

_FAR_WORLD = Sphere.from_center_and_radius(
    center=jnp.array([[100.0, 100.0, 100.0]]), radius=jnp.array([1e-3])
)


# ── Toolchain-gated fixtures ────────────────────────────────────────────────
def _vamp_checker():
    """Build the VAMP CPU checker or skip if the cricket toolchain isn't present."""
    if shutil.which("clang") is None:
        pytest.skip("clang not on PATH; cricket JIT cannot run")
    try:
        from pyroffi.collision import VAMPCPUCollisionChecker
    except Exception as exc:  # pragma: no cover - import guarded
        pytest.skip(f"VAMP checker unavailable: {exc}")
    try:
        return VAMPCPUCollisionChecker(URDF, srdf_path=SRDF)
    except RuntimeError as exc:  # cricket not built / not importable
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def vamp():
    return _vamp_checker()


@pytest.fixture(scope="module")
def robot():
    return pk.Robot.from_urdf(yourdfpy.URDF.load(str(URDF)))


@pytest.fixture(scope="module")
def fine():
    # Pass the SRDF so pyroffi's active self-pairs match the set VAMP compiles
    # from the same SRDF; otherwise pyroffi keeps adjacent pairs that overlap at
    # essentially every config and the self comparison is vacuous.
    return RobotCollisionSpherized.from_urdf(
        yourdfpy.URDF.load(str(URDF)), srdf_path=str(SRDF)
    )


# ── Geometric ground-truth margins (full Sᵢ×Sⱼ reduction) ───────────────────
def _world_margin(robot, model, cfg, world_geom):
    """Per-config min world signed distance over all link spheres × obstacles."""
    return jnp.min(
        jax.vmap(lambda c: model.compute_world_collision_distance(robot, c, world_geom))(cfg),
        axis=(-1, -2),
    )


def _self_margin_full(robot, model, cfg):
    """Per-config min self distance over EVERY Sᵢ×Sⱼ sphere pair of active links.

    This is the geometrically correct reduction (what VAMP and the binary kernel
    do); the library's ``compute_self_collision_distance`` only walks the S
    diagonal, so it is *not* used as ground truth here.
    """
    li = jnp.asarray(model.active_idx_i)
    lj = jnp.asarray(model.active_idx_j)

    def one(c):
        coll = model.at_config(robot, c)              # batch (S, N)
        ctr = coll.pose.translation()                 # [S, N, 3]
        rad = coll.radius                             # [S, N]
        ci, ri = ctr[:, li], rad[:, li]               # [S, P, 3], [S, P]
        cj, rj = ctr[:, lj], rad[:, lj]
        diff = ci[:, None, :, :] - cj[None, :, :, :]  # [S, S, P, 3]
        d = jnp.sqrt(jnp.sum(diff ** 2, axis=-1)) - (ri[:, None, :] + rj[None, :, :])
        valid = (ri[:, None, :] >= 0) & (rj[None, :, :] >= 0)
        d = jnp.where(valid, d, jnp.inf)
        return jnp.min(d, axis=(0, 1))                # [P]

    return jnp.min(jax.vmap(one)(cfg), axis=-1)


# ── Library SDF verdict (the reductions as actually shipped/used) ───────────
def _sdf_free(robot, model, cfg, world_geom):
    """Verdict from the library's own SDF reductions (diagonal-S self check)."""
    self_d = jax.vmap(lambda c: model.compute_self_collision_distance(robot, c))(cfg)
    self_free = jnp.all(self_d > 0.0, axis=-1)
    world_d = jax.vmap(lambda c: model.compute_world_collision_distance(robot, c, world_geom))(cfg)
    world_free = jnp.all(world_d > 0.0, axis=(-1, -2))
    return np.asarray(self_free & world_free)


def _sample_configs(robot, n, scale, seed):
    """Random configs scaled toward the joint limits to induce a free/coll mix."""
    lo = np.asarray(robot.joints.lower_limits)
    hi = np.asarray(robot.joints.upper_limits)
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) * scale
    u = np.random.RandomState(seed).uniform(-1.0, 1.0, size=(n, lo.shape[0]))
    return jnp.asarray(mid + u * half, dtype=jnp.float32)


def _summarise(label, true_margin, vamp_free, binary_free, sdf_free):
    """Print confusion vs VAMP and return (binary_agree, sdf_agree) on decisive."""
    decisive = np.abs(np.asarray(true_margin)) > BOUNDARY
    nd = int(np.sum(decisive))
    vf = np.asarray(vamp_free)

    def agree(other):
        return int(np.sum((np.asarray(other) == vf) & decisive))

    b_ok, s_ok = agree(binary_free), agree(sdf_free)
    n_free = int(np.sum(vf))
    # Directional SDF gap: SDF calls free where VAMP calls collision (the
    # under-report signature of the diagonal-S self check).
    sdf_overreport_free = int(
        np.sum((np.asarray(sdf_free) & ~vf) & decisive)
    )
    print(
        f"[{label}] n={len(vf)} decisive={nd} vamp-free={n_free}\n"
        f"        binary vs vamp: {b_ok}/{nd} agree ({nd - b_ok} mismatch)\n"
        f"        sdf    vs vamp: {s_ok}/{nd} agree ({nd - s_ok} mismatch; "
        f"{sdf_overreport_free} are sdf-free/vamp-collision)"
    )
    return nd, b_ok, s_ok, n_free


# ── Tests ───────────────────────────────────────────────────────────────────
def test_self_only_agreement(vamp, robot, fine):
    """Empty world → isolates the self-collision reductions against VAMP."""
    # Full joint range: folds the arm enough for a real self-free / self-collision
    # mix and surfaces cross-index sphere overlaps the diagonal-S SDF check misses.
    cfg = _sample_configs(robot, 1024, scale=1.0, seed=11)

    true_margin = _self_margin_full(robot, fine, cfg)
    vamp_free = np.asarray(vamp.check_collision_free(robot, cfg, _FAR_WORLD))
    binary_free = np.asarray(
        CUDABinaryCollisionChecker(fine).check_collision_free(robot, cfg, _FAR_WORLD)
    )
    sdf_free = _sdf_free(robot, fine, cfg, _FAR_WORLD)

    nd, b_ok, s_ok, n_free = _summarise(
        "self", true_margin, vamp_free, binary_free, sdf_free
    )
    assert nd > 0, "no decisive configs — widen the sampling range"
    assert 0 < n_free < len(vamp_free), "vacuous: need a free/collision mix under VAMP"
    # Binary tracks the canonical VAMP verdict away from the boundary.
    assert b_ok / nd >= 0.99, f"binary disagrees with VAMP on {nd - b_ok}/{nd} self configs"
    # The whole point of the cross-check: the binary kernel is at least as
    # faithful to VAMP as the library SDF reduction (which misses cross-index
    # sphere overlaps).
    assert b_ok >= s_ok, "binary should agree with VAMP at least as well as the SDF path"


def test_world_plus_self_agreement(vamp, robot, fine):
    """Obstacles in the workspace → exercise world + self jointly against VAMP."""
    cfg = _sample_configs(robot, 1024, scale=0.9, seed=23)
    world = Sphere.from_center_and_radius(
        center=jnp.array(
            [[0.3, 0.0, 0.6], [0.0, 0.35, 0.7], [-0.25, 0.0, 0.5], [0.2, -0.3, 0.8]]
        ),
        radius=jnp.array([0.13, 0.12, 0.12, 0.11]),
    )

    world_m = _world_margin(robot, fine, cfg, world)
    self_m = _self_margin_full(robot, fine, cfg)
    true_margin = jnp.minimum(world_m, self_m)

    vamp_free = np.asarray(vamp.check_collision_free(robot, cfg, world))
    binary_free = np.asarray(
        CUDABinaryCollisionChecker(fine).check_collision_free(robot, cfg, world)
    )
    sdf_free = _sdf_free(robot, fine, cfg, world)

    nd, b_ok, s_ok, n_free = _summarise(
        "world+self", true_margin, vamp_free, binary_free, sdf_free
    )
    assert nd > 0, "no decisive configs — adjust obstacles/sampling"
    assert 0 < n_free < len(vamp_free), "vacuous: need a free/collision mix under VAMP"
    assert b_ok / nd >= 0.99, f"binary disagrees with VAMP on {nd - b_ok}/{nd} configs"
    assert b_ok >= s_ok, "binary should agree with VAMP at least as well as the SDF path"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-s", "-v"]))
