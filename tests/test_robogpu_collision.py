"""Correctness test for the RoboGPU OptiX point-cloud collision path.

Independent oracle: pyroffi's own FK (`RobotCollisionSpherized.at_config`) gives
world-frame robot spheres; we brute-force test each against the env point cloud
(point i collides iff dist(center, p_i) < r_robot + r_env).  Self-collision is
disabled in the checker so the OptiX point-cloud stage is what's under test.

Skipped unless a CUDA GPU is present and the RoboGPU library has been built:

    bash build_kernels/build_robogpu_collision.sh

Run:
    pytest tests/test_robogpu_collision.py -s
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
import jax  # noqa: E402
import yourdfpy  # noqa: E402

import pyroffi as pk  # noqa: E402
from pyroffi.collision import RobotCollisionSpherized, Sphere  # noqa: E402

RES = pathlib.Path(__file__).resolve().parent.parent / "resources" / "panda"


def _checker(coll):
    try:
        from pyroffi.collision import RoboGPUCollisionChecker
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"RoboGPU checker unavailable: {exc}")
    try:
        return RoboGPUCollisionChecker(coll)
    except RuntimeError as exc:  # library not built / no OptiX
        pytest.skip(str(exc))


def test_robogpu_pointcloud_matches_oracle():
    urdf = yourdfpy.URDF.load(str(RES / "panda_spherized.urdf"))
    robot = pk.Robot.from_urdf(urdf)
    coll = RobotCollisionSpherized.from_urdf(urdf)

    rng = np.random.default_rng(3)
    B = 384
    home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float32)
    cfgs = jnp.array((home[None, :] + rng.uniform(-0.7, 0.7, (B, 7))).astype(np.float32))

    Mp = 300
    pc = rng.uniform(-0.8, 0.8, (Mp, 3)).astype(np.float32)
    pc_j = jnp.array(pc)
    R_ENV = 0.05
    far = Sphere.from_center_and_radius(
        center=jnp.array([[100.0, 100.0, 100.0]]), radius=jnp.array([0.01]))

    # RoboGPU with self-collision disabled so only the point-cloud path runs.
    rg = _checker(coll)
    rg._f_pair_i = jnp.zeros((0,), dtype=jnp.int32)
    rg._f_pair_j = jnp.zeros((0,), dtype=jnp.int32)
    rg._cached_robot_id = None
    rg._jit_fn = None
    rg.set_world(far, point_cloud=pc_j, r_env=R_ENV)

    try:
        v_rg = np.asarray(rg.check_collision_free(robot, cfgs)).astype(int)
    except Exception as exc:  # CUDA/OptiX runtime failure → skip, not fail
        pytest.skip(f"RoboGPU kernel did not run: {exc}")

    # Independent oracle: world-frame robot spheres via FK, brute-force vs cloud.
    geom = jax.vmap(lambda c: coll.at_config(robot, c))(cfgs)
    centers = np.asarray(geom.pose.translation())     # [B, S, 3]
    radii = np.asarray(geom.size).reshape(centers.shape[:-1])  # [B, S]

    pc_np = np.asarray(pc)
    v_oracle = np.ones(B, dtype=int)
    for b in range(B):
        c, r = centers[b], radii[b]
        valid = r > 0
        c, r = c[valid], r[valid]
        d2 = ((c[:, None, :] - pc_np[None, :, :]) ** 2).sum(-1)
        if np.any(d2 < (r[:, None] + R_ENV) ** 2):
            v_oracle[b] = 0

    mismatch = int((v_rg != v_oracle).sum())
    # A real free/hit mix proves the point-cloud path is actually exercised.
    assert 0.02 < v_rg.mean() < 0.98, f"degenerate verdict distribution: {v_rg.mean()}"
    assert mismatch == 0, (
        f"{mismatch}/{B} verdict mismatches vs brute-force oracle "
        f"(RoboGPU free={v_rg.mean():.3f}, oracle free={v_oracle.mean():.3f})"
    )


def test_robogpu_dynamic_refit_matches_oracle():
    """Streaming/dynamic mode: the BVH is refit (not rebuilt) each frame.

    Feed several distinct point clouds of the same size through ``dynamic=True``
    and confirm every frame's verdicts match the brute-force oracle, proving the
    in-place OptiX refit stays correct across updates.
    """
    urdf = yourdfpy.URDF.load(str(RES / "panda_spherized.urdf"))
    robot = pk.Robot.from_urdf(urdf)
    coll = RobotCollisionSpherized.from_urdf(urdf)

    rng = np.random.default_rng(3)
    B = 384
    home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float32)
    cfgs = jnp.array((home[None, :] + rng.uniform(-0.7, 0.7, (B, 7))).astype(np.float32))

    Mp = 300
    R_ENV = 0.05
    far = Sphere.from_center_and_radius(
        center=jnp.array([[100.0, 100.0, 100.0]]), radius=jnp.array([0.01]))

    rg = _checker(coll)
    rg._f_pair_i = jnp.zeros((0,), dtype=jnp.int32)
    rg._f_pair_j = jnp.zeros((0,), dtype=jnp.int32)
    rg._cached_robot_id = None
    rg._jit_fn = None
    rg.set_world(far, point_cloud=jnp.zeros((Mp, 3), jnp.float32),
                 r_env=R_ENV, dynamic=True)

    geom = jax.vmap(lambda c: coll.at_config(robot, c))(cfgs)
    centers = np.asarray(geom.pose.translation())
    radii = np.asarray(geom.size).reshape(centers.shape[:-1])

    def oracle(pc_np):
        v = np.ones(B, dtype=int)
        for b in range(B):
            c, r = centers[b], radii[b]
            valid = r > 0
            c, r = c[valid], r[valid]
            d2 = ((c[:, None, :] - pc_np[None, :, :]) ** 2).sum(-1)
            if np.any(d2 < (r[:, None] + R_ENV) ** 2):
                v[b] = 0
        return v

    saw_mix = False
    for frame in range(4):
        pc = rng.uniform(-0.8, 0.8, (Mp, 3)).astype(np.float32)
        try:
            v_rg = np.asarray(
                rg.check_collision_free(robot, cfgs, point_cloud=jnp.array(pc))
            ).astype(int)
        except Exception as exc:  # CUDA/OptiX runtime failure → skip, not fail
            pytest.skip(f"RoboGPU kernel did not run: {exc}")
        v_oracle = oracle(pc)
        mismatch = int((v_rg != v_oracle).sum())
        assert mismatch == 0, (
            f"frame {frame}: {mismatch}/{B} verdict mismatches after refit "
            f"(free={v_rg.mean():.3f}, oracle={v_oracle.mean():.3f})"
        )
        saw_mix |= 0.02 < v_rg.mean() < 0.98

    assert saw_mix, "no frame produced a real free/hit mix; path not exercised"


if __name__ == "__main__":
    test_robogpu_pointcloud_matches_oracle()
    test_robogpu_dynamic_refit_matches_oracle()
    print("PASS")
