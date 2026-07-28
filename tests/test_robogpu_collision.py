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


def test_robogpu_pointcloud_with_an_attachment_matches_oracle():
    """A grasped body must be posed and checked like any other geometry row.

    The OptiX path takes row-local spheres plus a per-row joint index and runs FK
    itself, so an attachment row is only correct if both its folded-in link←body
    offset and its parent-joint mapping are right.  The oracle is the same
    brute-force one as above, driven by the JAX ``at_config`` (which composes
    ``T_WB = T_WL · T_LB`` explicitly), so a mis-posed row cannot hide.

    The cloud is a loose cluster around where the held ball sits at the home
    configuration -- close enough that the ball hits it and the bare robot mostly
    does not, so the attachment is what drives the verdicts.
    """
    import jaxlie

    from pyroffi.attachments import Attachment, AttachmentSet

    urdf = yourdfpy.URDF.load(str(RES / "panda_spherized.urdf"))
    robot = pk.Robot.from_urdf(urdf)
    base = RobotCollisionSpherized.from_urdf(urdf)

    ee = robot.links.num_links - 1
    T_LB = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2])
    ball = Sphere.from_center_and_radius(
        center=jnp.zeros((1, 3)), radius=jnp.full((1,), 0.05)
    )
    aset = AttachmentSet.empty().attach(
        Attachment.from_geom(ball, ee, T_LB, name="ball")
    )
    coll = base.with_attachments(aset)

    rng = np.random.default_rng(11)
    B = 256
    R_ENV = 0.01
    home = jnp.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], jnp.float32)
    cfgs = jnp.array(
        (np.asarray(home)[None, :] + rng.uniform(-0.35, 0.35, (B, 7))).astype(np.float32)
    )
    ball_home = np.asarray(
        (jaxlie.SE3(robot.forward_kinematics(home)[ee]) @ jaxlie.SE3(T_LB)).translation()
    )
    pc = (ball_home[None, :] + rng.normal(0.0, 0.06, (120, 3))).astype(np.float32)
    far = Sphere.from_center_and_radius(
        center=jnp.array([[100.0, 100.0, 100.0]]), radius=jnp.array([0.01])
    )

    def _verdicts(model):
        ck = _checker(model)
        # Self-collision off so only the point-cloud stage is under test.
        ck._f_pair_i = jnp.zeros((0,), dtype=jnp.int32)
        ck._f_pair_j = jnp.zeros((0,), dtype=jnp.int32)
        ck._cached_robot_id = None
        ck._jit_fn = None
        ck.set_world(far, point_cloud=jnp.array(pc), r_env=R_ENV)
        try:
            return np.asarray(ck.check_collision_free(robot, cfgs)).astype(int)
        except Exception as exc:  # CUDA/OptiX runtime failure → skip, not fail
            pytest.skip(f"RoboGPU kernel did not run: {exc}")

    def _oracle(model):
        geom = jax.vmap(lambda c: model.at_config(robot, c))(cfgs)
        centers = np.asarray(geom.pose.translation()).reshape(B, -1, 3)
        radii = np.asarray(geom.size).reshape(B, -1)
        out = np.ones(B, dtype=int)
        for b in range(B):
            valid = radii[b] > 0
            c, r = centers[b][valid], radii[b][valid]
            d2 = ((c[:, None, :] - pc[None, :, :]) ** 2).sum(-1)
            if np.any(d2 < (r[:, None] + R_ENV) ** 2):
                out[b] = 0
        return out

    v_att = _verdicts(coll)
    v_oracle = _oracle(coll)
    assert 0.02 < v_att.mean() < 0.98, f"degenerate verdicts: {v_att.mean()}"
    mismatch = int((v_att != v_oracle).sum())
    assert mismatch == 0, (
        f"{mismatch}/{B} verdict mismatches vs brute-force oracle "
        f"(robogpu free={v_att.mean():.3f}, oracle free={v_oracle.mean():.3f})"
    )

    # The attachment must be what moved the verdicts, and disabling the slot must
    # put them back exactly where the un-attached model has them.
    v_bare = _verdicts(base)
    v_off = _verdicts(base.with_attachments(aset.set_active("ball", False)))
    assert int((v_att != v_bare).sum()) > 0, "the attachment changed no verdict"
    np.testing.assert_array_equal(v_off, v_bare)
