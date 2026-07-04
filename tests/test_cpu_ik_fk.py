"""Tests for the CPU-accelerated backends: QuIK IK, JAX-Halley IK, VAMP FK.

These exercise the pure-CPU planning stack (``JAX_PLATFORMS=cpu`` in production);
they run on whatever JAX platform is active here.  The QuIK and VAMP tests need
cricket's JIT and are skipped if it (or the required meshes) is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
import jax  # noqa: E402
import jaxlie  # noqa: E402
import yourdfpy  # noqa: E402

import pyroffi  # noqa: E402
from pyroffi.kinematics._dh import _wxyz_xyz_to_matrix, extract_dh, dh_fk  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
UR5 = REPO / "resources" / "ur5" / "ur5.urdf"
PANDA = REPO / "resources" / "panda" / "panda_spherized.urdf"
PANDA_SRDF = REPO / "resources" / "panda" / "panda.srdf"


def _robot(urdf_path):
    return pyroffi.Robot.from_urdf(yourdfpy.URDF.load(str(urdf_path)))


def _ee_matrix(robot, q, ee_idx):
    p = np.asarray(robot.forward_kinematics(jnp.asarray(q)))
    return _wxyz_xyz_to_matrix(p[ee_idx])


def _has_cricket():
    try:
        from pyroffi._jit_ffi import cricket_jit

        cricket_jit()
        return True
    except Exception:
        return False


# ── DH extraction ────────────────────────────────────────────────────────────


def test_dh_extraction_matches_fk_ur5():
    robot = _robot(UR5)
    model = extract_dh(robot, "ee_link")
    assert model.dof == 6
    rng = np.random.default_rng(1)
    ee_idx = robot.links.names.index("ee_link")
    for _ in range(8):
        q = rng.uniform(-1.5, 1.5, 6)
        T_ref = _ee_matrix(robot, q, ee_idx)
        T_dh = dh_fk(model, q[model.actuated_order])
        assert np.linalg.norm(T_ref[:3, 3] - T_dh[:3, 3]) < 1e-4


def test_dh_extraction_rejects_non_serial():
    robot = _robot(UR5)
    with pytest.raises(ValueError):
        extract_dh(robot, "base_link")  # base link -> no chain


# ── JAX-Halley IK (no cricket needed) ────────────────────────────────────────


def test_halley_ik_solves_ur5():
    robot = _robot(UR5)
    ee_idx = robot.links.names.index("ee_link")
    q_ref = np.array([0.3, -0.7, 0.9, -1.1, 0.6, 0.2])
    T = _ee_matrix(robot, q_ref, ee_idx)
    tgt = jaxlie.SE3.from_matrix(jnp.asarray(T))
    q = robot.inverse_kinematics("ee_link", tgt, solver="halley", num_seeds=32)
    T2 = _ee_matrix(robot, np.asarray(q), ee_idx)
    assert np.linalg.norm(T2[:3, 3] - T[:3, 3]) < 1e-3


def test_halley_rejects_multi_ee():
    robot = _robot(UR5)
    tgt = jaxlie.SE3.identity()
    with pytest.raises(ValueError):
        robot.inverse_kinematics(
            ["ee_link", "wrist_3_link"], [tgt, tgt], solver="halley"
        )


# ── QuIK C++ IK (needs cricket) ──────────────────────────────────────────────


@pytest.mark.skipif(not _has_cricket(), reason="cricket JIT unavailable")
def test_quik_ik_solves_ur5():
    robot = _robot(UR5)
    ee_idx = robot.links.names.index("ee_link")
    q_ref = np.array([0.2, -0.5, 0.8, -0.9, 0.4, 0.1])
    T = _ee_matrix(robot, q_ref, ee_idx)
    tgt = jaxlie.SE3.from_matrix(jnp.asarray(T))
    q = robot.inverse_kinematics("ee_link", tgt, solver="quik", num_seeds=32)
    T2 = _ee_matrix(robot, np.asarray(q), ee_idx)
    assert np.linalg.norm(T2[:3, 3] - T[:3, 3]) < 1e-3


@pytest.mark.skipif(not _has_cricket(), reason="cricket JIT unavailable")
def test_quik_and_halley_agree_on_reachability():
    """Both solvers should drive a reachable target to a tiny residual."""
    robot = _robot(UR5)
    ee_idx = robot.links.names.index("ee_link")
    rng = np.random.default_rng(3)
    for _ in range(5):
        q_ref = rng.uniform(-1.5, 1.5, 6)
        T = _ee_matrix(robot, q_ref, ee_idx)
        tgt = jaxlie.SE3.from_matrix(jnp.asarray(T))
        for solver in ("quik", "halley"):
            q = robot.inverse_kinematics("ee_link", tgt, solver=solver, num_seeds=48)
            T2 = _ee_matrix(robot, np.asarray(q), ee_idx)
            assert np.linalg.norm(T2[:3, 3] - T[:3, 3]) < 2e-3, solver


# ── VAMP CPU FK (needs cricket + meshes) ─────────────────────────────────────


@pytest.mark.skipif(
    not _has_cricket() or not PANDA.exists(), reason="cricket / panda meshes unavailable"
)
def test_vamp_cpu_fk_matches_pyroffi():
    from pyroffi.kinematics import make_vamp_cpu_fk

    robot = _robot(PANDA)
    fk = make_vamp_cpu_fk(str(PANDA), str(PANDA_SRDF))
    d = fk.dimension
    # VAMP's codegen default distal link is panda_grasptarget.
    ee = "panda_grasptarget"
    ee_idx = robot.links.names.index(ee)
    rng = np.random.default_rng(4)
    q = rng.uniform(-1.0, 1.0, (5, d)).astype(np.float32)
    T = np.asarray(fk.ee_poses(jnp.asarray(q)))
    for i in range(5):
        qf = np.zeros(int(robot.joints.num_actuated_joints))
        qf[:d] = q[i]
        T_ref = _ee_matrix(robot, qf, ee_idx)
        assert np.linalg.norm(T[i, :3, 3] - T_ref[:3, 3]) < 1e-5
