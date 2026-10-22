"""DH extraction: the standard-DH model of a serial chain reproduces pyroffi FK."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
import yourdfpy  # noqa: E402

import pyroffi  # noqa: E402
from pyroffi.kinematics._dh import _wxyz_xyz_to_matrix, extract_dh, dh_fk  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
UR5 = REPO / "resources" / "ur5" / "ur5.urdf"


def _robot(urdf_path):
    return pyroffi.Robot.from_urdf(yourdfpy.URDF.load(str(urdf_path)))


def _ee_matrix(robot, q, ee_idx):
    p = np.asarray(robot.forward_kinematics(jnp.asarray(q)))
    return _wxyz_xyz_to_matrix(p[ee_idx])


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
