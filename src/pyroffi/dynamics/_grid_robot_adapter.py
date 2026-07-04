"""Translate a yourdfpy-parsed URDF into a GRiD ``Robot`` object.

The robot-acceleration URDFParser re-parses the URDF XML with BeautifulSoup;
instead we populate its ``Robot``/``Link``/``Joint`` object model — vendored into
:mod:`pyroffi.dynamics._grid_urdf` so pyroffi carries no external ``URDFParser``
dependency — directly from the yourdfpy model pyroffi already loaded, then reuse
the vendored post-processing pipeline (fixed-joint elimination, DFS renumbering,
subtree lists) untouched.  ``GRiDCodeGenerator`` (kept external) consumes the
resulting ``Robot`` object unchanged.

Two impedance mismatches are handled here rather than by modifying GRiD:

* GRiD only supports joint axes that are *positive* unit coordinate axes.
  Joints with negated axes (e.g. ``0 -1 0``) are mapped to the positive axis
  with a per-joint sign flip recorded in ``axis_signs``; callers must negate
  ``q/qd/qdd/u`` entering GRiD and results leaving it for those joints.
* GRiD ignores the ``<inertial>`` origin rotation; we rotate the inertia
  tensor into link axes before handing it over.
"""

from __future__ import annotations

import dataclasses

import numpy as onp
import yourdfpy
from scipy.spatial.transform import Rotation

from ._grid_urdf import (
    Joint as GRiDJoint,
    Link as GRiDLink,
    Robot as GRiDRobot,
    renumber_links_joints,
)


@dataclasses.dataclass(frozen=True)
class GridRobotModel:
    """A GRiD Robot plus the mapping from pyroffi's actuated-joint vectors."""

    robot: object
    """GRiD ``URDFParser.Robot`` instance (post fixed-joint elimination and
    DFS renumbering)."""
    joint_perm: onp.ndarray
    """(n,) int: pyroffi actuated-joint index for each GRiD joint id, i.e.
    ``q_grid[g] = axis_signs[g] * q_act[joint_perm[g]]``."""
    axis_signs: onp.ndarray
    """(n,) float: +/-1 per GRiD joint id, from negated-axis normalization."""
    num_pos: int


def _rpy_from_matrix(R: onp.ndarray) -> list[float]:
    # URDF rpy is extrinsic x-y-z (R = Rz(y) @ Ry(p) @ Rx(r)).
    return [float(v) for v in Rotation.from_matrix(R).as_euler("xyz")]


def _canonical_axis(axis: onp.ndarray, joint_name: str) -> tuple[list[float], float]:
    """Return (positive unit coordinate axis, sign) or raise."""
    axis = onp.asarray(axis, dtype=onp.float64)
    norm = onp.linalg.norm(axis)
    if norm == 0:
        raise NotImplementedError(
            f"GRiD dynamics: joint '{joint_name}' has a zero axis."
        )
    axis = axis / norm
    idx = int(onp.argmax(onp.abs(axis)))
    unit = onp.zeros(3)
    unit[idx] = onp.sign(axis[idx])
    if not onp.allclose(axis, unit, atol=1e-8):
        raise NotImplementedError(
            f"GRiD dynamics: joint '{joint_name}' axis {axis.tolist()} is not "
            "aligned with a coordinate axis, which GRiDCodeGenerator does not "
            "support."
        )
    sign = float(onp.sign(axis[idx]))
    positive_axis = [0.0, 0.0, 0.0]
    positive_axis[idx] = 1.0
    return positive_axis, sign


def build_grid_robot(urdf: yourdfpy.URDF) -> GridRobotModel:
    """Build a GRiD Robot from a yourdfpy URDF (see module docstring)."""
    if any(j.mimic is not None for j in urdf.joint_map.values()):
        raise NotImplementedError(
            "GRiD dynamics does not support URDFs with mimic joints."
        )

    robot = GRiDRobot(urdf.robot.name if urdf.robot is not None else "robot")

    # --- Links (inertial data, rotated into link axes). ---------------------
    for lid, (link_name, link) in enumerate(urdf.link_map.items()):
        grid_link = GRiDLink(link_name, lid)
        inertial = link.inertial
        if inertial is None or inertial.mass is None:
            grid_link.set_origin_xyz([0.0, 0.0, 0.0])
            grid_link.set_origin_rpy([0.0, 0.0, 0.0])
            grid_link.set_inertia(0, 0, 0, 0, 0, 0, 0)
        else:
            T = inertial.origin if inertial.origin is not None else onp.eye(4)
            R, com = T[:3, :3], T[:3, 3]
            I3 = (
                R @ onp.asarray(inertial.inertia) @ R.T
                if inertial.inertia is not None
                else onp.zeros((3, 3))
            )
            grid_link.set_origin_xyz([float(v) for v in com])
            grid_link.set_origin_rpy([0.0, 0.0, 0.0])
            grid_link.set_inertia(
                float(inertial.mass),
                float(I3[0, 0]),
                float(I3[0, 1]),
                float(I3[0, 2]),
                float(I3[1, 1]),
                float(I3[1, 2]),
                float(I3[2, 2]),
            )
        robot.add_link(grid_link)

    # --- Joints (axis normalization + sign bookkeeping). --------------------
    sign_by_name: dict[str, float] = {}
    for jid, joint in enumerate(urdf.joint_map.values()):
        grid_joint = GRiDJoint(joint.name, jid, joint.parent, joint.child)
        T = joint.origin if joint.origin is not None else onp.eye(4)
        grid_joint.set_origin_xyz([float(v) for v in T[:3, 3]])
        grid_joint.set_origin_rpy(_rpy_from_matrix(T[:3, :3]))

        jtype = joint.type
        if jtype == "continuous":
            jtype = "revolute"
        if jtype in ("revolute", "prismatic"):
            axis, sign = _canonical_axis(joint.axis, joint.name)
            sign_by_name[joint.name] = sign
            grid_joint.set_type(jtype, axis)
        elif jtype == "fixed":
            grid_joint.set_type("fixed")
        else:
            raise NotImplementedError(
                f"GRiD dynamics: unsupported joint type '{joint.type}' "
                f"(joint '{joint.name}')."
            )

        damping = 0.0
        if joint.dynamics is not None and joint.dynamics.damping is not None:
            damping = float(joint.dynamics.damping)
        grid_joint.set_damping(damping)
        robot.add_joint(grid_joint)

    # --- Reuse the vendored URDFParser post-processing pipeline. ------------
    renumber_links_joints(robot, alpha_tie_breaker=False)

    # --- Joint ordering map (GRiD DFS order -> pyroffi actuated order). -----
    act_names = [j.name for j in urdf.actuated_joints]
    grid_names = [j.get_name() for j in robot.get_joints_ordered_by_id()]
    missing = set(grid_names) - set(act_names)
    if missing:
        raise NotImplementedError(
            f"GRiD dynamics: non-actuated movable joints {sorted(missing)} "
            "are not supported."
        )
    joint_perm = onp.array([act_names.index(n) for n in grid_names], dtype=onp.int32)
    axis_signs = onp.array([sign_by_name[n] for n in grid_names])

    n = robot.get_num_pos()
    assert n == len(act_names), (n, act_names)
    return GridRobotModel(
        robot=robot, joint_perm=joint_perm, axis_signs=axis_signs, num_pos=n
    )
