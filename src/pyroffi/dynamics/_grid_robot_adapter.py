"""Translate a yourdfpy-parsed URDF into a GRiD ``Robot`` object.

The A2R-Lab URDFParser re-parses the URDF XML with BeautifulSoup; instead we
populate its ``Robot``/``Link``/``Joint`` object model — vendored into
:mod:`pyroffi.dynamics._grid_urdf` so pyroffi carries no external ``URDFParser``
dependency — directly from the yourdfpy model pyroffi already loaded, then reuse
the vendored post-processing pipeline (fixed-joint elimination, DFS renumbering,
subtree lists) untouched.  ``grid_codegen.GRiDCodeGenerator`` (kept external in
``external/GRiD``) consumes the resulting ``Robot`` object unchanged.

One impedance mismatch is handled here rather than by modifying GRiD: GRiD
ignores the ``<inertial>`` origin rotation, so we rotate the inertia tensor into
link axes before handing it over.

Note on joint axes: the older robot-acceleration URDFParser only supported
*positive* unit coordinate axes, so this adapter used to canonicalize the axis
and carry a per-joint sign flip (``axis_signs``) that callers had to undo on
``q/qd/qdd/u``.  The A2R-Lab parser handles signed cardinal axes natively
(``Joint.set_type`` picks the ±1 ``axis_scale``, byte-identically to the old
emit) *and* general skew axes via a Rodrigues frame rotation, so the axis is now
passed through verbatim.  ``axis_signs`` is retained as an all-ones vector so the
downstream permutation code keeps a uniform shape.
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


def _unit_axis(axis: onp.ndarray, joint_name: str) -> list[float]:
    """Normalize a URDF joint axis, snapping near-cardinal axes to exactly ±1.

    The snap matters: the parser's cardinal/skew tier choice is an exact
    comparison, and a URDF axis written as ``0 0 0.9999999`` would otherwise
    drop the joint into the general (skew) Rodrigues path and change the
    generated kernel for what is really a z-axis joint.
    """
    axis = onp.asarray(axis, dtype=onp.float64)
    norm = onp.linalg.norm(axis)
    if norm == 0:
        raise NotImplementedError(
            f"GRiD dynamics: joint '{joint_name}' has a zero axis."
        )
    axis = axis / norm
    idx = int(onp.argmax(onp.abs(axis)))
    cardinal = onp.zeros(3)
    cardinal[idx] = onp.sign(axis[idx])
    if onp.allclose(axis, cardinal, atol=1e-8):
        return [float(v) for v in cardinal]
    return [float(v) for v in axis]


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
            # Upstream's strict-inertial validation distinguishes a genuinely
            # massless frame from a real body with a broken <inertial>; keep
            # that signal rather than silently presenting a zeroed body.
            grid_link.missing_inertial = True
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

    # --- Joints. ------------------------------------------------------------
    for jid, joint in enumerate(urdf.joint_map.values()):
        grid_joint = GRiDJoint(joint.name, jid, joint.parent, joint.child)
        T = joint.origin if joint.origin is not None else onp.eye(4)
        grid_joint.set_origin_xyz([float(v) for v in T[:3, 3]])
        grid_joint.set_origin_rpy(_rpy_from_matrix(T[:3, :3]))

        jtype = joint.type
        if jtype == "continuous":
            jtype = "revolute"
        if jtype in ("revolute", "prismatic"):
            grid_joint.set_type(jtype, _unit_axis(joint.axis, joint.name))
        elif jtype == "fixed":
            grid_joint.set_type("fixed")
        else:
            raise NotImplementedError(
                f"GRiD dynamics: unsupported joint type '{joint.type}' "
                f"(joint '{joint.name}')."
            )

        damping = friction = 0.0
        if joint.dynamics is not None:
            if joint.dynamics.damping is not None:
                damping = float(joint.dynamics.damping)
            if joint.dynamics.friction is not None:
                friction = float(joint.dynamics.friction)
        grid_joint.set_damping(damping)
        grid_joint.set_friction(friction)
        robot.add_joint(grid_joint)

    # --- Reuse the vendored URDFParser post-processing pipeline. ------------
    # urdf_order (not the upstream "pinocchio_order" default) preserves the
    # joint numbering pyroffi's cached kernels were generated against; the
    # actuated-order permutation below is computed by name either way, so this
    # is about kernel-cache stability rather than correctness.
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
    # Signed/skew axes are now modelled natively by the parser (see module
    # docstring), so no sign correction is ever needed.
    axis_signs = onp.ones(len(grid_names))

    n = robot.get_num_pos()
    assert n == len(act_names), (n, act_names)
    return GridRobotModel(
        robot=robot, joint_perm=joint_perm, axis_signs=axis_signs, num_pos=n
    )
