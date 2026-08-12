"""Kinematic structure detection for analytic inverse kinematics.

Whether a closed-form IK solution exists, and which decomposition computes it,
is a property of where the joint axes sit relative to each other — not of the
link lengths. Pieper's criterion is the classical statement: a 6-DOF arm is
solvable in closed form when three consecutive axes intersect at a point or are
mutually parallel.

This module reads that structure off a pyroffi robot rather than requiring the
user to assert it. It probes the world-frame joint-axis lines at the home
configuration and reports which consecutive runs of axes are **concurrent**
(sharing a common point, i.e. a spherical joint) or **parallel**.

The distinction matters concretely for the arms pyroffi ships with. The Franka
Panda and FR3 have a clean spherical shoulder — axes 1, 2, 3 meet to within
1e-9 m — but their wrists are *not* spherical: axes 6 and 7 miss each other by
88 mm. That single offset is why Pieper's criterion fails at the wrist, why
Shimizu et al.'s S-R-S arm-angle method does not apply to a Franka, and why the
solver in :mod:`._analytic_ik` must search over one redundancy parameter rather
than writing down a pure closed form. Detecting this automatically means the
solver refuses the wrong robot with a specific message instead of silently
returning garbage.

Everything here is host-side NumPy, run once when a solver is built for a given
robot; none of it is on the per-solve path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .._robot import Robot

#: Axes closer than this to sharing a point are treated as concurrent (metres).
#: Well below any real link offset and well above FK round-off.
CONCURRENT_TOL = 1e-6

#: Axis directions within this of collinear are treated as parallel (radians).
PARALLEL_TOL = 1e-6


@dataclass(frozen=True)
class AxisLine:
    """A joint axis as a line in the world frame at the home configuration."""

    point: np.ndarray        # (3,) any point on the axis
    direction: np.ndarray    # (3,) unit direction


@dataclass(frozen=True)
class Structure:
    """Detected kinematic structure of a serial chain.

    Attributes:
        axes: the ``n`` joint-axis lines, base to tip.
        concurrent_runs: maximal runs of consecutive axes sharing a common
            point, as ``(start, stop, point, residual)`` with ``stop``
            exclusive. A run of length 3 is a spherical joint.
        parallel_runs: maximal runs of consecutive mutually parallel axes, as
            ``(start, stop)``.
        family: a short label naming the decomposition that applies.
    """

    axes: tuple[AxisLine, ...]
    concurrent_runs: tuple[tuple[int, int, np.ndarray, float], ...]
    parallel_runs: tuple[tuple[int, int], ...]
    family: str

    @property
    def n(self) -> int:
        return len(self.axes)

    def concurrent(self, start: int, stop: int) -> tuple[np.ndarray, float] | None:
        """Common point of axes ``[start, stop)`` if they are concurrent."""
        pt, res = _common_point(self.axes[start:stop])
        return (pt, res) if res <= CONCURRENT_TOL else None

    def describe(self) -> str:
        lines = [f"{self.n}-axis serial chain, family={self.family!r}"]
        for s, e, pt, res in self.concurrent_runs:
            if e - s >= 2:
                kind = "spherical joint" if e - s >= 3 else "intersecting pair"
                lines.append(
                    f"  axes {s + 1}..{e} concurrent at "
                    f"[{pt[0]:+.4f} {pt[1]:+.4f} {pt[2]:+.4f}] "
                    f"(residual {res:.2e} m) — {kind}")
        for s, e in self.parallel_runs:
            if e - s >= 2:
                lines.append(f"  axes {s + 1}..{e} mutually parallel")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #

def _common_point(axes) -> tuple[np.ndarray, float]:
    """Least-squares point minimising distance to every axis line.

    For each line the squared distance from a point ``x`` is
    ``‖(I - dd^T)(x - p)‖²``, so stacking the projectors gives a linear least
    squares problem. The residual returned is the largest distance from the
    solution to any single line, which is the quantity to threshold — an
    average would hide one badly-offset axis among several good ones.
    """
    axes = list(axes)
    if len(axes) == 0:
        return np.zeros(3), np.inf
    if len(axes) == 1:
        return axes[0].point.copy(), 0.0

    A = np.zeros((3, 3))
    b = np.zeros(3)
    for ax in axes:
        P = np.eye(3) - np.outer(ax.direction, ax.direction)
        A += P
        b += P @ ax.point

    # A is singular exactly when every axis is parallel (no unique meeting
    # point); lstsq gives the minimum-norm answer and the residual check below
    # then correctly rejects it.
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    residual = max(_point_line_distance(x, ax) for ax in axes)
    return x, float(residual)


def _point_line_distance(x: np.ndarray, ax: AxisLine) -> float:
    v = x - ax.point
    return float(np.linalg.norm(v - ax.direction * np.dot(ax.direction, v)))


def line_line_distance(a: AxisLine, b: AxisLine) -> float:
    """Common-normal distance between two axis lines (0 if they intersect)."""
    n = np.cross(a.direction, b.direction)
    nn = np.linalg.norm(n)
    if nn < PARALLEL_TOL:                      # parallel: any perpendicular gap
        v = b.point - a.point
        return float(np.linalg.norm(v - a.direction * np.dot(a.direction, v)))
    return float(abs(np.dot(n / nn, b.point - a.point)))


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

def joint_axis_lines(robot: "Robot", ee_link_name: str) -> tuple[AxisLine, ...]:
    """World-frame axis lines of the actuated chain base -> ``ee_link_name``.

    Taken at the home configuration by probing pyroffi's own FK, so the result
    reflects the model actually used for solving rather than a re-parsed URDF.
    """
    import jax.numpy as jnp

    from ._dh import _serial_actuated_chain
    from ._fk import forward_kinematics_joints_jax

    ee_joint = _ee_parent_joint(robot, ee_link_name)
    chain = _serial_actuated_chain(robot, ee_joint)

    n_act = len(robot.joints.actuated_names)
    poses = np.asarray(forward_kinematics_joints_jax(robot, jnp.zeros(n_act)))

    twists = np.asarray(robot.joints.twists)
    act_idx = np.asarray(robot.joints.actuated_indices)

    lines: list[AxisLine] = []
    for j in chain:
        if act_idx[j] < 0:                      # fixed joint: not a DOF
            continue
        T = _wxyz_xyz_to_matrix(poses[j])
        axis_local = twists[j][3:]              # angular part of the screw
        if np.linalg.norm(axis_local) < PARALLEL_TOL:
            continue                            # prismatic; no rotation axis
        d = T[:3, :3] @ axis_local
        lines.append(AxisLine(point=T[:3, 3].copy(), direction=d / np.linalg.norm(d)))
    return tuple(lines)


def _wxyz_xyz_to_matrix(p: np.ndarray) -> np.ndarray:
    """``[w x y z px py pz]`` -> ``[4,4]`` homogeneous transform."""
    w, x, y, z, px, py, pz = p
    T = np.eye(4)
    T[:3, :3] = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    T[:3, 3] = (px, py, pz)
    return T


def _ee_parent_joint(robot: "Robot", ee_link_name: str) -> int:
    names = list(robot.links.names)
    if ee_link_name not in names:
        raise ValueError(
            f"Unknown end-effector link {ee_link_name!r}; "
            f"robot has {names}")
    return int(np.asarray(robot.links.parent_joint_indices)[names.index(ee_link_name)])


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #

def _maximal_concurrent_runs(axes):
    """Every maximal run of consecutive axes that share a common point."""
    runs = []
    n = len(axes)
    start = 0
    while start < n:
        stop = start + 1
        best = (start, stop, axes[start].point.copy(), 0.0)
        while stop < n:
            pt, res = _common_point(axes[start:stop + 1])
            if res > CONCURRENT_TOL:
                break
            stop += 1
            best = (start, stop, pt, res)
        runs.append(best)
        # Overlapping runs matter (a wrist pair may extend a shoulder run), so
        # advance by one rather than jumping to `stop`.
        start += 1
    # Keep only runs not contained in a longer one. Deduplicate on the index
    # span alone — the point/residual fields are arrays and unhashable.
    maximal = {}
    for r in runs:
        if any(o[0] <= r[0] and r[1] <= o[1] and (o[1] - o[0]) > (r[1] - r[0])
               for o in runs):
            continue
        maximal.setdefault((r[0], r[1]), r)
    return tuple(maximal.values())


def _maximal_parallel_runs(axes):
    runs = []
    n = len(axes)
    start = 0
    while start < n:
        stop = start + 1
        while stop < n and _is_parallel(axes[start].direction, axes[stop].direction):
            stop += 1
        runs.append((start, stop))
        start = stop
    return tuple(runs)


def _is_parallel(d1, d2) -> bool:
    return float(np.linalg.norm(np.cross(d1, d2))) < PARALLEL_TOL


def classify(axes) -> str:
    """Name the decomposition family for a chain of axis lines."""
    n = len(axes)
    conc = _maximal_concurrent_runs(axes)

    def has_run(start, length):
        return any(s == start and (e - s) >= length for s, e, _, _ in conc)

    par = _maximal_parallel_runs(axes)
    has_parallel_triple = any((e - s) >= 3 for s, e in par)

    if n == 6:
        if has_run(3, 3):
            return "6dof_spherical_wrist"
        if has_run(0, 3):
            return "6dof_spherical_shoulder"
        if has_parallel_triple:
            # Pieper's second branch: three mutually parallel consecutive axes
            # (the UR / anthropomorphic-offset family).
            return "6dof_parallel_triple"
        return "6dof_general"

    if n == 7:
        shoulder = has_run(0, 3)
        wrist = has_run(4, 3)
        if shoulder and wrist:
            return "7dof_srs"                  # Shimizu's arm-angle case
        if shoulder and any(s == 4 and (e - s) >= 2 for s, e, _, _ in conc):
            return "7dof_spherical_shoulder_offset_wrist"   # Panda / FR3
        if shoulder:
            return "7dof_spherical_shoulder"
        return "7dof_general"

    return f"{n}dof_general"


def detect(robot: "Robot", ee_link_name: str) -> Structure:
    """Detect the kinematic structure of ``robot``'s chain to ``ee_link_name``."""
    axes = joint_axis_lines(robot, ee_link_name)
    return Structure(
        axes=axes,
        concurrent_runs=_maximal_concurrent_runs(axes),
        parallel_runs=_maximal_parallel_runs(axes),
        family=classify(axes),
    )
