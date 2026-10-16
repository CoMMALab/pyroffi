"""Locate GRiM and adapt pyroffi robots to it.

GRiM (``commalab/GRiM``) holds every CUDA kernel pyroffi launches for dynamics, IK, trajectory
optimization and robot-side collision; it builds on GRiD (A2R-Lab/GRiD, GRiM's ``external/GRiD``
submodule), which generates the dynamics. GRiM lives in ``external/GRiM`` as a source checkout,
not a pip install: the ``grim`` package is under ``bindings/``, and importing it puts GRiD's
``grid_codegen`` and peer packages on ``sys.path``. ``PYROFFI_GRIM_PATH`` overrides the
checkout location.

GRiM's motion kernels are compiled per robot (and per collision setup) on first use and cached
on disk, so the first solve on a new robot pays an nvcc build.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path


def _root() -> Path:
    env = os.environ.get("PYROFFI_GRIM_PATH")
    # src/pyroffi/_grim.py -> repo root is parents[2].
    return Path(env) if env else Path(__file__).resolve().parents[2] / "external" / "GRiM"


def ensure_importable() -> None:
    """Put ``grim`` on ``sys.path``; importing it adds GRiD (``grid_codegen`` and its peers)."""
    root = _root()
    if not (root / "bindings" / "grim" / "__init__.py").is_file():
        raise ImportError(
            f"GRiM not found at {root}. Run `git submodule update --init --recursive external/GRiM` or set "
            "PYROFFI_GRIM_PATH to a GRiM checkout.")
    d = str(root / "bindings")
    if d not in sys.path:
        sys.path.insert(0, d)
    import grim  # noqa: F401  (adds external/GRiD to sys.path)


def motion():
    """The ``grim.motion`` package."""
    ensure_importable()
    import grim.motion

    return grim.motion


@lru_cache(maxsize=None)
def _motion_robot(urdf_xml: str, joint_names: tuple[str, ...], actuated_names: tuple[str, ...]):
    robot = motion().MotionRobot.from_urdf(urdf_xml)
    if (tuple(robot.joint_names), tuple(robot.actuated_names)) != (joint_names, actuated_names):
        raise RuntimeError(
            f"GRiM orders this robot's joints as {robot.joint_names} (actuated "
            f"{robot.actuated_names}); the pyroffi Robot has {joint_names} ({actuated_names}).")
    return robot


def motion_robot(robot):
    """The :class:`grim.motion.MotionRobot` of a pyroffi ``Robot`` built with
    ``Robot.from_urdf``. Joint indices agree between the two."""
    if robot._backends is None:
        raise RuntimeError("GRiM kernels need a Robot built with Robot.from_urdf.")
    names = tuple(robot.joints.names)
    return _motion_robot(robot._backends.urdf_xml(names), names,
                         tuple(robot.joints.actuated_names))
