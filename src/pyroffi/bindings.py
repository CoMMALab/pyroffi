"""Unified surface for translating a pyroffi :class:`~pyroffi.Robot` into the
external-accelerator backends.

pyroffi wraps three external kinematics/dynamics accelerators, each of which
wants the robot in its own representation:

============  =====================  =====================================
Backend       Representation         Translation
============  =====================  =====================================
GRiD (CUDA    ``URDFParser.Robot``   yourdfpy URDF -> vendored GRiD object
 dynamics)     (sympy Xmats)          model (:mod:`._grid_urdf`), fed to
                                      the external ``GRiDCodeGenerator``.
QuIK (CPU     standard-DH table      pyroffi POE ``Robot`` -> DH via FK
 IK)                                  probing (:func:`.kinematics.extract_dh`).
VAMP (CPU     ``vamp::robots::*``    URDF/SRDF *files* -> cricket codegen.
 FK)           C++ struct
============  =====================  =====================================

Historically each translation was reached through a different constructor
signature.  This module gives them one convention — a ``from_robot`` classmethod
— so callers can build any backend the same way:

    from pyroffi import bindings

    grid = bindings.GRiDDynamics.from_robot(robot)
    quik = bindings.QuIKSolver.from_robot(robot, "panda_hand")
    vfk  = bindings.VAMPCPUForwardKinematics.from_robot(robot, "panda.urdf")

GRiD and QuIK derive everything from the in-memory ``Robot``.  VAMP is the
documented exception: cricket parses the URDF/SRDF *files*, so ``from_robot``
takes the source path explicitly (``robot`` is accepted only for symmetry).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .kinematics import VAMPCPUForwardKinematics as VAMPCPUForwardKinematics
from .optimization_engines import QuIKSolver as QuIKSolver

if TYPE_CHECKING:
    from ._robot import Robot
    from .dynamics import GRiDDynamics as GRiDDynamics


@runtime_checkable
class RobotBinding(Protocol):
    """A backend constructible from a pyroffi :class:`~pyroffi.Robot`.

    All external-accelerator wrappers implement ``from_robot``; GRiD and QuIK
    take only the robot (plus backend options), while VAMP additionally requires
    the source URDF path (see the module docstring).
    """

    @classmethod
    def from_robot(cls, robot: "Robot", *args, **kwargs) -> "RobotBinding":
        ...


def __getattr__(name: str):
    # GRiDDynamics pulls in the codegen/vendor machinery; import lazily so the
    # bindings surface is usable without the external GRiDCodeGenerator present.
    if name == "GRiDDynamics":
        from .dynamics import GRiDDynamics

        return GRiDDynamics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "RobotBinding",
    "GRiDDynamics",
    "QuIKSolver",
    "VAMPCPUForwardKinematics",
]
