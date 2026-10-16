"""Unified surface for translating a pyroffi :class:`~pyroffi.Robot` into an
external-accelerator backend.

============  =====================  =====================================
Backend       Representation         Translation
============  =====================  =====================================
GRiD (CUDA    ``URDFParser.Robot``   yourdfpy URDF -> vendored GRiD object
 dynamics)     (sympy Xmats)          model (:mod:`._grid_urdf`), fed to
                                      GRiD's ``grid_codegen``.  
============  =====================  =====================================

Backends are built with a ``from_robot`` classmethod:

    from pyroffi import bindings

    grid = bindings.GRiDDynamics.from_robot(robot)

GRiM's motion kernels (IK, trajopt, collision) need no binding: the engines build them from
the Robot's URDF on first use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ._robot import Robot
    from .dynamics import GRiDDynamics as GRiDDynamics


@runtime_checkable
class RobotBinding(Protocol):
    """A backend constructible from a pyroffi :class:`~pyroffi.Robot`.

    External-accelerator wrappers implement ``from_robot``, taking the robot plus backend
    options.
    """

    @classmethod
    def from_robot(cls, robot: "Robot", *args, **kwargs) -> "RobotBinding":
        ...


def __getattr__(name: str):
    # GRiDDynamics pulls in the codegen/vendor machinery; import lazily so the
    # bindings surface is usable without the external GRiM checkout present.
    if name == "GRiDDynamics":
        from .dynamics import GRiDDynamics

        return GRiDDynamics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "RobotBinding",
    "GRiDDynamics",
]
