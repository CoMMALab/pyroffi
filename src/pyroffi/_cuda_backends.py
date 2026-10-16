"""Lazy, per-robot registry of CUDA/URDF-dependent backends.

Some accelerated backends need URDF-derived state that cannot live in the
``Robot`` JAX pytree: the collision checkers need trimesh geometry and the
GRiD dynamics backend JIT-compiles a per-robot CUDA library.  ``CudaBackends``
holds that state at the *Python* level (where side effects belong) and builds
each backend lazily on first use.

An instance is carried in ``Robot`` as a ``jdc.Static`` field, so it is part of
the pytree's ``treedef`` aux-data.  It is therefore **identity-hashed and never
compared by value** — do not give it ``__eq__``/``__hash__`` overrides, and do
not store traced arrays on it.  Two value-equal ``Robot`` objects with distinct
backend holders simply don't share cached wrappers, which is harmless.

The GRiM kernels (FK, IK, trajopt, robot-side collision) build from the URDF this holder
keeps (:meth:`CudaBackends.urdf_xml`) and cache per robot on GRiM's side.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import numpy as onp

if TYPE_CHECKING:
    import yourdfpy
    from jax import Array


def _eager():
    """Context manager forcing eager (compile-time) evaluation of array ops.

    Lets the URDF-dependent backends be *constructed* on their lazy first use
    even when that call happens inside a caller's ``jax.jit`` trace: the build
    performs concrete array math on the URDF, which would otherwise be captured
    as tracers.  Falls back to a no-op on older JAX without the helper.
    """
    try:
        return jax.ensure_compile_time_eval()
    except AttributeError:  # pragma: no cover - very old JAX
        import contextlib

        return contextlib.nullcontext()


def _urdf_xml(urdf: "yourdfpy.URDF") -> str:
    """The URDF as XML text, mimic tags included.

    yourdfpy's writer drops ``<mimic>`` (its ``_write_joint`` never calls ``_write_mimic``), and
    cricket would then trace each mimic joint as an independent coordinate.
    """
    xml = urdf.write_xml_string()
    mimics = {j.name: j.mimic for j in urdf.robot.joints if j.mimic is not None}
    if not mimics:
        return xml.decode()
    from lxml import etree

    root = etree.fromstring(xml)
    for joint in root.findall("joint"):  # direct children: transmissions also have <joint>
        m = mimics.get(joint.get("name"))
        if m is not None:
            etree.SubElement(joint, "mimic", joint=m.joint, multiplier=str(m.multiplier),
                             offset=str(m.offset))
    return etree.tostring(root, xml_declaration=True, encoding="utf-8").decode()


def _chart_urdf_xml(xml: str, root: str, joint_names: tuple[str, ...]) -> str:
    """The URDF of a floating robot's chart (see :mod:`pyroffi._floating_base`): its tree
    hung off the chart's virtual joints, which follow the real ones in ``joint_names``. The
    chart's reference rotation is baked as the identity."""
    from lxml import etree

    from ._floating_base import CHART_JOINTS, TANGENT_JOINTS, _CHART_TWISTS, _TANGENT_TWISTS

    twist = dict(zip(CHART_JOINTS + TANGENT_JOINTS, _CHART_TWISTS + _TANGENT_TWISTS))
    virtual = [n for n in joint_names if n in twist]
    tree = etree.fromstring(xml.encode())
    links = ["__chart_world"] + [f"__chart_link_{i}" for i in range(len(virtual) - 1)] + [root]
    for name in links[:-1]:
        etree.SubElement(tree, "link", name=name)
    for i, name in enumerate(virtual):
        tw = twist[name]
        kind = "prismatic" if any(tw[:3]) else "revolute"
        axis = tw[:3] if kind == "prismatic" else tw[3:]
        j = etree.SubElement(tree, "joint", name=name, type=kind)
        etree.SubElement(j, "parent", link=links[i])
        etree.SubElement(j, "child", link=links[i + 1])
        etree.SubElement(j, "origin", xyz="0 0 0", rpy="0 0 0")
        etree.SubElement(j, "axis", xyz=" ".join(str(float(a)) for a in axis))
        etree.SubElement(j, "limit", lower="-1e4", upper="1e4", effort="0", velocity="1e4")
    return etree.tostring(tree, xml_declaration=True, encoding="utf-8").decode()


class CudaBackends:
    """Per-robot holder that lazily builds URDF-dependent backends.

    Args:
        urdf:                The source URDF, retained so backends can be built
                             on demand.
        parent_indices:      Concrete (host) joint parent-index array, kept so
                             the CUDA IK ancestor masks can be precomputed
                             without materialising the *traced* Robot pytree —
                             this is what makes CUDA IK usable inside ``jax.jit``.
        parent_joint_indices: Concrete (host) link → parent-joint index array.
        num_joints:          Total joint count (mask width).
    """

    def __init__(
        self,
        urdf: "yourdfpy.URDF",
        parent_indices=None,
        parent_joint_indices=None,
        num_joints: int | None = None,
        floating_base: bool = False,
    ) -> None:
        self._urdf = urdf
        self.floating_base = floating_base
        # Lazily-populated caches (built on first use).
        self._robot_collision: Any = None            # pure-JAX RobotCollision
        self._robot_collision_spherized: Any = None  # pure-JAX RobotCollisionSpherized
        self._sdf_cuda_collision: Any = None         # CUDA differentiable SDF
        self._binary_cuda_collision: Any = None      # CUDA binary checker
        self._grid: dict[float, Any] = {}            # GRiDDynamics, keyed by gravity

        # Concrete kinematic structure for host-side (trace-free) precompute.
        self._parent_indices = (
            None if parent_indices is None else onp.asarray(parent_indices, dtype=onp.int32)
        )
        self._parent_joint_indices = (
            None
            if parent_joint_indices is None
            else onp.asarray(parent_joint_indices, dtype=onp.int32)
        )
        self._num_joints = num_joints

    @property
    def urdf(self) -> "yourdfpy.URDF":
        return self._urdf

    def urdf_xml(self, joint_names: tuple[str, ...]) -> str:
        """The URDF GRiM compiles kernels from: a floating robot's is its chart's."""
        xml = _urdf_xml(self._urdf)
        if getattr(self, "_chart_of", None) is not None:
            xml = _chart_urdf_xml(xml, self._urdf.base_link, joint_names)
        return xml

    # ── Collision backends ─────────────────────────────────────────────────

    def robot_collision(self) -> Any:
        """Pure-JAX :class:`RobotCollision` model (used when ``use_cuda=False``)."""
        if self._robot_collision is None:
            from .collision import RobotCollision

            # The build does array math on the (concrete) URDF; guard it so it
            # evaluates eagerly even when this lazy first-use happens to occur
            # inside a caller's jax.jit trace (otherwise the parser's arrays
            # become tracers and Python `if array:` checks raise).
            with _eager():
                self._robot_collision = RobotCollision.from_urdf(self._urdf)
        return self._robot_collision

    def robot_collision_spherized(self) -> Any:
        """Pure-JAX :class:`RobotCollisionSpherized` model (sphere-based)."""
        if self._robot_collision_spherized is None:
            from .collision import RobotCollisionSpherized

            with _eager():
                self._robot_collision_spherized = RobotCollisionSpherized.from_urdf(
                    self._urdf
                )
        return self._robot_collision_spherized

    def sdf_collision(self) -> Any:
        """Differentiable CUDA SDF collision checker (``method="sdf"``)."""
        if self._sdf_cuda_collision is None:
            from .collision import make_cuda_checker

            with _eager():
                self._sdf_cuda_collision = make_cuda_checker(self.robot_collision())
        return self._sdf_cuda_collision

    def binary_collision(self) -> Any:
        """Fused CUDA binary (collision-free) checker (``method="binary"``).

        The binary kernel only supports sphere-based models, so it is built from
        a :class:`RobotCollisionSpherized` inner (not the capsule
        :class:`RobotCollision` used by the SDF path).
        """
        if self._binary_cuda_collision is None:
            from .collision import make_cuda_binary_checker

            with _eager():
                self._binary_cuda_collision = make_cuda_binary_checker(
                    self.robot_collision_spherized()
                )
        return self._binary_cuda_collision

    # ── Dynamics backend ───────────────────────────────────────────────────

    def grid(self, gravity: float) -> Any:
        """GRiD CUDA dynamics backend for the given ``gravity``.

        Cached per gravity value, since ``gravity`` is baked into the
        generated kernels at construction time.
        """
        key = float(gravity)
        gd = self._grid.get(key)
        if gd is None:
            from .dynamics import GRiDDynamics

            with _eager():
                gd = GRiDDynamics(self._urdf, gravity=key, floating_base=self.floating_base)
            self._grid[key] = gd
        return gd
