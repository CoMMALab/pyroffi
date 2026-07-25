"""The viser scene graph pyroffi owns, driven by a :class:`WorldSource`.

Split from the source on purpose: this module knows how to draw a robot, a set
of named primitives, a path and a frame, and knows nothing about where any of
it came from. Point it at a simulator today and a perception stack later and
the picture is produced by the same code, which is what makes a render from
one comparable to a render from the other.

Node names are stable and namespaced (``/robot``, ``/objects/<name>``,
``/paths/<name>``), so an object's pose update is a handle write rather than a
scene rebuild — the difference between a viewer that tracks a 500 Hz rollout
and one that stutters.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from ._world import ObjectGeometry, Pose, WorldDescription, WorldState

_ROBOT_NODE = "/robot"
_OBJECT_ROOT = "/objects"
_PATH_ROOT = "/paths"
_FRAME_ROOT = "/frames"


class SceneView:
    """Builds and updates a viser scene from a description plus states."""

    def __init__(
        self,
        target,
        description: WorldDescription,
        show_collision_meshes: bool = False,
    ) -> None:
        """
        Args:
            target: a ``viser.ViserServer`` or ``viser.ClientHandle``.
            description: static scene content, from ``WorldSource.describe()``.
            show_collision_meshes: draw the URDF's collision geometry instead of
                its visual meshes. Worth turning on when a plan disagrees with
                what the collision checker thinks — the spherized Panda's
                collision hull is visibly not its visual mesh.
        """
        self._target = target
        self._description = description
        self._objects: dict[str, Any] = {}
        self._paths: dict[str, Any] = {}
        self._frames: dict[str, Any] = {}
        self._urdf = None
        self._last_state: WorldState | None = None

        if description.ground_plane:
            target.scene.add_grid(
                "/ground", width=4.0, height=4.0, cell_size=0.1, position=(0, 0, 0.0)
            )

        if description.robot_urdf is not None:
            from viser.extras import ViserUrdf

            self._urdf = ViserUrdf(
                target,
                description.robot_urdf,
                root_node_name=_ROBOT_NODE,
                load_meshes=not show_collision_meshes,
                load_collision_meshes=show_collision_meshes,
            )
            # ViserUrdf drives joints positionally; capture its ordering once so
            # every later update can be name-keyed without re-deriving it.
            self._urdf_joint_order = tuple(self._urdf.get_actuated_joint_names())
        else:
            self._urdf_joint_order = ()

        for geom in description.objects:
            self._add_object(geom)

    # ── static content ───────────────────────────────────────────────────

    def _add_object(self, geom: ObjectGeometry) -> None:
        node = f"{_OBJECT_ROOT}/{geom.name}"
        p = geom.params
        color = tuple(int(c) for c in geom.color)
        scene = self._target.scene
        try:
            if geom.shape == "box":
                handle = scene.add_box(
                    node,
                    dimensions=(
                        float(p["length"]),
                        float(p["width"]),
                        float(p["height"]),
                    ),
                    color=color,
                    opacity=float(geom.opacity),
                )
            elif geom.shape == "sphere":
                handle = scene.add_icosphere(
                    node, radius=float(p["radius"]), color=color,
                    opacity=float(geom.opacity),
                )
            elif geom.shape == "capsule":
                # viser has no capsule; a cylinder of the same radius and height
                # is the honest approximation and is labelled as such in docs.
                handle = scene.add_cylinder(
                    node,
                    radius=float(p["radius"]),
                    height=float(p["height"]),
                    color=color,
                    opacity=float(geom.opacity),
                )
            elif geom.shape == "mesh":
                handle = scene.add_mesh_trimesh(node, p["mesh"])
            else:
                logger.warning(f"skipping object {geom.name!r}: unknown shape {geom.shape!r}")
                return
        except Exception as exc:  # a scene we cannot draw is not a fatal error
            logger.warning(f"could not add object {geom.name!r}: {exc}")
            return
        self._objects[geom.name] = handle

    def add_object(self, geom: ObjectGeometry, pose: Pose | None = None) -> None:
        """Add an object that was not in the original description."""
        if geom.name in self._objects:
            self.remove_object(geom.name)
        self._add_object(geom)
        if pose is not None:
            self.set_object_pose(geom.name, pose)

    def remove_object(self, name: str) -> None:
        handle = self._objects.pop(name, None)
        if handle is not None:
            handle.remove()

    # ── dynamic content ──────────────────────────────────────────────────

    def update(self, state: WorldState) -> None:
        """Push one :class:`WorldState` into the scene.

        Missing entries are *held*, not defaulted: a source that reports the
        blocks but not the arm leaves the arm where it was, rather than
        snapping it to zero and showing a robot that never existed.
        """
        if self._urdf is not None and state.joint_values:
            cfg = np.array(
                [
                    float(state.joint_values.get(name, self._held(name)))
                    for name in self._urdf_joint_order
                ]
            )
            self._urdf.update_cfg(cfg)
        for name, pose in state.object_poses.items():
            self.set_object_pose(name, pose)
        self._last_state = state

    def _held(self, joint: str) -> float:
        if self._last_state is None:
            return 0.0
        return float(self._last_state.joint_values.get(joint, 0.0))

    def set_object_pose(self, name: str, pose: Pose) -> None:
        handle = self._objects.get(name)
        if handle is None:
            return
        handle.position = tuple(float(v) for v in pose.position)
        handle.wxyz = tuple(float(v) for v in pose.wxyz)

    # ── annotations an agent or a human asked for ────────────────────────

    def draw_path(
        self,
        name: str,
        positions: np.ndarray,
        color: tuple[int, int, int] = (255, 160, 40),
        width: float = 2.0,
    ) -> None:
        """Draw a Cartesian polyline — usually an end-effector path."""
        positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
        if positions.shape[0] < 2:
            return
        self.clear_path(name)
        self._paths[name] = self._target.scene.add_spline_catmull_rom(
            f"{_PATH_ROOT}/{name}", positions, color=color, line_width=width
        )

    def clear_path(self, name: str) -> None:
        handle = self._paths.pop(name, None)
        if handle is not None:
            handle.remove()

    def draw_frame(self, name: str, pose: Pose, axes_length: float = 0.1) -> None:
        """Draw a labelled coordinate frame — a grasp target, a goal pose."""
        handle = self._frames.get(name)
        if handle is None:
            self._frames[name] = self._target.scene.add_frame(
                f"{_FRAME_ROOT}/{name}",
                axes_length=axes_length,
                axes_radius=axes_length * 0.02,
                position=tuple(float(v) for v in pose.position),
                wxyz=tuple(float(v) for v in pose.wxyz),
            )
        else:
            handle.position = tuple(float(v) for v in pose.position)
            handle.wxyz = tuple(float(v) for v in pose.wxyz)

    def clear_frame(self, name: str) -> None:
        handle = self._frames.pop(name, None)
        if handle is not None:
            handle.remove()

    def clear_annotations(self) -> None:
        for name in list(self._paths):
            self.clear_path(name)
        for name in list(self._frames):
            self.clear_frame(name)
