"""What the render layer draws, and where it comes from.

The rendering stack is split so that **the source is the only thing that
changes between a simulator and the real world**. A MuJoCo rollout, a
kinematic pyroffi configuration, and a perception stack watching a physical
cell all describe the same thing — where the robot is and where the objects
are — and all three reduce to a :class:`WorldState` here. Everything
downstream (:mod:`._scene_view`, :mod:`._render_viewer`, and any agent looking
at the result) is written against that and never learns which one it got.

Two halves, because they change at different rates:

* :meth:`WorldSource.describe` — the *static* content: the robot's URDF and the
  named objects with their geometry. Read once, when the scene graph is built.
* :meth:`WorldSource.read` — the *dynamic* state: joint values and object
  poses, read every frame.

A source that cannot answer part of a read (a perception stack that sees the
blocks but not the arm) returns what it has; the view holds the last known
value for the rest rather than snapping to a default.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

IDENTITY_WXYZ = (1.0, 0.0, 0.0, 0.0)


@dataclasses.dataclass(frozen=True)
class Pose:
    """A world-frame SE(3) pose. Metres, quaternion scalar-first (``wxyz``)."""

    position: np.ndarray
    wxyz: np.ndarray = dataclasses.field(
        default_factory=lambda: np.asarray(IDENTITY_WXYZ, dtype=np.float64)
    )

    @staticmethod
    def of(position: Sequence[float], wxyz: Sequence[float] = IDENTITY_WXYZ) -> "Pose":
        return Pose(
            np.asarray(position, dtype=np.float64).reshape(3),
            np.asarray(wxyz, dtype=np.float64).reshape(4),
        )

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "position": [float(v) for v in self.position],
            "wxyz": [float(v) for v in self.wxyz],
        }


@dataclasses.dataclass(frozen=True)
class ObjectGeometry:
    """A drawable named object.

    ``shape`` is one of ``box`` / ``sphere`` / ``capsule`` / ``mesh``, with
    ``params`` in metres following the same convention as
    :mod:`pyroffi.toolbox._scene`: box takes ``length``/``width``/``height``
    (full extents), sphere ``radius``, capsule ``radius``/``height``, mesh a
    ``trimesh`` object under ``mesh``.
    """

    name: str
    shape: str
    params: Mapping[str, Any]
    color: tuple[int, int, int] = (180, 180, 190)
    opacity: float = 1.0


@dataclasses.dataclass
class WorldDescription:
    """The static content of a scene: what exists, not where it is."""

    robot_urdf: Any = None
    """A ``yourdfpy.URDF``, or None for a source with no articulated robot."""
    joint_names: tuple[str, ...] = ()
    """Actuated joint names, in the order ``ViserUrdf`` expects them."""
    objects: tuple[ObjectGeometry, ...] = ()
    ground_plane: bool = True
    name: str = "world"


@dataclasses.dataclass
class WorldState:
    """The dynamic state of a scene at one instant.

    Joint values are **name-keyed**, never positional: the render layer sits
    downstream of the same interop contract as the rest of pyroffi's
    boundaries (see :mod:`pyroffi.toolbox._exchange`), and a source that
    silently reorders joints produces a picture that is wrong in a way nobody
    notices.
    """

    joint_values: dict[str, float] = dataclasses.field(default_factory=dict)
    object_poses: dict[str, Pose] = dataclasses.field(default_factory=dict)
    time: float = 0.0
    extras: dict[str, Any] = dataclasses.field(default_factory=dict)
    """Whatever the source knows and the view does not need: contact counts, the
    held object, tracking error. Passed through to observers untouched."""


@runtime_checkable
class WorldSource(Protocol):
    """Anything that can say where the robot and the objects are.

    Implement this to render from somewhere new — a different simulator, a
    replay log, or a perception stack. Nothing above this protocol needs to
    change.
    """

    def describe(self) -> WorldDescription:
        """Static scene content. Called once, when the scene graph is built."""

    def read(self) -> WorldState:
        """Current state. Called every frame; must be cheap and must not block."""


# ── the sources that ship ────────────────────────────────────────────────────


class CallableSource:
    """A source built from two plain callables.

    The seam for anything pyroffi does not own — most usefully a perception
    stack: hand it a function returning the poses your estimator produced and
    the render layer is unchanged from the simulated case.
    """

    def __init__(
        self,
        description: WorldDescription,
        read: Callable[[], WorldState],
    ) -> None:
        self._description = description
        self._read = read

    def describe(self) -> WorldDescription:
        return self._description

    def read(self) -> WorldState:
        return self._read()


class ToolboxSource:
    """Renders a :class:`pyroffi.toolbox.Session` — robot config plus scene objects.

    Purely kinematic: it shows where a configuration *puts* the robot, which is
    what you want when inspecting an IK solution or a planned path, and is not
    a claim about what a simulator would do with it.
    """

    _COLORS = {
        "box": (150, 160, 200),
        "sphere": (200, 150, 150),
        "capsule": (150, 200, 160),
        "halfspace": (120, 120, 120),
    }

    def __init__(self, session, config: np.ndarray | None = None) -> None:
        self.session = session
        self.config = (
            np.asarray(session.robot_state, dtype=np.float64)
            if config is None
            else np.asarray(config, dtype=np.float64)
        )

    def set_config(self, config: np.ndarray) -> None:
        self.config = np.asarray(config, dtype=np.float64).reshape(-1)

    def describe(self) -> WorldDescription:
        objects = []
        for obj in self.session.scene.objects():
            if obj.shape == "halfspace":
                continue  # drawn as the ground grid, not as a solid
            objects.append(
                ObjectGeometry(
                    name=obj.name,
                    shape=obj.shape,
                    params=dict(obj.params),
                    color=self._COLORS.get(obj.shape, (180, 180, 190)),
                )
            )
        return WorldDescription(
            robot_urdf=self.session.urdf,
            joint_names=tuple(self.session.joint_names),
            objects=tuple(objects),
            name=str(self.session.robot_spec),
        )

    def read(self) -> WorldState:
        s = self.session
        return WorldState(
            joint_values={
                n: float(v) for n, v in zip(s.joint_names, self.config)
            },
            object_poses={
                o.name: Pose.of(o.position, o.wxyz)
                for o in s.scene.objects()
                if o.shape != "halfspace"
            },
            extras={"scene_version": s.scene.version, "source": "toolbox"},
        )


class MuJoCoSource:
    """Renders a live ``mujoco.MjData`` — the simulator's actual state.

    Reads the *simulated* joint positions and body poses, so what is drawn is
    what the physics did, not what the plan asked for. That distinction is the
    whole point of having a simulator in the loop, and it is preserved by
    reading ``MjData`` rather than the reference trajectory.

    Args:
        model: the compiled ``mujoco.MjModel``.
        data: the ``mujoco.MjData`` being stepped. Read in place, so a viewer
            bound to it follows the simulation with no plumbing.
        joint_names: actuated joint names **as pyroffi knows them**. These are
            what the returned state is keyed by, because everything downstream
            speaks pyroffi's names.
        object_bodies: ``{scene object name: mujoco body name}`` for the free
            bodies to track.
        geometry: static geometry for those objects, since ``MjModel`` describes
            them in its own terms and the view wants pyroffi's.
        mujoco_joint_names: the corresponding names *in the MuJoCo model*,
            positionally aligned with ``joint_names``. Needed whenever the two
            models disagree on spelling — the MuJoCo Menagerie Franka calls its
            joints ``joint1..7`` where pyroffi's URDF says ``panda_joint1..7``,
            and silently reading the wrong joint draws a robot that is not the
            one being simulated. Defaults to ``joint_prefix + joint_names``.
        joint_prefix: prefix MuJoCo applied when the arm was attached
            (``spec.attach(..., prefix="arm_")``).
    """

    def __init__(
        self,
        model,
        data,
        joint_names: Sequence[str],
        object_bodies: Mapping[str, str] | None = None,
        geometry: Sequence[ObjectGeometry] = (),
        urdf=None,
        joint_prefix: str = "",
        mujoco_joint_names: Sequence[str] | None = None,
    ) -> None:
        import mujoco

        self.model = model
        self.data = data
        self.urdf = urdf
        self.joint_names = tuple(joint_names)
        self.geometry = tuple(geometry)
        self.object_bodies = dict(object_bodies or {})

        if mujoco_joint_names is None:
            lookup = [f"{joint_prefix}{n}" for n in self.joint_names]
        else:
            lookup = [f"{joint_prefix}{n}" for n in mujoco_joint_names]
            if len(lookup) != len(self.joint_names):
                raise ValueError(
                    f"mujoco_joint_names has {len(lookup)} entries for "
                    f"{len(self.joint_names)} joint_names; they are positionally aligned"
                )
        self.mujoco_joint_names = tuple(lookup)

        self._qpos_adr: list[int] = []
        for name in self.mujoco_joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(
                    f"no MuJoCo joint named {name!r}; the model has "
                    f"{[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]}"
                )
            self._qpos_adr.append(int(model.jnt_qposadr[jid]))

        self._body_ids = {}
        for obj_name, body_name in self.object_bodies.items():
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if bid < 0:
                raise ValueError(f"no MuJoCo body named {body_name!r}")
            self._body_ids[obj_name] = int(bid)

    def describe(self) -> WorldDescription:
        return WorldDescription(
            robot_urdf=self.urdf,
            joint_names=self.joint_names,
            objects=self.geometry,
            name="mujoco",
        )

    def read(self) -> WorldState:
        qpos = self.data.qpos
        return WorldState(
            joint_values={
                n: float(qpos[adr]) for n, adr in zip(self.joint_names, self._qpos_adr)
            },
            object_poses={
                name: Pose.of(self.data.xpos[bid], self.data.xquat[bid])
                for name, bid in self._body_ids.items()
            },
            time=float(self.data.time),
            extras={"source": "mujoco", "n_contacts": int(self.data.ncon)},
        )
