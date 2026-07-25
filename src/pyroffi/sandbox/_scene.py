"""Building the MuJoCo world a task is executed in.

The scene is assembled from the same task specification the agent reads, so
the world the agent is told about and the world it is scored in cannot drift
apart — there is one source of truth and both sides derive from it.

**Two robot models, on purpose.** Planning runs on pyroffi's
``panda_spherized`` (7 DOF, primitive collision geometry, the model whose
self-collision calibration is actually reliable) while execution runs on the
MuJoCo Menagerie Franka, which has real finger meshes and a tendon-driven
gripper. That is only sound because their kinematics agree: measured over
random configurations, ``panda_hand`` and the Menagerie ``hand`` body differ
by **1e-7 m** in position and 1e-7 in quaternion, so a joint-space plan
transfers with no translation layer at all. The spherized URDF's own fingers
are *fixed*, and its finger collision spheres have 3.8 cm radii that
interpenetrate each other at every opening — it cannot express a pinch, which
is why it is not the execution model.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Sequence

import numpy as np

GRIPPER_OPEN = 255.0
"""Menagerie's gripper actuator is force-like on a split tendon: full scale
opens, zero closes. Named because the polarity is the opposite of what
'0 = no command' suggests."""
GRIPPER_CLOSE = 0.0

MENAGERIE_ARM_JOINTS: tuple[str, ...] = tuple(f"joint{i}" for i in range(1, 8))
"""Menagerie names its arm joints ``joint1..7``; pyroffi's URDF names the same
joints, in the same order, ``panda_joint1..7``. Kept as an explicit table
because the two names are close enough that a prefix rule would look right and
be wrong."""


@dataclasses.dataclass
class SceneHandles:
    """Everything the sandbox needs to address the compiled model."""

    model: Any
    data: Any
    hand_body: int
    finger_geoms: tuple[int, ...]
    block_bodies: dict[str, int]
    block_geoms: dict[str, int]
    block_qpos_adr: dict[str, int]
    obstacle_geoms: dict[str, int]
    block_size: np.ndarray


def build_scene(
    blocks: Sequence[Mapping[str, Any]],
    obstacles: Sequence[Mapping[str, Any]] = (),
    block_size: Sequence[float] = (0.05, 0.05, 0.05),
    block_mass: float = 0.1,
    block_friction: float = 1.5,
    timestep: float = 0.002,
) -> SceneHandles:
    """Compile the Franka plus a set of free blocks and static obstacles.

    Blocks are genuinely dynamic free bodies with real collision geometry:
    nothing is welded or teleported, so a tower that is knocked over falls
    over, and a block released from 10 cm up lands where physics puts it.
    """
    import mujoco
    from robot_descriptions import panda_mj_description

    spec = mujoco.MjSpec.from_file(panda_mj_description.MJCF_PATH)
    spec.option.timestep = float(timestep)

    spec.worldbody.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[2.0, 2.0, 0.1],
        rgba=[0.55, 0.56, 0.60, 1.0],
        friction=[1.0, 0.005, 0.0001],
    )

    half = np.asarray(block_size, dtype=np.float64) / 2.0
    for block in blocks:
        body = spec.worldbody.add_body(
            name=block["name"], pos=[float(v) for v in block["position"]]
        )
        body.add_freejoint()
        geom = body.add_geom(
            name=f"{block['name']}_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=half.tolist(),
            rgba=list(block.get("rgba", (0.8, 0.3, 0.3, 1.0))),
            friction=[float(block_friction), 0.01, 0.001],
        )
        geom.mass = float(block_mass)

    for obs in obstacles:
        params = obs["params"]
        spec.worldbody.add_geom(
            name=obs["name"],
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[float(v) for v in obs["position"]],
            quat=[float(v) for v in obs.get("wxyz", (1.0, 0.0, 0.0, 0.0))],
            size=[
                float(params["length"]) / 2.0,
                float(params["width"]) / 2.0,
                float(params["height"]) / 2.0,
            ],
            rgba=[0.45, 0.45, 0.50, 1.0],
        )

    model = spec.compile()
    data = mujoco.MjData(model)

    def gid(name: str) -> int:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if i < 0:
            raise ValueError(f"no geom named {name!r} in the compiled model")
        return int(i)

    def bid(name: str) -> int:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if i < 0:
            raise ValueError(f"no body named {name!r} in the compiled model")
        return int(i)

    finger_bodies = {bid("left_finger"), bid("right_finger")}
    finger_geoms = tuple(
        i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in finger_bodies
    )

    block_bodies = {b["name"]: bid(b["name"]) for b in blocks}
    return SceneHandles(
        model=model,
        data=data,
        hand_body=bid("hand"),
        finger_geoms=finger_geoms,
        block_bodies=block_bodies,
        block_geoms={b["name"]: gid(f"{b['name']}_geom") for b in blocks},
        block_qpos_adr={
            name: int(model.jnt_qposadr[model.body_jntadr[body]])
            for name, body in block_bodies.items()
        },
        obstacle_geoms={o["name"]: gid(o["name"]) for o in obstacles},
        block_size=np.asarray(block_size, dtype=np.float64),
    )


def viewer_geometry(
    blocks: Sequence[Mapping[str, Any]],
    obstacles: Sequence[Mapping[str, Any]],
    block_size: Sequence[float],
):
    """The same objects, described for :mod:`pyroffi.viewer`."""
    from ..viewer import ObjectGeometry

    size = np.asarray(block_size, dtype=np.float64)
    out = [
        ObjectGeometry(
            name=b["name"],
            shape="box",
            params={"length": size[0], "width": size[1], "height": size[2]},
            color=tuple(int(255 * c) for c in b.get("rgba", (0.8, 0.3, 0.3))[:3]),
        )
        for b in blocks
    ]
    out += [
        ObjectGeometry(
            name=o["name"],
            shape="box",
            params=dict(o["params"]),
            color=(115, 115, 128),
        )
        for o in obstacles
    ]
    return tuple(out)
