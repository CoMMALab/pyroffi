"""RoboSuite/MuJoCo scene bridge + kinematic plan replay.

Composes a RoboSuite scene (Panda robot model + BoxObjects for the cubes) in the
*planning frame* (Panda base at the origin, table top at z=0), then replays a
PDDLStream plan **kinematically** — SPaSM's plans are kinematic (REPORT.md §4.1),
so we drive the arm joints along the planned trajectories and carry the grasped
cube with the end-effector rather than relying on contact-based grasping. The
end-effector pose is taken from ``backend.get_ee_pose`` (identical Franka
kinematics to the MuJoCo model), so the same joint angles the pyroffi IK produced
place the hand exactly over each cube.

``execute_plan`` returns a frame timeline (arm config + every object's pose per
frame) consumed by :mod:`benchmarks.tamp.render`, plus a success flag (every cube
inside its goal region, mutually collision-free).
"""
from __future__ import annotations

import numpy as np

from . import _setup  # noqa: F401
from spasm import backend
from . import geometry as g

_COLORS = [
    [0.91, 0.08, 0.09, 1], [1.0, 0.65, 0.0, 1], [0.98, 0.92, 0.21, 1],
    [0.47, 0.76, 0.08, 1], [0.28, 0.49, 0.91, 1], [0.53, 0.21, 0.62, 1],
    [0.37, 0.71, 0.05, 1], [0.0, 0.6, 0.6, 1],
]


def yaw_to_wxyz(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def pose4_to_qpos7(pose):
    """[x,y,z,yaw] -> mujoco free-joint qpos [x,y,z, qw,qx,qy,qz]."""
    return np.concatenate([pose[:3], yaw_to_wxyz(pose[3])])


def compose_scene(world):
    """Build the MuJoCo model. Returns (model, info) where info carries qpos
    addresses for the arm joints and each cube's free joint."""
    import mujoco
    from robosuite.models import MujocoWorldBase
    from robosuite.models.objects import BoxObject
    from robosuite.models.robots import Panda

    import xml.etree.ElementTree as ET

    base = MujocoWorldBase()
    robot = Panda()
    robot.set_base_xpos([0, 0, 0])
    base.merge(robot)

    # Table surface (top at z=0, the planning frame) + a light, so the scene
    # reads as tabletop rearrangement rather than cubes floating in the void.
    table = ET.SubElement(base.worldbody, "geom")
    table.set("type", "box"); table.set("pos", "0.45 0.0 -0.021")
    table.set("size", "0.45 0.6 0.02"); table.set("rgba", "0.72 0.64 0.51 1")
    table.set("contype", "0"); table.set("conaffinity", "0")
    light = ET.SubElement(base.worldbody, "light")
    light.set("pos", "0.4 0.0 1.5"); light.set("dir", "0 0 -1")
    light.set("diffuse", "0.8 0.8 0.8"); light.set("directional", "true")

    for i, (name, pose) in enumerate(world.initial_poses.items()):
        half = world.block_half_height(name)
        box = BoxObject(name=name, size=[half, half, half],
                        rgba=_COLORS[i % len(_COLORS)], joints=[{"type": "free", "name": f"{name}_j"}])
        obj = box.get_obj()
        obj.set("pos", f"{pose[0]} {pose[1]} {pose[2]}")
        base.worldbody.append(obj)

    model = base.get_model(mode="mujoco")

    arm_adr = np.array([
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{j}")]
        for j in range(1, 8)])
    cube_adr = {
        name: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_j")]
        for name in world.initial_poses}
    return model, {"arm_adr": arm_adr, "cube_adr": cube_adr}


def _held_cube_pose(q, approach=g.TOP_DOWN_APPROACH):
    """World [x,y,z,yaw] of a cube grasped top-down at arm config q."""
    ee = np.asarray(backend.get_ee_pose(np.asarray(q, np.float32)))
    pos = ee[:3, 3]
    yaw = float(np.arctan2(ee[1, 0], ee[0, 0]))
    return np.array([pos[0], pos[1], pos[2] - approach, yaw])


def plan_to_timeline(world, plan, hold_frames=8):
    """Turn a PDDLStream plan into a list of frames: each frame is
    (arm_q7 np(7,), {cube_name: pose4}). Objects follow the eef while held."""
    obj_poses = {n: np.asarray(p, float).copy() for n, p in world.initial_poses.items()}
    arm_q = np.asarray(world.conf0, float).copy()
    holding = None
    frames = []

    def emit(q):
        poses = {n: p.copy() for n, p in obj_poses.items()}
        if holding is not None:
            obj_poses[holding] = _held_cube_pose(q)
            poses[holding] = obj_poses[holding].copy()
        frames.append((np.asarray(q, float).copy(), poses))

    for act in plan:
        name, args = act.name, act.args
        if name == "move":
            _, _q1, traj, _q2 = args
            for q in np.asarray(traj):
                arm_q = np.asarray(q, float)
                emit(arm_q)
        elif name == "pick":
            _, b, _p, _g, q = args
            arm_q = np.asarray(q, float)
            holding = b
            for _ in range(hold_frames):
                emit(arm_q)
        elif name in ("place", "place-on-block"):
            # rearrange place: (r,b,p,g,q); stack place: (r,b,p,g,q,bu,pu)
            b, p, q = args[1], args[2], args[4]
            arm_q = np.asarray(q, float)
            obj_poses[b] = np.asarray(p, float).copy()
            holding = None
            for _ in range(hold_frames):
                emit(arm_q)
    return frames


def success_from_poses(world, final_poses, tol=1e-6):
    """Goal check on the planned final poses: every cube inside its goal region
    and pairwise collision-free."""
    names = list(final_poses)
    for n in names:
        p = final_poses[n]
        reg = world.regions[world.goal_region[n]]
        if abs(p[0] - reg["cx"]) > reg["hx"] + tol or abs(p[1] - reg["cy"]) > reg["hy"] + tol:
            return False
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            if g.blocks_collide(world.blocks[a], final_poses[a],
                                world.blocks[b], final_poses[b]):
                return False
    return True


def stack_success_from_poses(world, final_poses, xy_tol=0.02):
    """Tower check: every stacked block sits directly above its support with the
    correct one-cube z gap (world.goal_region maps block -> support block)."""
    for b, bu in world.goal_region.items():
        pb, pu = final_poses[b], final_poses[bu]
        if abs(pb[0] - pu[0]) > xy_tol or abs(pb[1] - pu[1]) > xy_tol:
            return False
        gap = pb[2] - pu[2]
        want = world.block_half_height(b) + world.block_half_height(bu)
        if abs(gap - want) > 1e-3:
            return False
    return True


def execute_plan(world, plan, task="rearrange"):
    """Kinematic replay. Returns (frames, success, final_poses)."""
    frames = plan_to_timeline(world, plan)
    final_poses = frames[-1][1] if frames else dict(world.initial_poses)
    if task == "stack":
        success = stack_success_from_poses(world, final_poses)
    else:
        success = success_from_poses(world, final_poses)
    return frames, success, final_poses
