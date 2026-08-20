"""PDDLStream streams backed entirely by pyroffi / SPaSM geometry.

Every geometric primitive the classical planner calls — grasp sampling,
placement sampling, inverse kinematics, motion connection, collision tests —
routes through :mod:`spasm.tamp.geometry`, i.e. through ``spasm.backend`` and
``spasm.tetris.solve``. There is **no pybullet**: the geometry backend is shared
by every configuration compared here, so any difference between them is
attributable to the *motion backend* (see :mod:`spasm.tamp.motion`), not to a
faster collision/FK/IK implementation.

``make_stream_map(world)`` returns the ``stream_map`` dict wired into a
``PDDLProblem`` (see :mod:`spasm.tamp.problems`).
"""
from __future__ import annotations

import numpy as np

from pddlstream.language.generator import from_gen_fn, from_fn, from_test
from pddlstream.language.constants import Output

from . import geometry as g

# Object representations passed around as PDDL constants:
#   pose  : np.ndarray (4,)  [x, y, z, yaw]
#   grasp : np.ndarray (1,)  [wrist_yaw_offset]   (top-down)
#   conf  : np.ndarray (7,)  joint angles
#   traj  : np.ndarray (T,7) joint path


def make_stream_map(world, collisions=True, motion_samples=20, max_place_tries=200,
                    motion_backend="linear", motion_params=None):
    """Build the ``stream_map``.

    ``motion_backend`` selects how ``s-motion`` answers — ``"kinematic"``
    (straight-line, geometry-only: the stock-SPaSM regime) or ``"dynamics"``
    (pyroffi TOPP-RA under actuator torque limits, which *rejects* segments
    admitting no feasible schedule). See :mod:`spasm.tamp.motion`. Everything
    else about the stream map is identical across the two, so the symbolic
    search is unchanged and any difference is attributable to the motion
    backend alone.
    """
    from .motion import MotionParams, make_planner

    if motion_params is None:
        motion_params = MotionParams(n_waypoints=motion_samples)
    plan_motion = make_planner(motion_backend, motion_params)

    rng = np.random.default_rng(world.seed)

    # --- s-grasp: one canonical top-down grasp per block ------------------ #
    def s_grasp(b):
        return Output(np.array([0.0], dtype=float))

    # --- s-region: sample a resting placement inside region r ------------- #
    def s_region_gen(b, r):
        region = world.regions[r]
        half_h = world.block_half_height(b)
        z = region["z"] + half_h
        tries = 0
        while tries < max_place_tries:
            tries += 1
            x = rng.uniform(region["cx"] - region["hx"], region["cx"] + region["hx"])
            y = rng.uniform(region["cy"] - region["hy"], region["cy"] + region["hy"])
            yaw = rng.uniform(-np.pi, np.pi)
            yield Output(np.array([x, y, z, yaw], dtype=float))

    # --- s-ik: pyroffi analytic Franka IK for a top-down grasp ------------ #
    def s_ik(b, p, grasp):
        wrist_yaw = float(p[3] + grasp[0])
        q, reachable = g.ik_topdown(p, grasp_yaw=wrist_yaw)
        if not reachable:
            return None
        return Output(q.astype(float))

    # --- s-motion: motion connector (backend-selected; see spasm.tamp.motion) #
    def s_motion(q1, q2):
        path = plan_motion(q1, q2)
        if path is None:
            return None
        return Output(np.asarray(path).astype(float))

    # --- t-cfree: SPaSM sphere-sphere penetration between two placements -- #
    def t_cfree(b1, p1, b2, p2):
        if not collisions:
            return True
        return not g.blocks_collide(world.blocks[b1], p1, world.blocks[b2], p2)

    # --- t-region: pose lies within region rectangle --------------------- #
    def t_region(b, p, r):
        region = world.regions[r]
        return (abs(p[0] - region["cx"]) <= region["hx"] + 1e-6 and
                abs(p[1] - region["cy"]) <= region["hy"] + 1e-6)

    # --- dist: joint-space L2 (move-action cost) ------------------------- #
    def dist_fn(q1, q2):
        return float(np.linalg.norm(np.asarray(q1)[:7] - np.asarray(q2)[:7]))

    return {
        "s-grasp": from_fn(s_grasp),
        "s-region": from_gen_fn(s_region_gen),
        "s-ik": from_fn(s_ik),
        "s-motion": from_fn(s_motion),
        "t-cfree": from_test(t_cfree),
        "t-region": from_test(t_region),
        "dist": dist_fn,
    }


def make_stack_stream_map(world, motion_samples=20,
                          motion_backend="linear", motion_params=None):
    """Stream map for the stacking domain (deterministic stack poses).

    ``motion_backend`` has the same meaning as in :func:`make_stream_map`.
    """
    from .motion import MotionParams, make_planner

    if motion_params is None:
        motion_params = MotionParams(n_waypoints=motion_samples)
    plan_motion = make_planner(motion_backend, motion_params)

    def s_grasp(b):
        return Output(np.array([0.0], dtype=float))

    def s_stack_pose(b, bu, pu):
        # Rest b directly on top of bu: same xy/yaw, z lifted by one cube.
        half_b = world.block_half_height(b)
        half_u = world.block_half_height(bu)
        p = np.array([pu[0], pu[1], pu[2] + half_u + half_b, pu[3]], dtype=float)
        return Output(p)

    def s_ik(b, p, grasp):
        wrist_yaw = float(p[3] + grasp[0])
        q, reachable = g.ik_topdown(p, grasp_yaw=wrist_yaw)
        return None if not reachable else Output(q.astype(float))

    def s_motion(q1, q2):
        path = plan_motion(q1, q2)
        return None if path is None else Output(np.asarray(path).astype(float))

    def dist_fn(q1, q2):
        return float(np.linalg.norm(np.asarray(q1)[:7] - np.asarray(q2)[:7]))

    return {
        "s-grasp": from_fn(s_grasp),
        "s-stack-pose": from_fn(s_stack_pose),
        "s-ik": from_fn(s_ik),
        "s-motion": from_fn(s_motion),
        "dist": dist_fn,
    }
