"""pybullet geometric oracle for PDDLStream — the incumbent baseline.

PDDLStream's reference implementation is pybullet-backed (``ss-pybullet``), so
this is what a practitioner picking PDDLStream off the shelf actually runs. It
exists here to answer "is pyroffi a viable motion validator?" against the thing
it would be replacing, rather than only against SPaSM's hand-rolled kinematics.

It implements the same three primitives the other validators do — the interface
:mod:`spasm.tamp.geometry` exposes to the streams:

    ik_topdown(pose, grasp_yaw, approach) -> (q7, reachable)
    arm_path_valid(q_path)                -> bool
    blocks_collide(...)                   -> bool   (shared, pose-level)

Two deliberate choices, both to keep the comparison about the *backend* rather
than about incidental setup:

* **Same URDF as the pyroffi collision model** (``panda_spherized.urdf``). Using
  pybullet's usual mesh collision would conflate two differences at once —
  implementation *and* geometric representation. Spheres keep the geometry
  identical across validators so only the implementation differs.
* **Numerical IK.** ``calculateInverseKinematics`` is damped least squares, not
  a closed form. That *is* the honest comparison: it is what pybullet offers,
  and the contrast with an analytic solver is a real property of the backend,
  not a handicap imposed on it.

Runs headless (``DIRECT``); no GUI, no rendering.
"""
from __future__ import annotations

import functools

import numpy as np

from spasm.paths import PANDA_URDF

#: Grasp frame the top-down IK targets, matching ``backend.EE_LINK``.
EE_LINK_NAME = "panda_grasptarget"

#: Table height, matching ``geometry.FLOOR_Z``.
FLOOR_Z = -0.035

TOP_DOWN_APPROACH = 0.10

#: Seed configuration for the iterative solver, matching the neutral pose the
#: other validators use as their continuity reference.
REST_POSE = (0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4)

#: Position tolerance for calling a numerical IK solution "reached" (metres).
#: pybullet's IK is iterative and returns its best effort with no success flag,
#: so acceptance has to be decided by checking FK against the target — the
#: solver will happily return a configuration 20cm away.
IK_POS_TOL = 5e-3
IK_ORN_TOL = 5e-2


@functools.lru_cache(maxsize=1)
def _session():
    """Headless pybullet client with the Panda loaded. Built once."""
    import pybullet as pb
    import pybullet_data

    cid = pb.connect(pb.DIRECT)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
    robot = pb.loadURDF(PANDA_URDF, useFixedBase=True, physicsClientId=cid)

    # Actuated revolute joints, base to tip, plus the grasp-frame link index.
    joints, ee_index = [], None
    for j in range(pb.getNumJoints(robot, physicsClientId=cid)):
        info = pb.getJointInfo(robot, j, physicsClientId=cid)
        if info[2] == pb.JOINT_REVOLUTE:
            joints.append(j)
        if info[12].decode() == EE_LINK_NAME:
            ee_index = j
    if ee_index is None:
        raise RuntimeError(
            f"link {EE_LINK_NAME!r} not found in {PANDA_URDF}; pybullet IK has "
            "no frame to target")

    arm = joints[:7]
    lower = [pb.getJointInfo(robot, j, physicsClientId=cid)[8] for j in arm]
    upper = [pb.getJointInfo(robot, j, physicsClientId=cid)[9] for j in arm]
    ranges = [u - l for l, u in zip(lower, upper)]
    return dict(cid=cid, robot=robot, arm=arm, ee=ee_index,
                lower=lower, upper=upper, ranges=ranges)


def _set_arm(s, q):
    import pybullet as pb
    for j, v in zip(s["arm"], np.asarray(q)[:7]):
        pb.resetJointState(s["robot"], j, float(v), physicsClientId=s["cid"])


def _ee_pose(s):
    import pybullet as pb
    st = pb.getLinkState(s["robot"], s["ee"], computeForwardKinematics=True,
                         physicsClientId=s["cid"])
    return np.asarray(st[4]), np.asarray(st[5])   # world pos, world quat xyzw


def ik_topdown(pose, grasp_yaw=0.0, approach=TOP_DOWN_APPROACH):
    """Top-down grasp IK. Returns ``(q7, reachable)``.

    ``reachable`` is decided by checking FK against the target, not by trusting
    the solver: pybullet's IK returns a best-effort configuration with no
    success flag, so an unreachable target yields a plausible-looking answer
    that simply does not reach. Treating that as success would silently feed
    the planner invalid grasps.
    """
    import pybullet as pb

    s = _session()
    pose = np.asarray(pose, dtype=float)
    target = [float(pose[0]), float(pose[1]), float(pose[2]) + approach]

    yaw = float(pose[3] + grasp_yaw)
    # Top-down: z-axis pointing down, rotated by yaw about world z.
    orn = pb.getQuaternionFromEuler([np.pi, 0.0, yaw])

    # Plain damped-least-squares IK, no null-space arguments.
    #
    # pybullet's null-space form (lowerLimits/upperLimits/jointRanges/restPoses)
    # requires lists covering EVERY movable DOF in the chain -- 9 here, 7 arm
    # joints plus 2 fingers -- not just the joints being solved for. Passing 7
    # made pybullet silently ignore the limits: it returned q4 = +0.084 when
    # joint 4's range is entirely negative, and the solution missed the target
    # by 0.139m. Seeding from the rest pose and clamping afterwards is both
    # correct and closer to what ss-pybullet actually does.
    for j, v in zip(s["arm"], REST_POSE):
        pb.resetJointState(s["robot"], j, v, physicsClientId=s["cid"])
    sol = pb.calculateInverseKinematics(
        s["robot"], s["ee"], target, orn,
        maxNumIterations=200, residualThreshold=1e-4,
        physicsClientId=s["cid"])
    q = np.clip(np.asarray(sol[:7], dtype=float),
                np.asarray(s["lower"]), np.asarray(s["upper"]))

    _set_arm(s, q)
    pos, _ = _ee_pose(s)
    reached = float(np.linalg.norm(pos - np.asarray(target))) <= IK_POS_TOL
    within = bool(np.all(q >= np.asarray(s["lower"]) - 1e-6) and
                  np.all(q <= np.asarray(s["upper"]) + 1e-6))
    return q, bool(reached and within)


def arm_path_valid(q_path, floor_z=FLOOR_Z):
    """Joint limits, self-collision and floor clearance along a path.

    Self-collision uses pybullet's own broadphase rather than a sphere-pair
    list, which is the point of including it: this is what the incumbent
    backend actually checks.
    """
    import pybullet as pb

    s = _session()
    q_path = np.asarray(q_path)
    lower, upper = np.asarray(s["lower"]), np.asarray(s["upper"])
    if np.any(q_path[:, :7] < lower - 1e-3) or np.any(q_path[:, :7] > upper + 1e-3):
        return False

    for q in q_path:
        _set_arm(s, q)
        pb.performCollisionDetection(physicsClientId=s["cid"])
        if pb.getContactPoints(bodyA=s["robot"], bodyB=s["robot"],
                               physicsClientId=s["cid"]):
            return False
        # Floor clearance from the link AABBs; no plane body is loaded, so the
        # table is a half-space test rather than a collision pair.
        for link in [-1] + list(range(pb.getNumJoints(s["robot"],
                                                      physicsClientId=s["cid"]))):
            aabb_min, _ = pb.getAABB(s["robot"], link, physicsClientId=s["cid"])
            if aabb_min[2] < floor_z:
                return False
    return True


def interpolate(q1, q2, n=20):
    """Straight-line joint path, identical to the other validators'."""
    q1 = np.asarray(q1)[:7]
    q2 = np.asarray(q2)[:7]
    ts = np.linspace(0.0, 1.0, n)[:, None]
    return (1.0 - ts) * q1[None] + ts * q2[None]
