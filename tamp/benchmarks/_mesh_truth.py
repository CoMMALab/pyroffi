"""Mesh-level ground truth for validator quality, via pybullet.

Every backend in the throughput benchmark validates against a *sphere*
approximation of the Panda -- pyroffi and pybullet share a 59-sphere model,
cuRobo ships its own 61-sphere model plus a ``self_collision_buffer`` that
inflates it. Those are three different approximations of one robot, so
comparing them to each other only says they disagree, not which is right.

This is the thing they are all approximating: the actual collision meshes from
``panda.urdf``, checked with pybullet's mesh broadphase, with the same SRDF
pair exclusions the sphere models use and the same floor plane.

Against this reference a sphere model can be wrong in two directions, and they
are not equally bad:

    false ACCEPT   the meshes collide, the validator said the path was fine.
                   An unexecutable plan is handed to the robot.
    false REJECT   the meshes are clear, the validator refused the path.
                   Safe, but it shrinks the free space the planner can use --
                   this is what an inflated sphere model costs you.

A sphere model that circumscribes its meshes (the usual construction) should
show zero false accepts and some false rejects. The interesting quantity is how
*many* false rejects, because that is reachable workspace thrown away.
"""

from __future__ import annotations

import functools
import xml.etree.ElementTree as ET

import numpy as np

FLOOR_Z = -0.035

#: Actuated arm joints, in order. Fingers are held at the URDF default.
ARM_JOINTS = tuple(f"panda_joint{i}" for i in range(1, 8))


def _srdf_disabled(srdf_path):
    """``{(linkA, linkB)}`` pairs the SRDF marks as never-colliding."""
    root = ET.parse(srdf_path).getroot()
    out = set()
    for e in root.iter("disable_collisions"):
        a, b = e.get("link1"), e.get("link2")
        out.add((a, b))
        out.add((b, a))
    return out


@functools.lru_cache(maxsize=1)
def _session():
    import pybullet as pb
    import pybullet_data

    from spasm.paths import PYROFFI_ROOT
    import os

    urdf = os.path.join(PYROFFI_ROOT, "resources", "panda", "panda.urdf")
    srdf = os.path.join(PYROFFI_ROOT, "resources", "panda", "panda.srdf")

    cid = pb.connect(pb.DIRECT)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)

    # URDF_USE_SELF_COLLISION alone still excludes parent/child pairs, which is
    # what we want as a baseline; the SRDF list then removes the remaining pairs
    # the sphere models also ignore, so the reference and the models are asked
    # about the same set of link pairs.
    robot = pb.loadURDF(urdf, useFixedBase=True,
                        flags=pb.URDF_USE_SELF_COLLISION,
                        physicsClientId=cid)

    # Floor as a large thin box with its top face at FLOOR_Z.
    half = 2.0
    thick = 0.05
    shape = pb.createCollisionShape(pb.GEOM_BOX,
                                    halfExtents=[half, half, thick],
                                    physicsClientId=cid)
    floor = pb.createMultiBody(0, shape,
                               basePosition=[0, 0, FLOOR_Z - thick],
                               physicsClientId=cid)

    names = {}
    for j in range(pb.getNumJoints(robot, physicsClientId=cid)):
        info = pb.getJointInfo(robot, j, physicsClientId=cid)
        names[j] = info[12].decode()
    names[-1] = pb.getBodyInfo(robot, physicsClientId=cid)[0].decode()

    arm = []
    for want in ARM_JOINTS:
        for j in range(pb.getNumJoints(robot, physicsClientId=cid)):
            if pb.getJointInfo(robot, j, physicsClientId=cid)[1].decode() == want:
                arm.append(j)
                break
    assert len(arm) == 7, f"expected 7 arm joints, found {len(arm)}"

    return dict(cid=cid, robot=robot, floor=floor, arm=arm,
                link_name=names, disabled=_srdf_disabled(srdf))


def config_valid(q7, tol=0.0):
    """True when the *meshes* are clear at this configuration.

    ``tol`` is a penetration tolerance in metres: contacts shallower than this
    are ignored. Mesh contact at exactly 0 depth is numerically fragile, so a
    small positive value avoids counting grazing touches as collisions.
    """
    import pybullet as pb

    s = _session()
    cid, robot = s["cid"], s["robot"]
    for j, qi in zip(s["arm"], np.asarray(q7)[:7]):
        pb.resetJointState(robot, j, float(qi), physicsClientId=cid)

    pb.performCollisionDetection(physicsClientId=cid)

    for c in pb.getContactPoints(bodyA=robot, bodyB=robot, physicsClientId=cid):
        if c[8] >= -tol:                       # contactDistance
            continue
        a, b = s["link_name"].get(c[3]), s["link_name"].get(c[4])
        if (a, b) in s["disabled"]:
            continue
        return False

    for c in pb.getContactPoints(bodyA=robot, bodyB=s["floor"],
                                 physicsClientId=cid):
        if c[8] < -tol:
            return False
    return True


def path_valid(q_path, tol=0.0):
    """True when every waypoint on the path is mesh-clear."""
    return all(config_valid(q, tol) for q in np.asarray(q_path))


def paths_valid(paths, tol=0.0):
    """``[N, T, 7]`` -> ``[N]`` bool. Serial: this is a reference, not a rival."""
    return np.array([path_valid(p, tol) for p in np.asarray(paths)])
