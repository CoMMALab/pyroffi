"""Host-side collision data for the CUDA analytic-IK kernel.

The kernel already computes the cumulative screw transforms ``T_i = E₁…E_i``
while scoring a candidate's pose error. Those same transforms place the robot's
collision spheres, so collision costs almost nothing on top of scoring — as
long as the sphere data is expressed in a form the kernel can apply directly.

That form is: each sphere's **world position at the home configuration**, plus
the index of the last joint that moves it. A sphere on link *i* sits at
``T_i · p_home``, because the space-frame product of exponentials maps a point
from its home world position to its current one. So the kernel needs no link
home poses, no URDF traversal and no local-frame bookkeeping — one gather and
one transform per sphere.

Self-collision pairs are pre-expanded host-side from the SRDF-derived active
*link* pairs into an explicit list of *sphere* pairs, so the kernel loops a flat
array instead of maintaining per-link sphere ranges.

.. warning::
   The collision model must be built with an SRDF. Without one the spherized
   model's conservative enclosure leaves adjacent links overlapping by
   construction (self-clearance ≈ -0.03 m even at the neutral pose), so every
   configuration reports as colliding and collision-free IK silently returns
   "nothing is reachable" — a wrong answer rather than a crash.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .._robot import Robot


@dataclass(frozen=True)
class CollisionData:
    """Flattened collision description consumed by the CUDA kernel."""

    spheres_home: np.ndarray   # [K, 4] world-frame (x, y, z, r) at q = 0
    sphere_joint: np.ndarray   # [K] int32, last joint index (0..7) moving it
    self_pairs: np.ndarray     # [P, 2] int32, sphere-index pairs to check

    @property
    def num_spheres(self) -> int:
        return int(self.spheres_home.shape[0])

    @property
    def num_pairs(self) -> int:
        return int(self.self_pairs.shape[0])


def _link_joint_depth(robot: "Robot", n_dof: int = 7) -> np.ndarray:
    """For each link, how many actuated joints of the chain move it.

    A link's world pose under the space-frame PoE is ``E₁…E_d · (home pose)``
    where ``d`` is that count, so this is exactly the transform index the kernel
    should apply to spheres on that link.
    """
    parent_joint = np.asarray(robot.links.parent_joint_indices)
    joint_parent = np.asarray(robot.joints.parent_indices)
    act_idx = np.asarray(robot.joints.actuated_indices)

    depth = np.zeros(len(robot.links.names), dtype=np.int32)
    for li in range(len(robot.links.names)):
        j = int(parent_joint[li])
        count = 0
        guard = 0
        while j != -1:
            a = int(act_idx[j])
            # Only actuated joints within the arm's 7-DOF chain advance the
            # transform index; fixed joints and the fingers do not.
            if 0 <= a < n_dof:
                count = max(count, a + 1)
            j = int(joint_parent[j])
            guard += 1
            if guard > joint_parent.shape[0] + 1:
                raise ValueError("cycle walking the kinematic chain")
        depth[li] = count
    return depth


def build_collision_data(robot: "Robot", robot_coll, n_dof: int = 7) -> CollisionData:
    """Extract :class:`CollisionData` from a ``RobotCollisionSpherized``.

    Args:
        robot_coll: must have been built with an SRDF (see the module warning).
    """
    import jax.numpy as jnp
    import jaxlie

    n_link, n_sph = robot_coll.coll.get_batch_axes()
    local = np.asarray(robot_coll.coll.pose.translation()).reshape(n_link, n_sph, 3)
    radii = np.asarray(robot_coll.coll.radius).reshape(n_link, n_sph)

    # Link poses at the home configuration, to lift local sphere centres into
    # the world frame the kernel's screw transforms operate in.
    n_act = len(robot.joints.actuated_names)
    T_home = np.asarray(jaxlie.SE3(robot.forward_kinematics(jnp.zeros(n_act))).as_matrix())

    depth = _link_joint_depth(robot, n_dof)

    centres, rads, joints, link_of = [], [], [], []
    for li in range(n_link):
        T = T_home[li]
        for si in range(n_sph):
            r = float(radii[li, si])
            if r <= 0.0:            # padding slot (negative-radius sentinel)
                continue
            p = T[:3, :3] @ local[li, si] + T[:3, 3]
            centres.append(p)
            rads.append(r)
            joints.append(depth[li])
            link_of.append(li)

    spheres_home = np.concatenate(
        [np.asarray(centres, dtype=np.float64),
         np.asarray(rads, dtype=np.float64)[:, None]], axis=1)
    sphere_joint = np.asarray(joints, dtype=np.int32)
    link_of = np.asarray(link_of, dtype=np.int32)

    # Expand the SRDF-derived active LINK pairs into explicit SPHERE pairs.
    idx_i = np.asarray(robot_coll.active_idx_i)
    idx_j = np.asarray(robot_coll.active_idx_j)
    pairs = []
    for a, b in zip(idx_i, idx_j):
        sa = np.nonzero(link_of == int(a))[0]
        sb = np.nonzero(link_of == int(b))[0]
        for u in sa:
            for v in sb:
                pairs.append((u, v))
    self_pairs = (np.asarray(pairs, dtype=np.int32) if pairs
                  else np.zeros((0, 2), dtype=np.int32))

    return CollisionData(spheres_home=spheres_home,
                         sphere_joint=sphere_joint,
                         self_pairs=self_pairs)


@dataclass(frozen=True)
class WorldGeometry:
    """World obstacles in the layout every CUDA IK solver in the suite expects.

    Same four buffers and the same per-row layouts as ``ls``/``hjcd``/``sqp``/
    ``mppi``, so a world built for any of them works with the analytic solver
    unchanged (and vice versa).
    """

    spheres: np.ndarray      # [Ms, 4]  centre, radius
    capsules: np.ndarray     # [Mc, 7]
    boxes: np.ndarray        # [Mb, 15]
    halfspaces: np.ndarray   # [Mh, 6]

    @property
    def is_empty(self) -> bool:
        return not (len(self.spheres) or len(self.capsules)
                    or len(self.boxes) or len(self.halfspaces))


def _empty(n_cols: int) -> np.ndarray:
    return np.zeros((0, n_cols), dtype=np.float32)


EMPTY_WORLD = WorldGeometry(_empty(4), _empty(7), _empty(15), _empty(6))


def world_geometry(*geoms) -> WorldGeometry:
    """Pack one or more :mod:`pyroffi.collision` CollGeoms for the CUDA kernel.

    Accepts ``Sphere``, ``Capsule``, ``Box`` and ``HalfSpace`` batches in any
    combination. The robot side is spheres-only by construction
    (``RobotCollisionSpherized`` *is* a sphere model), so only sphere-vs-X
    distances are ever evaluated — but the world may be any mix of primitives.
    """
    from ..collision._geometry import Capsule, Sphere

    spheres, capsules, boxes, halfspaces = [], [], [], []

    for g in geoms:
        if g is None:
            continue
        name = type(g).__name__
        # Row layouts are dictated by the SDFs in _collision_cuda_helpers.cuh,
        # NOT by the CollGeom field order. They differ in ways that are easy to
        # get wrong and that fail silently rather than loudly:
        #   capsule    two endpoints + radius   (not centre + axis + radius)
        #   box        centre, axis1..3, halflen; the axes are the ROTATION'S
        #              COLUMNS, since the kernel dots the offset with each
        #              (world -> local is R^T)
        #   halfspace  NORMAL first, then a point on the plane
        if isinstance(g, Sphere) or name == "Sphere":
            c = np.asarray(g.pose.translation()).reshape(-1, 3)
            r = np.asarray(g.radius).reshape(-1, 1)
            spheres.append(np.concatenate([c, r], axis=1))
        elif isinstance(g, Capsule) or name == "Capsule":
            c = np.asarray(g.pose.translation()).reshape(-1, 3)
            R = np.asarray(g.pose.rotation().as_matrix()).reshape(-1, 3, 3)
            axis = R[:, :, 2]                       # capsule runs along local z
            half = np.asarray(g.height).reshape(-1, 1) * 0.5
            r = np.asarray(g.radius).reshape(-1, 1)
            capsules.append(np.concatenate(
                [c - half * axis, c + half * axis, r], axis=1))
        elif name == "Box":
            c = np.asarray(g.pose.translation()).reshape(-1, 3)
            R = np.asarray(g.pose.rotation().as_matrix()).reshape(-1, 3, 3)
            h = np.asarray(g.half_extents).reshape(-1, 3)
            boxes.append(np.concatenate(
                [c, R[:, :, 0], R[:, :, 1], R[:, :, 2], h], axis=1))
        elif name in ("HalfSpace", "Halfspace", "Plane"):
            p_ = np.asarray(g.pose.translation()).reshape(-1, 3)
            R = np.asarray(g.pose.rotation().as_matrix()).reshape(-1, 3, 3)
            n_ = R[:, :, 2]                          # plane normal = local z
            halfspaces.append(np.concatenate([n_, p_], axis=1))
        else:
            raise TypeError(
                f"world_geometry: unsupported CollGeom {name!r}; expected "
                "Sphere, Capsule, Box or HalfSpace")

    def _stack(parts, n_cols):
        return (np.concatenate(parts, axis=0).astype(np.float32) if parts
                else _empty(n_cols))

    return WorldGeometry(_stack(spheres, 4), _stack(capsules, 7),
                         _stack(boxes, 15), _stack(halfspaces, 6))


def world_spheres_array(world_geom) -> np.ndarray:
    """Deprecated: use :func:`world_geometry`, which handles every primitive."""
    return world_geometry(world_geom).spheres
