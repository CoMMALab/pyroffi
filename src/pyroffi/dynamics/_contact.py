"""Grasp / fixed-contact modeling for contact-rich trajectory optimization.

This module provides the differentiable JAX building blocks used by
:mod:`pyroffi.optimization_engines._contact_trajopt` to make a multi-manipulator
manipulation problem *dynamics-aware*, for **any** robot morphology (not just
a pair of arms):

* :class:`ManipulatorSpec` — one manipulator (a :class:`~pyroffi.Robot` plus a
  :class:`~pyroffi.dynamics.GRiDDynamics` kernel), with a world base transform
  and a designated grip link + contact point.
* :class:`GraspedObject` — a rigid object being grasped, described by a
  :class:`~pyroffi.collision.CollGeom` primitive (``Box``, ``Sphere``, ...)
  carrying its own ``mass`` / ``inertia_diag`` / ``friction``.
* :class:`ContactSystem` — an arbitrary number of :class:`ManipulatorSpec`
  rigidly grasping one :class:`GraspedObject`, with per-manipulator world base
  transforms. Concrete morphologies (bimanual, trimanual, single-arm + fixture,
  ...) are assembled by callers from these building blocks; nothing here is
  specific to any particular robot count or shape.
* :func:`grasp_closure_residual` — the *fixed-contact* holonomic constraint:
  the relative pose of every manipulator's gripper to a reference gripper must
  stay equal to the pose captured at grasp time (the object is rigid).
* :func:`object_dynamics_residual` — Newton-Euler force/torque balance of the
  grasped object, linking the per-contact forces to the object's motion.
* :func:`grip_validity_penalty` — friction-cone + minimum-normal-force
  feasibility of each contact, using the object's own friction coefficient by
  default.
* :func:`manipulator_contact_fext` — maps a world-frame contact force at a
  gripper into the per-body external wrench array consumed by
  ``GRiDDynamics.inverse_dynamics``.

Conventions
-----------
* Contact forces are 3-vectors in **world axes** (point contact; grip moments
  are supplied by the pinch, captured via :func:`grip_validity_penalty`).
* Each manipulator's world base transform is restricted to a translation plus
  a yaw about world +z, so gravity stays along the manipulator model's -z axis
  (GRiD applies gravity along its own model axis) and world<->base is a planar
  rotation.
* The grasped object's gravity acts at its centre, so it contributes no moment
  about the centre; the rotational balance reduces to the contact-force
  moments.
* The object's "centre" is approximated as the centroid of the world-frame
  contact points (no separate free-floating object state is tracked).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jaxlie
from jax import Array
from jaxtyping import Float

if TYPE_CHECKING:
    from .._robot import Robot
    from ..collision._geometry import CollGeom
    from ._grid_dynamics import GRiDDynamics


@dataclasses.dataclass(frozen=True, eq=False)
class ManipulatorSpec:
    """One manipulator in a contact system.

    ``base_xy_yaw`` is ``(x, y, yaw)``: the manipulator's base is placed at
    world ``(x, y, 0)`` and rotated by ``yaw`` about world +z. ``grip_link``
    is the link whose frame defines the gripper; ``p_local`` is the contact
    point in that link's frame.

    Parallel-jaw pinch model (used by :func:`parallel_jaw_grip_penalty`):
    ``close_axis_local`` is the gripper's finger-closing direction in the
    grip-link frame (the two pads face each other along this axis); ``+z`` and
    friction act in the plane perpendicular to it. ``f_grip_max`` is the applied
    clamp force magnitude (N) available at the pads -- friction capacity in the
    pad plane is ``2 * mu * f_grip_max``, squeeze capacity along the axis is
    ``f_grip_max``. For the Franka hand the fingers close along the hand-frame
    ``y`` axis, so ``close_axis_local = (0, 1, 0)``.
    """

    robot: "Robot"
    grid: "GRiDDynamics"
    grip_link: str
    base_xy_yaw: tuple[float, float, float] = (0.0, 0.0, 0.0)
    p_local: tuple[float, float, float] = (0.0, 0.0, 0.0)
    close_axis_local: tuple[float, float, float] = (0.0, 1.0, 0.0)
    f_grip_max: float = 40.0

    @property
    def grip_link_index(self) -> int:
        return self.robot.links.names.index(self.grip_link)

    @property
    def num_dof(self) -> int:
        return int(self.grid.num_dof)

    def base_se3(self) -> jaxlie.SE3:
        x, y, yaw = self.base_xy_yaw
        return jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.from_z_radians(jnp.asarray(yaw, jnp.float64)),
            jnp.array([x, y, 0.0], jnp.float64),
        )


@dataclasses.dataclass(frozen=True, eq=False)
class GraspedObject:
    """A rigid object being grasped, described by a collision primitive.

    ``geom`` carries its own physical parameters (``mass``, ``inertia_diag``,
    ``friction``) — see :class:`pyroffi.collision.CollGeom`.
    """

    geom: "CollGeom"

    @property
    def mass(self) -> Array:
        return jnp.asarray(self.geom.mass)

    @property
    def friction(self) -> Array:
        return jnp.asarray(self.geom.friction)

    @property
    def inertia_diag(self) -> Array:
        """Principal rotational inertia about the object's centre (local axes)."""
        return jnp.asarray(self.geom.inertia_diag)


@dataclasses.dataclass(frozen=True, eq=False)
class ContactSystem:
    """An arbitrary number of manipulators rigidly grasping one object.

    ``manipulators`` is an ordered tuple; ``manipulators[0]`` is the reference
    gripper. ``grasp_offsets[i]`` is the constant reference->manipulator[i+1]
    relative gripper transform (``T_ref^{-1} @ T_i`` in world), captured at
    the grasp configuration; the fixed-contact constraint pins the trajectory
    to it. ``len(grasp_offsets) == len(manipulators) - 1``.
    """

    manipulators: tuple[ManipulatorSpec, ...]
    body: GraspedObject
    grasp_offsets: tuple[jaxlie.SE3, ...]
    gravity: float = 9.81

    def __post_init__(self) -> None:
        if len(self.manipulators) < 1:
            raise ValueError("ContactSystem requires at least one manipulator.")
        if len(self.grasp_offsets) != len(self.manipulators) - 1:
            raise ValueError(
                "grasp_offsets must have exactly len(manipulators) - 1 entries "
                f"(got {len(self.grasp_offsets)} for {len(self.manipulators)} "
                "manipulators)."
            )

    @property
    def num_manipulators(self) -> int:
        return len(self.manipulators)

    @property
    def num_dof(self) -> int:
        return sum(m.num_dof for m in self.manipulators)

    def split_q(self, q: Float[Array, "*b n"]) -> tuple[Array, ...]:
        """Split a stacked ``[q_0 | q_1 | ...]`` config into per-manipulator configs."""
        qs = []
        idx = 0
        for m in self.manipulators:
            qs.append(q[..., idx : idx + m.num_dof])
            idx += m.num_dof
        return tuple(qs)


# ---------------------------------------------------------------------------
# Kinematics helpers
# ---------------------------------------------------------------------------

def _gripper_world_pose(m: ManipulatorSpec, q: Float[Array, "n"]) -> jaxlie.SE3:
    """World SE3 of the manipulator's gripper link at config ``q``."""
    link_poses = m.robot.forward_kinematics(q)  # (n_links, 7)
    grip_local = jaxlie.SE3(link_poses[m.grip_link_index])
    return m.base_se3() @ grip_local


def _contact_point_world(m: ManipulatorSpec, q: Float[Array, "n"]) -> Array:
    """World position of the manipulator's contact point."""
    return _gripper_world_pose(m, q).apply(jnp.asarray(m.p_local, jnp.float32))


def gripper_poses(
    system: ContactSystem, q: Float[Array, "n"]
) -> tuple[jaxlie.SE3, ...]:
    """World gripper poses for every manipulator, from a stacked config."""
    qs = system.split_q(q)
    return tuple(
        _gripper_world_pose(m, qm) for m, qm in zip(system.manipulators, qs)
    )


def contact_points_world(
    system: ContactSystem, q: Float[Array, "n"]
) -> tuple[Array, ...]:
    """World-frame contact point for every manipulator, from a stacked config."""
    qs = system.split_q(q)
    return tuple(
        _contact_point_world(m, qm) for m, qm in zip(system.manipulators, qs)
    )


def object_center_world(system: ContactSystem, q: Float[Array, "n"]) -> Array:
    """Grasped-object centre = centroid of the world-frame contact points."""
    pts = contact_points_world(system, q)
    return sum(pts) / len(pts)


# ---------------------------------------------------------------------------
# Fixed-contact (grasp closure) constraint
# ---------------------------------------------------------------------------

def grasp_closure_residual(
    system: ContactSystem, q: Float[Array, "n"]
) -> Float[Array, "6*(k-1)"]:
    """Fixed-contact residual: se(3) error of every non-reference gripper's
    pose (relative to the reference gripper) against its captured grasp
    offset. Zero iff the object is held rigidly. ``k`` is the manipulator
    count; the residual is empty for a single-manipulator system.
    """
    poses = gripper_poses(system, q)
    T_ref = poses[0]
    errs = []
    for T_i, offset_i in zip(poses[1:], system.grasp_offsets):
        rel = T_ref.inverse() @ T_i
        errs.append((rel @ offset_i.inverse()).log())  # (6,) — twist [v; omega]
    if not errs:
        return jnp.zeros((0,))
    return jnp.concatenate(errs)


# ---------------------------------------------------------------------------
# Object Newton-Euler balance
# ---------------------------------------------------------------------------

def object_dynamics_residual(
    system: ContactSystem,
    q: Float[Array, "n"],
    a_obj: Float[Array, "3"],
    forces: Float[Array, "k 3"],
) -> Float[Array, "6"]:
    """Newton-Euler residual of the grasped object (force ; torque about centre).

    ``a_obj`` is the world linear acceleration of the object's centroid.
    ``forces[i]`` is the world-frame contact force applied by manipulator
    ``i``. Gravity acts at the centre (no moment). The angular term is a
    quasi-static balance of the contact-force moments (object angular inertia
    is small for typical lift/carry motions).
    """
    g_vec = jnp.array([0.0, 0.0, -system.gravity], jnp.float32)
    c = object_center_world(system, q)
    pts = contact_points_world(system, q)

    force_res = system.body.mass * (a_obj - g_vec) - jnp.sum(forces, axis=0)
    torque_res = jnp.zeros((3,), forces.dtype)
    for p, f in zip(pts, forces):
        torque_res = torque_res + jnp.cross(p - c, f)
    return jnp.concatenate([force_res, torque_res])


# ---------------------------------------------------------------------------
# Grip validity (friction cone + pushing normal)
# ---------------------------------------------------------------------------

def _grip_inward_normal(m: ManipulatorSpec, q: Float[Array, "n"], toward: Array) -> Array:
    """Unit world vector from the contact point toward the object centre."""
    p = _contact_point_world(m, q)
    v = toward - p
    return v / (jnp.linalg.norm(v) + 1e-9)


def grip_validity_penalty(
    system: ContactSystem,
    q: Float[Array, "n"],
    forces: Float[Array, "k 3"],
    mu_friction: float | None,
    f_min: float,
) -> Array:
    """Squared-hinge penalty for an infeasible grip across all contacts.

    Each contact force must push *into* the object (normal component
    ``>= f_min``) and lie within the Coulomb friction cone
    ``||f_t|| <= mu * f_n``. If ``mu_friction`` is ``None``, the grasped
    object's own ``geom.friction`` is used.
    """
    mu = system.body.friction if mu_friction is None else mu_friction
    c = object_center_world(system, q)
    qs = system.split_q(q)

    def per_contact(m, qm, f):
        n = _grip_inward_normal(m, qm, c)
        f_n = jnp.dot(f, n)
        f_t = f - f_n * n
        push = jnp.maximum(0.0, f_min - f_n) ** 2
        cone = jnp.maximum(0.0, jnp.linalg.norm(f_t) - mu * f_n) ** 2
        return push + cone

    total = jnp.array(0.0, forces.dtype)
    for m, qm, f in zip(system.manipulators, qs, forces):
        total = total + per_contact(m, qm, f)
    return total


def _closing_axis_world(m: ManipulatorSpec, q: Float[Array, "n"]) -> Array:
    """Unit world direction of the gripper's finger-closing axis at ``q``."""
    R = _gripper_world_pose(m, q).rotation()
    a = R.apply(jnp.asarray(m.close_axis_local, jnp.float32))
    return a / (jnp.linalg.norm(a) + 1e-9)


def parallel_jaw_grip_penalty(
    system: ContactSystem,
    q: Float[Array, "n"],
    forces: Float[Array, "k 3"],
    mu_friction: float | None,
) -> Array:
    """Squared-hinge grip-feasibility penalty for a **parallel-jaw pinch**.

    Unlike :func:`grip_validity_penalty` (a unilateral fingertip-push model whose
    "inward normal" degenerates to zero whenever the contact point coincides with
    the object centroid -- i.e. every single-arm grasp), this models a two-pad
    clamp explicitly. Decompose each manipulator's world contact force ``f`` about
    the closing axis ``a``:

      * ``f_axis = (f . a) a``  -- along the closing axis: resisted by the clamp,
        capacity ``f_grip_max`` (pushing along the axis unloads one pad -> escape).
      * ``f_shear = f - f_axis`` -- in the pad plane: resisted by Coulomb friction
        at both pads, capacity ``2 * mu * f_grip_max``.

    Zero iff the grip is feasible. ``mu`` defaults to the grasped object's own
    ``geom.friction``. Physically correct behaviour: a top-down pinch (closing
    axis horizontal) bears the object weight as *shear*, so it slips when
    ``mass * g > 2 * mu * f_grip_max`` -- monotone in both ``mu`` and mass.
    """
    mu = system.body.friction if mu_friction is None else mu_friction
    qs = system.split_q(q)

    def per_contact(m, qm, f):
        a = _closing_axis_world(m, qm)
        f_ax = jnp.dot(f, a)
        f_shear = f - f_ax * a
        fg = m.f_grip_max
        squeeze = jnp.maximum(0.0, jnp.abs(f_ax) - fg) ** 2
        shear = jnp.maximum(0.0, jnp.linalg.norm(f_shear) - 2.0 * mu * fg) ** 2
        return squeeze + shear

    total = jnp.array(0.0, forces.dtype)
    for m, qm, f in zip(system.manipulators, qs, forces):
        total = total + per_contact(m, qm, f)
    return total


# ---------------------------------------------------------------------------
# External-wrench assembly for GRiD inverse dynamics
# ---------------------------------------------------------------------------

def manipulator_contact_fext(
    m: ManipulatorSpec,
    q: Float[Array, "n"],
    f_world: Float[Array, "3"],
) -> Float[Array, "n 6"]:
    """Per-body external wrench array for ``GRiDDynamics.inverse_dynamics``.

    The grasped object reacts on the manipulator with ``-f_world`` at the
    contact point. This is transferred to the last actuated body's frame
    origin and expressed in the manipulator's base-frame axes (the frame
    GRiD's dynamics live in), then placed on the last body row (all others
    zero).
    """
    n = m.num_dof
    base = m.base_se3()
    R_base_inv = base.rotation().inverse()  # world -> base rotation

    # Reaction force on the manipulator, in base axes.
    f_base = R_base_inv.apply(-f_world)

    # Contact point and last-body origin, both in base axes.
    _, r_world = m.grid.jacobian(q)  # r_world: (n_body, 3) in base frame
    r_last = r_world[..., -1, :]  # already base frame (grid uses manipulator base)
    grip_base = jaxlie.SE3(m.robot.forward_kinematics(q)[m.grip_link_index])
    p_contact_base = grip_base.apply(jnp.asarray(m.p_local, f_base.dtype))

    tau_base = jnp.cross(p_contact_base - r_last, f_base)
    wrench = jnp.concatenate([tau_base, f_base])  # [torque; force]

    fext = jnp.zeros((n, 6), f_base.dtype)
    return fext.at[-1].set(wrench)


def capture_grasp_offsets(
    manipulators: tuple[ManipulatorSpec, ...], qs: tuple[Array, ...]
) -> tuple[jaxlie.SE3, ...]:
    """Constant reference->manipulator[i] relative gripper transforms at a
    grasp config. ``manipulators[0]`` / ``qs[0]`` is the reference.
    """
    poses = [_gripper_world_pose(m, q) for m, q in zip(manipulators, qs)]
    T_ref = poses[0]
    return tuple(T_ref.inverse() @ T_i for T_i in poses[1:])
