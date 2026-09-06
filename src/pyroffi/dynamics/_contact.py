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

Relationship to :mod:`pyroffi.attachments`
------------------------------------------

An :class:`~pyroffi.attachments.Attachment` is the same physical idea as a grasp
here — a rigid body carried by a link — and it is now the single source of truth
for that bookkeeping.  :meth:`ContactSystem.from_attachments` takes one
attachment per manipulator (the object in *that* manipulator's grip-link frame)
and **derives** ``grasp_offsets`` from them:

    ``T_W_obj = T_W_L_i · A_i``  for every ``i``, so
    ``T_ref^{-1} · T_i = A_ref · A_i^{-1}``

which is exactly the relative-gripper transform the closure residual already
uses — the residual maths is unchanged, it just stops being independently
captured.  :func:`capture_attachments` is the "close the grippers here"
constructor, and :func:`capture_grasp_offsets` is now a thin wrapper over it.

The division of labour between the two modules is deliberate:

* ``attachments`` supplies the *nominal rigid model* — one SE(3) compose for
  kinematics and one ``I_body`` row update for dynamics, so
  ``robot.with_attachments(system.attachment_set(i))`` gives a manipulator whose
  ``inverse_dynamics`` already carries the payload
  (:meth:`ContactSystem.loaded_manipulator_robot`).
* this module supplies the *certificates* that the rigid model is valid —
  :func:`grasp_closure_residual` (does the closed chain hold?),
  :func:`grip_validity_penalty` / :func:`parallel_jaw_grip_penalty` (can the
  fingers actually apply the required force?) and
  :func:`object_dynamics_residual` (do the allocated forces move the object the
  way the trajectory says?).  Slip is a residual, not a modelling change.

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

    @staticmethod
    def from_attachment(attachment) -> "GraspedObject":
        """Build from an :class:`~pyroffi.attachments.Attachment`'s geometry.

        The attachment already carries the physical parameters on its
        ``CollGeom``, so this is a view, not a re-declaration: the same object
        can be handed to the collision path (``compose_collision``), the
        dynamics path (``compose_dynamics``) and the contact residuals here
        without its mass being stated three times.
        """
        if attachment.geom is None:
            raise ValueError(
                f"attachment {attachment.name!r} carries no collision geometry, "
                "so it cannot describe a grasped object; build it with "
                "Attachment.from_geom(...)."
            )
        return GraspedObject(geom=attachment.geom[0])


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
    attachments: tuple = ()
    """Optional per-manipulator :class:`~pyroffi.attachments.Attachment`, the
    object expressed in that manipulator's grip-link frame. Set by
    :meth:`from_attachments`, from which ``grasp_offsets`` is then derived.
    Empty for a system built the older way, which stays fully supported."""

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

    # ── attachment interop ────────────────────────────────────────────────

    @staticmethod
    def from_attachments(
        manipulators: tuple[ManipulatorSpec, ...],
        attachments: tuple,
        gravity: float = 9.81,
        body: "GraspedObject | None" = None,
    ) -> "ContactSystem":
        """Build a system from one :class:`~pyroffi.attachments.Attachment` per
        manipulator, deriving the grasp offsets rather than capturing them.

        ``attachments[i]`` is the grasped object in ``manipulators[i]``'s
        grip-link frame.  Because every manipulator's attachment must predict
        the *same* world object pose, the reference-relative gripper transform
        the closure residual pins is exactly ``A_ref · A_i^{-1}`` — see the
        module docstring.  ``body`` defaults to the reference attachment's
        geometry.
        """
        if len(attachments) != len(manipulators):
            raise ValueError(
                "from_attachments needs exactly one attachment per manipulator "
                f"(got {len(attachments)} for {len(manipulators)})."
            )
        A = [jaxlie.SE3(a.T_parent_body) for a in attachments]
        for i, a in enumerate(attachments):
            if a.parent_link_index != manipulators[i].grip_link_index:
                raise ValueError(
                    f"attachment {a.name!r} hangs off link index "
                    f"{a.parent_link_index}, but manipulator {i} grips with "
                    f"link {manipulators[i].grip_link!r} (index "
                    f"{manipulators[i].grip_link_index}). The attachment must be "
                    "expressed in the grip link's frame."
                )
        offsets = tuple(A[0] @ A_i.inverse() for A_i in A[1:])
        return ContactSystem(
            manipulators=tuple(manipulators),
            body=body if body is not None else GraspedObject.from_attachment(
                attachments[0]
            ),
            grasp_offsets=offsets,
            gravity=gravity,
            attachments=tuple(attachments),
        )

    def attachment_set(self, index: int):
        """The ``AttachmentSet`` for one manipulator, ready for composition."""
        from ..attachments import AttachmentSet

        if not self.attachments:
            raise ValueError(
                "This ContactSystem was not built from attachments; use "
                "ContactSystem.from_attachments (or capture_attachments) to get "
                "one that can hand its payload to the dynamics/collision paths."
            )
        return AttachmentSet.empty().attach(self.attachments[index])

    def loaded_manipulator_robot(self, index: int):
        """``manipulators[index].robot`` with the grasped object's inertia
        folded in, so its ``inverse_dynamics`` carries the payload.

        This is the concrete payoff of the unification: the transport torques a
        planner sees now include the thing being transported, without threading
        a payload argument through the solver.
        """
        return self.manipulators[index].robot.with_attachments(
            self.attachment_set(index)
        )


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


def _point_from_pose(m: ManipulatorSpec, T_w: jaxlie.SE3) -> Array:
    """World contact point from an already-computed world gripper pose."""
    return T_w.apply(jnp.asarray(m.p_local, jnp.float32))


def _axis_from_pose(m: ManipulatorSpec, T_w: jaxlie.SE3) -> Array:
    """Unit world finger-closing axis from an already-computed gripper pose."""
    a = T_w.rotation().apply(jnp.asarray(m.close_axis_local, jnp.float32))
    return a / (jnp.linalg.norm(a) + 1e-9)


def _closure_from_poses(
    system: ContactSystem, poses: tuple[jaxlie.SE3, ...]
) -> Float[Array, "6*(k-1)"]:
    """Grasp-closure residual from already-computed world gripper poses."""
    T_ref = poses[0]
    errs = []
    for T_i, offset_i in zip(poses[1:], system.grasp_offsets):
        rel = T_ref.inverse() @ T_i
        errs.append((rel @ offset_i.inverse()).log())  # (6,) — twist [v; omega]
    if not errs:
        return jnp.zeros((0,))
    return jnp.concatenate(errs)


def grasp_kinematics_with_poses(
    system: ContactSystem, q: Float[Array, "n"]
) -> tuple[Array, Array, Array, Array, Array]:
    """Every per-configuration grasp quantity, from **one** FK pass per manipulator.

    All the residuals in this module are projections of the same gripper poses,
    so calling them individually made a trajopt cost function re-run forward
    kinematics five or six times per timestep — twice inside
    :func:`object_dynamics_residual` alone, which recomputes both the centre and
    the contact points its caller usually already holds. This computes the poses
    once and hands back the four bundles the residuals actually consume.

    Returns ``(pose_params, points, center, axes, closure)``:

    * ``pose_params`` ``(k, 7)``    — world gripper pose per manipulator, as
      ``jaxlie.SE3`` parameters; rebuild one with ``jaxlie.SE3(pose_params[i])``.
    * ``points``  ``(k, 3)``        — world contact point per manipulator.
    * ``center``  ``(3,)``          — object centre (centroid of ``points``).
    * ``axes``    ``(k, 3)``        — world finger-closing axis per manipulator.
    * ``closure`` ``(6*(k-1),)``    — grasp-closure residual.

    Plain arrays rather than a dataclass, so the bundle survives a ``jax.vmap``
    over a trajectory without needing a pytree registration.
    """
    qs = system.split_q(q)
    poses = tuple(
        _gripper_world_pose(m, qm) for m, qm in zip(system.manipulators, qs)
    )
    points = jnp.stack(
        [_point_from_pose(m, T_w) for m, T_w in zip(system.manipulators, poses)]
    )
    axes = jnp.stack(
        [_axis_from_pose(m, T_w) for m, T_w in zip(system.manipulators, poses)]
    )
    center = jnp.mean(points, axis=0)
    pose_params = jnp.stack([T_w.wxyz_xyz for T_w in poses])
    return pose_params, points, center, axes, _closure_from_poses(system, poses)


def grasp_kinematics(
    system: ContactSystem, q: Float[Array, "n"]
) -> tuple[Array, Array, Array, Array]:
    """:func:`grasp_kinematics_with_poses` without the gripper poses.

    Callers that do not need the poses themselves (the contact-rich solver only
    wants points/centre/axes/closure) should use this, so the traced graph never
    contains the pose bundle at all.

    Returns ``(points, center, axes, closure)`` — see
    :func:`grasp_kinematics_with_poses` for the meaning of each.
    """
    qs = system.split_q(q)
    poses = tuple(
        _gripper_world_pose(m, qm) for m, qm in zip(system.manipulators, qs)
    )
    points = jnp.stack(
        [_point_from_pose(m, T_w) for m, T_w in zip(system.manipulators, poses)]
    )
    axes = jnp.stack(
        [_axis_from_pose(m, T_w) for m, T_w in zip(system.manipulators, poses)]
    )
    center = jnp.mean(points, axis=0)
    return points, center, axes, _closure_from_poses(system, poses)


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
    """Grasped-object centre = centroid of the world-frame contact points.

    An approximation (no separate free-floating object state is tracked), kept
    because the residuals and both trajopt solvers are calibrated against it.
    When the system carries attachments, :func:`object_pose_world` gives the
    exact pose instead.
    """
    pts = contact_points_world(system, q)
    return sum(pts) / len(pts)


def object_pose_world(
    system: ContactSystem, q: Float[Array, "n"], index: int = 0
) -> jaxlie.SE3:
    """Exact world pose of the grasped object, ``T_W_L_i(q) · A_i``.

    Requires attachments (see :meth:`ContactSystem.from_attachments`). Every
    manipulator predicts the same pose *iff* the grasp closure residual is zero,
    so the spread across ``index`` is itself a closure diagnostic — and unlike
    :func:`object_center_world` this carries orientation, which is what a viewer
    or a placement goal needs.
    """
    if not system.attachments:
        raise ValueError(
            "object_pose_world needs a ContactSystem built from attachments; "
            "use ContactSystem.from_attachments or capture_attachments."
        )
    qs = system.split_q(q)
    T_W_L = _gripper_world_pose(system.manipulators[index], qs[index])
    return T_W_L @ jaxlie.SE3(system.attachments[index].T_parent_body)


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

    Callers that also need the contact points, object centre or closing axes
    should use :func:`grasp_kinematics` instead, which produces all four from a
    single FK pass.
    """
    return _closure_from_poses(system, gripper_poses(system, q))


# ---------------------------------------------------------------------------
# Manipulator dynamics feasibility (torque limits)
# ---------------------------------------------------------------------------

def dynamics_feasibility_residual(
    grid: "GRiDDynamics",
    q: Float[Array, "T ndof"],
    dt: float,
    tau_max: float,
    f_ext: Float[Array, "T ndof 6"] | None = None,
) -> Float[Array, "T ndof"]:
    """Torque-limit infeasibility ``relu(|tau| - tau_max)`` over a trajectory.

    Wraps ``grid.inverse_dynamics(q, qd, qdd, f_ext)`` (vmap-fused, differentiable
    via GRiD's analytic ``custom_jvp`` — no float64 twin needed) with joint
    velocities/accelerations from central finite differences at timestep ``dt``.
    Returns the per-``(timestep, joint)`` amount by which the required torque
    exceeds ``tau_max`` (zero where feasible), shaped like ``q``.

    This is the shared residual behind tier 1's optional dynamics-feasibility term
    and tier 3's torque-limit AL term, mirroring
    :func:`object_dynamics_residual`'s shape/naming so both can enter a generic
    :class:`~pyroffi.optimization_engines._trajopt_core.AugmentedLagrangianTerm`
    (``kind="ineq"``) with one implementation. Feed it a ``robot.with_attachments``
    GRiD to include a carried payload's transport torque.
    """
    from ..optimization_engines._contact_trajopt import _fd_vel_acc

    qd, qdd = _fd_vel_acc(q, dt)
    tau = grid.inverse_dynamics(q, qd, qdd, f_ext=f_ext)
    return jnp.maximum(0.0, jnp.abs(tau) - tau_max)


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

    Recomputes the object centre and contact points from ``q``; if you already
    hold them (via :func:`grasp_kinematics`) call
    :func:`object_dynamics_residual_at` instead and skip two FK passes.
    """
    return object_dynamics_residual_at(
        system,
        object_center_world(system, q),
        jnp.stack(contact_points_world(system, q)),
        a_obj,
        forces,
    )


def object_dynamics_residual_at(
    system: ContactSystem,
    center: Float[Array, "3"],
    points: Float[Array, "k 3"],
    a_obj: Float[Array, "3"],
    forces: Float[Array, "k 3"],
) -> Float[Array, "6"]:
    """:func:`object_dynamics_residual` from a precomputed centre/contact points."""
    g_vec = jnp.array([0.0, 0.0, -system.gravity], jnp.float32)
    force_res = system.body.mass * (a_obj - g_vec) - jnp.sum(forces, axis=0)
    torque_res = jnp.sum(jnp.cross(points - center, forces), axis=0)
    return jnp.concatenate([force_res, torque_res])


# ---------------------------------------------------------------------------
# Grip validity (friction cone + pushing normal)
# ---------------------------------------------------------------------------

def _safe_unit(v: Array) -> Array:
    """``v`` normalized, or exactly zero when ``v`` is (numerically) zero.

    ``v / (norm(v) + eps)`` is *not* a safe normalization: it returns a finite
    value at ``v = 0`` but ``norm`` is non-differentiable there and hands back a
    NaN gradient, which then contaminates everything downstream. The
    ``where``-guard keeps the zero branch out of the differentiated path.
    """
    n2 = jnp.sum(v * v)
    ok = n2 > 1e-18
    return jnp.where(ok, v / jnp.sqrt(jnp.where(ok, n2, 1.0)), 0.0)


def _safe_norm(v: Array) -> Array:
    """``norm(v)`` with a zero (rather than NaN) gradient at the origin."""
    n2 = jnp.sum(v * v)
    ok = n2 > 1e-18
    return jnp.where(ok, jnp.sqrt(jnp.where(ok, n2, 1.0)), 0.0)


def _grip_inward_normal(m: ManipulatorSpec, q: Float[Array, "n"], toward: Array) -> Array:
    """Unit world vector from the contact point toward the object centre.

    Zero when the contact point *is* the object centre, which is the
    single-manipulator case: there is no inward direction to point in. See
    :func:`_safe_unit` for why the guard matters to the gradient.
    """
    p = _contact_point_world(m, q)
    return _safe_unit(toward - p)


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

    Recomputes the centre and contact points from ``q``; callers holding them
    (via :func:`grasp_kinematics`) should use :func:`grip_validity_penalty_at`.
    """
    return grip_validity_penalty_at(
        system,
        object_center_world(system, q),
        jnp.stack(contact_points_world(system, q)),
        forces,
        mu_friction,
        f_min,
    )


def grip_validity_penalty_at(
    system: ContactSystem,
    center: Float[Array, "3"],
    points: Float[Array, "k 3"],
    forces: Float[Array, "k 3"],
    mu_friction: float | None,
    f_min: float,
) -> Array:
    """:func:`grip_validity_penalty` from a precomputed centre/contact points."""
    mu = system.body.friction if mu_friction is None else mu_friction

    def per_contact(p, f):
        n = _safe_unit(center - p)
        f_n = jnp.dot(f, n)
        f_t = f - f_n * n
        push = jnp.maximum(0.0, f_min - f_n) ** 2
        cone = jnp.maximum(0.0, _safe_norm(f_t) - mu * f_n) ** 2
        return push + cone

    total = jnp.array(0.0, forces.dtype)
    for p, f in zip(points, forces):
        total = total + per_contact(p, f)
    return total


def _closing_axis_world(m: ManipulatorSpec, q: Float[Array, "n"]) -> Array:
    """Unit world direction of the gripper's finger-closing axis at ``q``."""
    return _axis_from_pose(m, _gripper_world_pose(m, q))


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

    Recomputes each closing axis from ``q``; callers holding the axes already
    (via :func:`grasp_kinematics`) should use
    :func:`parallel_jaw_grip_penalty_at`.
    """
    qs = system.split_q(q)
    axes = jnp.stack(
        [_closing_axis_world(m, qm) for m, qm in zip(system.manipulators, qs)]
    )
    return parallel_jaw_grip_penalty_at(system, axes, forces, mu_friction)


def parallel_jaw_grip_penalty_at(
    system: ContactSystem,
    axes: Float[Array, "k 3"],
    forces: Float[Array, "k 3"],
    mu_friction: float | None,
) -> Array:
    """:func:`parallel_jaw_grip_penalty` from precomputed world closing axes."""
    mu = system.body.friction if mu_friction is None else mu_friction

    def per_contact(m, a, f):
        f_ax = jnp.dot(f, a)
        f_shear = f - f_ax * a
        fg = m.f_grip_max
        squeeze = jnp.maximum(0.0, jnp.abs(f_ax) - fg) ** 2
        # _safe_norm, not jnp.linalg.norm: f_shear is exactly zero whenever the
        # force lies purely along the closing axis, which is a perfectly ordinary
        # state for a decision variable to pass through (and is where a naive
        # initialization puts it).
        shear = jnp.maximum(0.0, _safe_norm(f_shear) - 2.0 * mu * fg) ** 2
        return squeeze + shear

    total = jnp.array(0.0, forces.dtype)
    for m, a, f in zip(system.manipulators, axes, forces):
        total = total + per_contact(m, a, f)
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

    Runs its own FK for the contact point; callers holding the world contact
    point already (via :func:`grasp_kinematics`) should use
    :func:`manipulator_contact_fext_at`.
    """
    return manipulator_contact_fext_at(
        m, q, _contact_point_world(m, q), f_world
    )


def manipulator_contact_fext_at(
    m: ManipulatorSpec,
    q: Float[Array, "n"],
    p_contact_world: Float[Array, "3"],
    f_world: Float[Array, "3"],
) -> Float[Array, "n 6"]:
    """:func:`manipulator_contact_fext` from a precomputed world contact point.

    ``p_contact_base = base^-1 · p_contact_world`` is exactly the quantity the
    FK inside :func:`manipulator_contact_fext` reconstructs, so this is the same
    wrench for one fewer forward-kinematics pass.
    """
    n = m.num_dof
    base = m.base_se3()

    # Reaction force on the manipulator, in base axes.
    f_base = base.rotation().inverse().apply(-f_world)

    # Contact point and last-body origin, both in base axes.
    _, r_world = m.grid.jacobian(q)  # r_world: (n_body, 3) in base frame
    r_last = r_world[..., -1, :]  # already base frame (grid uses manipulator base)
    p_contact_base = base.inverse().apply(p_contact_world.astype(f_base.dtype))

    tau_base = jnp.cross(p_contact_base - r_last, f_base)
    wrench = jnp.concatenate([tau_base, f_base])  # [torque; force]

    fext = jnp.zeros((n, 6), f_base.dtype)
    return fext.at[-1].set(wrench)


def capture_attachments(
    manipulators: tuple[ManipulatorSpec, ...],
    qs: tuple[Array, ...],
    T_world_object: jaxlie.SE3,
    geom: "CollGeom | None" = None,
    name: str = "object",
):
    """"Close the grippers *here*": one attachment per manipulator, capturing
    the object's pose in each grip-link frame.

    ``A_i = (T_base_i · T_link_i(q_i))^{-1} · T_W_obj``.  Note the base
    transform: :class:`ManipulatorSpec` places each manipulator in the world
    with ``base_xy_yaw``, while ``Robot.forward_kinematics`` (and hence
    :meth:`Attachment.grasp_from_current_pose`) works in the manipulator's own
    model frame — so the capture has to go through :func:`_gripper_world_pose`
    rather than raw FK, or every non-origin manipulator gets a silently wrong
    grasp transform.

    ``geom`` is the object's collision primitive (with its mass / inertia /
    friction); pass ``None`` for a kinematics-only capture.
    """
    from ..attachments import Attachment

    out = []
    for i, (m, q) in enumerate(zip(manipulators, qs)):
        T_W_L = _gripper_world_pose(m, q)
        T_LB = T_W_L.inverse() @ T_world_object
        if geom is None:
            out.append(
                Attachment(
                    parent_link_index=m.grip_link_index,
                    name=f"{name}@{i}",
                    ignored_link_indices=(),
                    num_prims=0,
                    T_parent_body=T_LB.wxyz_xyz,
                    geom=None,
                    spatial_inertia=None,
                    active=jnp.asarray(True),
                )
            )
        else:
            g = geom if geom.get_batch_axes() else geom.broadcast_to((1,))
            out.append(
                Attachment.from_geom(
                    g,
                    m.grip_link_index,
                    T_LB.wxyz_xyz,
                    mass=jnp.asarray(geom.mass),
                    inertia_com=jnp.diag(jnp.asarray(geom.inertia_diag)),
                    name=f"{name}@{i}",
                )
            )
    return tuple(out)


def capture_grasp_offsets(
    manipulators: tuple[ManipulatorSpec, ...], qs: tuple[Array, ...]
) -> tuple[jaxlie.SE3, ...]:
    """Constant reference->manipulator[i] relative gripper transforms at a
    grasp config. ``manipulators[0]`` / ``qs[0]`` is the reference.

    This is algebraically the same quantity :meth:`ContactSystem.from_attachments`
    derives: capture the object at the reference gripper's own pose and ``A_0``
    becomes identity, so ``A_0 · A_i^{-1}`` collapses to ``T_ref^{-1} · T_i``.
    It is deliberately *not* routed through :func:`capture_attachments` even so.
    Going via attachments would compose two extra float32 SE(3) products for an
    identical result, and the ~1e-7 they perturb is enough to move the third
    decimal of a solver calibrated against this path. The unification buys
    nothing here; it buys the dynamics and collision composition, which is what
    :func:`capture_attachments` is for.
    """
    poses = [_gripper_world_pose(m, q) for m, q in zip(manipulators, qs)]
    T_ref = poses[0]
    return tuple(T_ref.inverse() @ T_i for T_i in poses[1:])
