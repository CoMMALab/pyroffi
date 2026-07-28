"""The :class:`Attachment` primitive and its fixed-capacity container.

See the package ``__init__`` for the design rationale.  This module holds only
the data model and its constructors; everything that *consumes* an attachment
(collision geometry, dynamics, tool frames) lives in :mod:`._compose`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import numpy as onp
from jaxtyping import Array, Bool, Float

from ..collision._geometry import CollGeom

if TYPE_CHECKING:
    from .._robot import Robot


def motion_transform(T_ab: jaxlie.SE3) -> Float[Array, "*batch 6 6"]:
    """Featherstone spatial *motion* transform ``X_ba`` from ``T_ab``.

    ``T_ab`` is the pose of frame ``b`` relative to frame ``a``; the returned
    6x6 maps a motion vector expressed in ``a`` to its coordinates in ``b``.
    Angular-first, matching :func:`pyroffi._robot_urdf_parser
    ._motion_transform_from_T` — this is its differentiable twin, for traced
    grasp transforms rather than parse-time constants.
    """
    R = T_ab.rotation().as_matrix()  # (*batch, 3, 3)
    p = T_ab.translation()  # (*batch, 3)
    E = jnp.swapaxes(R, -1, -2)
    zeros = jnp.zeros_like(E)
    top = jnp.concatenate([E, zeros], axis=-1)
    bottom = jnp.concatenate([-E @ _skew(p), E], axis=-1)
    return jnp.concatenate([top, bottom], axis=-2)


def _skew(v: Float[Array, "*batch 3"]) -> Float[Array, "*batch 3 3"]:
    zero = jnp.zeros_like(v[..., 0])
    return jnp.stack(
        [
            jnp.stack([zero, -v[..., 2], v[..., 1]], axis=-1),
            jnp.stack([v[..., 2], zero, -v[..., 0]], axis=-1),
            jnp.stack([-v[..., 1], v[..., 0], zero], axis=-1),
        ],
        axis=-2,
    )


def spatial_inertia(
    mass: Float[Array, "*batch"],
    com: Float[Array, "*batch 3"],
    inertia_com: Float[Array, "*batch 3 3"],
) -> Float[Array, "*batch 6 6"]:
    """6x6 spatial inertia about the *body frame origin*.

    ``inertia_com`` is the 3x3 rotational inertia about the centre of mass, in
    body axes; ``com`` is the centre of mass in the body frame.  The
    parallel-axis term ``m (cᵀc I − c cᵀ)`` is applied here — forgetting it is
    the classic way to get plausible-but-wrong torques (cf. the same term in
    ``_spatial_inertia_from_urdf``).
    """
    m = mass[..., None, None]
    cx = _skew(com)
    I = jnp.zeros(jnp.broadcast_shapes(mass.shape, com.shape[:-1]) + (6, 6))
    eye = jnp.broadcast_to(jnp.eye(3), cx.shape)
    I = I.at[..., :3, :3].set(inertia_com + m * (cx @ jnp.swapaxes(cx, -1, -2)))
    I = I.at[..., :3, 3:].set(m * cx)
    I = I.at[..., 3:, :3].set(m * jnp.swapaxes(cx, -1, -2))
    I = I.at[..., 3:, 3:].set(m * eye)
    return I


@jdc.pytree_dataclass
class Attachment:
    """A rigid body carried by a robot link — the tool-use primitive.

    An attachment is a *fixed joint*, not a degree of freedom: it adds no
    configuration variable, so every kernel shape (``MAX_JOINTS``, the FK/IK
    launches, the topological sort) is untouched.  Its two effects are one SE(3)
    compose on the kinematics side and a rank-6 additive update of one
    ``DynamicsInfo.I_body`` row on the dynamics side.

    The static/dynamic split is the load-bearing design decision:

    * **static** — which link it hangs off, how many collision primitives it
      contributes, and which links it is allowed to touch.  Changing these is a
      grasp-*topology* change and recompiles.
    * **leaves** — where it is attached (``T_parent_body``), its mass/inertia,
      its primitive dimensions, and whether the slot is live.  These are
      differentiable and ``vmap``-able, so a batch of candidate grasp transforms
      goes through one compiled call and ``∂cost/∂T_parent_body`` falls out.
      (Batched grasp search means ``vmap`` — see :func:`.compose_dynamics`.)
    """

    parent_link_index: jdc.Static[int]
    """Index into ``Robot.links.names`` of the link this body is fixed to."""
    name: jdc.Static[str]
    """Identifier, unique within an :class:`AttachmentSet`."""
    ignored_link_indices: jdc.Static[tuple[int, ...]]
    """Links this body is *allowed* to touch (the grasping fingers). The
    allowed-collision set belongs to the attachment, not to a global setting:
    the object is supposed to be in contact with whatever is holding it."""
    num_prims: jdc.Static[int]
    """Number of collision primitives contributed (0 for a dynamics-only body)."""

    T_parent_body: Float[Array, "*batch 7"]
    """``wxyz_xyz`` pose of the body frame relative to the parent link frame
    (link <- body).  **Differentiable.**  Note the direction: inverting it
    yields plausible but wrong behaviour, which is why
    :meth:`grasp_from_current_pose` exists."""
    geom: CollGeom | None
    """Collision geometry in the *body* frame, batched over ``num_prims``.
    ``None`` for a dynamics-only attachment."""
    spatial_inertia: Float[Array, "*batch 6 6"] | None
    """6x6 spatial inertia about the body frame origin. ``None`` for a
    collision-only attachment."""
    active: Bool[Array, "*batch"]
    """Whether this slot is live.  Toggling it is jit-safe (no recompile): an
    inactive slot contributes zero spatial inertia and no contact — its geometry
    stays in the array (shapes cannot depend on a traced value) with its radius
    driven negative, so it can neither create nor hide one."""

    @staticmethod
    def from_geom(
        geom: CollGeom,
        parent_link: int,
        T_parent_body: Float[Array, "*batch 7"],
        *,
        mass: Float[Array, "*batch"] | None = None,
        com: Float[Array, "*batch 3"] | None = None,
        inertia_com: Float[Array, "*batch 3 3"] | None = None,
        name: str = "attachment",
        ignored_links: tuple[int, ...] = (),
    ) -> Attachment:
        """Build from collision geometry plus optional inertial parameters.

        ``geom`` must be batched over its primitive axis only (shape
        ``(num_prims,)``); a bare unbatched geometry is promoted to one
        primitive.  When ``mass`` is given the body also enters the dynamics
        chain, with inertia about ``com`` (default: the body frame origin) and
        ``inertia_com`` (default: a point mass).
        """
        batch_axes = geom.get_batch_axes()
        if batch_axes == ():
            geom = geom.broadcast_to((1,))
            batch_axes = (1,)
        if len(batch_axes) != 1:
            raise ValueError(
                "Attachment geometry must be batched over the primitive axis "
                f"only, got batch axes {batch_axes}. Reshape it to (num_prims,)."
            )
        T_parent_body = jnp.asarray(T_parent_body)
        I = None
        if mass is not None:
            m = jnp.asarray(mass)
            c = jnp.zeros(m.shape + (3,)) if com is None else jnp.asarray(com)
            Ic = (
                jnp.zeros(m.shape + (3, 3))
                if inertia_com is None
                else jnp.asarray(inertia_com)
            )
            I = spatial_inertia(m, c, Ic)
        return Attachment(
            parent_link_index=int(parent_link),
            name=name,
            ignored_link_indices=tuple(int(i) for i in ignored_links),
            num_prims=int(batch_axes[0]),
            T_parent_body=T_parent_body,
            geom=geom,
            spatial_inertia=I,
            active=jnp.asarray(True),
        )

    @staticmethod
    def from_mass(
        mass: Float[Array, "*batch"],
        parent_link: int,
        T_parent_body: Float[Array, "*batch 7"],
        *,
        com: Float[Array, "*batch 3"] | None = None,
        inertia_com: Float[Array, "*batch 3 3"] | None = None,
        name: str = "payload",
    ) -> Attachment:
        """A dynamics-only attachment: a payload with no collision geometry."""
        m = jnp.asarray(mass)
        c = jnp.zeros(m.shape + (3,)) if com is None else jnp.asarray(com)
        Ic = (
            jnp.zeros(m.shape + (3, 3))
            if inertia_com is None
            else jnp.asarray(inertia_com)
        )
        return Attachment(
            parent_link_index=int(parent_link),
            name=name,
            ignored_link_indices=(),
            num_prims=0,
            T_parent_body=jnp.asarray(T_parent_body),
            geom=None,
            spatial_inertia=spatial_inertia(m, c, Ic),
            active=jnp.asarray(True),
        )

    def with_pose(self, T_parent_body: Float[Array, "*batch 7"]) -> Attachment:
        """Same body, new grasp transform.  A regrasp *search* is a ``vmap``
        over this — the transform is a leaf, so it costs no recompilation."""
        with jdc.copy_and_mutate(self, validate=False) as out:
            out.T_parent_body = jnp.asarray(T_parent_body)
        return out

    def with_active(self, active) -> Attachment:
        """Enable/disable the slot.  jit-safe: ``active`` is a leaf."""
        with jdc.copy_and_mutate(self, validate=False) as out:
            out.active = jnp.asarray(active)
        return out

    def grasp_from_current_pose(
        self,
        robot: "Robot",
        cfg: Float[Array, "*batch actuated_count"],
        T_world_body: Float[Array, "*batch 7"],
    ) -> Attachment:
        """The "close the gripper *here*" constructor.

        Computes ``T_LB = T_WL(cfg)^-1 · T_WB`` from the object's current world
        pose, so callers never write the link<-body transform by hand.
        """
        T_world_link = jaxlie.SE3(
            robot.forward_kinematics(cfg)[..., self.parent_link_index, :]
        )
        T_LB = T_world_link.inverse() @ jaxlie.SE3(jnp.asarray(T_world_body))
        return self.with_pose(T_LB.wxyz_xyz)

    def T_world_body(
        self, robot: "Robot", cfg: Float[Array, "*batch actuated_count"]
    ) -> jaxlie.SE3:
        """World pose of the body frame: ``T_WL(cfg) · T_LB``."""
        T_world_link = jaxlie.SE3(
            robot.forward_kinematics(cfg)[..., self.parent_link_index, :]
        )
        return T_world_link @ jaxlie.SE3(self.T_parent_body)


@jdc.pytree_dataclass
class AttachmentSet:
    """Fixed-capacity collection of attachment slots.

    Attach/detach across a plan skeleton is expressed by *masking* slots rather
    than reallocating them: :meth:`attach` / :meth:`detach` change the static
    topology (and so recompile — a handful of times across a whole TAMP
    problem), while :meth:`set_active` is a leaf update and is free inside a
    jitted region.  A "held by both arms" handoff state is representable as two
    mutually-ignoring active slots.
    """

    attachments: tuple[Attachment, ...]

    @staticmethod
    def empty() -> AttachmentSet:
        return AttachmentSet(attachments=())

    def __len__(self) -> int:
        return len(self.attachments)

    def __iter__(self):
        return iter(self.attachments)

    def names(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.attachments)

    def index_of(self, name: str) -> int:
        for i, a in enumerate(self.attachments):
            if a.name == name:
                return i
        raise KeyError(f"No attachment named {name!r}; have {self.names()}.")

    def attach(self, attachment: Attachment) -> AttachmentSet:
        """Add a slot (host-side; changes topology, so it recompiles)."""
        if attachment.name in self.names():
            raise ValueError(f"Attachment {attachment.name!r} is already present.")
        return AttachmentSet(attachments=self.attachments + (attachment,))

    def detach(self, name: str) -> AttachmentSet:
        """Remove a slot (host-side; changes topology, so it recompiles)."""
        i = self.index_of(name)
        return AttachmentSet(
            attachments=self.attachments[:i] + self.attachments[i + 1 :]
        )

    def set_active(self, name: str, flag) -> AttachmentSet:
        """Enable/disable a slot.  jit-safe — no topology change, no recompile."""
        i = self.index_of(name)
        atts = list(self.attachments)
        atts[i] = atts[i].with_active(flag)
        return AttachmentSet(attachments=tuple(atts))

    def replace(self, name: str, attachment: Attachment) -> AttachmentSet:
        """Swap a slot's contents, keeping its position."""
        i = self.index_of(name)
        atts = list(self.attachments)
        atts[i] = attachment
        return AttachmentSet(attachments=tuple(atts))

    @property
    def total_prims(self) -> int:
        return sum(a.num_prims for a in self.attachments)


def link_dof_bodies(robot: "Robot") -> dict[str, tuple[int, onp.ndarray]]:
    """Map each link name to ``(dof_index, T_dofbody_link)``.

    The "body" an attachment loads is the *DOF body* its link belongs to, i.e.
    the nearest actuated ancestor — grasping with a finger whose joint is fixed
    correctly loads the wrist DOF.  ``T_dofbody_link`` is the constant transform
    from that body's frame (the actuated joint's child link frame) down to the
    attachment's parent link, composed through the intervening fixed joints.

    Links above every actuated joint (the fixed base) map to ``dof_index == -1``
    and are silently non-loading, which is the physically correct answer: a
    fixture bolted to the base torques nothing.
    """
    urdf = robot.urdf
    dof_index = {j.name: i for i, j in enumerate(urdf.actuated_joints)}
    joint_by_child = {j.child: j for j in urdf.joint_map.values()}

    def origin(j) -> onp.ndarray:
        return onp.asarray(j.origin) if j.origin is not None else onp.eye(4)

    out: dict[str, tuple[int, onp.ndarray]] = {}
    for link_name in urdf.link_map:
        T = onp.eye(4)
        cur = link_name
        idx = -1
        while cur in joint_by_child:
            j = joint_by_child[cur]
            if j.name in dof_index:
                idx = dof_index[j.name]
                break
            T = origin(j) @ T
            cur = j.parent
        out[link_name] = (idx, T)
    return out
