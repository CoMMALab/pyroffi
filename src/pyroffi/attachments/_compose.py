"""Composition entry points: fold an :class:`AttachmentSet` into the structures
the rest of pyroffi already consumes.

The deliberate choice here is to compose *into* ``Robot`` / ``RobotCollision``
rather than thread a new argument through every call site.  ``robot
.with_attachments(aset).inverse_dynamics(...)`` needs no signature change, so
``motion_generators``, ``optimization_engines`` and both trajopt solvers keep
working with zero edits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import jaxlie
from jaxtyping import Array, Float

from ..collision._geometry import CollGeom
from ._attachment import Attachment, AttachmentSet, link_dof_bodies, motion_transform

if TYPE_CHECKING:
    from .._robot import Robot
    from ..collision._robot_collision import RobotCollision


def _T_world_parents(
    robot: "Robot",
    cfg: Float[Array, "*batch actuated_count"],
    parent_link_indices: tuple[int, ...],
) -> jaxlie.SE3:
    """World poses of the given parent links, as one gather off a single FK."""
    Ts = robot.forward_kinematics(cfg)  # (*batch, num_links, 7)
    idx = jnp.asarray(parent_link_indices, dtype=jnp.int32)
    return jaxlie.SE3(Ts[..., idx, :])


def pose_attachments(
    robot: "Robot",
    cfg: Float[Array, "*batch actuated_count"],
    aset: AttachmentSet,
) -> CollGeom | None:
    """World-frame collision geometry of every attachment primitive.

    Result batch axes are ``(*batch, total_prims)``, concatenated in slot order.
    Costs one gather off the existing FK plus one SE(3) compose per primitive.
    Returns ``None`` when no slot carries geometry.
    """
    geoms = [a for a in aset if a.geom is not None and a.num_prims > 0]
    if not geoms:
        return None
    parents = tuple(a.parent_link_index for a in geoms)
    T_world_parent = _T_world_parents(robot, cfg, parents)  # (*batch, n_slots)

    batch = cfg.shape[:-1]
    posed = []
    for k, a in enumerate(geoms):
        T_world_body = jaxlie.SE3(
            T_world_parent.wxyz_xyz[..., k, :]
        ) @ jaxlie.SE3(a.T_parent_body)
        # The geometry is batched over its primitive axis only; lift it to the
        # cfg batch before transforming so every leaf agrees on the batch axes.
        geom = cast(CollGeom, a.geom).broadcast_to(batch + (a.num_prims,))
        T = jaxlie.SE3(T_world_body.wxyz_xyz[..., None, :])
        posed.append(geom.transform(T))
    if len(posed) == 1:
        return posed[0]
    if len({type(p) for p in posed}) != 1:
        raise TypeError(
            "Attachments carrying different CollGeom types cannot be concatenated "
            "into one collision array; give every attachment the same primitive "
            f"type (got {sorted(type(p).__name__ for p in posed)})."
        )
    return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=-1), *posed)


def tool_frame(
    robot: "Robot",
    cfg: Float[Array, "*batch actuated_count"],
    aset: AttachmentSet,
    name: str,
    T_body_tool: Float[Array, "*batch 7"] | None = None,
) -> jaxlie.SE3:
    """World pose of an attachment's frame, optionally offset to a tool tip.

    This is what lets IK servo the pen nib rather than the flange:
    ``T_W_tip = T_WL(cfg) · T_LB · T_B_tip``.
    """
    a = aset.attachments[aset.index_of(name)]
    T = a.T_world_body(robot, cfg)
    if T_body_tool is not None:
        T = T @ jaxlie.SE3(jnp.asarray(T_body_tool))
    return T


def ik_target_for_tool(
    aset: AttachmentSet,
    name: str,
    T_world_tool: Float[Array, "*batch 7"],
    T_body_tool: Float[Array, "*batch 7"] | None = None,
) -> tuple[int, Float[Array, "*batch 7"]]:
    """Rewrite a *tool-tip* IK goal as an equivalent *link* goal.

    Returns ``(parent_link_index, T_world_link)`` for the existing
    ``(target_link_indices, target_poses)`` IK interface, so an attachment can
    be servoed — put the pen nib here, not the flange — on both the CUDA and the
    pure-JAX solvers with **no kernel change**.

    That matters, because the CUDA kernels genuinely cannot do it any other way:
    ``_ik_cuda_helpers.cuh`` compares ``T_world[target_jnt]`` against
    ``target_T`` directly and carries no per-target constant offset. It does not
    need one. The tool offset is constant, so

        ``T_W_tip = T_W_L · A``   ⟺   ``T_W_L = T_W_tip · A^{-1}``

    and solving the second problem solves the first exactly. Folding the offset
    into the goal on the host is a handful of flops once per solve, versus a new
    buffer through every IK kernel.

    One caveat worth stating: the *residual* being minimized becomes the link's
    pose error rather than the tip's. The zero sets are identical, so an
    exact solve is unaffected; but for a weighted least-squares that does not
    reach zero, position and orientation error trade off about the link frame,
    not about the tip. If tip-frame error weighting matters, minimize
    :func:`tool_frame` directly on the JAX path.
    """
    a = aset.attachments[aset.index_of(name)]
    T_LB = jaxlie.SE3(a.T_parent_body)
    if T_body_tool is not None:
        T_LB = T_LB @ jaxlie.SE3(jnp.asarray(T_body_tool))
    T_W_L = jaxlie.SE3(jnp.asarray(T_world_tool)) @ T_LB.inverse()
    return a.parent_link_index, T_W_L.wxyz_xyz


def compose_dynamics(robot: "Robot", aset: AttachmentSet) -> "Robot":
    """Return ``robot`` with each attachment's inertia folded into its body.

    A fixed child is absorbed into its parent by Featherstone's congruence

    ``I_D' = I_D + X_{B<-D}^T · I_B · X_{B<-D}``

    which is exactly the fixed-joint merge ``parse_dynamics`` already performs.
    So ``num_dof``, ``parent_dof_indices``, ``S``, ``X_tree`` and
    ``joint_is_prismatic`` are all untouched — only one row of ``I_body``
    changes, and RNEA/CRBA/ABA run unmodified on the result.  Gravity and the
    payload's Coriolis/centrifugal contribution then fall out automatically.

    ``B`` is referred to the **DOF body** the parent link belongs to (its
    nearest actuated ancestor), so grasping with a fixed-joint finger correctly
    loads the wrist DOF.  Attachments on the fixed base load nothing.

    Differentiable in both mass and ``T_parent_body``: ``∂τ/∂mass`` makes
    payload identification from measured torques a one-liner, and
    ``∂τ/∂T_parent_body`` makes "grasp where the transport torque stays inside
    the limits" an optimization rather than a search.
    """
    import jax_dataclasses as jdc

    if robot.dynamics is None:
        raise ValueError(
            "compose_dynamics requires a Robot with dynamics information; this "
            "URDF has none (see RobotURDFParser.parse_dynamics)."
        )
    bodies = [a for a in aset if a.spatial_inertia is not None]
    if not bodies:
        return robot

    link_map = link_dof_bodies(robot)
    link_names = robot.links.names
    I_body = robot.dynamics.I_body

    for a in bodies:
        link_name = link_names[a.parent_link_index]
        dof_index, T_body_link = link_map[link_name]
        if dof_index < 0:
            # Attached above every actuated joint: rigidly grounded, so it adds
            # no load to any DOF. Silently correct, not an error.
            continue
        # T_D_B = T_D_L (constant, through the fixed joints) · T_L_B (a leaf).
        T_DL = jaxlie.SE3.from_matrix(jnp.asarray(T_body_link, dtype=I_body.dtype))
        T_DB = T_DL @ jaxlie.SE3(a.T_parent_body)
        X = motion_transform(T_DB)  # X_{B<-D}
        I_att = cast(Array, a.spatial_inertia)
        # An inactive slot contributes exactly zero, so composing then
        # deactivating recovers the unattached DynamicsInfo bitwise.
        gate = jnp.asarray(a.active, dtype=I_body.dtype)
        dI = gate[..., None, None] * (
            jnp.swapaxes(X, -1, -2) @ I_att @ X
        )
        I_body = I_body.at[dof_index].add(dI)

    with jdc.copy_and_mutate(robot.dynamics, validate=False) as dyn:
        dyn.I_body = I_body
    with jdc.copy_and_mutate(robot, validate=False) as out:
        out.dynamics = dyn
    return out


def attachment_wrench_to_body(
    robot: "Robot",
    aset: AttachmentSet,
    name: str,
    wrench_body: Float[Array, "*batch 6"],
    T_body_frame: Float[Array, "*batch 7"] | None = None,
) -> tuple[int, Float[Array, "*batch 6"]]:
    """Map a wrench applied at an attachment (or tool-tip) frame to its DOF body.

    Spatial forces are the dual of spatial motions, so where motion transforms
    by ``m_B = X_{B<-D} m_D``, force transforms *back* by the transpose:

        ``f_D = X_{B<-D}ᵀ · f_B``

    (power ``fᵀm`` is frame-invariant, which forces exactly this pairing).  For
    a pure translation ``p`` that reduces to ``moment += p × force``, the
    familiar lever arm.  This is the entry point for pen-on-paper,
    push-with-stick and peg-in-hole reaction forces; the representation matches
    what ``dynamics._contact.manipulator_contact_fext`` already uses.

    Returns ``(dof_index, wrench_in_body_frame)``; ``dof_index == -1`` means the
    attachment is grounded and the wrench loads no DOF.
    """
    a = aset.attachments[aset.index_of(name)]
    dof_index, T_body_link = link_dof_bodies(robot)[
        robot.links.names[a.parent_link_index]
    ]
    if dof_index < 0:
        return -1, jnp.zeros_like(jnp.asarray(wrench_body))
    T_DB = jaxlie.SE3.from_matrix(jnp.asarray(T_body_link)) @ jaxlie.SE3(
        a.T_parent_body
    )
    if T_body_frame is not None:
        T_DB = T_DB @ jaxlie.SE3(jnp.asarray(T_body_frame))
    X = motion_transform(T_DB)  # X_{B<-D}
    return dof_index, jnp.einsum(
        "...ji,...j->...i", X, jnp.asarray(wrench_body)
    )


def _pad_to_slots(geom: CollGeom, num_slots: int) -> CollGeom:
    """Pad a ``(num_prims,)`` geometry out to ``(num_slots,)``.

    Reuses the spherized model's own convention: padding entries carry a
    negative-radius sentinel, which its distance reduction already masks out of
    the min (so a pad neither creates nor hides a contact).
    """
    from ..collision._geometry import Sphere

    n = geom.get_batch_axes()[0]
    if n > num_slots:
        raise ValueError(
            f"Attachment contributes {n} spheres but this spherized collision "
            f"model only carries {num_slots} per row. Simplify the attachment "
            "geometry, or rebuild the robot model with more spheres per link."
        )
    if n == num_slots:
        return geom
    if not isinstance(geom, Sphere):
        raise TypeError(
            "Padding an attachment out to the spherized model's per-row sphere "
            f"count is only defined for Sphere geometry, got {type(geom).__name__}."
        )
    pad = Sphere.from_center_and_radius(
        center=jnp.zeros((num_slots - n, 3)),
        radius=jnp.full((num_slots - n,), -1e9),
    )
    return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), geom, pad)


def compose_collision(rcoll, aset: AttachmentSet):
    """Return ``rcoll`` extended with every attachment's collision primitives.

    Accepts either :class:`~pyroffi.collision.RobotCollision` (capsules, one
    entry per primitive) or ``RobotCollisionSpherized`` (spheres, one row of S
    per link); the two layouts are distinguished by the geometry's rank.

    Geometry is concatenated onto the per-link array along the primitive axis,
    giving ``K' = num_links + Σ num_prims``; poses come from one gather of the
    parent link pose plus one SE(3) compose, so per-state cost is a handful of
    flops.  The world-collision kernels are shape-generic in ``K`` and
    self-collision takes the pair table as a runtime buffer, so a longer array
    is just a larger launch — no kernel or FFI change.

    The pair table gains, for each attachment primitive: pairs against every
    robot link except its own parent and its ``ignored_link_indices``; plus
    attachment-vs-attachment pairs across *different* slots, which is what makes
    two-arm handoff and tool-vs-workpiece checkable.

    The allowed-collision set is the attachment's own, deliberately: a grasped
    object is *supposed* to touch the fingers, and only the caller knows which
    links those are.  Inheriting the parent link's existing allowed set was
    tried and rejected — gripper links frequently carry no collision geometry
    and so appear in no pair at all, which would silently give the attachment an
    empty pair set (i.e. no collision checking) rather than a permissive one.
    """
    import jax_dataclasses as jdc

    slots = [a for a in aset if a.geom is not None and a.num_prims > 0]
    if not slots:
        return rcoll
    if rcoll.attach_parent_link_indices:
        raise ValueError(
            "This RobotCollision already carries attachments; compose from the "
            "un-attached model rather than stacking compositions."
        )

    n_link = rcoll.num_links
    geoms = [cast(CollGeom, a.geom) for a in slots]
    if {type(g) for g in geoms} != {type(rcoll.coll)}:
        raise TypeError(
            "Attachment geometry must be the same CollGeom type as the robot's "
            f"({type(rcoll.coll).__name__}); got "
            f"{sorted({type(g).__name__ for g in geoms})}."
        )

    # Two geometry layouts to serve. The capsule model is one entry per
    # primitive, batched (N,), so each attachment primitive becomes its own
    # entry. The spherized model is (N, S) -- one row of up to S spheres per
    # link -- so each attachment becomes one *row*, padded out to S. Either way
    # the extra entries land in the tail of the array and everything downstream
    # (pair table, world launch, cost reductions) just sees a larger N.
    per_row = len(rcoll.coll.get_batch_axes()) > 1
    spheres_per_row = rcoll.coll.get_batch_axes()[1] if per_row else 1
    if per_row:
        geoms = [_pad_to_slots(g, spheres_per_row) for g in geoms]

    coll = jax.tree.map(
        lambda *xs: jnp.concatenate(xs, axis=0),
        rcoll.coll,
        *[g[None] if per_row else g for g in geoms],
    )

    # Per-entry bookkeeping, flattened in slot order.
    parent_of_prim: list[int] = []
    slot_of_prim: list[int] = []
    T_list = []
    active_list = []
    names: list[str] = []
    for s, a in enumerate(slots):
        n_entries = 1 if per_row else a.num_prims
        parent_of_prim += [a.parent_link_index] * n_entries
        slot_of_prim += [s] * n_entries
        T_list.append(jnp.broadcast_to(a.T_parent_body, (n_entries, 7)))
        active_list.append(jnp.broadcast_to(a.active, (n_entries,)))
        names += [
            a.name if n_entries == 1 else f"{a.name}/{k}" for k in range(n_entries)
        ]

    new_i: list[int] = []
    new_j: list[int] = []
    for p, (parent, slot) in enumerate(zip(parent_of_prim, slot_of_prim)):
        prim_idx = n_link + p
        allowed = (
            set(range(n_link))
            - {parent}
            - set(slots[slot].ignored_link_indices)
        )
        for link in sorted(allowed):
            new_i.append(link)
            new_j.append(prim_idx)
        # attachment-vs-attachment, different slots only (primitives of one
        # body are rigidly co-moving and can never separate).
        for q in range(p + 1, len(parent_of_prim)):
            if slot_of_prim[q] != slot:
                new_i.append(prim_idx)
                new_j.append(n_link + q)

    idx_i = jnp.concatenate([rcoll.active_idx_i, jnp.asarray(new_i, dtype=jnp.int32)])
    idx_j = jnp.concatenate([rcoll.active_idx_j, jnp.asarray(new_j, dtype=jnp.int32)])

    with jdc.copy_and_mutate(rcoll, validate=False) as out:
        out.num_links = n_link + len(parent_of_prim)
        out.link_names = rcoll.link_names + tuple(names)
        out.coll = coll
        out.active_idx_i = idx_i
        out.active_idx_j = idx_j
        out.attach_parent_link_indices = tuple(parent_of_prim)
        out.attach_T_parent_body = jnp.concatenate(T_list, axis=0)
        out.attach_active = jnp.concatenate(active_list, axis=0)
    return out
