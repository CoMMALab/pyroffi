"""Tool use / attached bodies: objects a manipulator picks up and thereafter
carries as part of its own body.

Covers both halves of "carrying something": collision checking during transport,
and dynamics (RNEA/CRBA/ABA) when the object is load-bearing or is itself doing
the work — writing with a pen, pushing with a stick, carrying a full cup.

Two rules generate the whole design; ``docs/tool_use.md`` argues them at length.

**An attachment is a fixed joint.**  Kinematics is one SE(3) compose,
``T_WB(q) = T_WL(q) · T_LB``, with no new joint variable — so ``MAX_JOINTS`` /
``MAX_ACT``, the FK/IK CUDA kernels and the topological sort are untouched.
Dynamics is Featherstone's congruence ``I_L' = I_L + Xᵀ I_B X``, the same
fixed-joint merge :meth:`RobotURDFParser.parse_dynamics` already performs — a
rank-6 update of one ``DynamicsInfo.I_body`` row, after which RNEA/CRBA/ABA run
unmodified.

**Static topology, dynamic transform.**  Which link an object hangs off, how
many primitives it contributes and which pairs are allowed are ``jdc.Static``;
where it is attached, its mass and its dimensions are pytree leaves.  So a jit
recompiles per grasp-*topology* change (pick, place, handoff) rather than per
state, while ``T_parent_body`` stays differentiable and ``vmap``-able — which is
what makes batched grasp-candidate evaluation and ``∂cost/∂T_parent_body``
possible, and why this exists rather than a binding to VAMP's single
baked-at-codegen-time attachment.

Entry points
------------

``Attachment`` / ``AttachmentSet``    the data model (:mod:`._attachment`)
``pose_attachments``                  world-frame geometry of every primitive
``tool_frame``                        tool-tip pose, for IK targets and costs
``ik_target_for_tool``                rewrite a tip goal as a link goal for IK
``compose_dynamics``                  ``Robot`` with the load folded in
``compose_collision``                 ``RobotCollision`` with geometry + pairs
``attachment_wrench_to_body``         tool-tip wrench -> body-frame wrench

with sugar on the objects themselves: :meth:`Robot.with_attachments` and
:meth:`RobotCollision.with_attachments`.  Both compose from an *un-attached*
model and refuse to stack, since the composition is additive.

Out of scope
------------

Deliberately **not** modelled, so nobody assumes otherwise: deformable or
articulated attachments; objects whose inertia changes in transit (sloshing);
and slip.  Rigid composition assumes the grasp holds — that assumption is a
*residual*, not a modelling change, certified by ``dynamics._contact``'s
``grasp_closure_residual`` / ``grip_validity_penalty`` /
``parallel_jaw_grip_penalty``, with force-level treatment being
``contact_rich_trajopt``'s job.

Attachment collision works on the pure-JAX checkers and on both CUDA checkers,
with no kernel change: the extra geometry rows are posed off the parent link's
transform, which each backend already has (or, for the fused binary kernel,
computes).  When a CUDA checker is given a coarse guard model, the same
``AttachmentSet`` must be composed into both models — the guard flags fine
geometry by row, so an unmatched attachment row would go unchecked.
"""

from ._attachment import (
    Attachment as Attachment,
    AttachmentSet as AttachmentSet,
    link_dof_bodies as link_dof_bodies,
    motion_transform as motion_transform,
    spatial_inertia as spatial_inertia,
)
from ._compose import (
    attachment_wrench_to_body as attachment_wrench_to_body,
    compose_collision as compose_collision,
    compose_dynamics as compose_dynamics,
    ik_target_for_tool as ik_target_for_tool,
    pose_attachments as pose_attachments,
    tool_frame as tool_frame,
)

__all__ = [
    "Attachment",
    "AttachmentSet",
    "attachment_wrench_to_body",
    "compose_collision",
    "compose_dynamics",
    "ik_target_for_tool",
    "link_dof_bodies",
    "motion_transform",
    "pose_attachments",
    "spatial_inertia",
    "tool_frame",
]
