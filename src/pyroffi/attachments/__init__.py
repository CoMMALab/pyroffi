"""Tool use / attached bodies: objects a manipulator picks up and thereafter
carries as part of its own body.

Covers both halves of "carrying something": collision checking during transport,
and dynamics (RNEA/CRBA/ABA) when the object is load-bearing or is itself doing
the work — writing with a pen, pushing with a stick, carrying a full cup.

The whole design follows from one observation:

    **An attachment is a fixed joint.**

Attaching body ``B`` to link ``L`` with a constant transform ``T_LB`` adds a
fixed edge to the kinematic tree.  Kinematically that is ``T_WB(q) = T_WL(q) ·
T_LB`` — one SE(3) compose on top of the FK pyroffi already runs, with no new
joint variable, so ``MAX_JOINTS`` / ``MAX_ACT``, the FK and IK CUDA kernels and
the topological sort are all untouched.  Dynamically, Featherstone's congruence
``I_L' = I_L + Xᵀ I_B X`` absorbs a fixed child into its parent, which is
*exactly* the fixed-joint merge :meth:`RobotURDFParser.parse_dynamics` already
performs.  So dynamics support for a grasped object reduces to a rank-6 additive
update of a single row of ``DynamicsInfo.I_body``, and RNEA/CRBA/ABA run
unmodified.

The second design rule is what keeps it fast:

    **Static topology, dynamic transform.**

*Which* link an object hangs off, *how many* primitives it contributes and
*which* collision pairs are allowed are ``jdc.Static`` aux data; *where* it is
attached, its mass, its inertia and its dimensions are pytree leaves.  So a
``jit`` recompiles when the grasp *topology* changes (pick, place, handoff) — a
handful of times across a whole TAMP problem, not once per state — while
``T_parent_body`` and mass stay differentiable and ``vmap``-able.  That is what
makes batched grasp-candidate evaluation and ``∂cost/∂T_parent_body`` grasp
optimization possible, and it is the reason this exists rather than a binding to
VAMP's single baked-at-codegen-time attachment.

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
:meth:`RobotCollision.with_attachments`, which compose into the existing structs
rather than threading a new argument through every call site.

Out of scope
------------

Deliberately **not** modelled here, so nobody assumes otherwise:

* deformable or articulated attachments (an attachment is rigid, by definition);
* slip.  Rigid composition assumes the grasp holds.  That assumption is a
  *residual*, not a modelling change — ``dynamics._contact``'s
  ``grasp_closure_residual`` / ``grip_validity_penalty`` /
  ``parallel_jaw_grip_penalty`` certify it, and full force-level treatment is
  ``contact_rich_trajopt``'s job.  Attachment gives the *nominal* rigid model;
  those residuals say whether it is valid;
* objects whose inertia changes during transport (sloshing liquid).
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
