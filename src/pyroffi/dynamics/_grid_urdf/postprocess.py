"""Fixed-joint elimination + DFS/BFS renumbering + subtree building.

Lifted verbatim off the ``URDFParser`` class's post-processing pipeline (the XML
parsing methods are omitted) and wrapped in a parser-free ``_PostProcessor``
shim over a :class:`.robot.Robot` populated elsewhere.  Keeping the methods
byte-identical to upstream is deliberate: this pipeline decides joint/link ids,
and any drift from it silently changes every generated GRiD kernel's indexing.
See package ``__init__`` for provenance/licensing.
"""

import copy
import warnings

import numpy as np
import sympy as sp

from .errors import URDFParseError
from .joint import Joint, Fixed_Joint
from .link import Link


class _PostProcessor:
    """The non-XML half of ``URDFParser``, operating on an existing Robot."""

    def __init__(self, robot, strict_inertial=False):
        self.robot = robot
        self.strict_inertial = strict_inertial

    def remove_fixed_joints(self):
        # start at the leaves and work upwards
        for curr_joint in reversed(self.robot.get_joints_ordered_by_id()):
            if curr_joint.jtype == "fixed":
                # updated fixed transforms and parents of grandchild_joints
                # to account for the additional fixed transform
                # X_grandchild = X_granchild * X_child
                for gcjoint in self.robot.get_joints_by_parent_name(curr_joint.child):
                    gcjoint.set_parent(curr_joint.get_parent())
                    gcjoint.set_transformation_matrix(gcjoint.get_transformation_matrix() * curr_joint.get_transformation_matrix())
                    gcjoint.set_transformation_matrix_hom(
                        curr_joint.get_transformation_matrix_hom() * gcjoint.get_transformation_matrix_hom()
                    )
                # combine inertia tensors of child and parent at parent
                # note:  if X is the transform from A to B the I_B = X^T I_A X
                # note2: inertias in the same from add so I_parent_final = I_parent + X^T I_child X
                child_link = self.robot.get_link_by_name(curr_joint.child)
                parent_link = self.robot.get_link_by_name(curr_joint.parent)
                child_I = child_link.get_spatial_inertia()
                curr_Xmat = np.reshape(np.array(curr_joint.get_transformation_matrix()).astype(float),(6,6))
                transformed_Imat = np.matmul(np.matmul(np.transpose(curr_Xmat),child_I),curr_Xmat)
                parent_link.set_spatial_inertia(parent_link.get_spatial_inertia() + transformed_Imat)
                
                # save the fixed joint for later
                joint_hom = sp.matrix2numpy(curr_joint.get_transformation_matrix_hom()).astype(float)
                parent_joints = self.robot.get_joints_by_child_name(parent_link.get_name())
                parent_joint = parent_joints[0] if parent_joints else None
                parent_joint_name = parent_joint.get_name() if parent_joint is not None else -1
                fj = Fixed_Joint(curr_joint.get_id(), curr_joint.get_name(), parent_joint_name, joint_hom)
                self.robot.add_fixed_joint(fj)
                # update any fixed joints that had the current joint as the parent
                for fixed_joint in self.robot.fixed_joints:
                    if fixed_joint.parent_name == curr_joint.get_name():
                        fixed_joint.set_parent(parent_joint_name)
                        new_hom = joint_hom @ fixed_joint.get_transformation_matrix_hom()
                        fixed_joint.set_transformation_matrix_hom(new_hom)

                # delete the bypassed fixed joint and link
                self.robot.remove_joint(curr_joint)
                self.robot.remove_link(child_link)
        
        # renumber fixed joints (arbitarily) starting at the highest joint id to avoid conflicts with existing joint ids
        total_joints = self.robot.get_num_joints()
        for fj_id in range(len(self.robot.fixed_joints)):
            self.robot.fixed_joints[fj_id].set_id(total_joints + fj_id)

    def build_subtree_lists(self):
        subtree_lid_lists = {}
        # initialize all subtrees to include itself
        for lid in self.robot.get_links_dict_by_id().keys():
            subtree_lid_lists[lid] = [lid]
        # start at the leaves and build up!
        for curr_joint in self.robot.get_joints_ordered_by_id(reverse=True):
            parent_lid = self.robot.get_link_by_name(curr_joint.parent).get_id()
            child_lid = self.robot.get_link_by_name(curr_joint.child).get_id()
            # add the child's subtree list to the parent (includes the child)
            if child_lid in subtree_lid_lists.keys():
                subtree_lid_lists[parent_lid] = list(set(subtree_lid_lists[parent_lid]).union(set(subtree_lid_lists[child_lid])))
        # save to the links
        for link in self.robot.links:
            curr_subtree = subtree_lid_lists[link.get_id()]
            link.set_subtree(copy.deepcopy(curr_subtree))

    def sort_child_joints(self, child_joints, joint_ordering):
        if joint_ordering == "urdf_order":
            return child_joints
        if joint_ordering == "alphabetical_order":
            return sorted(child_joints, key=lambda joint: joint.name)
        if joint_ordering == "pinocchio_order":
            return sorted(child_joints, key=lambda joint: (joint.child, joint.name))
        raise ValueError(
            "joint_ordering must be one of 'urdf_order', 'alphabetical_order', or 'pinocchio_order'"
        )

    def dfs_order_update(self, parent_name, joint_ordering = "pinocchio_order", next_lid = 0, next_jid = 0):
        while True:
            child_joints = self.robot.get_joints_by_parent_name(parent_name)
            parent_id = self.robot.get_link_by_name(parent_name).lid
            child_joints = self.sort_child_joints(child_joints, joint_ordering)
            for curr_joint in child_joints:
                # save the new id
                curr_joint.set_id(next_jid)
                # save the next_lid to the child
                child = self.robot.get_link_by_name(curr_joint.child)
                child.set_id(next_lid)
                child.set_parent_id(parent_id)
                # recurse
                next_lid, next_jid = self.dfs_order_update(child.name, joint_ordering, next_lid + 1, next_jid + 1)
            # return to parent
            return next_lid, next_jid

    def bfs_order(self, root_name):
        # initialize
        next_lid = 0
        next_jid = 0
        next_parent_names = [(root_name,-1)]
        self.robot.get_link_by_name(root_name).set_bfs_id(-1)
        self.robot.get_link_by_name(root_name).set_bfs_level(-1)
        # until there are no parent to parse
        while len(next_parent_names) != 0:
            # get the next parent and save its level
            (parent_name, parent_level) = next_parent_names.pop(0)
            next_level = parent_level + 1
            # then until there are no children to parse (of that parent)
            child_joints = self.robot.get_joints_by_parent_name(parent_name)
            while len(child_joints) != 0:
                # update the current link
                curr_joint = child_joints.pop(0)
                curr_joint.set_bfs_id(next_jid)
                curr_joint.set_bfs_level(next_level)
                # append the child to the list of future possible parents
                curr_child_name = curr_joint.get_child()
                next_parent_names.append((curr_child_name,next_level))
                # update the child
                curr_link = self.robot.get_link_by_name(curr_child_name)
                curr_link.set_bfs_id(next_lid)
                curr_link.set_bfs_level(next_level)
                # update the global lid, jid
                next_lid += 1
                next_jid += 1

    def floating_base_adjust(self, root_link_name, using_quaternion = True):
        if not self.robot.floating_base:
            return root_link_name
        if root_link_name == "world":
            root_children = self.robot.get_joints_by_parent_name("world")
            if len(root_children) != 1:
                raise ValueError(
                    "Floating-base conversion for an explicit URDF world root currently expects "
                    f"exactly one child joint from 'world', found {len(root_children)}."
                )
            floating_joint = root_children[0]
            if floating_joint.get_child() == "world":
                raise ValueError(
                    "Floating-base conversion encountered an invalid self-loop from 'world' to 'world'."
                )
            floating_joint.name = "floating_base_joint"
            floating_joint.using_quaternion = using_quaternion
            floating_joint.set_type("floating")
            floating_joint.set_damping(0)
            return "world"
        # add world link
        world = Link("world",-2) # -2 is temporary and unique
        world.set_origin_xyz([0, 0, 0])
        world.set_origin_rpy([0, 0, 0])
        world.set_inertia(0, 0, 0, 0, 0, 0, 0)
        self.robot.add_link(copy.deepcopy(world))
        # add floating joint
        floating_joint = Joint("floating_base_joint", -2, "world", root_link_name, using_quaternion)
        floating_joint.set_origin_xyz([0,0,0])
        floating_joint.set_origin_rpy([0,0,0])
        floating_joint.set_type("floating")
        floating_joint.set_damping(0)
        self.robot.add_joint(copy.deepcopy(floating_joint))
        return "world" # world link is now the root

    def renumber_linksJoints(self, using_quaternion = True, joint_ordering = "pinocchio_order"):
        # find the root link
        link_names = set([link.name for link in self.robot.get_links_ordered_by_id()])
        links_that_are_children = set([joint.get_child() for joint in self.robot.get_joints_ordered_by_id()])
        root_link_name = list(link_names.difference(links_that_are_children))[0]
        # adjust for floating base if applicable
        root_link_name = self.floating_base_adjust(root_link_name, using_quaternion)
        # start renumbering at -1
        self.robot.get_link_by_name(root_link_name).set_id(-1)
        # generate the standard dfs ordering of joints/links
        self.dfs_order_update(root_link_name, joint_ordering)
        # remove all fixed joints where applicable (merge links)
        self.remove_fixed_joints()
        # recompute the dfs ordering of joints/links to account for removed fixed joints
        self.dfs_order_update(root_link_name, joint_ordering)
        # also save a bfs parse ordering and levels of joints/links and build subtree lists
        self.bfs_order(root_link_name)
        self.build_subtree_lists()
        # resolve <mimic> targets now that final jids are stable
        self.resolve_mimic_targets()
        self.robot.refresh_joint_metadata()
        # the renumbered root link is the (intentionally massless) base frame;
        # flag it so strict inertial validation never rejects it.
        root_link = self.robot.get_link_by_name(root_link_name)
        if root_link is not None:
            root_link.set_dummy(True)
        self.validate_inertials(root_link_name)

    def validate_inertials(self, root_link_name):
        """Guard against a degenerate/missing <inertial> on a real moving body
        (zero or non-positive-definite mass/inertia → singular/broken dynamics).
        The root/base frame and dummy links are exempt (intentionally massless).
        strict_inertial=True RAISES; lenient mode (the default) now WARNS instead
        of silently zeroing (the silent path was a footgun — e.g. rizon4's
        zero-inertia links produced broken dynamics with no signal)."""
        strict = getattr(self, "strict_inertial", False)
        bad = [link.get_name() for link in self.robot.get_links_ordered_by_id()
               if link.get_name() != root_link_name and not link.is_dummy_link()
               and (getattr(link, "missing_inertial", False) or link.has_degenerate_inertial())]
        if not bad:
            return
        msg = (f"Link(s) {bad} have a degenerate/missing <inertial> (zero or "
               "non-positive-definite mass/inertia), which yields singular/broken "
               "dynamics. Provide a valid <inertial>, or parse with "
               "strict_inertial=True to reject this as an error.")
        if strict:
            raise URDFParseError(msg)
        warnings.warn("URDFParser: " + msg, stacklevel=2)

    def resolve_mimic_targets(self):
        """Resolve each mimic joint's `mimic_joint_name` to its current jid.

        Must run after the final renumbering pass so `mimic_target_id` is
        stable. Fails loudly (`ValueError`) if a mimic joint references a
        joint that isn't part of the parsed model — silently dropping such
        a relation produces wrong dynamics derivatives downstream (the bug
        this support closes).
        """
        for joint in self.robot.get_joints_ordered_by_id():
            if not getattr(joint, "is_mimic", False):
                continue
            target_name = joint.get_mimic_joint_name()
            target = self.robot.get_joint_by_name(target_name)
            if target is None:
                # Allow mimic of a fixed joint: that's effectively a constant
                # coordinate, which means this mimic joint also degenerates
                # to a constant offset relative to its parent. Resolve by
                # leaving mimic_target_id as -1 and dof=0 (already the case).
                if self.robot.get_fixed_joint_by_name(target_name) is not None:
                    joint.mimic_target_id = -1
                    continue
                raise ValueError(
                    f"Joint '{joint.get_name()}' mimics unknown joint "
                    f"'{target_name}'. Available joints: "
                    f"{[j.get_name() for j in self.robot.get_joints_ordered_by_id()]}"
                )
            if getattr(target, "is_mimic", False):
                raise ValueError(
                    f"Joint '{joint.get_name()}' mimics '{target_name}', which is "
                    "itself a mimic joint. Chained mimics are not supported."
                )
            joint.mimic_target_id = target.get_id()


def renumber_links_joints(
    robot,
    alpha_tie_breaker=None,
    joint_ordering="pinocchio_order",
    using_quaternion=True,
    strict_inertial=False,
):
    """Remove fixed joints (merging links), then DFS/BFS-renumber, build subtree
    lists and refresh joint metadata.

    Mutates ``robot`` in place (matching ``URDFParser.renumber_linksJoints``);
    returns it for convenience.  ``alpha_tie_breaker`` is the legacy switch and
    wins over ``joint_ordering`` when not None (upstream ``resolve_joint_ordering``).
    """
    pp = _PostProcessor(robot, strict_inertial=strict_inertial)
    if alpha_tie_breaker is not None:
        joint_ordering = "alphabetical_order" if alpha_tie_breaker else "urdf_order"
    elif joint_ordering not in ("urdf_order", "alphabetical_order", "pinocchio_order"):
        raise ValueError(
            "joint_ordering must be one of 'urdf_order', 'alphabetical_order', "
            "or 'pinocchio_order'"
        )
    pp.renumber_linksJoints(using_quaternion, joint_ordering)
    return robot
