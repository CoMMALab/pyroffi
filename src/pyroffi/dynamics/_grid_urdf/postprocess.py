"""Fixed-joint elimination + DFS/BFS renumbering + subtree building.

Lifted verbatim off the ``URDFParser`` class's post-processing pipeline (the XML
parsing methods are omitted) and rephrased as free functions over a
:class:`.robot.Robot` populated elsewhere.  See package ``__init__`` for
provenance/licensing.
"""

import copy

import numpy as np


def _remove_fixed_joints(robot):
    for curr_joint in robot.get_joints_ordered_by_id():
        if curr_joint.jtype == "fixed":
            # updated fixed transforms and parents of grandchild_joints
            # to account for the additional fixed transform
            # X_grandchild = X_granchild * X_child
            for gcjoint in robot.get_joints_by_parent_name(curr_joint.child):
                gcjoint.set_parent(curr_joint.get_parent())
                gcjoint.set_transformation_matrix(gcjoint.get_transformation_matrix() * curr_joint.get_transformation_matrix())
            # combine inertia tensors of child and parent at parent
            # note:  if X is the transform from A to B the I_B = X^T I_A X
            # note2: inertias in the same from add so I_parent_final = I_parent + X^T I_child X
            child_link = robot.get_link_by_name(curr_joint.child)
            parent_link = robot.get_link_by_name(curr_joint.parent)
            child_I = child_link.get_spatial_inertia()
            curr_Xmat = np.reshape(np.array(curr_joint.get_transformation_matrix()).astype(float), (6, 6))
            transformed_Imat = np.matmul(np.matmul(np.transpose(curr_Xmat), child_I), curr_Xmat)
            parent_link.set_spatial_inertia(parent_link.get_spatial_inertia() + transformed_Imat)

            # delete the bypassed fixed joint and link
            robot.remove_joint(curr_joint)
            robot.remove_link(child_link)


def _build_subtree_lists(robot):
    subtree_lid_lists = {}
    # initialize all subtrees to include itself
    for lid in robot.get_links_dict_by_id().keys():
        subtree_lid_lists[lid] = [lid]
    # start at the leaves and build up!
    for curr_joint in robot.get_joints_ordered_by_id(reverse=True):
        parent_lid = robot.get_link_by_name(curr_joint.parent).get_id()
        child_lid = robot.get_link_by_name(curr_joint.child).get_id()
        # add the child's subtree list to the parent (includes the child)
        if child_lid in subtree_lid_lists.keys():
            subtree_lid_lists[parent_lid] = list(set(subtree_lid_lists[parent_lid]).union(set(subtree_lid_lists[child_lid])))
    # save to the links
    for link in robot.links:
        curr_subtree = subtree_lid_lists[link.get_id()]
        link.set_subtree(copy.deepcopy(curr_subtree))


def _dfs_order_update(robot, parent_name, alpha_tie_breaker=False, next_lid=0, next_jid=0):
    while True:
        child_joints = robot.get_joints_by_parent_name(parent_name)
        parent_id = robot.get_link_by_name(parent_name).lid
        if alpha_tie_breaker:
            child_joints.sort(key=lambda joint: joint.name)
        for curr_joint in child_joints:
            # save the new id
            curr_joint.set_id(next_jid)
            # save the next_lid to the child
            child = robot.get_link_by_name(curr_joint.child)
            child.set_id(next_lid)
            child.set_parent_id(parent_id)
            # recurse
            next_lid, next_jid = _dfs_order_update(robot, child.name, alpha_tie_breaker, next_lid + 1, next_jid + 1)
        # return to parent
        return next_lid, next_jid


def _bfs_order(robot, root_name):
    # initialize
    next_lid = 0
    next_jid = 0
    next_parent_names = [(root_name, -1)]
    robot.get_link_by_name(root_name).set_bfs_id(-1)
    robot.get_link_by_name(root_name).set_bfs_level(-1)
    # until there are no parent to parse
    while len(next_parent_names) != 0:
        # get the next parent and save its level
        (parent_name, parent_level) = next_parent_names.pop(0)
        next_level = parent_level + 1
        # then until there are no children to parse (of that parent)
        child_joints = robot.get_joints_by_parent_name(parent_name)
        while len(child_joints) != 0:
            # update the current link
            curr_joint = child_joints.pop(0)
            curr_joint.set_bfs_id(next_jid)
            curr_joint.set_bfs_level(next_level)
            # append the child to the list of future possible parents
            curr_child_name = curr_joint.get_child()
            next_parent_names.append((curr_child_name, next_level))
            # update the child
            curr_link = robot.get_link_by_name(curr_child_name)
            curr_link.set_bfs_id(next_lid)
            curr_link.set_bfs_level(next_level)
            # update the global lid, jid
            next_lid += 1
            next_jid += 1


def renumber_links_joints(robot, alpha_tie_breaker=False):
    """Remove fixed joints (merging links), then DFS/BFS-renumber and build subtree lists.

    Mutates ``robot`` in place (matching ``URDFParser.renumber_linksJoints``);
    returns it for convenience.
    """
    # remove all fixed joints where applicable (merge links)
    _remove_fixed_joints(robot)
    # find the root link
    link_names = set([link.name for link in robot.get_links_ordered_by_id()])
    links_that_are_children = set([joint.get_child() for joint in robot.get_joints_ordered_by_id()])
    root_link_name = list(link_names.difference(links_that_are_children))[0]
    # start renumbering at -1 as the base link is fixed by default
    robot.get_link_by_name(root_link_name).set_id(-1)
    # generate the standard dfs ordering of joints/links
    _dfs_order_update(robot, root_link_name, alpha_tie_breaker)
    # also save a bfs parse ordering and levels of joints/links
    _bfs_order(robot, root_link_name)
    # build subtree lists
    _build_subtree_lists(robot)
    return robot
