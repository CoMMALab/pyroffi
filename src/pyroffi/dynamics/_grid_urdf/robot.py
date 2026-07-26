from .link import Link
from .joint import Joint, Fixed_Joint
from .spatial_algebra import Quaternion_Tools

import numpy as np

class Robot:
    # initialization
    def __init__(self, name, floating_base = False, using_quaternion = True, floating_base_convention = "pinocchio"):
        self.name = name
        self.floating_base = floating_base
        self.links = []
        self.joints = []
        self.fixed_joints = []
        self.using_quaternion = using_quaternion
        if floating_base_convention not in ("pinocchio", "legacy"):
            raise ValueError("floating_base_convention must be 'pinocchio' or 'legacy'")
        self.floating_base_convention = floating_base_convention
        self.joint_type_by_id = {}
        self.joint_type_by_name = {}

    def next_none(self, iterable):
        try:
            return next(iterable)
        except:
            return None
        
    def _dense_q_offset(self, joint_id):
        # The dense map is populated by `refresh_joint_metadata`. If we're
        # asked before any joints have been registered (or for an unknown
        # jid), fall back to the joint_id itself to preserve legacy behavior
        # on models without mimic joints.
        return getattr(self, "_dense_q_offset_by_id", {}).get(joint_id, joint_id)

    def _dense_v_offset(self, joint_id):
        return getattr(self, "_dense_v_offset_by_id", {}).get(joint_id, joint_id)

    def get_joint_index_q(self, joint_id):
        if self.floating_base and joint_id == 0:
            return [0,1,2,3,4,5,6] if self.using_quaternion else [0,1,2,3,4,5]
        offset = self._dense_q_offset(joint_id)
        # Multi-DOF non-root joints (planar NQ=3, spherical NQ=4) own a
        # CONTIGUOUS BLOCK of q slots, not a single scalar. Return the range so
        # `q[get_joint_index_q(jid)]` feeds the whole local-q vector to the
        # joint's transform function (mirrors the floating-base root block).
        joint = self.get_joint_by_id(joint_id)
        local_q_dim = getattr(joint, "local_q_dim", 0) if joint is not None else 0
        if local_q_dim and local_q_dim > 1:
            return list(range(offset, offset + local_q_dim))
        return offset

    def get_joint_index_v(self, joint_id):
        if self.floating_base and joint_id == 0:
            return [0,1,2,3,4,5]
        offset = self._dense_v_offset(joint_id)
        # Multi-DOF non-root joints own a contiguous block of v slots (= dof).
        joint = self.get_joint_by_id(joint_id)
        dof = joint.get_num_dof() if joint is not None else 0
        if dof and dof > 1:
            return list(range(offset, offset + dof))
        return offset

    def get_joint_index_f(self, joint_id):
        # Generalized-force slots map identically to the velocity slots: the
        # RNEA backward pass projects c[inds_f] = S^T f, an NV-vector. So this
        # returns the SAME scalar (1-DOF) or contiguous block (multi-DOF:
        # floating root 6-wide, spherical/planar 3-wide) as get_joint_index_v.
        return self.get_joint_index_v(joint_id)

    def uses_legacy_floating_base_convention(self):
        return self.floating_base and self.floating_base_convention == "legacy"

    def get_floating_base_convention(self):
        return self.floating_base_convention

    def get_floating_base_q_input_permutation_to_internal(self):
        if not self.uses_legacy_floating_base_convention():
            return None
        if self.using_quaternion:
            return [0, 1, 2, 4, 5, 6, 3]
        return None

    def get_floating_base_q_output_permutation_from_internal(self):
        if not self.uses_legacy_floating_base_convention():
            return None
        if self.using_quaternion:
            return [0, 1, 2, 6, 3, 4, 5]
        return None

    def get_floating_base_v_permutation_to_internal(self):
        if not self.uses_legacy_floating_base_convention():
            return None
        return [3, 4, 5, 0, 1, 2]

    def get_floating_base_v_permutation_from_internal(self):
        if not self.uses_legacy_floating_base_convention():
            return None
        return [3, 4, 5, 0, 1, 2]

    def normalize_floating_base_q_input(self, q):
        q = np.asarray(q, dtype=np.float64).copy()
        permutation = self.get_floating_base_q_input_permutation_to_internal()
        if permutation is None:
            return q
        if q.shape[0] < len(permutation):
            raise ValueError("Floating-base quaternion input must have at least 7 entries.")
        q[: len(permutation)] = q[permutation]
        return q

    def normalize_floating_base_v_input(self, vec):
        vec = np.asarray(vec, dtype=np.float64).copy()
        permutation = self.get_floating_base_v_permutation_to_internal()
        if permutation is None:
            return vec
        if vec.shape[0] < len(permutation):
            raise ValueError("Floating-base velocity-like input must have at least 6 entries.")
        vec[: len(permutation)] = vec[permutation]
        return vec

    def denormalize_floating_base_q_output(self, q):
        q = np.asarray(q, dtype=np.float64).copy()
        permutation = self.get_floating_base_q_output_permutation_from_internal()
        if permutation is None:
            return q
        if q.shape[0] < len(permutation):
            raise ValueError("Floating-base quaternion output must have at least 7 entries.")
        q[: len(permutation)] = q[permutation]
        return q

    def denormalize_floating_base_v_output(self, vec):
        vec = np.asarray(vec, dtype=np.float64).copy()
        permutation = self.get_floating_base_v_permutation_from_internal()
        if permutation is None:
            return vec
        if vec.shape[0] < len(permutation):
            raise ValueError("Floating-base velocity-like output must have at least 6 entries.")
        vec[: len(permutation)] = vec[permutation]
        return vec

    #################
    #    Setters    #
    #################

    def add_joint(self, joint):
        self.joints.append(joint)
        self.refresh_joint_metadata()

    def add_link(self, link):
        self.links.append(link)

    def add_fixed_joint(self, fixed_joint):
        self.fixed_joints.append(fixed_joint)

    def remove_joint(self, joint):
        self.joints.remove(joint)
        self.refresh_joint_metadata()

    def remove_link(self, link):
        self.links.remove(link)

    def refresh_joint_metadata(self):
        self.joint_type_by_id = {joint.jid: joint.jtype for joint in self.joints}
        self.joint_type_by_name = {joint.name: joint.jtype for joint in self.joints}
        self._refresh_mimic_index_maps()

    def _refresh_mimic_index_maps(self):
        """Build dense (q, v) index maps that skip mimic joints.

        Mimic joints don't own a generalized coordinate. To preserve
        compatibility with downstream code that does `q[get_joint_index_q(j)]`
        (and similar for v), we map each jid to the dense slot that ACTUALLY
        carries its value: a mimic joint maps to the mimicked joint's slot
        (its transform sees `multiplier * q[target] + offset`), and
        non-mimic joints map to a fresh dense slot in joint-id order.
        """
        self._dense_q_offset_by_id = {}
        self._dense_v_offset_by_id = {}
        sorted_joints = self.get_joints_ordered_by_id()
        # First pass: assign dense (q, v) offsets to non-mimic joints in jid order.
        q_cursor = 0
        v_cursor = 0
        if self.floating_base and sorted_joints:
            # The floating root is always non-mimic, sits at jid=0, and reserves
            # the standard 7- or 6-wide block at the start of q / v.
            fb_joint = self.get_joint_by_id(0)
            self._dense_q_offset_by_id[0] = 0
            self._dense_v_offset_by_id[0] = 0
            q_cursor = 7 if self.using_quaternion else 6
            v_cursor = 6
            start_index = 1
        else:
            start_index = 0
        for joint in sorted_joints[start_index:]:
            if getattr(joint, "is_mimic", False):
                continue
            self._dense_q_offset_by_id[joint.jid] = q_cursor
            self._dense_v_offset_by_id[joint.jid] = v_cursor
            q_cursor += joint.local_q_dim if joint.local_q_dim else 1
            v_cursor += joint.dof
        # Second pass: mimic joints inherit the dense (q, v) offset of their
        # target.  At this stage `mimic_target_id` may not yet be resolved
        # (called from add_joint, before resolve_mimic_targets); fall back to
        # the joint's own jid which keeps `q[get_joint_index_q(j)]` defined for
        # debugging output before the parse completes.
        for joint in sorted_joints:
            if not getattr(joint, "is_mimic", False):
                continue
            tgt = joint.mimic_target_id if joint.mimic_target_id is not None else -1
            if tgt in self._dense_q_offset_by_id:
                self._dense_q_offset_by_id[joint.jid] = self._dense_q_offset_by_id[tgt]
                self._dense_v_offset_by_id[joint.jid] = self._dense_v_offset_by_id[tgt]
            else:
                # Resolution pending or target unparented (mimic of fixed
                # joint); point at slot 0 so transform-eval q_arg returns a
                # finite scalar (mimic_offset is applied on top in q_for_joint).
                self._dense_q_offset_by_id[joint.jid] = 0
                self._dense_v_offset_by_id[joint.jid] = 0

    def q_for_joint(self, jid, q):
        """Return the local-q block (scalar or array) to feed joint `jid`'s
        transform function, accounting for `<mimic>` joints.

        For a regular joint this is just `q[get_joint_index_q(jid)]`. For a
        mimic joint it's `multiplier * q[get_joint_index_q(target)] + offset`,
        i.e. the value the URDF mimic relation prescribes.
        """
        joint = self.get_joint_by_id(jid)
        inds = self.get_joint_index_q(jid)
        if not isinstance(inds, (list, tuple, np.ndarray)):
            inds = [inds]
        block = np.asarray(q, dtype=np.float64)[list(inds)]
        if getattr(joint, "is_mimic", False):
            block = joint.get_mimic_multiplier() * block + joint.get_mimic_offset()
        if block.size == 1:
            return float(block[0])
        return block

    #########################
    #    Generic Getters    #
    #########################

    def get_num_pos(self):
        """
        Returns the robot's total number of position degrees of freedom. 
        This corresponds to the size of the position(q) array.

        Output:
        - (int) - total position degrees of freedom
        """
        # NQ exceeds NV by one per quaternion-parameterized joint: the
        # floating-base root (when using_quaternion) and every spherical joint
        # (NQ=4, NV=3). All other joints have NQ==NV locally.
        quaternion_offset = 1 if (self.floating_base and self.using_quaternion) else 0
        quaternion_offset += sum(
            1
            for joint in self.joints
            if getattr(joint, "jtype", None) == "spherical"
            and not getattr(joint, "is_mimic", False)
        )
        return self.get_num_vel() + quaternion_offset

    def get_num_vel(self):
        """
        Returns the robot's total number of velocity degrees of freedom.
        This corresponds to the size of the velocity(qd) array.

        Output:
        - (int) - total velocity degrees of freedom
        """
        return sum([joint.get_num_dof() for joint in self.joints])
    
    def get_num_bodies(self):
        return self.get_num_links_effective()

    def get_num_cntrl(self):
        return self.get_num_joints()
    
    def get_num_fixed_joints(self):
        return len(self.fixed_joints)

    def get_name(self):
        return self.name

    def is_serial_chain(self):
        return all([jid - self.get_parent_id(jid) == 1 for jid in range(self.get_num_joints())])

    def get_parent_id(self, lid):
        return self.get_link_by_id(lid).get_parent_id()

    def get_parent_ids(self, lids):
        return [self.get_parent_id(lid) for lid in lids]

    def get_unique_parent_ids(self, lids):
        return list(set(self.get_parent_ids(lids)))

    def get_parent_id_array(self):
        return [tpl[1] for tpl in sorted([(link.get_id(),link.get_parent_id()) for link in self.links], key=lambda tpl: tpl[0])[1:]]

    def has_repeated_parents(self, jids):
        return len(self.get_parent_ids(jids)) != len(self.get_unique_parent_ids(jids))

    def get_subtree_by_id(self, lid):
        return sorted(self.get_link_by_id(lid).get_subtree())

    def get_total_subtree_count(self):
        return sum([len(self.get_subtree_by_id(lid)) for lid in range(self.get_num_joints())])

    def get_ancestors_by_id(self, jid):
        ancestors = []
        curr_id = jid
        while True:
            curr_id = self.get_parent_id(curr_id)
            if curr_id == -1:
                break
            else:
                ancestors.append(curr_id)
        return ancestors
    
    def get_max_num_ancestors(self):
        return max(len(self.get_ancestors_by_id(jid)) for jid in range(self.get_num_joints()))

    def get_total_ancestor_count(self):
        return sum([len(self.get_ancestors_by_id(jid)) for jid in range(self.get_num_joints())])

    def get_is_ancestor_of(self, jid, jid_of):
        return jid in self.get_ancestors_by_id(jid_of)

    def get_is_in_subtree_of(self, jid, jid_of):
        return jid in self.get_subtree_by_id(jid_of)

    def get_max_bfs_level(self):
        return sorted(self.joints, key=lambda joint: joint.bfs_level, reverse = True)[0].bfs_level

    def get_ids_by_bfs_level(self, level):
        return [joint.jid for joint in self.get_joints_by_bfs_level(level)]

    def get_bfs_level_by_id(self, jid):
        return(self.get_joint_by_id(jid).get_bfs_level())

    def get_max_bfs_width(self):
        return max([len(self.get_ids_by_bfs_level(level)) for level in range(self.get_max_bfs_level() + 1)])

    def get_is_leaf_node(self, jid):
        return len(self.get_subtree_by_id(jid)) == 1

    def get_leaf_nodes(self):
        return list(filter(lambda jid: self.get_is_leaf_node(jid), range(self.get_num_joints())))

    def get_total_leaf_nodes(self):
        return len(self.get_leaf_nodes())

    ###############
    #    Joint    #
    ###############

    def get_num_joints(self):
        return len(self.joints)

    def get_joint_by_id(self, jid):
        return self.next_none(filter(lambda fjoint: fjoint.jid == jid, self.joints))

    def get_joint_by_name(self, name):
        return self.next_none(filter(lambda fjoint: fjoint.name == name, self.joints))

    def get_joints_by_bfs_level(self, level):
        return list(filter(lambda fjoint: fjoint.bfs_level == level, self.joints))

    def get_joints_ordered_by_id(self, reverse = False):
        return sorted(self.joints, key=lambda item: item.jid, reverse = reverse)

    def get_joints_ordered_by_name(self, reverse = False):
        return sorted(self.joints, key=lambda item: item.name, reverse = reverse)

    def get_joints_dict_by_id(self):
        return {joint.jid:joint for joint in self.joints}

    def get_joints_dict_by_name(self):
        return {joint.name:joint for joint in self.joints}

    def get_joint_type_by_id(self, jid):
        return self.joint_type_by_id.get(jid)

    def get_joint_type_by_name(self, name):
        return self.joint_type_by_name.get(name)

    def get_joint_types_by_id(self):
        return dict(self.joint_type_by_id)

    def get_joint_types_by_name(self):
        return dict(self.joint_type_by_name)

    def get_joints_by_parent_name(self, parent_name):
        return list(filter(lambda fjoint: fjoint.parent == parent_name, self.joints))

    def get_joints_by_child_name(self, child_name):
        return list(filter(lambda fjoint: fjoint.child == child_name, self.joints))

    def get_joint_by_parent_child_name(self, parent_name, child_name):
        return self.next_none(filter(lambda fjoint: fjoint.parent == parent_name and fjoint.child == child_name, self.joints))

    def get_joint_limits_by_id(self, jid):
        return self.get_joint_by_id(jid).get_joint_limits()

    def get_velocity_limit_by_id(self, jid):
        return self.get_joint_by_id(jid).get_velocity_limit()

    def get_effort_limit_by_id(self, jid):
        return self.get_joint_by_id(jid).get_effort_limit()

    def get_damping_by_id(self, jid):
        return self.get_joint_by_id(jid).get_damping()

    def get_friction_by_id(self, jid):
        return self.get_joint_by_id(jid).get_friction()

    def robot_has_joint_damping(self):
        """True if ANY joint declares nonzero viscous damping. Codegen gates the
        additive damping bias on this so a robot without damping (the common
        case) emits byte-identical CUDA."""
        return any(float(getattr(j, "damping", 0) or 0) != 0.0 for j in self.joints)

    def robot_has_joint_friction(self):
        """True if ANY joint declares nonzero Coulomb friction. Same byte-neutral
        codegen gate as robot_has_joint_damping."""
        return any(float(getattr(j, "friction", 0) or 0) != 0.0 for j in self.joints)

    def get_joint_position_dim_by_id(self, jid):
        return self.get_joint_by_id(jid).get_local_q_dim()

    def get_children_by_id(self, jid):
        """
        Gets the joint children of a joint by its id.

        Inputs:
            - (int) jid - the joint id
        
        Returns:
            - [(int)] - the ids of the children of the joint
        """
        chilren = []
        for joint in range(self.get_num_joints()):
            # Check if joint is a child of jid => if jid is an ancestor of joint
            if jid in self.get_ancestors_by_id(joint):
                chilren.append(joint)
        return chilren
    
    
    def get_jid_ancestor_ids(self, include_joint=False):
        """
        Used to generate the ids of the joints and their ancestors
        as two lists.

        The output is formatted such that the first list contains the 
        ids of each joint and the second list contains the ids
        of the ancestors of that joint.

        Example: Joint 0 has ancestors [1, 2], Joint 1 has ancestors [2],
        then the output would be ([0, 0, 1], [1, 2, 2]).

        Inputs:
            - (bool) include_joint - whether to include the joint itself in the
                output as its own ancestor

        Returns:
            - ([(int)], [(int)]) - indices of joints & indices of ancestors
        """
        jids = []
        ancestors = []
        for joint in range(self.get_num_joints()):
            ancestors_j = self.get_ancestors_by_id(joint)
            if include_joint:
                jids.append(joint)
                ancestors.append(joint)
            for i, ancestor in enumerate(ancestors_j): 
                jids.append(joint)
                ancestors.append(ancestor)
        return jids, ancestors
    
    def get_jid_ancestor_st_ids(self, include_joint=False):
        """
        Used to generate the ids of the joints, their ancestors,
        and their subtree as three lists.

        The output is formatted such that the first list contains the 
        ids of each joint, the second list contains the ids
        of the ancestors of that joint, and the third list contains
        the subtree of that joint.

        Example: Joint 2 has ancestors [0, 1], and subtree [2, 3, 4],
        then the output will be [2, 2, 2, 2, 2, 2], [0, 0, 0, 1, 1, 1], [2, 3, 4, 2, 3, 4].

        Inputs:
            - (bool) include_joint - whether to include the joint itself in the
                output as its own ancestor

        Returns:
            - ([(int)], [(int)]) - indices of joints & indices of ancestors
        """
        jids = []
        ancestors = []
        st = []
        for joint in range(self.get_num_joints()):
            ancestors_j = self.get_ancestors_by_id(joint)
            st_j = self.get_subtree_by_id(joint)
            if include_joint:
                jids += [joint]*len(st_j)
                ancestors += [joint]*len(st_j)
                st += st_j
            for i, ancestor in enumerate(ancestors_j): 
                jids += [joint]*len(st_j)
                ancestors += [ancestor]*len(st_j)
                st += st_j
        return jids, ancestors, st
    
    def get_children_by_id(self, jid):
        """
        Gets the joint children of a joint by its id.

        Inputs:
            - (int) jid - the joint id
        
        Returns:
            - [(int)] - the ids of the children of the joint
        """
        chilren = []
        for joint in range(self.get_num_joints()):
            # Check if joint is a child of jid => if jid is an ancestor of joint
            if jid in self.get_ancestors_by_id(joint):
                chilren.append(joint)
        return chilren
    
    
    def get_jid_ancestor_ids(self, include_joint=False):
        """
        Used to generate the ids of the joints and their ancestors
        as two lists.

        The output is formatted such that the first list contains the 
        ids of each joint and the second list contains the ids
        of the ancestors of that joint.

        Example: Joint 0 has ancestors [1, 2], Joint 1 has ancestors [2],
        then the output would be ([0, 0, 1], [1, 2, 2]).

        Inputs:
            - (bool) include_joint - whether to include the joint itself in the
                output as its own ancestor

        Returns:
            - ([(int)], [(int)]) - indices of joints & indices of ancestors
        """
        jids = []
        ancestors = []
        for joint in range(self.get_num_joints()):
            ancestors_j = self.get_ancestors_by_id(joint)
            if include_joint:
                jids.append(joint)
                ancestors.append(joint)
            for i, ancestor in enumerate(ancestors_j): 
                jids.append(joint)
                ancestors.append(ancestor)
        return jids, ancestors
    
    def get_jid_ancestor_st_ids(self, include_joint=False):
        """
        Used to generate the ids of the joints, their ancestors,
        and their subtree as three lists.

        The output is formatted such that the first list contains the 
        ids of each joint, the second list contains the ids
        of the ancestors of that joint, and the third list contains
        the subtree of that joint.

        Example: Joint 2 has ancestors [0, 1], and subtree [2, 3, 4],
        then the output will be [2, 2, 2, 2, 2, 2], [0, 0, 0, 1, 1, 1], [2, 3, 4, 2, 3, 4].

        Inputs:
            - (bool) include_joint - whether to include the joint itself in the
                output as its own ancestor

        Returns:
            - ([(int)], [(int)]) - indices of joints & indices of ancestors
        """
        jids = []
        ancestors = []
        st = []
        for joint in range(self.get_num_joints()):
            ancestors_j = self.get_ancestors_by_id(joint)
            st_j = self.get_subtree_by_id(joint)
            if include_joint:
                jids += [joint]*len(st_j)
                ancestors += [joint]*len(st_j)
                st += st_j
            for i, ancestor in enumerate(ancestors_j): 
                jids += [joint]*len(st_j)
                ancestors += [ancestor]*len(st_j)
                st += st_j
        return jids, ancestors, st


    ##############
    #    Link    #
    ##############

    def get_num_links(self):
        return len(self.links)

    def get_num_links_effective(self):
        # subtracting base link from total # of links
        return self.get_num_links() - 1

    def get_link_by_id(self, lid):
        return self.next_none(filter(lambda flink: flink.lid == lid, self.links))

    def get_link_by_name(self, name):
        return self.next_none(filter(lambda flink: flink.name == name, self.links))

    def get_links_by_bfs_level(self, level):
        return list(filter(lambda flink: flink.bfs_level == level, self.links))

    def get_links_ordered_by_id(self, reverse = False):
        return sorted(self.links, key=lambda item: item.lid, reverse = reverse)

    def get_links_ordered_by_name(self, reverse = False):
        return sorted(self.links, key=lambda item: item.name, reverse = reverse)

    def get_links_dict_by_id(self):
        return {link.lid:link for link in self.links}

    def get_links_dict_by_name(self):
        return {link.name:link for link in self.links}

    ##############
    #    XMAT    #
    ##############

    def get_Xmat_by_id(self, jid):
        return self.get_joint_by_id(jid).get_transformation_matrix()

    def get_Xmat_by_name(self, name):
        return self.get_joint_by_name(name).get_transformation_matrix()

    def get_Xmats_by_bfs_level(self, level):
        return [joint.get_transformation_matrix() for joint in self.get_joints_by_bfs_level(level)]

    def get_Xmats_ordered_by_id(self, reverse = False):
        return [joint.get_transformation_matrix() for joint in self.get_joints_ordered_by_id(reverse)]

    def get_Xmats_ordered_by_name(self, reverse = False):
        return [joint.get_transformation_matrix() for joint in self.get_joints_ordered_by_name(reverse)]

    def get_Xmats_dict_by_id(self):
        return {joint.jid:joint.get_transformation_matrix() for joint in self.joints}

    def get_Xmats_dict_by_name(self):
        return {joint.name:joint.get_transformation_matrix() for joint in self.joints}

    ###################
    #    XMAT_Func    #
    ###################

    def get_Xmat_Func_by_id(self, jid):
        return self.get_joint_by_id(jid).get_transformation_matrix_function()

    def get_Xmat_Func_by_name(self, name):
        return self.get_joint_by_name(name).get_transformation_matrix_function()

    def get_Xmat_Funcs_by_bfs_level(self, level):
        return [joint.get_transformation_matrix_function() for joint in self.get_joints_by_bfs_level(level)]

    def get_Xmat_Funcs_ordered_by_id(self, reverse = False):
        return [joint.get_transformation_matrix_function() for joint in self.get_joints_ordered_by_id(reverse)]

    def get_Xmat_Funcs_ordered_by_name(self, reverse = False):
        return [joint.get_transformation_matrix_function() for joint in self.get_joints_ordered_by_name(reverse)]

    def get_Xmat_Funcs_dict_by_id(self):
        return {joint.jid:joint.get_transformation_matrix_function() for joint in self.joints}

    def get_Xmat_Funcs_dict_by_name(self):
        return {joint.name:joint.get_transformation_matrix_function() for joint in self.joints}

    ##################
    #    XMAT_hom    #
    ##################

    def get_Xmat_hom_by_id(self, jid):
        return self.get_joint_by_id(jid).get_transformation_matrix_hom()

    def get_Xmat_hom_by_name(self, name):
        return self.get_joint_by_name(name).get_transformation_matrix_hom()

    def get_Xmats_hom_by_bfs_level(self, level):
        return [joint.get_transformation_matrix_hom() for joint in self.get_joints_by_bfs_level(level)]

    def get_Xmats_hom_ordered_by_id(self, reverse = False, include_fixed_joints = False):
        base = [joint.get_transformation_matrix_hom() for joint in self.get_joints_ordered_by_id(reverse)]
        fixed = [joint.get_transformation_matrix_hom() for joint in self.get_fixed_joints_ordered_by_id(reverse)]
        return base + fixed if include_fixed_joints else base

    def get_Xmats_hom_ordered_by_name(self, reverse = False):
        return [joint.get_transformation_matrix_hom() for joint in self.get_joints_ordered_by_name(reverse)]

    def get_Xmats_hom_dict_by_id(self):
        return {joint.jid:joint.get_transformation_matrix_hom() for joint in self.joints}

    def get_Xmats_hom_dict_by_name(self):
        return {joint.name:joint.get_transformation_matrix_hom() for joint in self.joints}

    #######################
    #    Xmat_hom_Func    #
    #######################

    def get_Xmat_hom_Func_by_id(self, jid):
        return self.get_joint_by_id(jid).get_transformation_matrix_hom_function()

    def get_Xmat_hom_Func_by_name(self, name):
        return self.get_joint_by_name(name).get_transformation_matrix_hom_function()

    def get_Xmat_hom_Funcs_by_bfs_level(self, level):
        return [joint.get_transformation_matrix_hom_function() for joint in self.get_joints_by_bfs_level(level)]

    def get_Xmat_hom_Funcs_ordered_by_id(self, reverse = False):
        return [joint.get_transformation_matrix_hom_function() for joint in self.get_joints_ordered_by_id(reverse)]

    def get_Xmat_hom_Funcs_ordered_by_name(self, reverse = False):
        return [joint.get_transformation_matrix_hom_function() for joint in self.get_joints_ordered_by_name(reverse)]

    def get_Xmat_hom_Funcs_dict_by_id(self):
        return {joint.jid:joint.get_transformation_matrix_hom_function() for joint in self.joints}

    def get_Xmat_hom_Funcs_dict_by_name(self):
        return {joint.name:joint.get_transformation_matrix_hom_function() for joint in self.joints}

    ##################
    #    dXmat_hom    #
    ##################

    def get_dXmat_hom_by_id(self, jid):
        return self.get_joint_by_id(jid).get_dtransformation_matrix_hom()

    def get_dXmat_hom_by_name(self, name):
        return self.get_joint_by_name(name).get_dtransformation_matrix_hom()

    def get_dXmats_hom_by_bfs_level(self, level):
        return [joint.get_dtransformation_matrix_hom() for joint in self.get_joints_by_bfs_level(level)]

    def get_dXmats_hom_ordered_by_id(self, reverse = False):
        return [joint.get_dtransformation_matrix_hom() for joint in self.get_joints_ordered_by_id(reverse)]

    def get_dXmats_hom_ordered_by_name(self, reverse = False):
        return [joint.get_dtransformation_matrix_hom() for joint in self.get_joints_ordered_by_name(reverse)]

    def get_dXmats_hom_dict_by_id(self):
        return {joint.jid:joint.get_dtransformation_matrix_hom() for joint in self.joints}

    def get_dXmats_hom_dict_by_name(self):
        return {joint.name:joint.get_dtransformation_matrix_hom() for joint in self.joints}

    #######################
    #    dXmat_hom_Func    #
    #######################

    def get_dXmat_hom_Func_by_id(self, jid):
        return self.get_joint_by_id(jid).get_dtransformation_matrix_hom_function()

    def get_dXmat_hom_local_by_id(self, jid, local_index):
        return self.get_joint_by_id(jid).get_dtransformation_matrix_hom_local(local_index)

    def get_dXmat_hom_local_Func_by_id(self, jid, local_index):
        return self.get_joint_by_id(jid).get_dtransformation_matrix_hom_local_function(local_index)

    def get_dXmat_hom_Func_by_name(self, name):
        return self.get_joint_by_name(name).get_dtransformation_matrix_hom_function()

    def get_dXmat_hom_Funcs_by_bfs_level(self, level):
        return [joint.get_dtransformation_matrix_hom_function() for joint in self.get_joints_by_bfs_level(level)]

    def get_dXmat_hom_Funcs_ordered_by_id(self, reverse = False):
        return [joint.get_dtransformation_matrix_hom_function() for joint in self.get_joints_ordered_by_id(reverse)]

    def get_dXmat_hom_Funcs_ordered_by_name(self, reverse = False):
        return [joint.get_dtransformation_matrix_hom_function() for joint in self.get_joints_ordered_by_name(reverse)]

    def get_dXmat_hom_Funcs_dict_by_id(self):
        return {joint.jid:joint.get_dtransformation_matrix_hom_function() for joint in self.joints}

    def get_dXmat_hom_Funcs_dict_by_name(self):
        return {joint.name:joint.get_dtransformation_matrix_hom_function() for joint in self.joints}

    ##################
    #   d2Xmat_hom   #
    ##################

    def get_d2Xmat_hom_by_id(self, jid):
        return self.get_joint_by_id(jid).get_d2transformation_matrix_hom()

    def get_d2Xmat_hom_by_name(self, name):
        return self.get_joint_by_name(name).get_d2transformation_matrix_hom()

    def get_d2Xmats_hom_by_bfs_level(self, level):
        return [joint.get_d2transformation_matrix_hom() for joint in self.get_joints_by_bfs_level(level)]

    def get_d2Xmats_hom_ordered_by_id(self, reverse = False):
        return [joint.get_d2transformation_matrix_hom() for joint in self.get_joints_ordered_by_id(reverse)]

    def get_d2Xmats_hom_ordered_by_name(self, reverse = False):
        return [joint.get_d2transformation_matrix_hom() for joint in self.get_joints_ordered_by_name(reverse)]

    def get_d2Xmats_hom_dict_by_id(self):
        return {joint.jid:joint.get_d2transformation_matrix_hom() for joint in self.joints}

    def get_d2Xmats_hom_dict_by_name(self):
        return {joint.name:joint.get_d2transformation_matrix_hom() for joint in self.joints}

    #######################
    #   d2Xmat_hom_Func   #
    #######################

    def get_d2Xmat_hom_Func_by_id(self, jid):
        return self.get_joint_by_id(jid).get_d2transformation_matrix_hom_function()

    def get_d2Xmat_hom_local_by_id(self, jid, local_index_i, local_index_j):
        return self.get_joint_by_id(jid).get_d2transformation_matrix_hom_local(local_index_i, local_index_j)

    def get_d2Xmat_hom_local_Func_by_id(self, jid, local_index_i, local_index_j):
        return self.get_joint_by_id(jid).get_d2transformation_matrix_hom_local_function(local_index_i, local_index_j)

    def get_d2Xmat_hom_Func_by_name(self, name):
        return self.get_joint_by_name(name).get_d2transformation_matrix_hom_function()

    def get_d2Xmat_hom_Funcs_by_bfs_level(self, level):
        return [joint.get_d2transformation_matrix_hom_function() for joint in self.get_joints_by_bfs_level(level)]

    def get_d2Xmat_hom_Funcs_ordered_by_id(self, reverse = False):
        return [joint.get_d2transformation_matrix_hom_function() for joint in self.get_joints_ordered_by_id(reverse)]

    def get_d2Xmat_hom_Funcs_ordered_by_name(self, reverse = False):
        return [joint.get_d2transformation_matrix_hom_function() for joint in self.get_joints_ordered_by_name(reverse)]

    def get_d2Xmat_hom_Funcs_dict_by_id(self):
        return {joint.jid:joint.get_d2transformation_matrix_hom_function() for joint in self.joints}

    def get_d2Xmat_hom_Funcs_dict_by_name(self):
        return {joint.name:joint.get_d2transformation_matrix_hom_function() for joint in self.joints}

    ##############
    #    IMAT    #
    ##############

    def get_Imat_by_id(self, lid):
        return self.get_link_by_id(lid).get_spatial_inertia()

    def get_Imat_by_name(self, name):
        return self.get_joint_by_name(name).get_spatial_inertia()

    def get_Imats_by_bfs_level(self, level):
        return [link.get_spatial_inertia() for link in self.get_links_by_bfs_level()]

    def get_Imats_ordered_by_id(self, reverse = False):
        return [link.get_spatial_inertia() for link in self.get_links_ordered_by_id(reverse)]

    def get_Imats_ordered_by_name(self, reverse = False):
        return [link.get_spatial_inertia() for link in self.get_links_ordered_by_name(reverse)]

    def get_inertia_params_ordered_by_id(self, reverse = False):
        # Per-link 10-vector [m, h(3)=m*c, I_O(6)] in the frozen GRiD/URDF
        # regressor basis (see Link.get_inertia_params). Body-indexed, mimic-
        # agnostic (inertia is per LINK, never reduced DoF) — mirrors the
        # shipped inverse_dynamics_regressor's NB-wide parameter layout.
        return [link.get_inertia_params() for link in self.get_links_ordered_by_id(reverse)]

    def get_origin_params_ordered_by_id(self, reverse = False):
        # Per-joint 6-vector [x, y, z, roll, pitch, yaw] of raw URDF <origin>
        # scalars in the frozen runtime_transform basis (see
        # Joint.get_origin_params). Joint-indexed, mirrors get_Xmats_ordered_by_id
        # — the on-device prologue rebuilds each joint's constant Xfixed from
        # these 6 scalars once per launch (runtime_transform path).
        return [joint.get_origin_params() for joint in self.get_joints_ordered_by_id(reverse)]

    def get_runtime_transform_mats_ordered_by_id(self, reverse = False):
        # Per-joint spatial X transform with the origin block carried as named
        # xf_* symbols (see Joint.get_runtime_transform_matrix). Used by the
        # runtime_transform codegen to bake the DENSE rpy sparsity and hoist the
        # origin out of the hot sin/cos(q) loop into s_Xfixed scratch loads.
        return [joint.get_runtime_transform_matrix() for joint in self.get_joints_ordered_by_id(reverse)]

    def get_Imats_dict_by_id(self):
        return {link.lid:link.get_spatial_inertia() for link in self.links}

    def get_Imats_dict_by_name(self):
        return {link.name:link.get_spatial_inertia() for link in self.links}

    ###############
    #      S      #
    ###############

    def get_S_by_id(self, jid):
        return self.get_joint_by_id(jid).get_joint_subspace()

    def _get_flat_S_by_id(self, jid):
        return self.get_S_by_id(jid).reshape(-1).tolist()

    def _assert_single_axis_S(self, jid):
        """Guard the single-signed-index S assumption these helpers encode.

        The CUDA codegen's FAST PATH represents each joint's motion subspace as
        ONE signed index. Two distinct things break that:

        - A multi-DOF PLANAR / SPHERICAL joint (6xN, N>1 S) -> Tier C, deferred.
        - A single-column SKEW axis (revolute/prismatic with a non-cardinal
          <axis>): the column has >=2 nonzero entries, so there is no single
          signed unit index -> Tier B (Phase-6 STAGE 1). Tier B does NOT go
          through these signed-index helpers; it consumes the dense 6-vector
          via get_S_by_id. Reaching a signed-index helper with a skew joint is
          therefore a bug in an un-ported algorithm, so we fail loudly.

        The floating-base free-flyer root is EXEMPT (established 6-DOF path).
        """
        joint = self.get_joint_by_id(jid)
        dof = joint.get_num_dof() if joint is not None else 1
        jtype = getattr(joint, "jtype", "?") if joint is not None else "?"
        if dof and dof > 1 and jtype != "floating":
            raise ValueError(
                f"Joint {jid} (type '{jtype}', dof={dof}) has a multi-column motion "
                "subspace; the single-signed-index S helpers do not support it. "
                "Multi-column-S CUDA emit (Tier C) is deferred (see "
                "docs/open-tasks/phase6_joint_types_plan_REFRESH.md, item 4)."
            )
        if not self.S_is_cardinal_by_id(jid):
            raise ValueError(
                f"Joint {jid} (type '{jtype}') has a SKEW/general motion subspace "
                f"column ({self._get_flat_S_by_id(jid)}); it has no single signed "
                "unit index. Use the Tier-B dense-6-vector path (get_S_by_id). "
                "This algorithm has not been ported to Tier B yet "
                "(Phase-6 STAGE 1 covers inverse_dynamics + crba)."
            )

    def S_is_cardinal_by_id(self, jid):
        """True if joint `jid`'s motion subspace is a single signed cardinal
        unit axis (Tier A: exactly one |entry|==1, all others 0). The floating
        root is treated as cardinal here (its per-column identity subspace is
        handled by the established multi-column floating path, not Tier B)."""
        joint = self.get_joint_by_id(jid)
        if joint is not None and getattr(joint, "jtype", None) == "floating":
            return True
        S = self._get_flat_S_by_id(jid)
        unit_count = sum(1 for v in S if abs(v) == 1)
        nonzero_count = sum(1 for v in S if v != 0)
        return unit_count == 1 and nonzero_count == 1

    def joint_is_spherical(self, jid):
        """True if joint `jid` is a SPHERICAL (3-DoF ball) joint. Per-JOINT peer of
        S_is_cardinal_by_id / robot_has_spherical (the per-ROBOT any())."""
        return getattr(self.get_joint_by_id(jid), "jtype", None) == "spherical"

    def robot_has_spherical(self):
        """True if ANY (non-mimic) joint is a SPHERICAL (3-DoF ball) joint.

        Codegen uses this to decide AT CODEGEN TIME whether to emit the additive
        Tier-C spherical machinery (multi-column angular-identity S forward/back
        + the 4-wide quaternion q-block transform). A robot with no spherical
        joint never enters those branches, so its CUDA stays byte-identical to
        the all-cardinal path. (The planar 3-DoF joint is decomposed into
        cardinal sub-joints at parse time, so it never reaches here; only the
        spherical manifold joint survives as a true multi-column non-root S.)"""
        for joint in self.joints:
            if getattr(joint, "is_mimic", False):
                continue
            if getattr(joint, "jtype", None) == "spherical":
                return True
        return False

    def robot_has_skew_axis(self):
        """True if ANY joint carries a Tier-B skew/general single-column S.
        Codegen uses this to decide (at codegen time) whether to emit the
        additive Tier-B machinery; an all-cardinal robot stays byte-identical."""
        for joint in self.joints:
            jid = joint.get_id()
            j = self.get_joint_by_id(jid)
            if j is not None and getattr(j, "jtype", None) in ("floating", "planar", "spherical"):
                continue
            if j is not None and j.get_num_dof() == 0:
                continue
            if not self.S_is_cardinal_by_id(jid):
                return True
        return False

    def get_S_index_by_id(self, jid):
        self._assert_single_axis_S(jid)
        S = self._get_flat_S_by_id(jid)
        for index, value in enumerate(S):
            if abs(value) == 1:
                return index
        raise ValueError("Joint subspace does not contain a unit axis.")

    def get_S_sign_by_id(self, jid):
        self._assert_single_axis_S(jid)
        S = self._get_flat_S_by_id(jid)
        for value in S:
            if abs(value) == 1:
                return int(value)
        raise ValueError("Joint subspace does not contain a unit axis.")

    def get_signed_S_index_by_id(self, jid):
        return self.get_S_sign_by_id(jid) * (self.get_S_index_by_id(jid) + 1)

    def get_S_by_name(self, name):
        return self.get_joint_by_name(name).get_joint_subspace()

    def get_S_by_bfs_level(self, level):
        return [joint.get_joint_subspace() for joint in self.get_joints_by_bfs_level(level)]

    def get_Ss_ordered_by_id(self, reverse = False):
        return [joint.get_joint_subspace() for joint in self.get_joints_ordered_by_id(reverse)]

    def get_Ss_ordered_by_name(self, reverse = False):
        return [joint.get_joint_subspace() for joint in self.get_joints_ordered_by_name(reverse)]

    def get_Ss_dict_by_id(self):
        return {joint.jid:joint.get_joint_subspace() for joint in self.joints}

    def get_Ss_dict_by_name(self):
        return {joint.name:joint.get_joint_subspace() for joint in self.joints}

    def are_Ss_identical(self,jids):
        """
        Returns whether all joints have the same subspace matrix.
        If the robot has a floating base, the method will return False.
        This method is for optimizations during code generation.

        Outputs:
        - (bool) - True/False whether all joints have the same subspace matrix
        """
        if self.floating_base: return False
        # Multi-column motion subspaces (spherical 6x3) can't be "identical" to a
        # single-column revolute/prismatic S and would crash the elementwise
        # compare below (shape mismatch -> ambiguous truth value). A robot with
        # any multi-DoF non-root joint is never uniform-S, so report False (the
        # topology-helper path emits per-joint S). For an all-single-DoF robot
        # this is byte-identical to the legacy elementwise compare.
        S0 = np.asarray(self.get_S_by_id(jids[0]))
        for jid in jids:
            Sj = np.asarray(self.get_S_by_id(jid))
            if Sj.shape != S0.shape or not np.array_equal(Sj, S0):
                return False
        return True
    
    def get_S_inds(self, n):
        """
        Returns the index of the 1 in each joint's subspace matrix up to joint n.
        If the robot has a floating base, then the first six entries in the
        S_inds list will be the indices of the 1's in each column of the 
        floating base's subspace matrix.

        Inputs:
        -   (int) n - the total number of joints to get indices for

        Outputs:
        -   [(int)] - the index of the 1 in each of the n subspace matrices
        """
        # Tier-B (skew axis) joints have no signed unit index; their Tier-B emit
        # bakes the dense S column directly (it never reads this table), so we
        # emit a 0 placeholder to keep the per-jid S_inds table well-formed.
        # Cardinal robots contain no skew joints, so the table is byte-identical.
        def _signed_or_placeholder(jid):
            if self.S_is_cardinal_by_id(jid):
                return str(self.get_signed_S_index_by_id(jid))
            return "0"
        if self.floating_base:
            fb_S = self.get_S_by_id(0).T.tolist() # break fb S into each column
            S_inds = []
            for dof in fb_S:
                S_inds.append(str(next((1 if value > 0 else -1) * (index + 1) for index, value in enumerate(dof) if abs(value) == 1))) # signed one-based unit-axis index
            for jid in range(1,n):
                S_inds.append(_signed_or_placeholder(jid)) # take the rest
        else:
            S_inds = [_signed_or_placeholder(jid) for jid in range(n)]
        return S_inds

    ######################
    #    Fixed Joints    #
    ######################

    def get_fixed_joint_by_name(self, name):
        return self.next_none(filter(lambda fjoint: fjoint.name == name, self.fixed_joints))
    
    def get_fixed_joint_by_id(self, jid):
        return self.next_none(filter(lambda fjoint: fjoint.jid == jid, self.fixed_joints))

    def get_fixed_joint_by_parent_name(self, parent_name):
        return self.next_none(filter(lambda fjoint: fjoint.parent_name == parent_name, self.fixed_joints))

    def get_fixed_joint_names(self):
        return [fjoint.name for fjoint in self.fixed_joints]
    
    def get_fixed_joints_ordered_by_id(self, reverse = False):
        return sorted(self.fixed_joints, key=lambda item: item.jid, reverse = reverse)
