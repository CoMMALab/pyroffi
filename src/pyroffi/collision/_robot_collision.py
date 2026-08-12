from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, cast
import math

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import trimesh
import yourdfpy
from jaxtyping import Array, Bool, Float, Int
from jax import lax
from loguru import logger

if TYPE_CHECKING:
    from pyroffi._robot import Robot

from .._robot_urdf_parser import RobotURDFParser
from ._collision import collide, pairwise_collide
from ._geometry import Capsule, CollGeom, Sphere


class _AttachmentFieldsMixin:
    """Shared attachment behaviour for both collision layouts.

    The fields themselves must be declared on each ``jdc.pytree_dataclass``
    (a mixin cannot contribute dataclass fields), but everything derived from
    them lives here so the two classes cannot drift apart.
    """

    @property
    def num_robot_links(self) -> int:
        """Geometry entries that are actual robot links (excludes attachments)."""
        return self.num_links - len(self.attach_parent_link_indices)

    def with_attachments(self, aset):
        """Return a copy extended with an :class:`~pyroffi.attachments.AttachmentSet`.

        Sugar for :func:`pyroffi.attachments.compose_collision`.
        """
        from ..attachments import compose_collision

        return compose_collision(self, aset)

    def entry_active(self) -> Bool[Array, " num_links"]:
        """Per-entry activity mask (robot links are always active)."""
        if self.attach_active is None:
            return jnp.ones((self.num_links,), dtype=bool)
        return jnp.concatenate(
            [jnp.ones((self.num_robot_links,), dtype=bool), self.attach_active]
        )

    def _append_attachment_poses(self, Ts_link_world_wxyz_xyz):
        """Append ``T_WB = T_WL(cfg) · T_LB`` rows for every attachment entry.

        One gather off the FK already computed, then one SE(3) compose.
        """
        if not self.attach_parent_link_indices:
            return Ts_link_world_wxyz_xyz
        parents = jnp.asarray(self.attach_parent_link_indices, dtype=jnp.int32)
        T_world_parent = jaxlie.SE3(Ts_link_world_wxyz_xyz[..., parents, :])
        assert self.attach_T_parent_body is not None
        T_world_body = T_world_parent @ jaxlie.SE3(self.attach_T_parent_body)
        return jnp.concatenate(
            [Ts_link_world_wxyz_xyz, T_world_body.wxyz_xyz], axis=-2
        )

    def _mask_inactive_pairs(self, active_distances, idx_i, idx_j):
        """``+inf`` for pairs touching a disabled slot, so they drop out of the
        min-reductions downstream.

        Masked at the distance rather than by zeroing the radius: a zero-radius
        sphere is still a real point obstacle at the parent link and would
        poison those minima.  (``attachments._compose._gate_geom`` reaches the
        same end for geometry that has no pair table yet, via a large *negative*
        radius rather than zero.)
        """
        if self.attach_active is None:
            return active_distances
        live = self.entry_active()
        return jnp.where(live[idx_i] & live[idx_j], active_distances, jnp.inf)

    def _mask_inactive_world(self, dist_matrix):
        """``+inf`` for disabled slots against every obstacle."""
        if self.attach_active is None:
            return dist_matrix
        return jnp.where(self.entry_active()[:, None], dist_matrix, jnp.inf)


@jdc.pytree_dataclass
class RobotCollision(_AttachmentFieldsMixin):
    """Collision model for a robot, integrated with pyroffi kinematics."""

    num_links: jdc.Static[int]
    """Number of links in the model (matches kinematics links)."""
    link_names: jdc.Static[tuple[str, ...]]
    """Names of the links corresponding to link indices."""
    coll: CollGeom
    """Collision geometries for the robot (relative to their parent link frame)."""

    active_idx_i: Int[Array, " P"]
    """Row indices (first link) of active self-collision pairs to check."""
    active_idx_j: Int[Array, " P"]
    """Column indices (second link) of active self-collision pairs to check."""

    # --- Attached bodies (tools, grasped objects). --------------------------
    # Empty for an un-attached model, in which case every field below is inert.
    # Populated by ``pyroffi.attachments.compose_collision``: attachment
    # primitives live in the *tail* of ``coll``, so the pair table, the world
    # launch and every cost reducing over the geometry axis pick them up free.
    attach_parent_link_indices: jdc.Static[tuple[int, ...]] = ()
    """Parent link index per attachment primitive (static topology)."""
    attach_T_parent_body: Optional[Float[Array, "K 7"]] = None
    """``wxyz_xyz`` link<-body transform per primitive. A leaf, so it is
    differentiable and ``vmap``-able."""
    attach_active: Optional[Bool[Array, " K"]] = None
    """Per-primitive activity mask. A leaf, so toggling it never recompiles."""

    @staticmethod
    def from_urdf(
        urdf: yourdfpy.URDF,
        user_ignore_pairs: tuple[tuple[str, str], ...] = (),
        ignore_immediate_adjacents: bool = True,
    ):
        """
        Build a differentiable robot collision model from a URDF.

        Args:
            urdf: The URDF object (used to load collision meshes).
            user_ignore_pairs: Additional pairs of link names to ignore for self-collision.
            ignore_immediate_adjacents: If True, automatically ignore collisions
                between adjacent (parent/child) links based on the URDF structure.
        """
        # Re-load urdf with collision data if not already loaded.
        filename_handler = urdf._filename_handler  # pylint: disable=protected-access
        try:
            has_collision = any(link.collisions for link in urdf.link_map.values())
            if not has_collision:
                urdf = yourdfpy.URDF(
                    robot=urdf.robot,
                    filename_handler=filename_handler,
                    load_collision_meshes=True,
                )
        except Exception as e:
            logger.warning(f"Could not reload URDF with collision meshes: {e}")

        _, link_info = RobotURDFParser.parse(urdf)
        link_name_list = link_info.names  # Use names from parser

        # Gather all collision meshes.
        # The order of cap_list must match link_name_list.
        cap_list = list[Capsule]()
        for link_name in link_name_list:
            cap_list.append(
                Capsule.from_trimesh(
                    RobotCollision._get_trimesh_collision_geometries(urdf, link_name)
                )
            )

        # Convert list of trimesh objects into a batched Capsule object.
        capsules = cast(Capsule, jax.tree.map(lambda *args: jnp.stack(args), *cap_list))
        assert capsules.get_batch_axes() == (link_info.num_links,)

        # Directly compute active pair indices
        active_idx_i, active_idx_j = RobotCollision._compute_active_pair_indices(
            link_names=link_name_list,
            urdf=urdf,
            user_ignore_pairs=user_ignore_pairs,
            ignore_immediate_adjacents=ignore_immediate_adjacents,
        )

        logger.info(
            f"Created RobotCollision with {link_info.num_links} links and "
            f"{len(active_idx_i)} active self-collision pairs."
        )

        return RobotCollision(
            num_links=link_info.num_links,
            link_names=link_name_list,
            active_idx_i=active_idx_i,
            active_idx_j=active_idx_j,
            coll=capsules,
        )

    @staticmethod
    def _compute_active_pair_indices(
        link_names: tuple[str, ...],
        urdf: yourdfpy.URDF,
        user_ignore_pairs: tuple[tuple[str, str], ...],
        ignore_immediate_adjacents: bool,
    ) -> Tuple[Int[Array, " P"], Int[Array, " P"]]:
        """
        Computes the indices (i, j) of pairs where i < j and the pair should
        be actively checked for self-collision.

        Args:
            link_names: Tuple of link names in order.
            urdf: Parsed URDF object.
            user_ignore_pairs: List of (name1, name2) pairs to explicitly ignore.
            ignore_immediate_adjacents: Whether to ignore parent-child pairs from URDF.

        Returns:
            Tuple of (active_i, active_j) index arrays.
        """
        # --- Start: Logic combined from _build_ignore_matrix --- #
        num_links = len(link_names)
        link_name_to_idx = {name: i for i, name in enumerate(link_names)}
        ignore_matrix = jnp.zeros((num_links, num_links), dtype=bool)
        ignore_matrix = ignore_matrix.at[
            jnp.arange(num_links), jnp.arange(num_links)
        ].set(True)
        if ignore_immediate_adjacents:
            for joint in urdf.joint_map.values():
                parent_name = joint.parent
                child_name = joint.child
                if parent_name in link_name_to_idx and child_name in link_name_to_idx:
                    parent_idx = link_name_to_idx[parent_name]
                    child_idx = link_name_to_idx[child_name]
                    ignore_matrix = ignore_matrix.at[parent_idx, child_idx].set(True)
                    ignore_matrix = ignore_matrix.at[child_idx, parent_idx].set(True)
        for name1, name2 in user_ignore_pairs:
            if name1 in link_name_to_idx and name2 in link_name_to_idx:
                idx1 = link_name_to_idx[name1]
                idx2 = link_name_to_idx[name2]
                ignore_matrix = ignore_matrix.at[idx1, idx2].set(True)
                ignore_matrix = ignore_matrix.at[idx2, idx1].set(True)

        idx_i, idx_j = jnp.tril_indices(num_links, k=-1)
        should_check = ~ignore_matrix[idx_i, idx_j]
        active_i = idx_i[should_check]
        active_j = idx_j[should_check]

        # Sort by first link index for self-collision kernel cache locality
        sort_order = jnp.argsort(active_i, stable=True)
        return active_i[sort_order], active_j[sort_order]

    @staticmethod
    def _get_trimesh_collision_geometries(
        urdf: yourdfpy.URDF, link_name: str
    ) -> trimesh.Trimesh:
        """Extracts trimesh collision geometries for a given link name, applying relative transforms."""
        if link_name not in urdf.link_map:
            return trimesh.Trimesh()

        link = urdf.link_map[link_name]
        filename_handler = urdf._filename_handler
        coll_meshes = []

        for collision in link.collisions:
            geom = collision.geometry
            mesh: Optional[trimesh.Trimesh] = None

            # Get the transform of the collision geometry relative to the link frame
            if collision.origin is not None:
                transform = collision.origin
            else:
                transform = jaxlie.SE3.identity().as_matrix()

            if geom.box is not None:
                mesh = trimesh.creation.box(extents=geom.box.size)
            elif geom.cylinder is not None:
                mesh = trimesh.creation.cylinder(
                    radius=geom.cylinder.radius, height=geom.cylinder.length
                )
            elif geom.sphere is not None:
                mesh = trimesh.creation.icosphere(radius=geom.sphere.radius)
            elif geom.mesh is not None:
                try:
                    mesh_path = geom.mesh.filename
                    loaded_obj = trimesh.load(
                        file_obj=filename_handler(mesh_path), force="mesh"
                    )

                    scale = (
                        geom.mesh.scale
                        if geom.mesh.scale is not None
                        else [1.0, 1.0, 1.0]
                    )

                    if isinstance(loaded_obj, trimesh.Trimesh):
                        mesh = loaded_obj.copy()
                        mesh.apply_scale(scale)
                    elif isinstance(loaded_obj, trimesh.Scene):
                        if len(loaded_obj.geometry) > 0:
                            geom_candidate = list(loaded_obj.geometry.values())[0]
                            if isinstance(geom_candidate, trimesh.Trimesh):
                                mesh = geom_candidate.copy()
                                mesh.apply_scale(scale)
                            else:
                                continue
                        else:
                            continue
                    else:
                        continue  # Skip if load result is unexpected

                    if mesh:
                        mesh.fix_normals()

                except Exception as e:
                    logger.error(
                        f"Failed processing mesh '{geom.mesh.filename}' for link '{link_name}': {e}"
                    )
                    continue
            else:
                logger.warning(
                    f"Unsupported collision geometry type for link '{link_name}'."
                )
                continue

            if mesh is not None:
                # Apply the transform specified in the URDF collision tag
                mesh.apply_transform(transform)
                coll_meshes.append(mesh)

        coll_mesh = sum(coll_meshes, trimesh.Trimesh())
        return coll_mesh

    @jdc.jit
    def at_config(
        self, robot: Robot, cfg: Float[Array, "*batch actuated_count"]
    ) -> CollGeom:
        """
        Returns the collision geometry transformed to the given robot configuration.

        Ensures that the link transforms returned by forward kinematics are applied
        to the corresponding collision geometries stored in this object, based on link names.

        Args:
            robot: The Robot instance containing kinematics information.
            cfg: The robot configuration (actuated joints).

        Returns:
            The collision geometry (CollGeom) transformed to the world frame
            according to the provided configuration.
        """
        # Check if the link names match - this should be true if both Robot
        # and RobotCollision were created from the same URDF parser results.
        # Only the leading entries are links; any trailing entries are attached
        # bodies, which have no counterpart in the kinematics link list.
        n_link = self.num_robot_links
        assert self.link_names[:n_link] == robot.links.names, (
            "Link name mismatch between RobotCollision and Robot kinematics."
        )

        Ts_link_world_wxyz_xyz = self._append_attachment_poses(
            robot.forward_kinematics(cfg)
        )
        Ts_link_world = jaxlie.SE3(Ts_link_world_wxyz_xyz)
        return self.coll.transform(Ts_link_world)

    def get_swept_capsules(
        self,
        robot: Robot,
        cfg_prev: Float[Array, "*batch actuated_count"],
        cfg_next: Float[Array, "*batch actuated_count"],
    ) -> Capsule:
        """
        Computes swept-volume capsules between two configurations.

        For each link, the capsule at cfg_prev and cfg_next is decomposed into
        a fixed number of spheres (currently 5). Corresponding sphere pairs are
        then connected by capsules to represent the swept volume.

        Args:
            robot: The Robot instance.
            cfg_prev: The starting robot configuration.
            cfg_next: The ending robot configuration.

        Returns:
            A Capsule object representing the swept volumes.
            The batch axes will be (*batch, 5, num_links).
        """
        n_segments = 5

        # 1. Get collision geometries at start and end configurations
        # Shape: (*batch, num_links)
        coll_prev_world: Capsule = cast(Capsule, self.at_config(robot, cfg_prev))
        coll_next_world: Capsule = cast(Capsule, self.at_config(robot, cfg_next))
        assert isinstance(coll_prev_world, Capsule)
        assert isinstance(coll_next_world, Capsule)
        assert coll_prev_world.get_batch_axes() == coll_next_world.get_batch_axes()

        # 2. Decompose capsules into spheres
        # Shape: (n_segments, *batch, num_links)
        spheres_prev = coll_prev_world.decompose_to_spheres(n_segments)
        spheres_next = coll_next_world.decompose_to_spheres(n_segments)
        assert spheres_prev.get_batch_axes() == spheres_next.get_batch_axes(), (
            "Sphere batch axes mismatch after decomposition."
        )
        expected_sphere_batch_axes = (
            (n_segments,) + cfg_prev.shape[:-1] + (self.num_links,)
        )
        assert spheres_prev.get_batch_axes() == expected_sphere_batch_axes, (
            f"Unexpected sphere batch axes: {spheres_prev.get_batch_axes()} vs {expected_sphere_batch_axes}"
        )

        # 3. Create swept capsules by connecting corresponding sphere pairs
        # Shape: (n_segments, *batch, num_links)
        swept_capsules = Capsule.from_sphere_pairs(spheres_prev, spheres_next)
        assert swept_capsules.get_batch_axes() == expected_sphere_batch_axes, (
            "Swept capsule batch axes mismatch."
        )

        # The result contains capsules for each segment of each link.
        return swept_capsules

    def compute_self_collision_distance(
        self,
        robot: Robot,
        cfg: Float[Array, "*batch actuated_count"],
    ) -> Float[Array, "*batch num_active_pairs"]:
        """
        Computes the signed distances for active self-collision pairs.

        Args:
            robot_coll: The robot's collision model with precomputed active pair indices.
            robot: The robot's kinematic model.
            cfg: The robot configuration (actuated joints).

        Returns:
            Signed distances for each active pair.
            Shape: (*batch, num_active_pairs).
            Positive distance means separation, negative means penetration.
        """

        batch_axes = cfg.shape[:-1]

        # 1. Get collision geometry at the current config
        coll = self.at_config(robot, cfg)
        assert coll.get_batch_axes() == (*batch_axes, self.num_links)

        # 2. Compute all pairwise distances using the imported function
        dist_matrix = pairwise_collide(coll, coll)
        assert dist_matrix.shape == (
            *batch_axes,
            self.num_links,
            self.num_links,
        )
        # 3. Extract distances for the precomputed active pairs
        # Use advanced indexing with the stored indices
        active_distances = dist_matrix[..., self.active_idx_i, self.active_idx_j]

        active_distances = self._mask_inactive_pairs(
            active_distances, self.active_idx_i, self.active_idx_j
        )

        # Expected shape check
        num_active_pairs = len(self.active_idx_i)
        assert active_distances.shape == (*batch_axes, num_active_pairs)

        return active_distances

    def compute_world_collision_distance(
        self,
        robot: Robot,
        cfg: Float[Array, "*batch_cfg actuated_count"],
        world_geom: CollGeom,  # Shape: (*batch_world, M, ...)
    ) -> Float[Array, "*batch_combined N M"]:
        """
        Computes the signed distances between all robot links (N) and all world obstacles (M).

        Args:
            robot_coll: The robot's collision model.
            robot: The robot's kinematic model.
            cfg: The robot configuration (actuated joints).
            world_geom: Collision geometry representing world obstacles. If representing a
                single obstacle, it should have batch shape (). If multiple, the last axis
                is interpreted as the collection of world objects (M).
                The batch dimensions (*batch_world) must be broadcast-compatible with cfg's
                batch axes (*batch_cfg).

        Returns:
            Matrix of signed distances between each robot link and each world object.
            Shape: (*batch_combined, N, M), where N=num_links, M=num_world_objects.
            Positive distance means separation, negative means penetration.
        """
        # 1. Get robot collision geometry at the current config
        # Shape: (*batch_cfg, N, ...)
        coll_robot_world = self.at_config(robot, cfg)
        N = self.num_links
        assert coll_robot_world.get_batch_axes()[-1] == N
        batch_cfg_shape = coll_robot_world.get_batch_axes()[:-1]

        # 2. Normalize world_geom shape and determine M
        world_axes = world_geom.get_batch_axes()
        if len(world_axes) == 0:  # Single world object
            # Use the object's broadcast_to method to add the M=1 axis correctly
            _world_geom = world_geom.broadcast_to((1,))
            M = 1
            batch_world_shape = ()
        else:  # Multiple world objects
            _world_geom = world_geom
            M = world_axes[-1]
            batch_world_shape = world_axes[:-1]

        # 3. Compute distances: map collide over robot links vs _world_geom (None).
        # CollGeom batch axes are always LEADING (feature dims trail), and the
        # number of trailing feature dims varies per leaf (e.g. a Sphere's
        # `radius` has none, while `pose` has 7), so a single back-relative
        # axis like -2 does NOT point at the link axis for every leaf. Since
        # batch axes are leading, the link axis is at the same *front*-relative
        # position for every leaf: use that instead.
        link_axis = len(coll_robot_world.get_batch_axes()) - 1
        _collide_links_vs_world = jax.vmap(
            collide, in_axes=(link_axis, None), out_axes=(-2)
        )
        dist_matrix = _collide_links_vs_world(coll_robot_world, _world_geom)
        dist_matrix = self._mask_inactive_world(dist_matrix)

        # 4. Result shape check
        # Calculate expected shape based on broadcasting rules
        expected_batch_combined = jnp.broadcast_shapes(
            batch_cfg_shape, batch_world_shape
        )
        expected_shape = (*expected_batch_combined, N, M)

        # Perform the assertion without try-except or complex logic
        assert dist_matrix.shape == expected_shape, (
            f"Output shape mismatch. Expected {expected_shape}, Got {dist_matrix.shape}. "
            f"Robot axes: {coll_robot_world.get_batch_axes()}, Original World axes: {world_geom.get_batch_axes()}"
        )

        # 5. Return the distance matrix
        return dist_matrix



# --------------------------------------------------------------------------- #
# Fused CUDA fast path
# --------------------------------------------------------------------------- #
# `RobotCollisionSpherized`'s distance methods route through a fused FK +
# collision CUDA kernel when one is usable, falling back to the JAX
# implementation below otherwise. The kernel computes FK inside the collision
# kernel -- link transforms stay in shared memory and sphere positions are
# formed in registers -- instead of materialising a padded [B, S, N, 3] tensor
# to global memory between two XLA ops. Measured 2.7x (self) and 2.2x (world)
# on 2048 Panda configurations, with identical verdicts.
#
# It is used only when ALL of the following hold, and silently falls back
# otherwise, so behaviour never changes on a machine that cannot support it:
#   * the kernel library is built and a GPU is present
#   * the batch is at least MIN_BATCH (below that, launch overhead dominates
#     and the JAX path is genuinely faster)
#   * for world collision, the obstacles are spheres only (capsule/box/
#     half-space are implemented in the kernel but unverified against this
#     reference, so they take the JAX path rather than return unchecked numbers)
#
# Set PYROFFI_FUSED_COLLISION=0 to disable entirely.

_FUSED_CACHE: dict = {}


def _fused_checker(robot, model):
    """Cached fused checker for a (robot, model) pair, or None if unusable."""
    import os

    if os.environ.get("PYROFFI_FUSED_COLLISION", "1") != "1":
        return None
    key = (id(robot), id(model))
    if key not in _FUSED_CACHE:
        try:
            from ._cuda_collision import FusedCUDACollisionChecker

            _FUSED_CACHE[key] = (FusedCUDACollisionChecker(robot, model)
                                 if FusedCUDACollisionChecker.available()
                                 else None)
        except Exception:
            _FUSED_CACHE[key] = None
    return _FUSED_CACHE[key]


def _fused_eligible(checker, cfg) -> bool:
    """Batched, on-device configurations large enough to amortise the launch."""
    if checker is None:
        return False
    cfg = jnp.asarray(cfg)
    return cfg.ndim == 2 and cfg.shape[0] >= checker.MIN_BATCH


@jdc.pytree_dataclass
class RobotCollisionSpherized(_AttachmentFieldsMixin):
    """Collision model for a robot, integrated with pyroffi kinematics."""

    num_links: jdc.Static[int]
    """Number of links in the model (matches kinematics links)."""
    link_names: jdc.Static[tuple[str, ...]]
    """Names of the links corresponding to link indices."""
    coll: CollGeom
    """Collision geometries for the robot (relative to their parent link frame)."""

    active_idx_i: Int[Array, " P"]
    """Row indices (first link) of active self-collision pairs to check."""
    active_idx_j: Int[Array, " P"]
    """Column indices (second link) of active self-collision pairs to check."""

    # --- Attached bodies (tools, grasped objects). --------------------------
    # Same contract as RobotCollision, but here a *slot* (not a primitive) is
    # one extra row of the (N, S) sphere array: an attachment contributes up to
    # S spheres, padded with the same negative-radius sentinel the per-link rows
    # already use.
    attach_parent_link_indices: jdc.Static[tuple[int, ...]] = ()
    """Parent link index per attachment row (static topology)."""
    attach_T_parent_body: Optional[Float[Array, "K 7"]] = None
    """``wxyz_xyz`` link<-body transform per attachment row (a pytree leaf)."""
    attach_active: Optional[Bool[Array, " K"]] = None
    """Per-row activity mask (a pytree leaf; toggling it never recompiles)."""

    @staticmethod
    def from_urdf(
        urdf: yourdfpy.URDF,
        user_ignore_pairs: tuple[tuple[str, str], ...] = (),
        srdf_path: str = None,
        ignore_immediate_adjacents: bool = True,
    ):
        """
        Build a differentiable robot collision model from a URDF.

        Args:
            urdf: The URDF object (used to load collision meshes).
            user_ignore_pairs: Additional pairs of link names to ignore for self-collision.
            ignore_immediate_adjacents: If True, automatically ignore collisions
                between adjacent (parent/child) links based on the URDF structure.
        """
        # Re-load urdf with collision data if not already loaded.
        filename_handler = urdf._filename_handler  # pylint: disable=protected-access
        try:
            has_collision = any(link.collisions for link in urdf.link_map.values())
            if not has_collision:
                urdf = yourdfpy.URDF(
                    robot=urdf.robot,
                    filename_handler=filename_handler,
                    load_collision_meshes=True,
                )
        except Exception as e:
            logger.warning(f"Could not reload URDF with collision meshes: {e}")

        _, link_info = RobotURDFParser.parse(urdf)
        link_name_list = link_info.names  # Use names from parser

        # Gather all collision meshes.
        link_sphere_meshes: list[list[trimesh.Trimesh]] = []
        for link_name in link_name_list:
            spheres = RobotCollisionSpherized._get_trimesh_collision_spheres_for_link(
                urdf, link_name
            )
            link_sphere_meshes.append(spheres)

        sphere_list_per_link: list[list[Sphere]] = []
        for sphere_meshes in link_sphere_meshes:
            per_link_spheres = [
                Sphere.from_trimesh(mesh) for mesh in sphere_meshes if mesh is not None
            ]
            sphere_list_per_link.append(per_link_spheres)

        # Pad each link's sphere list to max_spheres so we can stack into a
        # uniform [N, S] array. Padding uses a degenerate sphere (radius=-1e9)
        # which colldist_from_sdf will never activate.
        max_spheres = max(len(spheres) for spheres in sphere_list_per_link)
        dummy_sphere = Sphere.from_center_and_radius(
            center=jnp.array([0.0, 0.0, 0.0]),
            radius=jnp.array(-1e9),
        )
        padded_sphere_list: list[list[Sphere]] = []
        for per_link_spheres in sphere_list_per_link:
            n_pad = max_spheres - len(per_link_spheres)
            padded_sphere_list.append(per_link_spheres + [dummy_sphere] * n_pad)

        # Stack into a single Sphere pytree with batch_axes (N, S):
        #   Step 1: for each link, stack its S scalar Spheres → Sphere(batch_axes=(S,))
        #   Step 2: stack the N per-link Spheres          → Sphere(batch_axes=(N, S))
        per_link_batched = [
            cast(Sphere, jax.tree.map(lambda *args: jnp.stack(args, axis=0), *link_spheres))
            for link_spheres in padded_sphere_list
        ]  # list of N Spheres, each batch_axes=(S,)
        spheres_2d = cast(
            Sphere,
            jax.tree.map(lambda *args: jnp.stack(args, axis=0), *per_link_batched),
        )  # Sphere with batch_axes=(N, S)

        # Directly compute active pair indices
        # Weihang: Have not checked this part yet!!!
        # Should be fine, generates active_indices per links - Sai
        if srdf_path:
            active_idx_i, active_idx_j = (
                RobotCollisionSpherized._compute_active_pair_indices_from_srdf(
                    link_names=link_name_list,
                    srdf_path=srdf_path,
                )
            )
        else:
            active_idx_i, active_idx_j = (
                RobotCollisionSpherized._compute_active_pair_indices(
                    link_names=link_name_list,
                    urdf=urdf,
                    user_ignore_pairs=user_ignore_pairs,
                    ignore_immediate_adjacents=ignore_immediate_adjacents,
                )
            )

        logger.info(
            f"Created RobotCollision with {link_info.num_links} links "
            f"and {len(active_idx_i)} active self-collision pairs, "
            f"using spherical collision models."
        )

        return RobotCollisionSpherized(
            num_links=link_info.num_links,
            link_names=link_name_list,
            active_idx_i=active_idx_i,
            active_idx_j=active_idx_j,
            coll=spheres_2d,  # Sphere pytree with batch_axes (N, S)
        )

    @staticmethod
    def _get_trimesh_collision_spheres_for_link(
        urdf: yourdfpy.URDF, link_name: str
    ) -> list[trimesh.Trimesh]:
        if link_name not in urdf.link_map:
            return [trimesh.Trimesh()]

        link = urdf.link_map[link_name]
        filename_handler = urdf._filename_handler
        coll_meshes = []

        for collision in link.collisions:
            geom = collision.geometry
            mesh: Optional[trimesh.Trimesh] = None

            # Get the transform of the collision geometry relative to the link frame
            if collision.origin is not None:
                transform = collision.origin
            else:
                transform = jaxlie.SE3.identity().as_matrix()
            if geom.sphere is not None:
                mesh = trimesh.creation.icosphere(radius=geom.sphere.radius)
            else:
                logger.warning(
                    f"Unsupported collision geometry type for link '{link_name}'."
                )
            if mesh is not None:
                mesh.apply_transform(transform)
                coll_meshes.append(mesh)
        return coll_meshes

    @staticmethod
    def _compute_active_pair_indices_from_srdf(
        link_names: tuple[str, ...],
        srdf_path: str,
    ) -> Tuple[Int[Array, " P"], Int[Array, " P"]]:
        """
        Computes the indices (i, j) of pairs from a srdf file where i < j and the pair should
        be actively checked for self-collision.

        Args:
            link_names: Tuple of link names in order.
            srdf_path: Path to the srdf file.

        Returns:
            Tuple of (active_i, active_j) index arrays.
        """
        from .._robot_srdf_parser import read_disabled_collisions_from_srdf

        num_links = len(link_names)
        link_name_to_idx = {name: i for i, name in enumerate(link_names)}
        ignore_matrix = jnp.zeros((num_links, num_links), dtype=bool)

        # Ignore self-collisions (diagonal)
        ignore_matrix = ignore_matrix.at[
            jnp.arange(num_links), jnp.arange(num_links)
        ].set(True)

        # Read disabled collision pairs from SRDF
        try:
            disabled_pairs = read_disabled_collisions_from_srdf(srdf_path)

            disabled_count = 0
            for pair in disabled_pairs:
                link1_name = pair["link1"]
                link2_name = pair["link2"]

                # Only process if both links are in our link_names
                if link1_name in link_name_to_idx and link2_name in link_name_to_idx:
                    idx1 = link_name_to_idx[link1_name]
                    idx2 = link_name_to_idx[link2_name]

                    # Set both directions as disabled (symmetric)
                    ignore_matrix = ignore_matrix.at[idx1, idx2].set(True)
                    ignore_matrix = ignore_matrix.at[idx2, idx1].set(True)
                    disabled_count += 1

            logger.info(f"Loaded {disabled_count} disabled collision pairs from SRDF")

        except FileNotFoundError:
            logger.warning(
                f"SRDF file not found: {srdf_path}. Using all collision pairs."
            )
        except Exception as e:
            logger.warning(f"Error parsing SRDF file: {e}. Using all collision pairs.")

        # Get all lower triangular indices (i < j)
        idx_i, idx_j = jnp.tril_indices(num_links, k=-1)

        # Filter out ignored pairs
        should_check = ~ignore_matrix[idx_i, idx_j]
        active_i = idx_i[should_check]
        active_j = idx_j[should_check]

        return active_i, active_j

    @staticmethod
    def _compute_active_pair_indices(
        link_names: tuple[str, ...],
        urdf: yourdfpy.URDF,
        user_ignore_pairs: tuple[tuple[str, str], ...],
        ignore_immediate_adjacents: bool,
    ) -> Tuple[Int[Array, " P"], Int[Array, " P"]]:
        """
        Computes the indices (i, j) of pairs where i < j and the pair should
        be actively checked for self-collision.

        Args:
            link_names: Tuple of link names in order.
            urdf: Parsed URDF object.
            user_ignore_pairs: List of (name1, name2) pairs to explicitly ignore.
            ignore_immediate_adjacents: Whether to ignore parent-child pairs from URDF.

        Returns:
            Tuple of (active_i, active_j) index arrays.
        """
        # --- Start: Logic combined from _build_ignore_matrix --- #
        num_links = len(link_names)
        link_name_to_idx = {name: i for i, name in enumerate(link_names)}
        ignore_matrix = jnp.zeros((num_links, num_links), dtype=bool)
        ignore_matrix = ignore_matrix.at[
            jnp.arange(num_links), jnp.arange(num_links)
        ].set(True)
        if ignore_immediate_adjacents:
            for joint in urdf.joint_map.values():
                parent_name = joint.parent
                child_name = joint.child
                if parent_name in link_name_to_idx and child_name in link_name_to_idx:
                    parent_idx = link_name_to_idx[parent_name]
                    child_idx = link_name_to_idx[child_name]
                    ignore_matrix = ignore_matrix.at[parent_idx, child_idx].set(True)
                    ignore_matrix = ignore_matrix.at[child_idx, parent_idx].set(True)
        for name1, name2 in user_ignore_pairs:
            if name1 in link_name_to_idx and name2 in link_name_to_idx:
                idx1 = link_name_to_idx[name1]
                idx2 = link_name_to_idx[name2]
                ignore_matrix = ignore_matrix.at[idx1, idx2].set(True)
                ignore_matrix = ignore_matrix.at[idx2, idx1].set(True)

        idx_i, idx_j = jnp.tril_indices(num_links, k=-1)
        should_check = ~ignore_matrix[idx_i, idx_j]
        active_i = idx_i[should_check]
        active_j = idx_j[should_check]

        return active_i, active_j

    @staticmethod
    def _get_trimesh_collision_spheres(urdf: yourdfpy.URDF) -> list[trimesh.Trimesh]:
        """
        Extracts all spherical collision geometries from a URDF model.

        Args:
            urdf (yourdfpy.URDF): The URDF model to extract collision spheres from.

        Returns:
            list[trimesh.Trimesh]: A list of trimesh sphere meshes (each transformed to link frame).

        Author:
        Sai Coumar
        """
        sphere_meshes = []

        for link_name, link in urdf.link_map.items():
            for collision in link.collisions:
                geom = collision.geometry

                # Only handle spherical geometries
                if geom.sphere is None:
                    continue

                try:
                    # Get transform for this collision geometry
                    if collision.origin is not None:
                        transform = collision.origin
                    else:
                        transform = jaxlie.SE3.identity().as_matrix()

                    # Create the sphere mesh
                    mesh = trimesh.creation.icosphere(radius=geom.sphere.radius)

                    # Apply the transform from the URDF
                    mesh.apply_transform(transform)

                    sphere_meshes.append(mesh)

                except Exception as e:
                    logger.error(
                        f"Failed to process sphere for link '{link_name}': {e}"
                    )
                    continue

        return sphere_meshes

    @jdc.jit
    def at_config(
        self, robot: Robot, cfg: Float[Array, "*batch actuated_count"]
    ) -> CollGeom:
        """
        Returns the collision geometry transformed to the given robot configuration.

        Ensures that the link transforms returned by forward kinematics are applied
        to the corresponding collision geometries stored in this object, based on link names.

        Args:
            robot: The Robot instance containing kinematics information.
            cfg: The robot configuration (actuated joints).

        Returns:
            The collision geometry (CollGeom) transformed to the world frame
            according to the provided configuration.
        """
        # Check if the link names match - this should be true if both Robot
        # and RobotCollision were created from the same URDF parser results.
        n_link = self.num_robot_links
        assert self.link_names[:n_link] == robot.links.names, (
            "Link name mismatch between RobotCollision and Robot kinematics."
        )
        # TODO: Override with passed in result of fk so i don't have to recompute
        # Attachment rows are appended along the link (N) axis, so the vmap
        # below covers them unchanged.
        Ts_link_world_wxyz_xyz = self._append_attachment_poses(
            robot.forward_kinematics(cfg)
        )
        Ts_link_world = jaxlie.SE3(Ts_link_world_wxyz_xyz)
        # self.coll has batch_axes (N, S).  Ts_link_world has batch_axes (*batch_cfg, N).
        #
        # We cannot call self.coll.transform(Ts_link_world) directly because jaxlie
        # broadcasts from the right: (N, 7) vs (N, S, 7) would align the N-dim of
        # Ts against the S-dim of self.coll instead of the N-dim.
        #
        # Instead we vmap over N: for each link n, apply Ts_link_world[..., n]
        # (a (*batch_cfg,) SE3) to self.coll[n, :] (an (S,) Sphere).
        #
        # in_axes:
        #   Ts_link_world leaves shape (*batch_cfg, N, leaf_dim) → axis -2 = N axis
        #   self.coll      leaves shape (N, S, leaf_dim)         → axis  0 = N axis
        # out_axes=0: stack N results at axis 0 → (N, *batch_cfg, S) batch_axes
        #
        # Inside the lambda the link transform still has batch axes
        # (*batch_cfg,) while the geometry has (S,); jaxlie broadcasts from the
        # right, so those two would be aligned against each other (and error
        # outright whenever B != S).  Insert an explicit trailing sphere axis on
        # the transform so the right-aligned broadcast gives (*batch_cfg, S).
        # `transform` requires matching batch axes, so the per-link geometry is
        # broadcast up to (*batch_cfg, S) first.
        batch_cfg = cfg.shape[:-1]
        num_spheres = self.coll.get_batch_axes()[1]

        def _apply(ts: jaxlie.SE3, c: CollGeom) -> CollGeom:
            return c.broadcast_to(batch_cfg + (num_spheres,)).transform(
                jaxlie.SE3(ts.wxyz_xyz[..., None, :])
            )

        coll_n_s = jax.vmap(_apply, in_axes=(-2, 0), out_axes=0)(
            Ts_link_world, self.coll
        )
        # coll_n_s.batch_axes = (N, *batch_cfg, S): vmap stacked N at axis 0, in
        # front of the config batch axes.  Downstream code expects
        # (*batch_cfg, S, N), so N must move to the *last* batch axis — not a
        # swapaxes(0, 1), which would exchange N with batch_cfg[0] and leave the
        # S axis in the link slot for any batched cfg.
        #
        # Leaves are not uniform: `pose`/`size`/`inertia_diag` carry a trailing
        # component dim while `mass`/`friction` are batch-shaped only.  Using
        # the batch rank (static) instead of negative indices keeps both correct.
        n_batch_axes = len(coll_n_s.get_batch_axes())
        return cast(CollGeom, jax.tree.map(
            lambda x: jnp.moveaxis(x, 0, n_batch_axes - 1),
            coll_n_s,
        ))

    def compute_self_collision_distance(
        self,
        robot: Robot,
        cfg: Float[Array, "*batch actuated_count"],
    ) -> Float[Array, "*batch num_active_pairs"]:
        """
        Computes the signed distances for active self-collision pairs.

        Args:
            robot_coll: The robot's collision model with precomputed active pair indices.
            robot: The robot's kinematic model.
            cfg: The robot configuration (actuated joints).

        Returns:
            Signed distances for each active pair.
            Shape: (*batch, num_active_pairs).
            Positive distance means separation, negative means penetration.

        Author: Sai Coumar
        """

        # Fused CUDA fast path; falls back to the JAX implementation below when
        # unavailable or when the batch is too small to amortise the launch.
        _fused = _fused_checker(robot, self)
        if _fused_eligible(_fused, cfg):
            return _fused.compute_self_collision_distance(robot, cfg)

        # 1. Transform all spheres to world frame.
        coll = self.at_config(robot, cfg)  # CollGeom batch axes: (*batch, S, N)

        # 2. Signed distance between every Sᵢ×Sⱼ sphere pair of each active link
        #    pair.  A link carries up to S spheres, so the closest contact between
        #    two links can be between spheres at *different* slot indices.  We
        #    therefore take the full cross product over both sphere axes; reducing
        #    only the matched-index (diagonal) spheres — as a naive
        #    `pairwise_collide(coll, coll)` over the link axis does — silently
        #    misses cross-index overlaps and under-reports self-collision.
        centers = coll.pose.translation()  # [*batch, S, N, 3]
        radii = coll.radius                # [*batch, S, N]
        li, lj = self.active_idx_i, self.active_idx_j  # [P], [P]

        ci = centers[..., :, li, :]        # [*batch, S, P, 3]
        cj = centers[..., :, lj, :]        # [*batch, S, P, 3]
        ri = radii[..., :, li]             # [*batch, S, P]
        rj = radii[..., :, lj]             # [*batch, S, P]

        # Broadcast the two sphere axes against each other: [*batch, Si, Sj, P].
        diff = ci[..., :, None, :, :] - cj[..., None, :, :, :]
        dist = jnp.linalg.norm(diff, axis=-1) - (ri[..., :, None, :] + rj[..., None, :, :])

        # Padding spheres carry a negative-radius sentinel; mask them out of the
        # min so they neither create nor mask a contact.
        valid = (ri[..., :, None, :] >= 0) & (rj[..., None, :, :] >= 0)
        dist = jnp.where(valid, dist, jnp.inf)

        # 3. Min over both sphere axes → one signed distance per active link pair.
        active_distances = jnp.min(dist, axis=(-3, -2))  # [*batch, P]
        return self._mask_inactive_pairs(active_distances, li, lj)

    @staticmethod
    def collide_link_vs_world(link_geom, world_geom):
        """Collide one link's ``S`` spheres against ``M`` world objects → ``(M,)``.

        Strictly single-configuration: ``link_geom`` must have batch axes
        ``(S,)`` exactly. Batched configurations are handled by the caller
        (see :meth:`compute_world_collision_distance`), not here.
        """
        collide_spheres_vs_world = jax.vmap(collide, in_axes=(0, None), out_axes=0)
        dist_spheres = collide_spheres_vs_world(link_geom, world_geom)  # (S, M)
        return dist_spheres.min(axis=0)  # reduce over spheres → (M,)

    @jdc.jit
    def compute_world_collision_distance(
        self,
        robot: Robot,
        cfg: Float[Array, "*batch_cfg actuated_count"],
        world_geom: CollGeom,  # Shape: (*batch_world, M, ...)
    ) -> Float[Array, "*batch_combined N M"]:
        """
        Computes signed distances between all robot links (N) and world obstacles (M),
        accounting for multiple primitives (S) per link.

        The maximum distance over all primitives in each link is used as the link’s
        representative distance to each world object.
        """
        # 1. Get robot collision geometry at configuration.
        # Shape: (*batch_cfg, S, N, ...). A single batched FK call — batching it
        # is the whole point of accepting a batched cfg, so the fix below must
        # not undo it.
        coll_robot_world = self.at_config(robot, cfg)
        batch_cfg_shape = coll_robot_world.get_batch_axes()[:-2]
        S, N = coll_robot_world.get_batch_axes()[-2:]

        # 2. Normalize world_geom shape and determine M
        world_axes = world_geom.get_batch_axes()
        if len(world_axes) == 0:
            _world_geom = world_geom.broadcast_to((1,))
            M = 1
            batch_world_shape = ()
        else:
            _world_geom = world_geom
            M = world_axes[-1]
            batch_world_shape = world_axes[:-1]

        # 3. Collide, mapping the (S, N) -> (N, M) core over any cfg batch axes.
        #
        # The core assumes geometry whose batch axes are *exactly* (S, N), so the
        # link axis is 1 and the sphere axis is 0. Running it directly on
        # (*batch_cfg, S, N) is what used to break: the link vmap left axis 0
        # pointing at a *batch* axis rather than the sphere axis, so
        # `collide_link_vs_world` reduced over configurations instead of spheres,
        # and `collide` was additionally asked to broadcast the cfg batch against
        # the world-object axis — two unrelated dimensions. Depending on the
        # sizes that produced either a silently scrambled distance matrix or a
        # "Cannot broadcast geometry shapes" error.
        #
        # One vmap per cfg batch axis fixes the axis bookkeeping without giving
        # up the batched FK above: vmap fuses these into the same parallel
        # execution, rather than falling back to a per-config loop.
        def _core(coll_sn):
            return jax.vmap(
                self.collide_link_vs_world, in_axes=(1, None), out_axes=-2
            )(coll_sn, _world_geom)  # (N, M)

        _mapped = _core
        for _ in batch_cfg_shape:
            _mapped = jax.vmap(_mapped)

        dist_matrix = _mapped(coll_robot_world)  # (*batch_cfg, N, M)
        dist_matrix = self._mask_inactive_world(dist_matrix)

        # 6. Verify shape consistency
        expected_batch_combined = jnp.broadcast_shapes(
            batch_cfg_shape, batch_world_shape
        )
        expected_shape = (*expected_batch_combined, N, M)
        assert dist_matrix.shape == expected_shape, (
            f"Output shape mismatch. Expected {expected_shape}, got {dist_matrix.shape}. "
            f"Robot axes: {coll_robot_world.get_batch_axes()}, "
            f"World axes: {world_geom.get_batch_axes()}"
        )

        return dist_matrix

    # @jdc.jit
    # def compute_world_collision_distance(
    #     self,
    #     robot: Robot,
    #     cfg: Float[Array, "*batch_cfg actuated_count"],
    #     world_geom: CollGeom,  # Shape: (*batch_world, M, ...)
    # ) -> Float[Array, "*batch_combined N M"]:
    #     """
    #     Computes signed distances between all robot links (N) and world obstacles (M),
    #     accounting for multiple primitives (S) per link.

    #     The minimum distance over all primitives in each link is used as the link’s
    #     representative distance to each world object.
    #     """
    #     CHUNK_SIZE = 1000

    #     # 1. Get robot collision geometry at configuration
    #     # Shape: (S, *batch_cfg, N, ...)
    #     coll_robot_world = self.at_config(robot, cfg)

    #     # 2. Normalize world_geom shape and determine M
    #     world_axes = world_geom.get_batch_axes()
    #     if len(world_axes) == 0:
    #         _world_geom = world_geom.broadcast_to((1,))
    #         M = 1
    #     else:
    #         _world_geom = world_geom
    #         M = world_axes[-1]

    #     # 3. Prepare for scan over links (N)
    #     # We want to iterate over the N axis.
    #     # coll_robot_world has N at the last batch axis.
    #     # We move N to the front (0) to scan over it.
    #     n_batch = len(coll_robot_world.get_batch_axes())
    #     coll_robot_world_scannable = jax.tree.map(
    #         lambda x: jnp.moveaxis(x, -2 if x.ndim > n_batch else -1, 0),
    #         coll_robot_world,
    #     )

    #     # Pad to multiple of CHUNK_SIZE
    #     N = self.num_links
    #     pad_size = (CHUNK_SIZE - (N % CHUNK_SIZE)) % CHUNK_SIZE

    #     @jdc.jit
    #     def pad_fn(x):
    #         # x shape: (N, ...)
    #         padding = [(0, 0)] * x.ndim
    #         padding[0] = (0, pad_size)
    #         return jnp.pad(x, padding, mode='edge')

    #     coll_padded = jax.tree.map(pad_fn, coll_robot_world_scannable)

    #     # Reshape to (num_chunks, CHUNK_SIZE, ...)
    #     num_chunks = (N + pad_size) // CHUNK_SIZE

    #     @jdc.jit
    #     def reshape_fn(x):
    #         return x.reshape((num_chunks, CHUNK_SIZE) + x.shape[1:])

    #     coll_chunked = jax.tree.map(reshape_fn, coll_padded)

    #     # 4. Define scan function
    #     @jdc.jit
    #     def scan_fn(carry, link_chunk_geom):
    #         # link_chunk_geom: (CHUNK_SIZE, S, *batch_cfg, ...)
    #         # _world_geom: (*batch_world, M, ...)

    #         # vmap over the chunk (axis 0)
    #         _collide_chunk = jax.vmap(
    #             self.collide_link_vs_world, in_axes=(0, None), out_axes=0
    #         )
    #         d_chunk = _collide_chunk(link_chunk_geom, _world_geom)
    #         return carry, d_chunk

    #     # 5. Run scan
    #     # dists_scanned: (num_chunks, CHUNK_SIZE, *batch_combined, M)
    #     _, dists_scanned = lax.scan(scan_fn, None, coll_chunked)

    #     # 6. Restore shape
    #     # Flatten chunks
    #     dists_flattened = dists_scanned.reshape((-1,) + dists_scanned.shape[2:])
    #     # Slice to original N
    #     dists_N = dists_flattened[:N]
    #     # Move N to -2
    #     dist_matrix = jnp.moveaxis(dists_N, 0, -2)

    #     return dist_matrix

    def get_swept_capsules(
        self,
        robot: Robot,
        cfg_prev: Float[Array, "*batch actuated_count"],
        cfg_next: Float[Array, "*batch actuated_count"],
    ) -> Capsule:
        """
        Computes swept-volume capsules between two configurations.

        For each link, each sphere at cfg_prev is connected to the corresponding
        sphere at cfg_next by a capsule to represent the swept volume.

        Args:
            robot: The Robot instance.
            cfg_prev: The starting robot configuration.
            cfg_next: The ending robot configuration.

        Returns:
            A Capsule object representing the swept volumes.
            The batch axes will be (*batch, S, num_links) where S is the number
            of spheres per link.
        """
        # 1. Get collision geometries at start and end configurations
        # Shape: (*batch, S, num_links) where S is the number of spheres per link
        coll_prev_world: Sphere = cast(Sphere, self.at_config(robot, cfg_prev))
        coll_next_world: Sphere = cast(Sphere, self.at_config(robot, cfg_next))
        assert isinstance(coll_prev_world, Sphere)
        assert isinstance(coll_next_world, Sphere)
        assert coll_prev_world.get_batch_axes() == coll_next_world.get_batch_axes(), (
            "Sphere batch axes mismatch between configurations."
        )

        # 2. Create swept capsules by connecting corresponding sphere pairs
        # For each sphere index s and link l, connect coll_prev[..., s, l] to coll_next[..., s, l]
        swept_capsules = Capsule.from_sphere_pairs(coll_prev_world, coll_next_world)
        assert swept_capsules.get_batch_axes() == coll_prev_world.get_batch_axes(), (
            "Swept capsule batch axes mismatch."
        )

        return swept_capsules

    @staticmethod
    def mask_collision_distance(
        solution: Float[Array, "1D array = num_links"],
        exclude_link_mask: Int[Array, " num_links"],
        replace_value: float = 1e6,
    ) -> Float[Array, "1D array = num_links"]:
        """Mask collision distances at specified link indices by replacing them with a value.

        Args:
            solution: Collision distance array with shape (*batch, actuated_count)
            exclude_link_indices: Indices of links to exclude from collision checking
            replace_value: Value to replace at the excluded indices (default: 1e6 for large distance)

        Returns:
            Masked collision distance array with the same shape as solution
        """
        return solution.at[exclude_link_mask].set(replace_value)
