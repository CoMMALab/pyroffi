"""Forward kinematics implementations (pure JAX and CUDA-accelerated).

These are the functional entry points backing ``Robot.forward_kinematics``;
the ``Robot`` methods delegate here so kinematics can also be used in a
free-function style (mirroring ``pyroffi.optimization_engines``).
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import jax
import jax_dataclasses as jdc
import jaxlie
from jax import Array
from jax import numpy as jnp
from jaxtyping import Float

if TYPE_CHECKING:
    from .._robot import Robot


def forward_kinematics(
    robot: Robot,
    cfg: Float[Array, "*batch actuated_count"],
    unroll_fk: jdc.Static[bool] = False,
    use_cuda: jdc.Static[bool] = False,
) -> Float[Array, "*batch link_count 7"]:
    """Run forward kinematics on the robot's links, in the provided configuration.

    Computes the world pose of each link frame. The result is ordered
    corresponding to `robot.links.names`.

    Args:
        robot: The robot model.
        cfg: The configuration of the actuated joints, in the format `(*batch actuated_count)`.
        unroll_fk: If True, unroll the JAX fori_loop over joints (ignored when use_cuda=True).
        use_cuda: If True, dispatch to an external CUDA kernel via the JAX FFI instead of
            the default JAX implementation.  Requires ``_fk_cuda.so`` to be compiled first
            (see ``build_kernels/build_fk_cuda.sh``).

    Returns:
        The SE(3) transforms of the links, ordered by `robot.links.names`,
        in the format `(*batch, link_count, wxyz_xyz)`.
    """
    batch_axes = cfg.shape[:-1]
    assert cfg.shape == (*batch_axes, robot.joints.num_actuated_joints)

    if use_cuda:
        Ts_world_joint = _fk_cuda_differentiable(cfg, robot, unroll_fk)
    else:
        Ts_world_joint = forward_kinematics_joints_jax(robot, cfg, unroll_fk)

    return link_poses_from_joint_poses(robot, Ts_world_joint)


def link_poses_from_joint_poses(
    robot: Robot, Ts_world_joint: Float[Array, "*batch actuated_count 7"]
) -> Float[Array, "*batch link_count 7"]:
    (*batch_axes, _, _) = Ts_world_joint.shape
    # Get the link poses.
    base_link_mask = robot.links.parent_joint_indices == -1
    parent_joint_indices = jnp.where(
        base_link_mask, 0, robot.links.parent_joint_indices
    )
    identity_pose = jaxlie.SE3.identity().wxyz_xyz
    Ts_world_link = jnp.where(
        base_link_mask[..., None],
        identity_pose,
        Ts_world_joint[..., parent_joint_indices, :],
    )
    assert Ts_world_link.shape == (*batch_axes, robot.links.num_links, 7)
    return Ts_world_link


def forward_kinematics_joints_jax(
    robot: Robot,
    cfg: Float[Array, "*batch actuated_count"],
    unroll_fk: jdc.Static[bool] = False,
) -> Float[Array, "*batch joint_count 7"]:
    (*batch_axes, _) = cfg.shape
    assert cfg.shape == (*batch_axes, robot.joints.num_actuated_joints)

    # Calculate full configuration using the dedicated method
    q_full = robot.joints.get_full_config(cfg)

    # Calculate delta transforms using the effective config and twists for all joints.
    tangents = robot.joints.twists * q_full[..., None]
    assert tangents.shape == (*batch_axes, robot.joints.num_joints, 6)
    delta_Ts = jaxlie.SE3.exp(tangents)  # Shape: (*batch_axes, self.joint.count, 7)

    # Combine constant parent transform with variable joint delta transform.
    Ts_parent_child = (
        jaxlie.SE3(robot.joints.parent_transforms) @ delta_Ts
    ).wxyz_xyz
    assert Ts_parent_child.shape == (*batch_axes, robot.joints.num_joints, 7)

    # Topological sort helpers
    topo_order = jnp.argsort(robot.joints._topo_sort_inv)
    Ts_parent_child_sorted = Ts_parent_child[..., robot.joints._topo_sort_inv, :]
    parent_orig_for_sorted_child = robot.joints.parent_indices[
        robot.joints._topo_sort_inv
    ]
    idx_parent_joint_sorted = jnp.where(
        parent_orig_for_sorted_child == -1,
        -1,
        topo_order[parent_orig_for_sorted_child],
    )

    # Compute link transforms relative to world, indexed by sorted *joint* index.
    def compute_transform(i: int, Ts_world_link_sorted: Array) -> Array:
        parent_sorted_idx = idx_parent_joint_sorted[i]
        T_world_parent_link = jnp.where(
            parent_sorted_idx == -1,
            jaxlie.SE3.identity().wxyz_xyz,
            Ts_world_link_sorted[..., parent_sorted_idx, :],
        )
        return Ts_world_link_sorted.at[..., i, :].set(
            (
                jaxlie.SE3(T_world_parent_link)
                @ jaxlie.SE3(Ts_parent_child_sorted[..., i, :])
            ).wxyz_xyz
        )

    Ts_world_link_init_sorted = jnp.zeros((*batch_axes, robot.joints.num_joints, 7))
    Ts_world_link_sorted = jax.lax.fori_loop(
        lower=0,
        upper=robot.joints.num_joints,
        body_fun=compute_transform,
        init_val=Ts_world_link_init_sorted,
        unroll=unroll_fk,
    )

    Ts_world_link_joint_indexed = Ts_world_link_sorted[..., topo_order, :]
    assert Ts_world_link_joint_indexed.shape == (
        *batch_axes,
        robot.joints.num_joints,
        7,
    )  # This is the link poses indexed by parent *joint* index.

    return Ts_world_link_joint_indexed


@functools.partial(jax.custom_jvp, nondiff_argnums=(2,))
def _fk_cuda_differentiable(
    cfg: Float[Array, "*batch actuated_count"],
    robot: Robot,
    unroll_fk: bool,
) -> Float[Array, "*batch joint_count 7"]:
    """Differentiable wrapper around the CUDA FK kernel.

    The FFI call is opaque to autodiff, so this ``custom_jvp`` provides:
      * primal  → the fast CUDA kernel (used on undifferentiated forward calls),
      * jvp rule → the differentiable pure-JAX FK ``forward_kinematics_joints_jax``,
                   which computes the identical ``(*batch, n_joints, 7)`` value.

    ``robot`` is an explicit (rather than closed-over) argument so the rule has
    no closed-over tracers and JAX can transpose it for reverse mode.  The FFI is
    confined to the primal function, so it is never invoked on a tangent-carrying
    input.  Both ``jax.jvp`` and ``jax.grad`` work; differentiated calls evaluate
    the JAX FK (the FFI itself is not differentiable).
    """
    from ..cuda_kernels.fk._fk_cuda import fk_cuda

    return fk_cuda(
        cfg=cfg,
        twists=robot.joints.twists,
        parent_tf=robot.joints.parent_transforms,
        parent_idx=robot.joints.parent_indices,
        act_idx=robot.joints.actuated_indices,
        mimic_mul=robot.joints.mimic_multiplier,
        mimic_off=robot.joints.mimic_offset,
        mimic_act_idx=robot.joints.mimic_act_indices,
        topo_inv=robot.joints._topo_sort_inv,
        fk_level_starts=robot.joints.fk_level_starts,
        fk_level_joints=robot.joints.fk_level_joints,
    )


@_fk_cuda_differentiable.defjvp
def _fk_cuda_differentiable_jvp(unroll_fk, primals, tangents):
    cfg, robot = primals
    dcfg, drobot = tangents
    primal_out, tangent_out = jax.jvp(
        lambda c, r: forward_kinematics_joints_jax(r, c, unroll_fk),
        (cfg, robot),
        (dcfg, drobot),
    )
    # Match the CUDA primal's dtype (float32; the JAX reference may run in x64).
    f32 = jnp.float32
    return primal_out.astype(f32), tangent_out.astype(f32)
