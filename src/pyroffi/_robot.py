from __future__ import annotations

import jax_dataclasses as jdc
import jaxlie
import jaxls
import yourdfpy
from jax import Array
from jax import numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Float

from loguru import logger

from ._robot_urdf_parser import DynamicsInfo, JointInfo, LinkInfo, RobotURDFParser
from . import kinematics as _kinematics


@jdc.pytree_dataclass
class Robot:
    """A differentiable robot kinematics tree."""

    joints: JointInfo
    """Joint information for the robot."""

    links: LinkInfo
    """Link information for the robot."""

    joint_var_cls: jdc.Static[type[jaxls.Var[Array]]]
    """Variable class for the robot configuration."""

    dynamics: DynamicsInfo | None = None
    """Rigid body dynamics information (None when the URDF lacks inertials
    or contains unsupported features such as mimic joints)."""

    @staticmethod
    def from_urdf(
        urdf: yourdfpy.URDF,
        default_joint_cfg: Float[ArrayLike, "*batch actuated_count"] | None = None,
    ) -> Robot:
        """
        Loads a robot kinematic tree from a URDF.
        Internally tracks a topological sort of the joints.

        Args:
            urdf: The URDF to load the robot from.
            default_joint_cfg: The default joint configuration to use for optimization.
        """
        joints, links = RobotURDFParser.parse(urdf)

        # Compute default joint configuration.
        if default_joint_cfg is None:
            default_joint_cfg = (joints.lower_limits + joints.upper_limits) / 2
        else:
            default_joint_cfg = jnp.array(default_joint_cfg)
        assert default_joint_cfg.shape == (joints.num_actuated_joints,)

        # Variable class for the robot configuration.
        class JointVar(  # pylint: disable=missing-class-docstring
            jaxls.Var[Array],
            default_factory=lambda: default_joint_cfg,
        ): ...

        try:
            dynamics = RobotURDFParser.parse_dynamics(urdf)
        except NotImplementedError as e:
            logger.warning(f"Dynamics unavailable for this URDF: {e}")
            dynamics = None

        robot = Robot(
            joints=joints,
            links=links,
            joint_var_cls=JointVar,
            dynamics=dynamics,
        )

        return robot

    @jdc.jit
    def forward_kinematics(
        self,
        cfg: Float[Array, "*batch actuated_count"],
        unroll_fk: jdc.Static[bool] = False,
        use_cuda: jdc.Static[bool] = False,
    ) -> Float[Array, "*batch link_count 7"]:
        """Run forward kinematics on the robot's links, in the provided configuration.

        Computes the world pose of each link frame. The result is ordered
        corresponding to `self.link.names`.

        Args:
            cfg: The configuration of the actuated joints, in the format `(*batch actuated_count)`.
            unroll_fk: If True, unroll the JAX fori_loop over joints (ignored when use_cuda=True).
            use_cuda: If True, dispatch to an external CUDA kernel via the JAX FFI instead of
                the default JAX implementation.  Requires ``_fk_cuda.so`` to be compiled first
                (see ``build_kernels/build_fk_cuda.sh``).

        Returns:
            The SE(3) transforms of the links, ordered by `self.link.names`,
            in the format `(*batch, link_count, wxyz_xyz)`.
        """
        return _kinematics.forward_kinematics(self, cfg, unroll_fk, use_cuda)

    @jdc.jit
    def inverse_kinematics(
        self,
        target_link_name: jdc.Static[str],
        target_pose: jaxlie.SE3,
        rng_key: Array | None = None,
        previous_cfg: Float[Array, "n_actuated_joints"] | None = None,
        num_seeds: jdc.Static[int] = 32,
        coarse_max_iter: jdc.Static[int] = 20,
        lm_max_iter: jdc.Static[int] = 40,
        epsilon: float = 0.02,
        nu: float = float(jnp.pi / 2),
        lambda_init: float = 5e-3,
        continuity_weight: float = 1e-3,
        fixed_joint_mask: Float[Array, "n_actuated_joints"] | None = None,
    ) -> Float[Array, "n_actuated_joints"]:
        """Solve inverse kinematics using the HJCD-IK two-phase optimizer.

        Phase 1 samples *num_seeds* configurations — the first ``top_k`` are
        warm-started near *previous_cfg* (or the joint-range midpoint when not
        provided) and the rest are random — then refines them via greedy
        coordinate descent.  Phase 2 selects the best solutions and polishes
        them with Levenberg-Marquardt.  A small *continuity_weight* penalty on
        distance from *previous_cfg* is added to the final selection criterion
        to stabilise the choice between equally valid IK solutions.

        Args:
            target_link_name:  Name of the link whose pose should match *target_pose*.
            target_pose:       Desired SE(3) world pose for that link.
            rng_key:           JAX PRNG key (defaults to PRNGKey(0) if None).
            previous_cfg:      Previous joint configuration for warm-starting and
                               continuity-aware selection.  Defaults to joint-range
                               midpoint when not provided.
            num_seeds:         Number of random seeds for the coarse phase.
            coarse_max_iter:   Coordinate-descent iteration budget.
            lm_max_iter:       Levenberg-Marquardt iteration budget.
            epsilon:           Position convergence threshold [m] (20 mm).
            nu:                Orientation convergence threshold [rad] (π/2).
            lambda_init:       Initial LM damping factor.
            continuity_weight: Weight on ‖q − previous_cfg‖² in best-solution
                               selection (default 1e-3).

        Returns:
            Best joint configuration found, shape ``(n_actuated_joints,)``.
        """
        return _kinematics.inverse_kinematics(
            self,
            target_link_name,
            target_pose,
            rng_key=rng_key,
            previous_cfg=previous_cfg,
            num_seeds=num_seeds,
            coarse_max_iter=coarse_max_iter,
            lm_max_iter=lm_max_iter,
            epsilon=epsilon,
            nu=nu,
            lambda_init=lambda_init,
            continuity_weight=continuity_weight,
            fixed_joint_mask=fixed_joint_mask,
        )

    @jdc.jit
    def inverse_dynamics(
        self,
        q: Float[Array, "*batch n_act_joints"],
        qd: Float[Array, "*batch n_act_joints"],
        qdd: Float[Array, "*batch n_act_joints"],
        gravity: jdc.Static[float] = -9.81,
    ) -> Float[Array, "*batch n_act_joints"]:
        """Joint torques realizing ``qdd`` at state ``(q, qd)`` (RNEA + viscous damping).

        Pure-JAX implementation; for large batches see
        ``pyroffi.dynamics.GRiDDynamics`` (CUDA, with analytic gradients).
        """
        from . import dynamics as _dynamics

        return _dynamics.inverse_dynamics(self, q, qd, qdd, gravity)

    @jdc.jit
    def forward_dynamics(
        self,
        q: Float[Array, "*batch n_act_joints"],
        qd: Float[Array, "*batch n_act_joints"],
        tau: Float[Array, "*batch n_act_joints"],
        gravity: jdc.Static[float] = -9.81,
    ) -> Float[Array, "*batch n_act_joints"]:
        """Joint accelerations produced by torques ``tau`` at state ``(q, qd)``."""
        from . import dynamics as _dynamics

        return _dynamics.forward_dynamics(self, q, qd, tau, gravity)

    @jdc.jit
    def mass_matrix(
        self,
        q: Float[Array, "*batch n_act_joints"],
    ) -> Float[Array, "*batch n_act_joints n_act_joints"]:
        """Joint-space mass matrix M(q) via the composite rigid body algorithm."""
        from . import dynamics as _dynamics

        return _dynamics.mass_matrix(self, q)

    def _link_poses_from_joint_poses(
        self, Ts_world_joint: Float[Array, "*batch actuated_count 7"]
    ) -> Float[Array, "*batch link_count 7"]:
        return _kinematics.link_poses_from_joint_poses(self, Ts_world_joint)

    def _forward_kinematics_joints(
        self,
        cfg: Float[Array, "*batch actuated_count"],
        unroll_fk: jdc.Static[bool] = False,
    ) -> Float[Array, "*batch joint_count 7"]:
        return _kinematics.forward_kinematics_joints_jax(self, cfg, unroll_fk)
