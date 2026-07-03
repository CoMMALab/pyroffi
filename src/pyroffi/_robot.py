from __future__ import annotations

from typing import Sequence

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
from ._cuda_backends import CudaBackends
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

    _backends: jdc.Static[CudaBackends | None] = None
    """Lazy holder for URDF-dependent CUDA backends (collision, GRiD dynamics).

    Static pytree aux-data, identity-hashed — see :class:`CudaBackends`.  ``None``
    only for robots not built via :meth:`from_urdf` (e.g. manual construction)."""

    @property
    def urdf(self):
        """The source ``yourdfpy.URDF`` this robot was parsed from.

        Available only for robots created via :meth:`from_urdf`.
        """
        if self._backends is None:
            raise AttributeError(
                "This Robot was not created via Robot.from_urdf, so its source "
                "URDF is unavailable (CUDA collision/dynamics backends require it)."
            )
        return self._backends.urdf

    def _require_backends(self) -> CudaBackends:
        if self._backends is None:
            raise RuntimeError(
                "CUDA backends are unavailable because this Robot was not created "
                "via Robot.from_urdf (they need the source URDF)."
            )
        return self._backends

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
            _backends=CudaBackends(
                urdf,
                parent_indices=joints.parent_indices,
                parent_joint_indices=links.parent_joint_indices,
                num_joints=joints.num_joints,
            ),
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
                (see ``build_kernels/build_fk_cuda.sh``).  jit-compatible: the kernel derives
                its buffers from ``robot.joints.*`` and calls the FFI, so it may be freely
                wrapped in ``jax.jit`` (``jax.vmap`` is not supported by the FFI kernel — use a
                leading batch dimension on ``cfg`` instead).

        Returns:
            The SE(3) transforms of the links, ordered by `self.link.names`,
            in the format `(*batch, link_count, wxyz_xyz)`.
        """
        return _kinematics.forward_kinematics(self, cfg, unroll_fk, use_cuda)

    def inverse_kinematics(
        self,
        target_link_name: str | Sequence[str],
        target_pose: jaxlie.SE3 | Sequence[jaxlie.SE3],
        rng_key: Array | None = None,
        previous_cfg: Float[Array, "n_actuated_joints"] | None = None,
        solver: str = "hjcd",
        num_seeds: int = 32,
        continuity_weight: float = 1e-3,
        fixed_joint_mask: Float[Array, "n_actuated_joints"] | None = None,
        constraints: Sequence = (),
        constraint_args: Sequence = (),
        constraint_weights=None,
        use_cuda: bool = False,
        **solver_kwargs,
    ) -> Float[Array, "n_actuated_joints"]:
        """Solve inverse kinematics, dispatching over solver and backend.

        Two solver families share this entry point, each with a pure-JAX and a
        CUDA (JAX-FFI) backend selected by ``use_cuda``:

        * ``solver="hjcd"`` — two-phase HJCD-IK.  Phase 1 samples *num_seeds*
          configurations (some warm-started near *previous_cfg*, the rest
          random) and refines them by greedy coordinate descent; phase 2
          polishes the best with Levenberg-Marquardt.
        * ``solver="ls"``   — seeded Levenberg-Marquardt least-squares.

        A small *continuity_weight* penalty on ‖q − previous_cfg‖² is folded
        into selection to stabilise the choice between equivalent IK solutions.

        Args:
            target_link_name:  Link name whose pose should match *target_pose*,
                               or a sequence of names for a multi-end-effector
                               (e.g. bimanual) solve.
            target_pose:       Matching single ``SE3`` or sequence of poses.
            rng_key:           JAX PRNG key (defaults to PRNGKey(0) if None).
            previous_cfg:      Previous configuration for warm-starting and
                               continuity-aware selection.  Defaults to the
                               joint-range midpoint.
            solver:            ``"hjcd"`` or ``"ls"``.
            num_seeds:         Number of random seeds for the coarse phase.
            continuity_weight: Weight on ‖q − previous_cfg‖² in selection.
            constraints:       Optional differentiable penalty callables
                               ``fn(cfg, robot, *args)`` folded into
                               selection/refinement (with *constraint_args* and
                               *constraint_weights*).
            use_cuda:          If True, offload the solver loops to the CUDA
                               kernels via the JAX FFI (requires the relevant
                               ``*_ik_cuda_lib.so``).  Fully compatible with a
                               caller's ``jax.jit``: the host-side ancestor-mask
                               precompute is derived from the robot's concrete
                               kinematic structure, so nothing traced is touched.
                               ``jax.vmap`` over this call is not supported (the
                               FFI kernels have no batching rule); batch instead
                               by adding leading dims to *previous_cfg*/targets
                               where the solver supports it.
            solver_kwargs:     Extra solver-specific options forwarded verbatim,
                               e.g. ``coarse_max_iter``/``lm_max_iter``/``epsilon``/
                               ``nu``/``lambda_init`` for ``hjcd``;
                               ``max_iter``/``pos_weight``/``ori_weight`` for
                               ``ls``; and CUDA collision options such as
                               ``collision_free``/``collision_checker``.

        Returns:
            Best joint configuration found, shape ``(n_actuated_joints,)``.
        """
        return _kinematics.inverse_kinematics(
            self,
            target_link_name,
            target_pose,
            rng_key=rng_key,
            previous_cfg=previous_cfg,
            solver=solver,
            num_seeds=num_seeds,
            continuity_weight=continuity_weight,
            fixed_joint_mask=fixed_joint_mask,
            constraints=constraints,
            constraint_args=constraint_args,
            constraint_weights=constraint_weights,
            use_cuda=use_cuda,
            **solver_kwargs,
        )

    @jdc.jit
    def inverse_dynamics(
        self,
        q: Float[Array, "*batch n_act_joints"],
        qd: Float[Array, "*batch n_act_joints"],
        qdd: Float[Array, "*batch n_act_joints"],
        gravity: jdc.Static[float] = -9.81,
        use_cuda: jdc.Static[bool] = False,
    ) -> Float[Array, "*batch n_act_joints"]:
        """Joint torques realizing ``qdd`` at state ``(q, qd)`` (RNEA + viscous damping).

        Args:
            use_cuda: If True, dispatch to the per-robot GRiD CUDA backend
                (:class:`pyroffi.dynamics.GRiDDynamics`, analytic gradients),
                built lazily on first use.  Otherwise use the pure-JAX RNEA.
                jit-compatible: the backend is built from the (concrete) URDF and
                its kernels operate purely on the traced state, so this may be
                wrapped in ``jax.jit`` (first in-trace use triggers a one-time
                kernel compile).  ``jax.vmap`` is not supported by the FFI
                kernels — batch via a leading dim on ``q``/``qd``/``qdd``.
        """
        if use_cuda:
            return self._require_backends().grid(gravity).inverse_dynamics(q, qd, qdd)

        from . import dynamics as _dynamics

        return _dynamics.inverse_dynamics(self, q, qd, qdd, gravity)

    @jdc.jit
    def forward_dynamics(
        self,
        q: Float[Array, "*batch n_act_joints"],
        qd: Float[Array, "*batch n_act_joints"],
        tau: Float[Array, "*batch n_act_joints"],
        gravity: jdc.Static[float] = -9.81,
        use_cuda: jdc.Static[bool] = False,
    ) -> Float[Array, "*batch n_act_joints"]:
        """Joint accelerations produced by torques ``tau`` at state ``(q, qd)``.

        Args:
            use_cuda: If True, dispatch to the per-robot GRiD CUDA backend;
                otherwise use the pure-JAX implementation.  jit-compatible (see
                ``inverse_dynamics``); ``jax.vmap`` unsupported for the CUDA path.
        """
        if use_cuda:
            return self._require_backends().grid(gravity).forward_dynamics(q, qd, tau)

        from . import dynamics as _dynamics

        return _dynamics.forward_dynamics(self, q, qd, tau, gravity)

    @jdc.jit
    def mass_matrix(
        self,
        q: Float[Array, "*batch n_act_joints"],
        use_cuda: jdc.Static[bool] = False,
    ) -> Float[Array, "*batch n_act_joints n_act_joints"]:
        """Joint-space mass matrix M(q) via the composite rigid body algorithm.

        Args:
            use_cuda: If True, dispatch to the GRiD CUDA backend, which computes
                M(q) as the (symmetrized) inverse of its direct ``Minv`` kernel;
                otherwise use the pure-JAX composite rigid body algorithm.
                jit-compatible (see ``inverse_dynamics``); ``jax.vmap``
                unsupported for the CUDA path.
        """
        if use_cuda:
            minv = self._require_backends().grid(-9.81).mass_matrix_inv(q)
            return jnp.linalg.inv(minv)

        from . import dynamics as _dynamics

        return _dynamics.mass_matrix(self, q)

    def collision_check(
        self,
        cfg: Float[Array, "*batch actuated_count"],
        world_geom=None,
        method: str = "sdf",
        use_cuda: bool = False,
    ):
        """Check collision for the robot in configuration(s) ``cfg``.

        This is a plain (non-jitted) dispatcher: the underlying collision
        checkers are stateful (they hold geometry/JIT caches) and manage their
        own JAX tracing internally, so this method itself is not decorated with
        ``jdc.jit``.

        jit-compatibility (CUDA path): fully supported for both self- and
        world-collision ``method="sdf"`` queries — the checker and its world
        geometry are constructed under ``jax.ensure_compile_time_eval`` so the
        one-time lazy build works even when this call is first hit inside a
        caller's ``jax.jit``.  Two caveats:

        * ``world_geom`` must be **concrete** (a fixed obstacle set), not a
          traced ``jax.jit`` argument — the world upload is host-side.
        * ``jax.vmap`` is not supported (the FFI kernels have no batching rule);
          batch by passing a leading dimension on ``cfg`` instead.

        Args:
            cfg:        Actuated-joint configuration(s), ``(*batch, actuated_count)``.
            world_geom: Optional world obstacle geometry (a ``CollGeom``).  For
                        ``method="sdf"`` it selects world- vs self-collision
                        distances; for ``method="binary"`` it is required.
            method:     ``"sdf"`` for signed-distance queries (differentiable),
                        or ``"binary"`` for a fused collision-free boolean check
                        (CUDA only).
            use_cuda:   If True, use the CUDA collision backend (built lazily on
                        first use); otherwise use the pure-JAX ``RobotCollision``
                        model.  Ignored for ``method="binary"`` (always CUDA).

        Returns:
            ``method="sdf"``:    signed distances (positive = separated,
                                 negative = penetration).  World distances of
                                 shape ``(*batch, N, M)`` when ``world_geom`` is
                                 given, else self-collision distances
                                 ``(*batch, num_active_pairs)``.
            ``method="binary"``: boolean array, ``True`` where collision-free,
                                 of shape ``cfg.shape[:-1]``.
        """
        backends = self._require_backends()

        if method == "sdf":
            checker = backends.sdf_collision() if use_cuda else backends.robot_collision()
            if world_geom is None:
                return checker.compute_self_collision_distance(self, cfg)
            return checker.compute_world_collision_distance(self, cfg, world_geom)

        if method == "binary":
            if world_geom is None:
                raise ValueError("method='binary' requires a world_geom argument.")
            return backends.binary_collision().check_collision_free(
                self, cfg, world_geom
            )

        raise ValueError(f"Unknown collision method {method!r}; expected 'sdf' or 'binary'.")

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
