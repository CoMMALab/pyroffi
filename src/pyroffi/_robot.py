from __future__ import annotations

from typing import Sequence

import jax_dataclasses as jdc
import jaxlie
import yourdfpy
from jax import Array
from jax import numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Float

from loguru import logger

from ._robot_urdf_parser import DynamicsInfo, JointInfo, LinkInfo, RobotURDFParser
from ._cuda_backends import CudaBackends
from . import kinematics as _kinematics


def _joint_var_cls(default_cfg):
    from cudals import Var

    class JointVar(Var[Array], default_factory=lambda: default_cfg): ...

    return JointVar


@jdc.pytree_dataclass
class Robot:
    """A differentiable robot kinematics tree."""

    joints: JointInfo
    """Joint information for the robot."""

    links: LinkInfo
    """Link information for the robot."""

    default_cfg: Float[Array, "nq"]
    """Default configuration, used to seed solvers: the actuated joints, preceded by the base
    pose (``wxyz_xyz``) for a floating-base robot."""

    dynamics: DynamicsInfo | None = None
    """Rigid body dynamics information (None when the URDF lacks inertials
    or contains unsupported joint types)."""

    _backends: jdc.Static[CudaBackends | None] = None
    """Lazy holder for URDF-dependent CUDA backends (collision, GRiD dynamics).

    Static pytree aux-data, identity-hashed — see :class:`CudaBackends`.  ``None``
    only for robots not built via :meth:`from_urdf` (e.g. manual construction)."""

    _mjcf_model: jdc.Static[object | None] = None
    """Optional compiled ``mujoco.MjModel``, set when :meth:`from_urdf` is
    given an ``mjcf_path``. Static pytree aux-data, identity-hashed. ``None``
    unless an MJCF file was supplied at construction."""

    floating_base: jdc.Static[bool] = False
    """True when the root link is free in SE(3): ``cfg = [base wxyz_xyz (7) | joints]`` and
    velocity-like vectors are ``[base twist (6) | joints]`` (see :mod:`pyroffi._floating_base`)."""

    joint_var_cls: jdc.Static[type | None] = None
    """:class:`pyroffi.Var` subclass for this robot's configuration, defaulting to
    ``default_cfg`` (PyRoKi's ``robot.joint_var_cls``)."""

    attached_names: jdc.Static[tuple[str, ...]] = ()
    """Names of the attachments already folded into ``dynamics.I_body``.

    Static aux data. Composition is additive and leaves no other trace, so
    without this a second :meth:`with_attachments` would silently double the
    carried load. See :func:`pyroffi.attachments.compose_dynamics`."""

    @property
    def nq(self) -> int:
        """Configuration dimension: ``7 + n`` for a floating base, else ``n`` actuated joints."""
        return self.joints.num_actuated_joints + (7 if self.floating_base else 0)

    @property
    def nv(self) -> int:
        """Velocity (tangent) dimension: ``6 + n`` for a floating base, else ``n``."""
        return self.joints.num_actuated_joints + (6 if self.floating_base else 0)

    @property
    def mjcf_model(self):
        """The compiled ``mujoco.MjModel`` this robot was loaded with.

        Available only when :meth:`from_urdf` was called with ``mjcf_path``.
        """
        if self._mjcf_model is None:
            raise AttributeError(
                "This Robot was not created with an mjcf_path, so no MJCF "
                "model is available. Pass mjcf_path=... to Robot.from_urdf."
            )
        return self._mjcf_model

    def geom_from_mjcf(self, body_name: str):
        """Build a physically-parameterized ``CollGeom`` from a body in this
        robot's MJCF model (mass/inertia/friction sourced from the MJCF, not
        guessed). See :func:`pyroffi.collision.geom_from_mjcf_body`.

        Requires :meth:`from_urdf` to have been called with ``mjcf_path``.
        """
        from .collision._geometry import geom_from_mjcf_body

        return geom_from_mjcf_body(self.mjcf_model, body_name)

    def with_attachments(self, aset) -> "Robot":
        """Return a copy carrying an :class:`~pyroffi.attachments.AttachmentSet`.

        The attachments' inertias are folded into ``dynamics.I_body``, so
        :meth:`inverse_dynamics` / :meth:`forward_dynamics` / :meth:`mass_matrix`
        / :meth:`step` account for the carried load with **no signature change**
        — an attachment is a fixed joint, so ``num_dof`` and every kernel shape
        are unchanged.  See :func:`pyroffi.attachments.compose_dynamics`.
        """
        from .attachments import compose_dynamics

        from . import _registry

        return _registry.register(compose_dynamics(self, aset))

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
        mjcf_path: str | None = None,
        floating_base: bool | None = None,
    ) -> Robot:
        """
        Loads a robot kinematic tree from a URDF.
        Internally tracks a topological sort of the joints.

        Args:
            urdf: The URDF to load the robot from.
            default_joint_cfg: The default joint configuration to use for optimization.
            mjcf_path: Optional path to an MJCF (MuJoCo XML) file describing
                this robot or a scene containing it — e.g. converted from the
                URDF via ``scripts/urdf_to_mjcf.py``, or a hand-authored scene
                with extra bodies (a grasped object, a fixture, ...). When
                given, the compiled model is available as ``robot.mjcf_model``
                and per-body physical parameters (mass, inertia, friction) can
                be read via ``robot.geom_from_mjcf(body_name)``. Optional,
                mirroring how an SRDF is optionally supplied to
                :meth:`pyroffi.collision.RobotCollision.from_urdf` — omit it
                if you don't need MJCF-sourced dynamics data.
            floating_base: Make the root link free in SE(3). ``None`` (default) detects it: a
                URDF whose root is connected to its first body by a ``floating`` joint (the
                ``world -> pelvis`` pattern) is floating, with that joint's origin as the default
                base pose, and the joint and its parent link are removed from the kinematic tree.
                ``True`` floats the URDF's root link as is (humanoid URDFs rooted at the pelvis).
                ``default_joint_cfg``, when given, covers the actuated joints only.
        """
        from . import _floating_base

        urdf, base_origin = _strip_floating_root(urdf)
        if floating_base is None:
            floating_base = base_origin is not None
        joints, links = RobotURDFParser.parse(urdf)

        # Compute default joint configuration.
        if default_joint_cfg is None:
            default_joint_cfg = (joints.lower_limits + joints.upper_limits) / 2
        else:
            default_joint_cfg = jnp.array(default_joint_cfg)
        assert default_joint_cfg.shape == (joints.num_actuated_joints,)
        if floating_base:
            base0 = (jaxlie.SE3.from_matrix(jnp.asarray(base_origin, jnp.float32)).wxyz_xyz
                     if base_origin is not None else _floating_base.neutral_base())
            default_joint_cfg = _floating_base.join_cfg(base0, default_joint_cfg)

        try:
            dynamics = RobotURDFParser.parse_dynamics(urdf, floating_base=floating_base)
        except NotImplementedError as e:
            logger.warning(f"Dynamics unavailable for this URDF: {e}")
            dynamics = None

        mjcf_model = None
        if mjcf_path is not None:
            import mujoco

            mjcf_model = mujoco.MjModel.from_xml_path(str(mjcf_path))

        robot = Robot(
            joints=joints,
            links=links,
            default_cfg=default_joint_cfg,
            dynamics=dynamics,
            _backends=CudaBackends(
                urdf,
                parent_indices=joints.parent_indices,
                parent_joint_indices=links.parent_joint_indices,
                num_joints=joints.num_joints,
                floating_base=bool(floating_base),
            ),
            _mjcf_model=mjcf_model,
            floating_base=bool(floating_base),
            joint_var_cls=_joint_var_cls(default_joint_cfg),
        )

        from . import _registry

        return _registry.register(robot)

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
            use_cuda: If True, dispatch to GRiM's kinematics kernel (compiled for this robot's
                URDF on first use) instead of the default JAX implementation. It reads the
                URDF's kinematics, not ``robot.joints.*``. jit-compatible (``jax.vmap`` is not
                supported by the FFI kernel — use a leading batch dimension on ``cfg``).

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
        base_region=None,
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
            base_region:       Floating base only: ``(lo, hi)`` world-frame xyz bounds the
                               base position is sampled from and kept within. Without it
                               the base is unbounded and seeds start at the warm start's base.
                               ``fixed_joint_mask`` covers the joints, not the base.
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
            base_region=base_region,
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
        f_ext: Float[Array, "*batch n_act_joints 6"] | None = None,
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
            f_ext: Optional per-body external wrenches ``[torque; force]``
                applied at each body's frame origin, in world axes.
        """
        if use_cuda:
            return (
                self._require_backends()
                .grid(gravity)
                .inverse_dynamics(q, qd, qdd, f_ext)
            )

        from . import dynamics as _dynamics

        return _dynamics.inverse_dynamics(self, q, qd, qdd, gravity, f_ext)

    @jdc.jit
    def forward_dynamics(
        self,
        q: Float[Array, "*batch n_act_joints"],
        qd: Float[Array, "*batch n_act_joints"],
        tau: Float[Array, "*batch n_act_joints"],
        gravity: jdc.Static[float] = -9.81,
        use_cuda: jdc.Static[bool] = False,
        f_ext: Float[Array, "*batch n_act_joints 6"] | None = None,
    ) -> Float[Array, "*batch n_act_joints"]:
        """Joint accelerations produced by torques ``tau`` at state ``(q, qd)``.

        Args:
            use_cuda: If True, dispatch to the per-robot GRiD CUDA backend;
                otherwise use the pure-JAX implementation.  jit-compatible (see
                ``inverse_dynamics``); ``jax.vmap`` unsupported for the CUDA path.
            f_ext: Optional per-body external wrenches (see ``inverse_dynamics``).
        """
        if use_cuda:
            return (
                self._require_backends()
                .grid(gravity)
                .forward_dynamics(q, qd, tau, f_ext)
            )

        from . import dynamics as _dynamics

        return _dynamics.forward_dynamics(self, q, qd, tau, gravity, f_ext)

    @jdc.jit
    def mass_matrix(
        self,
        q: Float[Array, "*batch n_act_joints"],
        use_cuda: jdc.Static[bool] = False,
    ) -> Float[Array, "*batch n_act_joints n_act_joints"]:
        """Joint-space mass matrix M(q) via the composite rigid body algorithm.

        Args:
            use_cuda: If True, dispatch to the GRiD CUDA backend's direct
                mass-matrix kernel (CRBA-equivalent, column-parallel);
                otherwise use the pure-JAX composite rigid body algorithm.
                jit-compatible (see ``inverse_dynamics``); ``jax.vmap``
                unsupported for the CUDA path.
        """
        if use_cuda:
            return self._require_backends().grid(-9.81).mass_matrix(q)

        from . import dynamics as _dynamics

        return _dynamics.mass_matrix(self, q)

    @jdc.jit
    def jacobian(
        self,
        q: Float[Array, "*batch n_act_joints"],
    ) -> tuple[
        Float[Array, "*batch n_body 6 n_act_joints"],
        Float[Array, "*batch n_body 3"],
    ]:
        """World-frame geometric Jacobians ``(J, r)`` for every dynamic body.

        ``J[..., i, :, :] @ qd`` is the angular-first spatial velocity
        ``[omega; v]`` of body ``i``'s frame, with the linear part measured at
        the frame origin ``r[..., i, :]`` in world axes. Bodies are indexed by
        actuated joint (``robot.dynamics.dof_names`` order).
        """
        from . import dynamics as _dynamics

        return _dynamics.jacobian(self, q)

    @jdc.jit
    def step(
        self,
        q: Float[Array, "*batch n_act_joints"],
        qd: Float[Array, "*batch n_act_joints"],
        tau: Float[Array, "*batch n_act_joints"],
        dt: jdc.Static[float],
        gravity: jdc.Static[float] = -9.81,
        use_cuda: jdc.Static[bool] = False,
        f_ext: Float[Array, "*batch n_act_joints 6"] | None = None,
        method: jdc.Static[str] = "semi_implicit",
        substeps: jdc.Static[int] = 1,
    ) -> tuple[
        Float[Array, "*batch n_act_joints"], Float[Array, "*batch n_act_joints"]
    ]:
        """Advance ``(q, qd)`` one timestep of size ``dt`` under torques ``tau``.

        Semi-implicit (symplectic) Euler by default; ``method`` may also be
        ``"euler"`` or ``"rk4"``. scan-compatible for rollouts; ``use_cuda``
        integrates over the GRiD forward-dynamics kernels.

        All fixed-step methods can silently diverge (blow up to NaN/Inf) once
        ``dt`` is large relative to the system's stiffness -- e.g.
        semi-implicit Euler under PD torques has been observed to diverge at
        ``dt`` as small as ~0.15s. ``substeps`` (default 1, identical to the
        prior behavior) subdivides ``dt`` into that many equal sub-steps and
        integrates each one in turn, trading extra ``fd``/kernel evaluations
        for a smaller effective step and much better stability.
        """
        if use_cuda:
            return (
                self._require_backends()
                .grid(gravity)
                .step(q, qd, tau, dt, f_ext, method, substeps)
            )

        from . import dynamics as _dynamics

        return _dynamics.step(self, q, qd, tau, dt, gravity, f_ext, method, substeps)

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


def _strip_floating_root(urdf: yourdfpy.URDF) -> tuple[yourdfpy.URDF, "object | None"]:
    """Remove a ``floating`` joint hanging the robot off its root link, and that root link.

    Returns the URDF of the remaining tree (rooted at the joint's child) and the joint's origin
    (4x4), or the URDF unchanged and ``None`` when there is no such joint. Floating joints
    anywhere else are left for the parser to reject.
    """
    import copy

    import numpy as onp

    root = urdf.base_link
    floating = [j for j in urdf.robot.joints if j.type == "floating" and j.parent == root]
    if not floating:
        return urdf, None
    if len(floating) > 1 or len([j for j in urdf.robot.joints if j.parent == root]) > 1:
        raise NotImplementedError(
            f"Floating base: the root link {root!r} must carry exactly one (floating) joint.")
    joint = floating[0]
    robot = copy.deepcopy(urdf.robot)
    robot.joints = [j for j in robot.joints if j.name != joint.name]
    robot.links = [link for link in robot.links if link.name != root]
    origin = onp.asarray(joint.origin) if joint.origin is not None else onp.eye(4)
    stripped = yourdfpy.URDF(robot=robot, load_meshes=False, build_scene_graph=True,
                             filename_handler=getattr(urdf, "_filename_handler", None))
    return stripped, origin
