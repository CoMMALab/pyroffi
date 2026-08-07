"""CUDA-accelerated rigid body dynamics via GRiD-generated kernels.

``GRiDDynamics`` JIT-generates and compiles per-robot CUDA kernels (see
``_grid_codegen``) and exposes them through the JAX FFI, following the
collision-checker pattern (a separate class rather than ``Robot`` methods,
since compiled-library handles cannot live in a jdc pytree).

Differentiability: ``inverse_dynamics`` and ``forward_dynamics`` carry
``jax.custom_jvp`` rules whose tangent maps use GRiD's *analytic gradient*
kernels (``inverse_dynamics_gradient`` / ``forward_dynamics_gradient``) —
this is the main payoff of GRiD for trajectory optimization inner loops.
Because the tangent rule is a plain linear map of the input tangents
(analytic-Jacobian @ tangent), JAX serves **both** forward-mode
(``jvp`` / ``jacfwd``) directly *and* reverse-mode (``grad`` / ``jacrev``) by
transposing that same linear rule — a single analytic rule covers both. For
second-order derivatives the analytic Jacobian kernels are treated as
constants (their own derivatives are not available); use the pure-JAX
``Robot.inverse_dynamics`` / ``Robot.forward_dynamics`` there instead.
"""

from __future__ import annotations

import ctypes
import functools
from pathlib import Path

import jax
import numpy as onp
import yourdfpy
from jax import Array
from jax import numpy as jnp
from jaxtyping import Float

from .._robot_urdf_parser import RobotURDFParser
from ._dynamics_jax import _DEFAULT_GRAVITY, jacobian_jax
from ._grid_codegen import compile_grid_library
from ._grid_robot_adapter import build_grid_robot
from ._integrators import StepMethod, step_with_fd

_SYMBOLS = {
    "id": "GridIdFfi",
    "fd": "GridFdFfi",
    "minv": "GridMinvFfi",
    "crba": "GridCrbaFfi",
    "id_grad": "GridIdGradFfi",
    "fd_grad": "GridFdGradFfi",
}

_registered_libs: dict[str, dict[str, str]] = {}


def _batchable(impl):
    """Give ``impl`` a *true* (single-launch) ``jax.vmap`` rule.

    ``impl`` must already accept arbitrary **leading** batch dimensions (all the
    GRiD kernel wrappers do, via :meth:`GRiDDynamics._flatten`). The vmap rule
    simply folds the mapped axis in as one more leading batch dimension and
    calls ``impl`` once, so a ``vmap`` becomes a single fused kernel launch
    (identically to batching over a leading dim) rather than a Python loop.
    """
    f = jax.custom_batching.custom_vmap(impl)

    @f.def_vmap
    def _rule(axis_size, in_batched, *args):
        # def_vmap moves each batched arg's mapped axis to front (axis 0);
        # broadcast any unbatched arg so every operand shares that leading axis.
        # ``impl`` then runs once over the merged leading batch — a single
        # kernel launch (not a Python loop). One vmap axis is folded per rule
        # application; for many batch axes at once prefer a single leading
        # batch dimension (reshape) over deeply-nested ``vmap``.
        args = [
            a if b else jnp.broadcast_to(a, (axis_size, *jnp.shape(a)))
            for a, b in zip(args, in_batched)
        ]
        out = impl(*args)
        return out, jax.tree_util.tree_map(lambda _: True, out)

    return f


def _register_library(so_path: Path) -> dict[str, str]:
    """CDLL the compiled library and register its FFI targets.

    Target names are namespaced by the library's cache key so multiple robots
    coexist in one process. Returns op-name -> FFI target name.
    """
    key = so_path.parent.name[:12]
    if key in _registered_libs:
        return _registered_libs[key]

    lib = ctypes.CDLL(str(so_path))
    _PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    _PyCapsule_New.restype = ctypes.py_object
    _PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

    targets = {}
    for op, symbol in _SYMBOLS.items():
        capsule = _PyCapsule_New(
            ctypes.cast(getattr(lib, symbol), ctypes.c_void_p),
            b"xla._CUSTOM_CALL_TARGET",
            None,
        )
        target = f"grid_{op}_{key}"
        jax.ffi.register_ffi_target(target, capsule, platform="CUDA")
        targets[op] = target
    _registered_libs[key] = targets
    return targets


class GRiDDynamics:
    """Per-robot GRiD dynamics kernels behind a batched JAX interface.

    All methods take/return vectors in **pyroffi actuated-joint order** (the
    same layout as ``Robot`` kinematics configurations) with arbitrary
    leading batch dimensions, in float32.
    """

    @classmethod
    def from_robot(
        cls,
        robot,
        gravity: float = _DEFAULT_GRAVITY,
        arch: str | None = None,
        runtime_inertia: bool = False,
    ) -> "GRiDDynamics":
        """Build from a pyroffi :class:`~pyroffi.Robot` (uses ``robot.urdf``).

        The unified ``from_robot`` entry point shared by the external-binding
        backends (see :mod:`pyroffi.bindings`).  Equivalent to
        ``GRiDDynamics(robot.urdf, ...)``; the robot must have been created via
        :meth:`Robot.from_urdf` so its source URDF is available.
        """
        return cls(
            robot.urdf,
            gravity=gravity,
            arch=arch,
            runtime_inertia=runtime_inertia,
        )

    def __init__(
        self,
        urdf: yourdfpy.URDF,
        gravity: float = _DEFAULT_GRAVITY,
        arch: str | None = None,
        runtime_inertia: bool = False,
    ):
        self._grid_model = build_grid_robot(urdf)
        self.num_dof = self._grid_model.num_pos
        # Sign convention: pyroffi's ``gravity`` is the signed z-component
        # (-9.81), and the A2R-Lab GRiD kernels seed the base acceleration as
        # ``-X * gravity``, i.e. they take that same signed value. (The older
        # robot-acceleration codegen omitted the negation, which is why this
        # used to be passed negated; passing the old sign now silently flips
        # every gravity torque while leaving M(q) correct.)
        self.gravity = float(gravity)
        so_path = compile_grid_library(
            self._grid_model, arch=arch, runtime_inertia=runtime_inertia
        )
        self._targets = _register_library(so_path)
        self.runtime_inertia = bool(runtime_inertia)
        self._model_state = None
        self._so_path = so_path

        # True single-launch vmap for every kernel: wrap the *raw* (non-diff)
        # kernel calls with :func:`_batchable`, which folds a vmapped axis in as
        # one more leading batch dim (see its docstring). The custom_vjp layer
        # sits on top and is vmap-transparent, descending into these on both the
        # forward and backward pass — so vmap and grad-of-vmap are single
        # launches. (Wrapping the custom_vjp function itself would close over
        # ``self`` and trip CustomVJPException.)
        self._id_raw_b = _batchable(self._id_raw)
        self._fd_raw_b = _batchable(self._fd_raw)
        self._minv_call = _batchable(self._minv_raw)
        self._crba_call = _batchable(self._crba_raw)
        self._idgrad_call = _batchable(
            lambda q, qd, qdd: self._grad_raw("id_grad", q, qd, qdd)
        )
        self._fdgrad_call = _batchable(
            lambda q, qd, u: self._grad_raw("fd_grad", q, qd, u)
        )

        # q_grid[g] = sign[g] * q_act[perm[g]]  (see GridRobotModel).
        perm = onp.asarray(self._grid_model.joint_perm)
        signs = onp.asarray(self._grid_model.axis_signs, dtype=onp.float32)
        inv_perm = onp.empty_like(perm)
        inv_perm[perm] = onp.arange(len(perm))
        self._perm = jnp.asarray(perm)
        self._inv_perm = jnp.asarray(inv_perm)
        self._signs = jnp.asarray(signs)
        self._signs_act = jnp.asarray(signs[inv_perm])

        # Pure-JAX dynamics info for the custom_vjp fallback pieces
        # (mass-matrix product in the ID backward pass).
        try:
            self._dyn_info = RobotURDFParser.parse_dynamics(urdf)
        except NotImplementedError:
            self._dyn_info = None

        # GRiD's generated kernels ignore <dynamics damping>; composite the
        # viscous damping terms around the kernels here so results match the
        # pure-JAX implementation: tau += d*qd, and FD runs on u - d*qd.
        self._damping = jnp.asarray(
            [
                float(j.dynamics.damping)
                if j.dynamics is not None and j.dynamics.damping is not None
                else 0.0
                for j in urdf.actuated_joints
            ],
            dtype=jnp.float32,
        )

    # ------------------------------------------------------------------
    # Joint-order / sign mapping helpers.
    # ------------------------------------------------------------------

    def _to_grid(self, x: Array) -> Array:
        return x[..., self._perm] * self._signs

    def _from_grid(self, y: Array) -> Array:
        return y[..., self._inv_perm] * self._signs_act

    def _mat_from_grid(self, M: Array) -> Array:
        """Un-permute both axes of a (..., n, n) grid-ordered matrix."""
        M = M[..., self._inv_perm, :][..., :, self._inv_perm]
        return M * (self._signs_act[:, None] * self._signs_act[None, :])

    def _flatten(self, *arrays: Array) -> tuple[tuple[int, ...], list[Array]]:
        n = self.num_dof
        batch_axes = arrays[0].shape[:-1]
        for a in arrays:
            assert a.shape == (*batch_axes, n), (a.shape, batch_axes, n)
        batch = int(onp.prod(batch_axes)) if batch_axes else 1
        return batch_axes, [
            self._to_grid(a).reshape(batch, n).astype(jnp.float32) for a in arrays
        ]

    def _call(self, op: str, out_shape: tuple[int, ...], *operands, **attrs):
        # The raw FFI primitive is not vmap-aware; ``jax.vmap`` support is
        # provided a level up by wrapping the public methods with
        # :func:`_batchable` (a true single-launch batching rule). Prefer a
        # leading batch dimension for the largest batches.
        return jax.ffi.ffi_call(
            self._targets[op],
            jax.ShapeDtypeStruct(out_shape, jnp.float32),
        )(*operands, **attrs)

    # ------------------------------------------------------------------
    # Raw (non-differentiable) kernel wrappers.
    # ------------------------------------------------------------------

    def _id_raw(self, q: Array, qd: Array, qdd: Array) -> Array:
        batch_axes, (qf, qdf, qddf) = self._flatten(q, qd, qdd)
        c = self._call(
            "id",
            (qf.shape[0], self.num_dof),
            qf,
            qdf,
            qddf,
            gravity=onp.float32(self.gravity),
        )
        c = self._from_grid(c).reshape(*batch_axes, self.num_dof)
        return c + self._damping * qd

    def _fd_raw(self, q: Array, qd: Array, u: Array) -> Array:
        u = u - self._damping * qd  # GRiD kernels have no damping term.
        batch_axes, (qf, qdf, uf) = self._flatten(q, qd, u)
        qdd = self._call(
            "fd",
            (qf.shape[0], self.num_dof),
            qf,
            qdf,
            uf,
            gravity=onp.float32(self.gravity),
        )
        return self._from_grid(qdd).reshape(*batch_axes, self.num_dof)

    def _minv_raw(self, q: Array) -> Array:
        n = self.num_dof
        batch_axes, (qf,) = self._flatten(q)
        buf = self._call("minv", (qf.shape[0], n, n), qf)
        # Per-timestep layout is column-major n x n with only the upper
        # triangle (row <= col) filled: buf[b, col, row]. Symmetrize, then
        # read as (row, col).
        upper_cr = jnp.tril(buf)  # [b, col, row], row <= col
        M = upper_cr + jnp.triu(jnp.swapaxes(upper_cr, -1, -2), k=1)
        M = self._mat_from_grid(jnp.swapaxes(M, -1, -2))
        return M.reshape(*batch_axes, n, n)

    def _grad_raw(self, op: str, a: Array, b: Array, c: Array) -> Array:
        """Shared wrapper for the analytic gradient kernels.

        Returns (*batch, n, 2n): rows = output joint, cols = [d/dq | d/dqd],
        in pyroffi actuated order.
        """
        n = self.num_dof
        batch_axes, (af, bf, cf) = self._flatten(a, b, c)
        buf = self._call(
            op,
            (af.shape[0], 2 * n, n),
            af,
            bf,
            cf,
            gravity=onp.float32(self.gravity),
        )
        # buf[b, col, row] with col in [0, 2n): column-major n x 2n blocks.
        G = jnp.swapaxes(buf, -1, -2)  # (B, n, 2n): [row, col]
        G = G[..., self._inv_perm, :]
        Gq = self._mat_col_from_grid(G[..., :n])
        Gqd = self._mat_col_from_grid(G[..., n:])
        sign_rows = self._signs_act[:, None]
        return (
            jnp.concatenate([Gq, Gqd], axis=-1) * sign_rows
        ).reshape(*batch_axes, n, 2 * n)

    def _mat_col_from_grid(self, G: Array) -> Array:
        """Un-permute/sign the column (input-joint) axis of (..., n, n)."""
        return G[..., :, self._inv_perm] * self._signs_act[None, :]

    # ------------------------------------------------------------------
    # Public API.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Runtime-mutable inertia (payload / tool use). See
    # :mod:`._grid_runtime_inertia` for the parameter basis and the purity
    # constraint this path carries.
    # ------------------------------------------------------------------

    @property
    def model_state(self):
        """Guarded handle on the device-resident inertia table.

        Only available when the library was built with ``runtime_inertia=True``.
        """
        if not self.runtime_inertia:
            raise AttributeError(
                "This GRiDDynamics was built without runtime_inertia, so its "
                "inertia is baked into the compiled kernels. Construct it with "
                "GRiDDynamics(urdf, runtime_inertia=True) to get a mutable "
                "inertia table."
            )
        if self._model_state is None:
            from ._grid_runtime_inertia import GridModelState

            self._model_state = GridModelState(
                self._so_path,
                num_bodies=self._grid_model.robot.get_num_joints(),
                baseline=onp.asarray(
                    self._grid_model.robot.get_inertia_params_ordered_by_id()[1:],
                    dtype=onp.float64,
                ),
            )
        return self._model_state

    def set_attachments(self, robot, aset) -> None:
        """Upload the payload implied by an ``AttachmentSet`` to the GPU.

        Composes each attachment's spatial inertia into its DOF body (the same
        ``Xᵀ I X`` congruence :func:`pyroffi.attachments.compose_dynamics` uses),
        converts to the 10-parameter regressor basis and adds it to the URDF's
        own parameters.  One blocking ``10·NB``-float memcpy; no recompile.

        Call this only at grasp-topology boundaries — it writes device-resident
        model state and will refuse to run under a tracer.  ``aset`` may not be
        batched: there is one table per model, so payload sweeps belong on the
        pure-JAX path (``robot.with_attachments(...)``).
        """
        from ..attachments import compose_dynamics
        from ._grid_runtime_inertia import (
            GridModelState,
            inertia_params_from_spatial,
        )

        # Guard at the entry point, before any numpy conversion, so a traced
        # payload reports the real constraint rather than an incidental
        # TracerArrayConversionError from deep inside the composition.
        GridModelState.reject_tracers(robot.dynamics, aset)
        base_I = onp.asarray(robot.dynamics.I_body, dtype=onp.float64)
        loaded_I = onp.asarray(
            compose_dynamics(robot, aset).dynamics.I_body, dtype=onp.float64
        )
        dI = loaded_I - base_I
        # pyroffi DOF order -> GRiD table rows: row k holds GRiD joint k, whose
        # pyroffi actuated index is joint_perm[k].
        perm = onp.asarray(self._grid_model.joint_perm)
        deltas = {
            k: inertia_params_from_spatial(dI[perm[k]])
            for k in range(len(perm))
            if onp.any(dI[perm[k]] != 0.0)
        }
        self.model_state.add_body_inertia(deltas)

    def reset_inertia(self) -> None:
        """Restore the URDF's own inertial parameters (drop any payload)."""
        self.model_state.reset()

    def _require_dyn_info(self, what: str):
        if self._dyn_info is None:
            raise NotImplementedError(
                f"{what} requires the pure-JAX dynamics info, which is "
                "unavailable for this URDF (see RobotURDFParser.parse_dynamics)."
            )
        return self._dyn_info

    def _tau_ext(self, q: Array, f_ext: Array) -> Array:
        """Generalized joint torques from per-body world-axis wrenches.

        ``tau_ext = sum_i J_i^T f_ext_i`` with the pure-JAX frame Jacobians
        (external forces enter the dynamics linearly as generalized forces,
        so the GRiD kernels themselves stay untouched). Differentiable in
        both ``q`` and ``f_ext``.
        """
        dyn = self._require_dyn_info("f_ext support")
        J, _ = jacobian_jax(dyn, q.astype(jnp.float32))  # (*b, n_body, 6, n)
        return jnp.einsum("...ijk,...ij->...k", J, f_ext.astype(jnp.float32))

    def inverse_dynamics(
        self,
        q: Float[Array, "*batch n"],
        qd: Float[Array, "*batch n"],
        qdd: Float[Array, "*batch n"],
        f_ext: Float[Array, "*batch n 6"] | None = None,
    ) -> Float[Array, "*batch n"]:
        """Joint torques via GRiD RNEA. Reverse-mode differentiable
        (analytic GRiD gradients for q/qd; mass-matrix product for qdd).

        ``f_ext`` are optional per-body wrenches ``[torque; force]`` at each
        body's frame origin in world axes (pyroffi actuated-joint body order);
        they are composited around the kernel as generalized forces.
        """
        tau = _id_differentiable(self, q, qd, qdd)
        if f_ext is not None:
            tau = tau - self._tau_ext(q, f_ext)
        return tau

    def forward_dynamics(
        self,
        q: Float[Array, "*batch n"],
        qd: Float[Array, "*batch n"],
        u: Float[Array, "*batch n"],
        f_ext: Float[Array, "*batch n 6"] | None = None,
    ) -> Float[Array, "*batch n"]:
        """Joint accelerations via GRiD ABA-style FD. Reverse-mode
        differentiable using GRiD's analytic gradient + Minv kernels.

        ``f_ext`` follows the same convention as :meth:`inverse_dynamics`.
        """
        if f_ext is not None:
            u = u + self._tau_ext(q, f_ext)
        return _fd_differentiable(self, q, qd, u)

    def mass_matrix_inv(
        self, q: Float[Array, "*batch n"]
    ) -> Float[Array, "*batch n n"]:
        """Inverse joint-space mass matrix (GRiD direct Minv), symmetrized."""
        return self._minv_call(q)

    def mass_matrix(
        self, q: Float[Array, "*batch n"]
    ) -> Float[Array, "*batch n n"]:
        """Joint-space mass matrix M(q) on the GPU.

        Computed by the custom ``CrbaKernel``: XImats are loaded once per
        timestep, then each column is ``ID(q, 0, e_j, g=0)`` via GRiD's
        thread-parallel inverse-dynamics inner routine (exactly CRBA's M).
        """
        return self._crba_call(q)

    def _crba_raw(self, q: Array) -> Array:
        n = self.num_dof
        batch_axes, (qf,) = self._flatten(q)
        buf = self._call("crba", (qf.shape[0], n, n), qf)  # [b, col, row]
        M = self._mat_from_grid(jnp.swapaxes(buf, -1, -2))
        return M.reshape(*batch_axes, n, n)

    def jacobian(
        self, q: Float[Array, "*batch n"]
    ) -> tuple[Float[Array, "*batch n 6 n"], Float[Array, "*batch n 3"]]:
        """World-frame geometric Jacobians ``(J, r)`` for every body.

        GRiD emits no Jacobian kernel; this delegates to the
        pure-JAX :func:`jacobian_jax` in float32 (same convention: angular
        first, linear part at each body frame origin, world axes).
        """
        dyn = self._require_dyn_info("jacobian")
        return jacobian_jax(dyn, q.astype(jnp.float32))

    def step(
        self,
        q: Float[Array, "*batch n"],
        qd: Float[Array, "*batch n"],
        u: Float[Array, "*batch n"],
        dt: float,
        f_ext: Float[Array, "*batch n 6"] | None = None,
        method: StepMethod = "semi_implicit",
        substeps: int = 1,
    ) -> tuple[Float[Array, "*batch n"], Float[Array, "*batch n"]]:
        """Advance ``(q, qd)`` one timestep using the GRiD forward dynamics.

        Same integrators as :func:`pyroffi.dynamics.step` (semi-implicit
        Euler default, ``"euler"``, ``"rk4"``); differentiable through the
        GRiD custom_vjp rules. See :func:`pyroffi.dynamics.step_with_fd` for
        the ``substeps`` divergence caveat and stabilization tradeoff.
        """
        return step_with_fd(
            lambda q_, qd_: self.forward_dynamics(q_, qd_, u, f_ext),
            q,
            qd,
            dt,
            method,
            substeps,
        )

    def inverse_dynamics_gradient(
        self,
        q: Float[Array, "*batch n"],
        qd: Float[Array, "*batch n"],
        qdd: Float[Array, "*batch n"],
    ) -> Float[Array, "*batch n 2n"]:
        """Analytic [d tau/d q | d tau/d qd], shape (*batch, n, 2n)."""
        G = self._idgrad_call(q, qd, qdd)
        # Damping is composited outside the GRiD kernels: d tau/d qd += diag(d).
        return G.at[..., :, self.num_dof :].add(jnp.diag(self._damping))

    def forward_dynamics_gradient(
        self,
        q: Float[Array, "*batch n"],
        qd: Float[Array, "*batch n"],
        u: Float[Array, "*batch n"],
    ) -> Float[Array, "*batch n 2n"]:
        """Analytic [d qdd/d q | d qdd/d qd], shape (*batch, n, 2n)."""
        n = self.num_dof
        # The kernel differentiates GRiD's FD at the effective (damping-
        # compensated) torque; add the -Minv @ diag(d) chain-rule term for qd.
        G = self._fdgrad_call(q, qd, u - self._damping * qd)
        if bool((self._damping != 0).any()):
            Minv = self._minv_call(q)
            G = G.at[..., :, n:].add(-Minv * self._damping[None, :])
        return G


# ---------------------------------------------------------------------------
# custom_jvp rules (module-level, with the GRiDDynamics instance nondiff).
#
# The tangent rule pushes the input tangents through GRiD's analytic Jacobian
# (a plain linear map, ``J @ tangent``, with ``J`` computed by the gradient
# kernels at the primal point). JAX gets forward-mode from this rule directly
# and reverse-mode by transposing it, so both modes share one analytic rule and
# a single fused kernel launch under ``vmap`` (the kernels are ``_batchable``).
# ``symbolic_zeros=True`` lets us skip the kernel work (and the mass-matrix
# solve) for any input whose tangent is a structural zero.
# ---------------------------------------------------------------------------

_SymZero = jax.custom_derivatives.SymbolicZero


def _nz(t) -> bool:
    """True iff tangent ``t`` is not a structural (symbolic) zero."""
    return not isinstance(t, _SymZero)


@functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
def _id_differentiable(gd: GRiDDynamics, q, qd, qdd):
    return gd._id_raw_b(q, qd, qdd)


def _id_jvp(gd, primals, tangents):
    q, qd, qdd = primals
    t_q, t_qd, t_qdd = tangents
    out = gd._id_raw_b(q, qd, qdd)
    n = gd.num_dof
    tan = jnp.zeros_like(out)
    if _nz(t_q) or _nz(t_qd):
        G = gd._idgrad_call(q, qd, qdd)  # (*b, n, 2n) = [d tau/d q | d tau/d qd]
        if _nz(t_q):
            tan = tan + jnp.einsum("...ij,...j->...i", G[..., :n], t_q)
        if _nz(t_qd):
            tan = (
                tan
                + jnp.einsum("...ij,...j->...i", G[..., n:], t_qd)
                + gd._damping * t_qd
            )
    if _nz(t_qdd):
        # d tau / d qdd = M(q), via the GPU CRBA kernel rather than the pure-JAX
        # one. This tangent is live on *every* gradient evaluation in the contact
        # solvers (qdd is finite-differenced from the decision variables), so it
        # is squarely on the hot path. Using the kernel also drops the old
        # dependency on ``_dyn_info``, so URDFs that carry no pure-JAX dynamics
        # tables are now differentiable w.r.t. qdd as well.
        M = gd._crba_call(q.astype(jnp.float32))
        tan = tan + jnp.einsum("...ij,...j->...i", M, t_qdd)
    # The GRiD kernels are float32 end to end, but the incoming tangents follow
    # the ambient x64 setting; a custom_jvp must return a tangent whose dtype
    # matches the primal, so pin it here.
    return out, tan.astype(out.dtype)


_id_differentiable.defjvp(_id_jvp, symbolic_zeros=True)


@functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
def _fd_differentiable(gd: GRiDDynamics, q, qd, u):
    return gd._fd_raw_b(q, qd, u)


def _fd_jvp(gd, primals, tangents):
    q, qd, u = primals
    t_q, t_qd, t_u = tangents
    out = gd._fd_raw_b(q, qd, u)
    n = gd.num_dof
    tan = jnp.zeros_like(out)
    if _nz(t_q) or _nz(t_qd):
        # Gradient kernel at the effective (damping-compensated) torque, matching
        # _fd_raw's forward pass. G = [d qdd/d q | d qdd/d qd] holding u_eff fixed.
        G = gd._fdgrad_call(q, qd, u - gd._damping * qd)
        if _nz(t_q):
            tan = tan + jnp.einsum("...ij,...j->...i", G[..., :n], t_q)
        if _nz(t_qd):
            tan = tan + jnp.einsum("...ij,...j->...i", G[..., n:], t_qd)
    # d qdd / d u_eff = Minv(q), with u_eff = u - d*qd, so both t_u and t_qd feed
    # it: d qdd += Minv @ (t_u - d*t_qd).
    if _nz(t_u) or _nz(t_qd):
        rhs = jnp.zeros_like(out)
        if _nz(t_u):
            rhs = rhs + t_u
        if _nz(t_qd):
            rhs = rhs - gd._damping * t_qd
        Minv = gd._minv_call(q)
        tan = tan + jnp.einsum("...ij,...j->...i", Minv, rhs)
    return out, tan.astype(out.dtype)


_fd_differentiable.defjvp(_fd_jvp, symbolic_zeros=True)
