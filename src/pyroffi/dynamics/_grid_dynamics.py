"""CUDA-accelerated rigid body dynamics via GRiD-generated kernels.

``GRiDDynamics`` JIT-generates and compiles per-robot CUDA kernels (see
``_grid_codegen``) and exposes them through the JAX FFI, following the
collision-checker pattern (a separate class rather than ``Robot`` methods,
since compiled-library handles cannot live in a jdc pytree).

Differentiability: ``inverse_dynamics`` and ``forward_dynamics`` carry
``jax.custom_vjp`` rules whose backward passes use GRiD's *analytic gradient*
kernels (``inverse_dynamics_gradient`` / ``forward_dynamics_gradient``) —
this is the main payoff of GRiD for trajectory optimization inner loops.
For forward-mode or higher-order derivatives use the pure-JAX
``Robot.inverse_dynamics`` / ``Robot.forward_dynamics`` instead.
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
from ._dynamics_jax import _DEFAULT_GRAVITY, mass_matrix_jax
from ._grid_codegen import compile_grid_library
from ._grid_robot_adapter import build_grid_robot

_SYMBOLS = {
    "id": "GridIdFfi",
    "fd": "GridFdFfi",
    "minv": "GridMinvFfi",
    "id_grad": "GridIdGradFfi",
    "fd_grad": "GridFdGradFfi",
}

_registered_libs: dict[str, dict[str, str]] = {}


def _register_library(so_path: Path) -> dict[str, str]:
    """CDLL the compiled library and register its five FFI targets.

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

    def __init__(
        self,
        urdf: yourdfpy.URDF,
        gravity: float = _DEFAULT_GRAVITY,
        arch: str | None = None,
    ):
        self._grid_model = build_grid_robot(urdf)
        self.num_dof = self._grid_model.num_pos
        self.gravity = float(gravity)
        so_path = compile_grid_library(self._grid_model, arch=arch)
        self._targets = _register_library(so_path)

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
            gravity=onp.float32(-self.gravity),
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
            gravity=onp.float32(-self.gravity),
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
            gravity=onp.float32(-self.gravity),
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

    def inverse_dynamics(
        self,
        q: Float[Array, "*batch n"],
        qd: Float[Array, "*batch n"],
        qdd: Float[Array, "*batch n"],
    ) -> Float[Array, "*batch n"]:
        """Joint torques via GRiD RNEA. Reverse-mode differentiable
        (analytic GRiD gradients for q/qd; mass-matrix product for qdd)."""
        return _id_differentiable(self, q, qd, qdd)

    def forward_dynamics(
        self,
        q: Float[Array, "*batch n"],
        qd: Float[Array, "*batch n"],
        u: Float[Array, "*batch n"],
    ) -> Float[Array, "*batch n"]:
        """Joint accelerations via GRiD ABA-style FD. Reverse-mode
        differentiable using GRiD's analytic gradient + Minv kernels."""
        return _fd_differentiable(self, q, qd, u)

    def mass_matrix_inv(
        self, q: Float[Array, "*batch n"]
    ) -> Float[Array, "*batch n n"]:
        """Inverse joint-space mass matrix (GRiD direct Minv), symmetrized."""
        return self._minv_raw(q)

    def inverse_dynamics_gradient(
        self,
        q: Float[Array, "*batch n"],
        qd: Float[Array, "*batch n"],
        qdd: Float[Array, "*batch n"],
    ) -> Float[Array, "*batch n 2n"]:
        """Analytic [d tau/d q | d tau/d qd], shape (*batch, n, 2n)."""
        G = self._grad_raw("id_grad", q, qd, qdd)
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
        G = self._grad_raw("fd_grad", q, qd, u - self._damping * qd)
        if bool((self._damping != 0).any()):
            Minv = self._minv_raw(q)
            G = G.at[..., :, n:].add(-Minv * self._damping[None, :])
        return G


# ---------------------------------------------------------------------------
# custom_vjp rules (module-level, with the GRiDDynamics instance nondiff).
# ---------------------------------------------------------------------------


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _id_differentiable(gd: GRiDDynamics, q, qd, qdd):
    return gd._id_raw(q, qd, qdd)


def _id_fwd(gd, q, qd, qdd):
    return gd._id_raw(q, qd, qdd), (q, qd, qdd)


def _id_bwd(gd, res, g):
    q, qd, qdd = res
    n = gd.num_dof
    G = gd._grad_raw("id_grad", q, qd, qdd)  # (*b, n, 2n)
    dq = jnp.einsum("...ij,...i->...j", G[..., :n], g)
    dqd = jnp.einsum("...ij,...i->...j", G[..., n:], g) + gd._damping * g
    if gd._dyn_info is None:
        raise NotImplementedError(
            "Differentiating inverse_dynamics w.r.t. qdd requires the "
            "pure-JAX mass matrix, which is unavailable for this URDF."
        )
    # d tau / d qdd = M(q); M is symmetric so the VJP is M @ g.
    M = mass_matrix_jax(gd._dyn_info, q.astype(jnp.float32))
    dqdd = jnp.einsum("...ij,...i->...j", M, g)
    return dq, dqd, dqdd


_id_differentiable.defvjp(_id_fwd, _id_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _fd_differentiable(gd: GRiDDynamics, q, qd, u):
    return gd._fd_raw(q, qd, u)


def _fd_fwd(gd, q, qd, u):
    return gd._fd_raw(q, qd, u), (q, qd, u)


def _fd_bwd(gd, res, g):
    q, qd, u = res
    n = gd.num_dof
    # Gradient kernel is evaluated at the effective (damping-compensated)
    # torque used by _fd_raw's forward pass.
    G = gd._grad_raw("fd_grad", q, qd, u - gd._damping * qd)  # (*b, n, 2n)
    dq = jnp.einsum("...ij,...i->...j", G[..., :n], g)
    dqd = jnp.einsum("...ij,...i->...j", G[..., n:], g)
    # d qdd / d u = Minv(q); symmetric, so the VJP is Minv @ g.
    Minv = gd._minv_raw(q)
    du = jnp.einsum("...ij,...i->...j", Minv, g)
    # Chain rule through u_eff = u - d*qd.
    dqd = dqd - gd._damping * du
    return dq, dqd, du


_fd_differentiable.defvjp(_fd_fwd, _fd_bwd)
