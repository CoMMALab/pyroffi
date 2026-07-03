"""Pure-JAX rigid body dynamics (Featherstone spatial-vector algorithms).

These mirror the CUDA GRiD kernels (see ``_grid_dynamics.py``) and serve as
the differentiable fallback, in the same spirit as the pure-JAX kinematics
kernels.  Spatial vectors are angular-first ``[omega; v]``, matching both
Featherstone's book and GRiD.

All public functions accept arbitrary leading batch dimensions on
``q``/``qd``/``qdd``/``tau``.

Gravity convention: the scalar ``gravity`` is the z-component of the
gravitational acceleration in the world frame (default ``-9.81``, i.e.
gravity pulls along -z).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
from jax import Array
from jax import numpy as jnp
from jaxtyping import Float

from .._robot_urdf_parser import DynamicsInfo

if TYPE_CHECKING:
    pass

_DEFAULT_GRAVITY = -9.81


def _topological_dof_order(dyn: DynamicsInfo) -> list[int]:
    """DOF indices ordered parent-before-child (computed on static topology)."""
    parents = dyn.parent_dof_indices
    order: list[int] = []
    remaining = set(range(dyn.num_dof))
    placed: set[int] = set()
    while remaining:
        progressed = False
        for i in sorted(remaining):
            p = parents[i]
            if p == -1 or p in placed:
                order.append(i)
                placed.add(i)
                remaining.remove(i)
                progressed = True
                break
        if not progressed:
            raise ValueError("Cyclic joint parenting in DynamicsInfo.")
    return order


def _skew(v: Array) -> Array:
    return jnp.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def _crm(v: Array) -> Array:
    """Spatial motion cross-product matrix (v x), angular-first."""
    wx = _skew(v[:3])
    vx = _skew(v[3:])
    return jnp.block([[wx, jnp.zeros((3, 3))], [vx, wx]])


def _crf(v: Array) -> Array:
    """Spatial force cross-product matrix (v x*), angular-first."""
    return -_crm(v).T


def _rotation_about_axis(axis: Array, angle: Array) -> Array:
    """Rodrigues rotation matrix for rotation by ``angle`` about ``axis``."""
    K = _skew(axis)
    s = jnp.sin(angle)
    c = jnp.cos(angle)
    return jnp.eye(3) + s * K + (1.0 - c) * (K @ K)

def _joint_motion_transform(
    axis_or_S: Array, is_prismatic: Array, q_i: Array
) -> Array:
    """Motion transform X_J from the pre-joint frame to the displaced joint
    frame for a revolute or prismatic joint (angular-first)."""
    axis = jnp.where(is_prismatic > 0.5, axis_or_S[3:], axis_or_S[:3])
    # Revolute: pure rotation by q about axis -> X = blockdiag(R^T, R^T).
    R = _rotation_about_axis(axis, q_i)
    X_rev = jnp.block(
        [[R.T, jnp.zeros((3, 3))], [jnp.zeros((3, 3)), R.T]]
    )
    # Prismatic: pure translation by q along axis -> X = [[I,0],[-skew(a q), I]].
    X_pri = jnp.block(
        [
            [jnp.eye(3), jnp.zeros((3, 3))],
            [-_skew(axis * q_i), jnp.eye(3)],
        ]
    )
    return jnp.where(is_prismatic > 0.5, X_pri, X_rev)


def _compute_Xup(dyn: DynamicsInfo, q: Array) -> list[Array]:
    """Per-joint spatial transform from parent body frame to joint frame."""
    return [
        _joint_motion_transform(dyn.S[i], dyn.joint_is_prismatic[i], q[i])
        @ dyn.X_tree[i]
        for i in range(dyn.num_dof)
    ]


def _rnea_single(
    dyn: DynamicsInfo,
    q: Array,
    qd: Array,
    qdd: Array,
    gravity: Array | float,
) -> Array:
    n = dyn.num_dof
    order = _topological_dof_order(dyn)
    Xup = _compute_Xup(dyn, q)

    # Gravity trick: base acceleration = -a_g.
    a_base = jnp.concatenate(
        [jnp.zeros(3), jnp.array([0.0, 0.0, 1.0]) * (-jnp.asarray(gravity))]
    )

    v: list[Array | None] = [None] * n
    a: list[Array | None] = [None] * n
    f: list[Array | None] = [None] * n

    for i in order:
        p = dyn.parent_dof_indices[i]
        vJ = dyn.S[i] * qd[i]
        if p == -1:
            v[i] = vJ
            a[i] = Xup[i] @ a_base + dyn.S[i] * qdd[i]
        else:
            v[i] = Xup[i] @ v[p] + vJ
            a[i] = Xup[i] @ a[p] + dyn.S[i] * qdd[i] + _crm(v[i]) @ vJ
        f[i] = dyn.I_body[i] @ a[i] + _crf(v[i]) @ (dyn.I_body[i] @ v[i])

    tau = jnp.zeros(n)
    for i in reversed(order):
        tau = tau.at[i].set(dyn.S[i] @ f[i] + dyn.damping[i] * qd[i])
        p = dyn.parent_dof_indices[i]
        if p != -1:
            f[p] = f[p] + Xup[i].T @ f[i]
    return tau


def _crba_single(dyn: DynamicsInfo, q: Array) -> Array:
    n = dyn.num_dof
    order = _topological_dof_order(dyn)
    Xup = _compute_Xup(dyn, q)

    Ic: list[Array] = [dyn.I_body[i] for i in range(n)]
    for i in reversed(order):
        p = dyn.parent_dof_indices[i]
        if p != -1:
            Ic[p] = Ic[p] + Xup[i].T @ Ic[i] @ Xup[i]

    M = jnp.zeros((n, n))
    for i in range(n):
        F = Ic[i] @ dyn.S[i]
        M = M.at[i, i].set(dyn.S[i] @ F)
        j = i
        while dyn.parent_dof_indices[j] != -1:
            F = Xup[j].T @ F
            j = dyn.parent_dof_indices[j]
            Mij = dyn.S[j] @ F
            M = M.at[i, j].set(Mij)
            M = M.at[j, i].set(Mij)
    return M


def _batched(fn, dyn, *arrays, out_extra_dims: int = 0):
    """Flatten arbitrary leading batch dims, vmap ``fn``, reshape back."""
    n = dyn.num_dof
    batch_axes = arrays[0].shape[:-1]
    for arr in arrays:
        assert arr.shape == (*batch_axes, n), (arr.shape, batch_axes, n)
    flat = [arr.reshape(-1, n) for arr in arrays]
    out = jax.vmap(lambda *xs: fn(dyn, *xs))(*flat)
    return out.reshape(*batch_axes, *out.shape[1:])


def inverse_dynamics_jax(
    dyn: DynamicsInfo,
    q: Float[Array, "*batch n_dof"],
    qd: Float[Array, "*batch n_dof"],
    qdd: Float[Array, "*batch n_dof"],
    gravity: float = _DEFAULT_GRAVITY,
) -> Float[Array, "*batch n_dof"]:
    """Joint torques from state and acceleration via RNEA (plus viscous damping)."""
    return _batched(
        lambda d, q_, qd_, qdd_: _rnea_single(d, q_, qd_, qdd_, gravity),
        dyn,
        q,
        qd,
        qdd,
    )


def mass_matrix_jax(
    dyn: DynamicsInfo,
    q: Float[Array, "*batch n_dof"],
) -> Float[Array, "*batch n_dof n_dof"]:
    """Joint-space mass matrix M(q) via the composite rigid body algorithm."""
    return _batched(_crba_single, dyn, q)


def forward_dynamics_jax(
    dyn: DynamicsInfo,
    q: Float[Array, "*batch n_dof"],
    qd: Float[Array, "*batch n_dof"],
    tau: Float[Array, "*batch n_dof"],
    gravity: float = _DEFAULT_GRAVITY,
) -> Float[Array, "*batch n_dof"]:
    """Joint accelerations from state and torques: qdd = M(q)^-1 (tau - bias).

    Matches GRiD's forward dynamics formulation
    ``qdd = Minv @ (u - RNEA(q, qd, 0))``.
    """

    def _fd(d, q_, qd_, tau_):
        bias = _rnea_single(d, q_, qd_, jnp.zeros_like(q_), gravity)
        M = _crba_single(d, q_)
        return jnp.linalg.solve(M, tau_ - bias)

    return _batched(_fd, dyn, q, qd, tau)
