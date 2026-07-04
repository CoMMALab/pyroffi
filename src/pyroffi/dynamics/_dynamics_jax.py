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
import numpy as onp
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


def _compute_X0(dyn: DynamicsInfo, Xup: list[Array]) -> list[Array]:
    """Per-body spatial motion transform from the world frame to the body frame."""
    order = _topological_dof_order(dyn)
    X0: list[Array | None] = [None] * dyn.num_dof
    for i in order:
        p = dyn.parent_dof_indices[i]
        X0[i] = Xup[i] if p == -1 else Xup[i] @ X0[p]
    return X0  # type: ignore[return-value]


def _invert_motion_transform(X: Array) -> Array:
    """Inverse of a spatial motion transform ``[[E, 0], [B, E]]`` (angular-first)."""
    E, B = X[:3, :3], X[3:, :3]
    Et = E.T
    return jnp.block([[Et, jnp.zeros((3, 3))], [-Et @ B @ Et, Et]])


def _body_origin_world(X0_i: Array) -> Array:
    """World position of a body frame origin from its world->body transform.

    For ``X = [[E, 0], [-E rhat, E]]`` the origin is ``r = unskew(-E^T B)``.
    """
    E, B = X0_i[:3, :3], X0_i[3:, :3]
    rhat = -E.T @ B
    return jnp.array([rhat[2, 1], rhat[0, 2], rhat[1, 0]])


def _fext_to_body(dyn: DynamicsInfo, X0: list[Array], f_ext: Array) -> list[Array]:
    """Rotate per-body world-axis wrenches (at body origins) into body coords.

    ``f_ext[i]`` is ``[torque; force]`` applied at body ``i``'s frame origin,
    expressed in world axes; the body frame differs only by the rotation
    ``E = X0[i][:3, :3]``, so forces map by ``blockdiag(E, E)``.
    """
    out = []
    for i in range(dyn.num_dof):
        E = X0[i][:3, :3]
        out.append(jnp.concatenate([E @ f_ext[i, :3], E @ f_ext[i, 3:]]))
    return out


def _rnea_single(
    dyn: DynamicsInfo,
    q: Array,
    qd: Array,
    qdd: Array,
    gravity: Array | float,
    f_ext: Array | None = None,
) -> Array:
    n = dyn.num_dof
    order = _topological_dof_order(dyn)
    Xup = _compute_Xup(dyn, q)
    f_ext_body = (
        None
        if f_ext is None
        else _fext_to_body(dyn, _compute_X0(dyn, Xup), f_ext)
    )

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
        if f_ext_body is not None:
            f[i] = f[i] - f_ext_body[i]

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


def _aba_single(
    dyn: DynamicsInfo,
    q: Array,
    qd: Array,
    tau: Array,
    gravity: Array | float,
    f_ext: Array | None = None,
) -> Array:
    """Joint accelerations via the O(n) Articulated Body Algorithm.

    Mirrors GRiD's ABA (Featherstone, Ch. 7), angular-first spatial vectors,
    including viscous joint damping so it matches the CRBA/RNEA formulation
    ``qdd = M^-1 (tau - RNEA(q, qd, 0))``.
    """
    n = dyn.num_dof
    order = _topological_dof_order(dyn)
    Xup = _compute_Xup(dyn, q)
    f_ext_body = (
        None
        if f_ext is None
        else _fext_to_body(dyn, _compute_X0(dyn, Xup), f_ext)
    )

    # Gravity trick: base acceleration = -a_g.
    a_base = jnp.concatenate(
        [jnp.zeros(3), jnp.array([0.0, 0.0, 1.0]) * (-jnp.asarray(gravity))]
    )

    # --- Pass 1: velocities, velocity-product accelerations, bias forces. ---
    v: list[Array | None] = [None] * n
    c: list[Array | None] = [None] * n
    IA: list[Array] = [dyn.I_body[i] for i in range(n)]
    pA: list[Array | None] = [None] * n
    for i in order:
        p = dyn.parent_dof_indices[i]
        vJ = dyn.S[i] * qd[i]
        if p == -1:
            v[i] = vJ
            c[i] = jnp.zeros(6)
        else:
            v[i] = Xup[i] @ v[p] + vJ
            c[i] = _crm(v[i]) @ vJ
        pA[i] = _crf(v[i]) @ (dyn.I_body[i] @ v[i])
        if f_ext_body is not None:
            pA[i] = pA[i] - f_ext_body[i]

    # --- Pass 2: articulated inertias and bias forces, tip to base. ---
    U: list[Array | None] = [None] * n
    D: list[Array | None] = [None] * n
    u: list[Array | None] = [None] * n
    for i in reversed(order):
        p = dyn.parent_dof_indices[i]
        U[i] = IA[i] @ dyn.S[i]
        D[i] = dyn.S[i] @ U[i]
        u[i] = tau[i] - dyn.damping[i] * qd[i] - dyn.S[i] @ pA[i]
        if p != -1:
            Ia = IA[i] - jnp.outer(U[i], U[i]) / D[i]
            pa = pA[i] + Ia @ c[i] + U[i] * (u[i] / D[i])
            IA[p] = IA[p] + Xup[i].T @ Ia @ Xup[i]
            pA[p] = pA[p] + Xup[i].T @ pa

    # --- Pass 3: accelerations, base to tip. ---
    a: list[Array | None] = [None] * n
    qdd = jnp.zeros(n)
    for i in order:
        p = dyn.parent_dof_indices[i]
        a_prime = (Xup[i] @ a_base if p == -1 else Xup[i] @ a[p]) + c[i]
        qdd_i = (u[i] - U[i] @ a_prime) / D[i]
        qdd = qdd.at[i].set(qdd_i)
        a[i] = a_prime + dyn.S[i] * qdd_i
    return qdd


def _ancestor_mask(dyn: DynamicsInfo) -> Array:
    """(n_body, n_dof) 0/1 mask: mask[i, j] = 1 iff DOF j is on the path from
    the world to body i (inclusive). Computed on static topology."""
    n = dyn.num_dof
    mask = onp.zeros((n, n))
    for i in range(n):
        j = i
        while j != -1:
            mask[i, j] = 1.0
            j = dyn.parent_dof_indices[j]
    return jnp.asarray(mask)


def _jacobian_single(dyn: DynamicsInfo, q: Array) -> tuple[Array, Array]:
    """World-frame geometric Jacobians for every body.

    Returns ``(J, r)`` where ``J`` is (n_body, 6, n_dof) angular-first with the
    linear part taken at each body's frame origin (MuJoCo/Pinocchio
    LOCAL_WORLD_ALIGNED convention), and ``r`` is (n_body, 3) body frame
    origins in world coordinates.
    """
    n = dyn.num_dof
    Xup = _compute_Xup(dyn, q)
    X0 = _compute_X0(dyn, Xup)
    # Column j of the *spatial* (world-origin) Jacobian is X0[j]^-1 @ S[j].
    cols = jnp.stack(
        [_invert_motion_transform(X0[j]) @ dyn.S[j] for j in range(n)], axis=-1
    )  # (6, n)
    r = jnp.stack([_body_origin_world(X0[i]) for i in range(n)])  # (n_body, 3)
    mask = _ancestor_mask(dyn)  # (n_body, n)
    Jang = cols[:3][None] * mask[:, None, :]  # (n_body, 3, n)
    Jlin0 = cols[3:][None] * mask[:, None, :]
    # Shift the linear part from the world origin to each body origin:
    # v(r) = v(0) + omega x r  =>  J_lin(r) = J_lin(0) - skew(r) @ J_ang.
    Jlin = Jlin0 - jax.vmap(_skew)(r) @ Jang
    return jnp.concatenate([Jang, Jlin], axis=1), r


def _batched(fn, dyn, *arrays, out_extra_dims: int = 0):
    """Flatten arbitrary leading batch dims, vmap ``fn``, reshape back.

    The first array must be (*batch, n_dof); the rest may carry extra
    trailing dims (e.g. per-body wrenches (*batch, n_body, 6)). ``None``
    entries are passed through unbatched-as-None.
    """
    n = dyn.num_dof
    batch_axes = arrays[0].shape[:-1]
    nb = len(batch_axes)
    flat = []
    for arr in arrays:
        if arr is None:
            flat.append(None)
            continue
        assert arr.shape[:nb] == batch_axes, (arr.shape, batch_axes)
        flat.append(arr.reshape(-1, *arr.shape[nb:]))
    in_axes = tuple(None if a is None else 0 for a in flat)
    out = jax.vmap(lambda *xs: fn(dyn, *xs), in_axes=in_axes)(*flat)
    return jax.tree.map(lambda o: o.reshape(*batch_axes, *o.shape[1:]), out)


def inverse_dynamics_jax(
    dyn: DynamicsInfo,
    q: Float[Array, "*batch n_dof"],
    qd: Float[Array, "*batch n_dof"],
    qdd: Float[Array, "*batch n_dof"],
    gravity: float = _DEFAULT_GRAVITY,
    f_ext: Float[Array, "*batch n_dof 6"] | None = None,
) -> Float[Array, "*batch n_dof"]:
    """Joint torques from state and acceleration via RNEA (plus viscous damping).

    ``f_ext`` are optional per-body external wrenches ``[torque; force]``
    applied at each body's frame origin, expressed in world axes.
    """
    return _batched(
        lambda d, q_, qd_, qdd_, f_: _rnea_single(d, q_, qd_, qdd_, gravity, f_),
        dyn,
        q,
        qd,
        qdd,
        f_ext,
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
    f_ext: Float[Array, "*batch n_dof 6"] | None = None,
) -> Float[Array, "*batch n_dof"]:
    """Joint accelerations from state and torques via the O(n) ABA.

    Matches GRiD's forward dynamics formulation
    ``qdd = Minv @ (u - RNEA(q, qd, 0))``, computed with the Articulated Body
    Algorithm rather than an explicit mass-matrix solve.  ``f_ext`` follows
    the same convention as :func:`inverse_dynamics_jax`.
    """
    return _batched(
        lambda d, q_, qd_, tau_, f_: _aba_single(d, q_, qd_, tau_, gravity, f_),
        dyn,
        q,
        qd,
        tau,
        f_ext,
    )


def jacobian_jax(
    dyn: DynamicsInfo,
    q: Float[Array, "*batch n_dof"],
) -> tuple[
    Float[Array, "*batch n_body 6 n_dof"], Float[Array, "*batch n_body 3"]
]:
    """World-frame geometric Jacobians (and frame origins) for every body.

    Returns ``(J, r)``: ``J[..., i, :, :]`` maps ``qd`` to the angular-first
    spatial velocity ``[omega; v]`` of body ``i``'s frame, with the linear
    part measured at the frame origin ``r[..., i, :]`` in world axes
    (LOCAL_WORLD_ALIGNED). For a point ``p`` rigidly attached to body ``i``,
    ``J_lin(p) = J[i, 3:] - skew(p - r[i]) @ J[i, :3]``.
    """
    return _batched(_jacobian_single, dyn, q)
