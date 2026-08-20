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

# Dense n x n mask GEMMs (and the einsum contractions below) run at JAX's
# default matmul precision, which is TF32 on Ampere (A5000) -- frax's dense
# dynamics are TF32-limited as a result (measured FD rel_err ~0.55). We pin
# the whole dense path to float32 "highest" precision so pyroffi_jax stays at
# its usual ~4e-5 and does not regress under the rewrite.
_HIGHEST = jax.lax.Precision.HIGHEST


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


def _body_origins_world(X0: Array) -> Array:
    """World positions of all body frame origins, batched over (n, 6, 6) X0.

    Vectorized version of :func:`_body_origin_world` for the stacked
    world->body transforms produced by :func:`_spatial_data`.
    """
    E = X0[:, :3, :3]
    B = X0[:, 3:, :3]
    rhat = -jnp.matmul(jnp.swapaxes(E, -1, -2), B)  # (n, 3, 3) skew of r
    return jnp.stack([rhat[:, 2, 1], rhat[:, 0, 2], rhat[:, 1, 0]], axis=-1)


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


def _skew_batched(v: Array) -> Array:
    """Batched 3x3 cross-product matrix: ``v`` (..., 3) -> (..., 3, 3).

    Built from explicit element rows (no per-vector ``jnp.block``), so it is
    jit- and vmap-friendly and emits a single batched scatter.
    """
    v0, v1, v2 = v[..., 0], v[..., 1], v[..., 2]
    z = jnp.zeros_like(v0)
    return jnp.stack(
        [
            jnp.stack([z, -v2, v1], axis=-1),
            jnp.stack([v2, z, -v0], axis=-1),
            jnp.stack([-v1, v0, z], axis=-1),
        ],
        axis=-2,
    )


def _motion_cross(X: Array, Y: Array) -> Array:
    """Batched spatial motion cross product ``X x Y`` (angular-first).

    ``X``, ``Y`` are (..., 6) ``[omega; v]``; the result is
    ``[omega_X x omega_Y; v_X x omega_Y + omega_X x v_Y]``.
    """
    return jnp.concatenate(
        [
            jnp.cross(X[..., :3], Y[..., :3]),
            jnp.cross(X[..., 3:], Y[..., :3]) + jnp.cross(X[..., :3], Y[..., 3:]),
        ],
        axis=-1,
    )


def _force_cross(X: Array, Y: Array) -> Array:
    """Batched spatial force cross product ``X x* Y`` (angular-first).

    ``X`` is (..., 6) motion ``[omega; v]`` and ``Y`` is (..., 6) force
    ``[torque; force]``; the result is
    ``[omega x torque + v x force; omega x force]``.
    """
    return jnp.concatenate(
        [
            jnp.cross(X[..., :3], Y[..., :3]) + jnp.cross(X[..., 3:], Y[..., 3:]),
            jnp.cross(X[..., :3], Y[..., 3:]),
        ],
        axis=-1,
    )


def _invert_motion_transform_batched(X: Array) -> Array:
    """Batched inverse of motion transforms ``[[E, 0], [B, E]]`` (angular-first).

    ``X`` is (..., 6, 6) -> (..., 6, 6); uses the closed form
    ``[[E^T, 0], [-E^T B E^T, E^T]]`` (same as ``_invert_motion_transform``).
    """
    E = X[..., :3, :3]
    B = X[..., 3:, :3]
    Et = E.swapaxes(-1, -2)
    C = -(Et @ B @ Et)
    z = jnp.zeros_like(E)
    return jnp.block([[Et, z], [C, Et]])


def _spatial_data(
    dyn: DynamicsInfo, q: Array
) -> tuple[Array, Array, Array, Array]:
    """World-frame spatial data shared by the dense RNEA/CRBA solvers.

    Returns per-sample ``(S_w, I_w, mask, r)``:
      - ``S_w``  (n, 6): world-frame joint axes, ``S_w[i] = X0[i]^-1 S[i]``.
      - ``I_w``  (n, 6, 6): world-frame link inertias,
        ``I_w[i] = X0[i]^T I_body[i] X0[i]`` (adjoint of the world->body
        transform, so ``v^T I v`` is invariant under ``v_w = X0^-1 v_body``).
      - ``mask`` (n, n): ancestor mask, ``mask[i, j] = 1`` iff ``j`` is on the
        world->body-i path (inclusive).
      - ``r``    (n, 3): world position of each body frame origin (used to
        re-reference an external wrench to the world origin in RNEA).

    All quantities are angular-first. These are dense per-sample arrays; the
    caller's ``vmap`` over the flattened batch turns them into batched GEMMs
    (the frax pattern). ``X0`` is built with an unrolled parent-before-child
    accumulation (general tree; frax's ``_unrolled_fk``) -- for n <= 43 this is
    a handful of tiny 6x6 GEMMs, negligible against the n x n mask GEMMs that
    dominate the cost.
    """
    n = dyn.num_dof
    is_p = dyn.joint_is_prismatic
    S = dyn.S

    # Per-joint motion transform X_joint (parent -> joint frame), batched
    # Rodrigues: stacked skew + ONE batched 3x3 K@K GEMM, then one 6x6 GEMM.
    axis = jnp.where(is_p[:, None] > 0.5, S[:, 3:], S[:, :3])  # (n, 3)
    K = _skew_batched(axis)  # (n, 3, 3)
    s = jnp.sin(q)[:, None, None]
    c = jnp.cos(q)[:, None, None]
    I3 = jnp.eye(3)[None]  # (1, 3, 3)
    R = I3 + s * K + (1.0 - c) * (K @ K)  # (n, 3, 3)
    Rt = R.swapaxes(-1, -2)  # (n, 3, 3)
    z33 = jnp.zeros((n, 3, 3))
    X_rev = jnp.block([[Rt, z33], [z33, Rt]])  # (n, 6, 6)
    I3n = jnp.broadcast_to(I3, (n, 3, 3))  # (n, 3, 3)
    X_pri = jnp.block([[I3n, z33], [-_skew_batched(axis * q[:, None]), I3n]])
    X_joint = jnp.where(is_p[:, None, None] > 0.5, X_pri, X_rev)  # (n, 6, 6)
    Xup = X_joint @ dyn.X_tree  # (n, 6, 6)

    # World->body transforms X0, unrolled parent-before-child (general tree).
    order = _topological_dof_order(dyn)
    X0 = jnp.zeros((n, 6, 6))
    for i in order:
        p = dyn.parent_dof_indices[i]
        X0_i = Xup[i] if p == -1 else Xup[i] @ X0[p]
        X0 = X0.at[i].set(X0_i)

    X0_inv = _invert_motion_transform_batched(X0)  # (n, 6, 6)
    S_w = jnp.einsum("nij,nj->ni", X0_inv, S, precision=_HIGHEST)  # (n, 6)
    # World-frame inertia. A spatial metric (inertia) transforms by the
    # ADJOINT of the world->body transform, not its inverse: the scalar
    # v^T I v is invariant under v_w = X0^-1 v_body iff I_w = X0^T I_body X0.
    I_w = X0.swapaxes(-1, -2) @ dyn.I_body @ X0  # (n, 6, 6)
    r = _body_origins_world(X0)  # (n, 3) body frame origins in world coords
    mask = _ancestor_mask(dyn)  # (n, n)
    return S_w, I_w, mask, r


def _rnea_from_spatial(
    S_w: Array,
    I_w: Array,
    mask: Array,
    r: Array,
    damping: Array,
    qd: Array,
    qdd: Array,
    gravity: Array | float,
    f_ext: Array | None = None,
) -> Array:
    """Dense world-frame RNEA from shared spatial data (one GEMM per term).

    ``r`` gives the world position of each body frame origin; it is needed to
    re-reference an externally applied wrench from its body origin to the world
    origin at which this RNEA is evaluated (see the ``f_ext`` block below).
    """
    a_base = jnp.concatenate(
        [jnp.zeros(3), jnp.array([0.0, 0.0, 1.0]) * (-jnp.asarray(gravity))]
    )
    vJ = S_w * qd[:, None]  # (n, 6) joint velocity contributions (world)
    v = mask @ vJ  # (n, 6) body spatial velocities (world)
    a = a_base[None] + (mask @ _motion_cross(v, vJ)) + (
        mask @ (S_w * qdd[:, None])
    )  # (n, 6) body spatial accelerations (world)
    Ia = jnp.einsum("ijk,ik->ij", I_w, a, precision=_HIGHEST)
    Iv = jnp.einsum("ijk,ik->ij", I_w, v, precision=_HIGHEST)
    f = Ia + _force_cross(v, Iv)  # (n, 6) body spatial forces (world)
    if f_ext is not None:
        # f_ext[i] is a wrench applied at body i's FRAME ORIGIN (world axes), but
        # this RNEA is referenced at the WORLD origin. Shift each wrench: the
        # force part is a free vector (unchanged); the torque gains the moment of
        # that force about the world origin, r_i x F_i.
        F_ext = f_ext[:, 3:]
        f = f - jnp.concatenate(
            [f_ext[:, :3] + jnp.cross(r, F_ext), F_ext], axis=-1
        )
    net = mask.T @ f  # (n, 6) net joint forces (world)
    return jnp.einsum("ij,ij->i", S_w, net, precision=_HIGHEST) + damping * qd


def _crba_from_spatial(S_w: Array, I_w: Array, mask: Array) -> Array:
    """Dense world-frame CRBA from shared spatial data (einsum composite).

    ``M[a, b] = S_a^T (sum_{k anc a} I_w_k) S_b`` for ``b`` on ``a``'s
    ancestor path (inclusive); the mask-region product selects exactly those
    entries and symmetrizes the lower triangle.
    """
    I_comp = jnp.einsum("ij,jkl->ikl", mask.T, I_w, precision=_HIGHEST)  # (n, 6, 6)
    M_all = jnp.einsum("ij,ijk,lk->il", S_w, I_comp, S_w, precision=_HIGHEST)
    M_lower = mask * M_all
    return M_lower + jnp.tril(M_lower, -1).T


def _rnea_dense(
    dyn: DynamicsInfo,
    q: Array,
    qd: Array,
    qdd: Array,
    gravity: Array | float,
    f_ext: Array | None = None,
) -> Array:
    with jax.default_matmul_precision("highest"):
        return _rnea_from_spatial(
            *_spatial_data(dyn, q), dyn.damping, qd, qdd, gravity, f_ext
        )


def _crba_dense(dyn: DynamicsInfo, q: Array) -> Array:
    with jax.default_matmul_precision("highest"):
        S_w, I_w, mask, _ = _spatial_data(dyn, q)
        return _crba_from_spatial(S_w, I_w, mask)


def _fd_dense(
    dyn: DynamicsInfo,
    q: Array,
    qd: Array,
    tau: Array,
    gravity: Array | float,
    f_ext: Array | None = None,
) -> Array:
    """Joint accelerations via an explicit mass-matrix solve (CRBA + RNEA).

    Forms ``qdd = M(q)^-1 (tau - bias)`` where ``bias = RNEA(q, qd, qdd=0)``
    collects the Coriolis/centrifugal, gravity, viscous-damping and external-
    wrench terms. Both ``M`` and ``bias`` consume a single world-frame
    spatial-data pass (:func:`_spatial_data`), so the forward-kinematics chain
    is evaluated once per call (the previous list-based solvers computed it
    separately for CRBA and RNEA). The solve uses a Cholesky factorization of
    the symmetric positive-definite mass matrix (``assume_a="pos"``),
    following ``frax``'s forward-dynamics formulation. Compared with the O(n)
    Articulated Body Algorithm (:func:`_aba_single`) this avoids the per-joint
    division by the projected articulated inertia ``S_i^T IA_i S_i`` -- which
    is where ABA can produce NaN/Inf for degenerate (near-massless) links -- at
    the cost of an O(n^3) factorization, negligible for manipulator-sized ``n``
    and numerically robust for well-conditioned ``M``.
    """
    import jax.scipy as jsp

    with jax.default_matmul_precision("highest"):
        S_w, I_w, mask, r = _spatial_data(dyn, q)
        M = _crba_from_spatial(S_w, I_w, mask)
        bias = _rnea_from_spatial(
            S_w, I_w, mask, r, dyn.damping, qd, jnp.zeros_like(q), gravity, f_ext
        )
        return jsp.linalg.solve(M, tau - bias, assume_a="pos")


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
        lambda d, q_, qd_, qdd_, f_: _rnea_dense(d, q_, qd_, qdd_, gravity, f_),
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
    return _batched(_crba_dense, dyn, q)


def forward_dynamics_jax(
    dyn: DynamicsInfo,
    q: Float[Array, "*batch n_dof"],
    qd: Float[Array, "*batch n_dof"],
    tau: Float[Array, "*batch n_dof"],
    gravity: float = _DEFAULT_GRAVITY,
    f_ext: Float[Array, "*batch n_dof 6"] | None = None,
) -> Float[Array, "*batch n_dof"]:
    """Joint accelerations from state and torques.

    Computes ``qdd = M(q)^-1 (tau - RNEA(q, qd, 0))`` via a Cholesky solve of the
    composite-rigid-body mass matrix (:func:`_fd_dense`, following
    ``frax``). This is numerically robust for degenerate/near-massless links,
    where the O(n) Articulated Body Algorithm (:func:`_aba_single`, kept for
    reference/benchmarking) can divide by a vanishing projected inertia and
    return NaN/Inf. ``f_ext`` follows the same convention as
    :func:`inverse_dynamics_jax`.
    """
    return _batched(
        lambda d, q_, qd_, tau_, f_: _fd_dense(d, q_, qd_, tau_, gravity, f_),
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
