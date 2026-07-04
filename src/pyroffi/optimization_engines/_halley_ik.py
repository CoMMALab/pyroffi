"""Pure-JAX reimplementation of QuIK's third-order Halley's-method IK.

This is the JAX counterpart to the QuIK C++ backend
(:mod:`pyroffi.optimization_engines._quik_ik`), provided for a like-for-like
comparison of the *same algorithm* across backends.  It reproduces QuIK's
Halley update (S. Lloyd et al., IEEE T-RO 2022) directly on pyroffi's
product-of-exponentials model, so it needs no DH extraction and runs on whatever
JAX platform is active (``JAX_PLATFORMS=cpu`` for CPU-only planning, or CUDA).

Per iteration (matching ``external/QuIK`` algorithm 0):

    e   = hgtDiff(FK(q), T_target)            # Sugihara 6-vec: [lin; ang]
    e   = clampMag(e)                         # saturate step
    dqn = -0.5 * Jpinv(J) @ e                 # half Newton step
    A   = J + (dJ/dq . dqn)                    # geometric Hessian product (jvp)
    dq  = -Jpinv(A) @ e
    q  += dq

where ``J`` is the geometric Jacobian and ``Jpinv`` is the damped right
pseudoinverse ``Jᵀ (J Jᵀ + λ²I)⁻¹`` (QuIK's ``lsolve``).  The whole solve is
``jit``/``vmap``-friendly: a fixed iteration count over a batch of seeds, then
pick the lowest-error solution.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Sequence

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
from jax import Array

from ..kinematics._dh import _serial_actuated_chain

if TYPE_CHECKING:
    from .._robot import Robot


# ── error / clamp / solve, mirroring QuIK's hgtDiff, clampMag, lsolve ─────────


def hgt_diff(T_cur: Array, T_tgt: Array) -> Array:
    """Sugihara pose error between two ``[4,4]`` transforms -> ``[6]`` ``[lin; ang]``.

    Reproduces QuIK's ``hgtDiff``: linear part is ``d_cur - d_tgt``; angular part
    is the axis-angle of ``R_cur R_tgtᵀ`` with the near-zero and near-pi Taylor
    branches handled to stay smooth for autodiff.
    """
    R1 = T_cur[:3, :3]
    R2 = T_tgt[:3, :3]
    d1 = T_cur[:3, 3]
    d2 = T_tgt[:3, 3]
    Re = R1 @ R2.T
    lin = d1 - d2
    t = jnp.trace(Re)
    eps = jnp.array(
        [Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]]
    )
    eps_norm = jnp.linalg.norm(eps)

    # Normal / near-zero branch: w = atan2(eps_norm, t-1)/eps_norm, with the
    # small-angle Taylor (0.75 - t/12) used where eps_norm is tiny (also keeps
    # the gradient finite at eps_norm -> 0).
    small = (0.75 - t / 12.0) * eps
    safe_norm = jnp.where(eps_norm < 1e-3, 1.0, eps_norm)
    normal = (jnp.arctan2(eps_norm, t - 1.0) / safe_norm) * eps
    ang_main = jnp.where(eps_norm < 1e-3, small, normal)

    # Near-pi branch (trace ~ -1): approximate from the diagonal.
    ang_pi = 1.570796326794897 * (jnp.diag(Re) + 1.0)
    near_pi = (t <= -0.99) & (eps_norm <= 1e-10)
    ang = jnp.where(near_pi, ang_pi, ang_main)
    return jnp.concatenate([lin, ang])


def clamp_mag(e: Array, max_lin: float, max_ang: float) -> Array:
    """QuIK's ``clampMag``: independently saturate the linear/angular sub-norms."""
    lin, ang = e[:3], e[3:]
    lin_n = jnp.linalg.norm(lin)
    ang_n = jnp.linalg.norm(ang)
    lin = jnp.where(lin_n > max_lin, lin * (max_lin / lin_n), lin)
    ang = jnp.where(ang_n > max_ang, ang * (max_ang / ang_n), ang)
    return jnp.concatenate([lin, ang])


def lsolve(J: Array, b: Array, lambda2: float) -> Array:
    """Damped right pseudoinverse solve ``x = Jᵀ (J Jᵀ + λ²I)⁻¹ b`` (QuIK ``lsolve``)."""
    JJt = J @ J.T
    JJt = JJt + lambda2 * jnp.eye(JJt.shape[0], dtype=J.dtype)
    y = jnp.linalg.solve(JJt, b)
    return J.T @ y


# ── geometric FK / Jacobian on the POE model ─────────────────────────────────


def _chain_info(robot: "Robot", ee_link_idx: int):
    """Static per-chain data: ordered actuated-joint ids, local axes, prismatic.

    Returns ``(chain_act_joint_ids, actuated_slots, local_axes, is_prismatic)``
    where ``actuated_slots`` maps each chain joint to its column in the robot's
    actuated-joint vector.
    """
    ee_parent_joint = int(np.asarray(robot.links.parent_joint_indices)[ee_link_idx])
    chain_joints = _serial_actuated_chain(robot, ee_parent_joint)
    actuated_indices = np.asarray(robot.joints.actuated_indices)
    twists = np.asarray(robot.joints.twists)
    ids, slots, axes, pris = [], [], [], []
    for j in chain_joints:
        if actuated_indices[j] == -1:
            continue
        tw = twists[j]
        ang = tw[3:6]
        if np.linalg.norm(ang) > 1e-8:
            axes.append(ang / np.linalg.norm(ang))
            pris.append(False)
        else:
            axes.append(tw[0:3] / np.linalg.norm(tw[0:3]))
            pris.append(True)
        ids.append(int(j))
        slots.append(int(actuated_indices[j]))
    return (
        np.array(ids, dtype=np.int64),
        np.array(slots, dtype=np.int64),
        np.array(axes, dtype=np.float64),
        np.array(pris, dtype=bool),
    )


def _fk_and_jacobian(
    robot: "Robot",
    q_chain: Array,
    chain_ids: Array,
    local_axes: Array,
    is_pris: Array,
    ee_link_idx: int,
    actuated_slots: Array,
    n_act: int,
):
    """Return ``(T_ee [4,4], J [6, dof])`` for a chain-ordered ``q_chain``."""
    from ..kinematics._fk import (
        forward_kinematics_joints_jax,
        link_poses_from_joint_poses,
    )

    cfg = jnp.zeros((n_act,), q_chain.dtype).at[actuated_slots].set(q_chain)
    joint_poses = forward_kinematics_joints_jax(robot, cfg)  # [num_joints, 7]
    link_poses = link_poses_from_joint_poses(robot, joint_poses)  # [num_links, 7]
    T_ee = jaxlie.SE3(link_poses[ee_link_idx]).as_matrix()
    o_ee = T_ee[:3, 3]

    # Per chain-joint world axis & origin.
    Ts = jaxlie.SE3(joint_poses[chain_ids]).as_matrix()  # [dof, 4, 4]
    R = Ts[:, :3, :3]
    o = Ts[:, :3, 3]
    z = jnp.einsum("dij,dj->di", R, local_axes.astype(q_chain.dtype))  # world axes

    pris = is_pris.astype(q_chain.dtype)[:, None]
    lin = (1.0 - pris) * jnp.cross(z, o_ee - o) + pris * z
    ang = (1.0 - pris) * z
    J = jnp.concatenate([lin, ang], axis=1).T  # [6, dof]
    return T_ee, J


def _halley_step(robot, q_chain, T_tgt, static, opts):
    chain_ids, local_axes, is_pris, ee_link_idx, actuated_slots, n_act = static
    max_lin, max_ang, lambda2, algorithm = opts

    def jac_fn(q):
        _, J = _fk_and_jacobian(
            robot, q, chain_ids, local_axes, is_pris, ee_link_idx, actuated_slots, n_act
        )
        return J

    T_ee, J = _fk_and_jacobian(
        robot, q_chain, chain_ids, local_axes, is_pris, ee_link_idx, actuated_slots, n_act
    )
    e = hgt_diff(T_ee, T_tgt)
    e = clamp_mag(e, max_lin, max_ang)

    # Newton half-step.
    dqn = (-0.5 * lsolve(J, e, lambda2)).astype(q_chain.dtype)

    def halley_branch(_):
        # A = J + (dJ/dq . dqn), the geometric Hessian product via a jvp.
        _, dJ = jax.jvp(jac_fn, (q_chain,), (dqn,))
        A = J + dJ
        return -lsolve(A, e, lambda2)

    def newton_branch(_):
        return -lsolve(J, e, lambda2)

    # algorithm 0 -> Halley, 1 -> Newton/LM.
    dq = jax.lax.cond(algorithm == 0, halley_branch, newton_branch, operand=None)
    return (q_chain + dq.astype(q_chain.dtype)), jnp.linalg.norm(hgt_diff(T_ee, T_tgt))


def _solve_one(robot, q0_chain, T_tgt, static, opts, iter_max):
    def body(_, q):
        q_new, _ = _halley_step(robot, q, T_tgt, static, opts)
        return q_new

    q = jax.lax.fori_loop(0, iter_max, body, q0_chain)
    # Final (unclamped) error norm for seed selection.
    chain_ids, local_axes, is_pris, ee_link_idx, actuated_slots, n_act = static
    T_ee, _ = _fk_and_jacobian(
        robot, q, chain_ids, local_axes, is_pris, ee_link_idx, actuated_slots, n_act
    )
    return q, jnp.linalg.norm(hgt_diff(T_ee, T_tgt))


class HalleyJAXSolver:
    """Reusable pure-JAX Halley IK solver bound to a robot + end-effector."""

    def __init__(self, robot: "Robot", ee_link_name: str) -> None:
        self.robot = robot
        self.ee_link_name = ee_link_name
        self.ee_link_idx = robot.links.names.index(ee_link_name)
        ids, slots, axes, pris = _chain_info(robot, self.ee_link_idx)
        self._chain_ids = jnp.asarray(ids)
        self._slots = jnp.asarray(slots)
        self._axes = jnp.asarray(axes)
        self._pris = jnp.asarray(pris)
        self._n_act = int(robot.joints.num_actuated_joints)
        self.dof = int(ids.shape[0])
        self._actuated_order = ids  # numpy, for scatter-back
        self._slots_np = slots

    @partial(jax.jit, static_argnames=("self", "iter_max", "algorithm"))
    def _solve_batch(
        self,
        seeds: Array,
        T_tgt: Array,
        max_lin: float,
        max_ang: float,
        lambda2: float,
        algorithm: int,
        iter_max: int,
    ):
        static = (
            self._chain_ids,
            self._axes,
            self._pris,
            self.ee_link_idx,
            self._slots,
            self._n_act,
        )
        opts = (max_lin, max_ang, lambda2, jnp.int32(algorithm))
        return jax.vmap(
            lambda q0: _solve_one(self.robot, q0, T_tgt, static, opts, iter_max)
        )(seeds)

    def solve(
        self,
        target_pose: Array,
        seeds: Array,
        *,
        algorithm: int = 0,
        iter_max: int = 100,
        lambda2: float = 0.0,
        max_lin_step: float = 0.34,
        max_ang_step: float = 1.0,
    ) -> dict[str, Array]:
        """Solve a batch of seeds against one ``[4,4]`` target. Returns chain-order q."""
        T = jnp.asarray(target_pose, jnp.float32).reshape(4, 4)
        seeds = jnp.asarray(seeds, jnp.float32).reshape(-1, self.dof)
        q, err = self._solve_batch(
            seeds, T, float(max_lin_step), float(max_ang_step),
            float(lambda2), int(algorithm), int(iter_max),
        )
        return {"q": q, "error": err}

    def solve_to_actuated(self, target_pose, seeds, **kwargs) -> dict[str, Array]:
        out = self.solve(target_pose, seeds, **kwargs)
        q_full = jnp.zeros(out["q"].shape[:-1] + (self._n_act,), out["q"].dtype)
        q_full = q_full.at[..., self._slots].set(out["q"])
        out["q_actuated"] = q_full
        return out


# Reuse one solver (and its jit'd batch fn) per (robot identity, ee link) so the
# dispatcher does not recompile the batched Halley loop on every IK call.
_SOLVER_CACHE: dict[tuple[int, str], "HalleyJAXSolver"] = {}


def _cached_solver(robot: "Robot", ee_link_name: str) -> "HalleyJAXSolver":
    key = (id(robot), ee_link_name)
    solver = _SOLVER_CACHE.get(key)
    if solver is None:
        solver = HalleyJAXSolver(robot, ee_link_name)
        _SOLVER_CACHE[key] = solver
    return solver


def halley_ik_solve(
    robot: "Robot",
    target_link_indices: Sequence[int],
    target_poses: Sequence,
    rng_key: Array,
    previous_cfg: Array,
    num_seeds: int = 32,
    continuity_weight: float = 1e-3,
    fixed_joint_mask: Array | None = None,
    *,
    algorithm: int = 0,
    iter_max: int = 100,
    lambda2: float = 0.0,
    **_ignored,
) -> Array:
    """Dispatcher-shaped single-target JAX-Halley solve (see ``_ik``)."""
    if len(target_link_indices) != 1:
        raise ValueError(
            "The Halley (JAX) backend solves a single serial chain; got "
            f"{len(target_link_indices)} end-effector targets."
        )
    ee_link_name = robot.links.names[target_link_indices[0]]
    solver = _cached_solver(robot, ee_link_name)

    pose = target_poses[0]
    T = pose.as_matrix() if isinstance(pose, jaxlie.SE3) else jnp.asarray(pose)

    slots = np.asarray(solver._slots)
    lower = np.asarray(robot.joints.lower_limits)[slots]
    upper = np.asarray(robot.joints.upper_limits)[slots]
    lower = np.where(np.isfinite(lower), lower, -np.pi)
    upper = np.where(np.isfinite(upper), upper, np.pi)
    key = rng_key if rng_key is not None else jax.random.PRNGKey(0)
    rand = jax.random.uniform(
        key, (num_seeds, solver.dof), minval=jnp.asarray(lower), maxval=jnp.asarray(upper)
    )
    prev = jnp.asarray(np.asarray(previous_cfg)[slots])[None]
    seeds = jnp.concatenate([prev, rand], axis=0)

    out = solver.solve_to_actuated(
        T, seeds, algorithm=algorithm, iter_max=iter_max, lambda2=lambda2
    )
    best = jnp.argmin(out["error"])
    return out["q_actuated"][best]
