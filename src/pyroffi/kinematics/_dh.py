"""Product-of-exponentials -> Denavit-Hartenberg extraction for serial chains.

pyroffi stores a robot as screw-axis twists + fixed parent transforms (a
product-of-exponentials / URDF formulation).  QuIK's C++ IK solver, by contrast,
is intrinsically **standard-DH** based: it wants a ``[N, 4]`` table of
``(a, alpha, d, theta)`` rows plus a base and tool transform, and rebuilds FK
from those with

    A_i = Rot_z(theta_i) . Trans_z(d_i) . Trans_x(a_i) . Rot_x(alpha_i)

(matching ``Robot<DOF>::FK`` in external/QuIK).  This module bridges the two by
*probing pyroffi's own FK* for the world-frame joint-axis lines at the home
configuration and running the classical (Spong/Craig) DH frame-assignment
algorithm on them.

The extraction only makes sense for an **unbranched serial chain** of revolute /
prismatic joints leading from the base to a single end-effector link — exactly
the class of robot (UR5, Panda, KUKA, ...) that the QuIK CPU backend targets.
The result is validated numerically against pyroffi FK across random
configurations; if the chain is not DH-representable (branched, or a geometry
DH cannot express) :func:`extract_dh` raises with a clear message so callers can
fall back to a JAX solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .._robot import Robot

_EPS = 1e-9


@dataclass(frozen=True)
class DHModel:
    """Standard-DH description of a serial chain, consumed by the QuIK backend.

    Attributes:
        dh:          ``[N, 4]`` float64 table of ``(a, alpha, d, theta)`` rows,
                     with the *constant* joint offset folded into ``theta`` (for
                     revolute joints) or ``d`` (prismatic).
        link_types:  ``[N]`` bool, ``True`` where joint is prismatic.
        qsign:       ``[N]`` +/-1, the sign relating the pyroffi joint coordinate
                     to the DH joint variable direction.
        t_base:      ``[4, 4]`` world -> DH-frame-0 transform.
        t_tool:      ``[4, 4]`` DH-frame-N -> end-effector transform.
        actuated_order: ``[N]`` indices into the robot's *actuated* joint vector
                     giving the base->tip chain order (QuIK's ``Q`` column order).
    """

    dh: np.ndarray
    link_types: np.ndarray
    qsign: np.ndarray
    t_base: np.ndarray
    t_tool: np.ndarray
    actuated_order: np.ndarray

    @property
    def dof(self) -> int:
        return int(self.dh.shape[0])


# ── small SE(3) / geometry helpers (numpy, host-side) ────────────────────────


def _wxyz_xyz_to_matrix(pose: np.ndarray) -> np.ndarray:
    """``[7] (wxyz, xyz)`` -> ``[4, 4]`` homogeneous transform."""
    w, x, y, z = pose[0], pose[1], pose[2], pose[3]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pose[4:7]
    return T


def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def _trans_x(a: float) -> np.ndarray:
    T = np.eye(4)
    T[0, 3] = a
    return T


def _trans_z(d: float) -> np.ndarray:
    T = np.eye(4)
    T[2, 3] = d
    return T


def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Standard-DH link transform ``Rot_z(theta) Trans_z(d) Trans_x(a) Rot_x(alpha)``."""
    return _rot_z(theta) @ _trans_z(d) @ _trans_x(a) @ _rot_x(alpha)


def _common_normal(
    pa: np.ndarray, za: np.ndarray, pb: np.ndarray, zb: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Common-normal geometry between two lines (pa, za) and (pb, zb).

    Returns ``(x_dir, foot_a, foot_b)``: the unit common-normal direction
    pointing from line a to line b, and the feet of the common perpendicular on
    each line.  Handles the parallel and intersecting degeneracies with the
    usual DH conventions (choose a perpendicular through pb; zero-length normal).
    """
    za = za / np.linalg.norm(za)
    zb = zb / np.linalg.norm(zb)
    cross = np.cross(za, zb)
    ncross = np.linalg.norm(cross)

    if ncross < 1e-7:
        # Parallel axes: common normal is the perpendicular from pb onto line a.
        # a_i is the (fixed) perpendicular distance; DH puts o_i at the foot on
        # zb nearest pa's origin, conventionally foot_b = pb.
        w = pb - pa
        foot_a = pa + np.dot(w, za) * za
        foot_b = pb
        x = foot_b - foot_a
        nx = np.linalg.norm(x)
        if nx < _EPS:
            # Coincident/collinear axes: pick any perpendicular to za.
            ref = np.array([1.0, 0.0, 0.0]) if abs(za[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            x = ref - np.dot(ref, za) * za
            x = x / np.linalg.norm(x)
            foot_a = foot_b = pa
        else:
            x = x / nx
        return x, foot_a, foot_b

    # Skew (or intersecting) axes: solve for the feet of the common perpendicular.
    x = cross / ncross
    w0 = pa - pb
    # Feet: pa + s*za and pb + t*zb minimize distance.
    a11 = 1.0
    a12 = -np.dot(za, zb)
    a22 = 1.0
    b1 = -np.dot(za, w0)
    b2 = np.dot(zb, w0)
    det = a11 * a22 - a12 * a12
    s = (b1 * a22 - a12 * b2) / det
    t = (a11 * b2 - b1 * a12) / det
    foot_a = pa + s * za
    foot_b = pb + t * zb
    # Orient x from line a to line b.
    if np.dot(foot_b - foot_a, x) < 0:
        x = -x
    return x, foot_a, foot_b


# ── FK probing ───────────────────────────────────────────────────────────────


def _serial_actuated_chain(robot: "Robot", ee_joint_idx: int) -> list[int]:
    """Ordered base->tip list of *joint* indices from base to ``ee_joint_idx``.

    Follows ``parent_indices`` up from the end-effector's parent joint, then
    reverses.  Raises if the walk does not terminate at the root (-1).
    """
    parent = np.asarray(robot.joints.parent_indices)
    chain: list[int] = []
    j = ee_joint_idx
    guard = 0
    while j != -1:
        chain.append(int(j))
        j = int(parent[j])
        guard += 1
        if guard > parent.shape[0] + 1:
            raise ValueError("Cycle detected walking the kinematic chain.")
    chain.reverse()
    return chain


def _joint_world_frames(robot: "Robot", cfg: np.ndarray) -> np.ndarray:
    """World pose ``[num_joints, 4, 4]`` of every joint frame at ``cfg``."""
    import jax.numpy as jnp
    from ._fk import forward_kinematics_joints_jax

    poses = np.asarray(
        forward_kinematics_joints_jax(robot, jnp.asarray(cfg))
    )  # [num_joints, 7]
    return np.stack([_wxyz_xyz_to_matrix(p) for p in poses], axis=0)


def extract_dh(
    robot: "Robot",
    ee_link_name: str,
    *,
    validate: bool = True,
    tol: float = 1e-5,
    n_validate: int = 16,
    seed: int = 0,
) -> DHModel:
    """Extract a standard-DH model for the serial chain base -> ``ee_link_name``.

    Args:
        robot:         pyroffi robot (product-of-exponentials formulation).
        ee_link_name:  Name of the end-effector link (chain tip).
        validate:      If True, check the reconstructed DH FK against pyroffi FK
                       at ``n_validate`` random configs and raise on mismatch.
        tol:           Max allowed position/orientation error during validation.

    Raises:
        ValueError: if the chain is branched / not a simple serial chain, or the
            reconstructed DH kinematics disagree with pyroffi FK (i.e. the chain
            is not expressible in standard DH form).
    """
    link_names = list(robot.links.names)
    if ee_link_name not in link_names:
        raise ValueError(f"Unknown end-effector link {ee_link_name!r}.")
    ee_link_idx = link_names.index(ee_link_name)
    ee_parent_joint = int(np.asarray(robot.links.parent_joint_indices)[ee_link_idx])
    if ee_parent_joint == -1:
        raise ValueError(
            f"Link {ee_link_name!r} is the base link; it has no parent joint to "
            "form a serial chain."
        )

    chain_joints = _serial_actuated_chain(robot, ee_parent_joint)

    # Keep only actuated joints (fixed joints are folded into transforms below).
    actuated_indices = np.asarray(robot.joints.actuated_indices)
    twists = np.asarray(robot.joints.twists)
    act_joint_ids = [j for j in chain_joints if actuated_indices[j] != -1]
    if not act_joint_ids:
        raise ValueError("Serial chain has no actuated joints.")
    dof = len(act_joint_ids)

    # Home-configuration world frames of every joint, and the EE link frame.
    n_act = int(robot.joints.num_actuated_joints)
    q0 = np.zeros(n_act, dtype=np.float64)
    frames0 = _joint_world_frames(robot, q0)

    # Axis line (point on axis, unit direction) for each actuated chain joint at
    # q=0, in world coordinates.  The URDF joint frame origin lies on the axis;
    # the local screw axis (angular part for revolute, linear for prismatic) is
    # the twist's rotational (resp. translational) component.
    p_list, z_list, is_pris = [], [], []
    for j in act_joint_ids:
        T = frames0[j]
        tw = twists[j]
        ang = tw[3:6]
        if np.linalg.norm(ang) > 1e-8:  # revolute
            axis_local = ang / np.linalg.norm(ang)
            is_pris.append(False)
        else:  # prismatic
            axis_local = tw[0:3] / np.linalg.norm(tw[0:3])
            is_pris.append(True)
        z_world = T[:3, :3] @ axis_local
        p_list.append(T[:3, 3].copy())
        z_list.append(z_world / np.linalg.norm(z_world))

    # End-effector frame at q=0.
    ee_pose = _ee_frame(robot, q0, ee_link_idx)

    # ── DH frame assignment ──────────────────────────────────────────────────
    # DH z-axes: zc[k] is the axis of joint (k+1).  We need N+1 frames F_0..F_N,
    # where F_{k}.z = zc[k].  The last frame reuses the previous z-axis (tool).
    zc = z_list + [z_list[-1]]
    pc = p_list + [p_list[-1]]

    # x-axes and origins per frame via successive common normals.
    x_axes: list[np.ndarray] = [None] * (dof + 1)  # type: ignore
    origins: list[np.ndarray] = [None] * (dof + 1)  # type: ignore
    for k in range(1, dof + 1):
        x_dir, foot_a, foot_b = _common_normal(pc[k - 1], zc[k - 1], pc[k], zc[k])
        x_axes[k] = x_dir
        origins[k] = foot_b
        if k == 1:
            # Frame 0 shares x_1 direction; its origin is the foot on axis 1.
            x_axes[0] = x_dir
            origins[0] = foot_a

    # Build world frames F_0..F_N and read DH parameters between them.
    dh = np.zeros((dof, 4), dtype=np.float64)
    qsign = np.ones(dof, dtype=np.float64)

    F_prev = _frame_from_axes(origins[0], x_axes[0], zc[0])
    t_base = F_prev.copy()
    for k in range(1, dof + 1):
        F_k = _frame_from_axes(origins[k], x_axes[k], zc[k])
        A = np.linalg.inv(F_prev) @ F_k  # should equal dh_transform(a,alpha,d,theta)
        a, alpha, d, theta = _params_from_A(A)
        dh[k - 1] = (a, alpha, d, theta)
        # By construction each DH frame's z-axis equals the joint axis direction
        # zc[k-1] that pyroffi rotates/translates about, so the DH joint variable
        # matches pyroffi's with a +1 sign.  (Validated against pyroffi FK below;
        # a wrong sign would surface there.)
        F_prev = F_k

    # Tool transform: residual from the last DH frame to the EE frame at q=0.
    t_tool = np.linalg.inv(F_prev) @ ee_pose

    link_types = np.asarray(is_pris, dtype=bool)
    # Map the chain's actuated joints to indices in the actuated-joint vector.
    actuated_order = np.array(
        [int(actuated_indices[j]) for j in act_joint_ids], dtype=np.int64
    )

    model = DHModel(
        dh=dh,
        link_types=link_types,
        qsign=qsign,
        t_base=t_base,
        t_tool=t_tool,
        actuated_order=actuated_order,
    )

    if validate:
        _validate(robot, model, ee_link_idx, tol=tol, n=n_validate, seed=seed)
    return model


def _frame_from_axes(origin: np.ndarray, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Build a right-handed homogeneous frame from an origin, x-axis and z-axis."""
    z = z / np.linalg.norm(z)
    x = x - np.dot(x, z) * z  # re-orthogonalise x against z
    nx = np.linalg.norm(x)
    if nx < _EPS:
        ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = ref - np.dot(ref, z) * z
        nx = np.linalg.norm(x)
    x = x / nx
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0] = x
    T[:3, 1] = y
    T[:3, 2] = z
    T[:3, 3] = origin
    return T


def _params_from_A(A: np.ndarray) -> tuple[float, float, float, float]:
    """Recover ``(a, alpha, d, theta)`` from a standard-DH link transform.

    Standard DH:
        A = [[ ct, -st*ca,  st*sa, a*ct],
             [ st,  ct*ca, -ct*sa, a*st],
             [  0,     sa,     ca,    d],
             [  0,      0,      0,    1]]
    """
    theta = np.arctan2(A[1, 0], A[0, 0])
    alpha = np.arctan2(A[2, 1], A[2, 2])
    d = A[2, 3]
    ct, st = np.cos(theta), np.sin(theta)
    # a*ct = A[0,3], a*st = A[1,3]; use the larger-magnitude component.
    if abs(ct) > abs(st):
        a = A[0, 3] / ct
    else:
        a = A[1, 3] / st
    return float(a), float(alpha), float(d), float(theta)


def _ee_frame(robot: "Robot", cfg: np.ndarray, ee_link_idx: int) -> np.ndarray:
    import jax
    import jax.numpy as jnp
    from ._fk import forward_kinematics

    # This extraction is validated against a tight tolerance (see _validate),
    # so it needs float64 FK precision here even though pyroffi no longer
    # forces x64 globally (see _ik_primitives.py). Scoped locally so callers
    # elsewhere stay in float32.
    with jax.enable_x64():
        poses = np.asarray(forward_kinematics(robot, jnp.asarray(cfg, dtype=jnp.float64)))  # [links, 7]
    return _wxyz_xyz_to_matrix(poses[ee_link_idx])


def dh_fk(model: DHModel, q_chain: np.ndarray) -> np.ndarray:
    """Reconstruct the EE ``[4,4]`` pose from the DH model for a chain-ordered q."""
    T = model.t_base.copy()
    for i in range(model.dof):
        a, alpha, d, theta = model.dh[i]
        if model.link_types[i]:
            d = d + q_chain[i] * model.qsign[i]
        else:
            theta = theta + q_chain[i] * model.qsign[i]
        T = T @ dh_transform(a, alpha, d, theta)
    return T @ model.t_tool


def _validate(
    robot: "Robot",
    model: DHModel,
    ee_link_idx: int,
    *,
    tol: float,
    n: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    lower = np.asarray(robot.joints.lower_limits, dtype=np.float64)
    upper = np.asarray(robot.joints.upper_limits, dtype=np.float64)
    lower = np.where(np.isfinite(lower), lower, -np.pi)
    upper = np.where(np.isfinite(upper), upper, np.pi)
    worst = 0.0
    for _ in range(n):
        q = rng.uniform(lower, upper)
        T_ref = _ee_frame(robot, q, ee_link_idx)
        T_dh = dh_fk(model, q[model.actuated_order])
        pos_err = np.linalg.norm(T_ref[:3, 3] - T_dh[:3, 3])
        R_err = T_ref[:3, :3].T @ T_dh[:3, :3]
        ang_err = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
        worst = max(worst, pos_err, ang_err)
        if pos_err > tol or ang_err > tol:
            raise ValueError(
                "POE->DH extraction failed validation: the chain to "
                f"link index {ee_link_idx} is not expressible in standard DH "
                f"form (pos err {pos_err:.2e} m, ang err {ang_err:.2e} rad at a "
                "random config exceed tol "
                f"{tol:.1e}). Use a JAX IK solver for this robot instead."
            )
    # Attach the observed worst error for callers that want it (debug).
    object.__setattr__(model, "_worst_validation_error", worst)
