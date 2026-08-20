"""Dynamics extension for spasm-pyroffi.

Wraps pyroffi's pure-JAX manipulator dynamics (RNEA/CRBA/forward dynamics,
exposed as Robot.inverse_dynamics/forward_dynamics/step) around SPaSM 7-dof
joint trajectories.

Why a separate robot object: `backend.ROBOT` is built from SPaSM's own URDF
(`spasm/kinematics/urdf/panda_sphere_visuals.urdf`), which has a mimic joint
(finger2 mimics finger1) and no <inertial> tags for most links. pyroffi's
dynamics explicitly refuses mimic-joint URDFs (`robot.dynamics is None` — see
`_robot.py:136`, "Dynamics does not support URDFs with mimic joints") and
would in any case have nothing to compute inertias from. So this module loads
pyroffi's own inertial-complete, mimic-free URDF
(`pyroffi/resources/panda/panda_spherized.urdf`) into a second `Robot`
(`DYN_ROBOT`) purely for dynamics.

Joint-order check (done at import, see `_verify_joint_order`): DYN_ROBOT's 7
actuated joint names are exactly `backend.ROBOT.joints.actuated_names[:7]`
in the same order (`panda_joint1..7`), i.e. the same order as the 7-dof `q`
used throughout the port's trajectories. No remap is needed. This is asserted
at import time so a future URDF swap can't silently reorder joints under us.

dtype: pyroffi's dynamics primitives (`dynamics/_dynamics_jax.py`) are pure
JAX and run fine in f32 -- unlike pyroffi's IK primitives (see backend.py's
docstring / PORT_NOTES.md), dynamics does NOT force jax_enable_x64. Verified
empirically (see tests): `robot.inverse_dynamics` on an f32 q returns f32
tau. So no global flag flip and no boundary casting is actually required
here; we still defensively cast to f32 at the module boundary in case pyroffi
promotes dtypes internally on some path, per the task's isolation
requirement.
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import yourdfpy
import pyroffi as pk


from spasm import backend  # noqa: E402  (registers jax_enable_x64=False as a side effect)

from spasm.paths import PANDA_URDF as DYN_URDF

_dyn_urdf_obj = yourdfpy.URDF.load(DYN_URDF, load_meshes=False)
DYN_ROBOT = pk.Robot.from_urdf(_dyn_urdf_obj)

if DYN_ROBOT.dynamics is None:
    raise RuntimeError(
        f"pyroffi dynamics unavailable for {DYN_URDF}; extensions/dynamics.py "
        "cannot function. This URDF was chosen specifically because it is "
        "inertial-complete and mimic-free -- if this fires, pyroffi's "
        "resource file changed underneath us."
    )

NUM_DOF = DYN_ROBOT.dynamics.num_dof
assert NUM_DOF == 7, f"expected 7-dof arm, got {NUM_DOF}"


def _verify_joint_order():
    dyn_names = DYN_ROBOT.joints.actuated_names
    port_names = backend.ROBOT.joints.actuated_names[:7]
    assert tuple(dyn_names) == tuple(port_names), (
        "Joint order mismatch between pyroffi's dynamics URDF and the port's "
        f"kinematics URDF: dynamics={dyn_names} vs port={port_names}. A remap "
        "would be required (not implemented)."
    )


_verify_joint_order()

# Franka torque limits (Nm), joint order 1..7.
TORQUE_LIMITS = jnp.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])

GRAVITY = -9.81


def _cast_f32(*arrays):
    return tuple(jnp.asarray(a, dtype=jnp.float32) for a in arrays)


def inverse_dynamics(q, qd, qdd):
    """tau (7,) that realizes qdd at state (q, qd). f32 in, f32 out."""
    q, qd, qdd = _cast_f32(q, qd, qdd)
    tau = DYN_ROBOT.inverse_dynamics(q, qd, qdd)
    return jnp.asarray(tau, dtype=jnp.float32)


inverse_dynamics_batched = jax.jit(jax.vmap(inverse_dynamics))


def forward_dynamics(q, qd, tau):
    q, qd, tau = _cast_f32(q, qd, tau)
    qdd = DYN_ROBOT.forward_dynamics(q, qd, tau)
    return jnp.asarray(qdd, dtype=jnp.float32)


def _finite_diff_qd_qdd(q_traj, dt):
    """q_traj: (T, 7) waypoints -> qd, qdd (T, 7) via central differences
    (forward/backward at the boundaries)."""
    T = q_traj.shape[0]
    qd = jnp.gradient(q_traj, dt, axis=0)
    qd_pad = jnp.concatenate([qd[:1], qd, qd[-1:]], axis=0)
    qdd = (qd_pad[2:] - qd_pad[:-2]) / (2 * dt)
    return qd, qdd


def torque_profile(q_traj, dt):
    """Finite-difference qd/qdd from a (T, 7) waypoint trajectory, then
    vmapped inverse dynamics -> per-waypoint torques.

    Returns dict with:
      tau: (T, 7) torque per waypoint
      max_abs_tau: (7,) max |tau| per joint across the trajectory
      limit_violation: (7,) bool, whether max_abs_tau exceeds TORQUE_LIMITS
      violation_frac: (T, 7) fraction over the limit at each waypoint (0 if ok)
    """
    q_traj = jnp.asarray(q_traj, dtype=jnp.float32)
    dt = float(dt)
    qd, qdd = _finite_diff_qd_qdd(q_traj, dt)

    tau = jax.vmap(inverse_dynamics)(q_traj, qd, qdd)  # (T, 7)
    max_abs_tau = jnp.max(jnp.abs(tau), axis=0)
    limit_violation = max_abs_tau > TORQUE_LIMITS
    violation_frac = jnp.maximum(0.0, (jnp.abs(tau) - TORQUE_LIMITS[None, :]) / TORQUE_LIMITS[None, :])

    return {
        'tau': tau,
        'qd': qd,
        'qdd': qdd,
        'max_abs_tau': max_abs_tau,
        'limit_violation': limit_violation,
        'violation_frac': violation_frac,
    }


def torque_cost(q_traj, dt):
    """Differentiable soft penalty for torque-limit violations, usable as an
    extra trajopt cost term. Scalar; 0 if all torques are within limits.

    Uses a smooth relu**2 hinge on (|tau| - limit) so it's grad-friendly
    (no jnp.max reductions in the loss path -- summed over time/joints
    instead, which keeps gradients dense across the whole trajectory rather
    than routing them through a single argmax waypoint).
    """
    q_traj = jnp.asarray(q_traj, dtype=jnp.float32)
    dt = float(dt)
    qd, qdd = _finite_diff_qd_qdd(q_traj, dt)
    tau = jax.vmap(inverse_dynamics)(q_traj, qd, qdd)  # (T, 7)
    over = jax.nn.relu(jnp.abs(tau) - TORQUE_LIMITS[None, :])
    return jnp.sum(over ** 2)


def track_rollout(q_traj, dt, kp, kd, substeps=None, method='linearly_implicit'):
    """PD-tracking rollout of a planned (T, 7) waypoint trajectory using
    Robot.step (semi-implicit Euler). Treats q_traj as the reference/setpoint
    sequence; at each step commands tau = PD(q_ref[t], qd_ref[t]; q, qd) and
    integrates the *true* dynamics one dt forward, so drift/lag from finite
    torque/actuator dynamics shows up as tracking error -- a dynamic-
    feasibility validator for planned trajectories (a trajopt plan can be
    kinematically fine and still be untrackable if it demands torques/
    accelerations the arm can't produce).

    kp, kd: either scalars or (7,) gain vectors.

    method: integrator method passed through to pyroffi's ``step_with_fd``.
    Defaults to ``'linearly_implicit'`` (Rosenbrock-Euler). The stiff part of a
    PD-tracked rollout is the feedback law itself, so we fold the PD torque
    *into* the forward-dynamics callable ``fd(q, qd)`` and let the implicit
    step linearize it -- this is what makes the rollout stable. With the plain
    ``'semi_implicit'`` integrator and the PD applied *outside* ``fd`` (the old
    path), the closed loop diverges to NaN for any realistic gains even at
    200 Hz substepping (kp=50 NaNs at every dt tested); folding PD into the
    implicit step keeps it stable at the full waypoint spacing (kp=50 tracks at
    dt up to ~0.1 s where the explicit path NaNs at 0.05 s). This is not a
    magic "any dt" fix -- the underlying arm dynamics have a ~0.02 s natural
    timescale, so very coarse dt still need substeps -- but it removes the
    spurious NaN that made the validator unusable.

    substeps: number of integrator substeps per waypoint interval `dt`
    (default: ceil(dt / 0.02), i.e. keep the effective step within the arm's
    natural timescale). The PD reference is sampled once per outer waypoint
    (fast inner control loop, slower outer trajectory reference).

    Returns dict with:
      q_actual: (T, 7) simulated joint trajectory (sampled once per waypoint,
        i.e. at the end of each substep block)
      qd_actual: (T, 7)
      tau_cmd: (T, 7) commanded torque at the first substep of each interval
      tracking_error: (T, 7) q_actual - q_ref
      max_tracking_error: scalar, max |tracking_error|
      rms_tracking_error: scalar
    """
    q_traj = jnp.asarray(q_traj, dtype=jnp.float32)
    dt = float(dt)
    T = q_traj.shape[0]
    kp = jnp.asarray(kp, dtype=jnp.float32) * jnp.ones(7, dtype=jnp.float32)
    kd = jnp.asarray(kd, dtype=jnp.float32) * jnp.ones(7, dtype=jnp.float32)

    if substeps is None:
        substeps = max(1, int(jnp.ceil(dt / 0.02)))
    substeps = int(substeps)
    sub_dt = dt / substeps

    qd_ref, _ = _finite_diff_qd_qdd(q_traj, dt)

    q0 = q_traj[0]
    qd0 = jnp.zeros(7, dtype=jnp.float32)

    def inner_step(carry, _):
        q, qd, q_ref, qd_ref_t = carry
        # Fold the PD law into the forward-dynamics callable so the implicit
        # integrator linearizes the (stiff) feedback term. tau is recomputed
        # for whatever (q, qd) the implicit solve evaluates fd at.
        def fd(q_, qd_):
            tau_ = jnp.clip(kp * (q_ref - q_) + kd * (qd_ref_t - qd_),
                            -TORQUE_LIMITS, TORQUE_LIMITS)
            return DYN_ROBOT.forward_dynamics(q_, qd_, tau_, gravity=GRAVITY)
        q_next, qd_next = pk.dynamics.step_with_fd(fd, q, qd, sub_dt, method=method)
        tau = jnp.clip(kp * (q_ref - q) + kd * (qd_ref_t - qd),
                       -TORQUE_LIMITS, TORQUE_LIMITS)
        return (q_next, qd_next, q_ref, qd_ref_t), tau

    def step_fn(carry, ref):
        q, qd = carry
        q_ref, qd_ref_t = ref
        (q_next, qd_next, _, _), taus = jax.lax.scan(
            inner_step, (q, qd, q_ref, qd_ref_t), None, length=substeps
        )
        return (q_next, qd_next), (q_next, qd_next, taus[0])

    (_, _), (q_actual, qd_actual, tau_cmd) = jax.lax.scan(
        step_fn, (q0, qd0), (q_traj, qd_ref)
    )

    tracking_error = q_actual - q_traj
    return {
        'q_actual': q_actual,
        'qd_actual': qd_actual,
        'tau_cmd': tau_cmd,
        'tracking_error': tracking_error,
        'max_tracking_error': jnp.max(jnp.abs(tracking_error)),
        'rms_tracking_error': jnp.sqrt(jnp.mean(tracking_error ** 2)),
    }


track_rollout_jit = jax.jit(track_rollout, static_argnames=('dt', 'method'))
