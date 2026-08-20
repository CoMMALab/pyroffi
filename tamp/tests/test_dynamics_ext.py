"""Tests for extensions/dynamics.py. Run from repo root:
    PYTHONPATH=. python tests/test_dynamics_ext.py
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import jax
import jax.numpy as jnp

from spasm import backend
from spasm.extensions.dynamics import (
    inverse_dynamics, forward_dynamics, torque_cost, track_rollout,
    TORQUE_LIMITS, DYN_ROBOT,
)


def test_gravity_torques_at_rest():
    """(a) Inverse dynamics of the panda at rest (qd=qdd=0) should equal pure
    gravity-compensation torque: nonzero (arm is not gravity-neutral at an
    arbitrary rest config), finite, and within Franka's torque limits."""
    lower, upper = backend.get_joint_limits()
    q = (lower + upper) / 2.0
    tau = inverse_dynamics(q, jnp.zeros(7), jnp.zeros(7))

    assert jnp.all(jnp.isfinite(tau)), f"non-finite gravity torque: {tau}"
    assert float(jnp.max(jnp.abs(tau))) > 1e-3, f"gravity torque suspiciously ~0: {tau}"
    assert jnp.all(jnp.abs(tau) < TORQUE_LIMITS), (
        f"gravity torque exceeds Franka limits: {tau} vs {TORQUE_LIMITS}"
    )
    print(f"  gravity tau at mid-range q: {tau}")
    print("PASS test_gravity_torques_at_rest")


def test_forward_inverse_consistency():
    """(b) forward_dynamics(inverse_dynamics(q, qd, qdd)) should reproduce
    qdd, on random states, within numerical tolerance (f32)."""
    lower, upper = backend.get_joint_limits()
    key = jax.random.PRNGKey(0)
    n_trials = 20
    max_err = 0.0
    for i in range(n_trials):
        key, kq, kqd, kqdd = jax.random.split(key, 4)
        q = jax.random.uniform(kq, (7,), minval=lower, maxval=upper)
        qd = jax.random.uniform(kqd, (7,), minval=-1.0, maxval=1.0)
        qdd = jax.random.uniform(kqdd, (7,), minval=-2.0, maxval=2.0)

        tau = inverse_dynamics(q, qd, qdd)
        qdd_out = forward_dynamics(q, qd, tau)
        tau_rt = inverse_dynamics(q, qd, qdd_out)

        assert jnp.all(jnp.isfinite(tau)) and jnp.all(jnp.isfinite(qdd_out))
        err = float(jnp.max(jnp.abs(tau_rt - tau)))
        max_err = max(max_err, err)

    print(f"  max ID(FD(tau)) round-trip error over {n_trials} trials: {max_err:.2e}")
    assert max_err < 1e-2, f"round-trip torque error too large: {max_err}"
    print("PASS test_forward_inverse_consistency")


def test_torque_cost_differentiable():
    """(c) torque_cost is differentiable; jax.grad returns finite gradients,
    including for a trajectory that intentionally violates torque limits
    (large joint swings over a short dt) so the hinge penalty is active."""
    lower, upper = backend.get_joint_limits()
    q0 = (lower + upper) / 2.0
    T = 12
    # Aggressive swing on joint 2 (a heavily-loaded joint) to trigger the
    # penalty, so this isn't just testing grad-of-zero.
    q_traj = jnp.tile(q0, (T, 1)) + jnp.linspace(0.0, 2.5, T)[:, None] * jnp.eye(7)[1]
    dt = 0.01

    val = torque_cost(q_traj, dt)
    assert jnp.isfinite(val), f"torque_cost not finite: {val}"
    assert float(val) > 0.0, "expected this aggressive trajectory to violate limits (cost==0)"

    grad = jax.grad(torque_cost)(q_traj, dt)
    assert jnp.all(jnp.isfinite(grad)), f"non-finite grad: {grad}"
    assert float(jnp.max(jnp.abs(grad))) > 0.0, "grad is exactly zero everywhere"

    print(f"  torque_cost={float(val):.4f}, max|grad|={float(jnp.max(jnp.abs(grad))):.4f}")
    print("PASS test_torque_cost_differentiable")


def test_track_rollout_slow_interp_small_error():
    """(d) PD-tracking a slow joint-space interpolation should produce a
    small tracking error (dynamic feasibility check on an easy trajectory)."""
    lower, upper = backend.get_joint_limits()
    q_start = (lower + upper) / 2.0
    q_end = q_start + 0.15 * jnp.eye(7)[3]  # small, slow move on one joint
    T = 20
    dt = 0.2  # slow: 4 s total for a 0.15 rad move
    q_traj = jnp.linspace(q_start, q_end, T)

    result = track_rollout(q_traj, dt, kp=200.0, kd=15.0)
    assert jnp.all(jnp.isfinite(result['q_actual'])), "rollout diverged (non-finite)"
    rms = float(result['rms_tracking_error'])
    max_err = float(result['max_tracking_error'])
    print(f"  slow-interp PD tracking: rms={rms:.4f} rad, max={max_err:.4f} rad")
    assert rms < 0.1, f"tracking rms too large for a slow trajectory: {rms}"
    print("PASS test_track_rollout_slow_interp_small_error")


if __name__ == '__main__':
    test_gravity_torques_at_rest()
    test_forward_inverse_consistency()
    test_torque_cost_differentiable()
    test_track_rollout_slow_interp_small_error()
    print("\nAll tests passed.")
