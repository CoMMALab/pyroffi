"""Fixed-step integrators over forward dynamics.

Written against a generic forward-dynamics callable ``fd(q, qd) -> qdd`` so
the same steppers drive both the pure-JAX ABA and the GRiD CUDA kernels.
All joints are 1-DOF revolute/prismatic (no floating base), so configuration
integration is plain vector addition; everything is jit/vmap/scan friendly.
"""

from __future__ import annotations

from typing import Callable, Literal

from jax import Array

StepMethod = Literal["semi_implicit", "euler", "rk4"]


def step_with_fd(
    fd: Callable[[Array, Array], Array],
    q: Array,
    qd: Array,
    dt: float | Array,
    method: StepMethod = "semi_implicit",
) -> tuple[Array, Array]:
    """Advance ``(q, qd)`` by one step of size ``dt`` under ``qdd = fd(q, qd)``.

    ``semi_implicit`` (symplectic Euler, the MuJoCo/brax default) updates the
    velocity first and integrates positions with the *new* velocity; ``euler``
    is explicit; ``rk4`` is classic fourth-order Runge-Kutta on ``(q, qd)``.
    """
    if method == "semi_implicit":
        qd_next = qd + dt * fd(q, qd)
        return q + dt * qd_next, qd_next
    if method == "euler":
        return q + dt * qd, qd + dt * fd(q, qd)
    if method == "rk4":
        k1_q, k1_qd = qd, fd(q, qd)
        k2_q = qd + 0.5 * dt * k1_qd
        k2_qd = fd(q + 0.5 * dt * k1_q, k2_q)
        k3_q = qd + 0.5 * dt * k2_qd
        k3_qd = fd(q + 0.5 * dt * k2_q, k3_q)
        k4_q = qd + dt * k3_qd
        k4_qd = fd(q + dt * k3_q, k4_q)
        q_next = q + (dt / 6.0) * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
        qd_next = qd + (dt / 6.0) * (k1_qd + 2 * k2_qd + 2 * k3_qd + k4_qd)
        return q_next, qd_next
    raise ValueError(f"Unknown integration method: {method!r}")
