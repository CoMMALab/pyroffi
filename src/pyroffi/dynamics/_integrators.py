"""Fixed-step integrators over forward dynamics.

Written against a generic forward-dynamics callable ``fd(q, qd) -> qdd`` so
the same steppers drive both the pure-JAX ABA and the GRiD CUDA kernels.
Configurations advance through ``retract(q, v, dt)``: plain ``q + v * dt`` for
1-DOF joints, and the SE(3) exponential for a floating base
(:func:`pyroffi._floating_base.integrate`), so ``q`` may be longer than ``qd``.
Everything is jit/vmap/scan friendly.
"""

from __future__ import annotations

from typing import Callable, Literal

import jax
from jax import Array
from jax import numpy as jnp

StepMethod = Literal["semi_implicit", "euler", "rk4", "linearly_implicit"]
Retract = Callable[[Array, Array, "float | Array"], Array]


def _add(q: Array, v: Array, dt) -> Array:
    return q + v * dt


def _linearly_implicit_step(
    fd: Callable[[Array, Array], Array],
    q: Array,
    qd: Array,
    dt: float | Array,
    retract: Retract = _add,
) -> tuple[Array, Array]:
    """One linearly-implicit (Rosenbrock--Euler) step on ``(q, qd)``.

    The explicit fixed-step methods are only conditionally stable: for a stiff
    system (large gravitational or PD stiffness, small distal-link inertia) they
    blow up to NaN once ``dt`` exceeds ``~2 / omega_max`` -- and for the Panda
    that limit is small enough that even physically reasonable ``dt`` diverge,
    *independent of the forward-dynamics accuracy* (the mass matrix is
    well-conditioned; the instability is purely the integrator).

    Linearly-implicit Euler treats the dynamics implicitly through a first-order
    expansion. Writing the state ``y = [q; qd]`` with ``y' = f(y) = [qd; fd(q,qd)]``
    and ``A = df/dy``, it solves ``(I - dt A) dy = dt f(y)`` and sets
    ``y_next = y + dy`` -- a single ``2n x 2n`` linear solve, cheap for
    manipulator-sized ``n``, capturing stiffness in *both* position (gravity)
    and velocity (damping/PD).

    Stability, stated honestly: for the *linearized* dynamics this is A-stable,
    so it markedly extends the usable ``dt`` range over the explicit methods
    (empirically ~2-3x for the Panda) and does not NaN where they do. It is NOT
    unconditionally stable for the full *nonlinear* system: at very large ``dt``
    on a fast, weakly-damped system the linearization point moves too far in one
    step and the update can still overshoot into a runaway (large-but-finite)
    trajectory. It is most effective when the stabilizing forces are inside
    ``fd`` -- e.g. joint damping, or a PD law folded into the torque -- in which
    case the damping is treated implicitly and closed-loop tracking stays stable
    at ``dt`` several times the explicit limit. For genuinely coarse ``dt`` on an
    undamped system, combine it with ``substeps`` to keep the effective step
    within the system's natural timescale (~0.02 s for the Panda).

    The position part is solved in local coordinates ``eps`` around ``q``
    (``q(eps) = retract(q, eps, 1)``), so a floating base's quaternion never
    enters the linear solve: the state is ``[eps; qd]``, both ``nv`` long, and
    the fixed-base case is unchanged (``eps`` is then ``q - q0``).
    """
    n = qd.shape[-1]

    def f_stacked(y: Array) -> Array:
        eps, qd_ = y[:n], y[n:]
        return jnp.concatenate([qd_, fd(retract(q, eps, 1.0), qd_)])

    y = jnp.concatenate([jnp.zeros_like(qd), qd])
    f0 = f_stacked(y)
    A = jax.jacobian(f_stacked)(y)  # (2n, 2n)
    lhs = jnp.eye(2 * n, dtype=y.dtype) - dt * A
    dy = jnp.linalg.solve(lhs, dt * f0)
    return retract(q, dy[:n], 1.0), qd + dy[n:]


def _step_once(
    fd: Callable[[Array, Array], Array],
    q: Array,
    qd: Array,
    dt: float | Array,
    method: StepMethod,
    retract: Retract = _add,
) -> tuple[Array, Array]:
    if method == "semi_implicit":
        qd_next = qd + dt * fd(q, qd)
        return retract(q, qd_next, dt), qd_next
    if method == "linearly_implicit":
        return _linearly_implicit_step(fd, q, qd, dt, retract)
    if method == "euler":
        return retract(q, qd, dt), qd + dt * fd(q, qd)
    if method == "rk4":
        # Stages in the tangent space, one retraction per stage point (exact for 1-DOF
        # joints; the usual tangent-space RK4 for a floating base).
        k1_q, k1_qd = qd, fd(q, qd)
        k2_q = qd + 0.5 * dt * k1_qd
        k2_qd = fd(retract(q, k1_q, 0.5 * dt), k2_q)
        k3_q = qd + 0.5 * dt * k2_qd
        k3_qd = fd(retract(q, k2_q, 0.5 * dt), k3_q)
        k4_q = qd + dt * k3_qd
        k4_qd = fd(retract(q, k3_q, dt), k4_q)
        q_next = retract(q, (k1_q + 2 * k2_q + 2 * k3_q + k4_q) / 6.0, dt)
        qd_next = qd + (dt / 6.0) * (k1_qd + 2 * k2_qd + 2 * k3_qd + k4_qd)
        return q_next, qd_next
    raise ValueError(f"Unknown integration method: {method!r}")


def step_with_fd(
    fd: Callable[[Array, Array], Array],
    q: Array,
    qd: Array,
    dt: float | Array,
    method: StepMethod = "semi_implicit",
    substeps: int = 1,
    retract: Retract = _add,
) -> tuple[Array, Array]:
    """Advance ``(q, qd)`` by one step of size ``dt`` under ``qdd = fd(q, qd)``.

    ``semi_implicit`` (symplectic Euler, the MuJoCo/brax default) updates the
    velocity first and integrates positions with the *new* velocity; ``euler``
    is explicit; ``rk4`` is classic fourth-order Runge-Kutta on ``(q, qd)``.

    All three fixed-step methods can silently diverge (blow up to NaN/Inf)
    once ``dt`` is large relative to the system's stiffness -- e.g.
    semi-implicit Euler under PD torques has been observed to diverge at
    ``dt`` as small as ~0.15s. ``substeps`` (default 1, matching the prior
    behavior exactly) subdivides ``dt`` into that many equal sub-intervals
    and applies the same one-step update repeatedly, trading extra ``fd``
    evaluations for a smaller effective step and much better stability.
    """
    if substeps <= 0:
        raise ValueError(f"substeps must be a positive int, got {substeps!r}")
    if substeps == 1:
        return _step_once(fd, q, qd, dt, method, retract)
    sub_dt = dt / substeps
    # lax.fori_loop (not a Python loop) so compile cost is O(1) in substeps -- a
    # Python loop would unroll the whole RNEA `substeps` times.
    q, qd = jax.lax.fori_loop(
        0, substeps, lambda _, s: _step_once(fd, s[0], s[1], sub_dt, method, retract), (q, qd)
    )
    return q, qd
