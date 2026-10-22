"""Canonical geometric subproblems for analytic inverse kinematics.

Analytic IK for a serial arm is not one formula; it is a decomposition of the
pose constraint into a handful of elementary rotation problems that each have a
closed-form answer. These are the classical Paden-Kahan subproblems, in the
generalised least-squares form introduced by Elias & Wen's canonical
subproblem decomposition (IK-Geo).

Four subproblems are enough for every decomposition pyroffi currently needs:

===========  ==========================================  ==================
subproblem   equation                                    unknowns
===========  ==========================================  ==================
1            ``rot(k, θ) p1 = p2``                       θ
2            ``rot(k1, θ1) p1 = rot(k2, θ2) p2``         θ1, θ2
3            ``‖rot(k, θ) p1 - p2‖ = d``                 θ
4            ``h · rot(k, θ) p = d``                     θ
===========  ==========================================  ==================

Two properties matter more than the formulae:

**Least-squares fallback.** When the geometry admits no exact solution — a
target fractionally out of reach, an axis triple that does not quite close — the
naive closed form evaluates ``arccos`` outside ``[-1, 1]`` and returns NaN. Every
subproblem here instead returns the *minimiser* of the residual and raises an
``is_ls`` flag. A downstream solver can then polish or reject deliberately,
rather than propagating NaN into a task planner. This is what makes the solver
usable as a geometric oracle: a near-miss reports a near-miss.

**Fixed shapes.** Each subproblem returns a fixed-size array of candidate
angles plus a boolean validity mask, never a variable-length list, so the whole
solver stays ``jit``- and ``vmap``-safe. Invalid slots hold a finite dummy angle
(not NaN) so that gradients through unused branches stay clean.

Subproblems 2 and 3 are expressed in terms of 1 and 4 rather than re-derived,
which keeps the trigonometric edge cases in one place.

All functions take and return plain JAX arrays and are differentiable where the
geometry is smooth. ``k`` axes are assumed unit-norm; callers normalise once.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float

# Tolerance for declaring a subproblem instance exactly solvable rather than
# least-squares. Loose enough to absorb f32 FK round-off on metre-scale links.
_TOL = 1e-6


def rot(k: Float[Array, "3"], theta: Float[Array, ""]) -> Float[Array, "3 3"]:
    """Rodrigues rotation matrix about unit axis ``k`` by ``theta``."""
    K = jnp.array([
        [0.0, -k[2], k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0],
    ])
    return jnp.eye(3) + jnp.sin(theta) * K + (1.0 - jnp.cos(theta)) * (K @ K)


# --------------------------------------------------------------------------- #
# Subproblem 4:  h · rot(k, θ) p = d
# --------------------------------------------------------------------------- #

def subproblem4(
    h: Float[Array, "3"],
    p: Float[Array, "3"],
    k: Float[Array, "3"],
    d: Float[Array, ""],
) -> tuple[Float[Array, "2"], Bool[Array, "2"], Bool[Array, ""]]:
    """Solve ``h · rot(k, θ) p = d`` for θ.

    Rodrigues expands the constraint into ``A cos θ + B sin θ + C = d``, i.e. a
    single sinusoid ``R cos(θ - φ) = d - C``. There are two solutions when
    ``|d - C| <= R``, one double root at the boundary, and none beyond it — in
    which case the minimiser is the sinusoid's nearest extremum.

    Returns:
        ``(theta[2], valid[2], is_ls)``. When ``is_ls`` both entries hold the
        same minimising angle and ``valid`` is all-True: a least-squares answer
        is still an answer, just not an exact one.
    """
    kp = jnp.dot(k, p)
    hk = jnp.dot(h, k)
    A = jnp.dot(h, p) - hk * kp
    B = jnp.dot(h, jnp.cross(k, p))
    C = hk * kp

    R = jnp.hypot(A, B)
    phi = jnp.arctan2(B, A)
    rhs = d - C

    # Exact branch: two roots symmetric about phi.
    ratio = jnp.clip(jnp.where(R > _TOL, rhs / jnp.where(R > _TOL, R, 1.0), 0.0),
                     -1.0, 1.0)
    delta = jnp.arccos(ratio)
    theta_exact = jnp.stack([phi + delta, phi - delta])

    # Least-squares branch: the sinusoid never reaches rhs, so take whichever
    # extremum is closer.
    theta_ls_val = jnp.where(rhs > 0.0, phi, phi + jnp.pi)
    theta_ls = jnp.stack([theta_ls_val, theta_ls_val])

    is_ls = jnp.abs(rhs) > R + _TOL
    theta = jnp.where(is_ls, theta_ls, theta_exact)
    valid = jnp.ones((2,), dtype=bool)
    return _wrap(theta), valid, is_ls


# --------------------------------------------------------------------------- #
# Subproblem 1:  rot(k, θ) p1 = p2
# --------------------------------------------------------------------------- #

def subproblem1(
    p1: Float[Array, "3"],
    p2: Float[Array, "3"],
    k: Float[Array, "3"],
) -> tuple[Float[Array, ""], Bool[Array, ""]]:
    """Solve ``rot(k, θ) p1 = p2`` for the unique θ.

    A rotation about ``k`` preserves both the component along ``k`` and the
    norm, so an exact solution needs ``k·p1 == k·p2`` and ``‖p1‖ == ‖p2‖``. The
    angle itself is read off the components perpendicular to ``k``; when the
    consistency conditions fail that angle is still the least-squares answer,
    which is why it is computed unconditionally.

    Returns:
        ``(theta, is_ls)``.
    """
    p1_perp = p1 - k * jnp.dot(k, p1)
    p2_perp = p2 - k * jnp.dot(k, p2)

    theta = jnp.arctan2(
        jnp.dot(k, jnp.cross(p1_perp, p2_perp)),
        jnp.dot(p1_perp, p2_perp),
    )

    is_ls = (
        (jnp.abs(jnp.dot(k, p1) - jnp.dot(k, p2)) > _TOL)
        | (jnp.abs(jnp.linalg.norm(p1) - jnp.linalg.norm(p2)) > _TOL)
    )
    return _wrap(theta), is_ls


# --------------------------------------------------------------------------- #
# Subproblem 3:  ‖rot(k, θ) p1 - p2‖ = d
# --------------------------------------------------------------------------- #

def subproblem3(
    p1: Float[Array, "3"],
    p2: Float[Array, "3"],
    k: Float[Array, "3"],
    d: Float[Array, ""],
) -> tuple[Float[Array, "2"], Bool[Array, "2"], Bool[Array, ""]]:
    """Solve ``‖rot(k, θ) p1 - p2‖ = d`` for θ.

    Expanding the norm turns the distance constraint into a projection
    constraint, ``p2 · rot(k, θ) p1 = (‖p1‖² + ‖p2‖² - d²) / 2``, which is
    subproblem 4 with ``h = p2``. This is the subproblem that sets an elbow
    angle from a shoulder-to-wrist distance, so its least-squares branch is
    exactly the "target out of reach" case.
    """
    rhs = 0.5 * (jnp.dot(p1, p1) + jnp.dot(p2, p2) - d * d)
    return subproblem4(p2, p1, k, rhs)


# --------------------------------------------------------------------------- #
# Subproblem 2:  rot(k1, θ1) p1 = rot(k2, θ2) p2
# --------------------------------------------------------------------------- #

def subproblem2(
    p1: Float[Array, "3"],
    p2: Float[Array, "3"],
    k1: Float[Array, "3"],
    k2: Float[Array, "3"],
) -> tuple[Float[Array, "2"], Float[Array, "2"], Bool[Array, "2"], Bool[Array, ""]]:
    """Solve ``rot(k1, θ1) p1 = rot(k2, θ2) p2`` for the pair ``(θ1, θ2)``.

    Projecting both sides onto ``k1`` removes θ1 entirely — ``k1`` is invariant
    under ``rot(k1, ·)`` — leaving ``k1 · rot(k2, θ2) p2 = k1 · p1``, which is
    subproblem 4. Each of its (up to two) roots then fixes θ1 by subproblem 1.
    This is the two-axis wrist / shoulder pair, hence the two solution branches
    that show up as "wrist flip".

    Returns:
        ``(theta1[2], theta2[2], valid[2], is_ls)``.
    """
    theta2, valid2, is_ls4 = subproblem4(k1, p2, k2, jnp.dot(k1, p1))

    def solve_theta1(t2):
        return subproblem1(p1, rot(k2, t2) @ p2, k1)

    theta1, is_ls1 = jax.vmap(solve_theta1)(theta2)

    # Norm mismatch makes the pair unsatisfiable regardless of the angles.
    norm_mismatch = jnp.abs(jnp.linalg.norm(p1) - jnp.linalg.norm(p2)) > _TOL
    is_ls = is_ls4 | norm_mismatch | jnp.any(is_ls1)
    return _wrap(theta1), theta2, valid2, is_ls


# --------------------------------------------------------------------------- #

def _wrap(theta):
    """Map angles into ``(-pi, pi]`` so branch comparisons are unambiguous."""
    return jnp.arctan2(jnp.sin(theta), jnp.cos(theta))


# --------------------------------------------------------------------------- #
# Residuals — used by the tests and by solvers that want to rank branches.
# --------------------------------------------------------------------------- #

def residual1(p1, p2, k, theta):
    return jnp.linalg.norm(rot(k, theta) @ p1 - p2)


def residual2(p1, p2, k1, k2, theta1, theta2):
    return jnp.linalg.norm(rot(k1, theta1) @ p1 - rot(k2, theta2) @ p2)


def residual3(p1, p2, k, d, theta):
    return jnp.abs(jnp.linalg.norm(rot(k, theta) @ p1 - p2) - d)


def residual4(h, p, k, d, theta):
    return jnp.abs(jnp.dot(h, rot(k, theta) @ p) - d)
