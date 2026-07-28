"""Two-variable linear programs, solved by vertex enumeration.

Every step of TOPP-RA's backward pass is an LP over exactly two scalars — the
path acceleration ``u`` and the squared path velocity ``x``. Two variables is
small enough that the usual simplex/Seidel machinery is the wrong tool: those
are sequential, branchy, and their pivot order depends on the data, none of
which survives ``jit``/``vmap`` well.

Vertex enumeration instead: the optimum of a bounded 2-D LP lies at a vertex,
every vertex is the intersection of some pair of constraint lines, and the
number of pairs is a compile-time constant. So the whole solve becomes
"intersect all ``m(m-1)/2`` pairs, mask the infeasible ones, reduce". Branch-
free, fixed shape, and it gives min and max of the objective from one pass.

The cost is ``O(m^3)`` rather than the ``O(m)`` an expected-linear-time 2-D LP
achieves. For the sizes here (``m ~ 34`` for a torque-limited 7-DOF arm, so
561 candidate vertices) that is a few tens of thousands of flops per gridpoint
— dominated by the memory traffic, and utterly negligible against the batch
parallelism it buys.

Degeneracy is handled by masking, not by perturbation: near-parallel pairs have
a small determinant and are dropped, because whatever vertex they define is
either duplicated by a better-conditioned pair or lies outside the feasible set
anyway. The bounding box that callers are required to include guarantees at
least one well-conditioned feasible pair exists whenever the problem is
feasible at all.
"""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import jax.numpy as jnp
import numpy as onp
from jax import Array
from jaxtyping import Float

_DET_EPS = 1e-12
"""Pairs whose lines are more parallel than this are not treated as vertices."""


class Interval(NamedTuple):
    """Range of a coordinate over a feasible set, plus whether it is nonempty."""

    lo: Float[Array, "..."]
    hi: Float[Array, "..."]
    feasible: Array
    """Boolean. When False, ``lo``/``hi`` are meaningless placeholders."""


@lru_cache(maxsize=None)
def _pair_indices(m: int) -> tuple[onp.ndarray, onp.ndarray]:
    """Upper-triangular index pairs, cached per constraint count."""
    i, j = onp.triu_indices(m, k=1)
    return i, j


def condition(
    A: Float[Array, "m 2"],
    h: Float[Array, " m"],
    u_scale: Float[Array, ""],
    x_scale: Float[Array, ""],
) -> tuple[Float[Array, "m 2"], Float[Array, " m"]]:
    """Rescale a system so every entry is O(1), for float32 sanity.

    Two rescalings, and both are needed:

    *Columns.* ``u`` and ``x`` have unrelated units and wildly different
    magnitudes — on a fine grid a feasible ``u`` runs to hundreds while ``x``
    stays below one. Cramer's rule then subtracts two large products to get a
    small answer, and float32 has nothing left. Substituting
    ``u = u_scale * û``, ``x = x_scale * x̂`` puts both variables on ``[-1, 1]``.

    *Rows.* Torque rows carry right-hand sides of order 100 while acceleration
    rows are order 1. Normalising each row by its largest coefficient makes the
    2x2 determinants comparable across constraint types, so the
    near-parallel test means the same thing for every pair.

    Callers un-scale the result by multiplying back by ``x_scale``.
    """
    A = A * jnp.stack([u_scale, x_scale])[None, :]
    row = jnp.max(jnp.abs(A), axis=1)
    row = jnp.where(row > 0.0, row, 1.0)
    return A / row[:, None], h / row


def x_range(
    A: Float[Array, "m 2"],
    h: Float[Array, " m"],
    u_scale: Float[Array, ""],
    x_scale: Float[Array, ""],
    *,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> Interval:
    """Range of ``x`` over ``{(u, x) : A @ [u, x] <= h}``.

    The caller **must** include rows bounding both ``u`` and ``x`` on both
    sides; an unbounded feasible set has no vertex realising the extremum and
    this function will silently report the bounded hull of whatever vertices
    exist.

    Args:
        A: ``(m, 2)`` constraint normals, columns ordered ``[u, x]``.
        h: ``(m,)`` right-hand sides.
        u_scale: Magnitude of the largest ``|u|`` the box admits.
        x_scale: Magnitude of the largest ``x`` the box admits. Together with
            ``u_scale`` these precondition the system; see :func:`condition`.
        atol: Absolute slack allowed when testing a candidate vertex, applied
            in the *scaled* coordinates where everything is O(1). It must stay
            comfortably above float32 eps (1.2e-7) and cannot be replaced by
            ``rtol`` alone: a row with ``h == 0`` gets no relative slack at all,
            and the backward pass generates exactly that every time the
            successor set is a single point -- which it is at the goal, where
            the trajectory is pinned to rest. At 1e-9 every genuine vertex on
            that equality line is rejected for a ~1e-7 residual and the
            controllable set collapses to {0} for the last several gridpoints,
            stalling the trajectory just before it arrives.
        rtol: Slack proportional to the scaled ``|h|``.

    Returns:
        An :class:`Interval` over ``x``, in the original units.
    """
    A, h = condition(A, h, u_scale, x_scale)

    m = A.shape[0]
    i, j = _pair_indices(m)
    i = jnp.asarray(i)
    j = jnp.asarray(j)

    a_i, a_j = A[i], A[j]  # (P, 2)
    h_i, h_j = h[i], h[j]  # (P,)

    det = a_i[:, 0] * a_j[:, 1] - a_i[:, 1] * a_j[:, 0]
    well_posed = jnp.abs(det) > _DET_EPS
    safe_det = jnp.where(well_posed, det, 1.0)

    # Cramer's rule on the 2x2 system [a_i; a_j] @ [u, x] = [h_i, h_j].
    u = (h_i * a_j[:, 1] - h_j * a_i[:, 1]) / safe_det
    x = (a_i[:, 0] * h_j - a_j[:, 0] * h_i) / safe_det

    # Feasibility of each candidate against the full constraint set.
    resid = A[:, 0:1] * u[None, :] + A[:, 1:2] * x[None, :] - h[:, None]  # (m, P)
    slack = atol + rtol * jnp.abs(h)[:, None]
    feasible_vertex = well_posed & jnp.all(resid <= slack, axis=0)

    any_feasible = jnp.any(feasible_vertex)
    lo = jnp.min(jnp.where(feasible_vertex, x, jnp.inf)) * x_scale
    hi = jnp.max(jnp.where(feasible_vertex, x, -jnp.inf)) * x_scale
    return Interval(lo=lo, hi=hi, feasible=any_feasible)


def u_range(
    A: Float[Array, "m 2"],
    h: Float[Array, " m"],
    x: Float[Array, ""],
    u_scale: Float[Array, ""],
    x_scale: Float[Array, ""],
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> Interval:
    """Range of ``u`` with ``x`` held fixed — a 1-D LP, so no enumeration needed.

    Substituting a known ``x`` collapses each row to ``A[k,0] * u <= rhs_k``,
    which is just a running intersection of half-lines. This is the forward
    pass's inner solve, and keeping it separate from :func:`x_range` is most of
    the reason the forward pass costs so much less than the backward one.

    Rows with ``A[k,0] == 0`` constrain nothing in ``u``; they instead assert
    something about the fixed ``x``, so a violated one makes the whole slice
    infeasible and is folded into the returned flag.

    Preconditioned the same way as :func:`x_range` — the cancellation in
    ``h - A[:,1] x`` is just as damaging as the one in Cramer's rule.
    """
    A, h = condition(A, h, u_scale, x_scale)
    x = x / x_scale

    rhs = h - A[:, 1] * x
    coef = A[:, 0]

    pos = coef > atol
    neg = coef < -atol
    zero = ~(pos | neg)

    hi = jnp.min(jnp.where(pos, rhs / jnp.where(pos, coef, 1.0), jnp.inf))
    lo = jnp.max(jnp.where(neg, rhs / jnp.where(neg, coef, 1.0), -jnp.inf))

    # A zero-coefficient row is satisfied iff its rhs is nonnegative.
    zero_ok = jnp.all(jnp.where(zero, rhs >= -atol - rtol * jnp.abs(h), True))
    hi = jnp.clip(hi, -1.0, 1.0)
    lo = jnp.clip(lo, -1.0, 1.0)

    # The interval collapses to a point wherever the forward pass rides the
    # boundary of the controllable set -- which is most of a time-optimal
    # trajectory, not an edge case. In float32 the two bounds then straddle each
    # other by a few ulps and a naive `lo <= hi` reports the optimum as
    # infeasible. The test is made in the *scaled* coordinates, where both
    # bounds live in [-1, 1] and a fixed absolute tolerance is meaningful; doing
    # it after un-scaling would need a tolerance that tracks ``u_scale``, which
    # varies by orders of magnitude with the grid resolution.
    #
    # ``atol`` is set well above float32 eps (1.2e-7) on purpose: each bound is
    # a quotient of a cancelled difference, so several eps of error accumulate
    # before the comparison. Too tight and correct trajectories get reported as
    # infeasible at isolated gridpoints -- a real path in the MBM set does
    # exactly that at 1e-6 while satisfying every limit to 1e-3.
    empty = lo > hi + atol + rtol * jnp.maximum(jnp.abs(lo), jnp.abs(hi))

    # Collapse an inverted interval onto ``hi``, never onto ``lo``. The
    # direction is not cosmetic: the forward pass takes ``hi`` as its greedy
    # ``u``, so widening upwards hands back a value that violates the very
    # upper bound that inverted the interval. Doing it the wrong way round
    # makes adding a constraint *speed the trajectory up* -- on the bundled MBM
    # paths, 10 of 64 came out up to 4.7% shorter once torque limits were
    # added, which is impossible for a correct solver.
    lo = jnp.minimum(lo, hi)
    return Interval(lo=lo * u_scale, hi=hi * u_scale, feasible=zero_ok & ~empty)
