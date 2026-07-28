"""Arc-length geometric path with its first two derivatives.

TOPP-RA never sees waypoints. It sees a curve ``q(s)`` and, at each gridpoint,
the pair ``(q'(s), q''(s))`` — those are what turn a joint-space limit into a
linear constraint on the two scalars TOPP-RA actually optimises. So everything
a sampling-based planner hands over (an ordered, variable-length, unevenly
spaced polyline) has to be converted into that form first.

Two things happen here, in this order, and the order is the whole trick for
staying jit-compatible:

1. **Resample to a fixed grid.** The polyline is measured in arc length, then
   re-sampled at ``n_grid`` points spaced uniformly in ``s``. The output shape
   no longer depends on the input waypoint count, which is what lets a padded
   ``[B, T_max, DOF]`` batch of planner outputs go through ``vmap`` unchanged.
2. **Fit a natural cubic spline to the resampled points.** Because step 1
   guarantees *uniform* knot spacing, the tridiagonal system for the spline's
   second derivatives has a constant coefficient matrix — no data-dependent
   pivoting, no degenerate knots, and ``q''`` is available in closed form.

Doing it the other way round (spline first, then resample) fails on exactly the
input we care about: planner paths routinely contain duplicated or
near-duplicated waypoints, and the non-uniform spline system is singular there.

Padding convention
------------------
A path is ``waypoints[:n_valid]``; rows at or after ``n_valid`` are ignored.
They are not masked out of the arithmetic — instead each padded segment is
assigned a fixed, large synthetic arc length (:data:`_PAD_SEGMENT_LEN`), which
pushes the padded rows far past the end of the real path in ``s``. The
resampling grid only ever spans ``[0, L]``, so it cannot reach them, and the
knot sequence stays strictly increasing so ``jnp.interp`` is unambiguous. This
costs nothing and avoids a ``where`` on every interpolation.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as onp
from jax import Array
from jaxtyping import Float, Integer

_PAD_SEGMENT_LEN = 1.0
"""Synthetic arc length given to each padded segment.

Any strictly positive constant works; it exists only to keep the knot sequence
increasing. It must not be so small that ``L + pad`` loses the separation in
float32, nor so large that it dominates the cumulative sum's magnitude.
"""

_MIN_LENGTH = 1e-9
"""Below this total arc length a path is treated as a single point."""


class GeometricPath(NamedTuple):
    """A path sampled on a uniform arc-length grid, with derivatives.

    All arrays share the leading gridpoint axis ``N``. ``qs`` and ``qss`` are
    derivatives with respect to arc length ``s``, *not* with respect to a
    normalised ``[0, 1]`` parameter — which means ``|qs| ~ 1`` and the
    constraint coefficients downstream are well scaled regardless of how long
    the path is.
    """

    q: Float[Array, "N DOF"]
    """Configurations at the gridpoints."""
    qs: Float[Array, "N DOF"]
    """dq/ds."""
    qss: Float[Array, "N DOF"]
    """d2q/ds2."""
    s: Float[Array, " N"]
    """Gridpoint arc lengths, ``linspace(0, length, N)``."""
    length: Float[Array, ""]
    """Total arc length of the path, in configuration-space units."""

    @property
    def n_grid(self) -> int:
        # Indexed from the right: a vmapped path carries a leading batch axis,
        # and the constraint builders are written to work on either rank.
        return self.q.shape[-2]

    @property
    def delta(self) -> Float[Array, ""]:
        """Uniform gridpoint spacing ``length / (N - 1)``."""
        return self.length / (self.n_grid - 1)

    @property
    def degenerate(self) -> Array:
        """True when the path has no extent, so no timing is meaningful."""
        return self.length < _MIN_LENGTH


# ---------------------------------------------------------------------------
# Uniform natural cubic spline
# ---------------------------------------------------------------------------


def _natural_spline_matrix(n: int) -> onp.ndarray:
    """Inverse of the natural-cubic-spline tridiagonal system on a uniform grid.

    Rows 1..n-2 enforce C² continuity, ``M[i-1] + 4 M[i] + M[i+1] = rhs[i]``;
    rows 0 and n-1 pin the second derivative to zero (the "natural" end
    condition). The matrix depends only on ``n``, so it is built in numpy at
    trace time and inverted once — turning the per-call spline fit into a
    single matmul.
    """
    A = onp.zeros((n, n), dtype=onp.float64)
    A[0, 0] = 1.0
    A[n - 1, n - 1] = 1.0
    idx = onp.arange(1, n - 1)
    A[idx, idx - 1] = 1.0
    A[idx, idx] = 4.0
    A[idx, idx + 1] = 1.0
    return onp.linalg.inv(A)


def _spline_derivatives(
    q: Float[Array, "N DOF"], h: Float[Array, ""]
) -> tuple[Float[Array, "N DOF"], Float[Array, "N DOF"]]:
    """First and second derivatives of the natural cubic spline through ``q``.

    ``q`` sits on a uniform grid of spacing ``h``. On segment ``i`` the spline is

        q(s_i + t) = q_i + A_i t + M_i/2 t^2 + (M_{i+1} - M_i)/(6h) t^3

    with ``M`` the knot second derivatives, so ``q''(s_i) = M_i`` directly and
    ``q'(s_i) = A_i``. The final knot's derivative is taken from the left of the
    last segment; the spline is C¹ so this agrees with the right-hand limit
    everywhere it is defined.
    """
    n = q.shape[0]
    inv = jnp.asarray(_natural_spline_matrix(n), dtype=q.dtype)

    # rhs[i] = 6 (q_{i+1} - 2 q_i + q_{i-1}) / h^2 for interior knots, 0 at ends.
    second_diff = q[2:] - 2.0 * q[1:-1] + q[:-2]
    rhs = jnp.concatenate(
        [
            jnp.zeros((1, q.shape[1]), q.dtype),
            6.0 * second_diff / (h * h),
            jnp.zeros((1, q.shape[1]), q.dtype),
        ],
        axis=0,
    )
    qss = inv @ rhs  # (N, DOF) knot second derivatives

    slope = (q[1:] - q[:-1]) / h  # (N-1, DOF)
    a = slope - h * (2.0 * qss[:-1] + qss[1:]) / 6.0  # q'(s_i), i < N-1
    last = a[-1] + h * (qss[-2] + qss[-1]) / 2.0  # q'(s_{N-1})
    qs = jnp.concatenate([a, last[None, :]], axis=0)
    return qs, qss


# ---------------------------------------------------------------------------
# Public constructor
# ---------------------------------------------------------------------------


def make_path(
    waypoints: Float[Array, "T DOF"],
    n_grid: int,
    n_valid: Integer[Array, ""] | int | None = None,
) -> GeometricPath:
    """Build a uniform arc-length :class:`GeometricPath` from a waypoint list.

    Args:
        waypoints: ``(T, DOF)`` ordered configurations. Rows from ``n_valid``
            onward are padding and are ignored (see the module docstring).
        n_grid: Number of gridpoints ``N`` in the output. Static; must be >= 4
            so the spline system is not all boundary rows. Larger ``N`` makes
            the TOPP-RA solution tighter and the scan longer — 100-200 is the
            usual range for a 7-DOF arm.
        n_valid: Number of real waypoints. May be a traced scalar. ``None``
            means all ``T`` rows are real.

    Returns:
        A :class:`GeometricPath`. For a degenerate (zero-length) path the
        derivatives are zero and ``length`` is zero; downstream code must check
        :attr:`GeometricPath.degenerate` rather than dividing by ``length``.
    """
    if n_grid < 4:
        raise ValueError(f"n_grid must be >= 4 for the spline fit, got {n_grid}")
    q = jnp.asarray(waypoints)
    if q.ndim != 2:
        raise ValueError(f"waypoints must be (T, DOF), got shape {q.shape}")
    n_wp = q.shape[0]

    if n_valid is None:
        n_valid_arr = jnp.asarray(n_wp, dtype=jnp.int32)
    else:
        n_valid_arr = jnp.asarray(n_valid, dtype=jnp.int32)
    n_valid_arr = jnp.clip(n_valid_arr, 1, n_wp)

    # Real segment lengths up to n_valid-1; synthetic length past that, which
    # keeps the knots strictly increasing without a masked interpolation.
    #
    # The norm is floored before the sqrt rather than after: duplicated
    # waypoints are common in planner output, and ``d/dx sqrt(x)`` at a
    # zero-length segment would put NaN into every gradient taken through the
    # path geometry.
    d = q[1:] - q[:-1]
    seg = jnp.sqrt(jnp.maximum(jnp.sum(d * d, axis=-1), 1e-24))  # (T-1,)
    is_real = jnp.arange(n_wp - 1) < (n_valid_arr - 1)
    seg = jnp.where(is_real, seg, _PAD_SEGMENT_LEN)
    knots = jnp.concatenate([jnp.zeros((1,), seg.dtype), jnp.cumsum(seg)])  # (T,)

    length = knots[n_valid_arr - 1]

    # Resample uniformly in s. Clamping the grid to [0, length] guarantees it
    # never reaches the padded knots.
    s_grid = jnp.linspace(0.0, 1.0, n_grid, dtype=q.dtype) * length
    q_grid = jax.vmap(lambda col: jnp.interp(s_grid, knots, col), in_axes=1, out_axes=1)(q)

    h = length / (n_grid - 1)
    # A zero-length path would divide by zero in the spline; substitute a
    # harmless spacing and zero the derivatives afterwards instead.
    safe_h = jnp.where(length < _MIN_LENGTH, 1.0, h)
    qs, qss = _spline_derivatives(q_grid, safe_h)
    alive = length >= _MIN_LENGTH
    qs = jnp.where(alive, qs, 0.0)
    qss = jnp.where(alive, qss, 0.0)

    return GeometricPath(q=q_grid, qs=qs, qss=qss, s=s_grid, length=length)


def pad_paths(
    paths: list,
    n_wp_max: int | None = None,
) -> tuple[Float[Array, "B T DOF"], Integer[Array, " B"]]:
    """Stack variable-length paths into one padded batch.

    The bridge between a sampling-based planner — which returns a different
    number of waypoints per problem — and the fixed tensor shapes JAX needs.
    Short paths are extended by repeating their final configuration, so the
    padding is a legal (if stationary) continuation of the path and cannot
    introduce a spurious jump if a consumer ignores ``n_valid``.

    Args:
        paths: Sequence of ``(T_i, DOF)`` arrays, ``T_i >= 1``.
        n_wp_max: Padded length. Defaults to the longest input. Passing a fixed
            value across calls keeps downstream jit caches warm.

    Returns:
        ``(padded, n_valid)`` with shapes ``(B, n_wp_max, DOF)`` and ``(B,)``.
    """
    if not paths:
        raise ValueError("pad_paths requires at least one path")
    arrs = [onp.asarray(p, dtype=onp.float32) for p in paths]
    for a in arrs:
        if a.ndim != 2 or a.shape[0] < 1:
            raise ValueError(f"each path must be (T, DOF) with T >= 1, got {a.shape}")
    dof = arrs[0].shape[1]
    if any(a.shape[1] != dof for a in arrs):
        raise ValueError("all paths must share the same DOF")

    longest = max(a.shape[0] for a in arrs)
    T = longest if n_wp_max is None else n_wp_max
    if T < longest:
        raise ValueError(f"n_wp_max={T} is shorter than the longest path ({longest})")

    out = onp.empty((len(arrs), T, dof), dtype=onp.float32)
    n_valid = onp.empty((len(arrs),), dtype=onp.int32)
    for b, a in enumerate(arrs):
        out[b, : a.shape[0]] = a
        out[b, a.shape[0] :] = a[-1]
        n_valid[b] = a.shape[0]
    return jnp.asarray(out), jnp.asarray(n_valid)
