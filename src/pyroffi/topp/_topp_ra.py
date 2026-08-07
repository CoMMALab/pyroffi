"""TOPP-RA: time-optimal path parameterisation by reachability analysis.

Given a *geometric* path — the ordered configurations a sampling-based planner
returns, with no notion of when — TOPP-RA assigns the fastest timing that
respects velocity, acceleration and (optionally) torque limits. It is the piece
that turns a planner's output into something a robot can actually execute at
speed, and it is genuinely time-*optimal*, unlike the uniform-timestep retiming
in :mod:`pyroffi.toolbox._retiming` which trades 2-3x duration for simplicity.

Why reachability rather than the classical approach
---------------------------------------------------
The textbook method integrates the maximum-velocity curve forwards and
backwards and stitches the segments at switching points. Finding those
switching points is a root-find on a discontinuous function, it is famously
fragile near singular points, and it does not vectorise. TOPP-RA replaces the
search with two sweeps over a fixed grid:

**Backward pass** computes the *controllable sets* ``K_i`` — the interval of
squared path velocities at gridpoint ``i`` from which the goal state is still
reachable without ever violating a limit. Each ``K_i`` follows from ``K_{i+1}``
by one 2-D linear program, so the sweep is a ``lax.scan`` of fixed length.

**Forward pass** then runs greedily: at every gridpoint take the largest path
acceleration that keeps the next state inside ``K_{i+1}``. Because the backward
pass already guaranteed controllability, greedy is safe — there is no
backtracking and no switching-point search. It is also optimal, under a
condition on the path's curvature that raw planner output does not always meet;
see "When optimal becomes feasible and fast" below.

The consequences that matter here: fixed-shape, branch-free, and every
expensive part (the dynamics evaluations that build the constraints) happens
*before* the scans. So a batch of paths vmaps cleanly, and the torque
coefficients for the whole batch can be produced by a single GRiD kernel launch.

Discretisation
--------------
Constraints are collocated at gridpoint ``i`` and held over the interval to
``i + 1``, with the exact integration ``x_{i+1} = x_i + 2 delta u_i`` (which is
exact because ``dx/ds = 2 sddot`` by definition of ``x = sdot^2``). Collocation
means the returned trajectory can exceed the limits *between* gridpoints by
O(delta); raise ``n_grid`` if that matters, or shrink the limits slightly.

When "optimal" becomes "feasible and fast"
------------------------------------------
The greedy forward pass is provably optimal on one condition: the largest
reachable ``x_{i+1}`` must be non-decreasing in ``x_i``, so that going as fast
as possible now cannot cost speed later. Reading that off the acceleration
constraint at a single joint,

    x_{i+1} <= x_i (1 - 2 delta q''/q') + 2 delta a_max / q'

the coefficient goes **negative** once ``2 delta |q''/q'| > 1`` — past that, a
faster state at ``i`` forces a *slower* one at ``i + 1``, greedy takes the bait,
and the result is merely feasible rather than optimal.

Raw sampling-planner output crosses that line routinely. A path from pRRTC is a
polyline, a corner in a polyline has unbounded curvature, and the spline fitted
through it inherits a large ``q''`` there — refining ``n_grid`` resolves the
corner *better* rather than smoothing it, so it does not reliably help. On the
bundled MBM Panda paths the ratio reaches ~10^3 and roughly 5% of paths come out
a few percent slower than they need to be; on smooth paths (a shortcut, a
trajectory-optimised path, anything C²) the ratio stays small and the solutions
are optimal to float precision.

Two things remain true regardless, and they are the ones that matter for
running on hardware: the trajectory always satisfies every constraint it was
given, and it is always far faster than uniform-timestep retiming. If the last
few percent matter, shortcut or smooth the path before retiming it — which is
standard practice with TOPP-RA anyway.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Integer

from ._constraints import Constraints, acceleration_constraints
from ._lp import u_range, x_range
from ._path import GeometricPath, make_path

_X_CAP = 1e6
"""Last-resort stand-in for an unbounded ``x``.

Vertex enumeration needs a bounded polygon, and an infinite box row would
produce ``0 * inf`` in the determinant. Only reached when no velocity limit was
supplied at all; normally the box comes from the velocity bound.
"""

_U_CAP = 1e6
"""Last-resort stand-in for an unbounded ``u``. See :func:`_solver_scales`."""

_MIN_SPEED = 1e-6
"""Floor on ``sdot`` when converting path velocity to elapsed time."""

_SQRT_FLOOR = 1e-12
"""Below this ``x``, ``sdot`` is taken as exactly zero. See :func:`_safe_sqrt`."""


def _safe_sqrt(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """``sqrt(x)`` with a finite derivative at ``x = 0``.

    Every trajectory that starts or ends at rest has ``x = 0`` at an endpoint,
    and ``d/dx sqrt(x) -> inf`` there. A plain ``jnp.sqrt`` therefore poisons
    the whole reverse-mode gradient with NaN the moment anyone differentiates a
    duration — which is the main reason to have a pure-JAX solver at all.

    The fix is the standard double-``where``: the branch that feeds ``sqrt``
    must never see the bad value, because ``jnp.where`` still propagates NaN
    from the untaken side through the backward pass.
    """
    x = jnp.maximum(x, 0.0)
    above = x > _SQRT_FLOOR
    return jnp.where(above, jnp.sqrt(jnp.where(above, x, _SQRT_FLOOR)), 0.0)


def _solver_scales(
    constraints: Constraints, delta: Float[Array, ""]
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Bounding-box magnitudes for ``x`` and ``u``, used to precondition the LPs.

    These are not tuning knobs — they have to be genuine upper bounds, or the
    box silently truncates the feasible set, and they have to be tight, or
    float32 loses the answer to cancellation (see :func:`~pyroffi.topp._lp.condition`).

    ``x`` is bounded by the tightest velocity limit anywhere on the grid.

    ``u`` is bounded by the reachability rows themselves: the step
    ``x_{i+1} = x_i + 2 delta u`` must land inside ``[0, x_cap]`` starting from
    inside it, so ``|2 delta u| <= x_cap``. That is a much tighter and far
    better-scaled bound than any constant, and it tightens automatically as the
    grid refines.
    """
    finite = jnp.where(jnp.isfinite(constraints.x_upper), constraints.x_upper, 0.0)
    x_cap = jnp.max(finite)
    # All-infinite means no velocity limit was given anywhere.
    x_cap = jnp.where(x_cap > 0.0, x_cap, _X_CAP)

    u_cap = jnp.where(delta > 0.0, x_cap / (2.0 * jnp.maximum(delta, 1e-12)), _U_CAP)
    return x_cap, jnp.clip(u_cap, 1e-6, _U_CAP)


class TOPPResult(NamedTuple):
    """A time-parameterised trajectory on the TOPP-RA grid."""

    times: Float[Array, " N"]
    """Waypoint times in seconds, starting at 0."""
    q: Float[Array, "N DOF"]
    """Configurations — the resampled geometric path, unchanged by retiming."""
    qd: Float[Array, "N DOF"]
    """Joint velocities, ``q'(s) sdot``."""
    qdd: Float[Array, "N DOF"]
    """Joint accelerations, ``q'(s) u + q''(s) x``."""
    x: Float[Array, " N"]
    """Squared path velocity ``sdot^2`` at each gridpoint."""
    u: Float[Array, " N"]
    """Path acceleration ``sddot`` held over each interval (last entry repeats)."""
    duration: Float[Array, ""]
    """Total trajectory time in seconds."""
    feasible: Array
    """Boolean. False means some gridpoint admitted no feasible ``(u, x)``, or
    that the schedule stalls (comes to rest mid-path and cannot continue). When
    False the timing is meaningless and must not be executed."""
    length: Float[Array, ""]
    """Arc length of the underlying geometric path."""


def _augment(
    A: Float[Array, "m 2"],
    h: Float[Array, " m"],
    x_hi: Float[Array, ""],
    u_cap: Float[Array, ""],
    delta: Float[Array, ""],
    reach_lo: Float[Array, ""],
    reach_hi: Float[Array, ""],
) -> tuple[Float[Array, "m+6 2"], Float[Array, " m+6"]]:
    """Add the bounding box and the reachability half-planes to a row set.

    The six extra rows are, in order: ``x <= x_hi``, ``-x <= 0``,
    ``u <= u_cap``, ``-u <= u_cap``, and the two sides of
    ``reach_lo <= x + 2 delta u <= reach_hi``.

    The box is not optional. Vertex enumeration only sees optima that are
    intersections of constraint lines, so an unbounded direction would be
    silently truncated to whatever the other rows happen to produce.
    """
    two_delta = 2.0 * delta
    extra_A = jnp.array(
        [
            [0.0, 1.0],
            [0.0, -1.0],
            [1.0, 0.0],
            [-1.0, 0.0],
        ],
        dtype=A.dtype,
    )
    reach_A = jnp.stack(
        [
            jnp.stack([two_delta, jnp.ones_like(two_delta)]),
            jnp.stack([-two_delta, -jnp.ones_like(two_delta)]),
        ]
    ).astype(A.dtype)
    extra_h = jnp.stack(
        [
            x_hi,
            jnp.zeros_like(x_hi),
            u_cap,
            u_cap,
            reach_hi,
            -reach_lo,
        ]
    ).astype(h.dtype)
    return (
        jnp.concatenate([A, extra_A, reach_A], axis=0),
        jnp.concatenate([h, extra_h], axis=0),
    )


def _backward_pass(
    constraints: Constraints,
    delta: Float[Array, ""],
    x_end: Float[Array, ""],
    x_cap: Float[Array, ""],
    u_cap: Float[Array, ""],
) -> tuple[Float[Array, " N"], Float[Array, " N"], Array]:
    """Controllable sets ``K_i = [K_lo_i, K_hi_i]`` for every gridpoint.

    Swept from the goal backwards. ``K_{N-1}`` is pinned to the requested
    terminal state; each earlier set is the projection onto ``x`` of the states
    at ``i`` that can reach ``K_{i+1}`` in one feasible step.

    An empty ``K_i`` means the goal is unreachable under the limits — with a
    correctly built constraint set that essentially only happens when the
    endpoint velocities are inconsistent with the limits, or when ``n_grid`` is
    so coarse that collocation has distorted the problem. The infeasible flag
    is latched and propagated rather than raised, so the function stays
    jit-safe; the empty set is replaced by ``[0, 0]`` so the sweep can continue.
    """
    x_hi_grid = jnp.minimum(constraints.x_upper, x_cap)
    n = constraints.n_grid

    k_last_hi = jnp.minimum(x_end, x_hi_grid[n - 1])
    k_last_lo = jnp.minimum(x_end, k_last_hi)

    def step(carry, inputs):
        lo_next, hi_next, ok = carry
        A_i, h_i, x_hi_i = inputs
        A_aug, h_aug = _augment(A_i, h_i, x_hi_i, u_cap, delta, lo_next, hi_next)
        rng = x_range(A_aug, h_aug, u_cap, x_cap)
        lo = jnp.where(rng.feasible, rng.lo, 0.0)
        hi = jnp.where(rng.feasible, rng.hi, 0.0)
        # Enumeration tolerances can push a vertex a hair outside the box.
        lo = jnp.clip(lo, 0.0, x_hi_i)
        hi = jnp.clip(hi, lo, x_hi_i)
        return (lo, hi, ok & rng.feasible), (lo, hi)

    (_, _, ok), (lo_seq, hi_seq) = jax.lax.scan(
        step,
        (k_last_lo, k_last_hi, jnp.asarray(True)),
        (constraints.A[:-1], constraints.h[:-1], x_hi_grid[:-1]),
        reverse=True,
    )
    k_lo = jnp.concatenate([lo_seq, k_last_lo[None]])
    k_hi = jnp.concatenate([hi_seq, k_last_hi[None]])
    return k_lo, k_hi, ok


def _forward_pass(
    constraints: Constraints,
    delta: Float[Array, ""],
    k_lo: Float[Array, " N"],
    k_hi: Float[Array, " N"],
    x_start: Float[Array, ""],
    x_cap: Float[Array, ""],
    u_cap: Float[Array, ""],
) -> tuple[Float[Array, " N"], Float[Array, " N"], Array]:
    """Greedy maximum-acceleration sweep inside the controllable sets.

    Greedy is optimal *here* only because the backward pass has already
    restricted every reachable successor to a state from which the goal is
    still attainable. Take that guarantee away and maximising ``u`` locally is
    exactly the classical method's failure mode: accelerate into a state the
    path cannot decelerate out of.
    """
    x0 = jnp.clip(x_start, k_lo[0], k_hi[0])
    start_ok = (x_start <= k_hi[0] + 1e-6) & (x_start >= k_lo[0] - 1e-6)

    def step(carry, inputs):
        x_i, ok = carry
        A_i, h_i, lo_next, hi_next = inputs
        A_aug, h_aug = _augment(A_i, h_i, x_cap, u_cap, delta, lo_next, hi_next)
        rng = u_range(A_aug, h_aug, x_i, u_cap, x_cap)
        u = jnp.where(rng.feasible, rng.hi, 0.0)
        x_next = jnp.clip(x_i + 2.0 * delta * u, lo_next, hi_next)
        x_next = jnp.maximum(x_next, 0.0)
        return (x_next, ok & rng.feasible), (x_i, u)

    (x_final, ok), (x_seq, u_seq) = jax.lax.scan(
        step,
        (x0, start_ok),
        (constraints.A[:-1], constraints.h[:-1], k_lo[1:], k_hi[1:]),
    )
    x = jnp.concatenate([x_seq, x_final[None]])

    # The final gridpoint has no interval after it, so the sweep never assigned
    # it a ``u``. It still needs one: the reported ``qdd`` there is
    # ``q' u + q'' x``, and simply repeating ``u_seq[-1]`` reports an
    # acceleration the solver never checked against any constraint -- on a
    # torque-limited Panda path that alone overshoots the limit by ~11% while
    # every other gridpoint sits at 1.005.
    #
    # Solving with ``delta = 0`` degenerates the reachability rows into
    # duplicates of the box, which is exactly right: there is no successor state
    # to reach. Of the feasible interval, take the value nearest zero -- the
    # path is over, so the natural terminal control is the smallest one. For a
    # trajectory ending at rest that gives ``u = 0`` and hence ``qdd = 0``.
    A_end, h_end = _augment(
        constraints.A[-1],
        constraints.h[-1],
        x_cap,
        u_cap,
        jnp.zeros_like(delta),
        jnp.zeros_like(x_final),
        x_cap,
    )
    rng_end = u_range(A_end, h_end, x_final, u_cap, x_cap)
    u_end = jnp.where(rng_end.feasible, jnp.clip(0.0, rng_end.lo, rng_end.hi), 0.0)

    u = jnp.concatenate([u_seq, u_end[None]])
    return x, u, ok & rng_end.feasible


def solve_topp_ra(
    path: GeometricPath,
    constraints: Constraints,
    sd_start: float | Float[Array, ""] = 0.0,
    sd_end: float | Float[Array, ""] = 0.0,
) -> TOPPResult:
    """Run TOPP-RA on a prepared path and constraint set.

    The low-level entry point: use it when the constraints were built
    separately — in particular when torque coefficients came from a batched
    GRiD launch covering many paths at once, which is the whole reason the
    constraint construction is decoupled from the solve.

    Args:
        path: Geometric path on a uniform arc-length grid.
        constraints: Canonical constraints on the *same* grid.
        sd_start: Path velocity ``sdot`` at the start, in path-length units per
            second. Zero means starting from rest.
        sd_end: Path velocity at the goal.

    Returns:
        A :class:`TOPPResult`. Check ``feasible`` before trusting the timing.
    """
    if path.n_grid != constraints.n_grid:
        raise ValueError(
            f"path has {path.n_grid} gridpoints but constraints have "
            f"{constraints.n_grid}"
        )
    delta = path.delta
    x_start = jnp.asarray(sd_start) ** 2
    x_end = jnp.asarray(sd_end) ** 2

    x_cap, u_cap = _solver_scales(constraints, delta)
    k_lo, k_hi, ok_back = _backward_pass(constraints, delta, x_end, x_cap, u_cap)
    x, u, ok_fwd = _forward_pass(
        constraints, delta, k_lo, k_hi, x_start, x_cap, u_cap
    )

    sdot = _safe_sqrt(x)
    qd = path.qs * sdot[:, None]
    qdd = path.qs * u[:, None] + path.qss * x[:, None]

    # Trapezoidal in sdot: over a segment of length delta traversed at speeds
    # sdot_i -> sdot_{i+1}, dt = 2 delta / (sdot_i + sdot_{i+1}). This is the
    # exact time under constant sddot, which is what the discretisation assumes.
    speed_sum = sdot[:-1] + sdot[1:]
    dt = 2.0 * delta / jnp.maximum(speed_sum, _MIN_SPEED)
    # A degenerate (zero-length) path has delta == 0 and therefore dt == 0.
    times = jnp.concatenate([jnp.zeros((1,), dt.dtype), jnp.cumsum(dt)])

    # A segment whose two endpoints are both at rest is not traversed at all;
    # only the ``_MIN_SPEED`` floor keeps its ``dt`` finite, and the resulting
    # duration is an artefact of that floor rather than a schedule. Reporting
    # such a result as feasible is worse than reporting no result: the number
    # looks like seconds and is off by orders of magnitude. This is the symptom
    # a collapsed controllable set produces, so it is also the backstop for any
    # future regression of that kind.
    stalled = jnp.any(speed_sum <= _MIN_SPEED)

    degenerate = path.degenerate
    feasible = jnp.where(degenerate, True, ok_back & ok_fwd & ~stalled)
    return TOPPResult(
        times=times,
        q=path.q,
        qd=jnp.where(degenerate, 0.0, qd),
        qdd=jnp.where(degenerate, 0.0, qdd),
        x=x,
        u=u,
        duration=times[-1],
        feasible=feasible,
        length=path.length,
    )


def topp_ra(
    waypoints: Float[Array, "T DOF"],
    velocity_limits: Float[Array, " DOF"],
    acceleration_limits: Float[Array, " DOF"],
    *,
    n_grid: int = 128,
    n_valid: Integer[Array, ""] | int | None = None,
    sd_start: float = 0.0,
    sd_end: float = 0.0,
    extra_constraints: Constraints | None = None,
) -> TOPPResult:
    """Time-optimally retime a waypoint path under velocity/acceleration limits.

    The convenience entry point, and the one to reach for unless torque limits
    are involved. Pure JAX: jit-, vmap- and grad-compatible.

    Args:
        waypoints: ``(T, DOF)`` ordered configurations, optionally padded.
        velocity_limits: ``(DOF,)`` positive ``|qd|`` bounds, rad/s.
        acceleration_limits: ``(DOF,)`` positive ``|qdd|`` bounds, rad/s².
        n_grid: Static number of TOPP-RA gridpoints. Cost is linear in it;
            accuracy of the collocation improves with it.
        n_valid: Number of real waypoints, for padded inputs. May be traced.
        sd_start: Initial path velocity; 0 starts from rest.
        sd_end: Terminal path velocity.
        extra_constraints: Additional canonical constraints on the same
            ``n_grid`` grid — e.g. the output of
            :func:`~pyroffi.topp.torque_constraints`.

    Returns:
        A :class:`TOPPResult` of ``n_grid`` samples. Check ``feasible`` before
        using the timing — an infeasible problem returns arrays, not an error.
    """
    path = make_path(waypoints, n_grid=n_grid, n_valid=n_valid)
    cons = acceleration_constraints(path, acceleration_limits, velocity_limits)
    if extra_constraints is not None:
        cons = cons.merge(extra_constraints)
    return solve_topp_ra(path, cons, sd_start=sd_start, sd_end=sd_end)


def topp_ra_batched(
    waypoints: Float[Array, "B T DOF"],
    velocity_limits: Float[Array, " DOF"],
    acceleration_limits: Float[Array, " DOF"],
    *,
    n_grid: int = 128,
    n_valid: Integer[Array, " B"] | None = None,
    sd_start: float = 0.0,
    sd_end: float = 0.0,
    extra_constraints: Constraints | None = None,
) -> TOPPResult:
    """:func:`topp_ra` over a padded batch of paths, in one vmapped solve.

    ``extra_constraints``, if given, must already carry the batch axis
    (``A`` of shape ``[B, n_grid, m, 2]``) — build it once for the flattened
    ``B * n_grid`` states so a GPU dynamics backend gets a single launch.

    Returns:
        A :class:`TOPPResult` whose fields all carry a leading ``B`` axis.
        ``feasible`` is per-path: check it and drop the failures rather than
        assuming the batch succeeded as a whole.
    """
    def _one(wp, nv, extra):
        return topp_ra(
            wp,
            velocity_limits,
            acceleration_limits,
            n_grid=n_grid,
            n_valid=nv,
            sd_start=sd_start,
            sd_end=sd_end,
            extra_constraints=extra,
        )

    # ``None`` operands are mapped over axis ``None`` so vmap leaves them alone;
    # the limits are closed over rather than passed, since they are shared.
    in_axes = (0, None if n_valid is None else 0, None if extra_constraints is None else 0)
    return jax.vmap(_one, in_axes=in_axes)(waypoints, n_valid, extra_constraints)


def sample_at_times(
    result: TOPPResult,
    query_times: Float[Array, " M"],
) -> tuple[Float[Array, "M DOF"], Float[Array, "M DOF"], Float[Array, "M DOF"]]:
    """Resample a result onto arbitrary times, e.g. a controller's fixed period.

    TOPP-RA's output grid is uniform in *arc length*, not in time — gridpoints
    bunch up in time wherever the trajectory slows down. A servo loop wants the
    opposite, so this interpolates ``(q, qd, qdd)`` onto whatever time grid the
    controller runs at. Linear interpolation is used throughout: the
    discretisation already assumes piecewise-constant ``sddot``, so a
    higher-order interpolant would add smoothness the solution never claimed.

    Queries outside ``[0, duration]`` clamp to the endpoints.
    """
    t = jnp.clip(jnp.asarray(query_times), result.times[0], result.times[-1])

    def _interp(mat):
        return jax.vmap(
            lambda col: jnp.interp(t, result.times, col), in_axes=1, out_axes=1
        )(mat)

    return _interp(result.q), _interp(result.qd), _interp(result.qdd)
