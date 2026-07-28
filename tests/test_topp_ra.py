"""Tests for TOPP-RA time-optimal path parameterisation.

There is no reference implementation to diff against in this environment, so
correctness is pinned three ways:

* **Closed form.** A straight-line path under symmetric velocity/acceleration
  limits has an exactly known bang-bang solution, so the duration, the peak
  speed, and the switching structure can be checked against analysis rather
  than against a golden file.
* **Invariants that only the optimum satisfies.** A time-optimal solution
  saturates at least one constraint at (almost) every instant, obeys the
  scaling law ``duration(k v, k^2 a) = duration(v, a) / k`` exactly, and is
  never slower than the uniform-timestep retiming it replaces.
* **Structural equivalence.** Padded and unpadded inputs, batched and
  per-path solves, and the pure-JAX and GRiD dynamics backends must agree.

Tolerances are sized for the process-default float32.
"""

import jax
import jax.numpy as jnp
import numpy as onp
import pytest
import yourdfpy

import pyroffi
from pyroffi import topp
from pyroffi.toolbox._retiming import retime_path

PANDA_URDF = "resources/panda/panda_spherized.urdf"

# Collocation holds the constraints at gridpoints only, so the interpolated
# trajectory may exceed a limit between them by O(delta). 1% at n_grid=128.
LIMIT_TOL = 1.02


def _has_gpu() -> bool:
    return any(d.platform == "gpu" for d in jax.devices())


@pytest.fixture(scope="module")
def panda():
    urdf = yourdfpy.URDF.load(PANDA_URDF, load_meshes=False)
    return urdf, pyroffi.Robot.from_urdf(urdf)


def _random_paths(key, batch, n_wp, dof, scale=1.0):
    return jax.random.uniform(key, (batch, n_wp, dof), minval=-scale, maxval=scale)


# ---------------------------------------------------------------------------
# Path geometry
# ---------------------------------------------------------------------------


def test_make_path_grid_is_uniform_in_arc_length():
    # Deliberately uneven spacing: the resampler has to redistribute it.
    wp = jnp.asarray([[0.0], [0.1], [0.15], [2.0], [3.0]])
    path = topp.make_path(wp, n_grid=64)

    assert float(path.length) == pytest.approx(3.0, rel=1e-5)
    step = jnp.diff(path.s)
    assert onp.allclose(step, step[0], rtol=1e-4)
    assert float(path.delta) == pytest.approx(3.0 / 63, rel=1e-5)
    # Monotone along a monotone path, and the endpoints are interpolated exactly.
    assert onp.all(onp.diff(onp.asarray(path.q[:, 0])) >= -1e-6)
    assert float(path.q[0, 0]) == pytest.approx(0.0, abs=1e-5)
    assert float(path.q[-1, 0]) == pytest.approx(3.0, rel=1e-5)


def test_spline_derivatives_match_finite_differences():
    # A smooth curve sampled densely: the spline's q' and q'' should agree with
    # central differences of its own knot values.
    # The input is sampled far more densely than the TOPP-RA grid on purpose.
    # ``make_path`` resamples by *linear* interpolation, and a second difference
    # amplifies the resulting jitter by 1/h^2 -- with only 200 input points that
    # aliasing, not the spline, dominates the comparison (it is ~20x larger).
    t = jnp.linspace(0.0, 1.0, 2000)
    wp = jnp.stack([jnp.sin(3.0 * t), jnp.cos(2.0 * t), t**2], axis=-1)
    path = topp.make_path(wp, n_grid=128)
    h = path.delta

    fd1 = (path.q[2:] - path.q[:-2]) / (2 * h)
    fd2 = (path.q[2:] - 2 * path.q[1:-1] + path.q[:-2]) / h**2
    assert onp.allclose(path.qs[1:-1], fd1, atol=2e-3)

    # The "natural" end condition pins q'' to zero at both ends, which is wrong
    # for any curve with real curvature there. That error decays over a few
    # knots, so the second derivative is only compared away from the boundary.
    assert onp.allclose(path.qss[6:-6], fd2[5:-5], atol=1e-2)


def test_straight_line_path_has_unit_speed_and_no_curvature():
    wp = jnp.asarray([[0.0, 0.0], [3.0, 4.0]])  # length 5
    path = topp.make_path(wp, n_grid=32)
    assert float(path.length) == pytest.approx(5.0, rel=1e-5)
    # Arc-length parameterisation means |q'| == 1 by construction.
    assert onp.allclose(jnp.linalg.norm(path.qs, axis=-1), 1.0, atol=1e-4)
    assert onp.allclose(path.qss, 0.0, atol=1e-4)


def test_degenerate_path_is_handled():
    """A path of duplicated waypoints has no extent and must not divide by zero."""
    wp = jnp.zeros((5, 3))
    path = topp.make_path(wp, n_grid=16)
    assert bool(path.degenerate)
    assert onp.all(onp.isfinite(onp.asarray(path.qs)))

    result = topp.topp_ra(wp, jnp.ones(3), jnp.ones(3), n_grid=16)
    assert bool(result.feasible)
    assert float(result.duration) == pytest.approx(0.0, abs=1e-6)
    assert onp.all(onp.isfinite(onp.asarray(result.times)))


def test_make_path_rejects_tiny_grid():
    with pytest.raises(ValueError, match="n_grid"):
        topp.make_path(jnp.zeros((4, 2)), n_grid=3)


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------


def test_pad_paths_repeats_final_configuration():
    paths = [onp.zeros((3, 2)), onp.ones((5, 2)), onp.full((2, 2), 7.0)]
    padded, n_valid = topp.pad_paths(paths)

    assert padded.shape == (3, 5, 2)
    assert onp.array_equal(onp.asarray(n_valid), [3, 5, 2])
    # Rows past n_valid repeat the last real waypoint, so a consumer that
    # ignores n_valid still sees a stationary continuation, not a jump.
    assert onp.allclose(padded[0, 3:], padded[0, 2])
    assert onp.allclose(padded[2, 2:], 7.0)


def test_pad_paths_rejects_too_small_n_wp_max():
    with pytest.raises(ValueError, match="shorter than the longest"):
        topp.pad_paths([onp.zeros((6, 2))], n_wp_max=4)


def test_padding_does_not_change_the_solution():
    """The whole point of ``n_valid``: padding must be invisible to the solve."""
    wp = jnp.asarray([[0.0, 0.0], [1.0, 0.5], [2.0, -0.5], [2.5, 0.25]])
    padded = jnp.concatenate([wp, jnp.repeat(wp[-1:], 6, axis=0)], axis=0)
    vmax, amax = jnp.ones(2), jnp.ones(2) * 2.0

    ref = topp.topp_ra(wp, vmax, amax, n_grid=96)
    pad = topp.topp_ra(padded, vmax, amax, n_grid=96, n_valid=4)

    assert float(pad.length) == pytest.approx(float(ref.length), rel=1e-5)
    assert float(pad.duration) == pytest.approx(float(ref.duration), rel=1e-4)
    assert onp.allclose(pad.q, ref.q, atol=1e-5)


def test_ignoring_n_valid_would_change_the_answer():
    """Guards against ``n_valid`` being silently dropped: padding must matter."""
    wp = jnp.asarray([[0.0], [1.0], [2.0]])
    padded = jnp.concatenate([wp, jnp.full((5, 1), 2.0)], axis=0)
    # Padding rows are stationary, so length is unchanged either way; use a
    # padding value that is *not* the last waypoint to make the difference real.
    mislabelled = padded.at[3:].set(9.0)
    honest = topp.topp_ra(mislabelled, jnp.ones(1), jnp.ones(1), n_grid=64, n_valid=3)
    naive = topp.topp_ra(mislabelled, jnp.ones(1), jnp.ones(1), n_grid=64)
    assert float(honest.length) == pytest.approx(2.0, rel=1e-4)
    assert float(naive.length) > 2.5


# ---------------------------------------------------------------------------
# Closed-form optimality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_grid", [64, 128, 256])
def test_straight_line_matches_analytic_bang_bang(n_grid):
    """Triangular profile: D=1, a=1, vmax=1 exactly reaches vmax at midpoint.

    Accelerate at ``a`` for ``t1 = sqrt(D/a) = 1`` covering ``D/2``, then
    decelerate symmetrically. Total ``2 sqrt(D/a) = 2``. Peak speed is
    ``sqrt(a D) = 1``, exactly the velocity limit, so this case pins the
    acceleration branch without the velocity limit truncating it.
    """
    wp = jnp.asarray([[0.0], [1.0]])
    r = topp.topp_ra(wp, jnp.array([1.0]), jnp.array([1.0]), n_grid=n_grid)

    assert bool(r.feasible)
    assert float(r.duration) == pytest.approx(2.0, rel=2e-3)
    assert float(jnp.max(jnp.abs(r.qd))) == pytest.approx(1.0, rel=1e-2)
    # Rest at both ends.
    assert float(jnp.abs(r.qd[0, 0])) < 1e-5
    assert float(jnp.abs(r.qd[-1, 0])) < 1e-5


def test_straight_line_with_cruise_phase():
    """Trapezoidal profile: D=10, a=1, vmax=1 -> t = D/v + v/a = 11."""
    wp = jnp.asarray([[0.0], [10.0]])
    r = topp.topp_ra(wp, jnp.array([1.0]), jnp.array([1.0]), n_grid=256)

    assert bool(r.feasible)
    assert float(r.duration) == pytest.approx(11.0, rel=2e-3)
    # A genuine cruise phase: most of the path runs at exactly vmax.
    at_vmax = onp.abs(onp.asarray(r.qd[:, 0])) > 0.999
    assert at_vmax.mean() > 0.7


def test_time_scaling_law():
    """Scaling ``v -> k v`` and ``a -> k^2 a`` scales duration by exactly ``1/k``.

    This is a property of the *optimal* solution, not of any feasible one: it
    holds because the substitution ``t -> t/k`` maps the feasible set onto
    itself. A suboptimal solver generally breaks it.
    """
    wp = jnp.asarray([[0.0, 0.0, 0.0], [0.5, 0.3, -0.2], [1.0, -0.4, 0.6]])
    vmax, amax = jnp.ones(3), jnp.ones(3) * 2.0
    base = topp.topp_ra(wp, vmax, amax, n_grid=128)
    for k in (2.0, 3.5):
        scaled = topp.topp_ra(wp, k * vmax, k * k * amax, n_grid=128)
        assert float(scaled.duration) == pytest.approx(
            float(base.duration) / k, rel=1e-4
        )


def test_faster_than_uniform_timestep_retiming():
    """TOPP-RA must decisively beat the uniform-timestep method it replaces.

    Compared on a smooth, densely sampled path, so both methods see the same
    geometry — the spline's arc length matches the polyline's to 7 digits here.
    That matters: on a *jagged* path the two are not solving the same problem.
    The uniform method treats a corner as an instantaneous direction change and
    never charges for it, while TOPP-RA sees the spline's real (large)
    curvature there and slows down for it, which can make TOPP-RA's duration
    the longer of the two. The honest comparison is this one.
    """
    t = onp.linspace(0.0, 1.0, 200)
    wp = onp.stack(
        [
            onp.sin(2 * t),
            0.7 * onp.cos(1.5 * t) - 0.7,
            0.5 * t**2,
            -0.4 * onp.sin(3 * t),
            0.3 * t,
            0.2 * onp.cos(t) - 0.2,
        ],
        axis=-1,
    )
    vmax = onp.ones(6) * 1.5
    amax = onp.ones(6) * 3.0

    uniform = retime_path(wp, vmax, amax)
    optimal = topp.topp_ra(
        jnp.asarray(wp, jnp.float32),
        jnp.asarray(vmax, jnp.float32),
        jnp.asarray(amax, jnp.float32),
        n_grid=200,
    )

    assert bool(optimal.feasible)
    assert uniform.feasible
    # The uniform step is set by the tightest junction on the whole path, so it
    # leaves most joints far below their limits; several-fold is expected.
    assert float(optimal.duration) < 0.5 * uniform.duration
    assert float(jnp.max(jnp.abs(optimal.qd) / 1.5)) <= LIMIT_TOL
    assert float(jnp.max(jnp.abs(optimal.qdd) / 3.0)) <= LIMIT_TOL


# ---------------------------------------------------------------------------
# Limit satisfaction and saturation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_limits_respected_on_random_paths(seed):
    dof = 7
    wp = _random_paths(jax.random.PRNGKey(seed), 1, 6, dof)[0]
    vmax = jnp.ones(dof) * 2.0
    amax = jnp.ones(dof) * 5.0
    r = topp.topp_ra(wp, vmax, amax, n_grid=128)

    assert bool(r.feasible)
    assert float(jnp.max(jnp.abs(r.qd) / vmax)) <= LIMIT_TOL
    assert float(jnp.max(jnp.abs(r.qdd) / amax)) <= LIMIT_TOL
    assert float(r.duration) > 0.0
    assert onp.all(onp.diff(onp.asarray(r.times)) >= -1e-9)


def test_solution_saturates_a_constraint_almost_everywhere():
    """The signature of time-optimality: something is always at its limit.

    A feasible-but-slow schedule can sit strictly inside every constraint. The
    optimum cannot -- at each gridpoint either the velocity bound or an
    acceleration bound is active.
    """
    wp = _random_paths(jax.random.PRNGKey(21), 1, 6, 5)[0]
    vmax, amax = jnp.ones(5) * 2.0, jnp.ones(5) * 4.0
    r = topp.topp_ra(wp, vmax, amax, n_grid=128)

    v_active = jnp.max(jnp.abs(r.qd) / vmax, axis=-1) > 0.99
    a_active = jnp.max(jnp.abs(r.qdd) / amax, axis=-1) > 0.99
    active = onp.asarray(v_active | a_active)
    # Endpoints are pinned at rest and need not saturate anything.
    assert active[1:-1].mean() > 0.95


def test_tighter_limits_never_shorten_the_trajectory():
    wp = _random_paths(jax.random.PRNGKey(5), 1, 7, 4)[0]
    amax = jnp.ones(4) * 4.0
    durations = [
        float(topp.topp_ra(wp, jnp.ones(4) * v, amax, n_grid=128).duration)
        for v in (0.5, 1.0, 2.0, 4.0)
    ]
    assert durations == sorted(durations, reverse=True)


def test_nonzero_terminal_velocity_shortens_the_trajectory():
    wp = jnp.asarray([[0.0], [2.0]])
    at_rest = topp.topp_ra(wp, jnp.array([1.0]), jnp.array([1.0]), n_grid=256)
    flying = topp.topp_ra(
        wp, jnp.array([1.0]), jnp.array([1.0]), n_grid=256, sd_end=1.0
    )
    assert float(flying.duration) < float(at_rest.duration)
    # Ending at sdot = 1 on a unit-speed path means |qd| = 1 at the goal.
    assert float(jnp.abs(flying.qd[-1, 0])) == pytest.approx(1.0, rel=1e-2)


def test_controllable_set_does_not_collapse_near_the_goal():
    """Regression: the deceleration ramp used to stall just short of the goal.

    The terminal controllable set is the single point ``x = 0``, so the backward
    pass emits reachability rows with a right-hand side of exactly zero. A
    relative feasibility tolerance gives such a row no slack at all, and with a
    1e-9 absolute floor every genuine vertex on that equality line was rejected
    for a float32-sized residual. ``K_hi`` then read zero for the last several
    gridpoints, the trajectory was forced to a standstill before arriving, and
    the duration blew up by five orders of magnitude -- while still reporting
    itself feasible, which is the part that makes it worth a test.
    """
    dof = 7
    t = jnp.linspace(0.0, 1.0, 300)
    phase = jax.random.uniform(
        jax.random.PRNGKey(0), (dof,), minval=0.0, maxval=2 * onp.pi
    )
    freq = jax.random.uniform(jax.random.PRNGKey(1), (dof,), minval=0.5, maxval=2.5)
    wp = 0.6 * jnp.sin(freq * t[:, None] * onp.pi + phase)

    r = topp.topp_ra(wp, jnp.ones(dof) * 2.0, jnp.full((dof,), 10.0), n_grid=128)
    assert bool(r.feasible)

    # The path is ~6 rad long at up to 2 rad/s, so a few seconds. Anything
    # near-stationary in the middle shows up as an absurd duration.
    assert float(r.duration) < 30.0
    # Only the two endpoints are pinned to rest.
    assert int(onp.sum(onp.asarray(r.x) < 1e-6)) <= 2
    assert float(jnp.max(jnp.diff(r.times))) < 1.0


def test_final_gridpoint_obeys_the_constraints():
    """Regression: the last gridpoint used to inherit its neighbour's ``u``.

    The forward sweep assigns one ``u`` per *interval*, and there is no interval
    after the final gridpoint. Copying ``u[-2]`` into ``u[-1]`` reported an
    acceleration nothing had ever checked -- on a torque-limited Panda path that
    single point overshot the limit by 11% while every other gridpoint sat at
    1.005. A path ending at rest must end with zero acceleration.
    """
    wp = _random_paths(jax.random.PRNGKey(97), 1, 6, 5)[0]
    vmax, amax = jnp.ones(5) * 2.0, jnp.ones(5) * 4.0
    r = topp.topp_ra(wp, vmax, amax, n_grid=128)

    assert bool(r.feasible)
    assert float(jnp.max(jnp.abs(r.qdd[-1]))) == pytest.approx(0.0, abs=1e-5)
    assert float(r.u[-1]) == pytest.approx(0.0, abs=1e-5)
    # The limit check must hold at *every* gridpoint, the last one included.
    assert float(jnp.max(jnp.abs(r.qdd[-1]) / amax)) <= LIMIT_TOL


def test_infeasible_problem_is_flagged_not_raised():
    """Gravity-free analogue: an acceleration limit of zero admits no motion."""
    wp = jnp.asarray([[0.0], [1.0]])
    r = topp.topp_ra(
        wp, jnp.array([1.0]), jnp.array([1.0]), n_grid=32, sd_start=5.0
    )
    # Starting at a speed the path cannot decelerate from within its length.
    assert not bool(r.feasible)


# ---------------------------------------------------------------------------
# Torque constraints
# ---------------------------------------------------------------------------


def test_torque_coefficients_reproduce_inverse_dynamics(panda):
    """``tau = a u + b x + c`` must match RNEA evaluated at the same state.

    This is the identity the three-RNEA trick relies on; if the ``c``
    subtraction were wrong the constraint would still look plausible.
    """
    _, robot = panda
    dof = robot.dynamics.num_dof
    wp = _random_paths(jax.random.PRNGKey(31), 1, 5, dof)[0]
    path = topp.make_path(wp, n_grid=48)
    idfn = topp.jax_inverse_dynamics_fn(robot)

    zeros = jnp.zeros_like(path.q)
    c = idfn(path.q, zeros, zeros)
    a = idfn(path.q, zeros, path.qs) - c
    b = idfn(path.q, path.qs, path.qss) - c

    u, x = 0.7, 1.3
    predicted = a * u + b * x + c
    qd = path.qs * jnp.sqrt(x)
    qdd = path.qs * u + path.qss * x
    actual = robot.inverse_dynamics(path.q, qd, qdd)

    # Scored against the magnitude of the torque vector rather than
    # element-wise: ``a`` and ``b`` are differences of float32 RNEA outputs of
    # order 100, so small entries carry the absolute error of the large ones.
    scale = float(jnp.max(jnp.abs(actual)))
    assert float(jnp.max(jnp.abs(predicted - actual))) < 1e-3 * scale


def test_torque_limits_bind_and_are_respected(panda):
    urdf, robot = panda
    dof = robot.dynamics.num_dof
    effort = jnp.asarray([float(j.limit.effort) for j in urdf.actuated_joints])
    vmax = jnp.asarray([float(j.limit.velocity) for j in urdf.actuated_joints])
    amax = jnp.ones(dof) * 3.0

    wp = _random_paths(jax.random.PRNGKey(3), 1, 5, dof)[0]
    path = topp.make_path(wp, n_grid=100)
    idfn = topp.jax_inverse_dynamics_fn(robot)

    unconstrained = topp.topp_ra(wp, vmax, amax, n_grid=100)

    # Loose enough that gravity alone is satisfiable, tight enough to bind.
    tau_lim = effort * 0.4
    cons = topp.torque_constraints(path, idfn, -tau_lim, tau_lim)
    constrained = topp.topp_ra(wp, vmax, amax, n_grid=100, extra_constraints=cons)

    assert bool(constrained.feasible)
    assert float(constrained.duration) > float(unconstrained.duration)

    tau = robot.inverse_dynamics(
        constrained.q, constrained.qd, constrained.qdd
    )
    ratio = float(jnp.max(jnp.abs(tau) / tau_lim))
    assert ratio <= LIMIT_TOL
    assert ratio > 0.9, "torque limit should actually be active"


def test_torque_limits_below_gravity_are_infeasible(panda):
    """A limit that gravity alone violates has no feasible (u, x) at all."""
    urdf, robot = panda
    dof = robot.dynamics.num_dof
    wp = _random_paths(jax.random.PRNGKey(3), 1, 5, dof)[0]
    path = topp.make_path(wp, n_grid=64)
    idfn = topp.jax_inverse_dynamics_fn(robot)

    gravity_torque = robot.inverse_dynamics(
        path.q, jnp.zeros_like(path.q), jnp.zeros_like(path.q)
    )
    tau_lim = jnp.max(jnp.abs(gravity_torque), axis=0) * 0.5
    cons = topp.torque_constraints(path, idfn, -tau_lim, tau_lim)
    r = topp.topp_ra(
        wp,
        jnp.ones(dof) * 2.0,
        jnp.ones(dof) * 3.0,
        n_grid=64,
        extra_constraints=cons,
    )
    assert not bool(r.feasible)


# ---------------------------------------------------------------------------
# Batching, jit, vmap, grad
# ---------------------------------------------------------------------------


def test_batched_matches_per_path_solves():
    dof = 6
    wps = _random_paths(jax.random.PRNGKey(41), 8, 7, dof)
    vmax, amax = jnp.ones(dof) * 1.5, jnp.ones(dof) * 4.0

    batched = topp.topp_ra_batched(wps, vmax, amax, n_grid=96)
    for b in range(8):
        single = topp.topp_ra(wps[b], vmax, amax, n_grid=96)
        # vmap reassociates the float32 reductions, so the two agree to a few
        # parts in 10^5 rather than bit-for-bit.
        assert float(batched.duration[b]) == pytest.approx(
            float(single.duration), rel=1e-4
        )
        assert onp.allclose(batched.qd[b], single.qd, atol=1e-4)


def test_batched_with_padding_and_per_path_n_valid():
    rng = onp.random.default_rng(0)
    paths = [rng.normal(size=(n, 5)).cumsum(0) for n in (3, 9, 5, 4)]
    padded, n_valid = topp.pad_paths(paths)
    vmax, amax = jnp.ones(5) * 2.0, jnp.ones(5) * 5.0

    batched = topp.topp_ra_batched(padded, vmax, amax, n_grid=96, n_valid=n_valid)
    assert batched.duration.shape == (4,)
    assert bool(jnp.all(batched.feasible))

    for b, p in enumerate(paths):
        single = topp.topp_ra(jnp.asarray(p, jnp.float32), vmax, amax, n_grid=96)
        assert float(batched.duration[b]) == pytest.approx(
            float(single.duration), rel=1e-3
        )


def test_jit_and_vmap_compatible():
    dof = 4
    wps = _random_paths(jax.random.PRNGKey(51), 5, 6, dof)
    vmax, amax = jnp.ones(dof), jnp.ones(dof) * 2.0

    fn = jax.jit(lambda w: topp.topp_ra(w, vmax, amax, n_grid=64).duration)
    eager = topp.topp_ra(wps[0], vmax, amax, n_grid=64).duration
    # jit fuses differently than the op-by-op path, so float32 results differ
    # in the last few bits; this checks equivalence, not bit-identity.
    assert float(fn(wps[0])) == pytest.approx(float(eager), rel=1e-4)

    vmapped = jax.jit(jax.vmap(fn))(wps)
    assert vmapped.shape == (5,)
    assert onp.all(onp.asarray(vmapped) > 0.0)


def test_duration_is_differentiable_wrt_waypoints():
    """The whole pipeline is pure JAX, so a duration cost can be optimised.

    Only checks that the gradient exists and is informative -- the LP's argmax
    is piecewise constant, so the gradient flows through the path geometry and
    the reachable-set boundaries, not through the vertex selection.
    """
    dof = 3
    wp = _random_paths(jax.random.PRNGKey(61), 1, 5, dof)[0]
    vmax, amax = jnp.ones(dof), jnp.ones(dof) * 2.0

    g = jax.grad(lambda w: topp.topp_ra(w, vmax, amax, n_grid=48).duration)(wp)
    assert g.shape == wp.shape
    assert onp.all(onp.isfinite(onp.asarray(g)))
    assert float(jnp.max(jnp.abs(g))) > 0.0


def test_sample_at_times_reproduces_the_trajectory():
    wp = _random_paths(jax.random.PRNGKey(71), 1, 6, 4)[0]
    r = topp.topp_ra(wp, jnp.ones(4), jnp.ones(4) * 2.0, n_grid=128)

    q, qd, qdd = topp.sample_at_times(r, r.times)
    assert onp.allclose(q, r.q, atol=1e-4)
    assert onp.allclose(qd, r.qd, atol=1e-4)

    # A controller's fixed period, and out-of-range queries clamped.
    dense = jnp.arange(0.0, float(r.duration) + 0.5, 0.01)
    q2, qd2, _ = topp.sample_at_times(r, dense)
    assert q2.shape == (dense.shape[0], 4)
    assert onp.allclose(q2[-1], r.q[-1], atol=1e-4)
    assert float(jnp.max(jnp.abs(qd2))) <= LIMIT_TOL


# ---------------------------------------------------------------------------
# Real planner output
# ---------------------------------------------------------------------------

MBM_PATHS = "resources/panda/topp_paths.npz"


@pytest.mark.skipif(
    not __import__("os").path.exists(MBM_PATHS),
    reason="MBM path bundle not present; see examples/19_00_batched_topp_ra.py",
)
def test_batched_solve_on_bundled_mbm_paths(panda):
    """End-to-end on real pRRTC output: padded, jagged, variable length.

    Synthetic paths are smooth and well conditioned in a way planner output is
    not — the bundle contains duplicated waypoints and corners whose spline
    curvature exceeds the acceleration limit, which is where the solver's
    numerics actually get tested.
    """
    urdf, robot = panda
    data = onp.load(MBM_PATHS, allow_pickle=True)
    waypoints = jnp.asarray(data["waypoints"], dtype=jnp.float32)
    n_valid = jnp.asarray(data["n_valid"], dtype=jnp.int32)
    batch, _, dof = waypoints.shape

    vmax = jnp.asarray([float(j.limit.velocity) for j in urdf.actuated_joints])
    effort = jnp.asarray([float(j.limit.effort) for j in urdf.actuated_joints])
    amax = jnp.full((dof,), 10.0)

    r = topp.topp_ra_batched(waypoints, vmax, amax, n_grid=128, n_valid=n_valid)
    assert bool(jnp.all(r.feasible)), "every bundled path should retime"
    assert float(jnp.max(jnp.abs(r.qd) / vmax)) <= LIMIT_TOL
    assert float(jnp.max(jnp.abs(r.qdd) / amax)) <= LIMIT_TOL
    assert onp.all(onp.asarray(r.duration) > 0.0)

    # With torque limits on top.
    paths = jax.vmap(lambda w, n: topp.make_path(w, 128, n))(waypoints, n_valid)
    cons = topp.torque_constraints(
        paths, topp.jax_inverse_dynamics_fn(robot), -0.8 * effort, 0.8 * effort
    )
    rt = topp.topp_ra_batched(
        waypoints, vmax, amax, n_grid=128, n_valid=n_valid, extra_constraints=cons
    )
    assert bool(jnp.all(rt.feasible))
    tau = robot.inverse_dynamics(rt.q, rt.qd, rt.qdd)
    assert float(jnp.max(jnp.abs(tau) / (0.8 * effort))) <= LIMIT_TOL

    # Adding a constraint should only slow the trajectory down, and mostly it
    # does -- but greedy TOPP-RA is optimal only where the reachable map is
    # monotone in x, and raw polyline corners violate that badly enough that a
    # handful of these paths come out a few percent *faster* once torque limits
    # nudge them off a bad state. See the "When optimal becomes feasible and
    # fast" section of pyroffi.topp._topp_ra. What must hold unconditionally is
    # that no path gains materially, and that the vast majority behave.
    deficit = (onp.asarray(r.duration) - onp.asarray(rt.duration)) / onp.asarray(
        r.duration
    )
    assert deficit.max() < 0.10
    assert (deficit > 1e-3).mean() < 0.25
    assert batch >= 8


def test_added_constraints_only_slow_smooth_paths_down():
    """Monotonicity in the constraint set, on paths where greedy *is* optimal.

    The counterpart to the tolerance in the MBM test above: with a C² path the
    reachable map stays monotone, and tightening the torque limit must increase
    every duration, to float precision. If this ever fails the solver is wrong,
    not merely discretisation-limited.
    """
    urdf = yourdfpy.URDF.load(PANDA_URDF, load_meshes=False)
    robot = pyroffi.Robot.from_urdf(urdf)
    dof = robot.dynamics.num_dof

    effort = jnp.asarray([float(j.limit.effort) for j in urdf.actuated_joints])
    vmax = jnp.asarray([float(j.limit.velocity) for j in urdf.actuated_joints])
    amax = jnp.full((dof,), 10.0)

    t = jnp.linspace(0.0, 1.0, 300)
    phase = jax.random.uniform(
        jax.random.PRNGKey(0), (8, dof), minval=0.0, maxval=2 * onp.pi
    )
    freq = jax.random.uniform(
        jax.random.PRNGKey(1), (8, dof), minval=0.5, maxval=2.5
    )
    wps = 0.6 * jnp.sin(freq[:, None, :] * t[None, :, None] * onp.pi + phase[:, None, :])

    paths = jax.vmap(lambda w: topp.make_path(w, 128))(wps)
    id_fn = topp.jax_inverse_dynamics_fn(robot)

    previous = onp.asarray(topp.topp_ra_batched(wps, vmax, amax, n_grid=128).duration)
    for frac in (3.0, 2.0, 1.5, 1.0, 0.8, 0.6):
        cons = topp.torque_constraints(paths, id_fn, -effort * frac, effort * frac)
        r = topp.topp_ra_batched(wps, vmax, amax, n_grid=128, extra_constraints=cons)
        assert bool(jnp.all(r.feasible))
        current = onp.asarray(r.duration)
        # Relative, because the tolerance being absorbed here is float32 noise
        # in the constraint coefficients, which scales with the duration.
        assert onp.all(current >= previous * (1 - 1e-3)), f"non-monotone at {frac}"
        previous = current


# ---------------------------------------------------------------------------
# The 2-D LP
# ---------------------------------------------------------------------------


def test_lp_x_range_on_a_known_polygon():
    from pyroffi.topp._lp import x_range

    # Unit square 0 <= u <= 1, 0 <= x <= 1, cut by u + x <= 1.2.
    A = jnp.asarray(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [1.0, 1.0]]
    )
    h = jnp.asarray([1.0, 0.0, 1.0, 0.0, 1.2])
    rng = x_range(A, h, jnp.asarray(1.0), jnp.asarray(1.0))
    assert bool(rng.feasible)
    assert float(rng.lo) == pytest.approx(0.0, abs=1e-6)
    assert float(rng.hi) == pytest.approx(1.0, abs=1e-6)


def test_lp_x_range_detects_empty_polygon():
    from pyroffi.topp._lp import x_range

    # x >= 2 and x <= 1 simultaneously.
    A = jnp.asarray([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    h = jnp.asarray([1.0, 1.0, 1.0, -2.0])
    rng = x_range(A, h, jnp.asarray(1.0), jnp.asarray(1.0))
    assert not bool(rng.feasible)


def test_lp_u_range_matches_manual_interval():
    from pyroffi.topp._lp import u_range

    # 2u + x <= 3 and -u <= 1, evaluated at x = 1  ->  u in [-1, 1].
    A = jnp.asarray([[2.0, 1.0], [-1.0, 0.0]])
    h = jnp.asarray([3.0, 1.0])
    rng = u_range(A, h, jnp.asarray(1.0), jnp.asarray(1.0), jnp.asarray(1.0))
    assert bool(rng.feasible)
    assert float(rng.lo) == pytest.approx(-1.0, abs=1e-5)
    assert float(rng.hi) == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# CUDA / GRiD backend
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_gpu(), reason="GRiD dynamics requires a CUDA device")
def test_grid_and_jax_backends_agree(panda):
    from pyroffi.dynamics import GRiDDynamics

    urdf, robot = panda
    gd = GRiDDynamics(urdf)
    dof = gd.num_dof

    effort = jnp.asarray([float(j.limit.effort) for j in urdf.actuated_joints]) * 0.5
    vmax = jnp.asarray([float(j.limit.velocity) for j in urdf.actuated_joints])
    amax = jnp.ones(dof) * 3.0

    wp = _random_paths(jax.random.PRNGKey(3), 1, 5, dof)[0]
    path = topp.make_path(wp, n_grid=100)

    jax_cons = topp.torque_constraints(
        path, topp.jax_inverse_dynamics_fn(robot), -effort, effort
    )
    grid_cons = topp.torque_constraints(
        path, topp.grid_inverse_dynamics_fn(gd), -effort, effort
    )
    # float32 RNEA on two different implementations; relative agreement.
    scale = float(jnp.max(jnp.abs(jax_cons.h)))
    assert float(jnp.max(jnp.abs(jax_cons.h - grid_cons.h))) < 1e-3 * scale

    r_jax = topp.topp_ra(wp, vmax, amax, n_grid=100, extra_constraints=jax_cons)
    r_grid = topp.topp_ra(wp, vmax, amax, n_grid=100, extra_constraints=grid_cons)
    assert bool(r_jax.feasible) and bool(r_grid.feasible)
    assert float(r_grid.duration) == pytest.approx(float(r_jax.duration), rel=1e-3)


@pytest.mark.skipif(not _has_gpu(), reason="GRiD dynamics requires a CUDA device")
def test_grid_batched_constraints_single_launch(panda):
    """A ``[B, N, DOF]`` batch must go through the FFI kernels in one shot."""
    from pyroffi.dynamics import GRiDDynamics

    urdf, robot = panda
    gd = GRiDDynamics(urdf)
    dof = gd.num_dof
    batch, n_grid = 16, 64

    effort = jnp.asarray([float(j.limit.effort) for j in urdf.actuated_joints]) * 0.8
    vmax = jnp.asarray([float(j.limit.velocity) for j in urdf.actuated_joints])
    amax = jnp.ones(dof) * 3.0

    wps = _random_paths(jax.random.PRNGKey(83), batch, 6, dof)
    paths = jax.vmap(lambda w: topp.make_path(w, n_grid))(wps)
    assert paths.q.shape == (batch, n_grid, dof)

    cons = topp.torque_constraints(
        paths, topp.grid_inverse_dynamics_fn(gd), -effort, effort
    )
    assert cons.A.shape == (batch, n_grid, 2 * dof, 2)

    r = topp.topp_ra_batched(wps, vmax, amax, n_grid=n_grid, extra_constraints=cons)
    assert r.duration.shape == (batch,)
    assert bool(jnp.all(r.feasible))

    tau = gd.inverse_dynamics(
        r.q.reshape(-1, dof), r.qd.reshape(-1, dof), r.qdd.reshape(-1, dof)
    )
    assert float(jnp.max(jnp.abs(tau) / effort)) <= LIMIT_TOL
