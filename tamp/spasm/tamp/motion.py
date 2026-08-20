"""Motion backends for the PDDLStream ``s-motion`` stream.

This module is the pyroffi paper's claim expressed as code. PDDLStream's
``s-motion`` stream is asked to connect two arm configurations; *how* it answers
is the only thing that differs between the configurations being compared.

Three backends, sharing every piece of geometry (same URDF, same 59 collision
spheres, same analytic IK, same ``sphere_sphere_penetration``), so the symbolic
search is identical and any difference is attributable to the motion backend
alone:

``linear``
    Straight-line joint interpolation, accepted if joint limits hold and no
    collision sphere penetrates the floor. Cheap, and a useful control — but it
    is not what SPaSM does, and it flatters the kinematic case: a straight line
    is already smooth, so its torques are unremarkable. Included so the
    comparison cannot be accused of choosing a strawman baseline.

``spasm`` — the stock-SPaSM regime
    SPaSM's own trajectory optimisation: hand-rolled gradient descent on the
    waypoints under collision, end-effector-orientation and path-length costs,
    exactly the cost structure of :mod:`spasm.tetris.traj` / :mod:`spasm.tower.traj`.
    It is purely *kinematic* — nothing in the objective knows the arm has mass.
    The optimiser happily produces jagged, high-curvature paths, because
    curvature costs it nothing.

``dynamics`` — pyroffi
    The identical optimiser and identical cost terms, plus pyroffi's
    differentiable torque penalty (:func:`spasm.extensions.dynamics.torque_cost`,
    which back-propagates through inverse dynamics). Nothing else changes.

This mirrors, at the level of a single TAMP motion segment, the whole-plan
comparison already established by :mod:`spasm.extensions.dynamic_tower`, where
the kinematic-only objective peaks at **2665 Nm** (per-joint up to 1343 Nm,
against Franka limits of 87/12 Nm) and adding the torque term brings the same
task down to **107 Nm** — a ~25x reduction in peak actuator torque for a *lower*
task cost and a shorter path. The point of this module is that the same effect
holds when the motion generator is driven by a task planner rather than by a
fixed script.

Reporting
---------
:func:`segment_metrics` scores any trajectory identically regardless of which
backend produced it. Torque is the primary quantity, in Nm, directly comparable
to ``TORQUE_LIMITS``. An equivalent end-effector force in newtons is also
reported (:func:`ee_force`) for readers who prefer a Cartesian number, but the
actuator-limit claim is the torque one.

Timing
------
SPaSM's trajectories carry no explicit timing — the optimiser works on
waypoints alone — so a uniform ``dt`` is assumed for the finite-difference
qd/qdd inside the torque computation. That is a modelling choice, and it is the
same choice :mod:`spasm.extensions.dynamic_tower` documents (``DEFAULT_DT``);
it is applied identically to every backend, so it cannot favour one.
:func:`retime` additionally offers pyroffi's TOPP-RA as a *reporting* tool: it
computes the fastest torque-feasible timing an already-planned path admits,
which is a bound the kinematic pipeline has no way to produce.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from pyroffi import topp

from . import _setup  # noqa: F401
from . import geometry as g
from spasm import backend
from spasm.extensions import dynamics as dyn
from spasm.extensions.dynamics import torque_cost, torque_profile

# Index of the grasp frame in DYN_ROBOT's link list; the payload hangs here.
_EE_BODY = list(dyn.DYN_ROBOT.links.names).index("panda_grasptarget")

#: Gravity as a world-frame acceleration vector (m/s^2), z-down.
GRAVITY_VEC = jnp.array([0.0, 0.0, dyn.GRAVITY])

#: Mass of a planning cube (kg): a 5cm ABS-ish cube. Default transport payload.
CUBE_MASS = 0.15

#: Seconds between adjacent waypoints. SPaSM trajectories carry no timing; this
#: is the same modelling assumption ``dynamic_tower.DEFAULT_DT`` makes, applied
#: identically to every backend.
DEFAULT_DT = 0.15

#: Franka joint velocity / acceleration limits (rad/s, rad/s^2), for TOPP-RA.
VELOCITY_LIMITS = jnp.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
ACCELERATION_LIMITS = jnp.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])


@dataclass(frozen=True)
class MotionParams:
    """Knobs shared by every backend. Frozen so it can be a static jit arg."""

    n_waypoints: int = 20
    """Waypoints per segment. Identical across backends."""

    dt: float = DEFAULT_DT
    """Assumed time between waypoints (see module docstring)."""

    steps: int = 60
    """Gradient steps for the ``spasm`` / ``dynamics`` optimiser."""

    lr: float = 0.01
    """Step size. Identical across those two backends."""

    w_collision: float = 0.005
    """Arm-collision weight (``TrajOptParams.arm_collision_weight``)."""

    w_orientation: float = 0.05
    """Keep-the-hand-pointing-down weight."""

    w_shortness: float = 0.50
    """Path-length weight."""

    torque_weight: float = 0.0
    """Weight on the differentiable torque penalty. **This is the only
    parameter that differs between the ``spasm`` and ``dynamics`` backends.**"""

    reject_infeasible: bool = False
    """If set, ``s-motion`` rejects segments whose peak torque exceeds the
    actuator limits, so infeasibility propagates back into PDDLStream's search
    instead of merely being reported."""

    payload_mass: float = 0.0
    """kg carried at the grasp frame. 0 for a free (empty-gripper) motion."""

    n_grid: int = 128
    """TOPP-RA gridpoints, used only by :func:`retime`."""

    torque_fraction: float = 0.85
    """Fraction of actuator torque limits TOPP-RA may use."""


# --------------------------------------------------------------------------- #
# Payload wrench -> joint torque
# --------------------------------------------------------------------------- #

def _ee_linear_jacobian(q):
    """Linear ``(3, 7)`` block of the grasp-frame Jacobian at ``q``."""
    J_all, _ = dyn.DYN_ROBOT.jacobian(jnp.asarray(q, jnp.float32))
    return J_all[_EE_BODY][:3, :]


def _payload_torque_state(q, qd, qdd, mass):
    """Payload joint torque at one state.

    ``a_ee = J qdd + Jdot qd``; the ``Jdot qd`` term comes from a JVP of
    ``q -> J(q) qd`` along ``qd``, so Jdot is never formed. The result is affine
    in ``qdd`` with its velocity-product term vanishing at ``qd = 0`` — the
    structure :func:`pyroffi.topp.torque_constraints` requires.
    """
    J = _ee_linear_jacobian(q)
    _, jdot_qd = jax.jvp(lambda qq: _ee_linear_jacobian(qq) @ qd, (q,), (qd,))
    a_ee = J @ qdd + jdot_qd
    return J.T @ (mass * (a_ee - GRAVITY_VEC))


def inverse_dynamics_fn(payload_mass=0.0):
    """Batched ``(q, qd, qdd) -> tau`` over ``(N, DOF)`` including the payload."""
    def _fn(q, qd, qdd):
        q, qd, qdd = (jnp.asarray(a, jnp.float32) for a in (q, qd, qdd))
        tau = jax.vmap(dyn.inverse_dynamics)(q, qd, qdd)
        if payload_mass:
            tau = tau + jax.vmap(_payload_torque_state, in_axes=(0, 0, 0, None))(
                q, qd, qdd, payload_mass)
        return tau

    return _fn


# --------------------------------------------------------------------------- #
# Scoring — identical for every backend. This is the comparison table.
# --------------------------------------------------------------------------- #

def ee_force(q, tau):
    """Equivalent EE force magnitude (N) per waypoint, from ``J_v^T f = tau``."""
    J = jax.vmap(_ee_linear_jacobian)(q)

    def one(Ji, ti):
        f, *_ = jnp.linalg.lstsq(Ji.T, ti, rcond=None)
        return jnp.linalg.norm(f)

    return jax.vmap(one)(J, tau)


@functools.partial(jax.jit, static_argnums=(1, 2))
def _score(q_traj, dt, payload_mass):
    qd, qdd = dyn._finite_diff_qd_qdd(q_traj, dt)
    tau = inverse_dynamics_fn(payload_mass)(q_traj, qd, qdd)
    return jnp.max(jnp.abs(tau), axis=0), ee_force(q_traj, tau), tau


def segment_metrics(q_traj, params: MotionParams):
    """Dynamic quantities for one trajectory, as numpy scalars.

    ``utilisation`` is the peak torque as a multiple of the actuator limit:
    1.0 is exactly at the limit, and anything above it is not executable.
    """
    q_traj = jnp.asarray(q_traj, jnp.float32)[:, :7]
    peak_tau, f, tau = _score(q_traj, params.dt, params.payload_mass)
    peak_tau = np.asarray(peak_tau)
    limits = np.asarray(dyn.TORQUE_LIMITS)
    return {
        "peak_tau_per_joint": peak_tau,
        "peak_tau_nm": float(np.max(peak_tau)),
        "utilisation": float(np.max(peak_tau / limits)),
        "torque_feasible": bool(np.all(peak_tau <= limits)),
        "frac_over_limit": float(np.mean(np.abs(np.asarray(tau)) > limits[None, :])),
        "peak_ee_force_n": float(np.max(np.asarray(f))),
        "mean_ee_force_n": float(np.mean(np.asarray(f))),
        "path_length": float(np.sum(np.linalg.norm(
            np.diff(np.asarray(q_traj), axis=0), axis=-1))),
    }


# --------------------------------------------------------------------------- #
# Backend 1: straight-line control
# --------------------------------------------------------------------------- #

def plan_linear(q1, q2, params: MotionParams, world_spheres=None):
    """Straight-line joint interpolation, accepted on geometry alone."""
    path = g.interpolate(q1, q2, params.n_waypoints)
    if not g.arm_path_valid(path):
        return None
    return path


# --------------------------------------------------------------------------- #
# Backends 2 & 3: SPaSM trajopt, optionally torque-augmented
# --------------------------------------------------------------------------- #
# Both share this optimiser. `torque_weight == 0` reproduces the stock-SPaSM
# objective; `torque_weight > 0` is the pyroffi dynamics-aware one. Keeping them
# in one function is deliberate: it makes it structurally impossible for the two
# arms of the comparison to differ in any way other than that weight.

def _orientation_cost(q):
    """Reuse SPaSM's own 'keep the hand pointing down' term."""
    from spasm.tetris.traj import error_to_down
    return error_to_down(q)


def _collision_cost(q, world_spheres):
    """Arm collision spheres vs. the floor and the world's block spheres.

    Uses SPaSM's own ``sphere_sphere_penetration``, so the geometric scoring is
    the same function the stock solver optimises against.
    """
    from spasm.tetris.solve import sphere_sphere_penetration
    pos, rad = backend.fk(q)
    arm = jnp.concatenate([pos, rad[:, None]], axis=-1)
    floor = jnp.sum(jax.nn.relu(-(pos[:, 2] - rad) + g.FLOOR_Z + 6e-2) ** 2)
    if world_spheres is None or world_spheres.shape[0] == 0:
        return floor
    blocks = jnp.sum(jax.vmap(
        lambda s: sphere_sphere_penetration(arm, s, 2e-2).sum())(world_spheres))
    return floor + blocks


@functools.partial(jax.jit, static_argnums=(2,))
def _trajopt(q1, q2, params: MotionParams, world_spheres):
    """SPaSM-style GD on the interior waypoints, endpoints pinned.

    Endpoints are pinned because the task planner has already committed to
    them: ``s-ik`` produced q1/q2 to place the hand exactly over a cube, so
    moving them would silently invalidate the symbolic plan.
    """
    T = params.n_waypoints
    ts = jnp.linspace(0.0, 1.0, T)[:, None]
    init = (1.0 - ts) * q1[None, :] + ts * q2[None, :]

    def cost(interior):
        traj = jnp.concatenate([q1[None, :], interior, q2[None, :]], axis=0)
        coll = jnp.sum(jax.vmap(
            lambda q: _collision_cost(q, world_spheres))(traj))
        orient = jnp.sum(jax.vmap(_orientation_cost)(traj))
        short = jnp.sum(jnp.linalg.norm(jnp.diff(traj, axis=0), axis=-1))
        total = (params.w_collision * coll
                 + params.w_orientation * orient
                 + params.w_shortness * short)
        # The one and only difference between the `spasm` and `dynamics`
        # backends. torque_cost back-propagates through pyroffi's inverse
        # dynamics, so this is a real gradient, not a post-hoc filter.
        if params.torque_weight:
            total = total + params.torque_weight * torque_cost(traj, params.dt)
        return total

    def step(_, interior):
        grad = jnp.nan_to_num(jax.grad(cost)(interior), nan=0.0)
        return interior - params.lr * grad

    interior = jax.lax.fori_loop(0, params.steps, step, init[1:-1])
    return jnp.concatenate([q1[None, :], interior, q2[None, :]], axis=0)


def plan_trajopt(q1, q2, params: MotionParams, world_spheres=None):
    """Run the shared optimiser and apply the acceptance test."""
    q1 = jnp.asarray(q1, jnp.float32)[:7]
    q2 = jnp.asarray(q2, jnp.float32)[:7]
    ws = (jnp.zeros((0, 1, 4), jnp.float32) if world_spheres is None
          else jnp.asarray(world_spheres, jnp.float32))
    path = np.asarray(_trajopt(q1, q2, params, ws))

    if not np.all(np.isfinite(path)) or not g.arm_path_valid(path):
        return None
    if params.reject_infeasible and not segment_metrics(path, params)["torque_feasible"]:
        return None
    return path


# --------------------------------------------------------------------------- #
# TOPP-RA retiming (reporting tool, not a planner)
# --------------------------------------------------------------------------- #

@functools.partial(jax.jit, static_argnums=(1,))
def _retime(waypoints, params: MotionParams):
    path = topp.make_path(waypoints, n_grid=params.n_grid)
    cons = topp.acceleration_constraints(path, ACCELERATION_LIMITS, VELOCITY_LIMITS)
    tau_max = params.torque_fraction * dyn.TORQUE_LIMITS
    cons = cons.merge(topp.torque_constraints(
        path, inverse_dynamics_fn(params.payload_mass), -tau_max, tau_max))
    return topp.solve_topp_ra(path, cons, sd_start=0.0, sd_end=0.0)


def retime(q_traj, params: MotionParams):
    """Fastest torque-feasible timing for an existing path, or ``None``.

    This does not change the geometry — it answers "how fast may this path
    actually be executed?", which a kinematic pipeline cannot ask. A jagged
    path is punished here in the currency of time: it retimes to a much slower
    schedule, or to none at all.
    """
    res = _retime(jnp.asarray(q_traj, jnp.float32)[:, :7], params)
    if not bool(res.feasible):
        return None
    return res


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #

#: Backend name -> (planner, torque_weight override). ``None`` keeps the
#: params' own weight, letting callers sweep it.
BACKENDS = {
    "linear": (plan_linear, 0.0),
    "spasm": (plan_trajopt, 0.0),
    "dynamics": (plan_trajopt, None),
}

#: Default torque weight for the ``dynamics`` backend, matching
#: ``dynamic_tower.py``'s value.
DEFAULT_TORQUE_WEIGHT = 1e-4


def make_planner(backend_name, params: MotionParams, world_spheres=None):
    """``(q1, q2) -> path | None`` for the named backend.

    The returned closure fixes the torque weight implied by the backend name,
    so a caller cannot accidentally give the two arms of the comparison
    different settings.
    """
    try:
        fn, forced_weight = BACKENDS[backend_name]
    except KeyError:
        raise ValueError(
            f"unknown motion backend {backend_name!r}; "
            f"choose from {sorted(BACKENDS)}") from None

    if forced_weight is not None:
        params = dataclasses_replace(params, torque_weight=forced_weight)
    elif params.torque_weight == 0.0:
        params = dataclasses_replace(params, torque_weight=DEFAULT_TORQUE_WEIGHT)

    return lambda q1, q2: fn(q1, q2, params, world_spheres)


def dataclasses_replace(params, **kw):
    import dataclasses
    return dataclasses.replace(params, **kw)
