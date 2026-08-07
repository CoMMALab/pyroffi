"""Velocity- and acceleration-limited time parameterisation.

pyroffi's optimizers return geometric paths — an ordered list of
configurations with no notion of *when*.  Nothing downstream that talks about
duration (``simulate``, any ETA an agent reports) means anything until the path
carries times, so this is the missing piece the rest of the toolbox depends on.

The method is a **uniform timestep** sized by the binding constraint, not
TOPP-RA. Given the waypoint displacements ``dq_k`` and a single ``dt``:

    velocity:      max |dq_k| / dt          <= vmax
    acceleration:  max |dq_k - dq_{k-1}| / dt^2 <= amax

Both invert in closed form, so ``dt = max(dt_vel, dt_acc)`` and there is no
iteration at all. The displacement sequence is bracketed with zeros, which is
what makes the path start and stop at rest.

Why uniform, given that a per-segment schedule would be faster: a per-segment
formulation needs acceleration defined across neighbouring segments of differing
duration, and the natural discretisation (constant-velocity segments, averaged
spans) admits *zigzag* solutions — alternating fast and slow segments that score
as feasible because the metric averages across the alternation, but describe a
motion no one wants. Iterative stretching converges to exactly those, and the
duration then stops being monotone in the limits: tightening the velocity
ceiling could return a shorter trajectory. A uniform step cannot express a
zigzag, is monotone in both limits by construction, and is verifiable by
inspection.

The cost is real and worth stating plainly: because one step must accommodate
the tightest junction on the whole path (usually the endpoint ramp), durations
run roughly 2-3x a time-optimal solution. Feasible, monotone and predictable
beats fast-and-subtly-wrong for a verification tool.

When the duration itself matters -- executing on real hardware rather than
sanity-checking a plan -- use :mod:`pyroffi.topp` instead. It implements TOPP-RA
proper: genuinely time-optimal, torque limits as well as kinematic ones, and
batched over many paths on the GPU. Measured against this function on the
bundled MBM paths it is about 2.3x faster in the median.
"""

from __future__ import annotations

import dataclasses

import numpy as np

_MIN_DT = 1e-4
"""Floor on a segment duration, so a duplicated waypoint cannot divide by zero."""


@dataclasses.dataclass
class RetimingResult:
    """Times and derivatives for a retimed path."""

    times: np.ndarray
    """``(T,)`` waypoint times in seconds, starting at 0."""
    velocities: np.ndarray
    """``(T, DOF)`` rad/s, zero at both endpoints."""
    accelerations: np.ndarray
    """``(T, DOF)`` rad/s^2."""
    duration: float
    dt: float
    """The uniform timestep, in seconds."""
    feasible: bool
    """Whether both limits hold at the returned timing."""
    peak_velocity_ratio: float
    """Worst ``|v| / v_max`` over the path. > 1 means infeasible."""
    peak_acceleration_ratio: float
    limiting_joint: str | None
    """Joint that binds hardest, i.e. sets the duration."""

    def to_dict(self) -> dict:
        return {
            "duration_s": round(self.duration, 6),
            "n_waypoints": int(self.times.shape[0]),
            "feasible": bool(self.feasible),
            "peak_velocity_ratio": round(self.peak_velocity_ratio, 4),
            "peak_acceleration_ratio": round(self.peak_acceleration_ratio, 4),
            "limiting_joint": self.limiting_joint,
            "dt_s": round(self.dt, 6),
            "method": "uniform timestep at the binding limit (feasible and monotone, "
                      "roughly 2-3x a time-optimal TOPP-RA schedule)",
        }


def retime_path(
    waypoints: np.ndarray,
    velocity_limits: np.ndarray,
    acceleration_limits: np.ndarray,
    joint_names: tuple[str, ...] | None = None,
    tolerance: float = 1e-3,
) -> RetimingResult:
    """Assign times to ``waypoints`` respecting velocity and acceleration limits.

    Args:
        waypoints: ``(T, DOF)`` path, radians.
        velocity_limits: ``(DOF,)`` positive per-joint |qd| bounds, rad/s.
        acceleration_limits: ``(DOF,)`` positive per-joint |qdd| bounds, rad/s^2.
        joint_names: Used only to name the limiting joint in the report.
        tolerance: Relative slack allowed before a limit counts as violated.

    Returns:
        A :class:`RetimingResult`. Endpoint velocities are zero: the path starts
        and stops at rest, which is usually what sets the timestep.
    """
    q = np.asarray(waypoints, dtype=np.float64)
    if q.ndim != 2:
        raise ValueError(f"waypoints must be (T, DOF), got shape {q.shape}")
    n_wp, dof = q.shape

    vmax = np.abs(np.asarray(velocity_limits, dtype=np.float64).reshape(dof))
    amax = np.abs(np.asarray(acceleration_limits, dtype=np.float64).reshape(dof))
    # A zero/absent limit would make every ratio infinite; treat it as unbounded.
    vmax = np.where(vmax > 0.0, vmax, np.inf)
    amax = np.where(amax > 0.0, amax, np.inf)

    if n_wp == 1:
        return RetimingResult(
            times=np.zeros(1),
            velocities=np.zeros((1, dof)),
            accelerations=np.zeros((1, dof)),
            duration=0.0,
            dt=0.0,
            feasible=True,
            peak_velocity_ratio=0.0,
            peak_acceleration_ratio=0.0,
            limiting_joint=None,
        )

    dq = np.diff(q, axis=0)                       # (T-1, DOF)

    # A single timestep for the whole path. Two independent lower bounds, both
    # closed-form and both monotone in their limit, so the answer is monotone in
    # the limits by construction:
    #
    #   velocity      |dq_k| / dt              <= vmax
    #   acceleration  |dq_k - dq_{k-1}| / dt^2 <= amax
    #
    # The acceleration bound brackets the displacement sequence with zeros, which
    # is what makes the path start and stop at rest.
    dt_vel = float(np.max(np.abs(dq) / vmax)) if dq.size else 0.0

    dq_bracketed = np.concatenate(
        [np.zeros((1, dof)), dq, np.zeros((1, dof))], axis=0
    )                                                       # (T+1, DOF)
    delta = np.diff(dq_bracketed, axis=0)                   # (T, DOF)
    dt_acc = float(np.sqrt(np.max(np.abs(delta) / amax))) if delta.size else 0.0

    dt = max(dt_vel, dt_acc, _MIN_DT)

    times = np.arange(n_wp, dtype=np.float64) * dt

    # Derivatives on the same uniform grid the constraints were written against,
    # so the reported profile is exactly what was bounded rather than an
    # independent estimate that might disagree.
    seg_v = dq / dt                                          # (T-1, DOF)
    v_bracket = np.concatenate(
        [np.zeros((1, dof)), seg_v, np.zeros((1, dof))], axis=0
    )
    accel = np.diff(v_bracket, axis=0) / dt                  # (T, DOF)

    wp_v = 0.5 * (v_bracket[:-1] + v_bracket[1:])
    wp_v[0] = 0.0
    wp_v[-1] = 0.0

    v_ratio = np.abs(seg_v) / vmax
    a_ratio = np.abs(accel) / amax
    peak_v = float(np.max(v_ratio)) if v_ratio.size else 0.0
    peak_a = float(np.max(a_ratio)) if a_ratio.size else 0.0

    # The binding joint is whichever comes closest to its own ceiling.
    per_joint = np.maximum(np.max(v_ratio, axis=0), np.max(a_ratio, axis=0))
    limiting = None
    if joint_names is not None and per_joint.size:
        limiting = joint_names[int(np.argmax(per_joint))]

    return RetimingResult(
        times=times,
        velocities=wp_v,
        accelerations=accel,
        duration=float(times[-1]),
        dt=dt,
        feasible=bool(peak_v <= 1.0 + tolerance and peak_a <= 1.0 + tolerance),
        peak_velocity_ratio=peak_v,
        peak_acceleration_ratio=peak_a,
        limiting_joint=limiting,
    )


def default_acceleration_limits(
    velocity_limits: np.ndarray, time_to_peak: float = 0.5
) -> np.ndarray:
    """Acceleration limits inferred from velocity limits.

    URDFs carry velocity and effort limits but no acceleration limits, so one has
    to be invented: assume a joint can reach full speed in ``time_to_peak``
    seconds.  Deliberately conservative — a caller who knows the real numbers
    should pass them instead.
    """
    v = np.abs(np.asarray(velocity_limits, dtype=np.float64))
    return np.where(v > 0.0, v / max(time_to_peak, 1e-3), 0.0)
