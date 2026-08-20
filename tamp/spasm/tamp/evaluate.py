"""Plan-level dynamic evaluation — where the paper's force numbers come from.

A PDDLStream plan is a sequence of actions whose ``move`` steps carry joint
trajectories. Scoring individual segments understates the difference between
the two motion backends, because the interesting failure is at the *junctions*:
a plan executed as one continuous motion arrives at the end of a segment at
full speed and immediately reverses direction into the next one. That velocity
discontinuity is what makes the implied accelerations — and therefore the
actuator torques and the equivalent end-effector force — explode.

Two things are measured here.

:func:`evaluate_plan_untimed`
    The stock regime. Concatenate the plan's segments into one path, hand it to
    an executor that knows nothing about dynamics, and let that executor spread
    the motion uniformly over some total duration. There is no principled way
    to choose that duration from a purely kinematic plan — which is the point —
    so :func:`tempo_sweep` sweeps it.

:func:`evaluate_plan_retimed`
    The pyroffi regime. Retime each segment with TOPP-RA under actuator torque
    limits, coming to rest at each junction (``sd_start = sd_end = 0``), so no
    discontinuity exists to begin with. The result is torque-feasible by
    construction, and TOPP-RA additionally reports the *fastest* duration the
    path admits — the number the kinematic side has no way to compute.

The headline comparison is therefore not "same plan, different force" but
"the kinematic plan is feasible only if executed slower than a bound it cannot
compute, and pyroffi computes that bound while guaranteeing feasibility."
"""
from __future__ import annotations

import numpy as np

from . import _setup  # noqa: F401
from . import motion as M


def plan_segments(plan):
    """Extract the ``(T, 7)`` joint trajectories from a PDDLStream plan.

    ``move`` actions carry their trajectory as the last argument. Actions with
    no trajectory (pick/place, which are instantaneous in this domain) are
    skipped. Returns a list of ``(traj, carrying)`` pairs, where ``carrying``
    marks segments executed while holding a cube — those are scored with the
    payload included.
    """
    segments = []
    holding = False
    for action in plan:
        name = action.name
        if name == "pick":
            holding = True
            continue
        if name == "place":
            holding = False
            continue
        traj = next((a for a in reversed(action.args)
                     if isinstance(a, np.ndarray) and a.ndim == 2), None)
        if traj is not None:
            segments.append((np.asarray(traj)[:, :7], holding))
    return segments


def concatenate(segments):
    """Join segments into one path, dropping the duplicated junction waypoint."""
    parts = [s for s, _ in segments]
    if not parts:
        return np.zeros((0, 7))
    return np.concatenate([p if i == 0 else p[1:] for i, p in enumerate(parts)],
                          axis=0)


def evaluate_plan_untimed(plan, total_duration, payload_mass=M.CUBE_MASS):
    """Score a plan executed as one continuous motion over ``total_duration``.

    The payload is applied throughout rather than per-segment: this is the
    optimistic reading for the stock side in transport-heavy plans and keeps the
    scoring a single pass. Returns ``None`` for an empty plan.
    """
    segments = plan_segments(plan)
    path = concatenate(segments)
    if path.shape[0] < 3:
        return None
    params = M.MotionParams(n_waypoints=path.shape[0],
                            nominal_duration=total_duration,
                            payload_mass=payload_mass)
    out = M.score_untimed_path(path, params)
    out["n_waypoints"] = int(path.shape[0])
    out["n_segments"] = len(segments)
    return out


def evaluate_plan_retimed(plan, payload_mass=M.CUBE_MASS, n_grid=128,
                          torque_fraction=0.85):
    """Retime every segment with TOPP-RA and score the result.

    Each segment is retimed independently and brought to rest at its endpoints,
    so the concatenation is continuous in velocity by construction. The reported
    duration is the sum over segments — the fastest torque-feasible execution of
    this plan.
    """
    segments = plan_segments(plan)
    if not segments:
        return None

    per_segment, total = [], 0.0
    for traj, carrying in segments:
        params = M.MotionParams(
            n_waypoints=traj.shape[0], n_grid=n_grid,
            torque_fraction=torque_fraction,
            payload_mass=payload_mass if carrying else 0.0)
        res = M._retime(np.asarray(traj, dtype=np.float32), params)
        if not bool(res.feasible):
            return {"feasible": False, "n_segments": len(segments)}
        s = M.score_trajectory(res.q, res.qd, res.qdd, params)
        s["duration_s"] = float(res.duration)
        s["carrying"] = bool(carrying)
        per_segment.append(s)
        total += float(res.duration)

    return {
        "feasible": all(s["torque_feasible"] for s in per_segment),
        "duration_s": total,
        "peak_ee_force_n": max(s["peak_ee_force_n"] for s in per_segment),
        "mean_ee_force_n": float(np.mean([s["mean_ee_force_n"] for s in per_segment])),
        "peak_tau_nm": max(s["peak_tau_nm"] for s in per_segment),
        "utilisation": max(s["utilisation"] for s in per_segment),
        "n_segments": len(segments),
        "per_segment": per_segment,
    }


def tempo_sweep(plan, durations, payload_mass=M.CUBE_MASS):
    """Score the untimed plan across a range of executor tempos.

    This is the figure the force claim rests on: peak end-effector force scales
    roughly as ``1/T**2``, so the kinematic plan is feasible at a leisurely
    tempo and catastrophically infeasible at a brisk one — with nothing in the
    plan itself indicating which side of the line it is on.
    """
    rows = []
    for T in durations:
        r = evaluate_plan_untimed(plan, T, payload_mass=payload_mass)
        if r is None:
            continue
        r["total_duration_s"] = float(T)
        rows.append(r)
    return rows


def format_comparison(untimed_rows, retimed, title="Plan dynamics"):
    """Render the comparison as a Markdown table."""
    lines = [f"## {title}", "",
             "### Stock regime — kinematic path, executor-chosen tempo", "",
             "| total duration (s) | peak torque (Nm) | limit utilisation | "
             "peak EE force (N) | executable |",
             "|--:|--:|--:|--:|:--|"]
    for r in untimed_rows:
        ok = "yes" if r["torque_feasible"] else "**NO**"
        lines.append(f"| {r['total_duration_s']:.2f} | {r['peak_tau_nm']:.0f} | "
                     f"{r['utilisation']:.2f}× | {r['peak_ee_force_n']:.0f} | {ok} |")

    lines += ["", "### pyroffi — TOPP-RA under actuator torque limits", ""]
    if retimed is None:
        lines.append("_no plan to retime_")
    elif not retimed.get("feasible"):
        lines.append("_no feasible schedule for this path_")
    else:
        lines += [
            f"- fastest torque-feasible duration: **{retimed['duration_s']:.2f} s**",
            f"- peak torque: **{retimed['peak_tau_nm']:.0f} Nm** "
            f"({retimed['utilisation']:.2f}× of limit)",
            f"- peak EE force: **{retimed['peak_ee_force_n']:.0f} N**",
            f"- executable: **yes, by construction**",
            "",
            "The kinematic plan is executable only for tempos slower than "
            "roughly this duration — a bound it provides no way to compute.",
        ]
    return "\n".join(lines)
