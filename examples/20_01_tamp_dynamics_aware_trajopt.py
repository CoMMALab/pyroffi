"""Dynamics-aware trajopt: why a kinematically valid plan can be unexecutable

A purely kinematic trajectory optimiser is optimising the wrong thing. It knows
where the arm may go, and it knows to keep the path short and collision-free,
but nothing in its objective knows the arm has mass. Curvature is free, so it
buys curvature freely — and curvature is exactly what actuator torque pays for.

This example runs one optimiser twice on the same task, with the same cost
terms, the same schedule and the same initialisation. The only difference is a
single extra term: pyroffi's differentiable torque penalty, which
back-propagates through inverse dynamics
(``spasm.extensions.dynamics.torque_cost``). Both runs call SPaSM's *own*
trajopt cost function — imported, not reimplemented — so "same optimiser, one
extra term" is literally true rather than approximately true.

What to look for:

* **peak torque** against the Franka's limits (87 Nm on joints 1-4, 12 Nm on
  joints 5-7). The kinematic-only run overruns its actuators by more than an
  order of magnitude: it is a trajectory that cannot be executed by the robot
  it was planned for.
* **task cost and path length**, which get *better*, not worse. This is not a
  quality-for-safety trade — the kinematic optimiser was simply unconstrained
  in a direction that never helped it.
* **PD tracking error**, from rolling both trajectories through pyroffi's
  forward dynamics. This is the independent check: a trajectory demanding
  impossible torques cannot be followed.

Run::

    python examples/20_01_tamp_dynamics_aware_trajopt.py
    python examples/20_01_tamp_dynamics_aware_trajopt.py --torque-weight 1e-3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

TAMP_ROOT = Path(__file__).resolve().parents[1] / "tamp"
sys.path.insert(0, str(TAMP_ROOT))

from spasm.tamp import _setup  # noqa: E402  (path shim + env pins)
from spasm.extensions import dynamic_tower as dt  # noqa: E402
from spasm.extensions.dynamics import TORQUE_LIMITS  # noqa: E402
from spasm.tower.env import TowerSimulation  # noqa: E402
from spasm.tower.traj import TrajOptParams  # noqa: E402
from spasm.paths import TAMP_ROOT as _ROOT  # noqa: E402


def run(label, sim, params, init_state, final_state, torque_weight, dt_step, kp, kd):
    t0 = time.perf_counter()
    traj = dt.optimize_jit(params, sim, init_state, final_state,
                           torque_weight, dt_step)
    traj.block_until_ready()
    wall = time.perf_counter() - t0
    metrics = dt.evaluate(params, sim, traj, init_state, dt_step, kp=kp, kd=kd)
    metrics["opt_wall_s"] = wall
    print(f"  {label:<28} peak {metrics['max_tau']:>8.1f} Nm   "
          f"({wall:.1f}s)", flush=True)
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-blocks", type=int, default=10)
    ap.add_argument("--num-obs", type=int, default=1)
    ap.add_argument("--torque-weight", type=float, default=1e-4)
    ap.add_argument("--dt", type=float, default=dt.DEFAULT_DT,
                    help="assumed seconds between waypoints (SPaSM trajectories "
                         "carry no explicit timing)")
    ap.add_argument("--kp", type=float, default=200.0)
    ap.add_argument("--kd", type=float, default=15.0)
    args = ap.parse_args()

    saved = Path(_ROOT) / "saved" / "tower.npz"
    if not saved.exists():
        sys.exit(f"missing {saved} — run tamp/spasm/tower/solve.py first")

    data = np.load(saved)
    init_state = data["init_state"]          # (num_blocks, 4) start poses
    final_state = data["opt_particles"][0]   # solved target placements

    sim = TowerSimulation(num_blocks=init_state.shape[0], num_obs=args.num_obs)
    sim.z_error_mul = 5.0
    sim.set_state(final_state)

    params = TrajOptParams()
    # Step counts the tower task is tuned for (matches dynamic_tower.py).
    if args.num_obs == 0:
        params.trajopt_steps = 10
    elif args.num_obs == 1:
        params.trajopt_steps = 30

    print(f"tower task: {init_state.shape[0]} blocks, {args.num_obs} obstacle(s), "
          f"{params.trajopt_steps} trajopt steps, dt={args.dt}s between waypoints\n")

    baseline = run("kinematic only", sim, params, init_state, final_state,
                   0.0, args.dt, args.kp, args.kd)
    augmented = run(f"+ torque penalty ({args.torque_weight:g})", sim, params,
                    init_state, final_state, args.torque_weight, args.dt,
                    args.kp, args.kd)

    print("\n=== Same optimiser, one extra term ===")
    print(f"{'metric':<28} {'kinematic only':>16} {'+ torque penalty':>18}")
    for key, fmt in [("max_tau", "{:.1f} Nm"),
                     ("final_tower_cost", "{:.3f}"),
                     ("trajectory_length", "{:.1f}"),
                     ("frac_waypoints_over_limit", "{:.1%}"),
                     ("mean_pd_tracking_rms", "{:.3f}"),
                     ("max_pd_tracking_rms", "{:.3f}")]:
        print(f"{key:<28} {fmt.format(baseline[key]):>16} "
              f"{fmt.format(augmented[key]):>18}")

    print(f"\n{'per-joint peak torque':<28}")
    limits = np.asarray(TORQUE_LIMITS)
    print(f"  {'joint':>5} {'limit':>7} {'kinematic':>11} {'+ torque':>10}")
    for j, (lim, b, a) in enumerate(zip(limits, baseline["max_tau_per_joint"],
                                        augmented["max_tau_per_joint"]), start=1):
        flag = "  <-- over limit" if b > lim else ""
        print(f"  {j:>5} {lim:>7.0f} {b:>11.1f} {a:>10.1f}{flag}")

    ratio = baseline["max_tau"] / max(augmented["max_tau"], 1e-9)
    print(f"\nPeak actuator torque reduced {ratio:.1f}x, with task cost "
          f"{baseline['final_tower_cost']:.3f} -> {augmented['final_tower_cost']:.3f} "
          f"and path length {baseline['trajectory_length']:.1f} -> "
          f"{augmented['trajectory_length']:.1f}.")
    print("The kinematic plan is not merely rougher — it exceeds the actuator "
          "limits\nof the robot it was planned for, and so cannot be executed at all.")


if __name__ == "__main__":
    main()
