"""Dynamics-augmented tower trajopt.

Runs the same tower trajopt as `spasm/tower_traj.py` (same cost terms,
schedules, boundary re-stitching) but adds a torque-limit penalty
(`extensions.dynamics.torque_cost`) to the per-step loss, weighted by
`--torque_weight`. Compares the resulting trajectory against a
`torque_weight=0` baseline (== `tower_traj.py` unmodified) on: final tower
cost, trajectory length (shortness), max |torque| per joint, and PD-tracking
error (dynamic feasibility).

Import-not-copy status: `spasm/tower_traj.py`'s trajopt cost (its own local
`cost(params, sim, initial_state, q_trajs, i)`) is imported directly and
called as a subroutine (`_augmented_cost` below wraps it, does not
reimplement it) -- so the collision/orientation/stability/shortness cost
math is exactly the original, byte-identical, never copied.

What *is* a thin copied driver: `tower_traj.py`'s `opt()` function calls its
own module-global `cost` by name inside a `jax.grad(cost, argnums=3)(...)`
closure, so there is no injection point to swap in an augmented cost without
either (a) monkeypatching `spasm.tower.traj.cost` (fragile, mutates shared
module state) or (b) copying the ~15-line `opt()` loop and pointing it at
`_augmented_cost` instead. We do (b): `optimize()` below is a copy of
`tower_traj.opt()`'s body with one line changed (which cost function is
differentiated). Everything it calls (`q_traj_init`, the schedule, the
return-trajectory re-stitching) is imported, not copied.
"""
import argparse
import os
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp


from spasm import backend
from spasm.tower.env import TowerSimulation
from spasm.tower.solve import cost as tower_cost
from spasm.tower.traj import TrajOptParams, cost as trajopt_cost
from spasm.conversions import q_traj_init, matrix_to_xyzyaw

from spasm.extensions.dynamics import torque_cost, torque_profile, track_rollout, TORQUE_LIMITS


# Assumed uniform time spacing between adjacent trajopt waypoints, used only
# for the finite-difference qd/qdd inside the torque penalty and for
# torque/tracking analysis after the fact. tower_traj.py's own trajectories
# carry no explicit timing (it's a pure GD-on-waypoints scheme), so this is a
# modeling choice, not a ported quantity -- documented rather than hidden.
DEFAULT_DT = 0.15


def _augmented_cost(params, sim, initial_state, q_trajs, i, torque_weight, dt):
    """tower_traj.py's original trajopt cost plus a torque-limit penalty
    summed over all trajectory segments (pick-place and return)."""
    base = trajopt_cost(params, sim, initial_state, q_trajs, i)
    if torque_weight == 0.0:
        return base
    full_trajs = jnp.concatenate([initial_state[:, None, :], q_trajs], axis=1)  # (num_trajs, T+1, 7)
    tc = jax.vmap(lambda qt: torque_cost(qt, dt))(full_trajs).sum()
    return base + torque_weight * tc


def optimize(params: TrajOptParams, sim: TowerSimulation, initial_state, final_state,
             torque_weight: float, dt: float):
    """Copy of `tower_traj.opt()`'s loop, differentiating `_augmented_cost`
    instead of the unaugmented `cost`. See module docstring for why this is
    copied rather than imported."""
    T = 10
    num_blocks = sim.num_blocks
    num_trajs = 2 * num_blocks - 1

    q_trajs_init = q_traj_init(initial_state, final_state, T)
    assert q_trajs_init.shape == (num_trajs, T + 2, 7)

    def schedule_lr(init_lr, step, total_steps):
        return (1.0 - step / total_steps) * init_lr

    def opt_step(i, q_trajs):
        lr = schedule_lr(params.trajopt_lr, i, params.trajopt_steps)
        grad = jax.grad(_augmented_cost, argnums=3)(
            params, sim, q_trajs[:, 0, :], q_trajs[:, 1:, :], i, torque_weight, dt
        )
        q_trajs = q_trajs.at[:, 1:, :].add(grad * -lr)

        return_starts = q_trajs[::2, -1, :][:-1]
        q_trajs = q_trajs.at[1::2, 0, :].set(return_starts)

        return_ends = q_trajs[::2, 0, :][1:]
        q_trajs = q_trajs.at[1::2, -1, :].set(return_ends)

        return q_trajs

    opt_q_traj = jax.lax.fori_loop(0, params.trajopt_steps, opt_step, q_trajs_init)
    return opt_q_traj


optimize_jit = jax.jit(optimize, static_argnames=('params', 'sim', 'dt', 'torque_weight'))


def evaluate(params, sim, q_opt_traj, init_state, dt, kp=200.0, kd=15.0):
    """Metrics used for the with/without-torque-cost comparison."""
    final_poses_xyzyaw = jax.vmap(lambda q: matrix_to_xyzyaw(backend.get_ee_pose(q)))(
        q_opt_traj[::2, -1, :]
    )
    final_tower_cost = float(tower_cost(sim, final_poses_xyzyaw, init_state))

    shortness = float(
        jnp.linalg.norm(q_opt_traj[:, 1:, :] - q_opt_traj[:, :-1, :], axis=-1).sum()
    )

    full_trajs = jnp.concatenate([q_opt_traj[:, 0, :][:, None, :], q_opt_traj[:, 1:, :]], axis=1)
    profiles = jax.vmap(lambda qt: torque_profile(qt, dt)['max_abs_tau'])(full_trajs)
    max_tau_per_joint = jnp.max(profiles, axis=0)
    max_tau = float(jnp.max(max_tau_per_joint))
    frac_over_limit = float(jnp.mean(profiles > TORQUE_LIMITS[None, :]))

    rollouts = jax.vmap(lambda qt: track_rollout(qt, dt, kp, kd)['rms_tracking_error'])(full_trajs)
    mean_track_err = float(jnp.mean(rollouts))
    max_track_err = float(jnp.max(rollouts))

    return {
        'final_tower_cost': final_tower_cost,
        'trajectory_length': shortness,
        'max_tau_per_joint': np_list(max_tau_per_joint),
        'max_tau': max_tau,
        'frac_waypoints_over_limit': frac_over_limit,
        'mean_pd_tracking_rms': mean_track_err,
        'max_pd_tracking_rms': max_track_err,
    }


def np_list(a):
    import numpy as np
    return [round(float(x), 3) for x in np.asarray(a)]


if __name__ == '__main__':
    from spasm.util import jax_cache_on
    jax_cache_on()

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_blocks', type=int, default=10)
    parser.add_argument('--num_obs', type=int, default=1)
    parser.add_argument('--torque_weight', type=float, default=1e-4,
                         help="Weight on the torque-limit penalty (default: small).")
    parser.add_argument('--dt', type=float, default=DEFAULT_DT,
                         help="Assumed uniform waypoint spacing (s) for finite-diff dynamics.")
    parser.add_argument('--kp', type=float, default=200.0)
    parser.add_argument('--kd', type=float, default=15.0)
    args = parser.parse_args()

    try:
        data = jnp.load('saved/tower.npz')
        solutions = data['opt_particles']
        init_state = data['init_state']
    except FileNotFoundError:
        print("Could not find 'saved/tower.npz'. Run 'spasm/tower_solve.py' first.")
        sys.exit(1)

    params = TrajOptParams()
    sim = TowerSimulation(num_blocks=init_state.shape[0], num_obs=args.num_obs)
    sim.z_error_mul = 5.0
    sim.set_state(solutions[0])

    if sim.num_obs == 0:
        params.trajopt_steps = 10
    if sim.num_obs == 1:
        params.trajopt_steps = 30

    print(f"num_blocks={init_state.shape[0]} num_obs={args.num_obs} "
          f"torque_weight={args.torque_weight} dt={args.dt}")

    results = {}
    for label, tw in [('baseline (torque_weight=0)', 0.0),
                       (f'dynamics-augmented (torque_weight={args.torque_weight})', args.torque_weight)]:
        print(f"\n--- {label} ---")
        # warmup + timed
        t0 = time.perf_counter()
        q_opt_traj = optimize_jit(params, sim, init_state, solutions[0], tw, args.dt)
        q_opt_traj.block_until_ready()
        t1 = time.perf_counter()
        print(f"opt time: {(t1 - t0) * 1000:.1f} ms (incl. compile)")

        metrics = evaluate(params, sim, q_opt_traj, init_state, args.dt, args.kp, args.kd)
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        results[label] = metrics

        os.makedirs('saved', exist_ok=True)
        fname = 'saved/tower_traj_dyn_baseline.npy' if tw == 0.0 else 'saved/tower_traj_dyn_augmented.npy'
        jnp.save(fname, q_opt_traj)

    print("\n=== Comparison table ===")
    keys = list(next(iter(results.values())).keys())
    labels = list(results.keys())
    header = f"{'metric':32s}" + "".join(f"{l[:28]:>30s}" for l in labels)
    print(header)
    for k in keys:
        row = f"{k:32s}" + "".join(f"{str(results[l][k]):>30s}" for l in labels)
        print(row)
