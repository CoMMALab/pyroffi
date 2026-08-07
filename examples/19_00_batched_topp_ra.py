"""Batched time-optimal path parameterisation (TOPP-RA)

A sampling-based planner answers *where* the robot goes. It says nothing about
*when*, and a geometric path executed naively either crawls or violates the
robot's limits. TOPP-RA closes that gap: it assigns the fastest timing the path
admits under joint velocity, acceleration and actuator torque limits.

This example runs it over a whole batch of paths at once. The inputs are real
pRRTC (CUDA RRT-Connect) solutions to MBM problems for the Panda, planned by
prrax and bundled into ``resources/panda/topp_paths.npz`` — so they have the
awkward shape planner output actually has: different waypoint counts per path,
uneven spacing, and sharp corners.

Three things the example is built to show:

1. **Padding is how variable-length planner output meets JAX.** Paths are
   right-padded to a common length and carry an ``n_valid`` count; the solver's
   internal grid is a fixed ``n_grid`` regardless, so the whole batch is one
   ``vmap``. The padded rows provably do not affect the answer.

2. **The CUDA path is a batching story, not a kernel story.** Torque limits
   need three RNEA evaluations per gridpoint. Done per path that is B tiny
   launches; done over the flattened ``[B * n_grid]`` state set it is three big
   ones. GRiD's FFI kernels want exactly that shape.

3. **Time-optimal is worth the machinery.** The same paths are retimed with the
   uniform-timestep method for comparison.

Requires ``resources/panda/topp_paths.npz``. Regenerate it from the prrax
checkout with::

    python generate_topp_path_bundle.py --robot panda --max-problems 64

The torque section additionally needs a CUDA device and nvcc for GRiD; it is
skipped automatically if unavailable.
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yourdfpy

import pyroffi
from pyroffi import topp
from pyroffi.toolbox._retiming import retime_path

REPO = Path(__file__).resolve().parents[1]
URDF = REPO / "resources" / "panda" / "panda_spherized.urdf"
PATHS = REPO / "resources" / "panda" / "topp_paths.npz"

N_GRID = 128
"""TOPP-RA gridpoints per path. Cost is linear in it; collocation error is not."""

TORQUE_FRACTION = 0.8
"""Fraction of the URDF effort limits to allow.

Below roughly 0.4 the Panda's own gravity torque exceeds the limit at some
configurations on these paths, which is genuinely infeasible rather than a
solver failure -- the example reports that case rather than hiding it.
"""


def load_batch():
    if not PATHS.exists():
        raise SystemExit(
            f"Missing {PATHS}.\n"
            "Generate it from the prrax checkout:\n"
            "  python generate_topp_path_bundle.py --robot panda --max-problems 64"
        )
    data = np.load(PATHS, allow_pickle=True)
    return (
        jnp.asarray(data["waypoints"], dtype=jnp.float32),
        jnp.asarray(data["n_valid"], dtype=jnp.int32),
        data["problem"],
    )


def report(name, result, vmax, amax, tau=None, tau_lim=None):
    feasible = np.asarray(result.feasible)
    dur = np.asarray(result.duration)
    v = np.asarray(jnp.max(jnp.abs(result.qd) / vmax, axis=(1, 2)))
    a = np.asarray(jnp.max(jnp.abs(result.qdd) / amax, axis=(1, 2)))

    print(f"\n{name}")
    print(f"  feasible          : {feasible.sum()}/{feasible.size}")
    if not feasible.any():
        return
    print(
        f"  duration (s)      : min {dur[feasible].min():.2f}  "
        f"median {np.median(dur[feasible]):.2f}  max {dur[feasible].max():.2f}"
    )
    # The peak ratios are the real check: a time-optimal solution rides its
    # limits, so anything much below 1.0 means the solver left speed on the
    # table, and anything much above means it cheated.
    print(f"  peak |qd| / vmax  : {v[feasible].max():.4f}")
    print(f"  peak |qdd| / amax : {a[feasible].max():.4f}")
    if tau is not None:
        ratio = np.asarray(jnp.max(jnp.abs(tau) / tau_lim, axis=(1, 2)))
        print(f"  peak |tau| / lim  : {ratio[feasible].max():.4f}")


def main():
    waypoints, n_valid, problem_names = load_batch()
    batch, n_wp_max, dof = waypoints.shape

    urdf = yourdfpy.URDF.load(str(URDF), load_meshes=False)
    robot = pyroffi.Robot.from_urdf(urdf)

    vmax = jnp.asarray([float(j.limit.velocity) for j in urdf.actuated_joints])
    effort = jnp.asarray([float(j.limit.effort) for j in urdf.actuated_joints])
    # URDFs carry velocity and effort limits but no acceleration limits, so one
    # has to be chosen. The Panda's datasheet figure is ~15 rad/s^2; 10 is a
    # conservative stand-in.
    amax = jnp.full((dof,), 10.0)

    counts = np.asarray(n_valid)
    print("=" * 70)
    print("Batched TOPP-RA on pRRTC / MBM paths")
    print("=" * 70)
    print(f"paths             : {batch} ({len(set(problem_names.tolist()))} problem sets)")
    print(f"dof               : {dof}")
    print(f"waypoints per path: min {counts.min()}, median {int(np.median(counts))}, "
          f"max {counts.max()}")
    print(f"padded to         : {n_wp_max}  "
          f"({100 * (1 - counts.mean() / n_wp_max):.0f}% of the tensor is padding)")
    print(f"TOPP-RA grid      : {N_GRID} points/path")

    # ------------------------------------------------------------------
    # 1. Kinematic limits only -- pure JAX, one vmapped solve.
    # ------------------------------------------------------------------
    solve = jax.jit(
        lambda wp, nv: topp.topp_ra_batched(
            wp, vmax, amax, n_grid=N_GRID, n_valid=nv
        )
    )
    kinematic = solve(waypoints, n_valid)
    jax.block_until_ready(kinematic.duration)

    t0 = time.perf_counter()
    kinematic = solve(waypoints, n_valid)
    jax.block_until_ready(kinematic.duration)
    t_kin = time.perf_counter() - t0

    report("[1] velocity + acceleration limits (pure JAX)", kinematic, vmax, amax)
    print(f"  solve wall time   : {1e3 * t_kin:.1f} ms for {batch} paths "
          f"({1e3 * t_kin / batch:.2f} ms/path)")

    # ------------------------------------------------------------------
    # 2. Padding really is inert.
    # ------------------------------------------------------------------
    # Solve one path on its own, unpadded, and check it lands in the same place.
    idx = int(np.argmax(counts))
    single = topp.topp_ra(
        waypoints[idx, : counts[idx]], vmax, amax, n_grid=N_GRID
    )
    print(
        f"\n[2] padding check (path {idx}, {counts[idx]} real waypoints)\n"
        f"  padded batch  : {float(kinematic.duration[idx]):.6f} s\n"
        f"  solved alone  : {float(single.duration):.6f} s"
    )

    # ------------------------------------------------------------------
    # 3. Against the uniform-timestep retiming.
    # ------------------------------------------------------------------
    # Same paths, same limits, numpy loop -- the method TOPP-RA replaces.
    uniform_durations = []
    for b in range(batch):
        r = retime_path(
            np.asarray(waypoints[b, : counts[b]], dtype=np.float64),
            np.asarray(vmax, dtype=np.float64),
            np.asarray(amax, dtype=np.float64),
        )
        uniform_durations.append(r.duration)
    uniform_durations = np.asarray(uniform_durations)
    feas = np.asarray(kinematic.feasible)
    speedup = uniform_durations[feas] / np.asarray(kinematic.duration)[feas]
    print(
        f"\n[3] vs uniform-timestep retiming\n"
        f"  uniform  median duration : {np.median(uniform_durations[feas]):.2f} s\n"
        f"  TOPP-RA  median duration : {np.median(np.asarray(kinematic.duration)[feas]):.2f} s\n"
        f"  speedup  median          : {np.median(speedup):.2f}x  "
        f"(range {speedup.min():.2f}-{speedup.max():.2f}x)"
    )
    if speedup.min() < 1.0:
        # Not a solver failure. The uniform method retimes the raw polyline and
        # treats each corner as an instantaneous direction change it never pays
        # for; TOPP-RA retimes the spline through it, sees the corner's real
        # (large) curvature, and slows down for it. On a path that is mostly
        # corners the honest answer is the slower one.
        print(
            f"  note: {(speedup < 1.0).sum()} path(s) come out slower than the "
            "uniform method,\n        which ignores the corner curvature that "
            "TOPP-RA is obliged to respect."
        )

    # ------------------------------------------------------------------
    # 4. Torque limits, coefficients built on the GPU.
    # ------------------------------------------------------------------
    tau_lim = effort * TORQUE_FRACTION
    has_gpu = any(d.platform == "gpu" for d in jax.devices())

    # The batched geometric path: [B, N_GRID, DOF]. Building it outside the
    # solver is what lets the dynamics see every gridpoint of every path as one
    # flat set of states.
    paths = jax.vmap(
        lambda wp, nv: topp.make_path(wp, N_GRID, nv)
    )(waypoints, n_valid)

    backends = [("pure JAX", topp.jax_inverse_dynamics_fn(robot))]
    if has_gpu:
        from pyroffi.dynamics import GRiDDynamics

        print("\nCompiling GRiD CUDA kernels for the Panda (cached after first run)...")
        t0 = time.perf_counter()
        gd = GRiDDynamics(urdf)
        print(f"  ready in {time.perf_counter() - t0:.1f} s")
        backends.append(("CUDA / GRiD", topp.grid_inverse_dynamics_fn(gd)))
    else:
        print("\nNo CUDA device: skipping the GRiD backend.")

    results = {}
    for label, id_fn in backends:
        # Three RNEA evaluations over B * N_GRID states -- 3 kernel launches for
        # the whole batch on the GRiD path, not 3 per path.
        jax.block_until_ready(
            topp.torque_constraints(paths, id_fn, -tau_lim, tau_lim).h
        )
        t0 = time.perf_counter()
        cons = topp.torque_constraints(paths, id_fn, -tau_lim, tau_lim)
        jax.block_until_ready(cons.h)
        t_cons = time.perf_counter() - t0

        r = topp.topp_ra_batched(
            waypoints, vmax, amax, n_grid=N_GRID, n_valid=n_valid,
            extra_constraints=cons,
        )
        jax.block_until_ready(r.duration)
        results[label] = r

        tau = robot.inverse_dynamics(r.q, r.qd, r.qdd)
        report(
            f"[4] + torque limits at {TORQUE_FRACTION:.0%} of effort ({label})",
            r, vmax, amax, tau=tau, tau_lim=tau_lim,
        )
        print(f"  constraint build  : {1e3 * t_cons:.1f} ms for "
              f"{batch * N_GRID} states x 3 RNEA")

    if len(results) == 2:
        ra, rb = results.values()
        a, b = np.asarray(ra.duration), np.asarray(rb.duration)
        both = np.asarray(ra.feasible) & np.asarray(rb.feasible)
        rel = np.abs(a[both] - b[both]) / b[both]
        # Two float32 RNEA implementations differ by ~1e-4 relative in the
        # constraint coefficients. That normally moves the duration by about as
        # much, but a path whose binding constraint switches between gridpoints
        # sits on a knife edge and can move far more -- hence median, not max.
        print(f"\n  backend agreement : relative duration diff, median "
              f"{np.median(rel):.1e}, max {rel.max():.1e} "
              f"({(rel > 1e-3).sum()}/{both.sum()} paths above 1e-3)")

    torque_result = results["CUDA / GRiD" if has_gpu else "pure JAX"]
    slowdown = np.asarray(torque_result.duration) / np.asarray(kinematic.duration)
    ok = np.asarray(torque_result.feasible) & feas
    if ok.any():
        print(f"  torque limits cost: {np.median(slowdown[ok]):.2f}x median duration")

    # ------------------------------------------------------------------
    # 5. Resample onto a controller's clock.
    # ------------------------------------------------------------------
    # TOPP-RA's grid is uniform in arc length, so its samples bunch up in time
    # wherever the trajectory slows. A servo loop needs the opposite.
    rate_hz = 1000.0
    b = int(np.argmax(np.asarray(kinematic.duration)))
    single_result = jax.tree.map(lambda z: z[b], kinematic)
    n_ticks = int(float(single_result.duration) * rate_hz) + 1
    t_grid = jnp.arange(n_ticks) / rate_hz
    q_c, qd_c, qdd_c = topp.sample_at_times(single_result, t_grid)

    print(
        f"\n[5] resampled for a {rate_hz:.0f} Hz controller (longest path, #{b})\n"
        f"  duration      : {float(single_result.duration):.3f} s -> {n_ticks} ticks\n"
        f"  q, qd, qdd    : {q_c.shape}, {qd_c.shape}, {qdd_c.shape}\n"
        f"  peak |qd|     : {float(jnp.max(jnp.abs(qd_c) / vmax)):.4f} x vmax\n"
        f"  time spacing  : TOPP grid {float(jnp.min(jnp.diff(single_result.times))):.4f}"
        f"-{float(jnp.max(jnp.diff(single_result.times))):.4f} s "
        f"(uneven), controller {1 / rate_hz:.4f} s (uniform)"
    )


if __name__ == "__main__":
    main()
