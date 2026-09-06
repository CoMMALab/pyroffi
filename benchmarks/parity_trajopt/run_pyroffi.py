"""Solve the shared trajopt problems with pyroffi (run under `pyroffi`).

pyroffi's L-BFGS dynamics_trajopt, batched over all problems (vmap): fixed
endpoints, straight-line seed, cost = smoothness + world/self collision hinge.
Reports batched throughput and per-problem latency, both excluding compile.
"""
from __future__ import annotations
import pathlib, time
import jax, jax.numpy as jnp, numpy as np
import yourdfpy
import pyroffi as pk
from pyroffi.collision import Box, RobotCollisionSpherized
from pyroffi.optimization_engines import dynamics_trajopt, DynamicsTrajOptConfig
from _problems import (OBSTACLE_CENTER, OBSTACLE_DIMS, CLEARANCE_MARGIN,
                       T_WAYPOINTS, load, save_result)

RESOURCE_ROOT = pathlib.Path(__file__).resolve().parents[2] / "resources"
# Collision-cost-bound solve: profiling showed the L-BFGS driver is only ~8% of
# runtime and cost is ~linear in n_iters, so 60 (early-stop still on) keeps
# coll-free ~95% at ~1.6x the throughput of 100. See benchmarks note.
N_ITERS = 60
REPS = 5


def main():
    q_start, q_goal, lo, hi = load()
    N, dof = q_start.shape
    T = int(T_WAYPOINTS)
    urdf = yourdfpy.URDF.load(str(RESOURCE_ROOT / "panda" / "panda_spherized.urdf"))
    robot = pk.Robot.from_urdf(urdf)
    coll = RobotCollisionSpherized.from_urdf(
        urdf, srdf_path=str(RESOURCE_ROOT / "panda" / "panda.srdf"))
    obstacle = Box.from_center_and_dimensions(
        jnp.asarray(OBSTACLE_CENTER, jnp.float32)[None],
        jnp.float32(OBSTACLE_DIMS[0])[None], jnp.float32(OBSTACLE_DIMS[1])[None],
        jnp.float32(OBSTACLE_DIMS[2])[None])
    qs = jnp.asarray(q_start, jnp.float32); qg = jnp.asarray(q_goal, jnp.float32)
    m = jnp.float32(CLEARANCE_MARGIN)

    steps = jnp.linspace(0.0, 1.0, T)[:, None]

    def build_full(x_flat, q0, q1):
        interior = x_flat.reshape(T - 2, dof)
        return jnp.concatenate([q0[None], interior, q1[None]], axis=0)  # (T, dof)

    def cost_of(x_flat, q0, q1):
        q = build_full(x_flat, q0, q1)
        smooth = jnp.sum((q[2:] - 2 * q[1:-1] + q[:-2]) ** 2)
        vel = jnp.sum((q[1:] - q[:-1]) ** 2)
        dw = coll.compute_world_collision_distance(robot, q, obstacle).reshape(T, -1)
        ds = coll.compute_self_collision_distance(robot, q).reshape(T, -1)
        cw = jnp.sum(jnp.maximum(0.0, m - dw) ** 2)
        cs = jnp.sum(jnp.maximum(0.0, m - ds) ** 2)
        return 1.0 * smooth + 0.2 * vel + 50.0 * cw + 50.0 * cs

    cfg = DynamicsTrajOptConfig(n_iters=N_ITERS, early_stop=True, grad_tol=1e-4, m_lbfgs=8)

    def solve_one(q0, q1):
        seed = (q0[None] * (1 - steps) + q1[None] * steps)[1:-1].reshape(-1)
        return dynamics_trajopt(seed, lambda x: cost_of(x, q0, q1), cfg)

    # ---- batched throughput: explicit JIT warmup, then best-of-REPS ----
    batched = jax.jit(jax.vmap(solve_one))
    jax.block_until_ready(batched(qs, qg))                 # JIT warmup (compile)
    jax.block_until_ready(batched(qs, qg))                 # warm caches/graphs
    t0 = time.perf_counter()
    for _ in range(REPS):
        x = batched(qs, qg); jax.block_until_ready(x)
    batch_t = (time.perf_counter() - t0) / REPS

    # ---- single-problem latency: separate JIT, warmed, best-of-REPS ----
    single = jax.jit(solve_one)
    jax.block_until_ready(single(qs[0], qg[0]))            # warmup
    lat = []
    for i in range(min(10, N)):
        t0 = time.perf_counter()
        jax.block_until_ready(single(qs[i], qg[i]))
        lat.append(time.perf_counter() - t0)
    single_ms = float(np.median(lat) * 1e3)

    trajs = jax.vmap(lambda xf, q0, q1: build_full(xf, q0, q1))(x, qs, qg)
    save_result("pyroffi_trajopt", trajectories=np.asarray(trajs, np.float64),
                batch_time_s=np.array(batch_t), n_problems=np.array(N),
                per_problem_ms=np.array(batch_t / N * 1e3),
                single_problem_ms=np.array(single_ms))
    print(f"pyroffi dynamics_trajopt (n_iters={N_ITERS}): "
          f"batch({N})={batch_t*1e3:.0f}ms => {batch_t/N*1e3:.2f} ms/prob amortized; "
          f"single-problem latency={single_ms:.1f} ms (all warm, excl. compile)")


if __name__ == "__main__":
    main()
