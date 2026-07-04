"""CPU-accelerated IK / FK benchmark: QuIK, JAX-Halley, JAX LS/HJCD, VAMP FK.

Motivation
    pyroffi's flagship solvers are CUDA kernels.  This benchmark measures the
    *CPU-only* planning stack intended for ``JAX_PLATFORMS=cpu``: the QuIK C++
    Halley's-method IK backend, its pure-JAX twin (``solver="halley"``), the
    existing JAX least-squares / HJCD solvers, and VAMP's SIMD end-effector FK
    kernel versus pyroffi's JAX FK.

    Run CPU-only (the intended deployment) with:

        JAX_PLATFORMS=cpu python tests/bench_cpu_ik_fk.py

    Without that env var JAX will use a GPU if present for the JAX solvers,
    while QuIK / VAMP always run on the CPU (their FFI targets are CPU-only), so
    the comparison is only apples-to-apples under ``JAX_PLATFORMS=cpu``.

What is reported
    * Per-target IK latency (sequential, one pose at a time).
    * Batch IK throughput (all targets at once) where the backend supports it.
    * Solution quality: median position (mm) and rotation (deg) error, and the
      fraction of targets solved under a tolerance.
    * FK throughput: VAMP CPU eefk vs pyroffi JAX FK over a large config batch.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np


def _now() -> float:
    return time.perf_counter()


def _pose_matrix(pose7_wxyz_xyz) -> np.ndarray:
    from pyroffi.kinematics._dh import _wxyz_xyz_to_matrix

    return _wxyz_xyz_to_matrix(np.asarray(pose7_wxyz_xyz))


def _errors(T_sol: np.ndarray, T_tgt: np.ndarray) -> tuple[float, float]:
    pos = float(np.linalg.norm(T_sol[:3, 3] - T_tgt[:3, 3]))
    Re = T_sol[:3, :3].T @ T_tgt[:3, :3]
    ang = float(np.arccos(np.clip((np.trace(Re) - 1) / 2, -1, 1)))
    return pos, ang


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default="resources/ur5/ur5.urdf")
    ap.add_argument("--ee", default="ee_link")
    ap.add_argument("--n-targets", type=int, default=100)
    ap.add_argument("--num-seeds", type=int, default=32)
    ap.add_argument("--fk-batch", type=int, default=100_000)
    ap.add_argument("--pos-tol", type=float, default=1e-3, help="metres")
    ap.add_argument("--ang-tol", type=float, default=1e-2, help="radians")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import jaxlie
    import yourdfpy

    import pyroffi

    plat = jax.default_backend()
    print(f"JAX backend: {plat}   (set JAX_PLATFORMS=cpu for the CPU comparison)")

    urdf = yourdfpy.URDF.load(args.urdf)
    robot = pyroffi.Robot.from_urdf(urdf)
    ee_idx = robot.links.names.index(args.ee)
    n_act = int(robot.joints.num_actuated_joints)

    # Random reachable targets: FK of random within-limit configs.
    rng = np.random.default_rng(0)
    lower = np.asarray(robot.joints.lower_limits)
    upper = np.asarray(robot.joints.upper_limits)
    lower = np.where(np.isfinite(lower), lower, -np.pi)
    upper = np.where(np.isfinite(upper), upper, np.pi)
    q_targets = rng.uniform(lower, upper, size=(args.n_targets, n_act))
    fk_all = np.asarray(robot.forward_kinematics(jnp.asarray(q_targets)))
    T_targets = np.stack([_pose_matrix(fk_all[i, ee_idx]) for i in range(args.n_targets)])
    se3_targets = [jaxlie.SE3.from_matrix(jnp.asarray(T)) for T in T_targets]

    prev = (lower + upper) / 2

    def eval_solver(name, solve_one):
        """Time sequential solves and score quality."""
        # Warm up (compile).
        try:
            solve_one(se3_targets[0])
        except Exception as e:  # noqa: BLE001
            print(f"  {name:16s}  SKIPPED ({type(e).__name__}: {str(e)[:60]})")
            return
        t0 = _now()
        sols = []
        for T in se3_targets:
            q = solve_one(T)
            sols.append(np.asarray(q))
        wall = _now() - t0
        pos_errs, ang_errs = [], []
        for q, T in zip(sols, T_targets):
            p = np.asarray(robot.forward_kinematics(jnp.asarray(q)))
            pe, ae = _errors(_pose_matrix(p[ee_idx]), T)
            pos_errs.append(pe)
            ang_errs.append(ae)
        pos_errs = np.array(pos_errs)
        ang_errs = np.array(ang_errs)
        solved = np.mean((pos_errs < args.pos_tol) & (ang_errs < args.ang_tol))
        print(
            f"  {name:16s}  {1e3 * wall / args.n_targets:8.3f} ms/target   "
            f"pos {1e3 * np.median(pos_errs):7.3f} mm   "
            f"rot {np.degrees(np.median(ang_errs)):6.3f} deg   "
            f"solved {100 * solved:5.1f}%"
        )

    ns = args.num_seeds
    key = jax.random.PRNGKey(0)

    print(f"\n== Sequential IK latency ({args.n_targets} targets, {ns} seeds) ==")

    # QuIK (Halley) and QuIK (Newton/LM) share one compiled solver.
    from pyroffi.optimization_engines._quik_ik import QuIKSolver

    quik = None
    try:
        quik = QuIKSolver(robot, args.ee)
    except Exception as e:  # noqa: BLE001
        print(f"  QuIK unavailable: {type(e).__name__}: {str(e)[:80]}")

    if quik is not None:
        order = quik.model.actuated_order

        def make_quik(alg, lam=0.0):
            def solve_one(T):
                Tm = np.asarray(T.as_matrix())
                lo = np.where(np.isfinite(lower[order]), lower[order], -np.pi)
                hi = np.where(np.isfinite(upper[order]), upper[order], np.pi)
                seeds = np.concatenate(
                    [prev[order][None], rng.uniform(lo, hi, (ns, quik.dof))]
                )
                poses = jnp.broadcast_to(jnp.asarray(Tm, jnp.float32), (ns + 1, 4, 4))
                out = quik.solve_to_actuated(
                    poses, jnp.asarray(seeds, jnp.float32), algorithm=alg, lambda2=lam
                )
                best = int(np.argmin(np.asarray(out["error"])))
                return out["q_actuated"][best]

            return solve_one

        eval_solver("QuIK (Halley)", make_quik(0))
        eval_solver("QuIK (NR/LM)", make_quik(1, lam=1e-6))

    # Dispatcher-based JAX solvers.
    def dispatch(solver, **kw):
        def solve_one(T):
            return robot.inverse_kinematics(
                args.ee, T, rng_key=key, previous_cfg=jnp.asarray(prev),
                solver=solver, num_seeds=ns, **kw,
            )
        return solve_one

    eval_solver("JAX-Halley", dispatch("halley"))
    eval_solver("JAX-LS", dispatch("ls"))
    eval_solver("JAX-HJCD", dispatch("hjcd"))

    # ── Batch IK throughput (QuIK / JAX-Halley solve every target's best seed
    #    in one fused CPU call — the planner throughput number). ──────────────
    print(f"\n== Batch IK throughput ({args.n_targets} targets x {ns} seeds) ==")
    if quik is not None:
        order = quik.model.actuated_order
        lo = np.where(np.isfinite(lower[order]), lower[order], -np.pi)
        hi = np.where(np.isfinite(upper[order]), upper[order], np.pi)
        seeds = rng.uniform(lo, hi, (args.n_targets, ns, quik.dof)).astype(np.float32)
        poses = np.repeat(T_targets[:, None].astype(np.float32), ns, axis=1)
        poses_flat = jnp.asarray(poses.reshape(-1, 4, 4))
        seeds_flat = jnp.asarray(seeds.reshape(-1, quik.dof))
        out = quik.solve(poses_flat, seeds_flat)  # warm
        jax.block_until_ready(out["q"])
        t0 = _now()
        out = quik.solve(poses_flat, seeds_flat)
        jax.block_until_ready(out["q"])
        wall = _now() - t0
        print(
            f"  {'QuIK batch':16s}  {1e3 * wall:8.2f} ms total   "
            f"{1e6 * wall / (args.n_targets * ns):6.2f} us / (target,seed)"
        )

    # ── FK throughput: VAMP CPU eefk vs pyroffi JAX FK. ──────────────────────
    print(f"\n== FK throughput ({args.fk_batch} configs) ==")
    cfgs = rng.uniform(lower, upper, (args.fk_batch, n_act)).astype(np.float32)
    cfgs_j = jnp.asarray(cfgs)

    fk_jit = jax.jit(lambda c: robot.forward_kinematics(c)[:, ee_idx])
    r = fk_jit(cfgs_j)
    jax.block_until_ready(r)
    t0 = _now()
    r = fk_jit(cfgs_j)
    jax.block_until_ready(r)
    wall = _now() - t0
    print(f"  {'pyroffi JAX FK':16s}  {1e3 * wall:8.2f} ms   {args.fk_batch / wall / 1e6:6.2f} M cfg/s")

    try:
        from pyroffi.kinematics import make_vamp_cpu_fk

        srdf = args.urdf.replace(".urdf", ".srdf")
        vfk = make_vamp_cpu_fk(args.urdf, srdf if os.path.exists(srdf) else None)
        d = vfk.dimension
        c = cfgs[:, :d]
        r = vfk.ee_poses(jnp.asarray(c))
        jax.block_until_ready(r)
        t0 = _now()
        r = vfk.ee_poses(jnp.asarray(c))
        jax.block_until_ready(r)
        wall = _now() - t0
        print(f"  {'VAMP CPU FK':16s}  {1e3 * wall:8.2f} ms   {args.fk_batch / wall / 1e6:6.2f} M cfg/s")
    except Exception as e:  # noqa: BLE001
        print(f"  VAMP CPU FK unavailable: {type(e).__name__}: {str(e)[:80]}")


if __name__ == "__main__":
    main()
