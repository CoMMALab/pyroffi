"""Solve the shared problem set with pyroffi's CUDA IK (run under `pyroffi`)."""
from __future__ import annotations

import pathlib
import time

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import yourdfpy

import pyroffi as pk
from pyroffi.collision import Box, RobotCollisionSpherized
from _problems import OBSTACLE_CENTER, OBSTACLE_DIMS, load, save_result

REPS = 5
RESOURCE_ROOT = pathlib.Path(__file__).resolve().parents[2] / "resources"


def main() -> None:
    q_ref, target_wxyz_xyz, ee = load()
    n = q_ref.shape[0]

    urdf = yourdfpy.URDF.load(str(RESOURCE_ROOT / "panda" / "panda_spherized.urdf"))
    robot = pk.Robot.from_urdf(urdf)
    coll = RobotCollisionSpherized.from_urdf(
        urdf, srdf_path=str(RESOURCE_ROOT / "panda" / "panda.srdf"))
    obstacle = Box.from_center_and_dimensions(
        jnp.asarray(OBSTACLE_CENTER, jnp.float32)[None],
        jnp.float32(OBSTACLE_DIMS[0])[None],
        jnp.float32(OBSTACLE_DIMS[1])[None],
        jnp.float32(OBSTACLE_DIMS[2])[None])

    targets = jaxlie.SE3(jnp.asarray(target_wxyz_xyz, jnp.float32))
    prev = jnp.zeros((n, q_ref.shape[1]), jnp.float32)

    from pyroffi.optimization_engines._ls_ik import ls_ik_solve_cuda_batch
    from pyroffi.optimization_engines._sqp_ik import sqp_ik_solve_cuda_batch
    from pyroffi.optimization_engines._mppi_ik import mppi_ik_solve_cuda_batch

    solvers = {
        # The apples-to-apples row: MPPI particles + L-BFGS is cuRobo's
        # architecture (particle_opt + newton_base).
        "pyroffi_mppi": mppi_ik_solve_cuda_batch,
        "pyroffi_ls": ls_ik_solve_cuda_batch,
        "pyroffi_sqp": sqp_ik_solve_cuda_batch,
    }
    kwargs = dict(collision_checker=coll, collision_world=obstacle,
                  collision_free=True)

    for name, fn in solvers.items():
        def run():
            return fn(robot, ee, targets, rng_key=jax.random.PRNGKey(0),
                      previous_cfgs=prev, **kwargs)

        out = run()                      # warm up with the EXACT timed config;
        jax.block_until_ready(out)       # a mismatched warmup times compilation
        best = float("inf")
        for _ in range(REPS):
            t0 = time.perf_counter()
            out = run()
            jax.block_until_ready(out)
            best = min(best, time.perf_counter() - t0)

        cfg = np.asarray(getattr(out, "cfg", out))
        if cfg.ndim == 3:
            cfg = cfg[:, 0]
        save_result(name, cfg=cfg.astype(np.float64),
                    seconds=np.array(best), n_problems=np.array(n))
        print(f"{name:14} {best*1e3:8.2f} ms   {n/best/1e3:7.1f} kIK/s")


if __name__ == "__main__":
    main()
