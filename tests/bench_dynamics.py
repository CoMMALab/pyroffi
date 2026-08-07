"""Benchmark: pure-JAX vs GRiD CUDA dynamics across batch sizes.

Usage:  python tests/bench_dynamics.py [urdf_path]
"""

import sys
import time

import jax
import jax.numpy as jnp
import yourdfpy

import pyroffi
from pyroffi.dynamics import GRiDDynamics

BATCH_SIZES = [1, 16, 64, 256, 1024, 4096, 16384]
N_ITERS = 50


def _median_ms(fn, *args):
    fn(*args)[0].block_until_ready() if isinstance(fn(*args), tuple) else fn(
        *args
    ).block_until_ready()
    times = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter()
        fn(*args).block_until_ready()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times) // 2]


def main(urdf_path: str = "resources/panda/panda_spherized.urdf") -> None:
    urdf = yourdfpy.URDF.load(urdf_path, load_meshes=False)
    robot = pyroffi.Robot.from_urdf(urdf)
    gd = GRiDDynamics(urdf)
    n = gd.num_dof

    ops = {
        "ID  ": (jax.jit(robot.inverse_dynamics), jax.jit(gd.inverse_dynamics)),
        "FD  ": (jax.jit(robot.forward_dynamics), jax.jit(gd.forward_dynamics)),
        "IDdu": (
            jax.jit(
                lambda q, qd, x: jax.vmap(
                    jax.jacobian(lambda q_, qd_, x_: robot.inverse_dynamics(q_, qd_, x_), argnums=(0, 1))
                )(q, qd, x)[0]
            ),
            jax.jit(gd.inverse_dynamics_gradient),
        ),
    }

    print(f"robot: {urdf_path} (n={n});  median of {N_ITERS} iters, ms")
    print(f"{'op':<6}{'batch':>8}{'jax':>12}{'grid-cuda':>12}{'speedup':>10}")
    key = jax.random.PRNGKey(0)
    for B in BATCH_SIZES:
        q, qd, x = jax.random.normal(key, (3, B, n), dtype=jnp.float32)
        for name, (f_jax, f_cuda) in ops.items():
            t_jax = _median_ms(f_jax, q, qd, x)
            t_cuda = _median_ms(f_cuda, q, qd, x)
            print(
                f"{name:<6}{B:>8}{t_jax:>12.3f}{t_cuda:>12.3f}{t_jax / t_cuda:>9.1f}x"
            )


if __name__ == "__main__":
    main(*sys.argv[1:])
