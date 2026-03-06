"""Benchmark and correctness test: CUDA FK vs JAX FK on the Panda robot.

Usage:
    python tests/test_fk_cuda.py

Prerequisites:
    1. A CUDA-capable GPU must be available.
    2. The CUDA FK library must be compiled:
           bash src/pyronot/cuda_kernels/build_fk_cuda.sh
    3. robot_descriptions must be installed (pip install robot_descriptions).
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import pyronot as pk
from robot_descriptions.loaders.yourdfpy import load_robot_description

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BATCH_SIZES = [1, 16, 64, 256, 1024, 4096, 8192, 16384, 32768, 65536]
N_WARMUP    = 5    # JIT warm-up repetitions (discarded)
N_TIMED     = 50   # timed repetitions per implementation
ATOL        = 1e-4 # absolute tolerance for numerical comparison (float32)
RTOL        = 1e-4 # relative tolerance

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _time_fn(fn, *args, n: int = N_TIMED) -> float:
    """Return median wall-clock time (seconds) over *n* calls.

    Each call is followed by block_until_ready() so async JAX dispatch is
    fully accounted for.
    """
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _print_row(label: str, batch: int, jax_ms: float, cuda_ms: float) -> None:
    speedup = jax_ms / cuda_ms if cuda_ms > 0 else float("nan")
    print(f"  {label:<10} batch={batch:<6} "
          f"JAX: {jax_ms*1e3:8.3f} ms   "
          f"CUDA: {cuda_ms*1e3:8.3f} ms   "
          f"speedup: {speedup:.2f}x")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(robot_name: str) -> None:
    print("=" * 70)
    print(f"FK correctness & performance: JAX vs CUDA  ({robot_name} robot)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load robot
    # ------------------------------------------------------------------
    print("\nLoading robot description ...")
    urdf = load_robot_description(f"{robot_name}_description")
    robot = pk.Robot.from_urdf(urdf)
    n_act = robot.joints.num_actuated_joints
    print(f"  {n_act} actuated joints, "
          f"{robot.joints.num_joints} total joints, "
          f"{robot.links.num_links} links")

    # ------------------------------------------------------------------
    # Build JIT-compiled callables for both backends
    # ------------------------------------------------------------------
    fk_jax  = jax.jit(lambda cfg: robot.forward_kinematics(cfg, use_cuda=False))
    fk_cuda = jax.jit(lambda cfg: robot.forward_kinematics(cfg, use_cuda=True))

    rng = np.random.default_rng(42)
    lo  = np.array(robot.joints.lower_limits)
    hi  = np.array(robot.joints.upper_limits)

    all_passed = True

    print("\n{:<10} {:<8} {:<16} {:<16} {:<12} {}".format(
        "Impl", "Batch", "JAX (ms)", "CUDA (ms)", "Speedup", "Max |err|"))
    print("-" * 70)

    for batch in BATCH_SIZES:
        cfg_np  = rng.uniform(lo, hi, size=(batch, n_act)).astype(np.float32)
        cfg_jax = jnp.array(cfg_np)

        # ---- warm-up (triggers JIT compilation) ----
        for _ in range(N_WARMUP):
            jax.block_until_ready(fk_jax(cfg_jax))
            jax.block_until_ready(fk_cuda(cfg_jax))

        # ---- correctness ----
        out_jax  = np.array(fk_jax(cfg_jax))
        out_cuda = np.array(fk_cuda(cfg_jax))
        max_err  = float(np.abs(out_jax - out_cuda).max())
        passed   = np.allclose(out_jax, out_cuda, atol=ATOL, rtol=RTOL)
        all_passed &= passed

        # ---- timing ----
        t_jax  = _time_fn(fk_jax,  cfg_jax)
        t_cuda = _time_fn(fk_cuda, cfg_jax)
        speedup = t_jax / t_cuda if t_cuda > 0 else float("nan")

        status = "OK" if passed else "FAIL"
        print(f"  {'JAX':<8} {batch:<8} {t_jax*1e3:>12.3f}   "
              f"{t_cuda*1e3:>12.3f}   {speedup:>8.2f}x   "
              f"|err|={max_err:.2e}  [{status}]")

    print("-" * 70)
    print()
    if all_passed:
        print("PASSED: CUDA and JAX outputs agree within tolerance "
              f"(atol={ATOL}, rtol={RTOL}).")
    else:
        print("FAILED: one or more batch sizes exceeded the tolerance threshold.")
        # raise SystemExit(1)


if __name__ == "__main__":
    main("panda")
    main("fetch")
    main("baxter")
