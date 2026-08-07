"""CPU-accelerated kinematics & planning with QuIK and VAMP.

pyroffi's flagship solvers are CUDA kernels, but a lot of real deployments have
no GPU.  This example shows the **CPU-only** stack that pyroffi can assemble from
two external libraries, all behind the usual JAX API:

  * QuIK   — a C++ Halley's-method inverse-kinematics solver (external/QuIK),
             wired in as ``robot.inverse_kinematics(..., solver="quik")``.  Its
             pure-JAX twin ``solver="halley"`` runs the *same* algorithm on any
             JAX platform, for comparison.
  * VAMP   — its SIMD end-effector forward-kinematics kernel
             (``pyroffi.kinematics.make_vamp_cpu_fk``) and its SIMD collision
             checker (``pyroffi.collision.make_vamp_cpu_checker``).

Run it CPU-only (the intended deployment) with:

    JAX_PLATFORMS=cpu python examples/15_00_cpu_accelerated_planning.py

Everything below runs on the CPU: QuIK and VAMP are CPU FFI kernels, and
``JAX_PLATFORMS=cpu`` keeps the JAX FK / IK on the CPU too, so a machine with no
CUDA device can still do fast IK, FK and collision checking for simple serial
arms.
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import yourdfpy

import pyroffi
from pyroffi.kinematics import make_vamp_cpu_fk
from pyroffi.kinematics._dh import _wxyz_xyz_to_matrix

REPO = Path(__file__).resolve().parents[1]
# panda_spherized ships with resolvable meshes, so cricket can codegen VAMP's FK
# kernel from it; its chain is also standard-DH representable for QuIK.
URDF = REPO / "resources" / "panda" / "panda_spherized.urdf"
SRDF = REPO / "resources" / "panda" / "panda.srdf"
EE = "panda_hand"


def pose_err(q, T_tgt, robot, ee_idx):
    p = np.asarray(robot.forward_kinematics(jnp.asarray(q)))
    T = _wxyz_xyz_to_matrix(p[ee_idx])
    pos = np.linalg.norm(T[:3, 3] - T_tgt[:3, 3])
    Re = T[:3, :3].T @ T_tgt[:3, :3]
    ang = np.arccos(np.clip((np.trace(Re) - 1) / 2, -1, 1))
    return pos, ang


def main():
    print(f"JAX backend: {jax.default_backend()}  "
          "(run with JAX_PLATFORMS=cpu for the CPU-only stack)\n")

    urdf = yourdfpy.URDF.load(str(URDF))
    robot = pyroffi.Robot.from_urdf(urdf)
    ee_idx = robot.links.names.index(EE)
    n_act = int(robot.joints.num_actuated_joints)

    # A reachable target: FK of a known within-limits configuration.
    lower0 = np.where(np.isfinite(np.asarray(robot.joints.lower_limits)),
                      np.asarray(robot.joints.lower_limits), -np.pi)
    upper0 = np.where(np.isfinite(np.asarray(robot.joints.upper_limits)),
                      np.asarray(robot.joints.upper_limits), np.pi)
    q_ref = 0.5 * (lower0 + upper0) + 0.2 * (upper0 - lower0)
    T_tgt = _wxyz_xyz_to_matrix(np.asarray(robot.forward_kinematics(jnp.asarray(q_ref)))[ee_idx])
    target = jaxlie.SE3.from_matrix(jnp.asarray(T_tgt))
    prev = jnp.asarray((np.asarray(robot.joints.lower_limits)
                        + np.asarray(robot.joints.upper_limits)) / 2)

    # ── Inverse kinematics: QuIK (C++) vs its JAX twin ───────────────────────
    print("Inverse kinematics (same target, 32 seeds):")
    for solver in ("quik", "halley"):
        # Warm up (DH extraction + kernel compile / JIT trace happen once).
        robot.inverse_kinematics(EE, target, previous_cfg=prev, solver=solver)
        t0 = time.perf_counter()
        q = robot.inverse_kinematics(EE, target, previous_cfg=prev, solver=solver)
        jax.block_until_ready(q)
        dt = time.perf_counter() - t0
        pos, ang = pose_err(q, T_tgt, robot, ee_idx)
        print(f"  solver={solver:7s}  {1e3 * dt:7.2f} ms   "
              f"pos err {1e3 * pos:.4f} mm   rot err {np.degrees(ang):.4f} deg")

    # ── Forward kinematics: VAMP SIMD kernel vs pyroffi JAX FK ───────────────
    print("\nForward kinematics throughput (100k configs):")
    lower = np.where(np.isfinite(np.asarray(robot.joints.lower_limits)),
                     np.asarray(robot.joints.lower_limits), -np.pi)
    upper = np.where(np.isfinite(np.asarray(robot.joints.upper_limits)),
                     np.asarray(robot.joints.upper_limits), np.pi)
    cfgs = np.random.default_rng(0).uniform(lower, upper, (100_000, n_act)).astype(np.float32)
    cfgs_j = jnp.asarray(cfgs)

    fk_jit = jax.jit(lambda c: robot.forward_kinematics(c)[:, ee_idx])
    jax.block_until_ready(fk_jit(cfgs_j))
    t0 = time.perf_counter()
    jax.block_until_ready(fk_jit(cfgs_j))
    dt = time.perf_counter() - t0
    print(f"  pyroffi JAX FK   {1e3 * dt:7.2f} ms   {cfgs.shape[0] / dt / 1e6:.2f} M cfg/s")

    try:
        vfk = make_vamp_cpu_fk(str(URDF), str(SRDF) if SRDF.exists() else None)
        c = jnp.asarray(cfgs[:, : vfk.dimension])
        jax.block_until_ready(vfk.ee_poses(c))
        t0 = time.perf_counter()
        jax.block_until_ready(vfk.ee_poses(c))
        dt = time.perf_counter() - t0
        print(f"  VAMP CPU FK      {1e3 * dt:7.2f} ms   {cfgs.shape[0] / dt / 1e6:.2f} M cfg/s")
    except Exception as e:  # noqa: BLE001  (VAMP/cricket optional)
        print(f"  VAMP CPU FK unavailable: {type(e).__name__}: {e}")

    print("\nFor a fuller comparison against the JAX/CUDA solvers, run:")
    print("    JAX_PLATFORMS=cpu python tests/bench_cpu_ik_fk.py")


if __name__ == "__main__":
    main()
