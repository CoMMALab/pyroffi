"""Example: CPU collision checking with the JIT-compiled VAMP backend.

``VAMPCPUCollisionChecker`` specialises VAMP's SIMD ``fkcc`` collision routine to
a concrete robot at runtime: cricket parses the URDF, emits a
``vamp::robots::<Robot>`` struct, and JIT-compiles a binary collision checker for
it (cached on disk for reuse).  Forward kinematics is baked into the binary, so
you only pass joint configurations — no pre-built pyroffi collision model.

This script:
  1. builds the checker for the Panda (first call compiles + caches; later calls
     reuse the cached binary),
  2. checks a batch of random configurations against a Sphere + Box world,
  3. reports how many are collision-free and the throughput.

Run (inside the `pyroffi` conda env, with cricket built — see
build_kernels/build_cricket_jit.sh):

    python examples/13_00_vamp_cpu_collision.py
"""

import os
import sys
import time
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import jax.numpy as jnp
import numpy as np

from pyroffi.collision import Box, Sphere, VAMPCPUCollisionChecker

REPO_ROOT = Path(__file__).resolve().parents[1]
SPHERIZED_URDF = REPO_ROOT / "resources" / "panda" / "panda_spherized.urdf"
SRDF = REPO_ROOT / "resources" / "panda" / "panda.srdf"


def main() -> None:
    print("Building VAMP CPU collision checker (first run JIT-compiles, then caches)...")
    t0 = time.perf_counter()
    checker = VAMPCPUCollisionChecker(SPHERIZED_URDF, srdf_path=SRDF)
    print(f"  ready in {time.perf_counter() - t0:.2f}s  "
          f"(dim={checker.dimension}, n_spheres={checker.n_spheres})")

    n = checker.dimension
    rng = np.random.RandomState(0)
    cfg = jnp.asarray(rng.uniform(-1.5, 1.5, size=(4096, n)), dtype=jnp.float32)

    # A mixed world: a few spheres and a box in the arm's workspace.
    spheres = Sphere.from_center_and_radius(
        center=jnp.array([[0.35, 0.0, 0.6], [0.0, 0.4, 0.7], [-0.3, 0.0, 0.5]]),
        radius=jnp.array([0.13, 0.12, 0.12]),
    )
    box = Box.from_center_and_half_lengths(
        center=jnp.array([[0.4, 0.0, 0.4]]),
        half_lengths=jnp.array([[0.1, 0.4, 0.1]]),
    )

    for name, world in (("sphere world", spheres), ("box world", box)):
        # warm up, then time
        free = np.asarray(checker.check_collision_free(None, cfg, world))
        t0 = time.perf_counter()
        for _ in range(5):
            free = np.asarray(checker.check_collision_free(None, cfg, world))
        dt = (time.perf_counter() - t0) / 5
        print(f"\n[{name}] {int(free.sum())}/{len(free)} configs collision-free")
        print(f"  {dt * 1e3:.2f} ms/call  ({dt * 1e6 / len(free):.2f} us/config) "
              f"for {len(free)} configs")

    # Single-config check (returns a scalar bool).
    home = jnp.zeros((n,), dtype=jnp.float32)
    print(f"\nhome configuration collision-free in empty box world: "
          f"{bool(checker.check_collision_free(None, home, box))}")


if __name__ == "__main__":
    main()
