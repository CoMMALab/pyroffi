"""Example: batch edge validation (+ point clouds) with the VAMP CPU backend.

Edge validation is the headline use of the VAMP backend for sampling-based
planning: given many candidate motions (edges), decide which are entirely
collision-free.  VAMP discretises each edge internally at the robot's planning
resolution and checks it with its SIMD ``fkcc`` routine, parallelised across the
batch with OpenMP.

``check_edges_collision_free`` takes the two endpoints of each edge in the
second-to-last axis (shape ``[*batch, 2, n_act]``) — VAMP fills in the
interior — and returns one verdict per edge.

This script also shows VAMP's CAPT (Collision-Affording Point Tree) path: a
point-cloud obstacle passed via ``point_cloud=`` with a per-point radius.

Run (inside the `pyroffi` conda env, with cricket built):

    python examples/13_01_vamp_edge_validation.py
"""

import os
import sys
import time
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import jax.numpy as jnp
import numpy as np

from pyroffi.collision import Sphere, VAMPCPUCollisionChecker

REPO_ROOT = Path(__file__).resolve().parents[1]
SPHERIZED_URDF = REPO_ROOT / "resources" / "panda" / "panda_spherized.urdf"
SRDF = REPO_ROOT / "resources" / "panda" / "panda.srdf"


def main() -> None:
    checker = VAMPCPUCollisionChecker(SPHERIZED_URDF, srdf_path=SRDF)
    n = checker.dimension
    rng = np.random.RandomState(5)

    world = Sphere.from_center_and_radius(
        center=jnp.array([[0.3, 0.0, 0.6], [0.0, 0.35, 0.7]]),
        radius=jnp.array([0.15, 0.14]),
    )

    # ── Batch edge validation ───────────────────────────────────────────────
    E = 4096
    a = jnp.asarray(rng.uniform(-1.2, 1.2, size=(E, n)), dtype=jnp.float32)
    b = jnp.asarray(rng.uniform(-1.2, 1.2, size=(E, n)), dtype=jnp.float32)
    edges = jnp.stack([a, b], axis=1)  # [E, 2, n]

    valid = np.asarray(checker.check_edges_collision_free(None, edges, world))  # warm up
    t0 = time.perf_counter()
    for _ in range(5):
        valid = np.asarray(checker.check_edges_collision_free(None, edges, world))
    dt = (time.perf_counter() - t0) / 5
    print(f"[edges] {int(valid.sum())}/{E} edges collision-free")
    print(f"  {dt * 1e3:.2f} ms/call  ({dt * 1e6 / E:.2f} us/edge) for {E} edges")

    # Sanity: VAMP samples (0, 1], so a valid edge must have its *goal* endpoint
    # collision-free (the start is assumed pre-validated by the planner).
    b_free = np.asarray(checker.check_collision_free(None, b, world))
    assert np.all(~valid | b_free)
    print("  consistency OK: no edge is valid with a colliding goal endpoint")

    # ── Point-cloud (CAPT) obstacle ─────────────────────────────────────────
    far = Sphere.from_center_and_radius(
        center=jnp.array([[100.0, 100.0, 100.0]]), radius=jnp.array([1e-3])
    )
    cfg = jnp.asarray(rng.uniform(-1.2, 1.2, size=(2048, n)), dtype=jnp.float32)
    gx, gz = np.meshgrid(np.linspace(0.15, 0.5, 25), np.linspace(0.2, 1.0, 25))
    cloud = np.stack([gx.ravel(), np.zeros(gx.size), gz.ravel()], axis=1).astype(np.float32)

    base = int(np.asarray(checker.check_collision_free(None, cfg, far)).sum())
    with_pc = int(
        np.asarray(
            checker.check_collision_free(
                None, cfg, far, point_cloud=jnp.asarray(cloud), capt=(0.0, 1.0, 0.04)
            )
        ).sum()
    )
    print(f"\n[CAPT point cloud] {cloud.shape[0]} points, r_point=0.04")
    print(f"  free without cloud: {base}/{cfg.shape[0]}")
    print(f"  free with cloud:    {with_pc}/{cfg.shape[0]}  "
          f"(point-cloud wall removed {base - with_pc} configs)")


if __name__ == "__main__":
    main()
