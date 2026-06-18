"""Validate the fused FK + binary collision checker against the JAX SDF reference.

A configuration is collision-free iff every world signed distance and every
self-collision signed distance is positive.  The binary CUDA kernel must agree
with that verdict (away from the zero-distance boundary, where float32 vs x64
rounding can flip a sign).

The validation isolates the two paths so each test exercises a real mix of
collision-free and in-collision configs:

  * world path — self-collision disabled, obstacles placed in the arm workspace.
  * self path  — empty world, and self pairs that already overlap at the rest
    pose (a spherized-model artifact) are filtered out so folding the arm is what
    drives the verdict.

Run:
    python tests/test_binary_collision.py
"""

from __future__ import annotations

import dataclasses
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import yourdfpy

import pyroffi as pk
from pyroffi.collision import (
    Box,
    CUDABinaryCollisionChecker,
    RobotCollisionSpherized,
    Sphere,
)

RESOURCE_ROOT = pathlib.Path(__file__).resolve().parent.parent / "resources"
FINE_URDF = RESOURCE_ROOT / "panda" / "panda_spherized.urdf"
COARSE_URDF = RESOURCE_ROOT / "panda" / "panda_spherized_coarse.urdf"

# Configs whose closest distance is within this band straddle the collision
# boundary; float32-vs-x64 rounding can flip their verdict, so they are excluded
# from the strict-agreement assertion.
BOUNDARY = 2e-3

_FAR_WORLD = Sphere.from_center_and_radius(
    center=jnp.array([[100.0, 100.0, 100.0]]), radius=jnp.array([1e-3])
)


def _load():
    robot = pk.Robot.from_urdf(yourdfpy.URDF.load(str(FINE_URDF)))
    fine = RobotCollisionSpherized.from_urdf(yourdfpy.URDF.load(str(FINE_URDF)))
    return robot, fine


def _no_self(model):
    """Copy of a spherized model with no active self-collision pairs."""
    return dataclasses.replace(
        model,
        active_idx_i=jnp.zeros((0,), dtype=jnp.int32),
        active_idx_j=jnp.zeros((0,), dtype=jnp.int32),
    )


def _filter_rest_colliding_pairs(robot, model):
    """Drop self pairs that already penetrate at the rest pose (model artifacts).

    Uses the same all-pairs reduction as the kernel so the filter matches what
    the kernel will actually flag.
    """
    li = jnp.asarray(model.active_idx_i)
    lj = jnp.asarray(model.active_idx_j)
    rest = jnp.zeros((robot.joints.num_actuated_joints,), dtype=jnp.float32)
    coll = model.at_config(robot, rest)          # batch (S, N)
    ctr = coll.pose.translation()                # [S, N, 3]
    rad = coll.radius                            # [S, N]
    ci, ri = ctr[:, li], rad[:, li]
    cj, rj = ctr[:, lj], rad[:, lj]
    diff = ci[:, None, :, :] - cj[None, :, :, :]
    d = jnp.sqrt(jnp.sum(diff ** 2, axis=-1)) - (ri[:, None, :] + rj[None, :, :])
    valid = (ri[:, None, :] >= 0) & (rj[None, :, :] >= 0)
    d = jnp.where(valid, d, jnp.inf)
    per_pair = np.asarray(jnp.min(d, axis=(0, 1)))  # [P]
    keep = per_pair > 0.05
    return dataclasses.replace(
        model,
        active_idx_i=jnp.asarray(np.asarray(model.active_idx_i)[keep], dtype=jnp.int32),
        active_idx_j=jnp.asarray(np.asarray(model.active_idx_j)[keep], dtype=jnp.int32),
    )


def _world_margin(robot, model, cfg, world_geom):
    return jnp.min(
        jax.vmap(lambda c: model.compute_world_collision_distance(robot, c, world_geom))(cfg),
        axis=(-1, -2),
    )


def _self_margin(robot, model, cfg):
    """Per-config min self-distance over ALL sphere pairs of each active link pair.

    The library's ``compute_self_collision_distance`` only compares same-index
    spheres across links (diagonal in S); the binary kernel — like pRRTC —
    checks every Sᵢ×Sⱼ sphere pair, so validate against that full reduction.
    """
    li = jnp.asarray(model.active_idx_i)
    lj = jnp.asarray(model.active_idx_j)

    def one(c):
        coll = model.at_config(robot, c)             # batch (S, N)
        ctr = coll.pose.translation()                # [S, N, 3]
        rad = coll.radius                            # [S, N]
        ci, ri = ctr[:, li], rad[:, li]              # [S, P, 3], [S, P]
        cj, rj = ctr[:, lj], rad[:, lj]
        diff = ci[:, None, :, :] - cj[None, :, :, :]  # [S, S, P, 3]
        d = jnp.sqrt(jnp.sum(diff ** 2, axis=-1)) - (ri[:, None, :] + rj[None, :, :])
        valid = (ri[:, None, :] >= 0) & (rj[None, :, :] >= 0)
        d = jnp.where(valid, d, jnp.inf)
        return jnp.min(d, axis=(0, 1))                # [P]

    return jnp.min(jax.vmap(one)(cfg), axis=-1)


def _assert_agreement(label, margin, bin_free, require_mix=True):
    ref_free = np.asarray(margin) > 0.0
    bin_free = np.asarray(bin_free)
    decisive = np.abs(np.asarray(margin)) > BOUNDARY
    bad = int(np.sum((bin_free != ref_free) & decisive))
    n_free = int(np.sum(ref_free))
    print(
        f"[{label}] {len(ref_free) - bad}/{len(ref_free)} agree "
        f"(reference free: {n_free}/{len(ref_free)}); decisive mismatches: {bad}"
    )
    assert bad == 0, f"{label}: binary disagrees on {bad} decisive configs"
    if require_mix:
        assert 0 < n_free < len(ref_free), (
            f"{label}: vacuous test — {n_free}/{len(ref_free)} free (need a mix)"
        )
    return ref_free, bin_free


def test_world_path_matches_sdf():
    robot, fine = _load()
    n = robot.joints.num_actuated_joints
    cfg = jnp.asarray(
        np.random.RandomState(1).uniform(-1.2, 1.2, size=(1024, n)), dtype=jnp.float32
    )

    # Obstacles seeded across the arm workspace so a real mix of configs collide.
    world = Sphere.from_center_and_radius(
        center=jnp.array(
            [[0.3, 0.0, 0.6], [0.0, 0.35, 0.7], [-0.25, 0.0, 0.5], [0.2, -0.3, 0.8]]
        ),
        radius=jnp.array([0.13, 0.12, 0.12, 0.11]),
    )

    fine_noself = _no_self(fine)
    margin = _world_margin(robot, fine_noself, cfg, world)

    checker = CUDABinaryCollisionChecker(fine_noself)
    bin_free = checker.check_collision_free(robot, cfg, world)
    _assert_agreement("world", margin, bin_free)


def test_world_path_box_matches_sdf():
    robot, fine = _load()
    n = robot.joints.num_actuated_joints
    cfg = jnp.asarray(
        np.random.RandomState(2).uniform(-1.2, 1.2, size=(1024, n)), dtype=jnp.float32
    )
    world = Box.from_center_and_half_lengths(
        center=jnp.array([[0.3, 0.0, 0.6], [-0.2, 0.2, 0.55]]),
        half_lengths=jnp.array([[0.12, 0.12, 0.3], [0.1, 0.1, 0.2]]),
    )
    fine_noself = _no_self(fine)
    margin = _world_margin(robot, fine_noself, cfg, world)

    checker = CUDABinaryCollisionChecker(fine_noself)
    bin_free = checker.check_collision_free(robot, cfg, world)
    _assert_agreement("world-box", margin, bin_free)


def test_self_path_matches_sdf():
    robot, fine = _load()
    fine_f = _filter_rest_colliding_pairs(robot, fine)
    n = robot.joints.num_actuated_joints
    # Wide range to fold the arm and induce genuine self-collisions.
    cfg = jnp.asarray(
        np.random.RandomState(3).uniform(-2.6, 2.6, size=(1024, n)), dtype=jnp.float32
    )
    margin = _self_margin(robot, fine_f, cfg)

    checker = CUDABinaryCollisionChecker(fine_f)
    bin_free = checker.check_collision_free(robot, cfg, _FAR_WORLD)
    _assert_agreement("self", margin, bin_free)


def test_coarse_guard_is_sound():
    """The coarse guard must never miss a collision the fine pass would catch."""
    robot, fine = _load()
    fine_noself = _no_self(fine)
    coarse = RobotCollisionSpherized.from_urdf(yourdfpy.URDF.load(str(COARSE_URDF)))
    coarse_noself = _no_self(coarse)

    n = robot.joints.num_actuated_joints
    cfg = jnp.asarray(
        np.random.RandomState(4).uniform(-1.2, 1.2, size=(1024, n)), dtype=jnp.float32
    )
    world = Sphere.from_center_and_radius(
        center=jnp.array([[0.3, 0.0, 0.6], [0.0, 0.35, 0.7], [-0.25, 0.0, 0.5]]),
        radius=jnp.array([0.13, 0.12, 0.12]),
    )

    fine_only = CUDABinaryCollisionChecker(fine_noself)
    guarded = CUDABinaryCollisionChecker(fine_noself, coarse_inner=coarse_noself)

    f_free = np.asarray(fine_only.check_collision_free(robot, cfg, world))
    g_free = np.asarray(guarded.check_collision_free(robot, cfg, world))

    missed = int(np.sum(g_free & ~f_free))  # guard said free, fine said collision
    print(
        f"[coarse-guard] matches fine-only on {int(np.sum(f_free == g_free))}/{len(f_free)}; "
        f"missed collisions: {missed}"
    )
    assert missed == 0, f"coarse guard missed {missed} collisions (not enclosing?)"


def test_edge_validation():
    robot, fine = _load()
    fine_noself = _no_self(fine)
    n = robot.joints.num_actuated_joints
    world = Sphere.from_center_and_radius(
        center=jnp.array([[0.3, 0.0, 0.6], [0.0, 0.35, 0.7]]),
        radius=jnp.array([0.15, 0.14]),
    )
    rng = np.random.RandomState(5)
    E, G = 128, 8
    a = jnp.asarray(rng.uniform(-1.2, 1.2, size=(E, n)), dtype=jnp.float32)
    b = jnp.asarray(rng.uniform(-1.2, 1.2, size=(E, n)), dtype=jnp.float32)
    ts = jnp.linspace(0.0, 1.0, G)
    edges = a[:, None, :] * (1 - ts)[None, :, None] + b[:, None, :] * ts[None, :, None]

    checker = CUDABinaryCollisionChecker(fine_noself)
    edge_ok = np.asarray(checker.check_edges_collision_free(robot, edges, world))

    pts = edges.reshape(E * G, n)
    pt_free = np.asarray(checker.check_collision_free(robot, pts, world)).reshape(E, G)
    ref_edge_ok = pt_free.all(axis=1)
    assert np.array_equal(edge_ok, ref_edge_ok)
    assert 0 < int(edge_ok.sum()) < E, "edge test vacuous — need a mix of valid/invalid"
    print(f"[edges] {int(edge_ok.sum())}/{E} edges valid; AND-reduction consistent")


if __name__ == "__main__":
    test_world_path_matches_sdf()
    test_world_path_box_matches_sdf()
    test_self_path_matches_sdf()
    test_coarse_guard_is_sound()
    test_edge_validation()
    print("\nAll binary-collision checks passed.")
