"""Analytic IK on the GPU: GRiM's closed-form kernel (``grim.motion.analytic_ik``).

GRiM bakes the arm geometry into a per-robot build, so a solve takes the build's FFI target
(:func:`build_target`, outside jit) rather than a geometry blob. This module adds what the
JAX front end needs on top: obstacle flattening, zero-derivative declaration, and a
``custom_vmap`` rule that folds a vmap into the kernel's own batch.
"""

from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from ._batching import constant_wrt_autodiff


def build_target(robot, ee_link_index: int) -> str:
    """GRiM's analytic-IK kernel for the chain ending at link ``ee_link_index``."""
    from .. import _grim

    _grim.ensure_importable()
    from grim.motion import analytic_ik

    joint = int(robot._backends._parent_joint_indices[int(ee_link_index)])
    return analytic_ik.build(_grim.motion_robot(robot), joint)


class _Coll:
    """Minimal carrier matching CollisionData's field names, for traced arrays."""

    __slots__ = ("spheres_home", "sphere_joint", "self_pairs")

    def __init__(self, spheres_home, sphere_joint, self_pairs):
        self.spheres_home = spheres_home
        self.sphere_joint = sphere_joint
        self.self_pairs = self_pairs


@lru_cache(maxsize=1)
def _empty_collision_buffers():
    """Cached zero-length device buffers for the no-collision path."""
    return (jnp.zeros((0, 4), jnp.float32),
            jnp.zeros((0,), jnp.int32),
            jnp.zeros((0, 2), jnp.int32))


def world_arrays(world):
    """Obstacle buffers from any obstacle description the suite accepts.

    The iterative solvers take ``collision_world`` as a geometry PRIMITIVE (a
    Box, Sphere, Capsule, HalfSpace, or a sequence of them). This path used to
    accept only a pre-flattened WorldGeometry wrapper, so passing the ordinary
    input raised ``AttributeError: 'Box' object has no attribute 'spheres'`` --
    an interface split with no reason behind it. Both forms are handled here, at
    the FFI boundary where the other buffers are marshalled, so one obstacle
    description works across every solver.
    """
    if world is None:
        return _empty_world()
    if all(hasattr(world, a) for a in ("spheres", "capsules", "boxes", "halfspaces")):
        return (jnp.asarray(world.spheres, jnp.float32),
                jnp.asarray(world.capsules, jnp.float32),
                jnp.asarray(world.boxes, jnp.float32),
                jnp.asarray(world.halfspaces, jnp.float32))

    from ..collision._cuda_collision import _extract_world_arrays
    items = world if isinstance(world, (list, tuple)) else (world,)
    chunks = ([], [], [], [])
    for w in items:
        for dst, arr in zip(chunks, _extract_world_arrays(w)):
            if arr.shape[0] > 0:
                dst.append(arr)
    return tuple(
        jnp.concatenate(c, axis=0).astype(jnp.float32) if c else empty
        for c, empty in zip(chunks, _empty_world()))


@lru_cache(maxsize=1)
def _empty_world():
    """Cached zero-length buffers for all four world primitive types."""
    return (jnp.zeros((0, 4), jnp.float32), jnp.zeros((0, 7), jnp.float32),
            jnp.zeros((0, 15), jnp.float32), jnp.zeros((0, 6), jnp.float32))


def _analytic_ik_cuda_raw(
    target,
    targets,
    q7_samples,
    previous_cfg=None,
    collision=None,
    world_spheres=None,
    *,
    respect_limits: bool = True,
    err_tol: float = 1e-4,
    margin: float = 0.005,
):
    """Batched analytic IK on the GPU.

    Args:
        target: FFI target from :func:`build_target`.
        targets: ``[B, 4, 4]`` target poses.
        q7_samples: ``[S]`` redundancy-parameter values to try.
        previous_cfg: optional ``[B, 7]``; when given the returned branch is the
            valid one closest to it in joint space (continuity resolution)
            rather than the lowest-error one.
        collision: spheres_home / sphere_joint / self_pairs carrier, or None.
        world_spheres: the four world buffers, or None.
        respect_limits: discard branches outside the joint limits.
        err_tol: pose-error threshold below which a branch counts as a solution.

    Returns:
        ``(q[B, 7], err[B], found[B], clearance[B])`` with ``found`` a boolean array. When
        ``found`` is False, ``q`` still holds the lowest-error near-miss.
    """
    from grim.motion import analytic_ik
    from grim.motion.ik import World

    targets = jnp.asarray(targets, jnp.float32)
    if targets.ndim == 2:
        targets = targets[None, ...]
    use_prev = previous_cfg is not None
    sph_home, sph_joint, pairs = (_empty_collision_buffers() if collision is None else
                                  (collision.spheres_home, collision.sphere_joint,
                                   collision.self_pairs))
    world = _empty_world() if world_spheres is None else world_spheres

    # Zero-derivative by declaration; the implicit rule supplies the real gradient. Every
    # array is an operand of the wrapped call -- see constant_wrt_autodiff.
    def _run(targets, q7, prev, sph_home, sph_joint, pairs, *world):
        return analytic_ik.run(target, targets, q7, prev if use_prev else None,
                               spheres_home=sph_home, sphere_joint=sph_joint,
                               self_pairs=pairs, world=World(*world),
                               respect_limits=bool(respect_limits), err_tol=float(err_tol),
                               margin=float(margin))

    prev = (jnp.zeros((targets.shape[0], 7), jnp.float32) if previous_cfg is None
            else previous_cfg)
    q, err, found, clearance = constant_wrt_autodiff(_run)(
        targets, q7_samples, prev, sph_home, sph_joint, pairs, *world)
    return q, err, found.astype(bool), clearance


# --------------------------------------------------------------------------- #
# vmap rule: fold the mapped axis into the kernel's own batch dimension
# --------------------------------------------------------------------------- #
# The kernel already takes a leading `[B, 4, 4]` batch, so a `vmap` over targets
# should become ONE fused launch rather than a Python-level loop. jax.ffi's
# default `vmap_method="sequential"` would do the latter, which for a kernel
# costing ~2 ms of launch overhead until batch ~64 is exactly the wrong trade.
# This mirrors the `_batchable` pattern the GRiD dynamics wrappers use.
#
# `custom_vmap` binds positional-only arguments, so the solver *options* are
# baked into a cached closure rather than passed as keywords. They are
# configuration, not traced data, so this costs nothing and keeps the public
# signature keyword-friendly.


def _flatten_leading(x, ndim_core):
    """Merge all leading axes of ``x`` into one, keeping ``ndim_core`` trailing."""
    core = x.shape[x.ndim - ndim_core:]
    return x.reshape((-1,) + core), x.shape[: x.ndim - ndim_core]


# Cached on the *hashable* options only. The collision buffers are passed as
# real arguments rather than closed over: they are arrays, so baking them into a
# cache key is impossible and closing over them would silently reuse the first
# scene's geometry for every later call — a stale-obstacle bug that produces
# plausible wrong answers rather than an error.
@lru_cache(maxsize=16)
def _make_batched(target: str, respect_limits: bool, err_tol: float, use_prev: bool,
                  margin: float, has_collision: bool):
    @jax.custom_batching.custom_vmap
    def f(targets, q7_samples, prev,
          sph_home, sph_joint, pairs, w_sph, w_cap, w_box, w_hs):
        coll = _Coll(sph_home, sph_joint, pairs) if has_collision else None
        return _analytic_ik_cuda_raw(
            target, targets, q7_samples, prev if use_prev else None,
            coll, (w_sph, w_cap, w_box, w_hs),
            respect_limits=respect_limits, err_tol=err_tol, margin=margin)

    @f.def_vmap
    def _rule(axis_size, in_batched, targets, q7_samples, prev,
              sph_home, sph_joint, pairs, w_sph, w_cap, w_box, w_hs):
        tgt_b, q7_b, prev_b = in_batched[:3]
        coll_b = any(in_batched[3:])

        # The q7 sweep and the collision scene are shared across a vmap in
        # every sane use; a per-example one would need its own launch, so
        # refuse loudly rather than silently applying example 0's to all.
        if q7_b or coll_b:
            raise NotImplementedError(
                "analytic_ik_cuda: vmap over the q7 samples or "
                "collision scene is not supported (one per launch). vmap over "
                "`targets`.")

        if not tgt_b:
            targets = jnp.broadcast_to(targets, (axis_size,) + targets.shape)
        flat_t, lead = _flatten_leading(targets, 2)

        if not prev_b:
            prev = jnp.broadcast_to(prev, (axis_size,) + prev.shape)
        flat_p, _ = _flatten_leading(prev, 1)

        coll = _Coll(sph_home, sph_joint, pairs) if has_collision else None
        q, err, found, clr = _analytic_ik_cuda_raw(
            target, flat_t, q7_samples, flat_p if use_prev else None,
            coll, (w_sph, w_cap, w_box, w_hs),
            respect_limits=respect_limits, err_tol=err_tol, margin=margin)

        return ((q.reshape(lead + q.shape[-1:]), err.reshape(lead),
                 found.reshape(lead), clr.reshape(lead)),
                (True, True, True, True))

    return f


def analytic_ik_cuda(target, targets, q7_samples, previous_cfg=None, *,
                     collision=None, world_spheres=None,
                     respect_limits: bool = True, err_tol: float = 1e-4,
                     margin: float = 0.005):
    """Batched analytic IK on the GPU, with a true single-launch ``vmap`` rule.

    Args:
        collision: optional :class:`~pyroffi.kinematics._analytic_collision.CollisionData`.
            When given, colliding branches are ranked behind collision-free ones
            and a per-target clearance is returned.
        world_spheres: ``[W, 4]`` obstacle spheres, or ``None`` for
            self-collision only.

    Returns:
        ``(q[B,7], err[B], found[B], clearance[B])``. Clearance is ``+1e30``
        when collision checking is disabled.
    """
    targets = jnp.asarray(targets, dtype=jnp.float32)
    if targets.ndim == 2:
        targets = targets[None, ...]
    batch = targets.shape[0]

    use_prev = previous_cfg is not None
    prev = (jnp.asarray(previous_cfg, dtype=jnp.float32).reshape(batch, 7)
            if use_prev else jnp.zeros((batch, 7), dtype=jnp.float32))

    # Empty placeholders are cached, not rebuilt per call. Allocating four
    # fresh device arrays (plus the extra clearance output) on every invocation
    # cost a constant ~1.8 ms of host-side dispatch — independent of batch size,
    # which is what identified it as marshalling rather than kernel time. It
    # nearly doubled the no-collision solve at small batches.
    has_coll = collision is not None
    if has_coll:
        sph_home = jnp.asarray(collision.spheres_home, dtype=jnp.float32)
        sph_joint = jnp.asarray(collision.sphere_joint, dtype=jnp.int32)
        pairs = jnp.asarray(collision.self_pairs, dtype=jnp.int32).reshape(-1, 2)
    else:
        sph_home, sph_joint, pairs = _empty_collision_buffers()
    world = world_arrays(world_spheres)

    f = _make_batched(target, bool(respect_limits), float(err_tol), use_prev,
                      float(margin), has_coll)
    return f(targets, q7_samples, prev,
             sph_home, sph_joint, pairs, *world)
