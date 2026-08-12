"""JAX FFI wrapper for the CUDA analytic-IK kernel.

The companion shared library ``_analytic_ik_cuda_lib.so`` must be compiled from
``_analytic_ik_cuda_kernel.cu`` before this module can be imported:

    bash build_kernels/build_analytic_ik_cuda.sh

Unlike the LM-based CUDA IK solvers, this one takes no robot-model buffers.
The arm geometry is resolved once on the host by
:func:`pyroffi.kinematics._analytic_ik.build_geometry` and packed into a single
flat float64 blob, so the kernel does no model traversal at all.

Requires JAX >= 0.4.14 (for jax.ffi).
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_LIB_NAME = "_analytic_ik_cuda_lib.so"

#: Scalars in the packed geometry blob. Must equal GEOM_N_SCALARS in the kernel;
#: the kernel re-checks and errors out rather than reading past the end.
GEOM_N_SCALARS = 95


@lru_cache(maxsize=1)
def _load_and_register() -> None:
    """Load the shared library and register the FFI target (runs once)."""
    lib_path = Path(__file__).parent / _LIB_NAME
    if not lib_path.exists():
        raise RuntimeError(
            f"Analytic-IK CUDA library not found at {lib_path}.\n"
            "Compile it first with:  bash build_kernels/build_analytic_ik_cuda.sh\n"
        )
    lib = ctypes.CDLL(str(lib_path))

    _PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    _PyCapsule_New.restype = ctypes.py_object
    _PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

    capsule = _PyCapsule_New(
        ctypes.cast(getattr(lib, "AnalyticIkCudaFfi"), ctypes.c_void_p),
        b"xla._CUSTOM_CALL_TARGET",
        None,
    )
    jax.ffi.register_ffi_target("analytic_ik_cuda", capsule, platform="CUDA")


def pack_geometry(geom) -> np.ndarray:
    """Flatten an :class:`~pyroffi.kinematics._analytic_ik.ArmGeometry` blob.

    Field order must match ``struct ArmGeom`` in the kernel exactly. It is
    asserted rather than trusted: a silent layout drift would not crash, it
    would return plausible-looking wrong joint angles.
    """
    m_home = np.asarray(geom.m_home, dtype=np.float64)
    parts = [
        np.asarray(geom.axes, dtype=np.float64).reshape(-1),      # 21
        np.asarray(geom.points, dtype=np.float64).reshape(-1),    # 21
        np.asarray(geom.shoulder, dtype=np.float64).reshape(-1),  # 3
        np.asarray(geom.wrist, dtype=np.float64).reshape(-1),     # 3
        m_home.reshape(-1),                                       # 16
        np.linalg.inv(m_home).reshape(-1),                        # 16
        np.atleast_1d(np.asarray(geom.cos_alpha, dtype=np.float64)),  # 1
        np.asarray(geom.lower, dtype=np.float64).reshape(-1),     # 7
        np.asarray(geom.upper, dtype=np.float64).reshape(-1),     # 7
    ]
    blob = np.concatenate(parts)
    if blob.size != GEOM_N_SCALARS:
        raise AssertionError(
            f"packed geometry has {blob.size} scalars, kernel expects "
            f"{GEOM_N_SCALARS}; the struct layout and packer have drifted apart")
    return blob


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
    return (jnp.zeros((0, 4), jnp.float64),
            jnp.zeros((0,), jnp.int32),
            jnp.zeros((0, 2), jnp.int32))


@lru_cache(maxsize=1)
def _empty_world():
    """Cached zero-length buffers for all four world primitive types."""
    return (jnp.zeros((0, 4), jnp.float32), jnp.zeros((0, 7), jnp.float32),
            jnp.zeros((0, 15), jnp.float32), jnp.zeros((0, 6), jnp.float32))


_EMPTY_SPH = np.zeros((0, 4), dtype=np.float64)
_EMPTY_IDX = np.zeros((0,), dtype=np.int32)
_EMPTY_PAIR = np.zeros((0, 2), dtype=np.int32)


def _analytic_ik_cuda_raw(
    geom_blob,
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
        geom_blob: ``[95]`` float64 blob from :func:`pack_geometry`.
        targets: ``[B, 4, 4]`` target poses.
        q7_samples: ``[S]`` redundancy-parameter values to try.
        previous_cfg: optional ``[B, 7]``; when given the returned branch is the
            valid one closest to it in joint space (continuity resolution)
            rather than the lowest-error one.
        respect_limits: discard branches outside the joint limits.
        err_tol: pose-error threshold below which a branch counts as a solution.

    Returns:
        ``(q[B, 7], err[B], found[B])`` with ``found`` a boolean array. When
        ``found`` is False, ``q`` still holds the lowest-error near-miss.
    """
    _load_and_register()

    targets = jnp.asarray(targets, dtype=jnp.float64)
    if targets.ndim == 2:
        targets = targets[None, ...]
    batch = targets.shape[0]

    q7_samples = jnp.asarray(q7_samples, dtype=jnp.float32).reshape(-1)
    geom_blob = jnp.asarray(geom_blob, dtype=jnp.float64).reshape(-1)

    use_prev = previous_cfg is not None
    prev = (jnp.asarray(previous_cfg, dtype=jnp.float32).reshape(batch, 7)
            if use_prev else jnp.zeros((batch, 7), dtype=jnp.float32))

    # Collision buffers; empty arrays disable the collision path entirely, and
    # the kernel then allocates no sphere scratch and keeps its full block size.
    if collision is None:
        sph_home, sph_joint, pairs = _EMPTY_SPH, _EMPTY_IDX, _EMPTY_PAIR
    else:
        sph_home = collision.spheres_home
        sph_joint = collision.sphere_joint
        pairs = collision.self_pairs
    world = _empty_world() if world_spheres is None else world_spheres
    w_sph, w_cap, w_box, w_hs = world

    call = jax.ffi.ffi_call(
        "analytic_ik_cuda",
        (
            jax.ShapeDtypeStruct((batch, 7), jnp.float32),
            jax.ShapeDtypeStruct((batch,), jnp.float32),
            jax.ShapeDtypeStruct((batch,), jnp.int32),
            jax.ShapeDtypeStruct((batch,), jnp.float32),
        ),
        vmap_method="sequential",
    )
    q, err, found, clearance = call(
        geom_blob, targets, q7_samples, prev,
        jnp.asarray(sph_home, dtype=jnp.float64),
        jnp.asarray(sph_joint, dtype=jnp.int32),
        jnp.asarray(pairs, dtype=jnp.int32).reshape(-1, 2),
        w_sph, w_cap, w_box, w_hs,
        respect_limits=np.int64(bool(respect_limits)),
        use_prev=np.int64(bool(use_prev)),
        err_tol=np.float32(err_tol),
        margin=np.float32(margin),
    )
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
def _make_batched(respect_limits: bool, err_tol: float, use_prev: bool,
                  margin: float, has_collision: bool):
    @jax.custom_batching.custom_vmap
    def f(geom_blob, targets, q7_samples, prev,
          sph_home, sph_joint, pairs, w_sph, w_cap, w_box, w_hs):
        coll = _Coll(sph_home, sph_joint, pairs) if has_collision else None
        return _analytic_ik_cuda_raw(
            geom_blob, targets, q7_samples, prev if use_prev else None,
            coll, (w_sph, w_cap, w_box, w_hs),
            respect_limits=respect_limits, err_tol=err_tol, margin=margin)

    @f.def_vmap
    def _rule(axis_size, in_batched, geom_blob, targets, q7_samples, prev,
              sph_home, sph_joint, pairs, w_sph, w_cap, w_box, w_hs):
        geom_b, tgt_b, q7_b, prev_b = in_batched[:4]
        coll_b = any(in_batched[4:])

        # Geometry, the q7 sweep and the collision scene are shared across a
        # vmap in every sane use; a per-example one would need its own launch,
        # so refuse loudly rather than silently applying example 0's to all.
        if geom_b or q7_b or coll_b:
            raise NotImplementedError(
                "analytic_ik_cuda: vmap over the geometry blob, q7 samples or "
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
            geom_blob, flat_t, q7_samples, flat_p if use_prev else None,
            coll, (w_sph, w_cap, w_box, w_hs),
            respect_limits=respect_limits, err_tol=err_tol, margin=margin)

        return ((q.reshape(lead + q.shape[-1:]), err.reshape(lead),
                 found.reshape(lead), clr.reshape(lead)),
                (True, True, True, True))

    return f


def analytic_ik_cuda(geom_blob, targets, q7_samples, previous_cfg=None, *,
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
    targets = jnp.asarray(targets, dtype=jnp.float64)
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
        sph_home = jnp.asarray(collision.spheres_home, dtype=jnp.float64)
        sph_joint = jnp.asarray(collision.sphere_joint, dtype=jnp.int32)
        pairs = jnp.asarray(collision.self_pairs, dtype=jnp.int32).reshape(-1, 2)
    else:
        sph_home, sph_joint, pairs = _empty_collision_buffers()
    if world_spheres is None:
        world = _empty_world()
    else:
        w = world_spheres
        world = (jnp.asarray(w.spheres, jnp.float32),
                 jnp.asarray(w.capsules, jnp.float32),
                 jnp.asarray(w.boxes, jnp.float32),
                 jnp.asarray(w.halfspaces, jnp.float32))

    f = _make_batched(bool(respect_limits), float(err_tol), use_prev,
                      float(margin), has_coll)
    return f(geom_blob, targets, q7_samples, prev,
             sph_home, sph_joint, pairs, *world)
