"""JAX FFI wrapper for the fused FK + self-collision kernel.

Computes self-collision distances straight from joint configurations in one
launch. The existing path runs FK as a separate XLA op, materialises a padded
``[B, S, N, 3]`` sphere tensor to global memory, then reads it back in the
collision kernel; this one keeps link transforms in shared memory and forms
sphere positions in registers on demand.

Output matches ``RobotCollisionSpherized.compute_self_collision_distance``:
``[B, P]`` signed distances over the active link pairs, in the same pair order.

Build first::

    bash build_kernels/build_fused_self_collision_cuda.sh
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .._ffi_dtypes import as_robot_buffers

_LIB_NAME = "_fused_self_collision_lib.so"


@lru_cache(maxsize=1)
def _load_and_register() -> None:
    lib_path = Path(__file__).parent / _LIB_NAME
    if not lib_path.exists():
        raise RuntimeError(
            f"Fused self-collision library not found at {lib_path}.\n"
            "Build it with:  bash build_kernels/build_fused_self_collision_cuda.sh")
    lib = ctypes.CDLL(str(lib_path))

    _PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    _PyCapsule_New.restype = ctypes.py_object
    _PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    capsule = _PyCapsule_New(
        ctypes.cast(getattr(lib, "FusedSelfCollisionFfi"), ctypes.c_void_p),
        b"xla._CUSTOM_CALL_TARGET", None)
    jax.ffi.register_ffi_target("fused_self_collision", capsule, platform="CUDA")


@lru_cache(maxsize=1)
def _register_esdf() -> None:
    _load_and_register()
    lib = ctypes.CDLL(str(Path(__file__).parent / _LIB_NAME))
    capsule_new = ctypes.pythonapi.PyCapsule_New
    capsule_new.restype = ctypes.py_object
    capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    capsule = capsule_new(ctypes.cast(lib.FusedWorldEsdfFfi, ctypes.c_void_p),
                          b"xla._CUSTOM_CALL_TARGET", None)
    jax.ffi.register_ffi_target("fused_world_esdf", capsule, platform="CUDA")


def fused_world_esdf(cfg, robot_buffers, static, grid, origin, voxel_size,
                     ffi_target="fused_world_esdf"):
    """Fused FK + robot-vs-ESDF collision: ``[B, N]`` per-link minimum over spheres of the
    trilinearly sampled signed distance at the sphere centre minus its radius (the
    :func:`~pyroffi.collision._esdf.esdf_query_jax` convention, edge-clamped).

    ``grid`` is ``[nx, ny, nz]`` float32, ``origin`` the world position of voxel (0, 0, 0)'s centre.
    """
    if ffi_target == "fused_world_esdf":
        _register_esdf()
    cfg = jnp.asarray(cfg, dtype=jnp.float32)
    if cfg.ndim == 1:
        cfg = cfg[None, :]
    sph_local, link_start, link_joint, _pi, _pj = static
    N = link_start.shape[0] - 1
    call = jax.ffi.ffi_call(ffi_target, jax.ShapeDtypeStruct((cfg.shape[0], N), jnp.float32),
                            vmap_method="sequential")
    return call(
        cfg,
        *as_robot_buffers(robot_buffers),
        jnp.asarray(sph_local, jnp.float32),
        jnp.asarray(link_start, jnp.int32),
        jnp.asarray(link_joint, jnp.int32),
        jnp.asarray(grid, jnp.float32),
        jnp.asarray(origin, jnp.float32),
        voxel=np.float32(voxel_size),
    )


def static_arrays(robot, model):
    """Flatten a spherized collision model into the kernel's static buffers.

    Returns ``(sph_local[K,4], link_start[N+1], link_joint[N], pair_i[P],
    pair_j[P])``.

    Padding slots (negative-radius sentinel) are dropped here rather than
    skipped in the kernel, and spheres are grouped by link so each link's run is
    contiguous — that is what lets the kernel walk a CSR range instead of
    scanning all ``S`` slots per link.
    """
    from ...collision._cuda_collision import link_parent_joint_for

    n_link, n_sph = model.coll.get_batch_axes()
    local = np.asarray(model.coll.pose.translation()).reshape(n_link, n_sph, 3)
    radii = np.asarray(model.coll.radius).reshape(n_link, n_sph)

    sph, starts = [], [0]
    for li in range(n_link):
        for si in range(n_sph):
            r = float(radii[li, si])
            if r <= 0.0:
                continue
            sph.append([*local[li, si], r])
        starts.append(len(sph))

    link_joint = np.asarray(link_parent_joint_for(robot, model), dtype=np.int32)

    return (np.asarray(sph, dtype=np.float32),
            np.asarray(starts, dtype=np.int32),
            link_joint,
            np.asarray(model.active_idx_i, dtype=np.int32),
            np.asarray(model.active_idx_j, dtype=np.int32))


def fused_self_collision(cfg, robot_buffers, static, ffi_target="fused_self_collision"):
    """Run the fused kernel.

    Args:
        cfg: ``[B, n_act]`` joint configurations.
        robot_buffers: ``(twists, parent_tf, parent_idx, act_idx, mimic_mul,
            mimic_off, mimic_act_idx, topo_inv)`` — the same model arrays the
            other CUDA kernels take.
        static: ``(sph_local, link_start, link_joint, pair_i, pair_j)``
            from :func:`static_arrays`.

    Returns:
        ``(dist[B, P], min_z[B])`` — signed distances per active link pair, and
        the lowest point reached by any sphere. ``min_z`` rides along because
        the kernel has already placed every sphere; computing it caller-side
        means a second FK, which measured more expensive than this entire
        kernel. Compare it against a floor height for a clearance test.
    """
    _load_and_register()

    cfg = jnp.asarray(cfg, dtype=jnp.float32)
    if cfg.ndim == 1:
        cfg = cfg[None, :]
    B = cfg.shape[0]

    sph_local, link_start, link_joint, pair_i, pair_j = static
    P = pair_i.shape[0]

    call = jax.ffi.ffi_call(
        ffi_target,
        (jax.ShapeDtypeStruct((B, P), jnp.float32),
         jax.ShapeDtypeStruct((B,), jnp.float32)),
        vmap_method="sequential",
    )
    return call(
        cfg,
        *as_robot_buffers(robot_buffers),
        jnp.asarray(sph_local, jnp.float32),
        jnp.asarray(link_start, jnp.int32),
        jnp.asarray(link_joint, jnp.int32),
        jnp.asarray(pair_i, jnp.int32),
        jnp.asarray(pair_j, jnp.int32),
    )


@lru_cache(maxsize=1)
def _register_world() -> None:
    lib = ctypes.CDLL(str(Path(__file__).parent / _LIB_NAME))
    _PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    _PyCapsule_New.restype = ctypes.py_object
    _PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    capsule = _PyCapsule_New(
        ctypes.cast(getattr(lib, "FusedWorldCollisionFfi"), ctypes.c_void_p),
        b"xla._CUSTOM_CALL_TARGET", None)
    jax.ffi.register_ffi_target("fused_world_collision", capsule, platform="CUDA")


def fused_world_collision(cfg, robot_buffers, static, world, ffi_target="fused_world_collision"):
    """Fused FK + robot-vs-world collision.

    Args:
        world: ``(spheres[Ms,4], capsules[Mc,7], boxes[Mb,15], halfspaces[Mh,6])``
            in the same row layouts every other CUDA IK kernel uses, so a world
            built for ls_ik/hjcd_ik/sqp_ik works here unchanged. Empty arrays
            disable a type.

    Returns:
        ``[B, N, M]`` signed distances, ``M = Ms + Mc + Mb + Mh`` in that order.
        Per link the value is the minimum over that link's spheres.
    """
    _register_world()

    cfg = jnp.asarray(cfg, dtype=jnp.float32)
    if cfg.ndim == 1:
        cfg = cfg[None, :]
    B = cfg.shape[0]

    sph_local, link_start, link_joint, _pi, _pj = static
    N = link_start.shape[0] - 1
    # jnp, not np: under jit the world arrays may be traced (only their shapes are static).
    w_sph, w_cap, w_box, w_hs = [jnp.asarray(x, dtype=jnp.float32) for x in world]
    M = sum(x.shape[0] for x in (w_sph, w_cap, w_box, w_hs))

    call = jax.ffi.ffi_call(
        ffi_target,
        jax.ShapeDtypeStruct((B, N, M), jnp.float32),
        vmap_method="sequential",
    )
    return call(
        cfg,
        *as_robot_buffers(robot_buffers),
        jnp.asarray(sph_local, jnp.float32),
        jnp.asarray(link_start, jnp.int32),
        jnp.asarray(link_joint, jnp.int32),
        jnp.asarray(w_sph), jnp.asarray(w_cap),
        jnp.asarray(w_box), jnp.asarray(w_hs),
    )


# ---------------------------------------------------------------------------
# Jacobian variants: the same outputs plus d(output)/dq from the GPU.

_JAC_TARGETS = {"FusedSelfCollisionJacFfi": "fused_self_collision_jac",
                "FusedWorldCollisionJacFfi": "fused_world_collision_jac",
                "FusedWorldEsdfJacFfi": "fused_world_esdf_jac"}


@lru_cache(maxsize=1)
def _register_jac() -> None:
    _load_and_register()
    lib = ctypes.CDLL(str(Path(__file__).parent / _LIB_NAME))
    capsule_new = ctypes.pythonapi.PyCapsule_New
    capsule_new.restype = ctypes.py_object
    capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    for symbol, target in _JAC_TARGETS.items():
        capsule = capsule_new(ctypes.cast(getattr(lib, symbol), ctypes.c_void_p),
                              b"xla._CUSTOM_CALL_TARGET", None)
        jax.ffi.register_ffi_target(target, capsule, platform="CUDA")


def _jac_call(target, stock, outs, cfg, robot_buffers, static, extra, **attrs):
    if target == stock:
        _register_jac()
    sph_local, link_start, link_joint, _pi, _pj = static
    call = jax.ffi.ffi_call(target, outs, vmap_method="sequential")
    return call(cfg, *as_robot_buffers(robot_buffers), jnp.asarray(sph_local, jnp.float32),
                jnp.asarray(link_start, jnp.int32), jnp.asarray(link_joint, jnp.int32), *extra, **attrs)


def fused_self_collision_jac(cfg, robot_buffers, static, ffi_target="fused_self_collision_jac"):
    """``(dist[B, P], d dist/dq [B, P, n], min_z[B], d min_z/dq [B, n])``."""
    B, n = cfg.shape
    P = static[3].shape[0]
    f32 = lambda *s: jax.ShapeDtypeStruct(s, jnp.float32)
    return _jac_call(ffi_target, "fused_self_collision_jac", (f32(B, P), f32(B, P, n), f32(B), f32(B, n)),
                     cfg, robot_buffers, static,
                     (jnp.asarray(static[3], jnp.int32), jnp.asarray(static[4], jnp.int32)))


def fused_world_collision_jac(cfg, robot_buffers, static, world, ffi_target="fused_world_collision_jac"):
    """``(dist[B, N, M], d dist/dq [B, N, M, n])`` for the world layout of :func:`fused_world_collision`."""
    B, n = cfg.shape
    N = static[1].shape[0] - 1
    w = [jnp.asarray(x, dtype=jnp.float32) for x in world]
    M = sum(x.shape[0] for x in w)
    f32 = lambda *s: jax.ShapeDtypeStruct(s, jnp.float32)
    return _jac_call(ffi_target, "fused_world_collision_jac", (f32(B, N, M), f32(B, N, M, n)),
                     cfg, robot_buffers, static, w)


def fused_world_esdf_jac(cfg, robot_buffers, static, grid, origin, voxel_size,
                         ffi_target="fused_world_esdf_jac"):
    """``(dist[B, N], d dist/dq [B, N, n])`` for :func:`fused_world_esdf`."""
    B, n = cfg.shape
    N = static[1].shape[0] - 1
    f32 = lambda *s: jax.ShapeDtypeStruct(s, jnp.float32)
    return _jac_call(ffi_target, "fused_world_esdf_jac", (f32(B, N), f32(B, N, n)), cfg, robot_buffers,
                     static, (jnp.asarray(grid, jnp.float32), jnp.asarray(origin, jnp.float32)),
                     voxel=np.float32(voxel_size))
