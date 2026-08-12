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


def fused_self_collision(cfg, robot_buffers, static):
    """Run the fused kernel.

    Args:
        cfg: ``[B, n_act]`` joint configurations.
        robot_buffers: ``(twists, parent_tf, parent_idx, act_idx, mimic_mul,
            mimic_off, mimic_act_idx, topo_inv)`` — the same model arrays the
            other CUDA kernels take.
        static: ``(sph_local, link_start, link_joint, pair_i, pair_j)``
            from :func:`static_arrays`.

    Returns:
        ``[B, P]`` signed distances per active link pair.
    """
    _load_and_register()

    cfg = jnp.asarray(cfg, dtype=jnp.float32)
    if cfg.ndim == 1:
        cfg = cfg[None, :]
    B = cfg.shape[0]

    sph_local, link_start, link_joint, pair_i, pair_j = static
    P = pair_i.shape[0]

    call = jax.ffi.ffi_call(
        "fused_self_collision",
        jax.ShapeDtypeStruct((B, P), jnp.float32),
        vmap_method="sequential",
    )
    return call(
        cfg,
        *[jnp.asarray(x) for x in robot_buffers],
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


def fused_world_collision(cfg, robot_buffers, static, world):
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
    w_sph, w_cap, w_box, w_hs = [np.asarray(x, dtype=np.float32) for x in world]
    M = sum(x.shape[0] for x in (w_sph, w_cap, w_box, w_hs))

    call = jax.ffi.ffi_call(
        "fused_world_collision",
        jax.ShapeDtypeStruct((B, N, M), jnp.float32),
        vmap_method="sequential",
    )
    return call(
        cfg,
        *[jnp.asarray(x) for x in robot_buffers],
        jnp.asarray(sph_local, jnp.float32),
        jnp.asarray(link_start, jnp.int32),
        jnp.asarray(link_joint, jnp.int32),
        jnp.asarray(w_sph), jnp.asarray(w_cap),
        jnp.asarray(w_box), jnp.asarray(w_hs),
    )
