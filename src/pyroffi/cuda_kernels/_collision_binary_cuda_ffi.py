"""JAX FFI wrapper for the fused FK + *binary* collision-check CUDA kernel.

The companion shared library ``_collision_binary_cuda_lib.so`` must be compiled
from ``_collision_binary_cuda_kernel.cu`` first:

    bash src/pyroffi/cuda_kernels/build_collision_binary_cuda.sh

Requires JAX >= 0.4.14 (for jax.ffi).

Unlike the signed-distance kernels in ``_collision_cuda_ffi.py`` (which return an
``[B, N, M]`` distance matrix), this kernel returns one int32 per configuration:
``1`` == collision-free, ``0`` == in collision (world OR self).  Forward
kinematics is fused into the kernel and a coarse sphere model guards the fine
geometry with per-config early exit — mirroring pRRTC's SIMT collision checker.
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import Array

_LIB_NAME = "_collision_binary_cuda_lib.so"


@lru_cache(maxsize=1)
def _load_and_register() -> None:
    """Load the shared library and register the FFI target (runs once)."""
    lib_path = Path(__file__).parent / _LIB_NAME
    if not lib_path.exists():
        raise RuntimeError(
            f"CUDA binary-collision library not found at {lib_path}.\n"
            "Compile it first with:\n"
            "  bash src/pyroffi/cuda_kernels/build_collision_binary_cuda.sh\n"
            "(This produces _collision_binary_cuda_lib.so alongside the kernel source.)"
        )
    lib = ctypes.CDLL(str(lib_path))

    _PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    _PyCapsule_New.restype = ctypes.py_object
    _PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    capsule = _PyCapsule_New(
        ctypes.cast(lib.CollisionBinaryFfi, ctypes.c_void_p),
        b"xla._CUSTOM_CALL_TARGET",
        None,
    )
    jax.ffi.register_ffi_target("collision_binary", capsule, platform="CUDA")


def collision_binary(
    cfg:               Array,  # [B, n_act]   float32
    twists:            Array,  # [J, 6]       float32
    parent_tf:         Array,  # [J, 7]       float32
    parent_idx:        Array,  # [J]          int32
    act_idx:           Array,  # [J]          int32
    mimic_mul:         Array,  # [J]          float32
    mimic_off:         Array,  # [J]          float32
    mimic_act_idx:     Array,  # [J]          int32
    topo_inv:          Array,  # [J]          int32
    link_parent_joint: Array,  # [NL]         int32
    f_local:           Array,  # [Kf, 4]      float32  (k = s*NL + n)
    c_local:           Array,  # [Kc, 4]      float32  (k = s*NL + n)
    world_spheres:     Array,  # [Ms, 4]      float32
    world_capsules:    Array,  # [Mc, 7]      float32
    world_boxes:       Array,  # [Mb, 15]     float32
    world_halfspaces:  Array,  # [Mh, 6]      float32
    f_pair_i:          Array,  # [Pf]         int32
    f_pair_j:          Array,  # [Pf]         int32
    c_pair_i:          Array,  # [Pc]         int32
    c_pair_j:          Array,  # [Pc]         int32
) -> Array:                    # [B]          int32  (1 = free, 0 = collision)
    """Fused FK + binary collision check, one CUDA block per configuration.

    Returns an int32 array of shape ``[B]``: ``1`` means the configuration is
    collision-free (world and self), ``0`` means it is in collision.

    The coarse model (``c_local`` / ``c_pair_*``) must *enclose* the fine model
    for the two-stage guard to be sound (a coarse "clear" implies a fine
    "clear").  Pass an empty ``c_local`` ([0, 4]) and empty coarse pairs to run
    the fine geometry directly with no coarse culling.
    """
    _load_and_register()

    B = cfg.shape[0]

    return jax.ffi.ffi_call(
        "collision_binary",
        jax.ShapeDtypeStruct((B,), jnp.int32),
    )(
        cfg.astype(jnp.float32),
        twists.astype(jnp.float32),
        parent_tf.astype(jnp.float32),
        parent_idx.astype(jnp.int32),
        act_idx.astype(jnp.int32),
        mimic_mul.astype(jnp.float32),
        mimic_off.astype(jnp.float32),
        mimic_act_idx.astype(jnp.int32),
        topo_inv.astype(jnp.int32),
        link_parent_joint.astype(jnp.int32),
        f_local.astype(jnp.float32),
        c_local.astype(jnp.float32),
        world_spheres.astype(jnp.float32),
        world_capsules.astype(jnp.float32),
        world_boxes.astype(jnp.float32),
        world_halfspaces.astype(jnp.float32),
        f_pair_i.astype(jnp.int32),
        f_pair_j.astype(jnp.int32),
        c_pair_i.astype(jnp.int32),
        c_pair_j.astype(jnp.int32),
    )
