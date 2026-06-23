"""JAX FFI wrapper for the RoboGPU sphere-octree collision-check kernel.

The companion shared library ``_robogpu_collision_lib.so`` must be compiled
before importing this module:

    bash build_kernels/build_robogpu_collision.sh

The build script also compiles ``_robogpu_optix_programs.ptx`` (OptiX device
programs); both files must sit alongside the .so at runtime.

The XLA FFI target ``"robogpu_collision"`` accepts the same FK model arrays as
``CUDABinaryCollisionChecker`` plus a point cloud [Mp, 3] and two scalar
attributes (``r_env``, ``r_robot_max``).  It returns ``int32[B]`` with 1 =
collision-free and 0 = in-collision.
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

_LIB_NAME = "_robogpu_collision_lib.so"
_FFI_TARGET = "robogpu_collision"


@lru_cache(maxsize=1)
def _load_and_register() -> None:
    lib_path = Path(__file__).parent / _LIB_NAME
    if not lib_path.exists():
        raise RuntimeError(
            f"RoboGPU collision library not found at {lib_path}.\n"
            "Compile it first with:\n"
            "  bash build_kernels/build_robogpu_collision.sh\n"
            "(Requires NVIDIA OptiX SDK 7.x and nvcc.)"
        )
    ptx_path = lib_path.parent / "_robogpu_optix_programs.ptx"
    if not ptx_path.exists():
        raise RuntimeError(
            f"RoboGPU OptiX PTX not found at {ptx_path}.\n"
            "Run build_kernels/build_robogpu_collision.sh to produce it."
        )

    lib = ctypes.CDLL(str(lib_path))
    _New = ctypes.pythonapi.PyCapsule_New
    _New.restype  = ctypes.py_object
    _New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    capsule = _New(
        ctypes.cast(lib.RoboGPUCollisionFfi, ctypes.c_void_p),
        b"xla._CUSTOM_CALL_TARGET",
        None,
    )
    jax.ffi.register_ffi_target(_FFI_TARGET, capsule, platform="CUDA")


def robogpu_collision(
    cfg:               Array,  # [B, n_act]    float32
    twists:            Array,  # [J, 6]        float32
    parent_tf:         Array,  # [J, 7]        float32
    parent_idx:        Array,  # [J]           int32
    act_idx:           Array,  # [J]           int32
    mimic_mul:         Array,  # [J]           float32
    mimic_off:         Array,  # [J]           float32
    mimic_act_idx:     Array,  # [J]           int32
    topo_inv:          Array,  # [J]           int32
    link_parent_joint: Array,  # [NL]          int32
    f_local:           Array,  # [K, 4]        float32  (k = s*NL + n)
    f_pair_i:          Array,  # [Pf]          int32
    f_pair_j:          Array,  # [Pf]          int32
    world_spheres:     Array,  # [Ms, 4]       float32
    world_capsules:    Array,  # [Mc, 7]       float32
    world_boxes:       Array,  # [Mb, 15]      float32
    world_halfspaces:  Array,  # [Mh, 6]       float32
    point_cloud:       Array,  # [Mp, 3]       float32
    r_env:             float,  # env sphere radius per point
    r_robot_max:       float,  # max robot sphere radius (BVH AABB expansion)
    dynamic:           bool = False,  # refit a persistent BVH instead of rebuilding
) -> Array:                    # [B]           int32   1=free, 0=collision
    """Fused FK + binary collision check with OptiX point-cloud BVH traversal.

    Regular world geometry (spheres, capsules, boxes, halfspaces) and self-
    collision are checked in a CUDA kernel (Stage 1).  The environment point
    cloud is indexed in an OptiX BVH; robot sphere centres query it via ray
    tracing with any-hit early exit (Stage 2).

    Pass ``point_cloud`` with shape ``[0, 3]`` to skip the OptiX stage and
    run only the regular world geometry + self-collision check.

    Set ``dynamic=True`` for streaming point clouds (e.g. a depth camera) whose
    contents change every frame but whose size and radii stay fixed.  The OptiX
    BVH is then *refit* in place each call rather than rebuilt, reusing the tree
    topology — far cheaper and fully asynchronous.  The first call (and any call
    whose point count or radii differ from the previous one) still does a full
    build.
    """
    _load_and_register()
    B = cfg.shape[0]
    r_env_np     = np.float32(r_env)
    r_robot_np   = np.float32(r_robot_max)
    dynamic_np   = np.int32(1 if dynamic else 0)
    return jax.ffi.ffi_call(
        _FFI_TARGET,
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
        f_pair_i.astype(jnp.int32),
        f_pair_j.astype(jnp.int32),
        world_spheres.astype(jnp.float32),
        world_capsules.astype(jnp.float32),
        world_boxes.astype(jnp.float32),
        world_halfspaces.astype(jnp.float32),
        point_cloud.astype(jnp.float32),
        # Scalar attributes (consumed by .Attr<>() in the FFI handler).
        r_env=r_env_np,
        r_robot_max=r_robot_np,
        dynamic=dynamic_np,
    )
