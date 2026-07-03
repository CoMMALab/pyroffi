"""JAX FFI wrapper for the fused FK + *binary* collision-check CUDA kernel.

The companion shared library ``_collision_binary_cuda_lib.so`` must be compiled
from ``_collision_binary_cuda_kernel.cu`` first:

    bash build_kernels/build_collision_binary_cuda.sh

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
            "  bash build_kernels/build_collision_binary_cuda.sh\n"
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


# The robot/world operands (everything after ``cfg``) are config-independent: a
# vmap/pmap over configurations leaves them untouched and only adds a batch axis
# to ``cfg`` (and the output).  We keep them as positional args after ``cfg`` so
# the custom batching rule can pass them straight through unchanged.
_N_MODEL_ARGS = 19


@jax.custom_batching.custom_vmap
def _collision_binary_call(
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
    f_local:           Array,  # [Kf, 4]      float32
    c_local:           Array,  # [Kc, 4]      float32
    world_spheres:     Array,  # [Ms, 4]      float32
    world_capsules:    Array,  # [Mc, 7]      float32
    world_boxes:       Array,  # [Mb, 15]     float32
    world_halfspaces:  Array,  # [Mh, 6]      float32
    f_pair_i:          Array,  # [Pf]         int32
    f_pair_j:          Array,  # [Pf]         int32
    c_pair_i:          Array,  # [Pc]         int32
    c_pair_j:          Array,  # [Pc]         int32
) -> Array:                    # [B]          int32  (1 = free, 0 = collision)
    """Single fused FK + binary collision kernel launch over a flat [B, n_act] batch.

    The kernel grids one CUDA block per configuration, so the whole [B] batch
    runs in parallel in a single launch.  Batching (vmap/pmap) is handled by the
    custom rule below, which folds any mapped axis into ``B`` instead of looping.
    """
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


@_collision_binary_call.def_vmap
def _collision_binary_vmap(axis_size, in_batched, cfg, *model_args):
    """Batching rule: fold the mapped axis of ``cfg`` into the kernel's [B] grid.

    The planner evaluates this cost inside ``pmap(vmap(gtmp_plan))``.  Rather than
    serialise the kernel over the vmapped axis (``vmap_method="sequential"``,
    one launch per slice), we reshape ``[V, B, n_act] -> [V*B, n_act]`` and issue
    a *single* launch with ``V*B`` blocks — full GPU occupancy — then reshape the
    ``[V*B]`` verdict back to ``[V, B]``.

    Only ``cfg`` carries a batch axis; the robot/world operands are
    config-independent and are passed through unbatched.  (If a caller ever
    vmaps over the model args too, that is unsupported here and raises.)
    """
    cfg_batched = in_batched[0]
    model_batched = in_batched[1:]
    if any(model_batched):
        raise NotImplementedError(
            "collision_binary batching only supports a mapped configuration "
            "axis; the robot/world model arguments must be loop-invariant."
        )
    if not cfg_batched:
        # cfg unbatched while something forced a rule call — broadcast manually.
        out = _collision_binary_call(cfg, *model_args)
        return out, False

    # cfg arrives with its mapped axis at the front: [V, B, n_act].
    V = cfg.shape[0]
    B = cfg.shape[1]
    cfg_flat = cfg.reshape(V * B, cfg.shape[-1])
    out_flat = _collision_binary_call(cfg_flat, *model_args)  # [V*B]
    return out_flat.reshape(V, B), True


def _local_gpu_devices() -> list:
    """Local GPU/CUDA devices, falling back to whatever local devices exist."""
    gpus = [d for d in jax.local_devices() if d.platform in ("gpu", "cuda")]
    return gpus if gpus else jax.local_devices()


def eager_pmap_batch(fn, cfg: Array, *static, devices=None) -> Array:
    """Run ``fn(cfg_chunk, *static)`` over a flat ``[B, ...]`` batch on all GPUs.

    ``fn`` maps a configuration chunk ``[b, n_act]`` to a per-config result
    ``[b]`` (e.g. a jitted binary-collision call); ``*static`` are broadcast,
    config-independent operands passed unchanged to every device.

    The ``B`` configs are split evenly across all local GPUs with ``jax.pmap``
    (one shard per device), padding ``B`` up to a multiple of the device count
    with dummy rows and trimming them from the result (pad-then-trim, so the
    per-device shape stays static).  Falls back to a single direct call when
    there is only one device, the batch is smaller than the device count, or the
    inputs are JAX tracers (i.e. we are *already* inside jit/vmap/pmap, where a
    nested pmap is illegal and the surrounding transform handles device
    placement itself).
    """
    devs = devices if devices is not None else _local_gpu_devices()
    D = len(devs)
    B = cfg.shape[0]
    tracing = isinstance(cfg, jax.core.Tracer) or any(
        isinstance(s, jax.core.Tracer) for s in static
    )
    if tracing or D <= 1 or B < D:
        return fn(cfg, *static)

    pad = (-B) % D
    if pad:
        cfg = jnp.concatenate(
            [cfg, jnp.zeros((pad,) + cfg.shape[1:], cfg.dtype)], axis=0
        )
    per = cfg.shape[0] // D
    cfg_sharded = cfg.reshape((D, per) + cfg.shape[1:])
    pfn = jax.pmap(fn, in_axes=(0,) + (None,) * len(static), devices=devs)
    out = pfn(cfg_sharded, *static)  # [D, per, ...]
    return out.reshape((D * per,) + out.shape[2:])[:B]


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
    collision-free (world and self), ``0`` means it is in collision.  Extra
    leading (vmap/pmap) batch axes on ``cfg`` are supported and run in a single
    kernel launch — see :func:`_collision_binary_vmap`.

    The coarse model (``c_local`` / ``c_pair_*``) must *enclose* the fine model
    for the two-stage guard to be sound (a coarse "clear" implies a fine
    "clear").  Pass an empty ``c_local`` ([0, 4]) and empty coarse pairs to run
    the fine geometry directly with no coarse culling.
    """
    _load_and_register()

    return _collision_binary_call(
        cfg,
        twists,
        parent_tf,
        parent_idx,
        act_idx,
        mimic_mul,
        mimic_off,
        mimic_act_idx,
        topo_inv,
        link_parent_joint,
        f_local,
        c_local,
        world_spheres,
        world_capsules,
        world_boxes,
        world_halfspaces,
        f_pair_i,
        f_pair_j,
        c_pair_i,
        c_pair_j,
    )
