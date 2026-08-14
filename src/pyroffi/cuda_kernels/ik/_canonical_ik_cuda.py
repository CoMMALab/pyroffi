"""FFI wrapper for the canonicalisation kernel.

Runs the bulk of the Gauss-Newton walk on the GPU. The JAX loop it replaces
cost ~4 ms per iteration in XLA dispatch alone -- 61x to 149x the IK solve it
was correcting -- while the arithmetic per iteration is one FK, one task
Jacobian and a 6x6 SPD solve.

float32, like the FK/Jacobian helper it calls, so it converges to roughly 1e-5.
Callers that need the manifold hit exactly finish with a few float64 steps in
JAX; see ``_canonical_ik.canonicalize``.
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .._ffi_dtypes import as_robot_buffers

_LIB_NAME = "_canonical_ik_lib.so"


@lru_cache(maxsize=1)
def _load_and_register() -> None:
    lib_path = Path(__file__).parent / _LIB_NAME
    if not lib_path.exists():
        raise RuntimeError(
            f"Canonical-IK CUDA library not found at {lib_path}.\n"
            "Compile it first with:  bash build_kernels/build_canonical_ik_cuda.sh\n"
        )
    lib = ctypes.CDLL(str(lib_path))

    _PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    _PyCapsule_New.restype = ctypes.py_object
    _PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

    capsule = _PyCapsule_New(
        ctypes.cast(getattr(lib, "CanonicalIkFfi"), ctypes.c_void_p),
        b"xla._CUSTOM_CALL_TARGET",
        None,
    )
    jax.ffi.register_ffi_target("canonical_ik_cuda", capsule, platform="CUDA")


def library_available() -> bool:
    return (Path(__file__).parent / _LIB_NAME).exists()


def canonicalize_cuda(
    cfgs,
    cfg_refs,
    robot_buffers,
    target_jnts,
    ancestor_masks,
    target_Ts,
    max_iters: int = 400,
    step: float = 0.05,
    tol: float = 1e-5,
    damping: float = 1e-9,
):
    """``(q_canon, iters_used)``, both with a leading problem axis.

    ``tol`` is on the STEP norm and defaults to 1e-5, not something tighter:
    the kernel is float32, so a smaller threshold is unreachable and the loop
    then runs past its noise floor and WANDERS -- at 1500 iterations the answer
    drifted 1.94 rad away from the converged one it had already found by 400.

    ``iters_used`` reports where each problem stopped, so a caller can tell a
    converged batch from one that ran out of iterations -- the failure the
    fixed-count JAX loop hid (its residual silently degraded to 1.8e-2 at
    B=1024).
    """
    _load_and_register()

    cfgs = jnp.asarray(cfgs, jnp.float32)
    n_problems, n_act = cfgs.shape
    n_ee = int(np.shape(target_jnts)[0])

    ops = (
        cfgs,
        jnp.asarray(cfg_refs, jnp.float32),
        *as_robot_buffers(robot_buffers),
        jnp.asarray(target_jnts, jnp.int32),
        jnp.asarray(ancestor_masks, jnp.int32),
        jnp.asarray(target_Ts, jnp.float32).reshape(n_problems, n_ee, 7),
    )

    return jax.ffi.ffi_call(
        "canonical_ik_cuda",
        (
            jax.ShapeDtypeStruct((n_problems, n_act), jnp.float32),
            jax.ShapeDtypeStruct((n_problems,), jnp.int32),
        ),
    )(
        *ops,
        max_iters=int(max_iters),
        step=np.float32(step),
        tol=np.float32(tol),
        damping=np.float32(damping),
    )
