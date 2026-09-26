"""JAX FFI wrapper for the PBA+ CUDA kernel (faithful-V2 ESDF port, Piece 2).

The companion shared library ``_pba_cuda_lib.so`` must be compiled from
``_pba_cuda_kernel.cu`` first:

    bash build_kernels/build_pba_cuda.sh

This is layer 1 only: the raw packed-int32 site-index grid in/out custom
call, a direct mirror of cuRobo's ``launch_pba3d`` (pba.py). Packing seed
coordinates into that format and unpacking the propagated result into
(site_xyz, has_site) -- matching ``jfa_propagate_jax``'s contract -- lives in
``pyroffi.collision._esdf.pba_propagate_jax`` (layer 2), which is the
``propagate_fn`` most callers should actually use (via ``edt_solver="pba"``
in ``pyroffi.collision._esdf.build_esdf_pipeline``).
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

_LIB_NAME = "_pba_cuda_lib.so"


@lru_cache(maxsize=1)
def _load_and_register() -> None:
    """Load the shared library and register the FFI target (runs once)."""
    lib_path = Path(__file__).parent / _LIB_NAME
    if not lib_path.exists():
        raise RuntimeError(
            f"PBA CUDA library not found at {lib_path}.\n"
            "Compile it first with:\n"
            "  bash build_kernels/build_pba_cuda.sh"
        )
    lib = ctypes.CDLL(str(lib_path))

    _PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    _PyCapsule_New.restype = ctypes.py_object
    _PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

    capsule = _PyCapsule_New(
        ctypes.cast(lib.PbaPropagateSitesFfi, ctypes.c_void_p),
        b"xla._CUSTOM_CALL_TARGET",
        None,
    )
    jax.ffi.register_ffi_target("pba_propagate_sites", capsule, platform="CUDA")


def pba_propagate_sites_raw(
    site_index: Array,  # int32 [nx, ny, nz], packed (site_encoding.cuh)
    m3: int = 2,
) -> Array:              # int32 [nx, ny, nz], packed, propagated
    """Run the 5-launch PBA+ 3D EDT on a packed site-index grid.

    Forward-only CUDA custom call -- deliberately no jax.custom_jvp bridge
    anywhere in this module or its callers; see
    ``pyroffi.collision._esdf``'s module docstring for why grid construction
    doesn't need one (unlike CUDADifferentiableSDFCollisionChecker's
    kernels, which do carry a real float tangent across the FFI boundary).

    Args:
        site_index: int32 [nx, ny, nz]. Sites are non-negative packed
            values (site_encoding.cuh::encode_site with empty_flag=0);
            non-sites are negative (empty_flag=1).
        m3: color-axis block Y dim, tuning parameter (cuRobo default 2).

    Returns:
        int32 [nx, ny, nz], same packing, propagated so every voxel now
        encodes the coordinates of its nearest site.
    """
    _load_and_register()
    nx, ny, nz = site_index.shape
    if max(nx, ny, nz) > 1024:
        raise ValueError(
            f"PBA packs coordinates into 10 bits each -- grid shape "
            f"{(nx, ny, nz)} exceeds the 1024-per-axis limit."
        )

    site_index = site_index.astype(jnp.int32)
    _, out = jax.ffi.ffi_call(
        "pba_propagate_sites",
        (
            jax.ShapeDtypeStruct((nx, ny, nz), jnp.int32),  # scratch, discarded
            jax.ShapeDtypeStruct((nx, ny, nz), jnp.int32),  # out
        ),
    )(
        site_index,
        nx=np.int64(nx),
        ny=np.int64(ny),
        nz=np.int64(nz),
        m3=np.int64(m3),
    )
    return out
