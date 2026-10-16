"""Content-keyed nvcc builds for pyroffi's own CUDA libraries.

pyroffi's motion and dynamics kernels live in GRiM and build through it. What stays here builds
the one library pyroffi still owns a robot-specific variant of: roboGPU's baked-table build.

Builds are cached on disk (``$PYROFFI_TRACED_CACHE``, default ``~/.cache/pyroffi/traced``)
keyed by the generated headers, sources and flags, so each one compiles once.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import jax
import numpy as np
from loguru import logger

_KERNELS_DIR = Path(__file__).parent
_REPO_ROOT = _KERNELS_DIR.parents[2]


def optix_sdk() -> Path:
    env = os.environ.get("OPTIX_SDK")
    for root in ([Path(env)] if env else []) + sorted(_REPO_ROOT.glob("NVIDIA-OptiX-SDK*")):
        if (root / "include" / "optix.h").is_file():
            return root
    raise RuntimeError("OptiX SDK not found; set OPTIX_SDK (see build_robogpu_collision.sh).")


def _cache_root() -> Path:
    env = os.environ.get("PYROFFI_TRACED_CACHE")
    if env:
        return Path(env)
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "pyroffi" / "traced"


def constant_table(ctype: str, name: str, values) -> str:
    """A ``__constant__`` array definition holding ``values`` (at least one element)."""
    values = np.asarray(values).reshape(-1)
    fmt = (lambda v: f"{float(v):.9e}f") if ctype == "float" else (lambda v: str(int(v)))
    body = ", ".join(fmt(v) for v in values) or "0"
    return f"__device__ __constant__ {ctype} {name}[{max(values.size, 1)}] = {{{body}}};\n"


def build_shared_library(name: str, source: Path, generated: dict[str, str], flags: list[str],
                         include_dirs: tuple[Path, ...] = (), key_files: tuple[Path, ...] = (),
                         copy_files: tuple[Path, ...] = (), so_name: str | None = None) -> Path:
    """nvcc ``source`` against the ``generated`` headers into a disk-cached shared library.

    The cache key covers the generated headers, the flags and every file in ``key_files``
    (``source`` is added when absent), so any input that changes the binary changes the key.
    The library is ``so_name`` (default ``<name>.so``) inside the key's directory.
    Builds land in a temporary directory that is renamed into place, so concurrent builds of
    the same library cannot leave a half-written one behind.
    """
    import jaxlib

    so_name = so_name or f"{name}.so"
    if source not in key_files:
        key_files = (source, *key_files)
    key = hashlib.sha1("\x00".join(
        [name, *generated.values(), " ".join(flags),
         *(p.read_text() for p in key_files)]).encode()).hexdigest()
    build_dir = _cache_root() / key
    if (build_dir / so_name).is_file():
        return build_dir / so_name

    _cache_root().mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=_cache_root()))
    try:
        for name, text in generated.items():
            (tmp / name).write_text(text)
        for f in copy_files:
            (tmp / f.name).write_bytes(f.read_bytes())
        cmd = ["nvcc", *flags, f"-I{tmp}", *(f"-I{d}" for d in include_dirs),
               f"-I{Path(jaxlib.__file__).parent / 'include'}", "-o", str(tmp / so_name), str(source)]
        logger.info(f"Compiling {so_name} (one-time, cached): {build_dir}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"nvcc failed to compile {so_name}:\n{result.stderr}")
        try:
            tmp.rename(build_dir)
        except OSError:  # another process finished the same build first
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return build_dir / so_name


def register_cuda_targets(so: Path, symbols: tuple[str, ...], tag: str) -> tuple[str, ...]:
    """Load ``so`` and register each FFI handler symbol as CUDA target ``<symbol>_<tag>``."""
    lib = ctypes.CDLL(str(so))
    capsule_new = ctypes.pythonapi.PyCapsule_New
    capsule_new.restype = ctypes.py_object
    capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    names = []
    for symbol in symbols:
        capsule = capsule_new(
            ctypes.cast(getattr(lib, symbol), ctypes.c_void_p), b"xla._CUSTOM_CALL_TARGET", None)
        names.append(f"{symbol}_{tag}")
        jax.ffi.register_ffi_target(names[-1], capsule, platform="CUDA")
    return tuple(names)
