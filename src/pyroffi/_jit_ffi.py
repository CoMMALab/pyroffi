"""Shared cricket-JIT + XLA-FFI plumbing for pyroffi's CPU-accelerated kernels.

Both the QuIK inverse-kinematics backend (:mod:`pyroffi.optimization_engines._quik_ik`)
and the VAMP forward-kinematics backend (:mod:`pyroffi.kinematics._vamp_fk`) compile
a small C++ translation unit at runtime with cricket's LLVM-ORC JIT and register
the resulting XLA FFI custom-call handlers with JAX for ``platform="cpu"``.  This
module collects the machinery those two share (originally private to
``collision._vamp_collision``): locating the JIT, the XLA FFI headers, Eigen,
preloading the C++/OpenMP runtimes into the global symbol scope, and pinning JIT
sessions for the process lifetime so registered targets never dangle.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def cricket_jit():
    """Import cricket's JIT submodule, with a helpful error if unavailable."""
    try:
        import cricket  # noqa: F401
        from cricket import _core_ext

        jit = getattr(_core_ext, "jit", None)
        if jit is None:
            raise RuntimeError(
                "cricket was built without JIT support "
                "(reconfigure with -DCRICKET_BUILD_JIT=ON)."
            )
        return cricket, jit
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "cricket is not importable. Build it from external/cricket with the "
            "Python extension and JIT enabled (CRICKET_BUILD_PYTHON=ON, "
            "CRICKET_BUILD_JIT=ON)."
        ) from exc


@lru_cache(maxsize=1)
def xla_ffi_include() -> str:
    """Locate the XLA FFI headers shipped with the installed jaxlib."""
    import jaxlib

    root = Path(jaxlib.__file__).resolve().parent / "include"
    if not (root / "xla" / "ffi" / "api" / "ffi.h").exists():
        raise RuntimeError(
            f"Could not find xla/ffi/api/ffi.h under {root}. "
            "Pass include_dirs=[...] pointing at the XLA FFI headers."
        )
    return str(root)


@lru_cache(maxsize=1)
def eigen_include_dirs() -> tuple[str, ...]:
    """Best-effort Eigen include roots (conda prefix, then system paths)."""
    conda = os.environ.get("CONDA_PREFIX")
    cands = [
        os.path.join(conda, "include", "eigen3") if conda else None,
        os.path.join(conda, "include") if conda else None,
        "/usr/include/eigen3",
        "/usr/local/include/eigen3",
    ]
    return tuple(c for c in cands if c and Path(c).exists())


@lru_cache(maxsize=1)
def preload_runtime_libs() -> None:
    """Load libstdc++ and the OpenMP runtime with RTLD_GLOBAL.

    The cricket JIT resolves external symbols via ``dlsym(RTLD_DEFAULT)``, so the
    C++ standard library and OpenMP runtime the JIT-compiled code calls into must
    be present in the global symbol scope.  Loaded from the active conda prefix
    when available, else by soname.
    """
    import ctypes

    candidates = ["libstdc++.so.6", "libomp.so"]
    conda = os.environ.get("CONDA_PREFIX")
    libdir = Path(conda) / "lib" if conda else None
    for name in candidates:
        loaded = False
        if libdir is not None and (libdir / name).exists():
            try:
                ctypes.CDLL(str(libdir / name), mode=ctypes.RTLD_GLOBAL)
                loaded = True
            except OSError:
                pass
        if not loaded:
            try:
                ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass  # surfaced later as a clear "Symbols not found" JIT error


# JIT sessions pinned for the process lifetime.  XLA FFI targets cannot be
# unregistered, so the JIT-owned code behind them must never be freed.
_SESSIONS: dict[str, object] = {}
_REGISTERED: set[str] = set()


def register_handlers(
    tu_source: str,
    opts,
    work_dir: Path,
    targets: dict[str, str],
) -> None:
    """JIT-compile ``tu_source`` (object-cached) and register FFI targets.

    Args:
        tu_source: The C++ translation unit source.
        opts:      A ``jit.CompileOptions`` (already populated).
        work_dir:  On-disk JIT cache / session directory.
        targets:   ``{ffi_target_name: extern_C_accessor_symbol}`` to register
                   for ``platform="cpu"``.

    Idempotent: keyed by the (source, opts) hash so repeated constructions for
    the same kernel reuse the pinned session and skip re-registration.
    """
    import jax

    _cricket, jit = cricket_jit()
    key = jit.hash_source(tu_source, opts)
    if key in _REGISTERED:
        return
    preload_runtime_libs()
    session = jit.JitSession(work_dir)
    if not session.try_load_cached(opts.module_id):
        session.add_source(tu_source, opts)
    for target_name, accessor in targets.items():
        jax.ffi.register_ffi_target(
            target_name,
            session.handler_capsule(accessor),
            platform="cpu",
        )
    _SESSIONS[key] = session
    _REGISTERED.add(key)


@lru_cache(maxsize=1)
def cpu_device():
    """The host CPU device (FFI targets are registered for ``platform="cpu"``)."""
    import jax

    return jax.devices("cpu")[0]
