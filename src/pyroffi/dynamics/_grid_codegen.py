"""Runtime code generation and compilation of GRiD dynamics kernels.

Pipeline (mirrors the cricket/VAMP JIT flow in ``collision/_vamp_collision.py``,
with nvcc in place of LLVM ORC):

  1. ``grid_codegen.GRiDCodeGenerator`` emits a robot-specific ``grid.cuh``.
  2. The static FFI translation unit ``cuda_kernels/dynamics/_grid_ffi_tu.cu``
     is compiled against it with ``nvcc --shared``.
  3. The resulting ``.so`` is cached on disk keyed by a sha1 over the
     generated header, the TU source, and the compile flags, so subsequent
     runs (and repeated constructions in one process) skip codegen entirely.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import subprocess
import tempfile
from pathlib import Path

from loguru import logger

from ._grid_robot_adapter import GridRobotModel
from ._vendor import ensure_grid_importable

_TU_PATH = Path(__file__).parent.parent / "cuda_kernels" / "dynamics" / "_grid_ffi_tu.cu"


def _cache_root() -> Path:
    env = os.environ.get("PYROFFI_GRID_CACHE")
    if env:
        return Path(env)
    xdg = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
    return Path(xdg) / "pyroffi" / "grid"


def _jaxlib_include_dir() -> Path:
    import jaxlib

    inc = Path(jaxlib.__file__).parent / "include"
    if not (inc / "xla" / "ffi" / "api" / "ffi.h").is_file():
        raise RuntimeError(
            f"xla/ffi/api/ffi.h not found under {inc}; jaxlib >= 0.4.14 required."
        )
    return inc


def generate_grid_cuh(
    grid_model: GridRobotModel, *, runtime_inertia: bool = False
) -> str:
    """Run GRiDCodeGenerator and return the generated header source.

    ``runtime_inertia`` emits the flag-gated mutable inertia table
    (``init_inertia_params`` / ``set_inertia_params`` plus the on-device
    ``s_XImats`` I-region rebuild), which is what lets an attached payload's
    inertia change without regenerating kernels.  With the flag off, upstream
    emits nothing for it and the baked header is byte-identical to before.
    """
    ensure_grid_importable()
    from grid_codegen import GRiDCodeGenerator

    codegen = GRiDCodeGenerator(
        grid_model.robot, False, False, FILE_NAMESPACE="grid"
    )
    # gen_all_code writes "<namespace>.cuh" into the *current directory*;
    # GRiD is used unmodified, so redirect cwd (and its prints) around it.
    with tempfile.TemporaryDirectory(prefix="pyroffi_grid_codegen_") as out_dir:
        prev_cwd = os.getcwd()
        try:
            os.chdir(out_dir)
            with contextlib.redirect_stdout(io.StringIO()):
                codegen.gen_all_code(runtime_inertia=runtime_inertia)
        finally:
            os.chdir(prev_cwd)
        return (Path(out_dir) / "grid.cuh").read_text()


def compile_grid_library(
    grid_model: GridRobotModel,
    arch: str | None = None,
    *,
    runtime_inertia: bool = False,
) -> Path:
    """Generate + compile the GRiD FFI library for this robot, with caching.

    Returns the path to the compiled shared library.
    """
    grid_cuh = generate_grid_cuh(grid_model, runtime_inertia=runtime_inertia)
    tu_source = _TU_PATH.read_text()
    arch_flag = arch or os.environ.get("PYROFFI_GRID_GPU_ARCH", "-arch=native")
    flags = [
        "-O3",
        "-std=c++17",
        arch_flag,
        "--shared",
        "--compiler-options",
        "-fPIC",
    ]
    if runtime_inertia:
        # Gates the set_inertia_params entry point in the TU; the generated
        # header only declares it under the same codegen flag.
        flags.append("-DPYROFFI_GRID_RUNTIME_INERTIA")
    key = hashlib.sha1(
        "\x00".join([grid_cuh, tu_source, " ".join(flags)]).encode()
    ).hexdigest()

    build_dir = _cache_root() / key
    so_path = build_dir / "grid_ffi.so"
    if so_path.is_file():
        return so_path

    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "grid.cuh").write_text(grid_cuh)
    tu_path = build_dir / "grid_ffi_tu.cu"
    tu_path.write_text(tu_source)

    cmd = [
        "nvcc",
        *flags,
        f"-I{_jaxlib_include_dir()}",
        f"-I{build_dir}",
        "-o",
        str(so_path),
        str(tu_path),
    ]
    logger.info(f"Compiling GRiD dynamics kernels (one-time, cached): {build_dir}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"nvcc failed to compile GRiD dynamics kernels:\n{result.stderr}"
        )
    return so_path
