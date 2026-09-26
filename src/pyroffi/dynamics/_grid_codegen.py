"""Runtime code generation and compilation of GRiD dynamics kernels.

Pipeline (mirrors the cricket/VAMP JIT flow in ``collision/_vamp_collision.py``,
with nvcc in place of LLVM ORC):

  1. ``grid_codegen.GRiDCodeGenerator`` emits a robot-specific ``grid.cuh``.
  2. The static FFI translation unit ``cuda_kernels/dynamics/_grid_ffi_tu.cu``
     is compiled against it by PyRoFFI's shared kernel builder
     (``cuda_kernels._traced.build_shared_library``), which caches the ``.so``
     on disk keyed by the generated header, the TU source and the flags, in the
     same cache as the traced kernels (``$PYROFFI_TRACED_CACHE``).
"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
from pathlib import Path

from ._grid_robot_adapter import GridRobotModel
from ._vendor import ensure_grid_importable

_TU_PATH = Path(__file__).parent.parent / "cuda_kernels" / "dynamics" / "_grid_ffi_tu.cu"


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
    from ..cuda_kernels._traced import build_shared_library

    arch_flag = arch or os.environ.get("PYROFFI_GRID_GPU_ARCH", "-arch=native")
    flags = ["-O3", "-std=c++17", arch_flag, "--shared", "--compiler-options", "-fPIC"]
    if runtime_inertia:
        # Gates the set_inertia_params entry point in the TU; the generated
        # header only declares it under the same codegen flag.
        flags.append("-DPYROFFI_GRID_RUNTIME_INERTIA")
    grid_cuh = generate_grid_cuh(grid_model, runtime_inertia=runtime_inertia)
    return build_shared_library("grid_ffi", _TU_PATH, {"grid.cuh": grid_cuh}, flags)
