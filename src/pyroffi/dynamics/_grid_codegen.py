"""Runtime code generation and compilation of GRiD dynamics kernels.

Pipeline:

  1. GRiD's ``grid_codegen.GRiDCodeGenerator`` (GRiM's ``external/GRiD``) emits a robot-specific ``grid.cuh`` from
     pyroffi's model of the URDF (:mod:`._grid_robot_adapter`).
  2. GRiM compiles its any-batch JAX FFI handlers around that header
     (:mod:`grim.motion.generated_dynamics`), cached on disk by content key.
"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
from pathlib import Path

from ._grid_robot_adapter import GridRobotModel
from .._grim import ensure_importable


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
    ensure_importable()
    from grid_codegen import GRiDCodeGenerator

    codegen = GRiDCodeGenerator(grid_model.robot, False, False)
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
    grid_model: GridRobotModel, *, runtime_inertia: bool = False
) -> tuple[dict[str, str], Path]:
    """Generate the header and build GRiM's handlers for this robot, with caching.

    Returns ``({op: FFI target name}, library path)``; the library path is what the runtime
    inertia table (:mod:`._grid_runtime_inertia`) loads its setter from.
    """
    ensure_importable()
    from grim.motion.generated_dynamics import build

    header = generate_grid_cuh(grid_model, runtime_inertia=runtime_inertia)
    return build(header, floating_base=grid_model.floating_base,
                 runtime_inertia=runtime_inertia)
