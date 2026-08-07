"""Read and enforce the compile-time capacity limits a CUDA kernel .so was built with.

The kernels size fixed per-thread arrays from ``MAX_JOINTS`` / ``MAX_ACT``
(``float delta[MAX_ACT]``, ``double A_s[MAX_ACT*MAX_ACT]``,
``float T_world[MAX_JOINTS*7]``, …) and do **no** runtime bounds checking. A robot
whose actual DOF exceeds the compiled limit therefore writes past the end of thread-local
storage — undefined behaviour, which in practice means silently wrong IK solutions rather
than a crash. There is no way to detect that after the fact.

So each ``.so`` exports the values it was actually compiled with (see
``_build_params.cuh``) and this module reads them back via ``ctypes`` and refuses the
launch. Reading the real values matters: the Python side previously *assumed* a
``MAX_JOINTS=64, MAX_ACT=16`` build (hardcoded in ``_region_ik.py``), which any
``--max-act`` rebuild would have silently invalidated.

Usage from an FFI wrapper::

    lib = ctypes.CDLL(str(lib_path))
    params = read_build_params(lib, _LIB_NAME)     # once, at load
    ...
    params.check(n_joints=twists.shape[0], n_act=seeds.shape[-1], kernel="ls_ik_cuda")
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

# Hard ceiling on MAX_ACT — no build supports more. Mirrors _build_params.cuh's
# static_assert and _build_params.sh's PYROFFI_MAX_ACT_CEILING; tests/test_build_params.py
# pins all three together. This is the block tier's shared-memory limit (the only tier
# that scales past 32 DOF), not a tuning choice.
MAX_ACT_CEILING = 64

# Largest DOF at which the thread/warp tiers are offered; above it the kernels are
# locked to the block tier and PYROFFI_IK_TIER is ignored. Mirrors _glass_solve.cuh's
# TIER_CHOICE_MAX_N. Here only so the tooling can describe the behaviour — the lock
# itself is enforced in the CUDA dispatch, not from Python.
TIER_CHOICE_MAX_ACT = 32


class BuildParams(NamedTuple):
    """Capacity limits a particular kernel library was compiled with."""

    max_joints: int
    max_act: int
    lib_name: str

    def check(self, *, n_joints: int | None = None, n_act: int | None = None,
              kernel: str = "kernel") -> None:
        """Raise if this robot exceeds what ``lib_name`` was compiled to hold.

        Cheap (two int compares); call it on every invocation rather than trusting
        a cached robot, since a Robot can be rebuilt in-process.
        """
        if n_joints is not None and n_joints > self.max_joints:
            raise ValueError(
                f"{kernel}: robot has {n_joints} joints but {self.lib_name} was built "
                f"with MAX_JOINTS={self.max_joints}. The kernel would write past the end "
                f"of its per-thread pose buffers (undefined behaviour, not a crash — you "
                f"would get silently wrong results).\n"
                f"Rebuild with:  bash build_kernels/build_all.sh --max-joints {_round_up(n_joints)}"
            )
        if n_act is not None and n_act > self.max_act:
            if n_act > MAX_ACT_CEILING:
                raise ValueError(
                    f"{kernel}: robot has {n_act} actuated DOF, which exceeds pyroffi's "
                    f"hard ceiling of {MAX_ACT_CEILING}. This is NOT a rebuild away — no "
                    f"--max-act value supports it.\n"
                    f"Above 32 DOF only the block tier fits, and its shared "
                    f"double A[N*N] must stay inside the 48KB static shared budget; "
                    f"{MAX_ACT_CEILING} is the last size that does (see "
                    f"_build_params.cuh). Supporting more means moving the solve to "
                    f"dynamic shared memory.\n"
                    f"If this robot has unused DOF (e.g. hands you are not solving for), "
                    f"a URDF with those joints fixed will bring it under the ceiling."
                )
            raise ValueError(
                f"{kernel}: robot has {n_act} actuated DOF but {self.lib_name} was built "
                f"with MAX_ACT={self.max_act}. The kernel would write past the end of its "
                f"per-thread state arrays (undefined behaviour, not a crash — you would get "
                f"silently wrong results).\n"
                f"Rebuild with:  bash build_kernels/build_all.sh "
                f"--max-act {min(_round_up(n_act), MAX_ACT_CEILING)}\n"
                f"MAX_ACT is capped at {MAX_ACT_CEILING}. Raising it is not free: it sizes "
                f"per-thread arrays in every kernel, so a big build slows the smaller arms "
                f"too. Build for the robot you deploy."
            )


def _round_up(n: int, step: int = 8) -> int:
    """Suggest a sensible rebuild limit (round up, leave a little headroom)."""
    return ((n + step - 1) // step) * step


def read_build_params(lib: ctypes.CDLL, lib_name: str) -> BuildParams:
    """Read the limits ``lib`` was compiled with, via its exported accessors.

    Raises if the library predates the exported accessors — that .so's limits are
    unknowable from here, and silently assuming the defaults is the exact failure
    mode this module exists to remove.
    """
    try:
        fn_joints = lib.pyroffi_max_joints
        fn_act = lib.pyroffi_max_act
    except AttributeError as exc:
        raise RuntimeError(
            f"{lib_name} does not export pyroffi_max_joints/pyroffi_max_act, so its "
            f"MAX_JOINTS/MAX_ACT limits cannot be verified. It was built before those "
            f"accessors were added (see _build_params.cuh).\n"
            f"Rebuild it:  bash build_kernels/build_all.sh"
        ) from exc
    fn_joints.restype = ctypes.c_int
    fn_joints.argtypes = []
    fn_act.restype = ctypes.c_int
    fn_act.argtypes = []
    return BuildParams(max_joints=int(fn_joints()), max_act=int(fn_act()), lib_name=lib_name)


@lru_cache(maxsize=None)
def _params_cached(lib_path: str, lib_name: str) -> BuildParams:
    # dlopen on an already-loaded library just bumps its refcount, so this is cheap;
    # the lru_cache makes it once-per-library regardless.
    return read_build_params(ctypes.CDLL(lib_path), lib_name)


def assert_built_for(lib_path: str, lib_name: str, *, max_joints: int, max_act: int,
                     what: str) -> None:
    """Assert a library was built with the exact limits some hardcoded constant assumes.

    For Python-side values that were derived by hand against a specific build (e.g. a
    shared-memory-derived thread-per-block ceiling). Such a constant is only valid for
    the build it was measured on, but nothing tied the two together — so a `--max-act`
    rebuild silently invalidated it. This turns that into a loud, actionable error.

    Prefer deriving the value from :func:`read_build_params` where the formula is known;
    use this only where the constant came from measurement rather than arithmetic.
    """
    p = _params_cached(lib_path, lib_name)
    if (p.max_joints, p.max_act) != (max_joints, max_act):
        raise RuntimeError(
            f"{what} was determined for a MAX_JOINTS={max_joints}, MAX_ACT={max_act} build, "
            f"but {lib_name} reports MAX_JOINTS={p.max_joints}, MAX_ACT={p.max_act}. "
            f"That constant does not transfer: shared-memory use scales with both limits, "
            f"so the bound is wrong (too low wastes occupancy; too high fails the launch).\n"
            f"Either rebuild at the default limits, or re-derive {what} for your build and "
            f"update it."
        )


def check_capacity(module_file: str, lib_name: str, *, n_joints: int | None = None,
                   n_act: int | None = None, kernel: str) -> None:
    """Refuse the launch if this robot exceeds what ``lib_name`` was compiled for.

    One-liner for the FFI wrappers, which sit next to their ``.so``::

        check_capacity(__file__, _LIB_NAME, n_joints=twists.shape[0],
                       n_act=seeds.shape[-1], kernel="ls_ik_cuda")

    Call it under ``jit`` tracing only — the shapes are static, so this costs nothing
    at runtime and the error surfaces at trace time rather than as corrupted output.
    """
    _params_cached(str(Path(module_file).parent / lib_name), lib_name).check(
        n_joints=n_joints, n_act=n_act, kernel=kernel
    )
