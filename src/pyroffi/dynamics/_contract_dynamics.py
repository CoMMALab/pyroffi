"""Forward dynamics through the traced-robot contract (M2 of the tiered trace-compilation design).

One generated header, ``_contract_robot_gen.cuh``, gives kernels ``pyroffi::traced`` with

* ``contract_version`` and ``n_q``;
* ``thread::forward_dynamics(q, qd, tau, qdd)``: cricket's traced articulated-body algorithm,
  one config per thread, in registers;
* ``block::forward_dynamics(s_q, s_qd, s_tau, s_qdd, s_scratch, model, gravity)``: GRiD's
  ``forward_dynamics_device``, one config per block. It owns the dynamic shared-memory arena
  (``forward_dynamics_smem_bytes``, GRiD's layout), so callers keep their own data in static
  shared memory.

Every argument is in PyRoFFI's actuated joint order. The maps to cricket's order and to GRiD's
order (a permutation plus axis signs) are generated at the boundary, so the kernels never see a
provider's conventions. Neither provider models joint damping; the URDF should carry none.
"""

from __future__ import annotations

import ctypes
import hashlib
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

_KERNEL = Path(__file__).parent.parent / "cuda_kernels" / "dynamics" / "_contract_dynamics_kernel.cu"
CONTRACT_VERSION = 1


def _cricket_source(urdf_xml: str) -> tuple[str, list[str]]:
    """cricket's CUDA header with forward dynamics traced, and its joint order."""
    import cricket

    from ..cuda_kernels._traced import _EMPTY_SRDF

    with tempfile.TemporaryDirectory(prefix="pyroffi_contract_") as tmp:
        urdf, srdf = Path(tmp) / "robot.urdf", Path(tmp) / "robot.srdf"
        urdf.write_text(urdf_xml)
        srdf.write_text(_EMPTY_SRDF)
        opts = cricket.GenOptions(urdf=urdf, srdf=srdf, language="cuda", data={"name": "Robot"})
        opts.forward_dynamics = True
        gen = cricket.generate_robot_source(opts)
    return gen.source, list(gen.data["joint_names"])


def contract_header(cricket_src: str, cricket_names: list[str], actuated: list[str],
                    grid_perm=None, grid_signs=None) -> str:
    """The contract header for one robot. GRiD's block tier is included when its map is given."""
    n = len(actuated)
    if sorted(cricket_names) != sorted(actuated):
        raise ValueError(f"cricket joints {cricket_names} do not match actuated joints {actuated}")
    src = [actuated.index(name) for name in cricket_names]
    gather = "".join(f"    x[{k * n + i}] = {arg}[{a}];\n"
                     for k, arg in enumerate(("q", "qd", "tau")) for i, a in enumerate(src))
    scatter = "".join(f"    qdd[{a}] = y[{i}];\n" for i, a in enumerate(src))
    out = [
        "#pragma once\n// PyRoFFI traced-robot contract, generated. Do not edit.\n",
        cricket_src,
        "\nnamespace pyroffi::traced {\n",
        f"constexpr int contract_version = {CONTRACT_VERSION};\nconstexpr int n_q = {n};\n",
        "namespace thread {\n",
        "// cricket: traced ABA, one config per thread.\n",
        "__device__ __forceinline__ void forward_dynamics(const float* q, const float* qd,\n"
        "                                                 const float* tau, float* qdd)\n",
        f"{{\n    float x[{3 * n}], y[{n}];\n{gather}"
        f"    cricket::robots::robot::forward_dynamics(x, y);\n{scatter}}}\n",
        "}  // namespace thread\n}  // namespace pyroffi::traced\n",
    ]
    if grid_perm is not None:
        perm = ", ".join(str(int(p)) for p in grid_perm)
        signs = ", ".join(f"{float(s):.1f}f" for s in grid_signs)
        out += [
            '\n#define PYROFFI_CONTRACT_BLOCK_FORWARD_DYNAMICS\n#include "grid.cuh"\n',
            "namespace pyroffi::traced::block {\n",
            "// GRiD: forward_dynamics_device, one config per block.\n",
            "using Model = const grid::robotModel<float>*;\n",
            "inline Model make_model() { return grid::init_robotModel<float>(); }\n",
            "constexpr int forward_dynamics_threads = grid::MAX_PERF_LEVEL_THREADS;\n",
            "constexpr unsigned forward_dynamics_smem_bytes =\n"
            "    grid::FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<float, grid::TIER_SHARED>();\n",
            "constexpr int forward_dynamics_scratch = 4 * n_q;  // floats of caller shared memory\n",
            "__device__ __forceinline__ void forward_dynamics(const float* s_q, const float* s_qd,\n"
            "    const float* s_tau, float* s_qdd, float* s_scratch, Model model, float gravity)\n{\n",
            # q_grid[g] = sign[g] * q_act[perm[g]]; signs are +-1, so the inverse map reuses them.
            f"    constexpr int perm[n_q] = {{{perm}}};\n",
            f"    constexpr float sign[n_q] = {{{signs}}};\n",
            "    float *g_q = s_scratch, *g_qd = g_q + n_q, *g_tau = g_qd + n_q, *g_qdd = g_tau + n_q;\n",
            "    for (int g = threadIdx.x; g < n_q; g += blockDim.x) {\n"
            "        g_q[g] = sign[g] * s_q[perm[g]];\n"
            "        g_qd[g] = sign[g] * s_qd[perm[g]];\n"
            "        g_tau[g] = sign[g] * s_tau[perm[g]];\n    }\n",
            "    __syncthreads();\n",
            "    grid::forward_dynamics_device<float, grid::TIER_SHARED>(g_qdd, g_q, g_qd, g_tau, model,\n"
            "                                                            nullptr, gravity);\n",
            "    __syncthreads();\n",
            "    for (int g = threadIdx.x; g < n_q; g += blockDim.x) s_qdd[perm[g]] = sign[g] * g_qdd[g];\n",
            "    __syncthreads();\n}\n",
            "}  // namespace pyroffi::traced::block\n",
        ]
    return "".join(out)


class ContractDynamics:
    """Forward dynamics of one robot at each tier its contract header provides.

    ``tier="thread"`` runs cricket's trace (one config per thread); ``tier="block"`` runs GRiD
    (one config per block). Both take and return ``(B, n_q)`` float32 in actuated order.
    """

    def __init__(self, urdf, gravity: float = -9.81, block: bool = True):
        from ..cuda_kernels._traced import build_shared_library
        from .._cuda_backends import _urdf_xml
        from .._robot import Robot

        self.gravity = float(gravity)
        actuated = list(Robot.from_urdf(urdf).joints.actuated_names)
        self.n_q = len(actuated)
        cricket_src, cricket_names = _cricket_source(_urdf_xml(urdf))
        generated = {}
        perm = signs = None
        if block:
            from ._grid_codegen import generate_grid_cuh
            from ._grid_robot_adapter import build_grid_robot

            grid_model = build_grid_robot(urdf)
            perm, signs = grid_model.joint_perm, grid_model.axis_signs
            generated["grid.cuh"] = generate_grid_cuh(grid_model)
        header = contract_header(cricket_src, cricket_names, actuated, perm, signs)
        generated["_contract_robot_gen.cuh"] = header
        flags = ["-O3", "-std=c++17", "-arch=native", "--shared", "--compiler-options", "-fPIC"]
        lib = ctypes.CDLL(str(build_shared_library("contract_dynamics", _KERNEL, generated, flags)))

        capsule_new = ctypes.pythonapi.PyCapsule_New
        capsule_new.restype = ctypes.py_object
        capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        tag = hashlib.sha1("".join(generated.values()).encode()).hexdigest()[:16]
        self.tiers = ("thread", "block") if block else ("thread",)
        self._targets = {}
        for tier in self.tiers:
            symbol = f"ContractFd{tier.capitalize()}Ffi"
            capsule = capsule_new(ctypes.cast(getattr(lib, symbol), ctypes.c_void_p),
                                  b"xla._CUSTOM_CALL_TARGET", None)
            self._targets[tier] = f"{symbol}_{tag}"
            jax.ffi.register_ffi_target(self._targets[tier], capsule, platform="CUDA")

    def forward_dynamics(self, q: Array, qd: Array, tau: Array, tier: str = "thread") -> Array:
        if tier not in self._targets:
            raise ValueError(f"tier {tier!r} not in this robot's contract header ({self.tiers})")
        q, qd, tau = (jnp.asarray(a, jnp.float32).reshape(-1, self.n_q) for a in (q, qd, tau))
        attrs = {"gravity": np.float32(self.gravity)} if tier == "block" else {}
        return jax.ffi.ffi_call(self._targets[tier], jax.ShapeDtypeStruct(q.shape, jnp.float32))(
            q, qd, tau, **attrs)
