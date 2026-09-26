"""Tiered rigid-body dynamics through the traced-robot contract (M2/M5 of the tiered design).

One generated header, ``_contract_robot_gen.cuh``, gives the kernels in
``cuda_kernels/dynamics/_contract_dynamics_kernel.cu`` these ops of one robot, each at two tiers:

======== ======================================= ==========================================
op       computes                                 inputs -> outputs (generator joint order)
======== ======================================= ==========================================
id       inverse dynamics (RNEA)                  [q, qd, qdd] -> tau
crba     mass matrix (CRBA)                       q -> M (column-major)
fd       forward dynamics (ABA)                   [q, qd, tau] -> qdd
fd_crba  forward dynamics as M qdd = tau - b      [q, qd, tau] -> qdd (GLASS Cholesky)
id_du    inverse-dynamics derivatives             [q, qd, qdd] -> [dtau/dq, dtau/dqd]
======== ======================================= ==========================================

* ``thread``: cricket's straight-line trace, one configuration per thread.
* ``block``: the same trace scheduled by ``cricket.tiered`` (warp-split): 32 configurations per
  block, the block's warps splitting each dataflow level.

``grid`` (GRiD's block-cooperative kernels, one configuration per block) is a third provider for
id/crba/fd/id_du, called through :class:`~pyroffi.dynamics.GRiDDynamics`.

``tier="auto"`` picks the provider per call at trace time from the batch size, using this robot's
calibration (:meth:`ContractDynamics.calibrate`) when one exists, else the measured default
table. Arguments and results are float32 in PyRoFFI's actuated joint order. Neither generator
models joint damping, so the URDF should carry none.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

_KERNEL = Path(__file__).parent.parent / "cuda_kernels" / "dynamics" / "_contract_dynamics_kernel.cu"
_REPO = Path(__file__).resolve().parents[3]
_DEFAULT_TABLE = _REPO / "resources" / "tier_tables" / "contract_dynamics_sm86.json"
CONTRACT_VERSION = 2
BLOCK_WARPS = 8
# A provider replaces thread only when calibration measures it this much faster: below it the
# difference is launch/dispatch noise (~50 us per call through JAX) and flips run to run.
_MARGIN = 0.85
_SMEM_LIMIT = 96 * 1024

# op -> (cricket GenOptions flag, inputs per joint, outputs as a function of n)
_TRACED = {
    "id": ("inverse_dynamics", 3, lambda n: n),
    "crba": ("mass_matrix", 1, lambda n: n * n),
    "fd": ("forward_dynamics", 3, lambda n: n),
    "id_du": ("inverse_dynamics_derivatives", 3, lambda n: 2 * n * n),
}
CANDIDATES = {
    "id": ("thread", "block", "grid"),
    "crba": ("thread", "block", "grid"),
    "fd": ("thread", "block", "grid", "thread_crba", "block_crba"),
    "id_du": ("thread", "block", "grid"),
}


def _cricket_traces(urdf_xml: str, with_derivatives: bool) -> tuple[dict, list[str]]:
    import cricket

    from ..cuda_kernels._traced import _EMPTY_SRDF

    with tempfile.TemporaryDirectory(prefix="pyroffi_contract_") as tmp:
        urdf, srdf = Path(tmp) / "robot.urdf", Path(tmp) / "robot.srdf"
        urdf.write_text(urdf_xml)
        srdf.write_text(_EMPTY_SRDF)
        opts = cricket.GenOptions(urdf=urdf, srdf=srdf, language="cuda", data={"name": "Robot"})
        for flag, *_ in _TRACED.values():
            setattr(opts, flag, flag != "inverse_dynamics_derivatives" or with_derivatives)
        gen = cricket.generate_robot_source(opts)
    return gen.data, list(gen.data["joint_names"])


def _thread_fn(name: str, code: str) -> str:
    n_v = max([int(v) for v in re.findall(r"\bv\[(\d+)\]", code)] + [0]) + 1
    body = re.sub(r"\by\[(\d+)\]", r"y[\1 * ys]", code)
    return (f"__device__ __forceinline__ void {name}_thread(const float* x, float* y, int ys)\n"
            f"{{\n    float v[{n_v}];\n{body}\n}}\n")


def _block_fn(name: str, code: str) -> tuple[str, int]:
    from cricket import tiered

    sched = tiered.schedule(code, BLOCK_WARPS)
    src = tiered.emit(sched, f"{name}_block", "warp", y_stride="ys")
    return src.replace("float* s)", "float* s, int ys)", 1), tiered.smem_floats(sched, "warp")


def contract_header(data: dict, n: int, ops: tuple[str, ...]) -> str:
    """The contract header: every op in ``ops`` at thread and block tier, generator joint order."""
    out = ["#pragma once\n// PyRoFFI traced-robot contract, generated. Do not edit.\n",
           '#include "glass.cuh"\n',
           f"namespace pyroffi::traced {{\nconstexpr int contract_version = {CONTRACT_VERSION};\n"
           f"constexpr int n_q = {n};\n}}\n",
           "namespace pyroffi::traced::ops {\n"]
    scratch, entries = {}, []
    for op in ("id", "crba", "fd", "id_du"):
        if op not in ops and not (op in ("id", "crba") and "fd_crba" in ops):
            continue
        flag, per_joint, n_out = _TRACED[op]
        code = data[f"{flag}_code"]
        block, floats = _block_fn(op, code)
        scratch[op] = floats
        out += [_thread_fn(op, code), block]
        if op in ops:
            entries.append((op, per_joint * n, n_out(n)))
    if "fd_crba" in ops:
        # Forward dynamics as M(q) qdd = tau - b(q, qd), b = RNEA(q, qd, 0), by GLASS Cholesky.
        m_off = scratch["crba"] + scratch["id"]
        b_off = m_off + 32 * n * n
        scratch["fd_crba"] = b_off + 32 * n
        out.append(f"""__device__ __forceinline__ void fd_crba_solve(float* M, float* b)
{{
    int fail = 0;
    glass::thread::potrf<float, {n}, true>(M, &fail);
    glass::thread::potrs<float, {n}>(M, b);
}}
__device__ __forceinline__ void fd_crba_thread(const float* x, float* y, int ys)
{{
    float M[{n * n}], xb[{3 * n}], b[{n}];
    crba_thread(x, M, 1);
    for (int i = 0; i < {2 * n}; ++i) xb[i] = x[i];
    for (int i = 0; i < {n}; ++i) xb[{2 * n} + i] = 0.f;
    id_thread(xb, b, 1);
    for (int i = 0; i < {n}; ++i) b[i] = x[{2 * n} + i] - b[i];
    fd_crba_solve(M, b);
    for (int i = 0; i < {n}; ++i) y[i * ys] = b[i];
}}
// Mass matrix and bias by the warp-split tier into per-lane scratch, then warp 0's lanes each
// solve their own configuration.
__device__ __forceinline__ void fd_crba_block(int rank, const float* x, float* y, float* s, int ys)
{{
    const int lane = threadIdx.x & 31;
    float xb[{3 * n}];
    for (int i = 0; i < {2 * n}; ++i) xb[i] = x[i];
    for (int i = 0; i < {n}; ++i) xb[{2 * n} + i] = 0.f;
    crba_block(rank, x, s + {m_off} + lane, s, 32);
    id_block(rank, xb, s + {b_off} + lane, s + {scratch["crba"]}, 32);
    if (rank != 0) return;
    float M[{n * n}], b[{n}];
    for (int i = 0; i < {n * n}; ++i) M[i] = s[{m_off} + i * 32 + lane];
    for (int i = 0; i < {n}; ++i) b[i] = x[{2 * n} + i] - s[{b_off} + i * 32 + lane];
    fd_crba_solve(M, b);
    for (int i = 0; i < {n}; ++i) y[i * ys] = b[i];
}}
""")
        entries.append(("fd_crba", 3 * n, n))
    for op, floats in scratch.items():
        out.append(f"constexpr int {op}_scratch = {floats};\n"
                   f"constexpr bool {op}_smem = {'true' if floats * 4 <= _SMEM_LIMIT else 'false'};\n"
                   f"constexpr int {op}_warps = {BLOCK_WARPS};\n")
    out.append("}  // namespace pyroffi::traced::ops\n#define PYROFFI_CONTRACT_OPS(X) "
               + " ".join(f"X({op}, {nin}, {nout})" for op, nin, nout in entries) + "\n")
    return "".join(out)


class ContractDynamics:
    """Dynamics of one robot at every tier its contract header and GRiD provide.

    Methods take ``(B, n)`` float32 arrays and a ``tier``: ``"auto"``, or one of
    :data:`CANDIDATES` for that op.
    """

    def __init__(self, urdf, gravity: float = -9.81, grid: bool = True,
                 ops: tuple[str, ...] = ("id", "crba", "fd", "fd_crba", "id_du")):
        from .._cuda_backends import _urdf_xml
        from .._robot import Robot
        from ..cuda_kernels._traced import _GLASS_DIR, build_shared_library

        self.gravity = float(gravity)
        self.actuated = list(Robot.from_urdf(urdf).joints.actuated_names)
        self.n_q = n = len(self.actuated)
        data, names = _cricket_traces(_urdf_xml(urdf), "id_du" in ops)
        if sorted(names) != sorted(self.actuated):
            raise ValueError(f"cricket joints {names} do not match actuated joints {self.actuated}")
        self._to_gen = np.array([self.actuated.index(nm) for nm in names])  # generator <- actuated
        self._from_gen = np.argsort(self._to_gen)
        self.ops = ops
        header = contract_header(data, n, ops)
        flags = ["-O3", "-std=c++17", os.environ.get("PYROFFI_TRACED_GPU_ARCH", "-arch=native"),
                 "--shared", "--compiler-options", "-fPIC"]
        glass = (*sorted(_GLASS_DIR.glob("*.cuh")), *sorted((_GLASS_DIR / "src").rglob("*.cuh")))
        so = build_shared_library("contract_dynamics", _KERNEL, {"_contract_robot_gen.cuh": header},
                                  flags, include_dirs=(_GLASS_DIR,), key_files=glass)
        self.key = hashlib.sha1(header.encode()).hexdigest()[:16]
        lib = ctypes.CDLL(str(so))
        capsule_new = ctypes.pythonapi.PyCapsule_New
        capsule_new.restype = ctypes.py_object
        capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        self._targets = {}
        for op in ops:
            for tier in ("thread", "block"):
                symbol = f"Contract_{op}_{tier}"
                capsule = capsule_new(ctypes.cast(getattr(lib, symbol), ctypes.c_void_p),
                                      b"xla._CUSTOM_CALL_TARGET", None)
                self._targets[op, tier] = f"{symbol}_{self.key}"
                jax.ffi.register_ffi_target(self._targets[op, tier], capsule, platform="CUDA")
        self._grid = None
        if grid:
            from ._grid_dynamics import GRiDDynamics

            self._grid = GRiDDynamics(urdf, gravity=gravity)
        self._table = self._load_table()

    # ── public ops ───────────────────────────────────────────────────────────────────────────
    def inverse_dynamics(self, q, qd, qdd, tier: str = "auto") -> Array:
        return self._run("id", tier, q, qd, qdd)

    def mass_matrix(self, q, tier: str = "auto") -> Array:
        return self._run("crba", tier, q)

    def forward_dynamics(self, q, qd, tau, tier: str = "auto") -> Array:
        return self._run("fd", tier, q, qd, tau)

    def inverse_dynamics_gradient(self, q, qd, qdd, tier: str = "auto") -> Array:
        """``[dtau/dq | dtau/dqd]``, shape ``(B, n, 2n)``, rows the output joint (GRiD's layout)."""
        return self._run("id_du", tier, q, qd, qdd)

    # ── tier selection ───────────────────────────────────────────────────────────────────────
    def available(self, op: str) -> tuple[str, ...]:
        have = {t for (o, t) in self._targets if o == op}
        if op == "fd":
            have |= {f"{t}_crba" for (o, t) in self._targets if o == "fd_crba"}
        if self._grid is not None:
            have.add("grid")
        return tuple(t for t in CANDIDATES[op] if t in have)

    def select(self, op: str, batch: int) -> str:
        """The provider ``tier="auto"`` uses for ``op`` at ``batch`` configurations.

        Precedence: ``$PYROFFI_TIER``, then the nearest calibrated (n, batch) point in log space
        (this robot's own calibration if it has one, else the default table), then ``thread``.
        """
        env = os.environ.get("PYROFFI_TIER")
        if env:
            return env
        rows = self._table.get(op)
        if not rows:
            return "thread"
        best = min(rows, key=lambda r: np.log(r["n"] / self.n_q) ** 2 + np.log(r["batch"] / batch) ** 2)
        return best["winner"] if best["winner"] in self.available(op) else "thread"

    def calibrate(self, batches=(16, 64, 256, 1024, 4096, 65536), repeats: int = 20) -> dict:
        """Time every provider of every op at ``batches`` on this GPU; cache and use the winners.

        A provider that fails at a size (GRiD's workspace runs out at large batches) is skipped
        there. The winner is the fastest provider if it beats ``thread`` by ``_MARGIN``, else
        ``thread``.
        """
        rows = {}
        for op in (o for o in ("id", "crba", "fd", "id_du") if o in self.ops):
            rows[op] = []
            for batch in batches:
                args = [jax.random.uniform(jax.random.PRNGKey(k), (batch, self.n_q), minval=-1, maxval=1)
                        for k in range(1 if op == "crba" else 3)]
                times = {}
                for tier in self.available(op):
                    f = jax.jit(lambda *a, tier=tier, op=op: self._run(op, tier, *a))
                    try:
                        for _ in range(3):
                            jax.block_until_ready(f(*args))
                        t0 = time.perf_counter()
                        for _ in range(repeats):
                            res = f(*args)
                        jax.block_until_ready(res)
                    except jax.errors.JaxRuntimeError:
                        continue
                    times[tier] = (time.perf_counter() - t0) / repeats * 1e3
                fastest = min(times, key=times.get)
                winner = fastest if times[fastest] < _MARGIN * times.get("thread", np.inf) else "thread"
                rows[op].append({"n": self.n_q, "batch": batch, "winner": winner,
                                 "ms": {k: round(v, 4) for k, v in times.items()}})
        self._table = rows
        path = self._calibration_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=1))
        return rows

    # ── internals ────────────────────────────────────────────────────────────────────────────
    def _calibration_path(self) -> Path:
        from ..cuda_kernels._traced import _cache_root

        gpu = jax.devices()[0].device_kind.replace(" ", "_")
        return _cache_root() / "tier_tables" / f"{self.key}_{gpu}.json"

    def _load_table(self) -> dict:
        for path in (self._calibration_path(), _DEFAULT_TABLE):
            if path.is_file():
                return json.loads(path.read_text())
        return {}

    def _run(self, op: str, tier: str, *args) -> Array:
        n = self.n_q
        args = [jnp.asarray(a, jnp.float32).reshape(-1, n) for a in args]
        batch = args[0].shape[0]
        if tier == "auto":
            tier = self.select(op, batch)
        if tier not in self.available(op):
            raise ValueError(f"{op}: tier {tier!r} not available ({self.available(op)})")
        if tier == "grid":
            g = self._grid
            return {"id": g.inverse_dynamics, "crba": g.mass_matrix, "fd": g.forward_dynamics,
                    "id_du": g.inverse_dynamics_gradient}[op](*args)
        kernel = "fd_crba" if tier.endswith("_crba") else op
        packed = jnp.concatenate([a[:, self._to_gen] for a in args], axis=1)
        n_out = {"id": n, "fd": n, "fd_crba": n, "crba": n * n, "id_du": 2 * n * n}[kernel]
        res = jax.ffi.ffi_call(self._targets[kernel, tier.removesuffix("_crba")],
                               jax.ShapeDtypeStruct((n_out, batch), jnp.float32))(packed).T
        g = self._from_gen
        if kernel in ("id", "fd", "fd_crba"):
            return res[:, g]
        if kernel == "crba":
            M = jnp.swapaxes(res.reshape(batch, n, n), 1, 2)  # column-major -> [row, col]
            return M[:, g][:, :, g]
        D = jnp.swapaxes(res.reshape(batch, 2, n, n), 2, 3)[:, :, g][:, :, :, g]
        return jnp.concatenate([D[:, 0], D[:, 1]], axis=-1)
