"""Compile a :class:`~pyroffi.LeastSquaresProblem` into one CUDA kernel.

``problem.analyze()`` traces the problem's residual once, together with its forward-mode
Jacobian, and emits it as straight-line code (see :mod:`._emit`) into ``_cost_gen.cuh``,
which a Levenberg-Marquardt / augmented-Lagrangian kernel includes. ``solve()`` is then a
single FFI call: no JAX runs inside the optimisation loop, and there is no JAX fallback.

What is baked into the build, and what stays a runtime input:

* **Baked** (a change is a different build): the problem's structure (which factories,
  which variables each cost touches, residual sizes), robot and robot-collision models,
  NumPy arrays and Python scalars (weights such as ``pos_weight=50.0``, margins).
* **Runtime** (changing it reuses the build): every ``jax.Array`` argument, such as target
  poses, rest poses, obstacle geometry, or a traced link index.

Builds are cached by that split, in memory and (the ``.so``) on disk, so re-analyzing a
problem that differs only in runtime inputs costs a dictionary lookup. Inside ``jax.jit``
a traced robot or collision model is resolved to the concrete one it was built from
(see :mod:`pyroffi._registry`).

Scope: Euclidean variables (e.g. ``robot.joint_var_cls``), at most 32 scalars in total
(the per-thread dense solve). Anything else raises.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .. import _registry
from . import _emit as E
from ._problem import (AugmentedLagrangianConfig, LeastSquaresProblem, SolveSummary,
                       TerminationConfig, TrustRegionConfig, Var, VarValues, _concrete)

_ROW_KIND = {"l2_squared": 0, "constraint_eq_zero": 1, "constraint_leq_zero": 2,
             "constraint_geq_zero": 2}
_MAX_VARS = 128   # block tier: one Jacobian column per thread of a 128-thread block
_LANE_TIERS_MAX_VARS = 32   # thread and warp tiers
_DEFAULTS = dict(max_iters=100, al_iters=20, lambda_initial=5e-4, tolerance=1e-5,
                 al_tolerance=1e-5, grad_tolerance=1e-4, grad_start=10, param_tolerance=1e-6)


def _baked_types() -> tuple[type, ...]:
    from .._robot import Robot
    from ..collision import RobotCollision, RobotCollisionSpherized

    return (Robot, RobotCollision, RobotCollisionSpherized)


def _bake_leaf(x) -> np.ndarray:
    if isinstance(x, jax.core.Tracer):
        raise ValueError(
            "A NumPy/Python argument of a cost is traced; pass values that should be runtime "
            "inputs as jax.Array, and values that should be baked in as concrete constants.")
    a = np.asarray(x)
    if a.dtype == np.float64:
        return a.astype(np.float32)
    if a.dtype == np.int64:
        return a.astype(np.int32)
    return a


def _concrete_var(v: Var) -> Var:
    return type(v)(_concrete(v.id)) if isinstance(v.id, jax.core.Tracer) else v


def _fingerprint(tree) -> tuple:
    leaves, treedef = jax.tree.flatten(tree)
    return (treedef, tuple((a.shape, str(a.dtype), hashlib.sha1(np.ascontiguousarray(a).tobytes())
                            .hexdigest()) for a in map(np.asarray, leaves)))


@dataclass
class _CostPlan:
    """One cost (possibly batched) split into variables, baked values and runtime params."""

    cost: Any
    treedef: Any
    kinds: list[str]           # per top-level item: "var" | "bake" | "param"
    items: list[Any]           # broadcast to the batch: Var / baked pytree / None for params
    params: list[Any]          # broadcast runtime arrays, in order (may be traced)
    batch: int | None          # None: unbatched

    def key(self) -> tuple:
        parts = []
        for kind, item in zip(self.kinds, self.items):
            if kind == "var":
                parts.append(("var", type(item), np.asarray(item.id).tobytes(), np.shape(item.id)))
            elif kind == "bake":
                parts.append(("bake", None if item is None else _fingerprint(item)))
            else:
                parts.append(("param",))
        shapes = tuple((tuple(p.shape), str(p.dtype)) for p in self.params)
        return (self.cost._get_name(), self.cost.kind, self.treedef, tuple(parts), shapes)


def _is_item(x) -> bool:
    return isinstance(x, (Var, *_baked_types()))


def _batch_axes(items) -> tuple[int, ...]:
    ids = [np.shape(v.id) for v in items if isinstance(v, Var)]
    if not ids:
        raise ValueError("every cost needs at least one pk.Var argument")
    axes = np.broadcast_shapes(*ids)
    if len(axes) > 1:
        raise NotImplementedError("costs with more than one batch axis")
    return axes


def _broadcast(x, axes, xp):
    x = xp.asarray(x)
    return xp.broadcast_to(x, axes + x.shape[len(axes):]) if axes else x


def _plan(cost) -> _CostPlan:
    items, treedef = jax.tree.flatten(cost.args, is_leaf=_is_item)
    items = [_concrete_var(it) if isinstance(it, Var) else it for it in items]
    axes = _batch_axes(items)
    kinds, out, params = [], [], []
    for it in items:
        if isinstance(it, Var):
            kinds.append("var")
            out.append(type(it)(np.broadcast_to(np.asarray(it.id), axes)) if axes else it)
        elif isinstance(it, _baked_types()):
            kinds.append("bake")
            concrete = _registry.resolve(it)
            out.append(jax.tree.map(lambda a: _broadcast(_bake_leaf(a), axes, np), concrete))
        elif isinstance(it, jax.Array):
            kinds.append("param")
            out.append(None)
            params.append(_broadcast(it, axes, jnp))
        else:
            kinds.append("bake")
            out.append(None if it is None else _broadcast(_bake_leaf(it), axes, np))
    return _CostPlan(cost, treedef, kinds, out, params, axes[0] if axes else None)


def _instance(plan: _CostPlan, i: int | None, params: list) -> tuple:
    it_params = iter(params)
    items = []
    for kind, item in zip(plan.kinds, plan.items):
        if kind == "var":
            items.append(item if i is None else type(item)(int(item.id[i])))
        elif kind == "param":
            p = next(it_params)
            items.append(p if i is None else p[i])
        elif item is None:
            items.append(None)
        else:
            items.append(item if i is None else jax.tree.map(lambda a: a[i], item))
    return jax.tree.unflatten(plan.treedef, items)


def _variable_layout(variables) -> tuple[dict, list, int]:
    """(type, id) -> (offset, size, shape); plus the ordered slots and total size."""
    layout, slots, n = {}, [], 0
    for var in variables:
        vtype = type(var)
        if vtype.retract_fn is not None:
            raise NotImplementedError(
                f"{vtype.__name__}: the compiled solver supports Euclidean variables only; "
                "Lie-group variables (SE3Var, SO3Var, ...) are not implemented yet")
        default = np.asarray(vtype.default_factory(), np.float32)
        for vid in np.atleast_1d(np.asarray(var.id)).ravel():
            key = (vtype, int(vid))
            if key in layout:
                continue
            layout[key] = (n, default.size, default.shape)
            slots.append(key)
            n += default.size
    return layout, slots, n


def _require_cuda() -> None:
    try:
        jax.devices("cuda")
    except RuntimeError as e:
        raise RuntimeError("pk.LeastSquaresProblem.analyze() compiles to CUDA and needs a CUDA "
                           "device visible to JAX; none was found.") from e


TIERS = ("thread", "warp", "block")
_TIER_INDEX = {"thread": 0, "warp": 1, "block": 2}
COOP_LOOP_MIN = 256
"""Loop threshold of the cooperative (warp/block) variant; the thread variant is fully
scalarized, GLASS's per-thread form."""
_SHARED_BUDGET = 48 * 1024
_THREAD_BUDGET = 64 * 1024


@lru_cache(maxsize=None)
def _build(header: str, tier: str) -> str:
    from ..cuda_kernels._traced import (_GLASS_DIR, _KERNELS_DIR, build_shared_library,
                                        register_cuda_targets)

    source = _KERNELS_DIR / "costs" / "_cost_lm_kernel.cu"
    flags = ["-O3", "-std=c++17", os.environ.get("PYROFFI_TRACED_GPU_ARCH", "-arch=native"),
             f"-DPYROFFI_COST_TIER={_TIER_INDEX[tier]}", "--shared", "--compiler-options", "-fPIC"]
    keys = (source, *sorted(_KERNELS_DIR.glob("*.cuh")), *sorted(_GLASS_DIR.glob("*.cuh")),
            *sorted((_GLASS_DIR / "src").rglob("*.cuh")))
    try:
        so = build_shared_library(f"cost_lm_{tier}", source, {"_cost_gen.cuh": header}, flags,
                                  include_dirs=(_KERNELS_DIR, _GLASS_DIR), key_files=keys)
    except FileNotFoundError as e:
        if shutil.which("nvcc") is None:
            raise RuntimeError("pk.LeastSquaresProblem.analyze() compiles to CUDA and needs "
                               "nvcc on PATH; it was not found.") from e
        raise
    tag = hashlib.sha1((header + tier).encode()).hexdigest()[:16]
    return register_cuda_targets(so, ("CostLmFfi",), tag)[0]


@dataclass
class _Variant:
    header: str
    scratch: int       # floats of scratch per problem
    max_extent: int    # largest loop (the parallelism a cooperative tier can use)


class _Kernel:
    """A compiled problem structure: the traced residual, its tier variants and FFI targets.

    Variants are generated, and tiers built, on first use: a problem only pays nvcc for the
    tiers it runs at.
    """

    def __init__(self, variables: list[Var], plans: list[_CostPlan]):
        self.layout, self.slots, self.n_x = _variable_layout(variables)
        if self.n_x > _MAX_VARS:
            raise NotImplementedError(
                f"{self.n_x} variables: the compiled dense solve supports up to {_MAX_VARS} "
                "(larger problems, e.g. trajectories, are not implemented yet)")
        self.plans = plans
        self.param_specs = [[(tuple(p.shape), p.dtype) for p in plan.params] for plan in plans]
        self.n_p = sum(int(np.prod(s, dtype=int)) for specs in self.param_specs for s, _ in specs)
        x0, p0 = np.zeros(self.n_x, np.float32), np.zeros(self.n_p, np.float32)

        def with_jac(x, p):
            residual = lambda x: jnp.concatenate(self._residual_rows(x, p))
            return jax.jacfwd(lambda x: (residual(x),) * 2, has_aux=True)(x)

        def with_jvp(x, p, t):
            return jax.jvp(lambda x: jnp.concatenate(self._residual_rows(x, p)), (x,), (t,))

        self.closed = jax.make_jaxpr(with_jac)(x0, p0)
        self.closed_jvp = jax.make_jaxpr(with_jvp)(x0, p0, x0)
        kinds, sign = [], []
        for plan, rows in zip(self.plans, jax.eval_shape(self._residual_rows, x0, p0)):
            kinds += [_ROW_KIND[plan.cost.kind]] * rows.shape[0]
            sign += [-1 if plan.cost.kind == "constraint_geq_zero" else 1] * rows.shape[0]
        self.row_kinds, self.row_sign = kinds, np.array(sign, np.float32)
        self.n_r = len(kinds)
        self.plans = None  # only needed to trace; may hold the caller's tracers
        self._variants: dict[str, _Variant] = {}
        self._targets: dict[str, str] = {}
        self.tier_table: dict[int, str] = {}   # batch size -> fastest tier (from calibrate)

    def _residual_rows(self, x, p) -> list[jax.Array]:
        """Each cost's residual rows (all batch instances), in problem order."""
        vals = VarValues({key: x[off:off + size].reshape(shape)
                          for key, (off, size, shape) in self.layout.items()})
        per_cost, off = [], 0
        for plan, specs in zip(self.plans, self.param_specs):
            params = []
            for shape, dtype in specs:
                size = int(np.prod(shape, dtype=int))
                params.append(p[off:off + size].reshape(shape).astype(dtype))
                off += size
            rows = []
            for i in ([None] if plan.batch is None else range(plan.batch)):
                out = plan.cost.compute_residual(vals, *_instance(plan, i, params))
                if isinstance(out, tuple):  # (residual, jacobian cache)
                    out = out[0]
                rows.append(jnp.ravel(out).astype(jnp.float32))
            per_cost.append(jnp.concatenate(rows))
        return per_cost

    # ── tiers ───────────────────────────────────────────────────────────────
    @staticmethod
    def variant_of(tier: str) -> str:
        return "thread" if tier == "thread" else "coop"

    def variant(self, name: str) -> _Variant:
        """``thread``: the residual and all Jacobian columns as one folded straight-line body
        (one problem per thread). ``coop``: residual plus ONE Jacobian column for a runtime
        tangent, so the lanes of a warp/block each take a column with identical code."""
        if name not in self._variants:
            with jax.ensure_compile_time_eval():
                self._variants[name] = (self._generate(float("inf")) if name == "thread"
                                        else self._generate_jvp())
        return self._variants[name]

    def slot_bytes(self, tier: str) -> int:
        """Per-problem solver state held in shared (warp/block) or local (thread) memory;
        the generated code's scratch is a separate global workspace."""
        n, r = self.n_x, self.n_r
        return 4 * (4 * n + n * n + 5 * r + r * n)

    def scratch(self, tier: str) -> int:
        return max(self.variant(self.variant_of(tier)).scratch, 1)

    def available(self, tier: str) -> bool:
        if tier != "block" and self.n_x > _LANE_TIERS_MAX_VARS:
            return False
        budget = _THREAD_BUDGET if tier == "thread" else _SHARED_BUDGET
        return self.slot_bytes(tier) + 16 <= budget

    def target(self, tier: str) -> str:
        if tier not in self._targets:
            if not self.available(tier):
                raise NotImplementedError(
                    f"tier {tier!r}: this problem needs {self.slot_bytes(tier)} bytes of "
                    "per-problem state, over the tier's budget")
            self._targets[tier] = _build(self.variant(self.variant_of(tier)).header, tier)
        return self._targets[tier]

    def auto_tier(self, batch: int) -> str:
        if self.tier_table:
            nearest = min(self.tier_table, key=lambda b: abs(np.log2(b) - np.log2(max(batch, 1))))
            return self.tier_table[nearest]
        # Before calibration (measured, RTX A5000, Panda collision/IK): the thread tier runs the
        # folded straight-line code and wins once there are enough problems to fill the GPU;
        # below that, a warp per problem (one Jacobian column per lane) cuts latency. The
        # block tier is for more columns than a warp has lanes (high-DOF robots).
        if self.n_x > 32:
            prefs = ["block"]
        elif batch >= 512:
            prefs = ["thread", "warp", "block"]
        else:
            prefs = ["warp", "thread", "block"]
        return next(t for t in prefs if self.available(t))

    def _signed(self, ev: E.Evaluator, v, row_of):
        """Negate the rows of >= constraints (they are solved as <= 0)."""
        if np.all(self.row_sign > 0):
            return v
        g = ev.g
        if isinstance(v, E.Lazy):
            sgn = g.table(self.row_sign, "f")
            fn = v.fn
            return E.Lazy(v.shape, v.kind, lambda i: g.op("mul", (fn(i), g.tload(sgn, row_of(g, i))), "f"),
                          v.cost + 1)
        out = np.empty(v.shape, dtype=object)
        for k in range(v.size):
            out.flat[k] = g.op("neg", (v.flat[k],), "f") if self.row_sign[row_of(None, k)] < 0 else v.flat[k]
        return out

    def _generate(self, loop_min: float) -> _Variant:
        g = E.Graph()
        ev = E.Evaluator(g, loop_min)
        J, r = ev.run(self.closed, [E.symbolic(g, "x", (self.n_x,)), E.symbolic(g, "p", (self.n_p,))])
        n = self.n_x
        r = self._signed(ev, r, lambda g_, i: i)
        J = self._signed(ev, J, lambda g_, i: g_.idiv(i, n) if g_ is not None else i // n)
        res_body, s1, e1 = g.emit_function([("r", r)])
        scalar = not isinstance(J, E.Lazy) and not isinstance(r, E.Lazy)
        # The scalar form uses normal_equations() below; residual_jacobian() would only bloat it.
        jac_body, s2, e2 = ("", 0, 1) if scalar else g.emit_function([("r", r), ("J", J)])
        scratch = max(s1, s2)
        kinds = self.row_kinds
        ne_src = ""
        if scalar:
            # Fully scalar (the thread tier's form): build H = sum w J J^T and g = sum w (r + s) J
            # symbolically, so zero Jacobian entries fold away and everything stays in registers.
            ne_body = self._normal_equations(g, r, J, kinds)
            ne_src = f"""
#define PYROFFI_COST_NORMAL_EQUATIONS 1
__device__ __forceinline__ void normal_equations(
    const float* __restrict__ x, const float* __restrict__ p, const float* __restrict__ w,
    const float* __restrict__ s, float* __restrict__ r, float* __restrict__ H,
    float* __restrict__ g, float* __restrict__ scratch, const int rank, const int size)
{{
    (void)p; (void)scratch; (void)size;
{ne_body}
}}
"""
        has_con = any(k != 0 for k in kinds)
        header = f"""// Generated by pyroffi (LeastSquaresProblem.analyze). Do not edit.
#pragma once
#include <cmath>

namespace pyroffi::costs_gen {{

constexpr int n_x = {n};
constexpr int n_r = {self.n_r};
constexpr int n_p = {self.n_p};
constexpr int scratch_size = {scratch};
constexpr bool has_constraints = {"true" if has_con else "false"};

__device__ __forceinline__ int row_kind(int i) {{
    constexpr int k[n_r] = {{{", ".join(map(str, kinds))}}};
    return k[i];
}}

{g.tables_source()}

__device__ __forceinline__ void residual(
    const float* __restrict__ x, const float* __restrict__ p, float* __restrict__ r,
    float* __restrict__ scratch, const int rank, const int size)
{{
    (void)p; (void)scratch; (void)size;
{res_body}
}}

__device__ __forceinline__ void residual_jacobian(
    const float* __restrict__ x, const float* __restrict__ p, float* __restrict__ r,
    float* __restrict__ J, float* __restrict__ scratch, const int rank, const int size)
{{
    (void)p; (void)scratch; (void)size;
{jac_body}
}}

{ne_src}
}}  // namespace pyroffi::costs_gen
"""
        return _Variant(header, scratch, max(e1, e2))

    def _generate_jvp(self) -> _Variant:
        g = E.Graph()
        ev = E.Evaluator(g, float("inf"))
        r, jt = ev.run(self.closed_jvp, [E.symbolic(g, "x", (self.n_x,)),
                                         E.symbolic(g, "p", (self.n_p,)),
                                         E.symbolic(g, "t", (self.n_x,))])
        r = self._signed(ev, r, lambda g_, i: i)
        jt = self._signed(ev, jt, lambda g_, i: i)
        res_body, _, _ = g.emit_function([("r", r)], sync=False)
        jvp_body, _, _ = g.emit_function([("r", r), ("jt", jt)], sync=False)
        kinds = self.row_kinds
        sparse_ne = self._sparse_normal_equations_tables(g, r)
        header = f"""// Generated by pyroffi (LeastSquaresProblem.analyze). Do not edit.
#pragma once
#include <cmath>

namespace pyroffi::costs_gen {{

constexpr int n_x = {self.n_x};
constexpr int n_r = {self.n_r};
constexpr int n_p = {self.n_p};
constexpr int scratch_size = 0;
constexpr bool has_constraints = {"true" if any(k != 0 for k in kinds) else "false"};
#define PYROFFI_COST_JVP 1

__device__ __forceinline__ int row_kind(int i) {{
    constexpr int k[n_r] = {{{", ".join(map(str, kinds))}}};
    return k[i];
}}

{g.tables_source()}
{sparse_ne}
// Lane-private straight-line code: one lane, no syncs.
__device__ __forceinline__ void residual(
    const float* __restrict__ x, const float* __restrict__ p, float* __restrict__ r,
    float* __restrict__ scratch, const int rank, const int size)
{{
    (void)p; (void)scratch; (void)size;
{res_body}
}}

// The residual and its directional derivative along t (one Jacobian column for t = e_j).
__device__ __forceinline__ void residual_jvp(
    const float* __restrict__ x, const float* __restrict__ p, const float* __restrict__ t,
    float* __restrict__ r, float* __restrict__ jt)
{{
    (void)p; const int rank = 0;
{jvp_body}
}}

}}  // namespace pyroffi::costs_gen
"""
        return _Variant(header, 0, self.n_x)

    def _sparse_normal_equations_tables(self, g: E.Graph, r) -> str:
        """Tables for assembling H = J^T W J and g = J^T W (r + s) from the Jacobian's
        structural sparsity (which variables each residual row can depend on, read off the
        compiled residual graph, so exact: a row inactive at some point is still counted).

        For each H entry (a <= c, every diagonal included) the rows touching both columns,
        and for each column the rows touching it. Rows such as joint limits or rest poses
        touch one variable each, so this is far less than the dense n_r * n_x^2 sum.
        """
        dep: dict[int, int] = {}
        for n in g.nodes:
            if n.op == "x":
                dep[n.idx] = 1 << int(n.args[0])
            elif n.op in ("p", "t", "w", "s", "idx", "table"):
                dep[n.idx] = 0
            else:
                m = 0
                for arg in n.args:
                    if isinstance(arg, E.Sym):
                        m |= dep[arg.idx]
                dep[n.idx] = m
        n = self.n_x
        masks = [dep[v.idx] if isinstance(v, E.Sym) else 0 for v in np.asarray(r).flat]
        rows_of = [[i for i, m in enumerate(masks) if (m >> a) & 1] for a in range(n)]
        h_a, h_c, h_ptr, h_rows = [], [], [0], []
        for a in range(n):
            for c in range(a, n):
                rows = [i for i in rows_of[a] if (masks[i] >> c) & 1]
                if rows or a == c:
                    h_a.append(a)
                    h_c.append(c)
                    h_rows += rows
                    h_ptr.append(len(h_rows))
        g_ptr, g_rows = [0], []
        for a in range(n):
            g_rows += rows_of[a]
            g_ptr.append(len(g_rows))

        def arr(name, vals):
            return f"__device__ const int {name}[] = {{{', '.join(map(str, vals)) or '0'}}};"

        return "\n".join([
            "#define PYROFFI_COST_SPARSE_NE 1",
            f"constexpr int h_nnz = {len(h_a)};  // H entries a <= c with any row (dense: {n * (n + 1) // 2})",
            f"constexpr int h_terms = {len(h_rows)};  // row contributions (dense: {self.n_r * n * (n + 1) // 2})",
            arr("h_a", h_a), arr("h_c", h_c), arr("h_ptr", h_ptr), arr("h_rows", h_rows),
            arr("g_ptr", g_ptr), arr("g_rows", g_rows), ""])

    def _normal_equations(self, g: E.Graph, r, J, kinds) -> str:
        n, n_r = self.n_x, self.n_r
        r, J = list(r.flat), np.asarray(J).reshape(n_r, n)
        wJ, res = [], []
        for i in range(n_r):
            if kinds[i] == 0:  # cost rows: w = 1, s = 0
                wJ.append(list(J[i]))
                res.append(r[i])
            else:
                w = g.input("w", i)
                wJ.append([g.op("mul", (w, v), "f") for v in J[i]])
                res.append(g.op("add", (r[i], g.input("s", i)), "f"))
        H = [[np.float32(0)] * n for _ in range(n)]
        grad = [np.float32(0)] * n
        for a in range(n):
            for i in range(n_r):
                grad[a] = g.op("add", (grad[a], g.op("mul", (wJ[i][a], res[i]), "f")), "f")
            for b in range(a, n):
                acc = np.float32(0)
                for i in range(n_r):
                    acc = g.op("add", (acc, g.op("mul", (wJ[i][a], J[i][b]), "f")), "f")
                H[a][b] = H[b][a] = acc
        H_arr = np.empty((n * n,), dtype=object)
        for a in range(n):
            for b in range(n):
                H_arr[a * n + b] = H[a][b]
        g_arr = np.empty((n,), dtype=object)
        g_arr[:] = grad
        r_arr = np.empty((n_r,), dtype=object)
        r_arr[:] = r
        body, _, _ = g.emit_function([("r", r_arr), ("H", H_arr), ("g", g_arr)])
        return body


# ponytail: unbounded; one entry per distinct problem structure in the process.
_KERNELS: dict[tuple, _Kernel] = {}


def analyze(problem: LeastSquaresProblem) -> "AnalyzedLeastSquaresProblem":
    _require_cuda()
    plans = [_plan(c) for c in problem.costs]
    variables = [_concrete_var(v) for v in problem.variables]
    key = (tuple(p.key() for p in plans),
           tuple((type(v), np.asarray(v.id).tobytes(), np.shape(v.id)) for v in variables))
    kernel = _KERNELS.get(key)
    if kernel is None:
        with jax.ensure_compile_time_eval():
            kernel = _KERNELS[key] = _Kernel(variables, plans)
    return AnalyzedLeastSquaresProblem(kernel, plans)


class AnalyzedLeastSquaresProblem:
    """A problem compiled to a CUDA kernel. Create with ``problem.analyze()``."""

    def __init__(self, kernel: _Kernel, plans: list[_CostPlan]):
        self._kernel = kernel
        self._plans = plans

    @property
    def n_x(self) -> int:
        """Total variable dimension."""
        return self._kernel.n_x

    @property
    def n_p(self) -> int:
        """Length of the runtime parameter vector."""
        return self._kernel.n_p

    def params(self) -> jax.Array:
        """This problem's runtime parameter vector ``(n_p,)``; stack several for
        :meth:`solve_batch`."""
        leaves = [p for plan in self._plans for p in plan.params]
        if not leaves:
            return jnp.zeros((0,), jnp.float32)
        return jnp.concatenate([jnp.ravel(a).astype(jnp.float32) for a in leaves])

    def initial_x(self, initial_vals: VarValues | None = None) -> jax.Array:
        """Flat initial variables ``(n_x,)``: ``initial_vals`` or each variable's default."""
        parts = []
        for vtype, vid in self._kernel.slots:
            v = initial_vals[vtype(vid)] if initial_vals is not None else vtype.default_factory()
            parts.append(jnp.ravel(jnp.asarray(v, jnp.float32)))
        return jnp.concatenate(parts)

    def unflatten(self, x: jax.Array) -> VarValues:
        """Flat ``(n_x,)`` variables to :class:`VarValues`."""
        return VarValues({key: x[off:off + size].reshape(shape)
                          for key, (off, size, shape) in self._kernel.layout.items()})

    def solve_batch(self, params: jax.Array, x0: jax.Array, *, tier: str = "auto", **settings):
        """Solve ``B`` problems of this structure in one kernel launch.

        Args:
            params: ``(B, n_p)``, or ``(n_p,)`` shared by every problem.
            x0: ``(B, n_x)`` initial variables.
            tier: ``"thread"`` (one problem per thread), ``"warp"``, ``"block"``, or
                ``"auto"`` (the :meth:`calibrate` table, else a batch-size heuristic).
            **settings: ``max_iters``, ``al_iters``, ``lambda_initial``, ``tolerance``
                (relative cost change), ``al_tolerance``, ``grad_tolerance``, ``grad_start``,
                ``param_tolerance`` (jaxls's termination defaults).

        Returns:
            ``(x, cost, violation)``: ``(B, n_x)`` solutions, ``(B,)`` sum of squared cost
            residuals, ``(B,)`` largest constraint violation.
        """
        k = self._kernel
        x0 = jnp.asarray(x0, jnp.float32)
        B = x0.shape[0]
        if k.n_p == 0:  # the kernel still takes a (1, 1) buffer
            params = jnp.zeros((1, 1), jnp.float32)
        else:
            params = jnp.asarray(params, jnp.float32).reshape(-1, k.n_p)
        s = {**_DEFAULTS, **settings}
        if tier == "auto":
            tier = k.auto_tier(B)
        elif tier not in TIERS:
            raise ValueError(f"tier must be 'auto' or one of {TIERS}, got {tier!r}")
        call = jax.ffi.ffi_call(k.target(tier), (
            jax.ShapeDtypeStruct((B, k.n_x), jnp.float32),
            jax.ShapeDtypeStruct((B,), jnp.float32),
            jax.ShapeDtypeStruct((B,), jnp.float32),
            jax.ShapeDtypeStruct((B, k.scratch(tier)), jnp.float32)))
        x, cost, viol, _ = call(x0, params, max_iters=np.int32(s["max_iters"]),
                    al_iters=np.int32(s["al_iters"]),
                    lambda_initial=np.float32(s["lambda_initial"]),
                    tolerance=np.float32(s["tolerance"]),
                    al_tolerance=np.float32(s["al_tolerance"]),
                    grad_tolerance=np.float32(s["grad_tolerance"]),
                    grad_start=np.int32(s["grad_start"]),
                    param_tolerance=np.float32(s["param_tolerance"]))
        return x, cost, viol

    def solve(self, initial_vals: VarValues | None = None, *, linear_solver: str = "dense_cholesky",
              trust_region: TrustRegionConfig | None = TrustRegionConfig(),
              termination: TerminationConfig = TerminationConfig(), sparse_mode: str = "blockrow",
              verbose: bool = True, augmented_lagrangian: AugmentedLagrangianConfig | None = None,
              return_summary: bool = False, tier: str = "auto"):
        """Solve, with jaxls's signature. Returns :class:`VarValues` (and a
        :class:`SolveSummary` with ``return_summary=True``).

        The kernel reads ``trust_region.lambda_initial``; ``termination.max_iterations``,
        ``cost_tolerance``, ``gradient_tolerance``, ``gradient_tolerance_start_step`` and
        ``parameter_tolerance``; and ``augmented_lagrangian.tolerance_absolute`` and
        ``max_iterations``. The other options are accepted for compatibility.
        """
        del linear_solver, sparse_mode, verbose
        s = dict(max_iters=termination.max_iterations, tolerance=float(termination.cost_tolerance),
                 grad_tolerance=float(termination.gradient_tolerance),
                 grad_start=int(termination.gradient_tolerance_start_step),
                 param_tolerance=float(termination.parameter_tolerance))
        if trust_region is not None:
            s["lambda_initial"] = float(trust_region.lambda_initial)
        if augmented_lagrangian is not None:
            s["al_tolerance"] = float(augmented_lagrangian.tolerance_absolute)
            s["al_iters"] = augmented_lagrangian.max_iterations
        x, cost, viol = self.solve_batch(self.params()[None], self.initial_x(initial_vals)[None],
                                         tier=tier, **s)
        vals = self.unflatten(x[0])
        return (vals, SolveSummary(cost[0], viol[0])) if return_summary else vals

    def calibrate(self, batch_sizes=(1, 16, 128, 1024, 8192), tiers=TIERS, repeats: int = 3,
                  **settings) -> dict[int, dict[str, float]]:
        """Time each tier at each batch size and make ``tier="auto"`` pick the fastest.

        The best tier depends on the problem's size *and* the batch jointly (a thread per
        problem wins at large batch, a warp or block when problems are few), so it is
        measured rather than assumed. Compiles every tier it times. Returns
        ``{batch: {tier: ms}}``; the choice is kept for this problem structure.
        """
        import time

        k = self._kernel
        results: dict[int, dict[str, float]] = {}
        for B in batch_sizes:
            params = jnp.broadcast_to(self.params(), (B, k.n_p))
            x0 = jnp.broadcast_to(self.initial_x(), (B, k.n_x))
            results[B] = {}
            for tier in tiers:
                if not k.available(tier):
                    continue
                run = jax.jit(lambda P, X, tier=tier: self.solve_batch(P, X, tier=tier, **settings)[0])
                jax.block_until_ready(run(params, x0))
                t = time.perf_counter()
                for _ in range(repeats):
                    jax.block_until_ready(run(params, x0))
                results[B][tier] = (time.perf_counter() - t) / repeats * 1e3
            k.tier_table[B] = min(results[B], key=results[B].get)
        return results
