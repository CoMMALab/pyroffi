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
_MAX_VARS = 32
_DEFAULTS = dict(max_iters=100, al_iters=20, lambda_initial=5e-4, tolerance=1e-5,
                 al_tolerance=1e-5)


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


@lru_cache(maxsize=None)
def _build(header: str) -> str:
    from ..cuda_kernels._traced import (_GLASS_DIR, _KERNELS_DIR, build_shared_library,
                                        register_cuda_targets)

    source = _KERNELS_DIR / "costs" / "_cost_lm_kernel.cu"
    flags = ["-O3", "-std=c++17", os.environ.get("PYROFFI_TRACED_GPU_ARCH", "-arch=native"),
             "--shared", "--compiler-options", "-fPIC"]
    keys = (source, *sorted(_KERNELS_DIR.glob("*.cuh")), *sorted(_GLASS_DIR.glob("*.cuh")),
            *sorted((_GLASS_DIR / "src").rglob("*.cuh")))
    try:
        so = build_shared_library("cost_lm", source, {"_cost_gen.cuh": header}, flags,
                                  include_dirs=(_KERNELS_DIR, _GLASS_DIR), key_files=keys)
    except FileNotFoundError as e:
        if shutil.which("nvcc") is None:
            raise RuntimeError("pk.LeastSquaresProblem.analyze() compiles to CUDA and needs "
                               "nvcc on PATH; it was not found.") from e
        raise
    tag = hashlib.sha1(header.encode()).hexdigest()[:16]
    return register_cuda_targets(so, ("CostLmFfi",), tag)[0]


class _Kernel:
    """A compiled problem structure: generated code, FFI target and variable layout."""

    def __init__(self, variables: list[Var], plans: list[_CostPlan]):
        self.layout, self.slots, self.n_x = _variable_layout(variables)
        if self.n_x > _MAX_VARS:
            raise NotImplementedError(
                f"{self.n_x} variables: the compiled dense solve supports up to {_MAX_VARS} "
                "(larger problems, e.g. trajectories, are not implemented yet)")
        self.plans = plans
        self.param_specs = [[(tuple(p.shape), p.dtype) for p in plan.params] for plan in plans]
        self.n_p = sum(int(np.prod(s, dtype=int)) for specs in self.param_specs for s, _ in specs)
        self.header, self.row_kinds = self._generate()
        self.plans = None  # only needed to trace; may hold the caller's tracers
        self.target = _build(self.header)

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

    def _generate(self) -> tuple[str, list[int]]:
        x0 = np.zeros(self.n_x, np.float32)
        p0 = np.zeros(self.n_p, np.float32)

        def with_jac(x, p):
            residual = lambda x: jnp.concatenate(self._residual_rows(x, p))
            return jax.jacfwd(lambda x: (residual(x),) * 2, has_aux=True)(x)

        closed = jax.make_jaxpr(with_jac)(x0, p0)
        g = E.Graph()
        J, r = E.Evaluator(g).run(closed, [E.symbolic(g, "x", (self.n_x,)),
                                          E.symbolic(g, "p", (self.n_p,))])
        kinds, sign = [], []
        for plan, rows in zip(self.plans, jax.eval_shape(self._residual_rows, x0, p0)):
            kinds += [_ROW_KIND[plan.cost.kind]] * rows.shape[0]
            sign += [-1 if plan.cost.kind == "constraint_geq_zero" else 1] * rows.shape[0]
        n_r = len(kinds)
        assert n_r == r.shape[0], (n_r, r.shape)
        r = [g.op("neg", (v,), "f") if s < 0 else v for v, s in zip(r.flat, sign)]
        J = [[g.op("neg", (v,), "f") if sign[i] < 0 else v for v in J[i]] for i in range(n_r)]

        # H = sum_i w_i J_i J_i^T,  grad = sum_i w_i (r_i + s_i) J_i; cost rows have w = 1, s = 0.
        n = self.n_x
        wJ, res = [], []
        for i in range(n_r):
            if kinds[i] == 0:
                wJ.append(J[i])
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

        res_body = g.emit([(f"r[{i}]", v) for i, v in enumerate(r)])
        ne_body = g.emit([(f"r[{i}]", v) for i, v in enumerate(r)]
                         + [(f"H[{a * n + b}]", H[a][b]) for a in range(n) for b in range(n)]
                         + [(f"g[{a}]", grad[a]) for a in range(n)])
        has_con = any(k != 0 for k in kinds)
        header = f"""// Generated by pyroffi (LeastSquaresProblem.analyze). Do not edit.
#pragma once
#include <cmath>

namespace pyroffi::costs_gen {{

constexpr int n_x = {n};
constexpr int n_r = {n_r};
constexpr int n_p = {self.n_p};
constexpr bool has_constraints = {"true" if has_con else "false"};

__device__ __forceinline__ int row_kind(int i) {{
    constexpr int k[n_r] = {{{", ".join(map(str, kinds))}}};
    return k[i];
}}

__device__ __forceinline__ void residual(
    const float* __restrict__ x, const float* __restrict__ p, float* __restrict__ r)
{{
{res_body}
}}

__device__ __forceinline__ void normal_equations(
    const float* __restrict__ x, const float* __restrict__ p, const float* __restrict__ w,
    const float* __restrict__ s, float* __restrict__ r, float* __restrict__ H,
    float* __restrict__ g)
{{
{ne_body}
}}

}}  // namespace pyroffi::costs_gen
"""
        return header, kinds


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

    def solve_batch(self, params: jax.Array, x0: jax.Array, **settings):
        """Solve ``B`` problems of this structure in one kernel launch.

        Args:
            params: ``(B, n_p)``, or ``(n_p,)`` shared by every problem.
            x0: ``(B, n_x)`` initial variables.
            **settings: ``max_iters``, ``al_iters``, ``lambda_initial``, ``tolerance``,
                ``al_tolerance``.

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
        call = jax.ffi.ffi_call(k.target, (
            jax.ShapeDtypeStruct((B, k.n_x), jnp.float32),
            jax.ShapeDtypeStruct((B,), jnp.float32),
            jax.ShapeDtypeStruct((B,), jnp.float32)))
        return call(x0, params, max_iters=np.int32(s["max_iters"]),
                    al_iters=np.int32(s["al_iters"]),
                    lambda_initial=np.float32(s["lambda_initial"]),
                    tolerance=np.float32(s["tolerance"]),
                    al_tolerance=np.float32(s["al_tolerance"]))

    def solve(self, initial_vals: VarValues | None = None, *, linear_solver: str = "dense_cholesky",
              trust_region: TrustRegionConfig | None = TrustRegionConfig(),
              termination: TerminationConfig = TerminationConfig(), sparse_mode: str = "blockrow",
              verbose: bool = True, augmented_lagrangian: AugmentedLagrangianConfig | None = None,
              return_summary: bool = False):
        """Solve, with jaxls's signature. Returns :class:`VarValues` (and a
        :class:`SolveSummary` with ``return_summary=True``).

        The kernel reads ``trust_region.lambda_initial``, ``termination.max_iterations`` and
        ``cost_tolerance``, and ``augmented_lagrangian.tolerance_absolute`` and
        ``max_iterations``; the other options are accepted for compatibility.
        """
        del linear_solver, sparse_mode, verbose
        s = dict(max_iters=termination.max_iterations, tolerance=float(termination.cost_tolerance))
        if trust_region is not None:
            s["lambda_initial"] = float(trust_region.lambda_initial)
        if augmented_lagrangian is not None:
            s["al_tolerance"] = float(augmented_lagrangian.tolerance_absolute)
            s["al_iters"] = augmented_lagrangian.max_iterations
        x, cost, viol = self.solve_batch(self.params()[None], self.initial_x(initial_vals)[None], **s)
        vals = self.unflatten(x[0])
        return (vals, SolveSummary(cost[0], viol[0])) if return_summary else vals
