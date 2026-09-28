"""Compile a ``jaxls.LeastSquaresProblem`` into one CUDA kernel.

The problem's costs are traced once, together with their forward-mode Jacobian, and
emitted as straight-line code (see :mod:`._emit`) into ``_cost_gen.cuh``, which a
Levenberg-Marquardt / augmented-Lagrangian kernel includes. A solve is then a single
FFI call: no JAX runs inside the optimisation loop.

What is baked into the build, and what stays a runtime input:

* **Baked** (a change recompiles): the problem's structure (which factories, which
  variables each cost touches, residual sizes), robot and robot-collision models,
  NumPy arrays and Python scalars (weights such as ``pos_weight=50.0``, margins).
* **Runtime** (changing it is free): every ``jax.Array`` argument, such as target poses,
  rest poses, obstacle geometry, or a traced link index.

Build the problem from concrete values once and :func:`compile` it; afterwards
:meth:`CompiledProblem.solve` accepts any problem of the same structure, including one
built inside ``jax.jit`` from traced targets.

Scope: Euclidean variables (e.g. ``robot.joint_var_cls``), at most 32 scalars in total
(the per-thread dense solve). Lie-group variables and larger problems stay on jaxls.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import jax
import jax.numpy as jnp
import jaxls
import numpy as np

from . import _emit as E

_ROW_KIND = {"l2_squared": 0, "constraint_eq_zero": 1, "constraint_leq_zero": 2,
             "constraint_geq_zero": 2}
_MAX_VARS = 32


def _baked_types() -> tuple[type, ...]:
    from .._robot import Robot
    from ..collision import RobotCollision, RobotCollisionSpherized

    return (Robot, RobotCollision, RobotCollisionSpherized)


def _bake_leaf(x) -> np.ndarray:
    if isinstance(x, jax.core.Tracer):
        raise ValueError(
            "pyroffi.costs.compile needs the robot, collision model and Python/NumPy "
            "arguments concrete: build the problem outside jax.jit before compiling "
            "(CompiledProblem.solve accepts traced problems afterwards).")
    a = np.asarray(x)
    if a.dtype == np.float64:
        a = a.astype(np.float32)
    elif a.dtype == np.int64:
        a = a.astype(np.int32)
    return a


@dataclass
class _CostPlan:
    """One cost (possibly batched) split into variables, baked values and runtime params."""

    cost: jaxls.Cost
    treedef: Any
    kinds: list[str]           # per top-level item: "var" | "bake" | "param"
    items: list[Any]           # broadcast to the batch: Var / baked pytree / None for params
    param_shapes: list[tuple]  # broadcast shape of each param item, in order
    param_dtypes: list[Any]
    batch: int | None          # None: unbatched


def _is_item(x) -> bool:
    return isinstance(x, (jaxls.Var, *_baked_types()))


def _split(cost: jaxls.Cost):
    return jax.tree.flatten(cost.args, is_leaf=_is_item)


def _batch_axes(items) -> tuple[int, ...]:
    ids = [() if isinstance(v.id, int) else jnp.shape(v.id) for v in items if isinstance(v, jaxls.Var)]
    if not ids:
        raise ValueError("every cost needs at least one jaxls.Var argument")
    axes = jnp.broadcast_shapes(*ids)
    if len(axes) > 1:
        raise NotImplementedError("costs with more than one batch axis")
    return axes


def _broadcast(x, axes, xp):
    x = xp.asarray(x)
    return xp.broadcast_to(x, axes + x.shape[len(axes):]) if axes else x


def _plan(cost: jaxls.Cost) -> _CostPlan:
    items, treedef = _split(cost)
    axes = _batch_axes(items)
    kinds, out, shapes, dtypes = [], [], [], []
    for it in items:
        if isinstance(it, jaxls.Var):
            kinds.append("var")
            out.append(type(it)(np.broadcast_to(np.asarray(it.id), axes)) if axes else it)
        elif isinstance(it, _baked_types()):
            kinds.append("bake")
            out.append(jax.tree.map(lambda a: _broadcast(_bake_leaf(a), axes, np), it))
        elif isinstance(it, jax.Array):
            kinds.append("param")
            out.append(None)
            shapes.append(tuple(axes) + tuple(jnp.shape(it)[len(axes):]))
            dtypes.append(jnp.result_type(it))
        else:
            kinds.append("bake")
            out.append(None if it is None else _broadcast(_bake_leaf(it), axes, np))
    return _CostPlan(cost, treedef, kinds, out, shapes, dtypes, axes[0] if axes else None)


def _same_baked(new, old, axes) -> bool:
    """Whether a non-param item of a new problem matches what the build baked in.

    Variables must match in type and ids. Robot/collision models must match in pytree
    structure (which includes their static fields, so a different robot fails) and, where
    their arrays are concrete, in value; traced arrays (a robot passed into jax.jit) cannot be
    compared and are taken as the compiled ones.
    """
    if isinstance(old, jaxls.Var):
        return type(new) is type(old) and np.array_equal(
            np.broadcast_to(np.asarray(new.id), np.shape(old.id)), np.asarray(old.id))
    if old is None or new is None:
        return old is None and new is None
    if jax.tree.structure(new) != jax.tree.structure(old):
        return False
    for a, b in zip(jax.tree.leaves(new), jax.tree.leaves(old)):
        if isinstance(a, jax.core.Tracer):
            continue
        try:
            if not np.array_equal(_broadcast(_bake_leaf(a), axes, np), b):
                return False
        except ValueError:  # not broadcastable to the compiled shape
            return False
    return True


def _param_leaves(plan: _CostPlan, cost: jaxls.Cost) -> list[jax.Array]:
    """The runtime-param items of ``cost`` (same structure as ``plan``), broadcast."""
    items, treedef = _split(cost)
    # Factories wrap each call in a fresh lambda, so compare names and kinds, not functions.
    if (treedef != plan.treedef or cost._get_name() != plan.cost._get_name()
            or cost.kind != plan.cost.kind):
        raise ValueError(f"cost {plan.cost._get_name()!r} does not match the compiled structure")
    axes = (plan.batch,) if plan.batch is not None else ()
    for it, kind, old in zip(items, plan.kinds, plan.items):
        if kind != "param" and not _same_baked(it, old, axes):
            raise ValueError(
                f"cost {plan.cost._get_name()!r}: a variable, robot/collision model or baked "
                "value differs from the compiled problem; compile this problem separately")
    out = [_broadcast(it, axes, jnp) for it, k in zip(items, plan.kinds) if k == "param"]
    for a, shape in zip(out, plan.param_shapes):
        if a.shape != shape:
            raise ValueError(f"parameter shape {a.shape} differs from the compiled {shape}")
    return out


class _Vals:
    """Stands in for ``jaxls.VarValues`` while tracing: variables are slices of ``x``."""

    def __init__(self, x, layout):
        self._x, self._layout = x, layout

    def get_value(self, var):
        vid = np.asarray(var.id)
        if vid.ndim > 0:
            return jnp.stack([self.get_value(type(var)(int(i))) for i in vid.ravel()]).reshape(
                vid.shape + self._layout[(type(var), int(vid.ravel()[0]))][2])
        off, size, shape = self._layout[(type(var), int(vid))]
        return self._x[off:off + size].reshape(shape)

    __getitem__ = get_value


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
        if vtype.retract_fn is not jaxls.Var._euclidean_retract:
            raise NotImplementedError(
                f"{vtype.__name__}: only Euclidean variables compile (Lie-group variables stay on jaxls)")
        default = np.asarray(vtype.default_factory(), np.float32)
        for vid in np.atleast_1d(np.asarray(var.id)).ravel():
            key = (vtype, int(vid))
            if key in layout:
                continue
            layout[key] = (n, default.size, default.shape)
            slots.append(key)
            n += default.size
    return layout, slots, n


@lru_cache(maxsize=None)
def _build(header: str) -> str:
    from ..cuda_kernels._traced import _GLASS_DIR, _KERNELS_DIR, build_shared_library, register_cuda_targets

    source = _KERNELS_DIR / "costs" / "_cost_lm_kernel.cu"
    flags = ["-O3", "-std=c++17", os.environ.get("PYROFFI_TRACED_GPU_ARCH", "-arch=native"),
             "--shared", "--compiler-options", "-fPIC"]
    keys = (source, *sorted(_KERNELS_DIR.glob("*.cuh")), *sorted(_GLASS_DIR.glob("*.cuh")),
            *sorted((_GLASS_DIR / "src").rglob("*.cuh")))
    so = build_shared_library("cost_lm", source, {"_cost_gen.cuh": header}, flags,
                              include_dirs=(_KERNELS_DIR, _GLASS_DIR), key_files=keys)
    tag = hashlib.sha1(header.encode()).hexdigest()[:16]
    return register_cuda_targets(so, ("CostLmFfi",), tag)[0]


class CompiledProblem:
    """A least-squares problem compiled to a CUDA kernel. Create with :func:`compile`."""

    def __init__(self, problem: jaxls.LeastSquaresProblem, *, max_iterations: int = 100,
                 al_iterations: int = 20, lambda_initial: float = 5e-4,
                 tolerance: float = 1e-5, al_tolerance: float = 1e-5):
        self._problem = problem
        self.settings = dict(max_iters=max_iterations, al_iters=al_iterations,
                             lambda_initial=lambda_initial, tolerance=tolerance,
                             al_tolerance=al_tolerance)
        costs = list(problem.costs)
        self._variables = list(problem.variables)
        self._layout, self._slots, self.n_x = _variable_layout(self._variables)
        if self.n_x > _MAX_VARS:
            raise NotImplementedError(
                f"{self.n_x} variables: the compiled dense solve supports up to {_MAX_VARS}")
        self._plans = [_plan(c) for c in costs]
        p0 = self.params(problem)
        self.n_p = int(p0.shape[-1])
        self.header, self.row_kinds = self._generate(p0)
        self._target = _build(self.header)

    # ── inputs ──────────────────────────────────────────────────────────────
    def params(self, problem: jaxls.LeastSquaresProblem | None = None) -> jax.Array:
        """The runtime parameter vector ``(n_p,)`` of ``problem`` (default: the compiled one).

        Stack several with ``jnp.stack`` to solve a batch of problems in one call.
        """
        costs = list((problem or self._problem).costs)
        if len(costs) != len(self._plans):
            raise ValueError("problem has a different number of costs than the compiled one")
        leaves = [a for plan, c in zip(self._plans, costs) for a in _param_leaves(plan, c)]
        if not leaves:
            return jnp.zeros((0,), jnp.float32)
        return jnp.concatenate([jnp.ravel(a).astype(jnp.float32) for a in leaves])

    def initial_x(self, initial_vals: jaxls.VarValues | None = None) -> jax.Array:
        """Flat initial variables ``(n_x,)``: ``initial_vals`` or each variable's default."""
        parts = []
        for vtype, vid in self._slots:
            v = initial_vals[vtype(vid)] if initial_vals is not None else vtype.default_factory()
            parts.append(jnp.ravel(jnp.asarray(v, jnp.float32)))
        return jnp.concatenate(parts)

    # ── solving ─────────────────────────────────────────────────────────────
    def solve_batch(self, params: jax.Array, x0: jax.Array, **settings):
        """Solve ``B`` problems in one kernel launch.

        Args:
            params: ``(B, n_p)`` or ``(n_p,)`` (shared by every problem).
            x0: ``(B, n_x)`` initial variables.

        Returns:
            ``(x, cost, violation)``: ``(B, n_x)`` solutions, ``(B,)`` sum of squared cost
            residuals, ``(B,)`` largest constraint violation.
        """
        x0 = jnp.asarray(x0, jnp.float32)
        B = x0.shape[0]
        if self.n_p == 0:  # the kernel still takes a (1, 1) buffer
            params = jnp.zeros((1, 1), jnp.float32)
        else:
            params = jnp.asarray(params, jnp.float32).reshape(-1, self.n_p)
        s = {**self.settings, **settings}
        call = jax.ffi.ffi_call(self._target, (
            jax.ShapeDtypeStruct((B, self.n_x), jnp.float32),
            jax.ShapeDtypeStruct((B,), jnp.float32),
            jax.ShapeDtypeStruct((B,), jnp.float32)))
        return call(x0, params, max_iters=np.int32(s["max_iters"]),
                    al_iters=np.int32(s["al_iters"]),
                    lambda_initial=np.float32(s["lambda_initial"]),
                    tolerance=np.float32(s["tolerance"]),
                    al_tolerance=np.float32(s["al_tolerance"]))

    def solve(self, problem: jaxls.LeastSquaresProblem | None = None,
              initial_vals: jaxls.VarValues | None = None, *,
              trust_region: jaxls.TrustRegionConfig | None = None,
              termination: jaxls.TerminationConfig | None = None,
              augmented_lagrangian: jaxls.AugmentedLagrangianConfig | None = None,
              verbose: bool = False, linear_solver: str | None = None) -> jaxls.VarValues:
        """Solve like ``problem.analyze().solve(...)`` and return a ``jaxls.VarValues``.

        ``problem`` defaults to the compiled one; any problem with the same structure works.
        The jaxls configs map onto the kernel's settings (``lambda_initial``,
        ``max_iterations``/``cost_tolerance``, ``tolerance_absolute``); ``verbose`` and
        ``linear_solver`` are accepted for signature compatibility.
        """
        del verbose, linear_solver
        s = {}
        if trust_region is not None:
            s["lambda_initial"] = float(trust_region.lambda_initial)
        if termination is not None:
            s["max_iters"] = termination.max_iterations
            s["tolerance"] = float(termination.cost_tolerance)
        if augmented_lagrangian is not None:
            s["al_tolerance"] = float(augmented_lagrangian.tolerance_absolute)
        x, _, _ = self.solve_batch(self.params(problem)[None], self.initial_x(initial_vals)[None], **s)
        return self.unflatten(x[0])

    def unflatten(self, x: jax.Array) -> jaxls.VarValues:
        """Flat ``(n_x,)`` variables to a ``jaxls.VarValues``."""
        return jaxls.VarValues.make([
            vtype(vid).with_value(x[off:off + size].reshape(shape))
            for (vtype, vid) in self._slots
            for off, size, shape in [self._layout[(vtype, vid)]]])

    # ── code generation ─────────────────────────────────────────────────────
    def _residual_rows(self, x, p) -> list[jax.Array]:
        """Each cost's residual rows (all batch instances), in problem order."""
        vals = _Vals(x, self._layout)
        per_cost, off = [], 0
        for plan in self._plans:
            params = []
            for shape, dtype in zip(plan.param_shapes, plan.param_dtypes):
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

    def _generate(self, p0: jax.Array) -> tuple[str, list[int]]:
        x0 = np.zeros(self.n_x, np.float32)
        p0 = np.asarray(p0, np.float32)

        def with_jac(x, p):
            residual = lambda x: jnp.concatenate(self._residual_rows(x, p))
            return jax.jacfwd(lambda x: (residual(x),) * 2, has_aux=True)(x)

        closed = jax.make_jaxpr(with_jac)(x0, p0)
        g = E.Graph()
        J, r = E.Evaluator(g).run(closed, [E.symbolic(g, "x", (self.n_x,)),
                                          E.symbolic(g, "p", (self.n_p,))])
        kinds, sign = [], []
        for plan, rows in zip(self._plans, jax.eval_shape(self._residual_rows, x0, p0)):
            rows = rows.shape[0]
            kinds += [_ROW_KIND[plan.cost.kind]] * rows
            sign += [-1 if plan.cost.kind == "constraint_geq_zero" else 1] * rows
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
        header = f"""// Generated by pyroffi.costs.compile. Do not edit.
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


def compile(problem: jaxls.LeastSquaresProblem, **settings) -> CompiledProblem:
    """Compile ``problem`` to a CUDA kernel (built once per structure, cached on disk).

    Args:
        problem: A ``jaxls.LeastSquaresProblem`` built from concrete values.
        **settings: ``max_iterations``, ``al_iterations``, ``lambda_initial``,
            ``tolerance``, ``al_tolerance`` (defaults follow jaxls).
    """
    return CompiledProblem(problem, **settings)
