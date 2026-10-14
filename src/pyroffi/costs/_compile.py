"""Compile a :class:`~pyroffi.LeastSquaresProblem` into one CUDA kernel.

``problem.analyze()`` traces the problem's residual once, together with its forward-mode
Jacobian, and emits it as straight-line code (see :mod:`._emit`) into ``cost_gen.cuh``,
which a Levenberg-Marquardt / augmented-Lagrangian kernel includes. ``solve()`` is then a
single FFI call: no JAX runs inside the optimisation loop, and there is no JAX fallback.

What is baked into the build, and what stays a runtime input:

* **Baked** (a change is a different build): the problem's structure (which factories,
  which variables each cost touches, residual sizes), robot and robot-collision models,
  NumPy arrays and Python scalars (weights such as ``pos_weight=50.0``, margins).
* **Runtime** (changing it reuses the build): every ``jax.Array`` argument, such as target
  poses, rest poses, obstacle geometry, or a traced link index.

Builds are cached by that split, in memory and on disk, so re-analyzing a problem that
differs only in runtime inputs costs a dictionary lookup, and a new process skips tracing,
code generation and nvcc for a structure it has seen: the generated headers live under
``~/.cache/pyroffi/costs`` (``$PYROFFI_COST_CACHE``; ``0`` disables), keyed by the problem's
structure, baked values, residual bytecode and pyroffi's source, and an entry is reused only
if the residual still evaluates to the probe stored with it. GRiM caches the ``.so`` by
header content. Inside ``jax.jit``
a traced robot or collision model is resolved to the concrete one it was built from
(see :mod:`pyroffi._registry`).

Scope: Euclidean variables (e.g. ``robot.joint_var_cls``) and Lie-group variables
(``pk.SE3Var`` / ``SO3Var`` / ``SE2Var`` / ``SO2Var``), at most 32 tangent dimensions at the
thread and warp tiers and 128 at the block tier. A Lie-group variable is stored by its group
parameters (7 floats for an SE3) and solved in its tangent space: the Jacobian is
``d/dδ r(retract(x, δ))`` at ``δ = 0`` and a step is ``x <- retract(x, dx)``, as in jaxls.
Anything else raises.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
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
        nat = None if self.cost.native is None else self.cost.native.digest
        return (self.cost._get_name(), self.cost.kind, self.treedef, tuple(parts), shapes, nat)


def _first(out):
    """A residual function's rows (some return ``(residual, jacobian cache)``)."""
    return out[0] if isinstance(out, tuple) else out


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


@dataclass(frozen=True)
class _Slot:
    """Where one variable lives: its parameters at ``q`` (ambient, e.g. 7 for an SE3) and its
    tangent coordinates at ``t``; ``treedef`` and ``shapes`` rebuild the value (a jaxlie group
    or an array) from the flat parameters."""

    q: int
    q_size: int
    t: int
    t_dim: int
    treedef: Any
    shapes: tuple

    def value(self, xq):
        leaves, off = [], self.q
        for shape in self.shapes:
            size = int(np.prod(shape, dtype=int))
            leaves.append(xq[off:off + size].reshape(shape))
            off += size
        return jax.tree.unflatten(self.treedef, leaves)


def _variable_layout(variables) -> tuple[dict, list, int, int]:
    """(type, id) -> :class:`_Slot`; plus the ordered keys, the parameter size ``n_q`` and the
    tangent size ``n_x`` (equal unless there are Lie-group variables)."""
    layout, slots, n_q, n_x = {}, [], 0, 0
    for var in variables:
        vtype = type(var)
        leaves, treedef = jax.tree.flatten(vtype.default_factory())
        shapes = tuple(np.shape(a) for a in leaves)
        q_size = sum(int(np.prod(sh, dtype=int)) for sh in shapes)
        t_dim = q_size if vtype.retract_fn is None else int(vtype.tangent_dim)
        for vid in np.atleast_1d(np.asarray(var.id)).ravel():
            key = (vtype, int(vid))
            if key in layout:
                continue
            layout[key] = _Slot(n_q, q_size, n_x, t_dim, treedef, shapes)
            slots.append(key)
            n_q += q_size
            n_x += t_dim
    return layout, slots, n_q, n_x


def _retract_fn(layout: dict, slots: list):
    """``retract(x, dx)``: apply a tangent step ``dx`` (n_x) to the parameters ``x`` (n_q)."""
    def retract(x, dx):
        parts = []
        for key in slots:
            sl, vtype = layout[key], key[0]
            d = dx[sl.t:sl.t + sl.t_dim]
            if vtype.retract_fn is None:
                parts.append(x[sl.q:sl.q + sl.q_size] + d)
            else:
                new = vtype.retract_fn(sl.value(x), d)
                parts += [jnp.ravel(a) for a in jax.tree.leaves(new)]
        return jnp.concatenate(parts)
    return retract


def _closure_tracers(layout: dict, slots: list, n_q: int, n_x: int) -> list:
    """Values of the caller's ``jax.jit`` trace that the variables' retractions close over
    (PyRoKi's mobile-base IK masks the base step by a traced ``fix_base``). They become
    runtime parameters, in this order; outside jit there are none."""
    closed = jax.make_jaxpr(_retract_fn(layout, slots))(np.zeros(n_q, np.float32),
                                                        np.zeros(n_x, np.float32))
    return [c for c in closed.consts if isinstance(c, jax.core.Tracer)]


def _lift_tracers(closed, hoisted: list) -> tuple[Any, list[int]]:
    """``closed`` with its constants that are ``hoisted`` tracers turned into trailing inputs;
    returns it and, per new input, the index into ``hoisted``."""
    if not hoisted:
        return closed, []
    jaxpr, keep_vars, keep_vals, lifted = closed.jaxpr, [], [], []
    for v, c in zip(jaxpr.constvars, closed.consts):
        if not isinstance(c, jax.core.Tracer):
            keep_vars.append(v)
            keep_vals.append(c)
            continue
        idx = next((i for i, h in enumerate(hoisted) if h is c), None)
        if idx is None:
            raise ValueError("a cost's residual closes over a value traced by jax.jit; pass it "
                             "to the cost factory as an argument instead")
        lifted.append((idx, v))
    lifted.sort(key=lambda t: t[0])
    info = jaxpr.debug_info
    if info is not None and info.arg_names is not None:
        info = info._replace(arg_names=(*info.arg_names, *(f"lifted{i}" for i, _ in lifted)))
    new = jaxpr.replace(constvars=keep_vars, invars=[*jaxpr.invars, *(v for _, v in lifted)],
                        debug_info=info)
    return jax.extend.core.ClosedJaxpr(new, keep_vals), [i for i, _ in lifted]


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


def _build(header: str, tier: str) -> str:
    """GRiM's LM / augmented-Lagrangian solver (``grim.motion.costs``) compiled around this
    problem's generated header, at one tier -> FFI target. Cached in memory and on disk."""
    from .. import _grim

    _grim.ensure_importable()
    from grim.motion.costs import dense_solver

    try:
        return dense_solver(header, tier)
    except FileNotFoundError as e:
        if shutil.which("nvcc") is None:
            raise RuntimeError("pk.LeastSquaresProblem.analyze() compiles to CUDA and needs "
                               "nvcc on PATH; it was not found.") from e
        raise


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

    def __init__(self, variables: list[Var], plans: list[_CostPlan], key: tuple | None = None):
        self.layout, self.slots, self.n_q, self.n_x = _variable_layout(variables)
        self.manifold = any(k[0].retract_fn is not None for k in self.slots)
        if self.n_x > _MAX_VARS:
            raise NotImplementedError(
                f"{self.n_x} variables: the compiled dense solve supports up to {_MAX_VARS} "
                "(larger problems, e.g. trajectories, are not implemented yet)")
        # Tracing never reads a plan's runtime values (they arrive as the symbolic p), so keep
        # plans without them: they may be the caller's tracers.
        self.plans = [dataclasses.replace(pl, params=[]) for pl in plans]
        self.param_specs = [[(tuple(p.shape), p.dtype) for p in plan.params] for plan in plans]
        n_p_costs = sum(int(np.prod(s, dtype=int)) for specs in self.param_specs for s, _ in specs)
        q0 = np.zeros(self.n_q, np.float32)
        t0, p0 = np.zeros(self.n_x, np.float32), np.zeros(n_p_costs, np.float32)
        self._retract = _retract_fn(self.layout, self.slots)
        # Traced values the retractions close over, appended to the runtime parameters.
        hoisted = _closure_tracers(self.layout, self.slots, self.n_q, self.n_x) if self.manifold else []
        self.hoist_shapes = [tuple(h.shape) for h in hoisted]
        self.hoist_offsets = list(np.cumsum([0] + [int(np.prod(s, dtype=int)) for s in self.hoist_shapes]))
        self.n_p = n_p_costs + int(self.hoist_offsets[-1])
        self._n_p_costs = n_p_costs
        self._hoisted = hoisted
        self._traced = False
        self._variants: dict[str, _Variant] = {}
        self._targets: dict[str, str] = {}
        self.tier_table: dict[int, str] = {}   # batch size -> fastest tier (from calibrate)

        # Disk cache: generated headers per variant, keyed by the problem's structure and
        # checked against a residual probe (see _disk_entry). A hit skips tracing and emitting.
        self._cache = _CostCache(_digest(key, self.plans)) if key is not None and _cache_dir() else None
        meta = self._cache.load(self) if self._cache else None
        if meta is None or hoisted:
            self._trace()  # also when the retractions close over this call's tracers
        if meta is None:
            self.row_kinds, self.row_sign, self.n_r = self._row_kinds()
            if self._cache:
                self._cache.save_meta(self)
        else:
            self.row_kinds, self.row_sign = meta["row_kinds"], np.array(meta["row_sign"], np.float32)
            self.n_r = len(self.row_kinds)
        self.natives = self._native_instances()

    def _trace(self) -> None:
        """The problem's residual with its Jacobian, its JVP and the retraction, as jaxprs."""
        if self._traced:
            return
        hoisted = self._hoisted
        q0 = np.zeros(self.n_q, np.float32)
        t0, p0 = np.zeros(self.n_x, np.float32), np.zeros(self._n_p_costs, np.float32)
        if self.manifold:
            # Differentiate in the tangent space at x, as jaxls: r(retract(x, d)) at d = 0.
            def local(x, p):
                return lambda d: jnp.concatenate(self._residual_rows(self._retract(x, d), p))
        else:
            def local(x, p):
                return lambda x: jnp.concatenate(self._residual_rows(x, p))

        def with_jac(x, p):
            at = jnp.zeros(self.n_x, jnp.float32) if self.manifold else x
            return jax.jacfwd(lambda d: (local(x, p)(d),) * 2, has_aux=True)(at)

        def with_jvp(x, p, t):
            at = jnp.zeros(self.n_x, jnp.float32) if self.manifold else x
            return jax.jvp(local(x, p), (at,), (t,))

        self.closed, self._lift = _lift_tracers(jax.make_jaxpr(with_jac)(q0, p0), hoisted)
        self.closed_jvp, self._lift_jvp = _lift_tracers(jax.make_jaxpr(with_jvp)(q0, p0, t0), hoisted)
        self.closed_retract, self._lift_retract = (
            _lift_tracers(jax.make_jaxpr(self._retract)(q0, t0), hoisted) if self.manifold
            else (None, []))
        self._traced = True

    def _native_instances(self) -> list[tuple]:
        """(native, row offset, x offset, J column) per instance of every native cost."""
        out, row = [], 0
        for plan in self.plans:
            nat = plan.cost.native
            n_inst = 1 if plan.batch is None else plan.batch
            if nat is None:
                row += self._plan_rows(plan) * n_inst
                continue
            var = next(it for k, it in zip(plan.kinds, plan.items) if k == "var")
            ids = np.atleast_1d(np.asarray(var.id)).ravel()
            for i in range(n_inst):
                sl = self.layout[(type(var), int(ids[i if len(ids) > 1 else 0]))]
                if sl.q_size != nat.dof or sl.t_dim != nat.dof:
                    raise NotImplementedError("native residuals need a Euclidean variable of "
                                              f"{nat.dof} coordinates")
                out.append((nat, row, sl.q, sl.t))
                row += nat.rows
        return out

    def _plan_rows(self, plan) -> int:
        """Residual rows of ONE instance of ``plan`` (by shape evaluation)."""
        i = None if plan.batch is None else 0
        vals = VarValues({key: sl.value(np.zeros(self.n_q, np.float32))
                          for key, sl in self.layout.items()})
        params = [np.zeros(s, d) for s, d in self.param_specs[self.plans.index(plan)]]
        out = jax.eval_shape(lambda: jnp.ravel(_first(plan.cost.compute_residual(
            vals, *_instance(plan, i, [jnp.asarray(p) for p in params])))))
        return int(out.shape[0])

    def _native_source(self) -> str:
        """Native device functions (each once) for the generated header."""
        seen, src = set(), []
        for nat, *_ in self.natives:
            if nat.fn not in seen:
                seen.add(nat.fn)
                src.append(nat.source)
        return "\n".join(src)

    def _native_calls(self, mode: str) -> str:
        """Calls that fill the native rows after the traced body. ``mode``: "r" (residual),
        "J" (residual + row-major Jacobian) or "jvp" (residual + J t)."""
        if not self.natives:
            return ""
        out = ["    if (rank == 0) {"]
        for nat, row, xq, xt in self.natives:
            if mode == "r":
                out.append(f"        {nat.fn}(x + {xq}, r + {row}, nullptr);")
            else:
                out.append("        {")
                out.append(f"            float Jl[{nat.rows * nat.dof}];")
                out.append(f"            {nat.fn}(x + {xq}, r + {row}, Jl);")
                out.append(f"            for (int a = 0; a < {nat.rows}; ++a) {{")
                if mode == "J":
                    out.append(f"                for (int b = 0; b < {nat.dof}; ++b) "
                               f"J[({row} + a) * n_x + {xt} + b] = Jl[a * {nat.dof} + b];")
                else:
                    out.append("                float acc = 0.f;")
                    out.append(f"                for (int b = 0; b < {nat.dof}; ++b) "
                               f"acc += Jl[a * {nat.dof} + b] * t[{xt} + b];")
                    out.append(f"                jt[{row} + a] = acc;")
                out.append("            }")
                out.append("        }")
        out.append("    }")
        if mode != "jvp":
            out.append("    GRIM_COST_SYNC();")
        return "\n".join(out)

    def _row_kinds(self) -> tuple[list[int], np.ndarray, int]:
        kinds, sign = [], []
        shapes = jax.eval_shape(self._residual_rows, np.zeros(self.n_q, np.float32),
                                np.zeros(self._n_p_costs, np.float32))
        for plan, rows in zip(self.plans, shapes):
            kinds += [_ROW_KIND[plan.cost.kind]] * rows.shape[0]
            sign += [-1 if plan.cost.kind == "constraint_geq_zero" else 1] * rows.shape[0]
        return kinds, np.array(sign, np.float32), len(kinds)

    def probe(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """The residual at ``(x, p)``, evaluated by JAX (the cache's staleness check)."""
        with jax.ensure_compile_time_eval(), jax.default_device(jax.devices("cpu")[0]):
            return np.asarray(jnp.concatenate(self._residual_rows(jnp.asarray(x), jnp.asarray(p))))

    def _symbolic_inputs(self, g: E.Graph, lift: list[int], *, tangent: bool = False) -> list:
        """Symbolic ``x``, the cost parameters ``p`` (``t`` too for ``tangent``), then each lifted
        tracer as its slice of ``p`` after the cost parameters."""
        args = [E.symbolic(g, "x", (self.n_q,)), E.symbolic(g, "p", (self._n_p_costs,))]
        if tangent:
            args.append(E.symbolic(g, "t", (self.n_x,)))
        return args + [E.symbolic(g, "p", self.hoist_shapes[i],
                                  offset=self._n_p_costs + int(self.hoist_offsets[i])) for i in lift]

    def _residual_rows(self, x, p) -> list[jax.Array]:
        """Each cost's residual rows (all batch instances), in problem order."""
        vals = VarValues({key: sl.value(x) for key, sl in self.layout.items()})
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
            cached = self._cache.load_variant(name) if self._cache else None
            if cached is None:
                with jax.ensure_compile_time_eval():
                    self._trace()
                    cached = (self._generate(float("inf")) if name == "thread"
                              else self._generate_jvp())
                if self._cache:
                    self._cache.save_variant(name, cached)
            self._variants[name] = cached
        return self._variants[name]

    def io_dtype(self):
        return jnp.float32

    def workspace_bytes(self, tier: str) -> int:
        return 4 * max(self.scratch(tier), 1)

    def slot_bytes(self, tier: str) -> int:
        """Per-problem solver state held in shared (warp/block) or local (thread) memory;
        the generated code's scratch is a separate global workspace."""
        n, r = self.n_x, self.n_r
        return 4 * (2 * self.n_q + 2 * n + n * n + 5 * r + r * n)

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
        J, r = ev.run(self.closed, self._symbolic_inputs(g, self._lift))
        n = self.n_x
        r = self._signed(ev, r, lambda g_, i: i)
        J = self._signed(ev, J, lambda g_, i: g_.idiv(i, n) if g_ is not None else i // n)
        res_body, s1, e1 = g.emit_function([("r", r)])
        scalar = (not isinstance(J, E.Lazy) and not isinstance(r, E.Lazy)
                  and not getattr(self, "natives", None))
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
#define GRIM_COST_NORMAL_EQUATIONS 1
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
        retract_src = self._retract_source(g)
        header = f"""// Generated by pyroffi (LeastSquaresProblem.analyze). Do not edit.
#pragma once
#include <cmath>

namespace grim::costs_gen {{

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
{self._native_source()}
{retract_src}
__device__ __forceinline__ void residual(
    const float* __restrict__ x, const float* __restrict__ p, float* __restrict__ r,
    float* __restrict__ scratch, const int rank, const int size)
{{
    (void)p; (void)scratch; (void)size;
{res_body}
{self._native_calls("r")}
}}

__device__ __forceinline__ void residual_jacobian(
    const float* __restrict__ x, const float* __restrict__ p, float* __restrict__ r,
    float* __restrict__ J, float* __restrict__ scratch, const int rank, const int size)
{{
    (void)p; (void)scratch; (void)size;
{jac_body}
{self._native_calls("J")}
}}

{ne_src}
}}  // namespace grim::costs_gen
"""
        return _Variant(header, scratch, max(e1, e2))

    def _generate_jvp(self) -> _Variant:
        g = E.Graph()
        ev = E.Evaluator(g, float("inf"))
        r, jt = ev.run(self.closed_jvp, self._symbolic_inputs(g, self._lift_jvp, tangent=True))
        r = self._signed(ev, r, lambda g_, i: i)
        jt = self._signed(ev, jt, lambda g_, i: i)
        res_body, _, _ = g.emit_function([("r", r)], sync=False)
        jvp_body, _, _ = g.emit_function([("r", r), ("jt", jt)], sync=False)
        kinds = self.row_kinds
        sparse_ne = self._sparse_normal_equations_tables(g, jt if self.manifold else r)
        retract_src = self._retract_source(g)
        header = f"""// Generated by pyroffi (LeastSquaresProblem.analyze). Do not edit.
#pragma once
#include <cmath>

namespace grim::costs_gen {{

constexpr int n_x = {self.n_x};
constexpr int n_r = {self.n_r};
constexpr int n_p = {self.n_p};
constexpr int scratch_size = 0;
constexpr bool has_constraints = {"true" if any(k != 0 for k in kinds) else "false"};
#define GRIM_COST_JVP 1

__device__ __forceinline__ int row_kind(int i) {{
    constexpr int k[n_r] = {{{", ".join(map(str, kinds))}}};
    return k[i];
}}

{g.tables_source()}
{self._native_source()}
{retract_src}{sparse_ne}
// Lane-private straight-line code: one lane, no syncs.
__device__ __forceinline__ void residual(
    const float* __restrict__ x, const float* __restrict__ p, float* __restrict__ r,
    float* __restrict__ scratch, const int rank, const int size)
{{
    (void)p; (void)scratch; (void)size;
{res_body}
{self._native_calls("r").replace("    GRIM_COST_SYNC();", "")}
}}

// The residual and its directional derivative along t (one Jacobian column for t = e_j).
__device__ __forceinline__ void residual_jvp(
    const float* __restrict__ x, const float* __restrict__ p, const float* __restrict__ t,
    float* __restrict__ r, float* __restrict__ jt)
{{
    (void)p; const int rank = 0;
{jvp_body}
{self._native_calls("jvp")}
}}

}}  // namespace grim::costs_gen
"""
        return _Variant(header, 0, self.n_x)

    def _retract_source(self, g: E.Graph) -> str:
        """``n_q`` and, for Lie-group variables, the step ``retract(x, t, xn)`` the kernel
        applies (``t`` is the tangent step): generated from the variables' retractions."""
        if not self.manifold:
            return ""
        x, t = E.symbolic(g, "x", (self.n_q,)), E.symbolic(g, "t", (self.n_x,))
        lifted = self._symbolic_inputs(g, self._lift_retract)[2:]
        (xn,) = E.Evaluator(g, float("inf")).run(self.closed_retract, [x, t, *lifted])
        body, _, _ = g.emit_function([("xn", xn)], sync=False)
        return f"""constexpr int n_q = {self.n_q};
#define GRIM_COST_MANIFOLD 1
__device__ __forceinline__ void retract(
    const float* __restrict__ x, const float* __restrict__ p, const float* __restrict__ t,
    float* __restrict__ xn)
{{
    (void)p; const int rank = 0;
{body}
}}
"""

    def _sparse_normal_equations_tables(self, g: E.Graph, rows) -> str:
        """Tables for assembling H = J^T W J and g = J^T W (r + s) from the Jacobian's
        structural sparsity (which variables each residual row can depend on, read off the
        compiled residual graph, so exact: a row inactive at some point is still counted).

        For each H entry (a <= c, every diagonal included) the rows touching both columns,
        and for each column the rows touching it. Rows such as joint limits or rest poses
        touch one variable each, so this is far less than the dense n_r * n_x^2 sum.

        Columns are tangent coordinates. For Euclidean variables they are the entries of
        ``x``, read off the residual ``rows``; with Lie-group variables a parameter of ``x``
        feeds several columns, so the pattern is read off the directional derivative (``rows``
        is then ``jt``) and its dependence on the tangent ``t``.
        """
        col = "t" if self.manifold else "x"
        dep: dict[int, int] = {}
        for n in g.nodes:
            if n.op == col:
                dep[n.idx] = 1 << int(n.args[0])
            elif n.op in ("x", "p", "t", "w", "s", "idx", "table"):
                dep[n.idx] = 0
            else:
                m = 0
                for arg in n.args:
                    if isinstance(arg, E.Sym):
                        m |= dep[arg.idx]
                dep[n.idx] = m
        n = self.n_x
        masks = [dep[v.idx] if isinstance(v, E.Sym) else 0 for v in np.asarray(rows).flat]
        # Native rows trace as zeros; they depend on every column of their variable.
        for nat, row, _xq, xt in getattr(self, "natives", ()):
            for a in range(nat.rows):
                masks[row + a] |= ((1 << nat.dof) - 1) << xt
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
            "#define GRIM_COST_SPARSE_NE 1",
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


# ── banded (trajectory) problems ─────────────────────────────────────────────────────────

def _build_banded(header: str, io: str) -> tuple[str, int]:
    """GRiM's banded LM / AL solver (double precision) around this header -> (FFI target,
    workspace bytes per problem)."""
    from .. import _grim

    _grim.ensure_importable()
    from grim.motion.costs import banded_solver

    try:
        return banded_solver(header, real="double", io=io)
    except FileNotFoundError as e:
        if shutil.which("nvcc") is None:
            raise RuntimeError("pk.LeastSquaresProblem.analyze() compiles to CUDA and needs "
                               "nvcc on PATH; it was not found.") from e
        raise


@dataclass
class _Group:
    """One cost: its instances share a residual over their local variables."""

    n_inst: int
    n_loc: int
    n_ploc: int
    n_r: int                 # rows per instance
    kinds: list[int]         # row kinds of one instance
    vtab: np.ndarray         # (n_inst, n_loc) global variable index of each local variable
    ptab: np.ndarray         # (n_inst, n_ploc) global parameter index
    residual: Any            # residual(xl, pl) of one instance
    closed_res: Any = None
    closed_jvp: Any = None
    native: Any = None       # NativeResidual: hand-written rows instead of the traced body
    sign: float = 1.0


class _BandedKernel:
    """Trajectory-shaped problems: more variables than the dense solve holds, with every cost
    a batch of instances that each touch a few neighbouring variables (knots).

    A batched cost is compiled ONCE, as a function of one instance's variables and runtime
    parameters, and run per instance; its baked arguments (robot, collision model, weights)
    must be the same for every instance. The normal equations are banded (bandwidth = the
    widest span of variables an instance touches) and solved in double precision by GRiM's
    banded Cholesky (``cost_lm_banded.cu``): one CUDA block per problem, state in a global
    workspace. Variables in ``fixed`` keep their initial value.
    """

    def __init__(self, variables: list[Var], plans: list[_CostPlan], fixed: tuple = (),
                 key: tuple | None = None):
        self.layout, self.slots, self.n_q, self.n_x = _variable_layout(variables)
        if self.n_q != self.n_x:
            raise NotImplementedError(
                "Lie-group variables (SE3Var, ...) in a banded (trajectory) problem are not "
                "supported yet; use structure='dense' (up to 128 tangent dimensions)")
        self.manifold, self.hoist_shapes, self.tier_table = False, [], {}
        self.param_specs = [[(tuple(p.shape), p.dtype) for p in plan.params] for plan in plans]
        self.n_p = self._n_p_costs = sum(int(np.prod(s, dtype=int))
                                         for specs in self.param_specs for s, _ in specs)
        self.fixed = np.zeros(self.n_x, np.int32)
        for var in fixed:
            for vid in np.atleast_1d(np.asarray(_concrete_var(var).id)).ravel():
                sl = self.layout[(type(var), int(vid))]
                self.fixed[sl.q:sl.q + sl.q_size] = 1
        self.groups, p_off = [], 0
        with jax.enable_x64(True):
            for plan, specs in zip(plans, self.param_specs):
                self.groups.append(self._group(dataclasses.replace(plan, params=[]), specs, p_off))
                p_off += sum(int(np.prod(sh, dtype=int)) for sh, _ in specs)
        self.row_kinds = [k for gr in self.groups for k in gr.kinds * gr.n_inst]
        self.row_sign = np.ones(len(self.row_kinds), np.float32)  # >= rows are negated per group
        self.n_r = len(self.row_kinds)
        self.band = max(int((gr.vtab.max(1) - gr.vtab.min(1)).max()) for gr in self.groups)
        self._variants: dict[str, _Variant] = {}
        self._targets: dict[str, tuple[str, int]] = {}
        self._cache = (_CostCache(_digest(("banded", key), [g for g in plans]))
                       if key is not None and _cache_dir() else None)
        if self._cache and self._cache.load(self) is None:
            self._cache.save_meta(self)

    def _group(self, plan: _CostPlan, specs, p_off: int) -> _Group:
        n_inst = 1 if plan.batch is None else plan.batch
        idx = [k for k, kind in enumerate(plan.kinds) if kind == "var"]
        for k, (kind, item) in enumerate(zip(plan.kinds, plan.items)):
            if kind == "bake" and item is not None and plan.batch is not None:
                for leaf in jax.tree.leaves(item):
                    if not np.array_equal(leaf, np.broadcast_to(leaf[:1], leaf.shape), equal_nan=True):
                        raise NotImplementedError(
                            f"{plan.cost._get_name()}: a baked argument differs between batch "
                            "instances; the banded solver compiles one residual per batched "
                            "cost, so pass per-instance values as jax.Array")

        def ids(i):  # local variables of instance i: the (type, id) its Var arguments name
            return [(type(plan.items[k]), int(np.asarray(plan.items[k].id).ravel()[i])
                     if plan.batch is not None else int(np.asarray(plan.items[k].id)))
                    for k in idx]
        keys0 = list(dict.fromkeys(ids(0)))
        local, n_loc = {}, 0
        for key in keys0:
            sl = self.layout[key]
            local[key] = (n_loc, sl.q_size, sl.shapes[0])
            n_loc += sl.q_size
        pos = [keys0.index(key) for key in ids(0)]
        vtab = np.zeros((n_inst, n_loc), np.int32)
        for i in range(n_inst):
            ki = ids(i)
            if len(dict(zip(pos, ki))) != len(set(zip(pos, ki))):
                raise NotImplementedError(
                    f"{plan.cost._get_name()}: arguments that name one variable in the first "
                    "instance name different variables in another")
            for m, key in zip(pos, ki):
                sl = self.layout[key]
                lo = local[keys0[m]][0]
                vtab[i, lo:lo + sl.q_size] = np.arange(sl.q, sl.q + sl.q_size)
        # Runtime parameters of instance i: slice i of each (batched) parameter array.
        inst_shapes = [sh[1:] if plan.batch is not None else sh for sh, _ in specs]
        sizes = [int(np.prod(sh, dtype=int)) for sh in inst_shapes]
        n_ploc = sum(sizes)
        ptab = np.zeros((n_inst, n_ploc), np.int32)
        for i in range(n_inst):
            col, off = 0, p_off
            for (shape, _), sz in zip(specs, sizes):
                ptab[i, col:col + sz] = off + i * sz + np.arange(sz)
                col += sz
                off += int(np.prod(shape, dtype=int))
        sign = -1.0 if plan.cost.kind == "constraint_geq_zero" else 1.0

        def residual(xl, pl):
            vals = VarValues({key: xl[o:o + sz].reshape(sh) for key, (o, sz, sh) in local.items()})
            params, c = [], 0
            for (shape, dtype), ish, sz in zip(specs, inst_shapes, sizes):
                params.append(pl[c:c + sz].reshape(ish).astype(dtype))
                c += sz
            it = iter(params)
            items = []
            for kind, item in zip(plan.kinds, plan.items):
                if kind == "var":
                    items.append(item if plan.batch is None
                                 else type(item)(int(np.asarray(item.id).ravel()[0])))
                elif kind == "param":
                    items.append(next(it))
                elif item is None or plan.batch is None:
                    items.append(item)
                else:
                    items.append(jax.tree.map(lambda a: a[0], item))
            out = plan.cost.compute_residual(vals, *jax.tree.unflatten(plan.treedef, items))
            if isinstance(out, tuple):
                out = out[0]
            return sign * jnp.ravel(out).astype(jnp.float64)

        n_r = int(jax.eval_shape(residual, np.zeros(n_loc), np.zeros(n_ploc)).shape[0])
        nat = plan.cost.native
        if nat is not None and (n_loc != nat.dof or n_r != nat.rows):
            raise NotImplementedError("native residuals need one Euclidean variable of "
                                      f"{nat.dof} coordinates")
        return _Group(n_inst, n_loc, n_ploc, n_r, [_ROW_KIND[plan.cost.kind]] * n_r, vtab, ptab,
                      residual, native=nat, sign=sign)

    def _trace(self) -> None:
        for gr in self.groups:
            if gr.closed_res is None and gr.native is None:
                xl0, pl0 = np.zeros(gr.n_loc), np.zeros(gr.n_ploc)
                gr.closed_res = jax.make_jaxpr(gr.residual)(xl0, pl0)
                gr.closed_jvp = jax.make_jaxpr(
                    lambda x, p, t, f=gr.residual: jax.jvp(lambda x: f(x, p), (x,), (t,)))(xl0, pl0, xl0)

    def probe(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Each group's residual at its first instance (the disk cache's staleness check)."""
        with jax.enable_x64(True), jax.ensure_compile_time_eval(), \
                jax.default_device(jax.devices("cpu")[0]):
            return np.concatenate([np.asarray(gr.residual(jnp.asarray(x[gr.vtab[0]], jnp.float64),
                                                          jnp.asarray(p[gr.ptab[0]], jnp.float64)))
                                   for gr in self.groups])

    # The block tier only; the interface the solve methods use.
    def auto_tier(self, batch: int) -> str:
        return "block"

    def available(self, tier: str) -> bool:
        return tier == "block"

    def io_dtype(self):
        """FFI buffers: float64 under JAX x64, else float32 (the solve is double either way)."""
        return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

    def variant(self, name: str = "banded") -> _Variant:
        if "banded" not in self._variants:
            cached = self._cache.load_variant("banded") if self._cache else None
            if cached is None:
                with jax.ensure_compile_time_eval(), jax.enable_x64(True), E.precision("double"):
                    self._trace()
                    cached = self._generate_banded()
                if self._cache:
                    self._cache.save_variant("banded", cached)
            self._variants["banded"] = cached
        return self._variants["banded"]

    def _built(self) -> tuple[str, int]:
        io = "double" if self.io_dtype() == jnp.float64 else "float"
        if io not in self._targets:
            self._targets[io] = _build_banded(self.variant().header, io)
        return self._targets[io]

    def target(self, tier: str) -> str:
        if tier != "block":
            raise NotImplementedError("banded (trajectory) problems run at the block tier only")
        return self._built()[0]

    def workspace_bytes(self, tier: str) -> int:
        return self._built()[1]

    def _generate_banded(self) -> _Variant:
        funcs, seen = [], set()
        for k, gr in enumerate(self.groups):
            if gr.native is not None:
                nat = gr.native
                if nat.fn not in seen:
                    seen.add(nat.fn)
                    funcs.append(nat.source)
                # The native contract is float; the banded solve is double.
                funcs.append(f"""
namespace g{k}_res {{
__device__ __forceinline__ void run(const real* __restrict__ x, const real* __restrict__ p,
                                     real* __restrict__ r)
{{
    (void)p; float q[{nat.dof}], rf[{nat.rows}];
    for (int b = 0; b < {nat.dof}; ++b) q[b] = (float)x[b];
    {nat.fn}(q, rf, nullptr);
    for (int a = 0; a < {nat.rows}; ++a) r[a] = (real)({gr.sign}f * rf[a]);
}}
}}
namespace g{k}_jvp {{
__device__ __forceinline__ void run(const real* __restrict__ x, const real* __restrict__ p,
                                     const real* __restrict__ t, real* __restrict__ r,
                                     real* __restrict__ jt)
{{
    (void)p; float q[{nat.dof}], rf[{nat.rows}], Jl[{nat.rows * nat.dof}];
    for (int b = 0; b < {nat.dof}; ++b) q[b] = (float)x[b];
    {nat.fn}(q, rf, Jl);
    for (int a = 0; a < {nat.rows}; ++a) {{
        real acc = 0;
        for (int b = 0; b < {nat.dof}; ++b) acc += (real)Jl[a * {nat.dof} + b] * t[b];
        r[a] = (real)({gr.sign}f * rf[a]); jt[a] = (real)({gr.sign}f) * acc;
    }}
}}
}}""")
                continue
            g = E.Graph()
            (r,) = E.Evaluator(g, float("inf")).run(
                gr.closed_res, [E.symbolic(g, "x", (gr.n_loc,)), E.symbolic(g, "p", (gr.n_ploc,))])
            res_body, _, _ = g.emit_function([("r", r)], sync=False)
            g2 = E.Graph()
            r2, jt = E.Evaluator(g2, float("inf")).run(
                gr.closed_jvp, [E.symbolic(g2, "x", (gr.n_loc,)), E.symbolic(g2, "p", (gr.n_ploc,)),
                                E.symbolic(g2, "t", (gr.n_loc,))])
            jvp_body, _, _ = g2.emit_function([("r", r2), ("jt", jt)], sync=False)
            funcs.append(f"""
namespace g{k}_res {{
{g.tables_source()}
__device__ __forceinline__ void run(const real* __restrict__ x, const real* __restrict__ p,
                                     real* __restrict__ r)
{{
    const int rank = 0; (void)rank; (void)p;
{res_body}
}}
}}
namespace g{k}_jvp {{
{g2.tables_source()}
__device__ __forceinline__ void run(const real* __restrict__ x, const real* __restrict__ p,
                                     const real* __restrict__ t, real* __restrict__ r,
                                     real* __restrict__ jt)
{{
    const int rank = 0; (void)rank; (void)p;
{jvp_body}
}}
}}""")
        G = self.groups

        def arr(name, vals, ctype="int"):
            vals = list(map(int, vals))
            return f"__device__ const {ctype} {name}[] = {{{', '.join(map(str, vals)) or '0'}}};"

        def prefix(vals):
            return np.concatenate([[0], np.cumsum(vals)]).astype(int)

        n_cols = [gr.n_inst * gr.n_loc for gr in G]
        j_sizes = [gr.n_inst * gr.n_r * gr.n_loc for gr in G]
        rows = [gr.n_inst * gr.n_r for gr in G]
        res_cases = "\n".join(f"        case {k}: g{k}_res::run(x, p, r); break;" for k in range(len(G)))
        jvp_cases = "\n".join(f"        case {k}: g{k}_jvp::run(x, p, t, r, jt); break;"
                              for k in range(len(G)))
        header = f"""// Generated by pyroffi (LeastSquaresProblem.analyze, banded). Do not edit.
#pragma once
#include <cmath>

namespace grim::costs_gen {{

constexpr int n_x = {self.n_x};
constexpr int n_r = {self.n_r};
constexpr int n_p = {self.n_p};
constexpr int band = {self.band};
constexpr int n_groups = {len(G)};
constexpr int n_inst_total = {sum(gr.n_inst for gr in G)};
constexpr int n_col_total = {sum(n_cols)};
constexpr long j_size = {sum(j_sizes)};
constexpr int max_nloc = {max(gr.n_loc for gr in G)};
constexpr int max_nploc = {max(gr.n_ploc for gr in G)};
constexpr int max_nr = {max(gr.n_r for gr in G)};
constexpr bool has_constraints = {"true" if any(k != 0 for k in self.row_kinds) else "false"};

{arr("g_nloc", [gr.n_loc for gr in G])}
{arr("g_nploc", [gr.n_ploc for gr in G])}
{arr("g_nr", [gr.n_r for gr in G])}
{arr("g_inst_prefix", prefix([gr.n_inst for gr in G]))}
{arr("g_col_prefix", prefix(n_cols))}
{arr("g_row0", prefix(rows)[:-1])}
{arr("g_vtab0", prefix([gr.vtab.size for gr in G])[:-1])}
{arr("g_ptab0", prefix([gr.ptab.size for gr in G])[:-1])}
{arr("g_j0", prefix(j_sizes)[:-1], "long")}
{arr("vtab", np.concatenate([gr.vtab.ravel() for gr in G]))}
{arr("ptab", np.concatenate([gr.ptab.ravel() for gr in G]))}
{arr("row_kind", self.row_kinds, "unsigned char")}
{arr("fixed", self.fixed, "unsigned char")}
{"".join(funcs)}

__device__ __noinline__ void stage_res(int g, const real* x, const real* p, real* r)
{{
    switch (g) {{
{res_cases}
    }}
}}

__device__ __noinline__ void stage_jvp(int g, const real* x, const real* p, const real* t,
                                       real* r, real* jt)
{{
    switch (g) {{
{jvp_cases}
    }}
}}

}}  // namespace grim::costs_gen
"""
        return _Variant(header, 0, 1)


# ── persistent cache of generated headers ───────────────────────────────────────────────

def _cache_dir() -> Path | None:
    """``$PYROFFI_COST_CACHE`` (``0`` or empty disables), else ``~/.cache/pyroffi/costs``."""
    env = os.environ.get("PYROFFI_COST_CACHE")
    if env is not None:
        return Path(env) if env not in ("", "0") else None
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "pyroffi" / "costs"


@lru_cache(maxsize=None)
def _library_digest() -> str:
    """pyroffi's own source and JAX's version: a new release never reuses an old header."""
    h = hashlib.sha256(f"{jax.__version__}".encode())
    root = Path(__file__).resolve().parents[1]
    for f in sorted(root.rglob("*.py")):
        h.update(str(f.relative_to(root)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def _code_digest(fn, h, seen: set) -> None:
    """Bytecode, constants and closure of ``fn`` (and of functions it closes over)."""
    if id(fn) in seen:
        return
    seen.add(id(fn))
    code = getattr(fn, "__code__", None)
    if code is None:
        h.update(repr(fn).encode() if not callable(fn) else getattr(fn, "__qualname__", "?").encode())
        return
    stack = [code]
    while stack:
        c = stack.pop()
        h.update(c.co_code)
        h.update(repr(c.co_names).encode())
        for k in c.co_consts:
            if hasattr(k, "co_code"):
                stack.append(k)
            else:
                h.update(repr(k).encode())
    for cell in fn.__closure__ or ():
        try:
            v = cell.cell_contents
        except ValueError:
            continue
        if callable(v):
            _code_digest(v, h, seen)
        elif isinstance(v, (np.ndarray, int, float, str, bool, tuple)) or v is None:
            h.update(repr(v).encode() if not isinstance(v, np.ndarray) else v.tobytes())


def _stable(obj) -> str:
    """A process-independent rendering of an analyze() key."""
    if isinstance(obj, (tuple, list)):
        return "(" + ",".join(_stable(o) for o in obj) + ")"
    if isinstance(obj, type):
        name = f"{obj.__module__}.{obj.__qualname__}"
        if getattr(obj, "retract_fn", None) is not None:  # a Lie-group variable's manifold
            h = hashlib.sha256()
            _code_digest(obj.retract_fn, h, set())
            name += "@" + h.hexdigest()[:16]
        return name
    if isinstance(obj, bytes):
        return obj.hex()
    # Static pytree fields can render objects by address (Robot's backend handle); the residual
    # probe, not the key, is what guarantees an entry still matches.
    return re.sub(r" at 0x[0-9a-f]+", "", repr(obj))


def _digest(key: tuple, plans: list[_CostPlan]) -> str:
    h = hashlib.sha256(_library_digest().encode())
    h.update(_stable(key).encode())
    for plan in plans:
        _code_digest(plan.cost.compute_residual, h, set())
    return h.hexdigest()[:32]


class _CostCache:
    """One problem structure's directory: ``meta.json`` (rows, and a residual probe) and one
    header per variant. Writes are atomic renames, so concurrent processes are safe."""

    def __init__(self, digest: str):
        self.dir = _cache_dir() / digest

    def _write(self, name: str, text: str) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.dir, prefix=f".{name}.")
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, self.dir / name)

    @staticmethod
    def _probe_point(k: "_Kernel") -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        return (rng.uniform(-1, 1, k.n_q).astype(np.float32),
                rng.uniform(-1, 1, k._n_p_costs).astype(np.float32))

    def load(self, k: "_Kernel") -> dict | None:
        """The entry's metadata if it exists and the residual still evaluates to the stored
        probe (a changed residual, robot or library never reuses a header), else None."""
        try:
            meta = json.loads((self.dir / "meta.json").read_text())
        except (OSError, ValueError):
            return None
        try:
            got = k.probe(*self._probe_point(k))
        except Exception:  # noqa: BLE001 - any failure means: rebuild
            return None
        want = np.array(meta["probe"], np.float32)
        if got.shape != want.shape or not np.allclose(got, want, rtol=1e-5, atol=1e-6, equal_nan=True):
            return None
        return meta

    def save_meta(self, k: "_Kernel") -> None:
        """Start the entry afresh (a stale entry's headers must not be reused)."""
        for f in self.dir.glob("*") if self.dir.is_dir() else ():
            if f.suffix in (".cuh", ".json"):
                f.unlink(missing_ok=True)
        probe = k.probe(*self._probe_point(k))
        self._write("meta.json", json.dumps({
            "row_kinds": list(map(int, k.row_kinds)), "row_sign": k.row_sign.tolist(),
            "probe": probe.tolist()}))  # json round-trips NaN / inf

    def load_variant(self, name: str) -> _Variant | None:
        try:
            info = json.loads((self.dir / f"{name}.json").read_text())
            return _Variant((self.dir / f"{name}.cuh").read_text(), info["scratch"], info["max_extent"])
        except (OSError, ValueError, KeyError):
            return None

    def save_variant(self, name: str, v: _Variant) -> None:
        self._write(f"{name}.cuh", v.header)
        self._write(f"{name}.json", json.dumps({"scratch": v.scratch, "max_extent": v.max_extent}))


def analyze(problem: LeastSquaresProblem, structure: str = "auto") -> "AnalyzedLeastSquaresProblem":
    """``structure``: ``"dense"`` (up to 128 tangent dimensions, every tier), ``"banded"``
    (trajectories: batched costs over neighbouring knots, block tier, double precision), or
    ``"auto"`` (banded above 128 variables or when the problem fixes variables)."""
    _require_cuda()
    plans = [_plan(c) for c in problem.costs]
    variables = [_concrete_var(v) for v in problem.variables]
    fixed = tuple(_concrete_var(v) for v in getattr(problem, "fixed", ()))
    if structure == "auto":
        structure = "banded" if fixed or _variable_layout(variables)[3] > _MAX_VARS else "dense"
    if structure not in ("dense", "banded"):
        raise ValueError(f"structure must be 'auto', 'dense' or 'banded', got {structure!r}")
    if structure == "dense" and fixed:
        raise NotImplementedError("fixed variables need structure='banded'")
    key = (tuple(p.key() for p in plans),
           tuple((type(v), np.asarray(v.id).tobytes(), np.shape(v.id)) for v in variables))
    if structure == "banded":
        key = ("banded", key, tuple((type(v), np.asarray(v.id).tobytes()) for v in fixed))
    kernel = _KERNELS.get(key)
    if kernel is None:
        with jax.ensure_compile_time_eval():
            kernel = _KERNELS[key] = (_BandedKernel(variables, plans, fixed, key)
                                      if structure == "banded" else _Kernel(variables, plans, key))
    hoisted = []
    if kernel.hoist_shapes:  # this call's values of the tracers the retractions close over
        layout, slots, n_q, n_x = _variable_layout(variables)
        hoisted = _closure_tracers(layout, slots, n_q, n_x)
    return AnalyzedLeastSquaresProblem(kernel, plans, hoisted)


class AnalyzedLeastSquaresProblem:
    """A problem compiled to a CUDA kernel. Create with ``problem.analyze()``."""

    def __init__(self, kernel: _Kernel, plans: list[_CostPlan], hoisted: list | None = None):
        self._kernel = kernel
        self._plans = plans
        self._hoisted = hoisted or []

    @property
    def n_x(self) -> int:
        """Total tangent dimension (the size of the solve)."""
        return self._kernel.n_x

    @property
    def n_q(self) -> int:
        """Total parameter dimension (``n_x`` plus the extra parameters of Lie-group
        variables, e.g. 7 - 6 = 1 per SE3); the length of a flat solution."""
        return self._kernel.n_q

    @property
    def n_p(self) -> int:
        """Length of the runtime parameter vector."""
        return self._kernel.n_p

    def params(self) -> jax.Array:
        """This problem's runtime parameter vector ``(n_p,)``; stack several for
        :meth:`solve_batch`."""
        leaves = [p for plan in self._plans for p in plan.params] + list(self._hoisted)
        io = self._kernel.io_dtype()
        if not leaves:
            return jnp.zeros((0,), io)
        return jnp.concatenate([jnp.ravel(a).astype(io) for a in leaves])

    def initial_x(self, initial_vals: VarValues | None = None) -> jax.Array:
        """Flat initial variables ``(n_q,)``: ``initial_vals`` or each variable's default."""
        parts = []
        for vtype, vid in self._kernel.slots:
            v = initial_vals[vtype(vid)] if initial_vals is not None else vtype.default_factory()
            parts += [jnp.ravel(jnp.asarray(a, self._kernel.io_dtype())) for a in jax.tree.leaves(v)]
        return jnp.concatenate(parts)

    def unflatten(self, x: jax.Array) -> VarValues:
        """Flat ``(n_q,)`` variables to :class:`VarValues`."""
        return VarValues({key: sl.value(x) for key, sl in self._kernel.layout.items()})

    def solve_batch(self, params: jax.Array, x0: jax.Array, *, tier: str = "auto", **settings):
        """Solve ``B`` problems of this structure in one kernel launch.

        Args:
            params: ``(B, n_p)``, or ``(n_p,)`` shared by every problem.
            x0: ``(B, n_q)`` initial variables (see :meth:`initial_x`).
            tier: ``"thread"`` (one problem per thread), ``"warp"``, ``"block"``, or
                ``"auto"`` (the :meth:`calibrate` table, else a batch-size heuristic).
            **settings: ``max_iters``, ``al_iters``, ``lambda_initial``, ``tolerance``
                (relative cost change), ``al_tolerance``, ``grad_tolerance``, ``grad_start``,
                ``param_tolerance`` (jaxls's termination defaults).

        Returns:
            ``(x, cost, violation)``: ``(B, n_q)`` solutions, ``(B,)`` sum of squared cost
            residuals, ``(B,)`` largest constraint violation.
        """
        k = self._kernel
        io = k.io_dtype()
        x0 = jnp.asarray(x0, io)
        B = x0.shape[0]
        if k.n_p == 0:  # the kernel still takes a (1, 1) buffer
            params = jnp.zeros((1, 1), io)
        else:
            params = jnp.asarray(params, io).reshape(-1, k.n_p)
        s = {**_DEFAULTS, **settings}
        if tier == "auto":
            tier = k.auto_tier(B)
        elif tier not in TIERS:
            raise ValueError(f"tier must be 'auto' or one of {TIERS}, got {tier!r}")
        call = jax.ffi.ffi_call(k.target(tier), (
            jax.ShapeDtypeStruct((B, k.n_q), io),
            jax.ShapeDtypeStruct((B,), io),
            jax.ShapeDtypeStruct((B,), io),
            jax.ShapeDtypeStruct((B, k.workspace_bytes(tier)), jnp.uint8)))
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
            x0 = jnp.broadcast_to(self.initial_x(), (B, k.n_q))
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
