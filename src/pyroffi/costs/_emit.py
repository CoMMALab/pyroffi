"""jaxpr -> tier-generic CUDA C++ (straight-line code plus fused loops).

A jaxpr is evaluated over *symbolic* values, building an expression graph that is then
printed as the body of a ``template <Tier TIER>`` device function. Two value forms:

* **Scalarized** (arrays smaller than ``LOOP_MIN``): every element is its own node. Constants
  fold as nodes are built -- the robot's kinematic constants, and the zero tangents
  ``jax.jacfwd`` produces for variables a residual does not touch, disappear here (nvcc
  may not fold ``x * 0.0f`` under IEEE rules) -- and identical expressions are shared.
* **Lazy** (larger arrays): an element function ``fn(index) -> node`` over a symbolic loop
  index. Elementwise ops compose element functions, so chains of them fuse into one loop
  body with no intermediate buffers; data movement composes index maps. A lazy array is
  *materialized* into scratch memory only where recomputation would be wasteful: large
  reductions and contractions, and reuse of an expensive array.

Materializations become loops written in GLASS's form, ``for (i = rank; i < n; i += size)``,
followed by ``PYROFFI_COST_SYNC()``: sequential at ``Tier::Thread``, spread over a warp or
block at the cooperative tiers. Scalarized code runs redundantly on every lane of a group;
only the leader writes scalar outputs. Scratch buffers share one float array, with offsets
reused once a buffer is dead.

Data movement (reshape, gather, concatenate, scan slicing, ...) is never reimplemented: the
primitive itself is run on integer codes naming the input elements, and its output says
which element lands where. Anything that cannot be resolved (a data-dependent
``while_loop``, a sort of symbolic values) raises with the primitive's name.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import jax
import numpy as np
from jax.extend import core as jcore

LOOP_MIN = 1024
"""Arrays at least this large are lazy (loop-emitted); smaller ones are scalarized (this keeps
kinematic chains, whose tangent-stacked arrays are a few hundred elements, fully folded)."""
UNROLL_MAX = 8
"""Contractions and reductions over at most this many elements unroll inside a fused body."""
FANOUT_COST = 8
"""A lazy array whose elements cost at least this many ops is materialized before reuse..."""
MATERIALIZE_MAX = 4096
"""...if it has at most this many elements; larger ones are recomputed (scratch is per problem)."""
SCALARIZE_BUDGET = 4096
"""A small lazy result is scalarized only if size x element cost stays within this many ops;
otherwise it is computed by a (parallel) loop and read back."""


class EmitError(NotImplementedError):
    """The traced function uses something the emitter cannot express."""


def _kind(dtype) -> str:
    dtype = np.dtype(dtype)
    if dtype == np.bool_:
        return "b"
    if np.issubdtype(dtype, np.integer):
        return "i"
    return "f"


_NP_DTYPE = {"f": np.float32, "i": np.int32, "b": np.bool_}


def _const(value, kind: str):
    return _NP_DTYPE[kind](value)


class Sym:
    """A symbolic scalar: an input, a loop index, or an operation on other nodes.

    Buffer-producing statements (``loop``, ``pack``) are nodes too, so that dead-code
    elimination and ordering treat them uniformly; their details live in ``extra``.
    """

    __slots__ = ("op", "args", "kind", "idx", "extra")

    def __init__(self, op: str, args: tuple, kind: str, idx: int, extra=None):
        self.op, self.args, self.kind, self.idx, self.extra = op, args, kind, idx, extra


def _is_sym(v) -> bool:
    return isinstance(v, Sym)


def _key(v):
    return ("s", v.idx) if isinstance(v, Sym) else ("c", _kind(v.dtype), v.item())


_COMMUTATIVE = {"add", "mul", "max", "min", "eq", "ne", "and", "or", "xor"}


def _div(a, b):
    if np.issubdtype(np.asarray(a).dtype, np.integer):
        return np.int32(int(a / b)) if b != 0 else np.int32(0)
    return np.float32(a) / np.float32(b)


def _rem(a, b):
    if np.issubdtype(np.asarray(a).dtype, np.integer):
        return np.int32(int(math.fmod(int(a), int(b)))) if b != 0 else np.int32(0)
    return np.fmod(a, b)


# Constant evaluation of every elementwise op, in float32/int32 like the device.
_FOLD: dict[str, Callable] = {
    "add": np.add, "sub": np.subtract, "mul": np.multiply, "div": _div,
    "neg": np.negative, "max": np.maximum, "min": np.minimum, "abs": np.abs,
    "sign": np.sign, "floor": np.floor, "ceil": np.ceil, "round": np.round,
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "asin": np.arcsin, "acos": np.arccos,
    "atan": np.arctan, "atan2": np.arctan2, "sinh": np.sinh, "cosh": np.cosh,
    "tanh": np.tanh, "asinh": np.arcsinh, "acosh": np.arccosh, "atanh": np.arctanh,
    "exp": np.exp, "exp2": np.exp2, "log": np.log, "log1p": np.log1p, "expm1": np.expm1,
    "sqrt": np.sqrt, "rsqrt": lambda a: np.float32(1) / np.sqrt(a), "cbrt": np.cbrt,
    "pow": np.power, "square": np.square, "rem": _rem,
    "logistic": lambda a: np.float32(1) / (np.float32(1) + np.exp(-a)),
    "erf": lambda a: np.float32(math.erf(float(a))),
    "is_finite": np.isfinite,
    "eq": np.equal, "ne": np.not_equal, "lt": np.less, "le": np.less_equal,
    "gt": np.greater, "ge": np.greater_equal,
    "and": np.logical_and, "or": np.logical_or, "not": np.logical_not, "xor": np.logical_xor,
    "select": lambda c, a, b: b if c else a,  # select(c, a, b) = c ? b : a (select_n order)
}

_CPP_UNARY_F = {
    "abs": "fabsf", "floor": "floorf", "ceil": "ceilf", "round": "roundf", "sin": "sinf",
    "cos": "cosf", "tan": "tanf", "asin": "asinf", "acos": "acosf", "atan": "atanf",
    "sinh": "sinhf", "cosh": "coshf", "tanh": "tanhf", "asinh": "asinhf", "acosh": "acoshf",
    "atanh": "atanhf", "exp": "expf", "exp2": "exp2f", "log": "logf", "log1p": "log1pf",
    "expm1": "expm1f", "sqrt": "sqrtf", "rsqrt": "rsqrtf", "cbrt": "cbrtf", "erf": "erff",
}
_CPP_BINARY_F = {"max": "fmaxf", "min": "fminf", "atan2": "atan2f", "pow": "powf", "rem": "fmodf"}
_CPP_INFIX = {"add": "+", "sub": "-", "mul": "*", "div": "/", "eq": "==", "ne": "!=",
              "lt": "<", "le": "<=", "gt": ">", "ge": ">=", "and": "&&", "or": "||"}
_CTYPE = {"f": "float", "i": "int", "b": "bool"}
_INPUTS = ("x", "p", "w", "s", "t")
_REDUCE_INIT = {"add": 0, "mul": 1, "max": -np.inf, "min": np.inf, "and": True, "or": False}


def _reduce_init(op: str, kind: str):
    if kind == "i" and op in ("max", "min"):
        return -(2**31) if op == "max" else 2**31 - 1
    return _REDUCE_INIT[op]


class Graph:
    """Hash-consed expression graph. Nodes are created in topological order."""

    def __init__(self):
        self.nodes: list[Sym] = []
        self._table: dict[tuple, Sym] = {}
        self.tables: dict[int, np.ndarray] = {}   # table node idx -> values
        self._table_keys: dict[tuple, Sym] = {}

    def _append(self, op, args, kind, extra=None) -> Sym:
        node = Sym(op, args, kind, len(self.nodes), extra)
        self.nodes.append(node)
        return node

    def input(self, name: str, index: int, kind: str = "f") -> Sym:
        return self._make(name, (index,), kind)

    def index(self) -> Sym:
        """A fresh loop index."""
        return self._append("idx", (), "i")

    def table(self, values: np.ndarray, kind: str) -> Sym:
        """A constant table (namespace-scope device array), deduplicated by content."""
        values = np.ascontiguousarray(np.asarray(values, _NP_DTYPE[kind]).ravel())
        key = (kind, values.tobytes())
        node = self._table_keys.get(key)
        if node is None:
            node = self._table_keys[key] = self._append("table", (), kind)
            self.tables[node.idx] = values
        return node

    def tload(self, table: Sym, i):
        if not _is_sym(i):
            return _const(self.tables[table.idx][int(i)], table.kind)
        return self._make("tload", (table, i), table.kind)

    def bload(self, buf: Sym, i, kind: str):
        return self._make("bload", (buf, i if _is_sym(i) else np.int32(i)), kind)

    def loop(self, extent: int, idx: Sym, value, kind: str, reduce=None, target=None) -> Sym:
        """``dst[idx] = value`` for idx < extent; ``reduce=(r_idx, R, op)`` folds ``value``
        over an inner index first. ``target``: an output array name (default: scratch)."""
        args = (idx, value) + ((reduce[0],) if reduce else ())
        return self._append("loop", args, kind,
                            dict(extent=extent, reduce=reduce, target=target))

    def pack(self, values: list, kind: str) -> Sym:
        """A scratch buffer holding ``values`` (written by the group leader); identical packs
        are shared."""
        key = ("pack", kind, tuple(_key(v) for v in values))
        node = self._table.get(key)
        if node is None:
            node = self._table[key] = self._append("pack", tuple(values), kind,
                                                   dict(extent=len(values)))
        return node

    def _make(self, op: str, args: tuple, kind: str) -> Sym:
        key = (op, kind, tuple(a if isinstance(a, int) and op in _INPUTS else _key(a)
                               for a in args))
        node = self._table.get(key)
        if node is None:
            node = self._append(op, args, kind)
            self._table[key] = node
        return node

    # ── elementwise ops with folding ────────────────────────────────────────
    def op(self, name: str, args: Sequence, kind: str):
        if not any(_is_sym(a) for a in args):
            with np.errstate(all="ignore"):
                return _const(_FOLD[name](*args), kind)
        folded = self._fold(name, args, kind)
        if folded is not None:
            return folded
        if name in _COMMUTATIVE:
            args = tuple(sorted(args, key=lambda a: (not _is_sym(a), repr(_key(a)))))
        return self._make(name, tuple(args), kind)

    def _fold(self, name, args, kind):
        def is_c(a, v):
            return not _is_sym(a) and a == v

        if name == "add":
            a, b = args
            if is_c(a, 0):
                return self.cast(b, kind)
            if is_c(b, 0):
                return self.cast(a, kind)
        elif name == "sub":
            a, b = args
            if is_c(b, 0):
                return a
            if a is b:
                return _const(0, kind)
            if is_c(a, 0):
                return self.op("neg", (b,), kind)
        elif name == "mul":
            a, b = args
            if is_c(a, 0) or is_c(b, 0):
                return _const(0, kind)
            if is_c(a, 1):
                return b
            if is_c(b, 1):
                return a
            if is_c(a, -1):
                return self.op("neg", (b,), kind)
            if is_c(b, -1):
                return self.op("neg", (a,), kind)
        elif name == "div":
            a, b = args
            if is_c(b, 1):
                return a
            if is_c(a, 0) and kind == "f":
                return _const(0, kind)
        elif name == "neg":
            (a,) = args
            if _is_sym(a) and a.op == "neg":
                return a.args[0]
        elif name in ("max", "min"):
            if args[0] is args[1]:
                return args[0]
        elif name == "select":
            c, a, b = args
            if not _is_sym(c):
                return b if c else a
            if a is b or (not _is_sym(a) and not _is_sym(b) and _key(a) == _key(b)):
                return a
        elif name in ("and", "or"):
            a, b = args
            for x, y in ((a, b), (b, a)):
                if not _is_sym(x):
                    if name == "and":
                        return y if x else _const(False, "b")
                    return _const(True, "b") if x else y
        return None

    def cast(self, v, kind: str):
        if _is_sym(v):
            return v if v.kind == kind else self._make("cast", (v,), kind)
        return _const(v, kind)

    # ── index arithmetic (folds when concrete) ──────────────────────────────
    def iadd(self, a, b):
        return self.op("add", (a, np.int32(b) if not _is_sym(b) else b), "i")

    def imul(self, a, c: int):
        return self.op("mul", (a, np.int32(c)), "i")

    def idiv(self, a, c: int):
        return a if c == 1 else self.op("div", (a, np.int32(c)), "i")

    def irem(self, a, c: int):
        return self.op("rem", (a, np.int32(c)), "i")

    # ── C++ emission ────────────────────────────────────────────────────────
    def tables_source(self) -> str:
        """Namespace-scope declarations of every constant table."""
        out = []
        for idx, vals in self.tables.items():
            kind = self.nodes[idx].kind
            body = ", ".join(_lit(np.asarray(v), kind) for v in vals) or "0"
            out.append(f"__device__ const {_CTYPE[kind]} t{idx}[] = {{{body}}};")
        return "\n".join(out)

    def schedule(self, outputs: Sequence[tuple[str, Any]]) -> "Schedule":
        """Order the statements that write each ``(array_name, value)`` output.

        ``value`` is a scalarized object array or a :class:`Lazy`, written to
        ``array_name[0:size]`` row-major. The schedule is what both the C++ printer and
        the test interpreter execute.
        """
        roots, writes = [], []
        for name, value in outputs:
            if isinstance(value, Lazy):
                idx = self.index()
                roots.append(self.loop(value.size, idx, value.fn(idx), value.kind, target=name))
            else:
                for i, v in enumerate(np.asarray(value, dtype=object).flat):
                    writes.append((f"{name}[{i}]", v))
                    if _is_sym(v):
                        roots.append(v)

        live = np.zeros(len(self.nodes), bool)
        stack = list(roots)
        while stack:
            n = stack.pop()
            if live[n.idx]:
                continue
            live[n.idx] = True
            stack.extend(a for a in n.args if _is_sym(a))

        # Which loop indices each node depends on; a loop node hides its own indices.
        deps: dict[int, frozenset] = {}
        owner: dict[int, Sym] = {}          # index node idx -> its loop
        empty = frozenset()
        for n in self.nodes:
            if not live[n.idx]:
                continue
            if n.op == "idx":
                deps[n.idx] = frozenset((n.idx,))
                continue
            d = empty
            for a in n.args:
                if _is_sym(a):
                    da = deps[a.idx]
                    if da:
                        d = d | da
            if n.op == "loop":
                own = {n.args[0].idx} | ({n.extra["reduce"][0].idx} if n.extra["reduce"] else set())
                for i in own:
                    owner[i] = n
                d = d - own
                if d:
                    raise EmitError("a loop body depends on another loop's index")
            deps[n.idx] = d

        top: list[Sym] = []
        body: dict[int, list[Sym]] = {}
        for n in self.nodes:
            if not live[n.idx] or n.op in _INPUTS or n.op in ("idx", "table"):
                continue
            d = deps[n.idx]
            if not d:
                top.append(n)
            else:
                loop = owner[next(iter(d))]
                body.setdefault(loop.idx, []).append(n)

        # Scratch allocation: each buffer lives from its statement to its last reader.
        pos = {n.idx: t for t, n in enumerate(top)}
        last_use: dict[int, int] = {}
        for n in self.nodes:
            if live[n.idx] and n.op == "bload":
                d = deps[n.idx]
                t = pos[n.idx] if not d else pos[owner[next(iter(d))].idx]
                b = n.args[0].idx
                last_use[b] = max(last_use.get(b, -1), t)
        offsets: dict[int, int] = {}
        active: list[tuple[int, int, int]] = []   # (end, offset, size)
        free: list[tuple[int, int]] = []          # (offset, size)
        scratch = 0
        for t, n in enumerate(top):
            for item in [a for a in active if a[0] < t]:
                active.remove(item)
                free.append((item[1], item[2]))
            if n.op not in ("loop", "pack") or (n.op == "loop" and n.extra["target"]):
                continue
            size = n.extra["extent"]
            free.sort()
            for k, (off, fsize) in enumerate(free):
                if fsize >= size:
                    free[k] = (off + size, fsize - size)
                    break
            else:
                off = scratch
                scratch += size
            offsets[n.idx] = off
            active.append((last_use.get(n.idx, t), off, size))

        return Schedule(top, body, deps, offsets, writes, scratch)

    def emit_function(self, outputs: Sequence[tuple[str, Any]], sync: bool = True
                      ) -> tuple[str, int, int]:
        """Body of a tier-generic function writing ``outputs`` (see :meth:`schedule`).

        ``sync=False`` for lane-private functions (straight-line code one lane runs alone):
        no trailing group sync, which would deadlock when only some lanes call it.
        Returns ``(body, scratch_floats, max_loop_extent)``.
        """
        sch = self.schedule(outputs)
        top, body, offsets, writes = sch.top, sch.body, sch.offsets, sch.writes
        ctx = _EmitCtx(self, offsets, sch.deps)
        lines, max_extent = [], 1
        for n in top:
            if n.op == "loop":
                lines += ctx.loop(n, body.get(n.idx, []))
                max_extent = max(max_extent, n.extra["extent"])
            elif n.op == "pack":
                off = offsets[n.idx]
                stores = " ".join(f"scratch[{off + i}] = (float)({ctx.ref(v)});"
                                  for i, v in enumerate(n.args))
                lines.append(f"    if (rank == 0) {{ {stores} }}")
                lines.append("    PYROFFI_COST_SYNC();")
            else:
                lines.append(f"    const {_CTYPE[n.kind]} v{n.idx} = {ctx.expr(n)};")
        if writes:
            lines.append("    if (rank == 0) {")
            lines += [f"        {lhs} = {ctx.ref(v)};" for lhs, v in writes]
            lines.append("    }")
        if sync:
            lines.append("    PYROFFI_COST_SYNC();")
        return "\n".join(lines), sch.scratch, max_extent


class Schedule:
    """Top-level statements in order, each loop's body, dependency sets, scratch offsets,
    scalar output writes and scratch size."""

    def __init__(self, top, body, deps, offsets, writes, scratch):
        self.top, self.body, self.deps = top, body, deps
        self.offsets, self.writes, self.scratch = offsets, writes, scratch


def _lit(v, kind: str) -> str:
    if kind == "b":
        return "true" if bool(v) else "false"
    if kind == "i":
        return f"({int(v)})" if int(v) != -(2**31) else "(-2147483647 - 1)"
    f = float(np.float32(v))
    if math.isnan(f):
        return "NAN"
    if math.isinf(f):
        return "INFINITY" if f > 0 else "(-INFINITY)"
    return f"{f!r}f"


class _EmitCtx:
    """Prints nodes, given scratch offsets."""

    def __init__(self, g: Graph, offsets: dict[int, int], deps: dict[int, frozenset]):
        self.g, self.offsets, self.deps = g, offsets, deps

    def ref(self, v) -> str:
        if not _is_sym(v):
            return _lit(v, _kind(v.dtype))
        if v.op == "x":
            return f"x[{v.args[0]}]"
        if v.op in ("w", "s", "t"):
            return f"{v.op}[{v.args[0]}]"
        if v.op == "p":
            return {"f": f"p[{v.args[0]}]", "i": f"((int)p[{v.args[0]}])",
                    "b": f"(p[{v.args[0]}] != 0.0f)"}[v.kind]
        return f"v{v.idx}"

    def loop(self, n: Sym, members: list[Sym]) -> list[str]:
        idx, value = n.args[0], n.args[1]
        red = n.extra["reduce"]
        inner_idx = red[0].idx if red else None
        outer = [m for m in members if inner_idx not in self.deps[m.idx]]
        inner = [m for m in members if inner_idx in self.deps[m.idx]]
        out = [f"    for (int v{idx.idx} = rank; v{idx.idx} < {n.extra['extent']}; "
               f"v{idx.idx} += size) {{"]
        out += [f"        const {_CTYPE[m.kind]} v{m.idx} = {self.expr(m)};" for m in outer]
        if red:
            r_idx, extent, op = red
            init = _lit(np.asarray(_reduce_init(op, n.kind)), n.kind)
            out.append(f"        {_CTYPE[n.kind]} acc = {init};")
            out.append(f"        for (int v{r_idx.idx} = 0; v{r_idx.idx} < {extent}; ++v{r_idx.idx}) {{")
            out += [f"            const {_CTYPE[m.kind]} v{m.idx} = {self.expr(m)};" for m in inner]
            out.append(f"            acc = {_format(op, n.kind, ['acc', self.ref(value)])};")
            out.append("        }")
            val = "acc"
        else:
            val = self.ref(value)
        dst = n.extra["target"] or f"scratch + {self.offsets[n.idx]}"
        out.append(f"        ({dst})[v{idx.idx}] = (float)({val});")
        out.append("    }")
        out.append("    PYROFFI_COST_SYNC();")
        return out

    def expr(self, n: Sym) -> str:
        if n.op == "tload":
            return f"t{n.args[0].idx}[{self.ref(n.args[1])}]"
        if n.op == "bload":
            buf, i = n.args
            base = f"scratch[{self.offsets[buf.idx]} + ({self.ref(i)})]"
            return {"f": base, "i": f"((int){base})", "b": f"({base} != 0.0f)"}[n.kind]
        a = [self.ref(x) for x in n.args]
        return _format(n.op, n.kind, a)


def _format(op: str, k: str, a: list[str]) -> str:
    if op == "cast":
        return f"({_CTYPE[k]})({a[0]})"
    if op == "select":
        return f"({a[0]} ? {a[2]} : {a[1]})"
    if op == "neg":
        return f"-({a[0]})"
    if op == "not":
        return f"!({a[0]})" if k == "b" else f"~({a[0]})"
    if op == "xor":
        return f"({a[0]} != {a[1]})" if k == "b" else f"({a[0]} ^ {a[1]})"
    if op == "square":
        return f"({a[0]} * {a[0]})"
    if op == "sign":
        return f"(({a[0]} > 0) - ({a[0]} < 0))"
    if op == "logistic":
        return f"(1.0f / (1.0f + expf(-({a[0]}))))"
    if op == "is_finite":
        return f"isfinite({a[0]})"
    if op in _CPP_INFIX:
        return f"({a[0]} {_CPP_INFIX[op]} {a[1]})"
    if k in ("i", "b") and op in ("max", "min", "abs"):
        return f"{op}({', '.join(a)})"
    if k == "i" and op == "rem":
        return f"({a[0]} % {a[1]})"
    if op in _CPP_UNARY_F:
        return f"{_CPP_UNARY_F[op]}({a[0]})"
    if op in _CPP_BINARY_F:
        return f"{_CPP_BINARY_F[op]}({a[0]}, {a[1]})"
    raise EmitError(f"no C++ form for {op!r}")


# ── values ──────────────────────────────────────────────────────────────────

class Lazy:
    """A large array as an element function over a flat row-major index."""

    __slots__ = ("shape", "kind", "fn", "cost", "buffered")

    def __init__(self, shape, kind: str, fn: Callable, cost: int = 1, buffered: bool = False):
        self.shape, self.kind, self.fn, self.cost, self.buffered = (
            tuple(shape), kind, fn, cost, buffered)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=int))


def _obj(arr) -> np.ndarray:
    arr = np.asarray(arr)
    out = np.empty(arr.shape, dtype=object)
    for i, v in enumerate(arr.ravel()):
        out.flat[i] = v
    return out


def _all_const(a) -> bool:
    return isinstance(a, np.ndarray) and not any(_is_sym(v) for v in a.flat)


def _typed(a: np.ndarray, dtype) -> np.ndarray:
    return np.array([np.asarray(v).item() for v in a.flat], dtype=dtype).reshape(a.shape)


def _size(shape) -> int:
    return int(np.prod(shape, dtype=int))


def _strides(shape) -> list[int]:
    out, s = [], 1
    for d in reversed(shape):
        out.append(s)
        s *= d
    return out[::-1]


_ELEMENTWISE = set(_FOLD) - {"select"} | {"select_n", "convert_element_type", "integer_pow",
                                          "clamp", "add_any", "copy", "copy_p",
                                          "reduce_precision", "stop_gradient"}
_INDEX_OPERANDS = {"gather": (1,), "dynamic_slice": None, "dynamic_update_slice": None,
                   "scatter": (1,)}
_MOVEMENT = {"reshape", "broadcast_in_dim", "transpose", "squeeze", "expand_dims", "slice",
             "rev", "concatenate", "pad", "split", "stack", "gather", "dynamic_slice",
             "dynamic_update_slice", "scatter"}
_REDUCE = {"reduce_sum": "add", "reduce_max": "max", "reduce_min": "min",
           "reduce_prod": "mul", "reduce_and": "and", "reduce_or": "or"}
_CUMULATIVE = {"cumsum": "add", "cumprod": "mul", "cummax": "max", "cummin": "min"}
_CALLS = ("jit", "pjit", "closed_call", "core_call", "remat", "checkpoint",
          "custom_jvp_call", "custom_vjp_call", "custom_vjp_call_jaxpr")
_LAZY_OPS = _ELEMENTWISE | _MOVEMENT | {"dot_general"} | set(_REDUCE) | {"argmax", "argmin"}


class Evaluator:
    """Evaluates a jaxpr over symbolic values, building ``graph``."""

    def __init__(self, graph: Graph, loop_min: float = LOOP_MIN):
        """``loop_min``: arrays at least this large become loops (``inf``: fully scalarized,
        the thread tier's per-thread form)."""
        self.g = graph
        self.loop_min = loop_min
        self._probes: dict[int, Sym] = {}
        self._loops: dict[tuple, Sym] = {}

    def run(self, closed: jcore.ClosedJaxpr, args: Sequence) -> list:
        # Constant folding and data-movement probes bind primitives eagerly (also when called
        # inside a caller's jax.jit trace); keep them off the GPU.
        with jax.ensure_compile_time_eval(), jax.default_device(jax.devices("cpu")[0]):
            return self._jaxpr(closed.jaxpr, [_obj(c) for c in closed.consts], list(args))

    # ── conversions ─────────────────────────────────────────────────────────
    def as_obj(self, v) -> np.ndarray:
        """Every element as its own node (for small arrays and outputs)."""
        if not isinstance(v, Lazy):
            return v
        out = np.empty(v.shape, dtype=object)
        for k in range(out.size):
            out.flat[k] = v.fn(k)
        return out

    def as_lazy(self, v, kind: str) -> Lazy:
        if isinstance(v, Lazy):
            return v
        arr, g = v, self.g
        if arr.size == 1:
            s = arr.flat[0]
            return Lazy(arr.shape, kind, lambda i: s)
        if _all_const(arr):
            t = g.table(_typed(arr, _NP_DTYPE[kind]), kind)
            return Lazy(arr.shape, kind, lambda i: arr.flat[int(i)] if not _is_sym(i) else g.tload(t, i))
        buf = g.pack([g.cast(e, kind) if _is_sym(e) else e for e in arr.flat], kind)
        return Lazy(arr.shape, kind,
                    lambda i: arr.flat[int(i)] if not _is_sym(i) else g.bload(buf, i, kind),
                    buffered=True)

    def materialize(self, lz: Lazy, force: bool = False) -> Lazy:
        if lz.buffered or (not force and lz.size > MATERIALIZE_MAX):
            return lz
        g, kind = self.g, lz.kind
        # Identical arrays (e.g. one robot's kinematics, traced once per residual) share one
        # buffer: key the loop by its body evaluated at a probe index shared per extent.
        probe = self._probes.setdefault(lz.size, g.index())
        key = (lz.size, kind, _key(lz.fn(probe)))
        buf = self._loops.get(key)
        if buf is None:
            idx = g.index()
            buf = self._loops[key] = g.loop(lz.size, idx, lz.fn(idx), kind)
        return Lazy(lz.shape, kind, lambda i: g.bload(buf, i, kind), buffered=True)

    def _out(self, v, shape):
        """Scalarize results small enough to keep folding."""
        if isinstance(v, Lazy) and _size(shape) < self.loop_min:
            if not v.buffered and v.cost * v.size > SCALARIZE_BUDGET:
                v = self.materialize(v, force=True)
            return self.as_obj(v)
        return v

    def _reusable(self, lz: Lazy) -> bool:
        """Whether an expensive lazy array should be stored rather than recomputed."""
        return not lz.buffered and lz.cost >= FANOUT_COST and lz.size <= MATERIALIZE_MAX

    # ── jaxpr walk ──────────────────────────────────────────────────────────
    def _jaxpr(self, jaxpr, consts, args):
        env: dict = {}
        uses: dict = {}
        for eqn in jaxpr.eqns:
            for v in eqn.invars:
                if not isinstance(v, jcore.Literal):
                    uses[v] = uses.get(v, 0) + 1
        for v in jaxpr.outvars:
            if not isinstance(v, jcore.Literal):
                uses[v] = uses.get(v, 0) + 1

        def read(v):
            if isinstance(v, jcore.Literal):
                return _obj(np.asarray(v.val, dtype=v.aval.dtype))
            x = env[v]
            if isinstance(x, Lazy) and self._reusable(x) and uses.get(v, 0) > 1:
                x = env[v] = self.materialize(x)
            return x

        for v, x in zip(jaxpr.constvars, consts):
            env[v] = x
        for v, x in zip(jaxpr.invars, args):
            env[v] = x
        for eqn in jaxpr.eqns:
            outs = self._eqn(eqn, [read(v) for v in eqn.invars])
            for v, o in zip(eqn.outvars, outs):
                if tuple(o.shape) != tuple(v.aval.shape):
                    raise EmitError(f"internal: {eqn.primitive.name} produced shape {o.shape}, "
                                    f"expected {v.aval.shape}")
                if not isinstance(v, jcore.DropVar):
                    env[v] = o
        return [read(v) for v in jaxpr.outvars]

    def _eqn(self, eqn, ins: list) -> list:
        name = eqn.primitive.name
        if name in _CALLS:
            sub = next(eqn.params[k] for k in ("jaxpr", "call_jaxpr", "fun_jaxpr") if k in eqn.params)
            if isinstance(sub, jcore.ClosedJaxpr):
                return self._jaxpr(sub.jaxpr, [_obj(c) for c in sub.consts], ins)
            return self._jaxpr(sub, [], ins)
        if name == "scan":
            return self._scan(eqn, ins)
        if name == "while":
            return self._while(eqn, ins)
        if name == "cond":
            return self._cond(eqn, ins)
        if all(_all_const(a) for a in ins):
            return self._const_eval(eqn, ins)
        big = max([_size(v.aval.shape) for v in eqn.outvars] + [0]) >= self.loop_min
        if name in _LAZY_OPS and (big or any(isinstance(a, Lazy) for a in ins)):
            outs = self._lazy_eqn(eqn, ins)
            return [self._out(o, v.aval.shape) for o, v in zip(outs, eqn.outvars)]
        ins = [self.as_obj(a) for a in ins]
        if name in _ELEMENTWISE:
            return [self._elementwise(eqn, ins)]
        if name in _MOVEMENT:
            return self._move(eqn, ins)
        if name == "dot_general":
            return [self._dot(eqn, *ins)]
        if name in _REDUCE:
            return [self._reduce(eqn, ins[0], _REDUCE[name])]
        if name in ("argmax", "argmin"):
            return [self._arg(eqn, ins[0], name)]
        if name in _CUMULATIVE:
            return [self._cumulative(eqn, ins[0], _CUMULATIVE[name])]
        from jax._src import source_info_util

        raise EmitError(
            f"primitive {name!r} on symbolic values cannot be compiled "
            f"(at {source_info_util.summarize(eqn.source_info)})")

    def _const_eval(self, eqn, ins):
        vals = [_typed(a, v.aval.dtype) for a, v in zip(ins, eqn.invars)]
        outs = eqn.primitive.bind(*vals, **eqn.params)
        if not eqn.primitive.multiple_results:
            outs = [outs]
        return [_obj(np.asarray(o)) for o in outs]

    # ── scalar forms ────────────────────────────────────────────────────────
    def _ew(self, name, params, a: list, kind: str):
        """One element of an elementwise op."""
        g = self.g
        if name == "add_any":
            return g.op("add", a, kind)
        if name in ("copy", "copy_p", "reduce_precision", "stop_gradient"):
            return a[0]
        if name == "convert_element_type":
            return g.cast(a[0], kind)
        if name == "integer_pow":
            return self._ipow(a[0], params["y"], kind)
        if name == "clamp":
            lo, x, hi = a
            return g.op("min", (g.op("max", (x, lo), kind), hi), kind)
        if name == "select_n":
            which = a[0]
            if len(a) == 3:
                return g.op("select", (g.cast(which, "b") if _is_sym(which) else np.bool_(which),
                                       a[1], a[2]), kind)
            v = a[1]
            for c in range(1, len(a) - 1):
                v = g.op("select", (g.op("eq", (which, np.int32(c)), "b"), v, a[c + 1]), kind)
            return v
        if name in ("pow", "atan2", "rem", "max", "min", "add", "sub", "mul", "div") and kind == "f":
            a = [g.cast(x, "f") for x in a]
        return g.op(name, a, kind)

    def _elementwise(self, eqn, ins):
        kind = _kind(eqn.outvars[0].aval.dtype)
        shape = eqn.outvars[0].aval.shape
        ins = [np.broadcast_to(a, shape) for a in ins]
        out = np.empty(shape, dtype=object)
        for i in range(out.size):
            out.flat[i] = self._ew(eqn.primitive.name, eqn.params, [x.flat[i] for x in ins], kind)
        return out

    def _ipow(self, x, y: int, kind):
        g = self.g
        if y == 0:
            return _const(1, kind)
        base, n, acc = x, abs(y), None
        while n:
            if n & 1:
                acc = base if acc is None else g.op("mul", (acc, base), kind)
            n >>= 1
            if n:
                base = g.op("mul", (base, base), kind)
        return acc if y > 0 else g.op("div", (_const(1, kind), acc), kind)

    def _codes(self, eqn, ins):
        """Run a movement primitive on integer codes naming its data inputs' elements.

        Returns ``(bases, run, sym)``: each data input's first code, a function running the
        primitive for given index-operand values, and the one symbolic index (or None).
        """
        name = eqn.primitive.name
        idx_pos = _INDEX_OPERANDS.get(name, ())
        if idx_pos is None:  # dynamic_slice / dynamic_update_slice: trailing scalar starts
            idx_pos = tuple(range(1 if name == "dynamic_slice" else 2, len(ins)))
        bases, codes, n = {}, [], 0
        for k, a in enumerate(ins):
            if k in idx_pos:
                codes.append(None)
                continue
            size = a.size if isinstance(a, (np.ndarray, Lazy)) else 1
            shape = a.shape
            bases[k] = n
            codes.append(np.arange(n, n + size, dtype=np.int32).reshape(shape))
            n += size
        sym = []
        for k in idx_pos:
            if isinstance(ins[k], Lazy):
                raise EmitError(f"{name} with a large runtime index array is not supported")
            sym += [(k, j) for j, v in enumerate(ins[k].flat) if _is_sym(v)]
        const_idx = {k: _typed(ins[k], eqn.invars[k].aval.dtype) for k in idx_pos
                     if not any(kk == k for kk, _ in sym)}

        def run(index_vals: dict) -> list[np.ndarray]:
            args = [index_vals[k] if k in idx_pos else codes[k] for k in range(len(ins))]
            outs = eqn.primitive.bind(*args, **eqn.params)
            if not eqn.primitive.multiple_results:
                outs = [outs]
            return [np.asarray(o) for o in outs]

        if len(sym) > 1 or (sym and name == "scatter"):
            raise EmitError(f"{name} with more than one runtime index is not supported")
        return bases, n, run, const_idx, (sym[0] if sym else None)

    def _move(self, eqn, ins):
        """Scalarized data movement."""
        bases, n, run, const_idx, sym = self._codes(eqn, ins)
        pool = [None] * n
        for k, base in bases.items():
            for j, v in enumerate(ins[k].flat):
                pool[base + j] = v
        if sym is None:
            return [self._pick(pool, o) for o in run(const_idx)]
        k, j = sym
        s = ins[k].flat[j]
        count = self._index_range(eqn, ins, k, j)
        variants = []
        for v in range(count):
            idx = _typed(np.where(np.vectorize(_is_sym, otypes=[bool])(ins[k]),
                                  np.int32(v), ins[k]), eqn.invars[k].aval.dtype)
            variants.append(run({**const_idx, k: idx}))
        g = self.g
        sc = self._clamp_index(s, count)
        outs = []
        for o in range(len(variants[0])):
            picked = [self._pick(pool, var[o]) for var in variants]
            kind = _kind(eqn.outvars[o].aval.dtype)
            out = np.empty(picked[0].shape, dtype=object)
            for i in range(out.size):
                val = picked[0].flat[i]
                for v in range(1, count):
                    val = g.op("select", (g.op("eq", (sc, np.int32(v)), "b"), val, picked[v].flat[i]), kind)
                out.flat[i] = val
            outs.append(out)
        return outs

    def _clamp_index(self, s, count):
        g = self.g
        return g.op("min", (g.op("max", (g.cast(s, "i"), np.int32(0)), "i"), np.int32(count - 1)), "i")

    def _index_range(self, eqn, ins, k, j) -> int:
        shape = ins[0].shape
        if eqn.primitive.name == "gather":
            dn = eqn.params["dimension_numbers"]
            axis = dn.start_index_map[j % len(dn.start_index_map)]
            return shape[axis]
        axis = k - (1 if eqn.primitive.name == "dynamic_slice" else 2)
        size = (eqn.params["slice_sizes"][axis] if eqn.primitive.name == "dynamic_slice"
                else ins[1].shape[axis])
        return shape[axis] - size + 1

    @staticmethod
    def _pick(pool, code: np.ndarray) -> np.ndarray:
        out = np.empty(code.shape, dtype=object)
        for i, c in enumerate(code.ravel()):
            if not 0 <= c < len(pool):
                raise EmitError("data movement produced an out-of-bounds element (fill mode)")
            out.flat[i] = pool[c]
        return out

    def _dot(self, eqn, lhs, rhs):
        (lc, rc), (lb, rb) = eqn.params["dimension_numbers"]
        kind = _kind(eqn.outvars[0].aval.dtype)
        lf = [d for d in range(lhs.ndim) if d not in lc and d not in lb]
        rf = [d for d in range(rhs.ndim) if d not in rc and d not in rb]
        L = np.transpose(lhs, (*lb, *lf, *lc))
        R = np.transpose(rhs, (*rb, *rf, *rc))
        nb = _size([lhs.shape[d] for d in lb])
        nl = _size([lhs.shape[d] for d in lf])
        nr = _size([rhs.shape[d] for d in rf])
        nk = _size([lhs.shape[d] for d in lc])
        L = L.reshape(nb, nl, nk)
        R = R.reshape(nb, nr, nk)
        out = np.empty((nb, nl, nr), dtype=object)
        g = self.g
        for b in range(nb):
            for i in range(nl):
                for j in range(nr):
                    acc = _const(0, kind)
                    for c in range(nk):
                        acc = g.op("add", (acc, g.op("mul", (g.cast(L[b, i, c], kind),
                                                            g.cast(R[b, j, c], kind)), kind)), kind)
                    out[b, i, j] = acc
        return out.reshape(eqn.outvars[0].aval.shape)

    def _reduce(self, eqn, x, op):
        axes = tuple(eqn.params["axes"])
        kind = _kind(eqn.outvars[0].aval.dtype)
        keep = [d for d in range(x.ndim) if d not in axes]
        xt = np.transpose(x, (*keep, *axes)).reshape(_size([x.shape[d] for d in keep]), -1)
        out = np.empty(xt.shape[0], dtype=object)
        for i in range(xt.shape[0]):
            acc = xt[i, 0]
            for v in xt[i, 1:]:
                acc = self.g.op(op, (acc, v), kind)
            out[i] = acc
        return out.reshape(eqn.outvars[0].aval.shape)

    def _arg(self, eqn, x, name):
        (axis,) = eqn.params["axes"]
        xt = np.moveaxis(x, axis, -1)
        out = np.empty(xt.shape[:-1], dtype=object)
        cmp = "gt" if name == "argmax" else "lt"
        g = self.g
        for i in np.ndindex(out.shape):
            best, bi = xt[i][0], np.int32(0)
            for j, v in enumerate(xt[i][1:], start=1):
                better = g.op(cmp, (v, best), "b")
                best = g.op("select", (better, best, v), _kind(eqn.invars[0].aval.dtype))
                bi = g.op("select", (better, bi, np.int32(j)), "i")
            out[i] = bi
        return out

    def _cumulative(self, eqn, x, op):
        axis, reverse = eqn.params["axis"], eqn.params["reverse"]
        kind = _kind(eqn.outvars[0].aval.dtype)
        xt = np.moveaxis(x, axis, -1).copy()
        n = xt.shape[-1]
        order = range(n - 1, -1, -1) if reverse else range(n)
        for i in np.ndindex(xt.shape[:-1]):
            acc = None
            for j in order:
                acc = xt[i][j] if acc is None else self.g.op(op, (acc, xt[i][j]), kind)
                xt[i + (j,)] = acc
        return np.moveaxis(xt, -1, axis)

    # ── lazy forms ──────────────────────────────────────────────────────────
    def _lazy_eqn(self, eqn, ins) -> list:
        name = eqn.primitive.name
        if name in _ELEMENTWISE:
            return [self._lazy_ew(eqn, ins)]
        if name in _MOVEMENT:
            return self._lazy_move(eqn, ins)
        if name == "dot_general":
            return [self._lazy_dot(eqn, *ins)]
        if name in _REDUCE:
            return [self._lazy_reduce(eqn, ins[0], _REDUCE[name])]
        return [self._lazy_arg(eqn, ins[0], name)]

    def _lazy_ew(self, eqn, ins) -> Lazy:
        kind = _kind(eqn.outvars[0].aval.dtype)
        shape = eqn.outvars[0].aval.shape
        size = _size(shape)
        args, cost = [], 1
        for a, v in zip(ins, eqn.invars):
            if not isinstance(a, Lazy) and a.size == 1:
                args.append(a.flat[0])           # broadcast scalar
                continue
            lz = self.as_lazy(a, _kind(v.aval.dtype))
            if tuple(lz.shape) != tuple(shape):
                # Size-1 dimensions broadcast (numpy rules; binary ops allow it).
                if lz.cost >= FANOUT_COST:
                    lz = self.materialize(lz)
                lz = self._broadcast_lazy(lz, shape)
            args.append(lz)
            cost += lz.cost
        name, params = eqn.primitive.name, eqn.params

        def fn(i):
            return self._ew(name, params, [a.fn(i) if isinstance(a, Lazy) else a for a in args], kind)

        return Lazy(shape, kind, fn, cost)

    def _broadcast_lazy(self, lz: Lazy, shape) -> Lazy:
        src_shape = (1,) * (len(shape) - len(lz.shape)) + tuple(lz.shape)
        if any(s not in (1, d) for s, d in zip(src_shape, shape)):
            raise EmitError(f"cannot broadcast {lz.shape} to {tuple(shape)}")
        g = self.g

        def fn(i):
            coords = [c if s != 1 else np.int32(0)
                      for c, s in zip(self._unravel(i, shape), src_shape)]
            return lz.fn(self._ravel(coords, src_shape))

        return Lazy(shape, lz.kind, fn, lz.cost + 1)

    def _lazy_move(self, eqn, ins) -> list:
        g = self.g
        bases, n, run, const_idx, sym = self._codes(eqn, ins)
        ks = sorted(bases)
        base_arr = np.array([bases[k] for k in ks])

        def lazy_input(k):
            a = ins[k]
            return self.as_lazy(a, _kind(eqn.invars[k].aval.dtype))

        if sym is not None:
            if len(ks) != 1:
                raise EmitError(f"{eqn.primitive.name} with a runtime index and several inputs")
            k, j = sym
            count = self._index_range(eqn, ins, k, j)
            variants = []
            for v in range(count):
                idx = _typed(np.where(np.vectorize(_is_sym, otypes=[bool])(ins[k]),
                                      np.int32(v), ins[k]), eqn.invars[k].aval.dtype)
                variants.append(run({**const_idx, k: idx}))
            sc = self._clamp_index(ins[k].flat[j], count)
            src = lazy_input(ks[0])
            outs = []
            for o in range(len(variants[0])):
                maps = np.stack([var[o].ravel() for var in variants]) - bases[ks[0]]
                if maps.min() < 0 or maps.max() >= src.size:
                    raise EmitError("data movement produced an out-of-bounds element (fill mode)")
                shape = variants[0][o].shape
                size = _size(shape)
                t = g.table(maps.ravel(), "i")
                src_o = self.materialize(src) if src.cost >= FANOUT_COST else src
                outs.append(Lazy(shape, src.kind, (lambda s, t, size: lambda i: s.fn(
                    g.tload(t, g.iadd(g.imul(sc, size), i))))(src_o, t, size), src_o.cost + 1))
            return outs

        outs = []
        for code in run(const_idx):
            shape = code.shape
            flat = code.ravel()
            if flat.size and (flat.min() < 0 or flat.max() >= n):
                raise EmitError("data movement produced an out-of-bounds element (fill mode)")
            which = np.searchsorted(base_arr, flat, side="right") - 1 if flat.size else flat
            used = sorted(set(which.tolist()))
            srcs = {w: lazy_input(ks[w]) for w in used}
            if len(used) == 1:
                w = used[0]
                local = flat - base_arr[w]
                src = srcs[w]
                if np.array_equal(local, np.arange(local.size) + (local[0] if local.size else 0)):
                    off = int(local[0]) if local.size else 0
                    fn = (lambda s, off: lambda i: s.fn(g.iadd(i, off)))(src, off)
                    outs.append(Lazy(shape, src.kind, fn, src.cost))
                    continue
                if src.cost >= FANOUT_COST and len(np.unique(local)) < local.size:
                    src = self.materialize(src)
                t = g.table(local, "i")
                outs.append(Lazy(shape, src.kind, (lambda s, t: lambda i: s.fn(g.tload(t, i)))(src, t),
                                 src.cost + 1))
                continue
            # Several inputs: select by a per-element source table, each input read at a
            # clamped index (so the unselected reads stay in bounds).
            kind = _kind(eqn.outvars[0].aval.dtype)
            sel = g.table(which, "i")
            parts = []
            for w in used:
                src = srcs[w]
                if src.cost >= FANOUT_COST:
                    src = self.materialize(src)
                local = np.where(which == w, flat - base_arr[w], 0)
                parts.append((w, src, g.table(local, "i")))

            def fn(i, parts=parts, sel=sel, kind=kind):
                s = g.tload(sel, i)
                val = parts[0][1].fn(g.tload(parts[0][2], i))
                for w, src, t in parts[1:]:
                    val = g.op("select", (g.op("eq", (s, np.int32(w)), "b"), val,
                                          src.fn(g.tload(t, i))), kind)
                return val

            outs.append(Lazy(shape, kind, fn, sum(p[1].cost for p in parts) + len(parts)))
        return outs

    def _unravel(self, i, shape) -> list:
        g, coords = self.g, []
        for d, s in zip(shape, _strides(shape)):
            coords.append(g.irem(g.idiv(i, s), d) if d > 1 else np.int32(0))
        return coords

    def _ravel(self, coords, shape):
        g, acc = self.g, np.int32(0)
        for c, s in zip(coords, _strides(shape)):
            acc = g.iadd(acc, g.imul(c, s))
        return acc

    def _lazy_dot(self, eqn, lhs, rhs) -> Lazy:
        (lc, rc), (lb, rb) = eqn.params["dimension_numbers"]
        kind = _kind(eqn.outvars[0].aval.dtype)
        L = self.as_lazy(lhs, _kind(eqn.invars[0].aval.dtype))
        R = self.as_lazy(rhs, _kind(eqn.invars[1].aval.dtype))
        lf = [d for d in range(len(L.shape)) if d not in lc and d not in lb]
        rf = [d for d in range(len(R.shape)) if d not in rc and d not in rb]
        bshape = [L.shape[d] for d in lb]
        lfshape = [L.shape[d] for d in lf]
        rfshape = [R.shape[d] for d in rf]
        cshape = [L.shape[d] for d in lc]
        nl, nr, nk = _size(lfshape), _size(rfshape), _size(cshape)
        if L.cost >= FANOUT_COST and nr > 1:
            L = self.materialize(L)
        if R.cost >= FANOUT_COST and nl > 1:
            R = self.materialize(R)
        g = self.g
        out_shape = eqn.outvars[0].aval.shape

        def index(o, c, X, is_lhs, batch_dims, free_dims, contract_dims, free_shape):
            b = g.idiv(o, nl * nr)
            f = g.irem(g.idiv(o, nr), nl) if is_lhs else g.irem(o, nr)
            coords = [np.int32(0)] * len(X.shape)
            for d, v in zip(batch_dims, self._unravel(b, bshape)):
                coords[d] = v
            for d, v in zip(free_dims, self._unravel(f, free_shape)):
                coords[d] = v
            for d, v in zip(contract_dims, self._unravel(c, cshape)):
                coords[d] = v
            return self._ravel(coords, X.shape)

        def term(o, c):
            a = L.fn(index(o, c, L, True, lb, lf, lc, lfshape))
            b = R.fn(index(o, c, R, False, rb, rf, rc, rfshape))
            return g.op("mul", (g.cast(a, kind), g.cast(b, kind)), kind)

        if nk <= UNROLL_MAX:
            def fn(o):
                acc = _const(0, kind)
                for c in range(nk):
                    acc = g.op("add", (acc, term(o, c)), kind)
                return acc

            return Lazy(out_shape, kind, fn, (L.cost + R.cost + 2) * nk)
        o, r = g.index(), g.index()
        buf = g.loop(_size(out_shape), o, term(o, r), kind, reduce=(r, nk, "add"))
        return Lazy(out_shape, kind, lambda i: g.bload(buf, i, kind), buffered=True)

    def _lazy_reduce(self, eqn, x, op) -> Lazy:
        axes = tuple(eqn.params["axes"])
        kind = _kind(eqn.outvars[0].aval.dtype)
        X = self.as_lazy(x, _kind(eqn.invars[0].aval.dtype))
        keep = [d for d in range(len(X.shape)) if d not in axes]
        kshape = [X.shape[d] for d in keep]
        rshape = [X.shape[d] for d in axes]
        R = _size(rshape)
        g = self.g
        out_shape = eqn.outvars[0].aval.shape

        def index(k, r):
            coords = [np.int32(0)] * len(X.shape)
            for d, v in zip(keep, self._unravel(k, kshape)):
                coords[d] = v
            for d, v in zip(axes, self._unravel(r, rshape)):
                coords[d] = v
            return self._ravel(coords, X.shape)

        if R <= UNROLL_MAX:
            def fn(k):
                acc = X.fn(index(k, 0))
                for r in range(1, R):
                    acc = g.op(op, (acc, X.fn(index(k, r))), kind)
                return acc

            return Lazy(out_shape, kind, fn, (X.cost + 1) * R)
        k, r = g.index(), g.index()
        buf = g.loop(_size(out_shape), k, X.fn(index(k, r)), kind, reduce=(r, R, op))
        return Lazy(out_shape, kind, lambda i: g.bload(buf, i, kind), buffered=True)

    def _lazy_arg(self, eqn, x, name) -> Lazy:
        (axis,) = eqn.params["axes"]
        X = self.as_lazy(x, _kind(eqn.invars[0].aval.dtype))
        R = X.shape[axis]
        if R > 64:
            raise EmitError(f"{name} over more than 64 elements is not supported")
        if X.cost >= FANOUT_COST:
            X = self.materialize(X)
        keep = [d for d in range(len(X.shape)) if d != axis]
        kshape = [X.shape[d] for d in keep]
        g = self.g
        cmp = "gt" if name == "argmax" else "lt"
        vkind = X.kind

        def fn(k):
            base = self._unravel(k, kshape)

            def at(j):
                coords = [np.int32(0)] * len(X.shape)
                for d, v in zip(keep, base):
                    coords[d] = v
                coords[axis] = np.int32(j)
                return X.fn(self._ravel(coords, X.shape))

            best, bi = at(0), np.int32(0)
            for j in range(1, R):
                v = at(j)
                better = g.op(cmp, (v, best), "b")
                best = g.op("select", (better, best, v), vkind)
                bi = g.op("select", (better, bi, np.int32(j)), "i")
            return bi

        return Lazy(eqn.outvars[0].aval.shape, "i", fn, (X.cost + 3) * R)

    # ── control flow ────────────────────────────────────────────────────────
    def _step(self, x, t):
        """x[t, ...] for an object array or lazy array."""
        if not isinstance(x, Lazy):
            return x[t, ...]
        m = _size(x.shape[1:])
        g = self.g
        lz = Lazy(x.shape[1:], x.kind, lambda i: x.fn(g.iadd(i, t * m)), x.cost)
        return self._out(lz, lz.shape)

    def _stack(self, items: list, kind: str):
        if not any(isinstance(y, Lazy) for y in items):
            return np.stack(items)
        g = self.g
        parts = [self.as_lazy(y, kind) for y in items]
        parts = [self.materialize(p) if p.cost >= FANOUT_COST else p for p in parts]
        m = parts[0].size

        def fn(i):
            if not _is_sym(i):
                return parts[int(i) // m].fn(int(i) % m)
            step, local = g.idiv(i, m), g.irem(i, m)
            val = parts[0].fn(local)
            for t in range(1, len(parts)):
                val = g.op("select", (g.op("eq", (step, np.int32(t)), "b"), val,
                                      parts[t].fn(local)), kind)
            return val

        return Lazy((len(parts), *parts[0].shape), kind, fn, sum(p.cost for p in parts))

    def _scan(self, eqn, ins):
        p = eqn.params
        body, n, nc, ncar = p["jaxpr"], p["length"], p["num_consts"], p["num_carry"]
        consts, carry, xs = ins[:nc], ins[nc:nc + ncar], ins[nc + ncar:]
        ys_steps = []
        for t in (range(n - 1, -1, -1) if p["reverse"] else range(n)):
            outs = self._jaxpr(body.jaxpr, [_obj(c) for c in body.consts],
                               [*consts, *carry, *(self._step(x, t) for x in xs)])
            carry, ys = outs[:ncar], outs[ncar:]
            ys_steps.append(ys)
        if p["reverse"]:
            ys_steps.reverse()
        ys = []
        for k in range(len(eqn.outvars) - ncar):
            var = eqn.outvars[ncar + k]
            if n == 0:
                ys.append(np.empty(var.aval.shape, dtype=object))
            else:
                ys.append(self._out(self._stack([s[k] for s in ys_steps], _kind(var.aval.dtype)),
                                    var.aval.shape))
        return [*carry, *ys]

    def _while(self, eqn, ins):
        p = eqn.params
        cj, bj, cn, bn = p["cond_jaxpr"], p["body_jaxpr"], p["cond_nconsts"], p["body_nconsts"]
        cconsts, bconsts, carry = ins[:cn], ins[cn:cn + bn], ins[cn + bn:]
        for _ in range(100_000):
            (pred,) = self._jaxpr(cj.jaxpr, [_obj(c) for c in cj.consts], [*cconsts, *carry])
            pv = self.as_obj(pred).flat[0]
            if _is_sym(pv):
                raise EmitError("while_loop with a data-dependent trip count")
            if not pv:
                return carry
            carry = self._jaxpr(bj.jaxpr, [_obj(c) for c in bj.consts], [*bconsts, *carry])
        raise EmitError("while_loop did not terminate within 100000 unrolled iterations")

    def _cond(self, eqn, ins):
        branches = eqn.params["branches"]
        index, ops = self.as_obj(ins[0]).flat[0], ins[1:]
        run = lambda b: self._jaxpr(b.jaxpr, [_obj(c) for c in b.consts], list(ops))
        if not _is_sym(index):
            return run(branches[int(index)])
        results = [run(b) for b in branches]
        g = self.g
        outs = []
        for k, var in enumerate(eqn.outvars):
            kind = _kind(var.aval.dtype)
            vals = [r[k] for r in results]
            if any(isinstance(v, Lazy) for v in vals):
                parts = [self.as_lazy(v, kind) for v in vals]

                def fn(i, parts=parts, kind=kind):
                    val = parts[0].fn(i)
                    for b in range(1, len(parts)):
                        val = g.op("select", (g.op("eq", (index, np.int32(b)), "b"), val,
                                              parts[b].fn(i)), kind)
                    return val

                outs.append(Lazy(var.aval.shape, kind, fn, sum(p.cost for p in parts)))
                continue
            out = np.empty(var.aval.shape, dtype=object)
            for i in range(out.size):
                val = vals[0].flat[i]
                for b in range(1, len(branches)):
                    val = g.op("select", (g.op("eq", (index, np.int32(b)), "b"), val,
                                          vals[b].flat[i]), kind)
                out.flat[i] = val
            outs.append(out)
        return outs


def symbolic(graph: Graph, name: str, shape: tuple[int, ...], kind: str = "f",
             offset: int = 0) -> np.ndarray:
    """An object array of input nodes ``name[offset + i]``."""
    out = np.empty(shape, dtype=object)
    for i in range(out.size):
        out.flat[i] = graph.input(name, offset + i, kind)
    return out
