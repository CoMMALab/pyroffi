"""jaxpr -> straight-line scalar CUDA C++.

A jaxpr is evaluated with every array element as one scalar node, so the output is
loop-free code for one fixed problem structure. Three things make that code small:

* constants fold as nodes are built (the robot's kinematic constants, and the zero tangents
  ``jax.jacfwd`` produces for variables a residual does not touch, disappear here, since
  nvcc may not fold ``x * 0.0f`` under IEEE rules);
* identical expressions are shared (hash-consing);
* only nodes an output reaches are emitted.

Data movement (reshape, gather, concatenate, scan slicing, ...) is never reimplemented:
the primitive itself is run on integer codes naming the input elements, and its output
says which element lands where. Anything that cannot be resolved to straight-line code
(a data-dependent ``while_loop``, a sort of symbolic values) raises with the primitive's
name.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import jax
import numpy as np
from jax.extend import core as jcore


class EmitError(NotImplementedError):
    """The traced function uses something straight-line code cannot express."""


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
    """A symbolic scalar: an input, or an operation on other nodes."""

    __slots__ = ("op", "args", "kind", "idx")

    def __init__(self, op: str, args: tuple, kind: str, idx: int):
        self.op, self.args, self.kind, self.idx = op, args, kind, idx


def _is_sym(v) -> bool:
    return isinstance(v, Sym)


def _key(v):
    return ("s", v.idx) if isinstance(v, Sym) else ("c", _kind(v.dtype), v.item())


_COMMUTATIVE = {"add", "mul", "max", "min", "eq", "ne", "and", "or", "xor"}


def _div(a, b):
    if np.issubdtype(np.asarray(a).dtype, np.integer):
        return np.int32(int(a / b)) if b != 0 else np.int32(0)
    return np.float32(a) / np.float32(b)


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
    "pow": np.power, "square": np.square, "rem": np.fmod,
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


class Graph:
    """Hash-consed scalar expression graph. Nodes are created in topological order."""

    def __init__(self):
        self.nodes: list[Sym] = []
        self._table: dict[tuple, Sym] = {}

    def input(self, name: str, index: int, kind: str = "f") -> Sym:
        return self._make(name, (index,), kind)

    def _make(self, op: str, args: tuple, kind: str) -> Sym:
        key = (op, kind, tuple(a if isinstance(a, int) and op in ("x", "p", "w", "s") else _key(a)
                               for a in args))
        node = self._table.get(key)
        if node is None:
            node = Sym(op, args, kind, len(self.nodes))
            self.nodes.append(node)
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

    # ── C++ emission ────────────────────────────────────────────────────────
    def emit(self, outputs: Sequence[tuple[str, Any]], indent: str = "    ") -> str:
        """Statements computing each ``(lvalue, node)`` in ``outputs``, dead code removed."""
        live = np.zeros(len(self.nodes), bool)
        stack = [v for _, v in outputs if _is_sym(v)]
        while stack:
            n = stack.pop()
            if live[n.idx]:
                continue
            live[n.idx] = True
            stack.extend(a for a in n.args if _is_sym(a))
        lines = []
        for n in self.nodes:
            if live[n.idx] and n.op not in ("x", "p", "w", "s"):
                lines.append(f"{indent}const {_CTYPE[n.kind]} v{n.idx} = {self._expr(n)};")
        for lhs, v in outputs:
            lines.append(f"{indent}{lhs} = {_ref(v, 'f')};")
        return "\n".join(lines)

    def _expr(self, n: Sym) -> str:
        a = [_ref(x, n.kind if n.op not in ("cast", "select") else None) for x in n.args]
        op, k = n.op, n.kind
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
        if k == "i" and op in ("max", "min", "abs"):
            return f"{op}({', '.join(a)})"
        if k == "i" and op == "rem":
            return f"({a[0]} % {a[1]})"
        if op in _CPP_UNARY_F:
            return f"{_CPP_UNARY_F[op]}({a[0]})"
        if op in _CPP_BINARY_F:
            return f"{_CPP_BINARY_F[op]}({a[0]}, {a[1]})"
        raise EmitError(f"no C++ form for {op!r}")


_CTYPE = {"f": "float", "i": "int", "b": "bool"}


def _ref(v, kind: str | None) -> str:
    if _is_sym(v):
        if v.op == "x":
            return f"x[{v.args[0]}]"
        if v.op == "w":
            return f"w[{v.args[0]}]"
        if v.op == "s":
            return f"s[{v.args[0]}]"
        if v.op == "p":
            return {"f": f"p[{v.args[0]}]", "i": f"((int)p[{v.args[0]}])",
                    "b": f"(p[{v.args[0]}] != 0.0f)"}[v.kind]
        return f"v{v.idx}"
    k = _kind(v.dtype)  # constants carry their own dtype; `kind` only matters for nodes
    if k == "b":
        return "true" if v else "false"
    if k == "i":
        return f"({int(v)})" if int(v) != -(2**31) else "(-2147483647 - 1)"
    f = float(np.float32(v))
    if math.isnan(f):
        return "NAN"
    if math.isinf(f):
        return "INFINITY" if f > 0 else "(-INFINITY)"
    return f"{f!r}f"


# ── jaxpr evaluation ────────────────────────────────────────────────────────

def _obj(arr) -> np.ndarray:
    arr = np.asarray(arr)
    out = np.empty(arr.shape, dtype=object)
    for i, v in enumerate(arr.ravel()):
        out.flat[i] = v
    return out


def _all_const(a: np.ndarray) -> bool:
    return not any(_is_sym(v) for v in a.flat)


def _typed(a: np.ndarray, dtype) -> np.ndarray:
    return np.array([np.asarray(v).item() for v in a.flat], dtype=dtype).reshape(a.shape)


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


class Evaluator:
    """Evaluates a jaxpr over object arrays of scalar nodes, building ``graph``."""

    def __init__(self, graph: Graph):
        self.g = graph

    def run(self, closed: jcore.ClosedJaxpr, args: Sequence[np.ndarray]) -> list[np.ndarray]:
        # Constant folding and data-movement probes bind primitives eagerly; keep them off the GPU.
        with jax.default_device(jax.devices("cpu")[0]):
            return self._jaxpr(closed.jaxpr, [_obj(c) for c in closed.consts], list(args))

    def _jaxpr(self, jaxpr, consts, args):
        env: dict = {}

        def read(v):
            if isinstance(v, jcore.Literal):
                return _obj(np.asarray(v.val, dtype=v.aval.dtype))
            return env[v]

        for v, x in zip(jaxpr.constvars, consts):
            env[v] = x
        for v, x in zip(jaxpr.invars, args):
            env[v] = x
        for eqn in jaxpr.eqns:
            outs = self._eqn(eqn, [read(v) for v in eqn.invars])
            for v, o in zip(eqn.outvars, outs):
                if not isinstance(v, jcore.DropVar):
                    env[v] = o
        return [read(v) for v in jaxpr.outvars]

    def _eqn(self, eqn, ins: list[np.ndarray]) -> list[np.ndarray]:
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
            f"primitive {name!r} on symbolic values cannot be compiled to straight-line code "
            f"(at {source_info_util.summarize(eqn.source_info)})")

    def _const_eval(self, eqn, ins):
        vals = [_typed(a, v.aval.dtype) for a, v in zip(ins, eqn.invars)]
        outs = eqn.primitive.bind(*vals, **eqn.params)
        if not eqn.primitive.multiple_results:
            outs = [outs]
        return [_obj(np.asarray(o)) for o in outs]

    def _elementwise(self, eqn, ins):
        name = eqn.primitive.name
        kind = _kind(eqn.outvars[0].aval.dtype)
        shape = eqn.outvars[0].aval.shape
        ins = [np.broadcast_to(a, shape) for a in ins]
        out = np.empty(shape, dtype=object)
        g = self.g
        for i in range(out.size):
            a = [x.flat[i] for x in ins]
            if name in ("add_any",):
                v = g.op("add", a, kind)
            elif name in ("copy", "copy_p", "reduce_precision", "stop_gradient"):
                v = a[0]
            elif name == "convert_element_type":
                v = g.cast(a[0], kind)
            elif name == "integer_pow":
                v = self._ipow(a[0], eqn.params["y"], kind)
            elif name == "clamp":
                lo, x, hi = a
                v = g.op("min", (g.op("max", (x, lo), kind), hi), kind)
            elif name == "select_n":
                v = a[-1]
                # select_n(which, c0, c1, ...): cases[which]. Bool which: c1 if true.
                which = a[0]
                if len(a) == 3:
                    v = g.op("select", (g.cast(which, "b") if _is_sym(which) else np.bool_(which),
                                        a[1], a[2]), kind)
                else:
                    v = a[1]
                    for c in range(1, len(a) - 1):
                        v = g.op("select", (g.op("eq", (which, np.int32(c)), "b"), v, a[c + 1]), kind)
            elif name == "round" and eqn.params.get("rounding_method") not in (None, 0):
                v = g.op("round", a, kind)  # ties-to-even; roundf is away-from-zero
            else:
                if name in ("pow", "atan2", "rem", "max", "min", "add", "sub", "mul", "div") and kind == "f":
                    a = [g.cast(x, "f") for x in a]
                v = g.op(name, a, kind)
            out.flat[i] = v
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

    def _move(self, eqn, ins):
        """Data movement: run the primitive on integer codes naming the input elements."""
        name = eqn.primitive.name
        idx_pos = _INDEX_OPERANDS.get(name, ())
        if idx_pos is None:  # dynamic_slice / dynamic_update_slice: trailing scalar starts
            idx_pos = tuple(range(1 if name == "dynamic_slice" else 2, len(ins)))
        pool: list = []
        codes: list = []
        for k, a in enumerate(ins):
            if k in idx_pos:
                codes.append(None)
                continue
            codes.append(np.arange(len(pool), len(pool) + a.size, dtype=np.int32).reshape(a.shape))
            pool.extend(a.flat)

        def run(index_vals: dict[int, np.ndarray]) -> list[np.ndarray]:
            args = [index_vals[k] if k in idx_pos else codes[k] for k in range(len(ins))]
            outs = eqn.primitive.bind(*args, **eqn.params)
            if not eqn.primitive.multiple_results:
                outs = [outs]
            return [np.asarray(o) for o in outs]

        sym = [(k, j) for k in idx_pos for j, v in enumerate(ins[k].flat) if _is_sym(v)]
        const_idx = {k: _typed(ins[k], eqn.invars[k].aval.dtype) if not any(
            (kk == k) for kk, _ in sym) else None for k in idx_pos}
        if not sym:
            return [self._pick(pool, o) for o in run(const_idx)]
        if len(sym) > 1 or name == "scatter":
            raise EmitError(f"{name} with more than one runtime index is not supported")
        (k, j), = sym
        s = ins[k].flat[j]
        n = self._index_range(eqn, ins, k, j)
        variants = []
        for v in range(n):
            idx = _typed(np.where(np.vectorize(_is_sym, otypes=[bool])(ins[k]),
                                  np.int32(v), ins[k]), eqn.invars[k].aval.dtype)
            variants.append(run({**const_idx, k: idx}))
        # Out-of-range runtime indices clamp, as gather/dynamic_slice do.
        g = self.g
        sc = g.op("min", (g.op("max", (g.cast(s, "i"), np.int32(0)), "i"), np.int32(n - 1)), "i")
        outs = []
        for o in range(len(variants[0])):
            picked = [self._pick(pool, var[o]) for var in variants]
            kind = _kind(eqn.outvars[o].aval.dtype)
            out = np.empty(picked[0].shape, dtype=object)
            for i in range(out.size):
                val = picked[0].flat[i]
                for v in range(1, n):
                    val = g.op("select", (g.op("eq", (sc, np.int32(v)), "b"), val, picked[v].flat[i]), kind)
                out.flat[i] = val
            outs.append(out)
        return outs

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
        nb = int(np.prod([lhs.shape[d] for d in lb], dtype=int))
        nl = int(np.prod([lhs.shape[d] for d in lf], dtype=int))
        nr = int(np.prod([rhs.shape[d] for d in rf], dtype=int))
        nk = int(np.prod([lhs.shape[d] for d in lc], dtype=int))
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
        xt = np.transpose(x, (*keep, *axes)).reshape(
            int(np.prod([x.shape[d] for d in keep], dtype=int)), -1)
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

    def _scan(self, eqn, ins):
        p = eqn.params
        body, n, nc, ncar = p["jaxpr"], p["length"], p["num_consts"], p["num_carry"]
        consts, carry, xs = ins[:nc], ins[nc:nc + ncar], ins[nc + ncar:]
        ys_steps = []
        for t in (range(n - 1, -1, -1) if p["reverse"] else range(n)):
            outs = self._jaxpr(body.jaxpr, [_obj(c) for c in body.consts],
                               [*consts, *carry, *(x[t, ...] for x in xs)])
            carry, ys = outs[:ncar], outs[ncar:]
            ys_steps.append(ys)
        if p["reverse"]:
            ys_steps.reverse()
        n_ys = len(eqn.outvars) - ncar
        ys = []
        for k in range(n_ys):
            if n == 0:
                ys.append(np.empty(eqn.outvars[ncar + k].aval.shape, dtype=object))
            else:
                ys.append(np.stack([s[k] for s in ys_steps]))
        return [*carry, *ys]

    def _while(self, eqn, ins):
        p = eqn.params
        cj, bj, cn, bn = p["cond_jaxpr"], p["body_jaxpr"], p["cond_nconsts"], p["body_nconsts"]
        cconsts, bconsts, carry = ins[:cn], ins[cn:cn + bn], ins[cn + bn:]
        for _ in range(100_000):
            (pred,) = self._jaxpr(cj.jaxpr, [_obj(c) for c in cj.consts], [*cconsts, *carry])
            pv = pred.flat[0]
            if _is_sym(pv):
                raise EmitError("while_loop with a data-dependent trip count")
            if not pv:
                return carry
            carry = self._jaxpr(bj.jaxpr, [_obj(c) for c in bj.consts], [*bconsts, *carry])
        raise EmitError("while_loop did not terminate within 100000 unrolled iterations")

    def _cond(self, eqn, ins):
        branches = eqn.params["branches"]
        index, ops = ins[0].flat[0], ins[1:]
        run = lambda b: self._jaxpr(b.jaxpr, [_obj(c) for c in b.consts], list(ops))
        if not _is_sym(index):
            return run(branches[int(index)])
        results = [run(b) for b in branches]
        outs = []
        for k, var in enumerate(eqn.outvars):
            kind = _kind(var.aval.dtype)
            out = np.empty(var.aval.shape, dtype=object)
            for i in range(out.size):
                val = results[0][k].flat[i]
                for b in range(1, len(branches)):
                    val = self.g.op("select", (self.g.op("eq", (index, np.int32(b)), "b"), val,
                                               results[b][k].flat[i]), kind)
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
