#!/usr/bin/env python
"""Generate a Mermaid flowchart of the PyRoKi API, annotated with what pyroffi adds.

pyroffi is meant to be a strict superset of PyRoKi: ``import pyroffi as pk`` (and
``jaxls.X`` -> ``pk.X``) should run any PyRoKi program. This tool walks both packages'
public surfaces and diffs them, so the diagram doubles as a drop-in readiness check.

The walk is static (``ast`` only): nothing is imported, so it needs no JAX, GPU or
GRiM build, and a broken import in either package shows up as a finding instead of a
crash. Public = ``__all__`` if present, else ``import x as x``-style re-exports, public
top-level defs/classes/assignments, and names served by a module ``__getattr__``.

jaxls is part of the PyRoKi surface in practice (its examples build problems with
``jaxls.LeastSquaresProblem`` etc.), so the jaxls names PyRoKi's examples and sources
actually use are included and matched against pyroffi's top level.

Node status:
  same      present in pyroffi with a compatible signature
  extended  compatible, pyroffi adds defaulted params / methods
  changed   present but a PyRoKi param is missing or a new param is required
  missing   absent from pyroffi: a drop-in break
  new       pyroffi only
Edges are data flow taken from return annotations: callable -> public class it returns.

    python tools/api_flowchart.py --pyroki ../pyroffi/baselines/pyroki/src/pyroki
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Sym:
    kind: str  # module | class | function | attr | lazy | missing-module
    params: dict[str, bool] | None = None  # name -> has default; None = unknown
    returns: str | None = None
    members: dict[str, "Sym"] = field(default_factory=dict)


class Walker:
    """Resolve the public API of one package from source."""

    def __init__(self, pkg_dir: Path):
        self.pkg_dir = pkg_dir
        self.trees: dict[Path, ast.Module | None] = {}

    def _file(self, mod: Path) -> Path | None:
        for p in (mod / "__init__.py", mod.with_suffix(".py")):
            if p.exists():
                return p
        return None

    def _tree(self, f: Path) -> ast.Module:
        """Parsed module, with top-level `if`/`try` blocks flattened to their first branch."""
        if f not in self.trees:
            tree = ast.parse(f.read_text())
            body = []
            for node in tree.body:
                body += node.body if isinstance(node, ast.If | ast.Try) else [node]
            tree.body = body
            self.trees[f] = tree
        return self.trees[f]

    def _target(self, f: Path, node: ast.ImportFrom) -> Path | None:
        """Module path an ImportFrom refers to; None for absolute (external) imports."""
        if not node.level:
            return None
        base = f.parent
        for _ in range(node.level - 1):
            base = base.parent
        return base.joinpath(*node.module.split(".")) if node.module else base

    def module(self, mod: Path) -> Sym:
        f = self._file(mod)
        if f is None:
            return Sym("missing-module")
        tree, out = self._tree(f), Sym("module")
        names = self._public_names(tree)
        if f.name == "__init__.py":
            names += [c.stem for c in sorted(mod.iterdir()) if not c.name.startswith("_")
                      and (c.suffix == ".py" or (c / "__init__.py").exists()) and c.stem not in names]
        for name in names:
            out.members[name] = self.resolve(mod, name)
        return out

    def _public_names(self, tree: ast.Module) -> list[str]:
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "__all__" for t in node.targets):
                return list(ast.literal_eval(node.value))
        names = []
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                names += [a.asname for a in node.names if a.asname and not a.asname.startswith("_")]
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names.append(node.name)
            elif isinstance(node, ast.Assign | ast.AnnAssign):
                names += [t.id for t in _targets(node) if isinstance(t, ast.Name)]
        names += self._lazy_names(tree)
        return [n for n in dict.fromkeys(names) if not n.startswith("_")]

    @staticmethod
    def _getattr_fn(tree: ast.Module) -> ast.FunctionDef | None:
        return next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "__getattr__"), None)

    def _lazy_names(self, tree: ast.Module) -> list[str]:
        fn = self._getattr_fn(tree)
        if fn is None:
            return []
        return [c.value for cmp in ast.walk(fn) if isinstance(cmp, ast.Compare)
                for c in ast.walk(cmp) if isinstance(c, ast.Constant) and isinstance(c.value, str)]

    def resolve(self, mod: Path, name: str, depth: int = 0) -> Sym:
        f = self._file(mod)
        if f is None or depth > 20:
            return Sym("missing-module")
        tree = self._tree(f)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == name:
                return _function(node)
            if isinstance(node, ast.ClassDef) and node.name == name:
                return self._class(mod, node, depth)
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                tgt = self._target(f, node)
                for a in node.names:
                    if (a.asname or a.name) != name:
                        continue
                    if tgt is None:
                        return Sym("attr")  # external re-export
                    if (tgt / a.name).is_dir() or (tgt / f"{a.name}.py").exists():
                        return self.module(tgt / a.name)
                    return self.resolve(tgt, a.name, depth + 1)
            if isinstance(node, ast.Assign | ast.AnnAssign) and any(getattr(t, "id", None) == name for t in _targets(node)):
                return self._assigned(f, mod, tree, node.value, depth)
        if (fn := self._getattr_fn(tree)) and name in self._lazy_names(tree):
            for imp in (n for n in ast.walk(fn) if isinstance(n, ast.ImportFrom)):
                tgt = self._target(f, imp)
                for a in imp.names:
                    if tgt and a.name == name:
                        return self.resolve(tgt, name, depth + 1)
                    if tgt and ((tgt / a.name).is_dir() or (tgt / f"{a.name}.py").exists()):
                        sym = self.resolve(tgt / a.name, name, depth + 1)
                        if sym.kind != "missing-module":
                            return sym
            return Sym("lazy")
        if (mod / f"{name}.py").exists() or (mod / name).is_dir():
            return self.module(mod / name)
        return Sym("missing-module")

    def _class(self, mod: Path, node: ast.ClassDef, depth: int) -> Sym:
        out = Sym("class")
        for base in node.bases:  # inherit in-package bases (mixins etc.)
            if isinstance(base, ast.Name) and (b := self.resolve(mod, base.id, depth + 1)).kind == "class":
                out.members |= b.members
        for n in node.body:
            if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and not n.name.startswith("_"):
                out.members[n.name] = _function(n, method=True)
            elif isinstance(n, ast.Assign | ast.AnnAssign) and isinstance(t := getattr(n, "target", None) or n.targets[0], ast.Name):
                if not t.id.startswith("_") and n.value is not None and isinstance(n.value, ast.Name):
                    out.members[t.id] = Sym("attr")  # method alias, e.g. `half_extent = half_lengths`
        return out

    def _assigned(self, f: Path, mod: Path, tree: ast.Module, value: ast.expr, depth: int) -> Sym:
        """`x = Cost.factory(residual_fn)`-style: borrow the wrapped function's signature."""
        if not isinstance(value, ast.Call):
            return Sym("attr")
        # The innermost bare/dotted callable argument is the wrapped function.
        args = [a for c in ast.walk(value) if isinstance(c, ast.Call) for a in c.args
                if isinstance(a, ast.Name | ast.Attribute)]
        for arg in reversed(args):
            if isinstance(arg, ast.Name):
                sym = self.resolve(mod, arg.id, depth + 1)
            else:  # alias.fn where `from . import _mod as alias`
                sym = Sym("attr")
                for node in tree.body:
                    if isinstance(node, ast.ImportFrom) and (tgt := self._target(f, node)):
                        for a in node.names:
                            if (a.asname or a.name) == getattr(arg.value, "id", None):
                                sym = self.resolve(tgt / a.name, arg.attr, depth + 1)
            if sym.kind == "function":
                return Sym("function", params=dict(list(sym.params.items())[1:]), returns="Cost")  # drop `vals`
        return Sym("attr")


def _targets(node: ast.Assign | ast.AnnAssign) -> list[ast.expr]:
    return node.targets if isinstance(node, ast.Assign) else [node.target]


def _ann(node: ast.expr | None) -> str | None:
    if node is None:
        return None
    s = ast.unparse(node).strip("'\"")
    m = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", s)
    return m[-1] if m else None


def _function(node: ast.FunctionDef, method: bool = False) -> Sym:
    a = node.args
    pos = a.posonlyargs + a.args
    defaults = [False] * (len(pos) - len(a.defaults)) + [True] * len(a.defaults)
    params = {p.arg: d for p, d in zip(pos, defaults)}
    params |= {p.arg: d is not None for p, d in zip(a.kwonlyargs, a.kw_defaults)}
    if a.vararg:
        params["*" + a.vararg.arg] = True
    if a.kwarg:
        params["**" + a.kwarg.arg] = True
    decos = {ast.unparse(d) for d in node.decorator_list}
    if method and "staticmethod" not in decos:
        params.pop(next(iter(params), None), None)  # self / cls
    if any("factory" in d for d in decos):
        params.pop(next(iter(params), None), None)  # `vals`
        return Sym("function", params=params, returns="Cost")
    kind = "attr" if "property" in decos else "function"
    return Sym(kind, params=params, returns=_ann(node.returns))




# ---------------------------------------------------------------- diff


def _compat(old: Sym, new: Sym) -> tuple[str, str]:
    """(status, note) of `new` as a replacement for `old`."""
    if new.kind == "missing-module":
        return "missing", ""
    notes = []
    if old.params is not None and new.params is not None and not any(p.startswith("**") for p in new.params):
        gone = [p for p in old.params if p not in new.params and not p.startswith("*")]
        required = [p for p, d in new.params.items() if p not in old.params and not d]
        added = [p for p, d in new.params.items() if p not in old.params and d]
        if gone or required:
            return "changed", "; ".join(filter(None, [
                gone and "drops " + ", ".join(gone), required and "requires " + ", ".join(required)]))
        if added:
            notes.append("+" + ", ".join(added))
    if old.kind == "class" and new.kind == "class":
        bad = [m for m in old.members if m not in new.members or _compat(old.members[m], new.members[m])[0] in ("missing", "changed")]
        if bad:
            return "changed", "methods: " + ", ".join(bad)
        if set(new.members) - set(old.members) or any(_compat(old.members[m], new.members[m])[0] == "extended" for m in old.members):
            notes.append("+methods")
    return ("extended" if notes else "same"), " ".join(notes)


@dataclass
class Row:
    path: str  # dotted, relative to pk
    status: str
    sym: Sym
    note: str = ""
    methods: list[tuple[str, str, bool]] = field(default_factory=list)  # (name, status, callable)


def diff(old: Sym, new: Sym, prefix: str = "") -> list[Row]:
    rows = []
    for name, o in old.members.items():
        n = new.members.get(name, Sym("missing-module"))
        path = prefix + name
        if o.kind == "module":
            rows.append(Row(path, "missing" if n.kind != "module" else "same", o))
            rows += diff(o, n if n.kind == "module" else Sym("module"), path + ".")
            continue
        status, note = _compat(o, n)
        row = Row(path, status, n if status != "missing" else o, note)
        if o.kind == "class" and n.kind == "class":
            for m, om in o.members.items():
                row.methods.append((m, _compat(om, n.members.get(m, Sym("missing-module")))[0], om.kind == "function"))
            row.methods += [(m, "new", nm.kind == "function") for m, nm in n.members.items() if m not in o.members]
        rows.append(row)
    for name, n in new.members.items():
        if name in old.members:
            continue
        path = prefix + name
        status = "missing" if n.kind == "missing-module" else "new"
        if n.kind == "module":
            if child := diff(Sym("module"), n, path + "."):  # skip modules with no public surface
                rows += [Row(path, status, n)] + child
            continue
        rows.append(Row(path, status, n, "imported but module not found" if status == "missing" else "",
                        [(m, "new", s.kind == "function") for m, s in n.members.items()] if n.kind == "class" else []))
    return rows


# ---------------------------------------------------------------- render

STYLE = {
    "same": "fill:#eceff1,stroke:#90a4ae,color:#263238",
    "extended": "fill:#e3f2fd,stroke:#1e88e5,color:#0d47a1",
    "changed": "fill:#fff3e0,stroke:#fb8c00,color:#e65100",
    "missing": "fill:#ffebee,stroke:#e53935,color:#b71c1c",
    "new": "fill:#e8f5e9,stroke:#43a047,color:#1b5e20",
}
MARK = {"same": "", "extended": "~", "changed": "!", "missing": "✗", "new": "+"}


def _id(path: str) -> str:
    return "n_" + re.sub(r"\W", "_", path)


def _esc(s: str) -> str:
    return s.replace('"', "#quot;").replace("<", "#lt;").replace(">", "#gt;")


def mermaid(rows: list[Row], max_methods: int, jaxls_names: set[str]) -> str:
    out = ["flowchart LR", '  pk(["import pyroffi as pk"])']
    modules = {r.path: r for r in rows if r.sym.kind == "module"}
    classes = {r.path.rsplit(".", 1)[-1]: r.path for r in rows if r.sym.kind == "class"}
    by_parent: dict[str, list[Row]] = {}
    for r in rows:
        if r.path not in modules:
            by_parent.setdefault(r.path.rpartition(".")[0], []).append(r)

    def node(r: Row) -> str:
        name = r.path.rsplit(".", 1)[-1]
        label = f"{MARK[r.status]} {name}".strip()
        if r.sym.kind == "class":
            label = f"<b>{label}</b>"
            meths = [f"{MARK[s]}.{m}{'()' if fn else ''}" for m, s, fn in r.methods]
            if len(meths) > max_methods:
                n_new = sum(s == "new" for _, s, _ in r.methods[max_methods:])
                meths = meths[:max_methods] + [f"… {len(r.methods) - max_methods} more ({n_new} new)"]
            label += "".join("<br/>" + _esc(m) for m in meths)
        elif r.sym.kind == "function":
            label += "()"
        if r.path in jaxls_names:
            label += "<br/><i>(was jaxls.%s)</i>" % name
        if r.note:
            label += f"<br/><i>{_esc(r.note)}</i>"
        return f'{_id(r.path)}["{label}"]:::{r.status}'

    def emit(prefix: str, indent: str):
        for r in by_parent.get(prefix, []):
            out.append(indent + node(r))
        for path, m in modules.items():
            if path.rpartition(".")[0] == prefix:
                title = f"{MARK[m.status]} pk.{path}".strip()
                out.append(f'{indent}subgraph {_id(path)}["{title}"]')
                out.append(f"{indent}  direction TB")
                emit(path, indent + "  ")
                out.append(f"{indent}end")
                out.append(f"{indent}class {_id(path)} {m.status}")

    emit("", "  ")
    for path in modules:
        if "." not in path:
            out.append(f"  pk --> {_id(path)}")
    # Data flow: callable -> public class it returns.
    edges = []
    for r in rows:
        targets = [(r.path, r.sym.returns)] if r.sym.kind == "function" else []
        if r.sym.kind == "class":
            targets += [(r.path, ms.returns) for ms in r.sym.members.values() if ms.kind == "function"]
        for src, ret in targets:
            if ret in classes and classes[ret] != src:
                edges.append(f"  {_id(src)} -.-> {_id(classes[ret])}")
    out += dict.fromkeys(edges)
    out += [f"  classDef {k} {v}" for k, v in STYLE.items()]
    return "\n".join(out)


def jaxls_surface(pyroki_dir: Path, jaxls_dir: Path) -> Sym:
    """The jaxls names PyRoKi's examples and sources reference, as a pseudo-module."""
    text = "".join(p.read_text() for d in (pyroki_dir, pyroki_dir.parents[1] / "examples") for p in d.rglob("*.py"))
    used = sorted(set(re.findall(r"\bjaxls\.([A-Za-z_]\w*)", text)))
    full = Walker(jaxls_dir.parent).module(jaxls_dir)
    return Sym("module", members={n: full.members[n] for n in used if n in full.members})


def main():
    if sys.version_info < (3, 12):
        sys.exit("needs Python >= 3.12 to parse jaxls's `type X[...]` syntax (no other deps)")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pyroki", type=Path, default=ROOT.parent / "pyroffi/baselines/pyroki/src/pyroki")
    ap.add_argument("--pyroffi", type=Path, default=ROOT / "src/pyroffi")
    ap.add_argument("--jaxls", type=Path, default=None, help="jaxls package dir (default: found on sys.path)")
    ap.add_argument("--out", type=Path, default=ROOT / "docs/api_flowchart.md")
    ap.add_argument("--max-methods", type=int, default=12, help="methods listed per class node")
    args = ap.parse_args()

    pyroki = Walker(args.pyroki.parent).module(args.pyroki)
    jaxls_dir = args.jaxls or Path(importlib.util.find_spec("jaxls").origin).parent
    # jaxls.X is spelled pk.X in pyroffi: merge into the top level for the diff.
    jaxls = jaxls_surface(args.pyroki, jaxls_dir).members
    for name, sym in jaxls.items():
        pyroki.members.setdefault(name, sym)
    rows = diff(pyroki, Walker(args.pyroffi.parent).module(args.pyroffi))

    counts = {s: sum(r.status == s for r in rows) for s in STYLE}
    problems = [r for r in rows if r.status in ("missing", "changed")]
    lines = [
        "# PyRoKi → pyroffi API flowchart",
        "",
        f"Generated by `tools/api_flowchart.py` (static, from source). PyRoKi surface includes the "
        f"jaxls names its examples use (`jaxls.X` → `pk.X`).",
        "",
        "Legend: `~` extended (compatible superset) · `!` changed (breaks drop-in) · `✗` missing · "
        "`+` new in pyroffi · unmarked = same. Dotted edges: return-type data flow.",
        "",
        "| " + " | ".join(counts) + " |",
        "|" + "---|" * len(counts),
        "| " + " | ".join(str(v) for v in counts.values()) + " |",
        "",
        "## Drop-in breaks",
        "",
        *([f"- `pk.{r.path}` — **{r.status}**{': ' + r.note if r.note else ''}" for r in problems] or ["None."]),
        *[f"- `pk.{r.path}.{m}` — **{s}**" for r in rows for m, s, _ in r.methods if s in ("missing", "changed")],
        "",
        "## Flowchart",
        "",
        "```mermaid",
        mermaid(rows, args.max_methods, set(jaxls)),
        "```",
        "",
    ]
    args.out.write_text("\n".join(lines))
    print(f"wrote {args.out}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
