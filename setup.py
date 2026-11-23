"""Refuse to build pyroffi without its submodules; all metadata lives in pyproject.toml.

pyroffi needs GRiM (with GRiD and mantis) and cudals from external/. Install them together,
editable, with `pip install -r requirements.txt` from the repo root.
"""

from pathlib import Path

from setuptools import setup

_ROOT = Path(__file__).resolve().parent
_REQUIRED = [
    "external/GRiM/bindings/grim/__init__.py",
    "external/GRiM/external/mantis/python/mantis/__init__.py",
    "external/GRiM/external/GRiD/grid_codegen/__init__.py",
    "external/cudals/src/cudals/__init__.py",
]
_missing = [p for p in _REQUIRED if not (_ROOT / p).is_file()]
if _missing:
    raise SystemExit(
        "pyroffi: submodules not checked out (missing " + ", ".join(_missing) + ").\n"
        "Run `git submodule update --init --recursive`, then `pip install -r requirements.txt`.")

setup()
