"""Locate and import the external GRiDCodeGenerator module.

It lives in ``external/`` at the repo root (like vamp and cricket) and is a plain
importable source package, not pip-installable. ``PYROFFI_GRID_PATH`` overrides
the search location (it must be the directory *containing* the
``GRiDCodeGenerator`` package directory).

The companion ``URDFParser`` package is no longer required: its ``Robot`` object
model and post-processing are vendored into :mod:`pyroffi.dynamics._grid_urdf`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _candidate_dirs() -> list[Path]:
    dirs = []
    env = os.environ.get("PYROFFI_GRID_PATH")
    if env:
        dirs.append(Path(env))
    # src/pyroffi/dynamics/_vendor.py -> repo root is parents[3].
    dirs.append(Path(__file__).resolve().parents[3] / "external")
    return dirs


def ensure_grid_importable() -> None:
    """Put the directory containing GRiDCodeGenerator on sys.path."""
    try:
        import GRiDCodeGenerator  # noqa: F401

        return
    except ImportError:
        pass
    for d in _candidate_dirs():
        if (d / "GRiDCodeGenerator" / "__init__.py").is_file():
            if str(d) not in sys.path:
                sys.path.insert(0, str(d))
            return
    raise ImportError(
        "Could not locate the external GRiDCodeGenerator module. Clone it into "
        "<repo>/external/ or set PYROFFI_GRID_PATH to the directory containing it."
    )
