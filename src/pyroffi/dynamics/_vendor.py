"""Locate and import the external GRiD modules (GRiDCodeGenerator, URDFParser).

These live in ``external/`` at the repo root (like vamp and cricket) and are
plain importable source packages, not pip-installable. ``PYROFFI_GRID_PATH``
overrides the search location (it must be the directory *containing* the
``GRiDCodeGenerator`` and ``URDFParser`` package directories).
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
    """Put the directory containing GRiDCodeGenerator/URDFParser on sys.path."""
    try:
        import GRiDCodeGenerator  # noqa: F401
        import URDFParser  # noqa: F401

        return
    except ImportError:
        pass
    for d in _candidate_dirs():
        if (d / "GRiDCodeGenerator" / "__init__.py").is_file() and (
            d / "URDFParser" / "__init__.py"
        ).is_file():
            if str(d) not in sys.path:
                sys.path.insert(0, str(d))
            return
    raise ImportError(
        "Could not locate the external GRiD modules (GRiDCodeGenerator, "
        "URDFParser). Clone them into <repo>/external/ or set "
        "PYROFFI_GRID_PATH to the directory containing them."
    )
