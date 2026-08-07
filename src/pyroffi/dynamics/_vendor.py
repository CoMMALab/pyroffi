"""Locate and import the external GRiD codegen module.

Upstream is ``A2R-Lab/GRiD`` (branch ``modernizing-tests``), where the codegen
package was renamed ``GRiDCodeGenerator/`` -> ``grid_codegen/``.  It lives in
``external/GRiD`` at the repo root (like vamp and cricket) and is a plain
importable source package, not pip-installable.  ``PYROFFI_GRID_PATH`` overrides
the search location (it must be the directory *containing* the ``grid_codegen``
package directory, i.e. the GRiD checkout root).

The companion ``URDFParser`` package is not required: its ``Robot`` object model
and post-processing are vendored into :mod:`pyroffi.dynamics._grid_urdf`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_PACKAGE = "grid_codegen"
_LEGACY_PACKAGE = "GRiDCodeGenerator"


def _candidate_dirs() -> list[Path]:
    dirs = []
    env = os.environ.get("PYROFFI_GRID_PATH")
    if env:
        dirs.append(Path(env))
    # src/pyroffi/dynamics/_vendor.py -> repo root is parents[3].
    external = Path(__file__).resolve().parents[3] / "external"
    dirs.append(external / "GRiD")
    dirs.append(external)
    return dirs


def ensure_grid_importable() -> None:
    """Put the directory containing the ``grid_codegen`` package on sys.path."""
    try:
        import grid_codegen  # noqa: F401

        return
    except ImportError:
        pass
    for d in _candidate_dirs():
        if (d / _PACKAGE / "__init__.py").is_file():
            if str(d) not in sys.path:
                sys.path.insert(0, str(d))
            return
        if (d / _LEGACY_PACKAGE / "__init__.py").is_file():
            raise ImportError(
                f"Found the legacy '{_LEGACY_PACKAGE}' package at {d}, but pyroffi "
                f"now requires A2R-Lab/GRiD (package '{_PACKAGE}'). Update the "
                "external/GRiD checkout."
            )
    raise ImportError(
        f"Could not locate the external '{_PACKAGE}' package (A2R-Lab/GRiD). Clone "
        "it into <repo>/external/GRiD or set PYROFFI_GRID_PATH to the directory "
        "containing it."
    )
