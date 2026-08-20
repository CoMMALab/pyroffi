"""Path + environment shim for the TAMP experiments.

PDDLStream ships no setup.py — it is meant to be used from PYTHONPATH. Import
this module first (``from spasm.tamp import _setup``) to put the vendored
``tamp/external/pddlstream`` on ``sys.path`` and pin the env quirks the port
needs:

* ``jax_enable_x64`` stays False (``spasm/backend.py`` contract).
* ``MUJOCO_GL=egl`` for headless RoboSuite offscreen rendering.

No pybullet is imported anywhere under ``spasm/tamp`` — every geometric
primitive routes through ``spasm.backend`` / ``spasm.tetris.solve``, so the
motion-primitive backend is pyroffi throughout and any difference against the
stock-SPaSM baseline is attributable to the motion backend, not to a faster
collision/FK/IK implementation.
"""
import os
import sys

from spasm.paths import TAMP_ROOT, PDDLSTREAM_ROOT, require

require(PDDLSTREAM_ROOT, "vendored pddlstream")

for p in (TAMP_ROOT, PDDLSTREAM_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("MUJOCO_GL", "egl")

# Guard the numpy>=2 / jax contract early with a clear message.
import numpy as _np  # noqa: E402

if int(_np.__version__.split(".")[0]) < 2:
    raise RuntimeError(
        f"numpy {_np.__version__} < 2 breaks jax (asarray(copy=)). "
        "Run `pip install 'numpy>=2'` in the pyroffi-tamp env "
        "(robosuite pulls numpy 1.26 — reinstall numpy 2 after)."
    )


def sanity_check():
    """Assert the whole stack imports (used by tests / bench preflight)."""
    import pddlstream  # noqa: F401
    from pddlstream.algorithms.meta import solve  # noqa: F401
    from spasm import backend  # noqa: F401
    return True
