"""Repo-relative path resolution for the TAMP experiments.

The port originally hardcoded absolute paths to three sibling checkouts
(``/home/sadmin/Work/{pyroffi,spasm}``). Under ``pyroffi/tamp`` everything is
resolved relative to this file instead, so the experiments run from a fresh
clone on any machine:

    <pyroffi>/tamp/spasm/paths.py   <- this file
    <pyroffi>/tamp/external/pddlstream        (pinned clone)
    <pyroffi>/tamp/external/fast-downward     (pinned clone)
    <pyroffi>/tamp/external/spasm_stock       (commalab/spasm, the baseline)
    <pyroffi>/resources/panda/...             (pyroffi's own URDFs)

Each location honours an environment-variable override (``PYROFFI_ROOT``,
``SPASM_STOCK_ROOT``, ``PDDLSTREAM_ROOT``) for the case where a user wants to
point at an existing sibling checkout rather than the vendored clone.
"""
from __future__ import annotations

import os

_HERE = os.path.dirname(os.path.abspath(__file__))

#: ``<pyroffi>/tamp`` — the experiment root (package dir's parent).
TAMP_ROOT = os.path.abspath(os.path.join(_HERE, ".."))

#: The pyroffi checkout this ``tamp/`` lives inside.
PYROFFI_ROOT = os.environ.get(
    "PYROFFI_ROOT", os.path.abspath(os.path.join(TAMP_ROOT, "..")))

EXTERNAL = os.path.join(TAMP_ROOT, "external")

#: Vendored clone of commalab/spasm — the stock (kinematic-only) baseline.
SPASM_STOCK_ROOT = os.environ.get(
    "SPASM_STOCK_ROOT", os.path.join(EXTERNAL, "spasm_stock"))

PDDLSTREAM_ROOT = os.environ.get(
    "PDDLSTREAM_ROOT", os.path.join(EXTERNAL, "pddlstream"))

FD_STANDALONE = os.environ.get(
    "FD_ROOT", os.path.join(EXTERNAL, "fast-downward"))

# --------------------------------------------------------------------------- #
# Robot models
# --------------------------------------------------------------------------- #

#: SPaSM's own sphere-visual Panda. Used by ``spasm.backend`` so the collision
#: sphere set is bit-identical to stock SPaSM's — this is what makes the
#: baseline comparison fair (same geometry, different motion backend).
SPASM_URDF = os.path.join(
    SPASM_STOCK_ROOT, "kinematics", "urdf", "panda_sphere_visuals.urdf")

#: pyroffi's inertially-complete, mimic-free Panda. Required for dynamics:
#: SPaSM's URDF carries no usable inertial data, so the torque path must use
#: this one. (The two agree kinematically; see tests/test_backend.py.)
PANDA_URDF = os.path.join(PYROFFI_ROOT, "resources", "panda", "panda_spherized.urdf")
PANDA_MESH_DIR = os.path.join(PYROFFI_ROOT, "resources", "panda", "meshes")


def require(path, what):
    """Fail loudly and actionably when a vendored dependency is missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{what} not found at {path}. Run `tamp/setup_externals.sh` to "
            f"clone the vendored dependencies, or set the corresponding "
            f"environment variable to an existing checkout.")
    return path
