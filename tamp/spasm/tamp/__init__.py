"""Classical-TAMP library backed entirely by pyroffi / SPaSM geometry.

This is the *library* half of the former ``benchmarks/tamp/`` package (the
runnable bench scripts stay under ``benchmarks/tamp/``). It provides the PDDL
domains, the geometric problem builders (:mod:`spasm.tamp.problems`), the
pyroffi-backed PDDLStream stream map (:mod:`spasm.tamp.streams_pyroffi`), the
shared collision geometry (:mod:`spasm.tamp.geometry`), and the RoboSuite
execution bridge (:mod:`spasm.tamp.robosuite_bridge`).

Importing this package runs :mod:`spasm.tamp._setup`, which puts the vendored
``external/pddlstream`` on ``sys.path`` (it ships no setup.py) and pins the env
quirks the port needs (numpy>=2 guard, ``MUJOCO_GL=egl``). This makes anything
that touches ``spasm.tamp`` self-bootstrapping instead of relying on callers to
import the path shim first.
"""
from . import _setup  # noqa: F401
