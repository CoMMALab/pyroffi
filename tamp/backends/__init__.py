"""Geometric-oracle backends that cannot share an interpreter with the JAX stack.

``spasm.tamp`` imports JAX and enforces numpy>=2 at import time. cuRobo's
environment pins numpy<2 and has no JAX, so a backend written against cuRobo
cannot live inside that package -- importing it would fail on the environment
contract before reaching any cuRobo code. Backends here are deliberately
dependency-light and importable standalone, and the benchmark harness runs each
in its own environment via subprocess.
"""
