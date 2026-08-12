"""Shared pytest configuration.

Contains global JAX config leakage between tests.

``pyroffi.toolbox`` deliberately mutates the process-wide ``jax_enable_x64``
setting (``toolbox/_session.py``) — that is a documented feature of a session,
since several IK and trajopt paths want float64. In a test process it is also a
side channel: a test that opens a toolbox session changes the dtype every test
after it in the same process runs under, and several solvers are genuinely
borderline in float32.

That produced three order-dependent failures that all pass in isolation:
``test_attachments`` (a CUDA SDF checker), ``test_toolbox_integration`` and
``test_topp_ra`` (torque limits reporting an absurd 321019 s duration, the
signature of a degenerate solve rather than a wrong answer). Each was
misdiagnosed at least once as a real regression before the ordering was
controlled for, which is the expensive kind of confusion.

This does not restore x64 settings made at import time (``test_subproblems``
does that at module scope, which runs during collection); it contains mutations
made *during* a test, which is where the toolbox session does its work.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _contain_jax_x64_leakage():
    """Snapshot ``jax_enable_x64`` before each test and restore it after."""
    import jax

    before = bool(jax.config.read("jax_enable_x64"))
    try:
        yield
    finally:
        if bool(jax.config.read("jax_enable_x64")) != before:
            jax.config.update("jax_enable_x64", before)
