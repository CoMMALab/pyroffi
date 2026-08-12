"""Dtype contract at the CUDA FFI boundary.

Every CUDA kernel in pyroffi is compiled for float32/int32. XLA's FFI decodes
operands by declared type and rejects a mismatch outright rather than
converting, so a 64-bit array reaching a handler fails the whole call::

    INVALID_ARGUMENT: [execute] Failed to decode all FFI handler operands
    (bad operands at: 2, 3, 6, 7, 8) ... expected S32 but got S64

That is easy to hit by accident, because ``jax_enable_x64`` is process-wide
global state: ``pyroffi.toolbox`` turns it on by design (several JAX/CPU IK and
trajopt paths want float64), so the same call works or fails depending on
whether a toolbox session was opened first. The casts therefore belong at the
boundary itself, not at the call sites, where one omission is invisible until
someone enables x64.

``jnp.asarray(x)`` is NOT sufficient: without an explicit dtype it preserves
whatever it was given, which under x64 is exactly the 64-bit array the handler
rejects.
"""

from __future__ import annotations

import jax.numpy as jnp

#: Dtypes of the shared robot-model buffers, in the order every kernel takes
#: them: twists, parent_tf, parent_idx, act_idx, mimic_mul, mimic_off,
#: mimic_act_idx, topo_inv.
ROBOT_BUFFER_DTYPES = (
    jnp.float32,  # twists
    jnp.float32,  # parent_tf
    jnp.int32,    # parent_idx
    jnp.int32,    # act_idx
    jnp.float32,  # mimic_mul
    jnp.float32,  # mimic_off
    jnp.int32,    # mimic_act_idx
    jnp.int32,    # topo_inv
)


def robot_buffers(twists, parent_tf, parent_idx, act_idx,
                  mimic_mul, mimic_off, mimic_act_idx, topo_inv) -> tuple:
    """Cast the robot-model arrays to the dtypes the C++ kernels expect.

    Single definition shared by every kernel wrapper. It was previously copied
    per wrapper, and the fused collision path skipped it entirely -- which is
    the bug this module exists to prevent recurring.
    """
    arrays = (twists, parent_tf, parent_idx, act_idx,
              mimic_mul, mimic_off, mimic_act_idx, topo_inv)
    return tuple(jnp.asarray(a, dtype=dt)
                 for a, dt in zip(arrays, ROBOT_BUFFER_DTYPES))


def as_robot_buffers(buffers) -> tuple:
    """``robot_buffers`` for callers that already hold the 8-tuple."""
    buffers = tuple(buffers)
    if len(buffers) != len(ROBOT_BUFFER_DTYPES):
        raise ValueError(
            f"expected {len(ROBOT_BUFFER_DTYPES)} robot buffers "
            f"(twists, parent_tf, parent_idx, act_idx, mimic_mul, mimic_off, "
            f"mimic_act_idx, topo_inv), got {len(buffers)}")
    return robot_buffers(*buffers)
