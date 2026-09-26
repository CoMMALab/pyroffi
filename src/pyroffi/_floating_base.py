"""Floating-base configuration conventions.

A floating-base robot's configuration is ``cfg = [base (7) | joints (n)]``, the base being the
root link's world pose as ``wxyz_xyz``. Velocities, accelerations and generalized forces live in
the tangent space, ``[base (6) | joints (n)]``, where the base part is a spatial vector in the
*base body frame*, angular first (``[omega; v]``, Featherstone's convention). Integration is
right-multiplicative on SE(3):

    T_base(t + dt) = T_base(t) @ exp(v_base * dt)

which is pinocchio's free-flyer convention (pinocchio orders its base twist linear-first and its
quaternion ``xyzw``; convert when comparing). Fixed-base robots have ``cfg`` = joints and
``n_q == n_v == n``, and every function here is the identity split for them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jaxlie
from jax import Array
from jax import numpy as jnp

if TYPE_CHECKING:
    from ._robot import Robot

BASE_NQ = 7
BASE_NV = 6


def split_cfg(robot: "Robot", cfg: Array) -> tuple[Array | None, Array]:
    """``(base wxyz_xyz or None, joint cfg)``."""
    if not robot.floating_base:
        return None, cfg
    return cfg[..., :BASE_NQ], cfg[..., BASE_NQ:]


def split_tangent(robot: "Robot", v: Array) -> tuple[Array | None, Array]:
    """``(base spatial vector or None, joint part)`` of a velocity-like vector."""
    if not robot.floating_base:
        return None, v
    return v[..., :BASE_NV], v[..., BASE_NV:]


def join_cfg(base: Array, joints: Array) -> Array:
    return jnp.concatenate([base, joints], axis=-1)


def _twist_to_jaxlie(v6: Array) -> Array:
    """Angular-first ``[omega; v]`` -> jaxlie's ``[v; omega]`` tangent."""
    return jnp.concatenate([v6[..., 3:], v6[..., :3]], axis=-1)


def _twist_from_jaxlie(t6: Array) -> Array:
    return jnp.concatenate([t6[..., 3:], t6[..., :3]], axis=-1)


def integrate(robot: "Robot", cfg: Array, v: Array, dt: float | Array = 1.0) -> Array:
    """``cfg (+) v * dt``: base by the SE(3) exponential in the body frame, joints additively."""
    base, q = split_cfg(robot, cfg)
    vb, qd = split_tangent(robot, v)
    q_new = q + qd * dt
    if base is None:
        return q_new
    T = jaxlie.SE3(base) @ jaxlie.SE3.exp(_twist_to_jaxlie(vb * dt))
    return join_cfg(T.wxyz_xyz, q_new)


def difference(robot: "Robot", cfg0: Array, cfg1: Array) -> Array:
    """Tangent ``v`` with ``integrate(cfg0, v) == cfg1`` (``log(T0^-1 T1)`` for the base)."""
    b0, q0 = split_cfg(robot, cfg0)
    b1, q1 = split_cfg(robot, cfg1)
    if b0 is None:
        return q1 - q0
    vb = _twist_from_jaxlie((jaxlie.SE3(b0).inverse() @ jaxlie.SE3(b1)).log())
    return jnp.concatenate([vb, q1 - q0], axis=-1)


def neutral_base() -> Array:
    return jaxlie.SE3.identity().wxyz_xyz
