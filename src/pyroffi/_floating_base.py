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

import jax
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


def fixed_tree(robot: "Robot") -> "Robot":
    """``robot``'s joint tree as a fixed-base robot, i.e. seen from its own base frame (for
    tools that model one chain relative to its root: DH extraction, QuIK, analytic IK)."""
    if not robot.floating_base:
        return robot
    import jax_dataclasses as jdc

    return jdc.replace(robot, floating_base=False, default_cfg=robot.default_cfg[BASE_NQ:])


# ---------------------------------------------------------------------------
# Chart: the floating robot seen by joint-space solvers
# ---------------------------------------------------------------------------
#
# Solvers written for a joint vector (seeding, clipping, coordinate descent, sampling noise,
# smoothness stencils, and every CUDA kernel's `q += step`) run unchanged on the *chart robot*:
# a fixed-base robot whose tree hangs off six extra joints appended after the real ones,
#
#     T_base = Trans(x, y, z) @ R_ref @ Rz(yaw) @ Ry(pitch) @ Rx(roll),
#
# so its configuration is ``[joints (n) | x y z yaw pitch roll]``. The chart is exact: every
# base pose has chart coordinates and maps back to the same SE(3) pose. Its only singularity is
# pitch = +-pi/2 *relative to R_ref*, so callers centre it on the warm start (``ref``) and the
# solver starts at pitch 0; pitch is clipped short of the singularity. Translation limits are
# +-inf (the base is unbounded) unless the caller gives a base region; `sample_in_limits` and
# `finite_range` handle infinite limits wherever solvers sample or scale by the joint range.
#
# ponytail: one chart per call. A solve whose base pitches ~90 deg away from its warm start
# loses conditioning; the upgrade is re-centring inside the solver loop (a device retraction).

CHART_JOINTS = ("__base_x", "__base_y", "__base_z", "__base_yaw", "__base_pitch", "__base_roll")
_CHART_TWISTS = ((1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0),
                 (0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 1, 0), (0, 0, 0, 1, 0, 0))
# Evaluation chart only: six more joints, held at zero, whose Jacobian columns are the base
# twist's [omega; v] in the body frame (revolute then prismatic about the base's own axes).
TANGENT_JOINTS = ("__base_wx", "__base_wy", "__base_wz", "__base_vx", "__base_vy", "__base_vz")
_TANGENT_TWISTS = ((0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 0, 1),
                   (1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0))
_PITCH_MAX = float(jnp.pi / 2 - 0.05)
_YAW_JOINT = 3  # the chart joint whose parent transform carries R_ref


def _chart_structure(robot: "Robot", k: int):
    """Concrete tree of the chart robot with ``k`` virtual joints (cached per robot)."""
    import numpy as onp

    backends = robot._require_backends()
    cache = backends.__dict__.setdefault("_chart_structure", {})
    if k in cache:
        return cache[k]
    from ._cuda_backends import CudaBackends

    n = robot.joints.num_joints
    parents = onp.asarray(backends._parent_indices).copy()
    parents[parents == -1] = n + k - 1
    parents = onp.concatenate([parents, [-1], onp.arange(n, n + k - 1)]).astype(onp.int32)
    link_parents = onp.asarray(backends._parent_joint_indices).copy()
    link_parents[link_parents == -1] = n + k - 1
    depth = onp.zeros(n + k, onp.int32)
    for _ in range(n + k):  # relax to the fixed point; trees are shallow
        depth = onp.where(parents == -1, 0, depth[parents] + 1)
    topo = onp.argsort(depth, kind="stable").astype(onp.int32)  # parents precede children
    starts = onp.searchsorted(depth[topo], onp.arange(depth.max() + 2)).astype(onp.int32)
    chart_backends = CudaBackends(backends.urdf, parent_indices=parents,
                                  parent_joint_indices=link_parents, num_joints=n + k)
    chart_backends._chart_of = backends
    cache[k] = (parents, link_parents, topo, starts, chart_backends)
    return cache[k]


def chart_robot(robot: "Robot", ref: Array, base_region=None, tangent: bool = False) -> "Robot":
    """The fixed-base chart of floating ``robot``, centred on base pose ``ref`` (wxyz_xyz).

    ``base_region`` = ``(lo, hi)``, world-frame xyz bounds for the base position. It bounds the
    chart's translation joints, so seeds are drawn from it and solutions stay in it; without it
    the translation is unbounded (+-inf limits). ``tangent`` appends the six zero-valued
    `TANGENT_JOINTS` (see `to_eval_chart`)."""
    import jax_dataclasses as jdc

    k = 12 if tangent else 6
    parents, link_parents, topo, starts, chart_backends = _chart_structure(robot, k)
    J = robot.joints
    n, m = J.num_joints, J.num_actuated_joints
    dt = J.twists.dtype
    ref_tf = jnp.concatenate([jnp.asarray(ref, dt)[:4], jnp.zeros(3, dt)])
    parent_tf = jnp.concatenate([J.parent_transforms,
                                 jnp.tile(neutral_base().astype(dt), (k, 1))])
    parent_tf = parent_tf.at[n + _YAW_JOINT].set(ref_tf)
    if base_region is None:
        lo_xyz, hi_xyz = jnp.full(3, -jnp.inf, dt), jnp.full(3, jnp.inf, dt)
    else:
        lo_xyz, hi_xyz = (jnp.broadcast_to(jnp.asarray(b, dt), (3,)) for b in base_region)
    lo = jnp.concatenate([lo_xyz, jnp.asarray([-jnp.pi, -_PITCH_MAX, -jnp.pi], dt),
                          jnp.full(k - 6, -jnp.inf, dt)])
    hi = jnp.concatenate([hi_xyz, jnp.asarray([jnp.pi, _PITCH_MAX, jnp.pi], dt),
                          jnp.full(k - 6, jnp.inf, dt)])
    vel = jnp.full(k, jnp.inf, dt)
    names = CHART_JOINTS + (TANGENT_JOINTS if tangent else ())
    twists = _CHART_TWISTS + (_TANGENT_TWISTS if tangent else ())
    joints = jdc.replace(
        J,
        num_joints=n + k,
        num_actuated_joints=m + k,
        names=J.names + names,
        actuated_names=J.actuated_names + names,
        twists=jnp.concatenate([J.twists, jnp.asarray(twists, dt)]),
        parent_transforms=parent_tf,
        parent_indices=jnp.asarray(parents),
        actuated_indices=jnp.concatenate([J.actuated_indices,
                                          jnp.arange(m, m + k, dtype=J.actuated_indices.dtype)]),
        lower_limits=jnp.concatenate([J.lower_limits, lo]),
        upper_limits=jnp.concatenate([J.upper_limits, hi]),
        velocity_limits=jnp.concatenate([J.velocity_limits, vel]),
        lower_limits_all=jnp.concatenate([J.lower_limits_all, lo]),
        upper_limits_all=jnp.concatenate([J.upper_limits_all, hi]),
        velocity_limits_all=jnp.concatenate([J.velocity_limits_all, vel]),
        mimic_multiplier=jnp.concatenate([J.mimic_multiplier, jnp.ones(k, dt)]),
        mimic_offset=jnp.concatenate([J.mimic_offset, jnp.zeros(k, dt)]),
        mimic_act_indices=jnp.concatenate([J.mimic_act_indices,
                                           jnp.full(k, -1, J.mimic_act_indices.dtype)]),
        _topo_sort_inv=jnp.asarray(topo),
        fk_level_starts=jnp.asarray(starts),
        fk_level_joints=jnp.asarray(topo),
        num_fk_levels=len(starts) - 1,
    )
    links = jdc.replace(robot.links, parent_joint_indices=jnp.asarray(link_parents))
    chart = jdc.replace(robot, joints=joints, links=links, dynamics=None,
                        floating_base=False, _backends=chart_backends)
    q0 = to_chart(robot, robot.default_cfg, ref)
    if tangent:
        q0 = jnp.concatenate([q0, jnp.zeros(6, q0.dtype)])
    return jdc.replace(chart, default_cfg=q0)


def eval_chart(robot: "Robot") -> "Robot":
    """The chart evaluators (collision checkers) run on: centred on the world frame, with the
    tangent joints. Static per robot (cached, so kernel buffers and jit caches keyed on it
    stay put)."""
    backends = robot._require_backends()
    cached = backends.__dict__.get("_eval_chart")
    if cached is not None:
        return cached
    with jax.ensure_compile_time_eval():
        chart = chart_robot(robot, neutral_base(), tangent=True)
    if not any(isinstance(x, jax.core.Tracer) for x in jax.tree_util.tree_leaves(chart)):
        backends._eval_chart = chart
    return chart


def to_eval_chart(robot: "Robot", cfg: Array) -> Array:
    """Floating ``cfg`` -> `eval_chart` configuration, for exact values *and* exact gradients.

    The pose goes through the Euler chart with its derivative stopped, and the tangent joints
    carry ``log(stop_gradient(T)^-1 T)``: zero in value, the base's body twist in derivative. So a
    kernel's Jacobian w.r.t. the tangent joints is its gradient w.r.t. the base twist, with no
    Euler singularity anywhere."""
    base = cfg[..., :BASE_NQ]
    T = jaxlie.SE3(base)
    tw = (jaxlie.SE3(jax.lax.stop_gradient(base)).inverse() @ T).log()  # jaxlie [v; omega]
    return jnp.concatenate([jax.lax.stop_gradient(to_chart(robot, cfg, neutral_base()))
                            .at[..., :-6].set(cfg[..., BASE_NQ:]),
                            _twist_from_jaxlie(tw)], axis=-1)


def _yaw_pitch_roll(R: jaxlie.SO3) -> Array:
    """``(yaw, pitch, roll)`` with ``R = Rz(yaw) Ry(pitch) Rx(roll)``, exact to float precision
    everywhere: pitch by atan2 (not asin), and roll read off the residual ``(Rz Ry)^-1 R`` so the
    yaw/roll ambiguity near pitch = +-pi/2 cancels instead of amplifying rounding."""
    M = R.as_matrix()
    yaw = jnp.arctan2(M[..., 1, 0], M[..., 0, 0])
    pitch = jnp.arctan2(-M[..., 2, 0], jnp.hypot(M[..., 0, 0], M[..., 1, 0]))
    res = (jaxlie.SO3.from_z_radians(yaw) @ jaxlie.SO3.from_y_radians(pitch)).inverse() @ R
    Mr = res.as_matrix()
    roll = jnp.arctan2(Mr[..., 2, 1], Mr[..., 1, 1])
    return jnp.stack([yaw, pitch, roll], axis=-1)


def to_chart(robot: "Robot", cfg: Array, ref: Array) -> Array:
    """Floating ``cfg`` -> chart configuration ``[joints | x y z yaw pitch roll]``."""
    base, q = split_cfg(robot, cfg)
    T = jaxlie.SE3(base)
    rel = jaxlie.SO3(jnp.broadcast_to(ref[..., :4], base[..., :4].shape)).inverse() @ T.rotation()
    return jnp.concatenate([q, T.translation(), _yaw_pitch_roll(rel)], axis=-1)


def from_chart(robot: "Robot", q_chart: Array, ref: Array) -> Array:
    """Chart configuration -> floating ``cfg = [base wxyz_xyz | joints]``."""
    n = robot.joints.num_actuated_joints
    q, xyz, ypr = q_chart[..., :n], q_chart[..., n:n + 3], q_chart[..., n + 3:n + 6]
    R = jaxlie.SO3(jnp.broadcast_to(ref[..., :4], (*ypr.shape[:-1], 4))) @ jaxlie.SO3.from_rpy_radians(
        ypr[..., 2], ypr[..., 1], ypr[..., 0])
    return join_cfg(jaxlie.SE3.from_rotation_and_translation(R, xyz).wxyz_xyz, q)


def curve_tangents(q: Array, qd: Array, qdd: Array) -> tuple[Array, Array]:
    """Velocity and acceleration of the curve through floating ``q`` with configuration-space
    derivatives ``qd``, ``qdd`` (the base's 7 ambient coordinates, e.g. a spline of poses), as
    tangent vectors ``[base twist | joints]``: the body twist ``(R^T dR, R^T dp)`` and its rate
    ``(dR^T dR + R^T ddR, dR^T dp + R^T ddp)``. (Not via ``SE3.log``: jaxlie's small-angle branch
    gets the second derivative wrong.)"""
    def rot(t):
        w = q[..., :4] + t * qd[..., :4] + 0.5 * t * t * qdd[..., :4]
        return jaxlie.SO3(w / jnp.linalg.norm(w, axis=-1, keepdims=True)).as_matrix()

    one = jnp.ones((), q.dtype)
    d1 = lambda t: jax.jvp(rot, (t,), (one,))[1]
    t0 = jnp.zeros((), q.dtype)
    R, dR, ddR = rot(t0), d1(t0), jax.jvp(d1, (t0,), (one,))[1]
    Rt, dRt = jnp.swapaxes(R, -1, -2), jnp.swapaxes(dR, -1, -2)
    vee = lambda W: jnp.stack([W[..., 2, 1], W[..., 0, 2], W[..., 1, 0]], axis=-1)
    mv = lambda M, x: jnp.einsum("...ij,...j->...i", M, x)
    dp, ddp = qd[..., 4:7], qdd[..., 4:7]
    vel = jnp.concatenate([vee(Rt @ dR), mv(Rt, dp), qd[..., BASE_NQ:]], axis=-1)
    acc = jnp.concatenate([vee(dRt @ dR + Rt @ ddR), mv(dRt, dp) + mv(Rt, ddp),
                           qdd[..., BASE_NQ:]], axis=-1)
    return vel, acc


def sample_in_limits(key: Array, shape: tuple[int, ...], lower: Array, upper: Array,
                     fallback: Array) -> Array:
    """Uniform samples in ``[lower, upper]``; an unbounded coordinate (infinite limit) takes
    ``fallback`` (the warm start) instead, since a uniform draw over it does not exist."""
    u = jax.random.uniform(key, shape, dtype=lower.dtype)
    bounded = jnp.isfinite(lower) & jnp.isfinite(upper)
    lo, hi = jnp.where(bounded, lower, 0.0), jnp.where(bounded, upper, 0.0)
    return jnp.where(bounded, lo + u * (hi - lo), jnp.broadcast_to(fallback, shape))


def finite_range(lower: Array, upper: Array, unbounded: float = 1.0) -> Array:
    """``upper - lower``, with ``unbounded`` (a length scale) where a limit is infinite."""
    return jnp.where(jnp.isfinite(lower) & jnp.isfinite(upper), upper - lower, unbounded)
