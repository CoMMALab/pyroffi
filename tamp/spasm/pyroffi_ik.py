"""pyroffi-backed replacement for SPaSM's hand-rolled Franka analytic IK.

``spasm.conversions.grasp_to_q`` implements the He & Liu (2021) Franka IK: a
closed form given q7, swept over a fan of 9 q7 values, keeping the valid branch
closest to a reference configuration. It is correct, fast, and hardcoded to the
Panda — every DH constant of that one arm is a literal in the source.

This module provides the same function backed by
:mod:`pyroffi.kinematics._analytic_ik`, which derives the identical decomposition
from *the robot model* instead. The interface is unchanged, so it is a drop-in:

    from spasm.pyroffi_ik import grasp_to_q      # instead of spasm.conversions

What that buys, concretely:

* **It is not Franka-specific.** The same code path serves the FR3 (or any arm
  with a spherical shoulder, intersecting axes 5-6 and an offset axis 7) with no
  new constants, because the geometry is read off the URDF at construction.
* **Near-misses report as near-misses.** The subproblem layer returns a
  least-squares minimiser with a flag rather than NaN when a target is
  fractionally out of reach, so an unreachable grasp is distinguishable from a
  numerically failed one — a distinction the task planner can act on.
* **Collision-free branch selection** is available (``collision_free=True``),
  choosing among the ~18 valid branches rather than returning the first one and
  leaving ``s-motion`` to discover it is unusable.

The geometry is built once at import and cached; per-call cost is the solve.
"""
from __future__ import annotations

import functools
import os

import jax
import jax.numpy as jnp
import numpy as np

from spasm import backend
from spasm.paths import SPASM_URDF

from pyroffi.kinematics import _analytic_ik as _aik

#: q7 fan matching ``spasm.conversions.grasp_to_q`` exactly, so any difference
#: between the two solvers is the decomposition and not the sweep density.
Q7_FAN = jnp.linspace(-2.8970, 2.8970, 9)

NEUTRAL_Q9 = jnp.array([0.0, -jnp.pi / 4, 0.0, -3 * jnp.pi / 4, 0.0,
                        jnp.pi / 2, jnp.pi / 4, 0.0, 0.0])

#: Flange-to-grasp offset SPaSM applies before solving. Kept identical so the
#: two solvers are handed the same target pose.
DIST_MOUNT_TO_GRASP = 0.0016

#: Link the grasp pose refers to. SPaSM's kinematics target the grasp frame.
EE_LINK = backend.EE_LINK


@functools.lru_cache(maxsize=1)
def _geometry():
    """Arm geometry for the SPaSM URDF, resolved once."""
    return _aik.build_geometry(backend.ROBOT, EE_LINK)


@functools.lru_cache(maxsize=1)
def _collision():
    """Collision model + flattened data, for the collision-free variant.

    Built with the SRDF: without it the spherized model's adjacent links overlap
    by construction and every configuration reports as colliding.
    """
    import os

    import yourdfpy

    import pyroffi as pk
    from pyroffi.kinematics import _analytic_collision as _ac

    srdf = os.path.join(os.path.dirname(SPASM_URDF), "panda.srdf")
    urdf = yourdfpy.URDF.load(SPASM_URDF, load_collision_meshes=True)
    rc = (pk.collision.RobotCollisionSpherized.from_urdf(urdf, srdf_path=srdf)
          if os.path.exists(srdf)
          else pk.collision.RobotCollisionSpherized.from_urdf(urdf))
    return rc, _ac.build_collision_data(backend.ROBOT, rc)


# Build the geometry eagerly at import. `build_geometry` calls np.asarray on FK
# results, and inside a jit trace *every* jnp operation produces a tracer — even
# on constant inputs — so resolving it lazily on first call explodes when that
# call happens under `jax.jit` (which is exactly how the oracle invokes IK).
_geometry()


def _target_matrix(grasp_pose):
    """``[x y z quat_xyzw]`` -> ``[4,4]``, with SPaSM's mount offset applied."""
    import jaxlie

    rot = jaxlie.SO3.from_quaternion_xyzw(grasp_pose[3:]).as_matrix()
    T = jnp.eye(4).at[:3, :3].set(rot).at[:3, 3].set(grasp_pose[:3])
    return T.at[:3, 3].set(T[:3, 3] - DIST_MOUNT_TO_GRASP * T[:3, 2])


def grasp_to_q(grasp_pose, nearest_q=NEUTRAL_Q9):
    """Drop-in for :func:`spasm.conversions.grasp_to_q`.

    Args:
        grasp_pose: ``(7,)`` ``[x y z quat_xyzw]``.
        nearest_q: ``(9,)`` reference; the returned branch is the valid one
            closest to it, matching SPaSM's continuity behaviour.

    Returns:
        ``(q (9,), error)`` where error is ``0.0`` on success and ``inf``
        otherwise — the same contract SPaSM's callers already branch on.
    """
    assert grasp_pose.shape == (7,), \
        f"Expected grasp_pose to be (7,), got {grasp_pose.shape}"

    T = _target_matrix(grasp_pose)
    q7, found = _aik.analytic_ik_solve_batched(
        backend.ROBOT, EE_LINK, T,
        q7_samples=Q7_FAN,
        previous_cfg=jnp.asarray(nearest_q)[:7],
        geometry=_geometry(),
        backend="jax",
        differentiable=False,
    )
    q9 = jnp.pad(q7, (0, 2), constant_values=0.0)
    return q9, jnp.where(found, 0.0, jnp.inf)


def grasp_to_q_collision_free(grasp_pose, world_geom=None, nearest_q=NEUTRAL_Q9):
    """As :func:`grasp_to_q`, but prefers a collision-free branch.

    Returns ``(q (9,), error, collision_free)``. Falls back to the
    maximum-clearance valid branch when nothing is collision-free, so the caller
    can tell "unreachable" from "reachable but blocked".
    """
    rc, _ = _collision()
    T = _target_matrix(grasp_pose)
    q7, found, free, _clr = _aik.analytic_ik_solve_collision_free(
        backend.ROBOT, EE_LINK, T, rc, world_geom,
        q7_samples=Q7_FAN,
        previous_cfg=jnp.asarray(nearest_q)[:7],
        geometry=_geometry(),
    )
    q9 = jnp.pad(q7, (0, 2), constant_values=0.0)
    return q9, jnp.where(found, 0.0, jnp.inf), free


def grasp_to_q_self_free(grasp_pose, nearest_q=NEUTRAL_Q9):
    """:func:`grasp_to_q` restricted to *self*-collision-free branches.

    This is the collision-free variant in the form a PDDLStream ``s-ik`` stream
    can actually use. Streams are context-free: ``s-ik`` is invoked with a block
    and a candidate pose, with no knowledge of where the *other* blocks are at
    that point in the eventual plan — in a rearrangement task they are all
    moving. Screening against movable obstacles there would be checking against
    a world state that may never occur.

    Self-collision has no such problem. An arm folded into itself is invalid
    regardless of what the rest of the scene is doing, and the branch enumerator
    produces such configurations routinely (the 8 branches per q7 are exactly
    the elbow-up/down, wrist-flip and shoulder-pair alternatives). So this
    screens self-collision only, and movable-obstacle checking stays where it
    belongs, in ``s-motion``.
    """
    rc, _ = _collision()
    T = _target_matrix(grasp_pose)
    q7, found, _free, _clr = _aik.analytic_ik_solve_collision_free(
        backend.ROBOT, EE_LINK, T, rc, None,      # world=None -> self only
        q7_samples=Q7_FAN,
        previous_cfg=jnp.asarray(nearest_q)[:7],
        geometry=_geometry(),
    )
    q9 = jnp.pad(q7, (0, 2), constant_values=0.0)
    return q9, jnp.where(found, 0.0, jnp.inf)


# Same eager-warm rationale as _geometry(): building the collision model inside
# a jit trace is impossible, and the oracle jits its IK.
if os.environ.get("PYROFFI_ANALYTIC_IK", "") == "pyroffi-cfree":
    _collision()
