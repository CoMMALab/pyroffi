"""Analytic 7-DOF IK, behind the same interface as the iterative CUDA solvers.

The kernel underneath is a different animal: closed-form rather than iterative,
sweeping the redundancy parameter q7 instead of refining seeds, and
deterministic. But a caller choosing between solvers should not have to learn a
second calling convention to try it, so this module gives it the signature every
other CUDA solver has -- ``(robot, target_link_indices, target_poses, rng_key,
previous_cfg, ...)`` returning a differentiable solution.

Two places where the shared signature is a deliberate lie, both documented at
the argument rather than left to be discovered:

``rng_key`` is accepted and IGNORED. The analytic solve has nothing to
randomise: it enumerates branches over a fixed q7 sweep and picks among them.
Dropping the argument would make this the one solver you cannot substitute into
a generic call site, which costs more than the honesty is worth -- so it is
taken and its irrelevance is stated.

``num_seeds`` maps onto the q7 SWEEP RESOLUTION rather than a seed count.
Nothing is seeded; the parameter controls how finely the redundant degree of
freedom is sampled. Same knob shape, different meaning, and pretending
otherwise would make the two incomparable in a benchmark.

Collision follows the same three paths as the rest of the suite: self-collision
alone (path 1), plus world obstacles (path 2), and arbitrary JAX constraints by
null-space projection afterwards (path 3, see ``_nullspace``).

REQUIRES ``jax_enable_x64``. This kernel is the one solver in the suite that
takes float64 buffers -- the closed-form solve chains trigonometric branch
selection where float32 loses the sign of a nearly-degenerate quantity, and the
iterative solvers can absorb that error in another iteration while this one
cannot. The other four cast to float32 at their FFI boundary and so run under
either setting, which means the two CAN share a process, but only with x64 ON.
With it off, the kernel rejects its own operands and the error names buffer
indices rather than the cause, so `_require_x64` raises something readable
first.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
from jax import Array
from jaxtyping import Float

from .._robot import Robot
from ..cuda_kernels.ik._analytic_ik_cuda import (
    _Coll,
    _analytic_ik_cuda_raw,
    _empty_collision_buffers,
    _empty_world,
    analytic_ik_cuda,
    pack_geometry,
    world_arrays,
)
from ..kinematics._analytic_collision import CollisionData, build_collision_data
from ..kinematics._analytic_ik import build_geometry, default_err_tol, default_q7_samples
from ._batching import make_sharded_pmap, run_sharded, sharding_enabled
from ._implicit_diff import (
    detached_robot,
    differentiable_ik_solution,
    differentiable_ik_solution_batch,
)

#: Geometry blobs keyed on (robot identity, end-effector link). Building one
#: walks the URDF and packs 95 float64s; it depends only on the kinematic chain,
#: so recomputing it per call would be pure overhead. Holds a reference to the
#: robot so the id it keys on cannot be recycled onto a different object.
_GEOM_CACHE: dict = {}


def _require_x64() -> None:
    """Fail readably when x64 is off, instead of at the FFI boundary.

    Without this the kernel reports ``Failed to decode all FFI handler operands
    (bad operands at: 1, 2, 5) ... expected F64 but got F32``, which names
    buffer positions and not the one-line fix. JAX also only WARNS when a
    float64 request is truncated, so nothing upstream stops it either.
    """
    if not bool(jax.config.read("jax_enable_x64")):
        raise RuntimeError(
            "analytic IK requires 64-bit JAX: set jax.config.update("
            "'jax_enable_x64', True) or JAX_ENABLE_X64=1 before use. Its "
            "closed-form branch selection is float64-only, unlike the "
            "iterative CUDA solvers, which cast to float32 and run under "
            "either setting.")


def _geometry_for(robot: Robot, ee_link_index: int):
    """Geometry blob for this chain, cached on a key that survives tracing.

    Keyed on the LINK NAMES rather than ``id(robot)``. Under ``jax.grad`` with
    respect to robot parameters the model arrives as tracers, a fresh object per
    call, so an identity key misses every time -- and rebuilding is not merely
    slow but impossible: ``build_geometry`` runs ``detect()``, a discrete
    structural classification (which axes are concurrent), on numpy arrays. It
    raises TracerArrayConversionError on a traced robot.

    Reusing the blob under differentiation is CORRECT, not a shortcut. The
    kernel only has to produce q*; the gradient comes from the implicit rule
    evaluated on the pure-JAX residual with the LIVE parameters. The solver's
    internal copy of the geometry never needs to track the perturbation -- the
    same reason `detached_robot` exists for the iterative solvers.

    Structure is not differentiable anyway: perturbing a twist does not change
    which axes intersect, and if it did, the closed-form solution would not
    apply at all.
    """
    key = (tuple(robot.links.names), int(ee_link_index))
    entry = _GEOM_CACHE.get(key)
    if entry is not None:
        return entry
    try:
        geom = build_geometry(robot, robot.links.names[int(ee_link_index)])
    except Exception as exc:                       # tracer, or a bad chain
        raise RuntimeError(
            "analytic IK could not build its geometry. Under jax.grad with "
            "respect to robot parameters the model is traced and the structural "
            "analysis cannot run on it, so the blob must already be cached: "
            "call the solver once outside the differentiated function with the "
            "same robot to warm it."
        ) from exc
    entry = _GEOM_CACHE[key] = (pack_geometry(geom), geom)
    return entry


#: Analytic collision data, cached on the same tracing-safe key as the geometry.
_COLL_CACHE: dict = {}


def _analytic_collision(robot, checker):
    """Adapt the suite's ``RobotCollisionSpherized`` to analytic's CollisionData.

    Analytic's kernel consumes a flattened home-frame sphere list, not the
    spherized checker the other four solvers take. Accepting ``collision_checker``
    in the signature and then requiring a different type is exactly the kind of
    inconsistency standardising this solver was meant to remove -- a caller who
    passes what works everywhere else got ``AttributeError: 'RobotCollisionSpherized'
    object has no attribute 'spheres_home'``.

    Cached on link names for the same reason the geometry is: under jax.grad the
    robot arrives as tracers, and build_collision_data walks it with numpy.
    """
    if checker is None:
        return None
    if isinstance(checker, CollisionData):
        return checker
    key = (tuple(robot.links.names), id(type(checker)))
    hit = _COLL_CACHE.get(key)
    if hit is None:
        hit = _COLL_CACHE[key] = build_collision_data(robot, checker)
    return hit


def _as_matrix(target_poses: jaxlie.SE3) -> Array:
    """``SE3`` -> the ``[B, 4, 4]`` homogeneous form the kernel expects."""
    return jnp.asarray(target_poses.as_matrix(), dtype=jnp.float64)


def _solve(robot, ee, targets_se3, previous_cfgs, num_seeds, err_tol,
           collision, world, respect_limits):
    """Shared core: both entry points differ only in batch shape."""
    geom_blob, geom = _geometry_for(robot, ee)
    q7 = default_q7_samples(geom, int(num_seeds))
    q, err, found, clearance = analytic_ik_cuda(
        geom_blob, _as_matrix(targets_se3), q7, previous_cfgs,
        collision=_analytic_collision(robot, collision),
        world_spheres=world,
        respect_limits=bool(respect_limits),
        err_tol=float(err_tol if err_tol is not None else default_err_tol()))
    return q, err, found, clearance


def analytic_ik_solve_cuda(
    robot:               Robot,
    target_link_indices: int | tuple[int, ...],
    target_poses:        jaxlie.SE3,
    rng_key:             Array | None = None,
    previous_cfg:        Float[Array, "n_act"] | None = None,
    num_seeds:           int = 32,
    err_tol:             float | None = None,
    respect_limits:      bool = True,
    collision_checker:   Any | None = None,
    collision_world:     Any | None = None,
) -> Float[Array, "n_act"]:
    """Solve one pose analytically. See the module docstring on ``rng_key``.

    ``target_link_indices`` accepts an int or a 1-tuple. Multiple end-effectors
    are refused rather than silently solved for the first: a closed-form 7-DOF
    solve addresses ONE chain, and quietly ignoring the rest would return a
    confident answer to a different question.
    """
    _require_x64()
    ee = _single_ee(target_link_indices)
    del rng_key                       # deterministic; see module docstring

    # Detached for the kernel; the LIVE robot and target reach the implicit rule
    # below, which is what carries the gradient. Nothing with a live tangent may
    # cross the FFI boundary or linearisation fails.
    _robot_k = detached_robot(robot)
    _tgt_k = jax.tree.map(jax.lax.stop_gradient, target_poses)

    targets = jax.tree.map(lambda x: x[None] if x.ndim == 1 else x, _tgt_k)
    prev = None if previous_cfg is None else jnp.asarray(previous_cfg).reshape(1, -1)
    q, _err, _found, _clr = _solve(_robot_k, ee, targets, prev, num_seeds, err_tol,
                                   collision_checker, collision_world, respect_limits)
    return differentiable_ik_solution(q[0], robot, (ee,), target_poses)


def analytic_ik_solve_cuda_batch(
    robot:               Robot,
    target_link_indices: int | tuple[int, ...],
    target_poses:        jaxlie.SE3,
    rng_key:             Array | None = None,
    previous_cfgs:       Float[Array, "n_problems n_act"] | None = None,
    num_seeds:           int = 32,
    err_tol:             float | None = None,
    respect_limits:      bool = True,
    collision_checker:   Any | None = None,
    collision_world:     Any | None = None,
) -> Float[Array, "n_problems n_act"]:
    """Solve a batch of poses analytically, sharded across local GPUs.

    Sharding splits the problem axis exactly as it does for the iterative
    solvers, and is skipped below the threshold where the pmap's split and
    gather cost more than the parallel solve saves.

    Collision handling here is SELECTION, not optimisation, and that makes its
    guarantee weaker than the iterative solvers'. Analytic IK solves the pose
    exactly and its only freedom is the redundancy parameter q7, so it can
    reject colliding q7 samples but cannot push away from an obstacle. When
    every sample collides it returns the least-bad one rather than failing.

    MEASURED, panda + a 0.3 m box, 256 targets drawn from random reference
    configurations (58 of which are themselves inside the box):

        analytic, self only          66/256 in obstacle, pose <1mm 255/256
        analytic, self + world       33/256 in obstacle, pose <1mm 255/256
        ls, collision_free            0/256 in obstacle, pose <1mm 205/256

    Passing `collision_world` halves obstacle hits, and raising `num_seeds`
    from 16 to 1024 only moves it 34 -> 30 -- the residual is targets with no
    collision-free q7 at all, not an under-sampled sweep. `ls` clears them by
    giving up the pose on a fifth of the batch, which is the trade analytic
    structurally cannot make. Use analytic when the pose is hard and collision
    is a preference; use sqp when collision is the hard constraint.
    """
    # Detached for the kernel; the live `robot` reaches the implicit rule,
    # which is what carries dq*/dtheta. See detached_robot.
    _require_x64()
    ee = _single_ee(target_link_indices)
    del rng_key

    # Detached for the kernel; the LIVE robot and target reach the implicit rule
    # below, which is what carries the gradient.
    _robot_k = detached_robot(robot)
    _tgt_k = jax.tree.map(jax.lax.stop_gradient, target_poses)

    n_problems = target_poses.wxyz_xyz.shape[0]
    n_act = robot.joints.num_actuated_joints
    prev = (jnp.zeros((n_problems, n_act), jnp.float32)
            if previous_cfgs is None else jnp.asarray(previous_cfgs))

    n_devices = jax.local_device_count()
    # MEASURED crossover, 4x A5000, against one device:
    #     B=4096     79.3 -> 177.7 ms   2.2x SLOWER
    #     B=32768   603.8 -> 474.4 ms   1.27x
    #     B=131072 2404.8 -> 1555.3 ms  1.55x
    # Far higher than the iterative solvers' because this kernel is roughly 7x
    # faster per problem (51.6 kIK/s against ~14), so the fixed pmap cost --
    # padding, the gather, re-broadcasting geometry to every device -- takes
    # that much more work to amortise. Sharding this at the suite default of 512
    # would have made it more than twice as slow.
    if sharding_enabled(n_problems, n_devices, "PYROFFI_ANALYTIC_IK_PMAP_MIN",
                        min_problems=32768):
        # Geometry and the q7 sweep are built OUTSIDE the pmap. They are static
        # per (robot, end-effector) and building one walks the URDF with numpy;
        # passing `robot` in as a broadcast argument makes it a tracer and that
        # walk dies with TracerArrayConversionError. Hoisting is also simply
        # correct -- there is nothing per-device about the kinematic chain.
        geom_blob, geom = _geometry_for(robot, ee)
        q7 = default_q7_samples(geom, int(num_seeds))
        tol = float(err_tol if err_tol is not None else default_err_tol())

        # Collision geometry travels as RAW ARRAYS rather than as the checker
        # object: pmap traces once, so a captured object's arrays would become
        # compile-time constants and a later call with a different scene would
        # silently keep solving against the first one. The same reason the
        # iterative solvers pass their collision buffers as broadcast args.
        sph_home, sph_joint, pairs = _collision_arrays(collision_checker)
        world = _world_arrays(collision_world)

        def _body(_rng, prev_sh, target_wxyz, geom_b, q7_s,
                  sph_h, sph_j, prs, w_sph, w_cap, w_box, w_hs):
            coll = _Coll(sph_h, sph_j, prs) if collision_checker is not None else None
            q_, _e, _f, _c = _analytic_ik_cuda_raw(
                geom_b, jaxlie.SE3(target_wxyz).as_matrix(), q7_s, prev_sh,
                coll, (w_sph, w_cap, w_box, w_hs),
                respect_limits=bool(respect_limits), err_tol=tol)
            return q_

        pmapped = make_sharded_pmap(_body, 9)
        q = run_sharded(pmapped, target_poses, jax.random.PRNGKey(0), prev,
                        n_devices, geom_blob, q7, sph_home, sph_joint, pairs, *world)
    else:
        q, _err, _found, _clr = _solve(
            _robot_k, ee, _tgt_k, prev, num_seeds, err_tol,
            collision_checker, collision_world, respect_limits)

    return differentiable_ik_solution_batch(q, robot, (ee,), target_poses)


def _collision_arrays(checker):
    """Raw self-collision buffers, or the cached empties when disabled."""
    if checker is None:
        return _empty_collision_buffers()
    return (jnp.asarray(checker.spheres_home, jnp.float64),
            jnp.asarray(checker.sphere_joint, jnp.int32),
            jnp.asarray(checker.self_pairs, jnp.int32).reshape(-1, 2))


def _world_arrays(world):
    """Obstacle buffers, shared with the single-problem FFI path.

    The sharded path needs these as raw arrays (a captured checker object would
    be frozen into the trace), while the plain path marshals them inside the FFI
    wrapper. One flattener serves both so the two cannot disagree about what an
    obstacle argument means.
    """
    return world_arrays(world)


def _single_ee(target_link_indices: int | Sequence[int]) -> int:
    """One end-effector, or a clear refusal."""
    if isinstance(target_link_indices, (int, np.integer)):
        return int(target_link_indices)
    idx = tuple(target_link_indices)
    if len(idx) != 1:
        raise NotImplementedError(
            f"analytic IK solves ONE 7-DOF chain in closed form; got "
            f"{len(idx)} end-effectors. Use an iterative solver (ls/sqp/mppi/"
            f"hjcd) for multi-EE targets.")
    return int(idx[0])
