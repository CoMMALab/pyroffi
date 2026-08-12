"""Collision constraints in the CUDA IK solvers.

The load-bearing test here is `test_sqp_self_collision_is_weight_independent`.

A *soft* penalty and a *hard* constraint look identical at a well-tuned weight —
both return collision-free solutions. They only diverge when the penalty is made
negligible: a soft term stops mattering, a hard constraint does not care. So the
sweep below deliberately includes a weight low enough to be worthless as a
penalty, and asserts SQP still returns feasible configurations there. Drop that
case and the suite can no longer tell the two apart, which is the whole point.

`ls` is the control: it has correct collision gradients but no constraint
mechanism, so it is expected to *fail* at low weight and pass at the default.
That asymmetry is the evidence the SQP result is real and not an artefact of the
harness.

Requires the spherized Panda plus its SRDF. Without an SRDF the spherized model
reports adjacent links as permanently overlapping and every configuration is
rejected, so the SRDF is not optional decoration.
"""

from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pytest
import yourdfpy

import pyroffi as pk
from pyroffi.collision import RobotCollisionSpherized, Sphere

@pytest.fixture(scope="module", autouse=True)
def _float32_regime():
    """Pin ``jax_enable_x64`` off for this module.

    The CUDA IK kernels are float32/int32 at their FFI boundary and reject
    64-bit operands outright (``expected S32 but got S64``). x64 is process-wide
    global state that other test modules turn on -- ``test_subproblems`` at
    module scope, ``pyroffi.toolbox`` sessions during a test -- so whether these
    tests see 32- or 64-bit inputs depends on collection order, not on anything
    they do. Declaring the regime is therefore part of the fixture, not a
    workaround.

    That the kernels break under x64 at all is a real limitation and is tracked
    separately; pinning here keeps THIS module testing collision constraints
    rather than re-discovering the dtype issue.

    Module-scoped and an explicit dependency of `panda`: the robot and its
    buffers are built once per module, and if that happens while x64 is on they
    are int64/float64 for good -- flipping the flag afterwards does not re-cast
    arrays that already exist.
    """
    import jax

    before = bool(jax.config.read("jax_enable_x64"))
    if before:
        jax.config.update("jax_enable_x64", False)
    try:
        yield
    finally:
        if bool(jax.config.read("jax_enable_x64")) != before:
            jax.config.update("jax_enable_x64", before)


RESOURCE_ROOT = pathlib.Path(__file__).resolve().parent.parent / "resources"
SPHERIZED_URDF = RESOURCE_ROOT / "panda" / "panda_spherized.urdf"
SRDF = RESOURCE_ROOT / "panda" / "panda.srdf"

# Deliberately spans "useless as a penalty" to "the shipped default". A hard
# constraint must hold across all of it; a soft one only at the top.
NEGLIGIBLE_WEIGHT = 1e0
DEFAULT_WEIGHT = 1e4

N_PROBLEMS = 32


@pytest.fixture(scope="module")
def panda(_float32_regime):
    if not SPHERIZED_URDF.exists() or not SRDF.exists():
        pytest.skip(f"spherized Panda + SRDF not found under {RESOURCE_ROOT}")
    urdf = yourdfpy.URDF.load(str(SPHERIZED_URDF))
    robot = pk.Robot.from_urdf(urdf)
    coll = RobotCollisionSpherized.from_urdf(urdf, srdf_path=str(SRDF))
    ee = len(robot.links.names) - 1
    return robot, coll, ee


@pytest.fixture(scope="module")
def reachable_targets(panda):
    """Targets generated from real configurations, so every one is reachable.

    A target the arm cannot reach would let a solver look collision-free for the
    wrong reason (it never gets near the obstacle), so poses come from forward
    kinematics on sampled in-limit configurations rather than from thin air.
    """
    robot, _, ee = panda
    rng = np.random.default_rng(0)
    lo = np.asarray(robot.joints.lower_limits)
    hi = np.asarray(robot.joints.upper_limits)
    q = lo + (hi - lo) * rng.random((N_PROBLEMS, lo.shape[0]))
    T = jax.vmap(lambda c: robot.forward_kinematics(c)[ee])(jnp.asarray(q))
    return jaxlie.SE3(T), lo.shape[0]


def _as_configs(solution, n_problems):
    cfg = np.asarray(getattr(solution, "cfg", solution))
    if cfg.ndim == 3:
        cfg = cfg[:, 0]
    assert cfg.shape[0] == n_problems
    return cfg


def _n_self_colliding(robot, coll, cfgs):
    d = coll.compute_self_collision_distance(robot, jnp.asarray(cfgs))
    return int((np.asarray(d).min(axis=-1) < 0).sum())


def _solve(solver, panda, targets, n_act, **kwargs):
    robot, _, ee = panda
    return solver(
        robot, ee, targets,
        rng_key=jax.random.PRNGKey(0),
        previous_cfgs=jnp.zeros((N_PROBLEMS, n_act)),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Self-collision
# ---------------------------------------------------------------------------

def test_self_collision_fires_at_all(panda, reachable_targets):
    """Passing a spherized checker must change the result.

    Guards the failure mode that cost this feature two sessions: results
    *identical* to the no-checker baseline meant the code was not executing, not
    that it had no effect.
    """
    from pyroffi.optimization_engines._ls_ik import ls_ik_solve_cuda_batch

    robot, coll, _ = panda
    targets, n_act = reachable_targets

    off = _as_configs(_solve(ls_ik_solve_cuda_batch, panda, targets, n_act,
                             collision_checker=None), N_PROBLEMS)
    on = _as_configs(_solve(ls_ik_solve_cuda_batch, panda, targets, n_act,
                            collision_checker=coll,
                            collision_weight=DEFAULT_WEIGHT), N_PROBLEMS)

    baseline = _n_self_colliding(robot, coll, off)
    assert baseline > 0, (
        "the no-checker baseline produced no self-collisions, so this fixture "
        "cannot detect whether the constraint does anything -- reseed it")
    assert not np.allclose(off, on), (
        "enabling the collision checker left the solutions byte-identical, "
        "which means the self-collision path did not execute")


def test_sqp_self_collision_is_weight_independent(panda, reachable_targets):
    """SQP's constraint is hard: it holds even when the penalty is negligible."""
    from pyroffi.optimization_engines._sqp_ik import sqp_ik_solve_cuda_batch

    robot, coll, _ = panda
    targets, n_act = reachable_targets

    for weight in (NEGLIGIBLE_WEIGHT, DEFAULT_WEIGHT):
        cfgs = _as_configs(
            _solve(sqp_ik_solve_cuda_batch, panda, targets, n_act,
                   collision_checker=coll, collision_weight=weight),
            N_PROBLEMS)
        n_bad = _n_self_colliding(robot, coll, cfgs)
        assert n_bad == 0, (
            f"sqp returned {n_bad}/{N_PROBLEMS} self-colliding configurations at "
            f"collision_weight={weight:g}. At the low weight this means the "
            f"constraint has degraded to a penalty; at the default it means the "
            f"constraint is broken outright.")


def test_ls_self_collision_is_a_soft_penalty(panda, reachable_targets):
    """LS is the control, and documents its own ceiling.

    LS has correct collision gradients but no constraint mechanism, so it cleans
    up at the default weight and degrades as the weight falls. Only the default
    case is asserted as an equality: pinning the low-weight count would make the
    suite fail the day LS is improved, which is backwards. What IS asserted is
    the ordering -- low weight must be no better than default -- because the SQP
    test above leans on that contrast being real and not a fixture artefact.
    """
    from pyroffi.optimization_engines._ls_ik import ls_ik_solve_cuda_batch

    robot, coll, _ = panda
    targets, n_act = reachable_targets

    def n_bad(weight):
        return _n_self_colliding(robot, coll, _as_configs(
            _solve(ls_ik_solve_cuda_batch, panda, targets, n_act,
                   collision_checker=coll, collision_weight=weight),
            N_PROBLEMS))

    at_default = n_bad(DEFAULT_WEIGHT)
    assert at_default == 0, (
        f"ls left {at_default}/{N_PROBLEMS} self-colliding at the DEFAULT weight; "
        f"its collision term reaches the normal equations, so it should clear here")

    assert n_bad(NEGLIGIBLE_WEIGHT) >= at_default, (
        "ls did no worse at a negligible collision_weight than at the default, so "
        "this fixture no longer distinguishes a penalty from a constraint -- the "
        "weight-independence test for sqp is only meaningful against this contrast")


# ---------------------------------------------------------------------------
# World obstacles
# ---------------------------------------------------------------------------

def test_world_obstacle_is_avoided_and_margin_is_respected(panda, reachable_targets):
    """Obstacle avoidance, and the shape of the converged answer.

    The clearance assertion is the interesting half. A solver that merely
    *rejects* colliding candidates lands wherever it happens to; one that
    descends a correct constraint gradient settles ON the margin. Asserting
    clearance is bounded below by the margin (not just positive) is what
    distinguishes those.
    """
    from pyroffi.optimization_engines._ls_ik import ls_ik_solve_cuda_batch

    robot, coll, _ = panda
    targets, n_act = reachable_targets

    margin = 0.02
    obstacle = Sphere.from_center_and_radius(
        jnp.array([[0.45, 0.0, 0.45]]), jnp.array([0.14]))

    def clearance(cfgs):
        d = coll.compute_world_collision_distance(robot, jnp.asarray(cfgs), obstacle)
        return np.asarray(d).reshape(len(cfgs), -1).min(axis=-1)

    off = _as_configs(_solve(ls_ik_solve_cuda_batch, panda, targets, n_act,
                             collision_checker=None), N_PROBLEMS)
    assert (clearance(off) < 0).sum() > 0, (
        "no configuration hits the obstacle without collision enabled, so this "
        "test cannot detect avoidance -- move or grow the obstacle")

    on = _as_configs(
        _solve(ls_ik_solve_cuda_batch, panda, targets, n_act,
               collision_checker=coll, collision_world=obstacle,
               collision_free=True, collision_margin=margin,
               collision_weight=1e6),
        N_PROBLEMS)
    d_on = clearance(on)

    assert (d_on < 0).sum() == 0, (
        f"{int((d_on < 0).sum())}/{N_PROBLEMS} configurations penetrate the obstacle")
    assert d_on.min() >= margin - 1e-3, (
        f"min clearance {d_on.min():.4f} m sits inside the {margin} m margin; the "
        f"solver is avoiding contact but not respecting the requested standoff")
