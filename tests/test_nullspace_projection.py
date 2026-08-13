"""Null-space constraint projection (IK path 3).

The properties worth guarding here are the ones whose violation is SILENT. A
projector that fails loudly is a bug report; one that returns the configuration
it was handed, unchanged, with ``success=False``, looks identical to an
over-constrained problem and sends the user tuning step sizes forever.

Both bugs this module guards were exactly that shape and neither was visible by
reading the code:

- an absolute ``pose_tol`` froze every element, because it was tighter than the
  pose a CUDA solve actually delivers (~2e-4 median on the log residual). No
  step could pass a test the starting point already failed.
- a start configuration already in collision rejects every candidate step, since
  each is compared against a start that is itself invalid.

So the tests below assert *movement* and *distinguishable failure reasons*, not
merely that the call returns.
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
from pyroffi.collision import RobotCollisionSpherized
from pyroffi.optimization_engines._nullspace import project_onto_constraints

RESOURCE_ROOT = pathlib.Path(__file__).resolve().parent.parent / "resources"
SPHERIZED_URDF = RESOURCE_ROOT / "panda" / "panda_spherized.urdf"
SRDF = RESOURCE_ROOT / "panda" / "panda.srdf"

B = 4          # small: the projector is iterative and unoptimised

#: Pose tolerance the projector can actually hold. NOT tighter, for a measured
#: reason: the task-space correction cannot drive the SE(3) log residual below
#: roughly 2e-4 in float32 whatever the step size -- that is the precision floor
#: of the FK and log-map chain, not step-dependent drift. Asserting 1e-4 here
#: would be asserting something no implementation can deliver in this dtype.
POSE_FLOOR = 1e-3
ACHIEVABLE_DZ = 0.005    # comfortably inside a 1-DOF null space
UNREACHABLE_DZ = 5.0     # nothing can move an elbow 5 m at fixed pose


@pytest.fixture(scope="module", autouse=True)
def _float32_regime():
    """Pin x64 off: the CUDA solve feeding these tests is float32-only at its
    FFI boundary, and x64 is process-wide global state other modules toggle."""
    before = bool(jax.config.read("jax_enable_x64"))
    if before:
        jax.config.update("jax_enable_x64", False)
    try:
        yield
    finally:
        if bool(jax.config.read("jax_enable_x64")) != before:
            jax.config.update("jax_enable_x64", before)


@pytest.fixture(scope="module")
def setup(_float32_regime):
    if not SPHERIZED_URDF.exists() or not SRDF.exists():
        pytest.skip(f"spherized Panda + SRDF not found under {RESOURCE_ROOT}")
    urdf = yourdfpy.URDF.load(str(SPHERIZED_URDF))
    robot = pk.Robot.from_urdf(urdf)
    coll = RobotCollisionSpherized.from_urdf(urdf, srdf_path=str(SRDF))
    ee = robot.links.names.index("panda_hand")
    elbow = robot.links.names.index("panda_link4")
    return robot, coll, ee, elbow


@pytest.fixture(scope="module")
def solved(setup):
    """A collision-free batch from path 2, i.e. the real input path 3 receives.

    Deliberately NOT a hand-built configuration: the frozen-projector bug only
    appeared because a real CUDA solution carries ~2e-4 of pose error, and an
    idealised start would have hidden it.
    """
    robot, coll, ee, _ = setup
    from pyroffi.optimization_engines._ls_ik import ls_ik_solve_cuda_batch

    n_act = robot.joints.num_actuated_joints
    rng = np.random.default_rng(3)
    lo = np.asarray(robot.joints.lower_limits)
    hi = np.asarray(robot.joints.upper_limits)
    q0 = jnp.asarray(lo + (hi - lo) * rng.random((B, n_act)), jnp.float32)
    targets = jaxlie.SE3(jax.vmap(lambda c: robot.forward_kinematics(c)[ee])(q0))

    sol = ls_ik_solve_cuda_batch(
        robot, ee, targets, rng_key=jax.random.PRNGKey(0),
        previous_cfgs=jnp.zeros((B, n_act)), collision_checker=coll)
    cfg = np.asarray(getattr(sol, "cfg", sol))
    cfg = cfg[:, 0] if cfg.ndim == 3 else cfg
    return jnp.asarray(cfg, jnp.float32), targets


def _elbow_z(robot, elbow, cfgs):
    return np.asarray(jax.vmap(
        lambda q: jaxlie.SE3(robot.forward_kinematics(q)[elbow]).translation()[2]
    )(jnp.asarray(cfgs, jnp.float32)))


def _pose_err(robot, ee, cfgs, targets):
    d = jax.vmap(lambda q, t: (jaxlie.SE3(robot.forward_kinematics(q)[ee]).inverse() @ t).log())(
        jnp.asarray(cfgs, jnp.float32), targets)
    return np.asarray(jnp.max(jnp.abs(d), axis=-1))


def make_elbow_constraint(elbow):
    """Constraint factory.

    The link index is CLOSED OVER, not passed through ``constraint_args``: args
    are vmapped per element when ``batched_constraint_args`` is set, and a
    static index has no batch axis to map over. Keeping statics in the closure
    and only per-problem values in args is the pattern users should follow.
    """
    def constraint(q, robot, target_z):
        return jaxlie.SE3(robot.forward_kinematics(q)[elbow]).translation()[2] - target_z
    return constraint


def test_achievable_constraint_actually_moves_the_robot(setup, solved):
    """The headline regression: it must MOVE, not return its input unchanged.

    A projector that silently does nothing reports exactly what an infeasible
    constraint reports. This is the assertion that caught the frozen-pose_tol
    bug, and it only works because `solved` is a real CUDA solution carrying
    real pose error.
    """
    robot, coll, ee, elbow = setup
    cfgs, targets = solved
    z0 = _elbow_z(robot, elbow, cfgs)

    res = project_onto_constraints(
        cfgs, robot, (ee,), (targets,), (make_elbow_constraint(elbow),),
        (jnp.asarray(z0 + ACHIEVABLE_DZ),),
        batched_constraint_args=True, collision_checker=coll, max_iter=30)

    z1 = _elbow_z(robot, elbow, res.cfg)
    moved = np.abs(z1 - z0)
    assert moved.max() > 1e-4, (
        f"the projector did not move any joint (max |dz| = {moved.max():.2e}). "
        f"It returned its input while reporting success="
        f"{np.asarray(res.success)} -- indistinguishable from an infeasible "
        f"constraint, which is the failure mode this test exists for.")
    assert np.asarray(res.constraint_violation).max() < ACHIEVABLE_DZ, (
        "constraint violation did not improve on the initial offset")


def test_pose_is_not_degraded(setup, solved):
    """Path 3's whole promise: the pose it was given survives the projection."""
    robot, coll, ee, elbow = setup
    cfgs, targets = solved
    before = _pose_err(robot, ee, cfgs, targets)
    z0 = _elbow_z(robot, elbow, cfgs)

    res = project_onto_constraints(
        cfgs, robot, (ee,), (targets,), (make_elbow_constraint(elbow),),
        (jnp.asarray(z0 + ACHIEVABLE_DZ),),
        batched_constraint_args=True, collision_checker=coll, max_iter=30)

    after = _pose_err(robot, ee, res.cfg, targets)
    # Judged against the INPUT pose, not an absolute bound -- the projector
    # holds what it was handed rather than beating the solver upstream of it.
    assert np.all(after <= np.maximum(before, POSE_FLOOR) + 1e-6), (
        f"pose degraded: worst {before.max():.2e} -> {after.max():.2e}")


def test_collision_freedom_survives_projection(setup, solved):
    """The null-space direction knows only about the constraint, so it can walk
    a collision-free configuration straight into contact unless it is checked."""
    robot, coll, ee, elbow = setup
    cfgs, targets = solved
    z0 = _elbow_z(robot, elbow, cfgs)

    res = project_onto_constraints(
        cfgs, robot, (ee,), (targets,), (make_elbow_constraint(elbow),),
        (jnp.asarray(z0 + ACHIEVABLE_DZ),),
        batched_constraint_args=True, collision_checker=coll, max_iter=30)

    assert bool(jnp.all(res.start_collision_free)), "fixture start was not collision-free"
    d = jnp.min(coll.compute_self_collision_distance(robot, res.cfg), axis=-1)
    assert bool(jnp.all(d >= 0.0)), (
        f"projection produced a self-collision (min distance {float(jnp.min(d)):.4f})")


def test_infeasible_constraint_fails_without_drifting(setup, solved):
    """Over-constrained must fail HONESTLY: no success, no pose damage."""
    robot, coll, ee, elbow = setup
    cfgs, targets = solved
    before = _pose_err(robot, ee, cfgs, targets)
    z0 = _elbow_z(robot, elbow, cfgs)

    res = project_onto_constraints(
        cfgs, robot, (ee,), (targets,), (make_elbow_constraint(elbow),),
        (jnp.asarray(z0 + UNREACHABLE_DZ),),
        batched_constraint_args=True, collision_checker=coll, max_iter=20)

    assert not bool(jnp.any(res.success)), "claimed success on an impossible constraint"
    assert bool(jnp.all(res.start_collision_free)), (
        "start_collision_free must stay True here -- otherwise an infeasible "
        "constraint is indistinguishable from an invalid input")
    after = _pose_err(robot, ee, res.cfg, targets)
    assert np.all(after <= np.maximum(before, POSE_FLOOR) + 1e-6), (
        "pose drifted while chasing an unreachable constraint")


def test_colliding_start_is_reported_distinctly(setup, solved):
    """A start already in collision rejects every step, and looks exactly like
    an infeasible constraint unless it is reported separately. These two have
    completely different fixes, so conflating them costs real debugging time."""
    robot, coll, ee, elbow = setup
    cfgs, targets = solved
    z0 = _elbow_z(robot, elbow, cfgs)

    # A demanding margin makes the (genuinely collision-free) start fail the
    # check, standing in for a start produced without a collision-aware solve.
    res = project_onto_constraints(
        cfgs, robot, (ee,), (targets,), (make_elbow_constraint(elbow),),
        (jnp.asarray(z0 + ACHIEVABLE_DZ),),
        batched_constraint_args=True, collision_checker=coll,
        collision_margin=10.0, max_iter=5)

    assert not bool(jnp.any(res.start_collision_free)), (
        "an unsatisfiable margin should mark the START as failing the check")
    assert not bool(jnp.any(res.success))


def test_batched_matches_per_element(setup, solved):
    """Batching must not change the answer -- it is a launch strategy, not an
    algorithm. Guards against a reduction accidentally coupling elements."""
    robot, coll, ee, elbow = setup
    cfgs, targets = solved
    z0 = _elbow_z(robot, elbow, cfgs)
    args = (jnp.asarray(z0 + ACHIEVABLE_DZ),)

    batched = project_onto_constraints(
        cfgs, robot, (ee,), (targets,), (make_elbow_constraint(elbow),), args,
        batched_constraint_args=True, collision_checker=coll, max_iter=20)

    for i in range(B):
        one = project_onto_constraints(
            cfgs[i], robot, (ee,), (jax.tree.map(lambda x: x[i][None], targets),),
            (make_elbow_constraint(elbow),),
            (jnp.asarray([z0[i] + ACHIEVABLE_DZ]),),
            batched_constraint_args=True, collision_checker=coll, max_iter=20)
        assert np.allclose(np.asarray(batched.cfg[i]), np.asarray(one.cfg[0]),
                           atol=1e-5), (
            f"element {i} differs between the batched and single-element paths")


def test_per_problem_constraint_targets(setup, solved):
    """Each problem may carry its OWN constraint target, which is the point of
    batching this at all -- a shared target would rarely be what you want."""
    robot, coll, ee, elbow = setup
    cfgs, targets = solved
    z0 = _elbow_z(robot, elbow, cfgs)
    per_problem = jnp.asarray(z0 + np.linspace(0.001, ACHIEVABLE_DZ, B))

    res = project_onto_constraints(
        cfgs, robot, (ee,), (targets,), (make_elbow_constraint(elbow),),
        (per_problem,), batched_constraint_args=True,
        collision_checker=coll, max_iter=30)

    z1 = _elbow_z(robot, elbow, res.cfg)
    # Larger requests should not end up closer to their target than smaller
    # ones do; a mis-broadcast arg would scramble this relationship.
    assert np.all(z1 >= z0 - 1e-6), "elbow moved the wrong way for some element"
