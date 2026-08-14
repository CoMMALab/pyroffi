"""The CUDA task-Jacobian kernel and the implicit rule that consumes it.

Two properties carry this feature, and both are easy to break silently:

1. the kernel's J matches ``jax.jacobian`` of the KERNEL'S residual -- not of
   the SE(3) log-map residual, which is a different function;
2. it matches only AT A CONVERGED SOLUTION, because the kernel returns the
   geometric Jacobian and its orientation rows coincide with
   ``d(log(R_ee R_tgt^-1))/dq`` only where the orientation error vanishes.

A regression in either would not raise -- it would return a wrong gradient.
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
from pyroffi.optimization_engines._ik_primitives import (
    _ik_residual_kernel_convention,
)

RESOURCES = pathlib.Path(__file__).resolve().parents[1] / "resources"

_ik_jacobian = pytest.importorskip("pyroffi.cuda_kernels.ik._ik_jacobian")


@pytest.fixture(scope="module")
def panda():
    urdf = yourdfpy.URDF.load(str(RESOURCES / "panda" / "panda_spherized.urdf"))
    robot = pk.Robot.from_urdf(urdf)
    return robot, robot.links.names.index("panda_hand")


def _buffers(robot):
    j = robot.joints
    return (j.twists, j.parent_transforms, j.parent_indices, j.actuated_indices,
            j.mimic_multiplier, j.mimic_offset, j.mimic_act_indices,
            j._topo_sort_inv)


def _random_cfgs(robot, n, seed=0):
    rng = np.random.default_rng(seed)
    lo = np.asarray(robot.joints.lower_limits)
    hi = np.asarray(robot.joints.upper_limits)
    return jnp.asarray(lo + (hi - lo) * rng.random((n, robot.joints.num_actuated_joints)),
                       jnp.float32)


@pytest.mark.skipif(not _ik_jacobian.library_available(),
                    reason="_ik_jacobian_lib.so not built")
def test_kernel_residual_matches_the_jax_convention(panda):
    robot, ee = panda
    q = _random_cfgs(robot, 32)
    targets = jax.vmap(lambda c: robot.forward_kinematics(c)[ee])(
        _random_cfgs(robot, 32, seed=1))
    tj, am = _ik_jacobian.ancestor_tables(robot, (ee,))

    r_cuda, _ = _ik_jacobian.task_jacobian(
        q, _buffers(robot), tj, am, targets[:, None, :])
    r_jax = jax.vmap(
        lambda c, t: _ik_residual_kernel_convention(c, robot, ee, jaxlie.SE3(t))
    )(q, targets)

    assert float(jnp.max(jnp.abs(r_cuda - r_jax))) < 1e-2


@pytest.mark.skipif(not _ik_jacobian.library_available(),
                    reason="_ik_jacobian_lib.so not built")
def test_position_rows_are_exact_everywhere(panda):
    """The position block is d(p_ee)/dq exactly, converged or not."""
    robot, ee = panda
    q = _random_cfgs(robot, 32)
    targets = jax.vmap(lambda c: robot.forward_kinematics(c)[ee])(
        _random_cfgs(robot, 32, seed=1))
    tj, am = _ik_jacobian.ancestor_tables(robot, (ee,))

    _, J_cuda = _ik_jacobian.task_jacobian(
        q, _buffers(robot), tj, am, targets[:, None, :])
    J_jax = jax.vmap(
        lambda c, t: jax.jacobian(_ik_residual_kernel_convention)(
            c, robot, ee, jaxlie.SE3(t))
    )(q, targets)

    assert float(jnp.max(jnp.abs(J_cuda[:, :3] - J_jax[:, :3]))) < 1e-2


@pytest.mark.skipif(not _ik_jacobian.library_available(),
                    reason="_ik_jacobian_lib.so not built")
def test_full_jacobian_matches_only_at_a_converged_solution(panda):
    """Guards the assumption the implicit rule depends on.

    At a solved configuration the whole J matches. At random configurations the
    orientation rows do NOT -- asserted explicitly, so that if someone ever
    reuses this kernel away from a solution the test says why it is wrong
    instead of quietly passing.
    """
    from pyroffi.optimization_engines._ls_ik import ls_ik_solve_cuda_batch

    robot, ee = panda
    n_act = robot.joints.num_actuated_joints
    targets = jax.vmap(lambda c: robot.forward_kinematics(c)[ee])(
        _random_cfgs(robot, 32, seed=1))
    tj, am = _ik_jacobian.ancestor_tables(robot, (ee,))

    def jac_pair(q):
        _, J_cuda = _ik_jacobian.task_jacobian(
            q, _buffers(robot), tj, am, targets[:, None, :])
        J_jax = jax.vmap(
            lambda c, t: jax.jacobian(_ik_residual_kernel_convention)(
                c, robot, ee, jaxlie.SE3(t))
        )(q, targets)
        return J_cuda, J_jax

    solved = ls_ik_solve_cuda_batch(
        robot, ee, jaxlie.SE3(targets), rng_key=jax.random.PRNGKey(0),
        previous_cfgs=jnp.zeros((32, n_act), jnp.float32))
    solved = jnp.asarray(getattr(solved, "cfg", solved), jnp.float32)

    J_cuda, J_jax = jac_pair(solved)
    assert float(jnp.max(jnp.abs(J_cuda - J_jax))) < 1e-2, (
        "kernel J must match at a converged solution -- the implicit rule "
        "differentiates r = 0 and relies on exactly this")

    J_cuda_r, J_jax_r = jac_pair(_random_cfgs(robot, 32))
    assert float(jnp.max(jnp.abs(J_cuda_r[:, 3:] - J_jax_r[:, 3:]))) > 1e-1, (
        "orientation rows are the GEOMETRIC Jacobian and must differ away from "
        "a solution; if this now matches, the kernel changed convention and the "
        "implicit rule's comments are stale")


@pytest.mark.skipif(not _ik_jacobian.library_available(),
                    reason="_ik_jacobian_lib.so not built")
def test_gradient_agrees_with_the_pure_jax_rule(panda):
    """End to end: swapping J_q's source must not change the gradient."""
    import pyroffi.optimization_engines._implicit_diff as ID
    from pyroffi.optimization_engines._ls_ik import ls_ik_solve_cuda_batch

    robot, ee = panda
    n_act = robot.joints.num_actuated_joints
    targets = jax.vmap(lambda c: robot.forward_kinematics(c)[ee])(
        _random_cfgs(robot, 64, seed=2))
    prev = jnp.zeros((64, n_act), jnp.float32)

    def loss(w):
        out = ls_ik_solve_cuda_batch(
            robot, ee, jaxlie.SE3(w), rng_key=jax.random.PRNGKey(0),
            previous_cfgs=prev)
        return jnp.sum(jnp.asarray(getattr(out, "cfg", out)) ** 2)

    original = ID._batch_task_jacobians
    try:
        # jax.jit caches on the traced function, so the two variants must be
        # separated by clear_caches() -- without it both runs execute the SAME
        # compiled graph and the comparison is vacuously equal.
        jax.clear_caches()
        g_cuda = jax.grad(loss)(targets)

        ID._batch_task_jacobians = lambda *a, **k: None
        jax.clear_caches()
        g_jax = jax.grad(loss)(targets)
    finally:
        ID._batch_task_jacobians = original
        jax.clear_caches()

    # Compared PER PROBLEM and by median, not by batch norm. Near a kinematic
    # singularity dq*/dt is genuinely ill-conditioned -- pinv amplifies a
    # float32 difference in J without bound -- so a single near-singular
    # problem dominates a batch-norm comparison and says nothing about the
    # other 63. MEASURED at B=64 on a panda: J agrees to 5.6e-4 median, pinv to
    # 9.4e-4 median, but the one problem with cond(J) = 4.5e3 differs by 19%,
    # which alone drives the batch-norm relative error to 0.43.
    num = jnp.linalg.norm(g_cuda - g_jax, axis=-1)
    den = jnp.linalg.norm(g_jax, axis=-1) + 1e-30
    rel = np.asarray(num / den)
    assert float(np.median(rel)) < 5e-2, (
        f"gradient changed with J_q's source (median rel={np.median(rel):.2e})")
    # The tail is allowed to be large, but only for a small minority; if most
    # problems disagree, the conventions have drifted apart rather than the
    # conditioning being bad.
    assert float(np.mean(rel > 5e-2)) < 0.1, (
        f"{100*np.mean(rel > 5e-2):.0f}% of problems disagree -- that is a "
        "convention mismatch, not singularity amplification")


# ---------------------------------------------------------------------------
# Documented limits of the implicit rule.
#
# These pin behaviour that is WRONG or MISSING today. They are here so the
# limits are discoverable from the test suite instead of being folklore, and so
# that fixing either one fails loudly and forces the docs to be updated.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _ik_jacobian.library_available(),
                    reason="_ik_jacobian_lib.so not built")
def test_hessian_is_not_trustworthy(panda):
    """`jax.hessian` through IK returns a plausible WRONG number.

    The rule differentiates r(q*, t) = 0 at a FIXED q*, so J_q carries no
    q*-dependence and the dJ_q/dq* term a second derivative needs is absent.
    Both J_q sources are affected; they merely disagree about HOW wrong.

    MEASURED at a verified-smooth panda target, against a finite difference of
    the (correct) gradient, true |H@v| = 138:  CUDA J_q gives 19.6 (7x small),
    jax.jacobian gives 68.3 (2x small).

    If this test starts FAILING because the two paths now agree, someone has
    implemented the second-order rule -- delete this test and the first-order
    caveats in _implicit_diff and the IK docs.
    """
    import pyroffi.optimization_engines._implicit_diff as ID
    from pyroffi.optimization_engines._ls_ik import ls_ik_solve_cuda_batch

    robot, ee = panda
    n_act = robot.joints.num_actuated_joints
    targets = jax.vmap(lambda c: robot.forward_kinematics(c)[ee])(
        _random_cfgs(robot, 4, seed=0))
    prev = jnp.zeros((4, n_act), jnp.float32)

    def loss(w):
        out = ls_ik_solve_cuda_batch(
            robot, ee, jaxlie.SE3(w), rng_key=jax.random.PRNGKey(0),
            previous_cfgs=prev)
        return jnp.sum(jnp.asarray(getattr(out, "cfg", out)) ** 2)

    original = ID._batch_task_jacobians
    try:
        jax.clear_caches()
        H_cuda = jax.hessian(loss)(targets)
        ID._batch_task_jacobians = lambda *a, **k: None
        jax.clear_caches()
        H_jax = jax.hessian(loss)(targets)
    finally:
        ID._batch_task_jacobians = original
        jax.clear_caches()

    rel = float(jnp.linalg.norm(H_cuda - H_jax)
                / (jnp.linalg.norm(H_jax) + 1e-30))
    assert rel > 0.5, (
        "the two J_q paths now agree on the Hessian; if the second-order rule "
        "was implemented, drop this test and the first-order caveats")


@pytest.mark.skipif(not _ik_jacobian.library_available(),
                    reason="_ik_jacobian_lib.so not built")
def test_vmap_over_grad_is_unsupported(panda):
    """`jax.vmap(jax.grad(...))` does not work through a CUDA IK solve.

    `dispatch_vmap_to_batched` gives vmap-over-SOLVE a rule, but vmap over the
    DIFFERENTIATED solve reaches the raw ffi_call, which has no vmap_method.
    Pre-existing, and load-bearing for differentiable TAMP (a batch of problems
    each carrying a gradient), so it is pinned rather than left to be
    rediscovered. `jax.grad` of a batched solve DOES work -- that is the
    supported shape.
    """
    from pyroffi.optimization_engines._ls_ik import ls_ik_solve_cuda_batch

    robot, ee = panda
    n_act = robot.joints.num_actuated_joints
    targets = jax.vmap(lambda c: robot.forward_kinematics(c)[ee])(
        _random_cfgs(robot, 4, seed=0))
    prev = jnp.zeros((4, n_act), jnp.float32)

    def loss(w):
        out = ls_ik_solve_cuda_batch(
            robot, ee, jaxlie.SE3(w), rng_key=jax.random.PRNGKey(0),
            previous_cfgs=prev)
        return jnp.sum(jnp.asarray(getattr(out, "cfg", out)) ** 2)

    jax.grad(loss)(targets)          # the supported shape still works

    with pytest.raises(NotImplementedError, match="vmap"):
        jax.vmap(lambda s: jax.grad(loss)(targets * s))(jnp.ones((3,)))
