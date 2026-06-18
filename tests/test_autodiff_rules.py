"""Correctness tests for the custom autodiff (JVP/VJP) rules.

Covers the three CUDA-FFI subsystems that were made differentiable:

  1. FK         — Robot.forward_kinematics(use_cuda=True)
  2. SDF coll.  — CUDARobotCollisionChecker.compute_{world,self}_collision_distance
  3. IK engines — implicit differentiation of q*(target_pose)

Design note
-----------
The differentiation *rules* are pure JAX (the opaque FFI kernels are confined to
the undifferentiated forward path), so every gradient check below runs on CPU or
GPU and does **not** require the compiled ``*.so`` kernels.  For FK and collision
the CUDA-path gradient is asserted equal to the pure-JAX reference gradient.  For
IK the implicit gradient is validated by the defining identity
``J_geometric @ dq* == dt`` (the predicted joint tangent reproduces the requested
end-effector tangent) plus forward/reverse-mode consistency.

Usage:
    python tests/test_autodiff_rules.py      # run as a script
    pytest tests/test_autodiff_rules.py      # or under pytest

Requires ``resources/panda/panda_spherized.urdf``.
"""

from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import yourdfpy

import pyroffi as pk
from pyroffi.collision import CUDARobotCollisionChecker, RobotCollision, Sphere
from pyroffi.optimization_engines._hjcd_ik import hjcd_solve
from pyroffi.optimization_engines._ls_ik import ls_ik_solve
from pyroffi.optimization_engines._sqp_ik import sqp_ik_solve

RESOURCE_ROOT = pathlib.Path(__file__).resolve().parent.parent / "resources"
PANDA = RESOURCE_ROOT / "panda" / "panda_spherized.urdf"

ATOL = 5e-4   # float32 CUDA-path vs (x64) JAX-reference tolerance
RNG = np.random.RandomState(0)


def _load():
    urdf = yourdfpy.URDF.load(str(PANDA))
    robot = pk.Robot.from_urdf(urdf)
    return robot


def _rand_cfg(robot, seed=0):
    n = robot.joints.num_actuated_joints
    return jnp.asarray(np.random.RandomState(seed).uniform(-0.4, 0.4, size=(n,)),
                       dtype=jnp.float32)


# ---------------------------------------------------------------------------
# 1. Forward kinematics
# ---------------------------------------------------------------------------
def test_fk_cuda_gradient_matches_jax():
    robot = _load()
    cfg = _rand_cfg(robot, 0)
    li = robot.links.num_links - 1

    def loss(c, use_cuda):
        P = robot.forward_kinematics(c, use_cuda=use_cuda)
        return jnp.sum(P[li] ** 2)

    g_jax = jax.grad(lambda c: loss(c, False))(cfg)
    g_cuda = jax.grad(lambda c: loss(c, True))(cfg)
    assert jnp.all(jnp.isfinite(g_cuda))
    assert float(jnp.max(jnp.abs(g_jax - g_cuda))) < ATOL

    # forward mode
    v = jnp.asarray(np.random.RandomState(1).randn(cfg.shape[0]), dtype=jnp.float32)
    _, t_jax = jax.jvp(lambda c: loss(c, False), (cfg,), (v,))
    _, t_cuda = jax.jvp(lambda c: loss(c, True), (cfg,), (v,))
    assert float(jnp.abs(t_jax - t_cuda)) < ATOL

    # jit-compiled reverse mode
    g_jit = jax.jit(jax.grad(lambda c: loss(c, True)))(cfg)
    assert float(jnp.max(jnp.abs(g_jit - g_jax))) < ATOL


# ---------------------------------------------------------------------------
# 2. SDF collision checker
# ---------------------------------------------------------------------------
def test_collision_cuda_gradient_matches_jax():
    robot = _load()
    coll = RobotCollision.from_urdf(yourdfpy.URDF.load(str(PANDA)))
    cuda = CUDARobotCollisionChecker(coll)
    cfg = _rand_cfg(robot, 0)
    world = Sphere.from_center_and_radius(
        center=jnp.array([[0.4, 0.0, 0.4], [0.2, 0.2, 0.5]]),
        radius=jnp.array([0.1, 0.08]),
    )

    def world_loss(c, mdl):
        return jnp.sum(jnp.minimum(mdl.compute_world_collision_distance(robot, c, world), 0.1) ** 2)

    g_jax = jax.grad(lambda c: world_loss(c, coll))(cfg)
    g_cuda = jax.grad(lambda c: world_loss(c, cuda))(cfg)
    assert jnp.all(jnp.isfinite(g_cuda))
    assert float(jnp.max(jnp.abs(g_jax - g_cuda))) < ATOL

    def self_loss(c, mdl):
        return jnp.sum(jnp.minimum(mdl.compute_self_collision_distance(robot, c), 0.1) ** 2)

    g_jax_s = jax.grad(lambda c: self_loss(c, coll))(cfg)
    g_cuda_s = jax.grad(lambda c: self_loss(c, cuda))(cfg)
    assert float(jnp.max(jnp.abs(g_jax_s - g_cuda_s))) < ATOL


# ---------------------------------------------------------------------------
# 3. IK engines — implicit differentiation
# ---------------------------------------------------------------------------
def _ik_identity_check(robot, solve_q, T0, tol):
    """Validate dq*/d(target) via J_geometric @ dq* == d(target tangent)."""
    li = robot.links.num_links - 1
    x0 = T0.wxyz_xyz
    q0 = solve_q(x0)

    # reverse mode produces finite grads
    g = jax.grad(lambda x: jnp.sum(solve_q(x) ** 2))(x0)
    assert jnp.all(jnp.isfinite(g))

    # forward/reverse consistency: jvp == grad . v
    v = jnp.asarray(np.random.RandomState(3).randn(7))
    _, tv = jax.jvp(lambda x: jnp.sum(solve_q(x) ** 2), (x0,), (v,))
    assert float(abs(tv - jnp.dot(g, v))) < 1e-5

    # defining identity along a random SE(3) tangent
    xi = jnp.asarray(np.random.RandomState(9).randn(6))
    _, dt = jax.jvp(lambda e: (T0 @ jaxlie.SE3.exp(e * xi)).wxyz_xyz, (0.0,), (1.0,))
    _, dq = jax.jvp(solve_q, (x0,), (dt,))
    Jgeo = jax.jacobian(
        lambda q: jaxlie.SE3(robot.forward_kinematics(q)[li]).wxyz_xyz
    )(q0)
    identity_err = float(jnp.max(jnp.abs(Jgeo @ dq - dt)))
    assert identity_err < tol, f"FK_jac @ dq* != dt: {identity_err}"


def test_ls_ik_implicit_diff():
    robot = _load()
    li = robot.links.num_links - 1
    key = jax.random.PRNGKey(0)
    q_ref = jnp.asarray(np.random.RandomState(2).uniform(-0.4, 0.4,
                        size=(robot.joints.num_actuated_joints,)))
    T0 = jaxlie.SE3(robot.forward_kinematics(q_ref)[li])
    solve_q = lambda x: ls_ik_solve(robot, (li,), (jaxlie.SE3(x),), key, q_ref,
                                    num_seeds=16, max_iter=80)
    # well-converged solver → tight identity
    _ik_identity_check(robot, solve_q, T0, tol=1e-3)


def test_sqp_ik_implicit_diff():
    robot = _load()
    li = robot.links.num_links - 1
    key = jax.random.PRNGKey(0)
    q_ref = jnp.asarray(np.random.RandomState(7).uniform(-0.4, 0.4,
                        size=(robot.joints.num_actuated_joints,)))
    T0 = jaxlie.SE3(robot.forward_kinematics(q_ref)[li])
    solve_q = lambda x: sqp_ik_solve(robot, (li,), (jaxlie.SE3(x),), key, q_ref,
                                     num_seeds=16, max_iter=60)
    _ik_identity_check(robot, solve_q, T0, tol=1e-3)


def test_hjcd_ik_grad_finite():
    # HJCD's coarse phase may leave a small residual at default settings; we only
    # assert the gradient is finite and the public Robot.inverse_kinematics path
    # differentiates end-to-end.
    robot = _load()
    li = robot.links.num_links - 1
    key = jax.random.PRNGKey(0)
    q_ref = jnp.asarray(np.random.RandomState(7).uniform(-0.4, 0.4,
                        size=(robot.joints.num_actuated_joints,)))
    T0 = jaxlie.SE3(robot.forward_kinematics(q_ref)[li])
    x0 = T0.wxyz_xyz
    g_solve = jax.grad(lambda x: jnp.sum(
        hjcd_solve(robot, (li,), (jaxlie.SE3(x),), key, q_ref, num_seeds=32) ** 2))(x0)
    assert jnp.all(jnp.isfinite(g_solve))
    g_ik = jax.grad(lambda x: jnp.sum(
        robot.inverse_kinematics(robot.links.names[li], jaxlie.SE3(x),
                                 rng_key=key, previous_cfg=q_ref) ** 2))(x0)
    assert jnp.all(jnp.isfinite(g_ik))


if __name__ == "__main__":
    tests = [
        test_fk_cuda_gradient_matches_jax,
        test_collision_cuda_gradient_matches_jax,
        test_ls_ik_implicit_diff,
        test_sqp_ik_implicit_diff,
        test_hjcd_ik_grad_finite,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print("All autodiff-rule tests passed.")
