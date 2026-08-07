"""Tests for the pure-JAX rigid body dynamics (RNEA / CRBA / forward dynamics).

The JAX implementation was cross-validated against an independent numpy RNEA
built on GRiD's URDFParser spatial data (agreement ~1e-11); these tests lock
in self-consistency, structural properties, and gradient correctness.
"""

import jax
# These tests run in the process-default f32 (pyroffi no longer forces a global
# jax_enable_x64=True at import). Tolerances and finite-difference steps below
# are sized for f32, so they cost some accuracy vs the old f64 runs.
import jax.numpy as jnp
import numpy as onp
import pytest
import yourdfpy

import pyroffi
from pyroffi import dynamics

PANDA_URDF = "resources/panda/panda_spherized.urdf"
BAXTER_URDF = "resources/baxter/baxter_spherized_coarse.urdf"  # branching tree

ATOL = 1e-6


@pytest.fixture(scope="module", params=[PANDA_URDF, BAXTER_URDF])
def robot(request) -> pyroffi.Robot:
    urdf = yourdfpy.URDF.load(request.param, load_meshes=False)
    return pyroffi.Robot.from_urdf(urdf)


def _rand_state(robot, key, batch=()):
    n = robot.dynamics.num_dof
    return jax.random.normal(key, (3, *batch, n))


def test_dynamics_parsed(robot):
    dyn = robot.dynamics
    assert dyn is not None
    assert dyn.num_dof == len(robot.joints.actuated_names)
    assert dyn.dof_names == robot.joints.actuated_names
    assert dyn.I_body.shape == (dyn.num_dof, 6, 6)
    # Spatial inertias are symmetric with positive masses on the diagonal.
    assert onp.allclose(dyn.I_body, onp.swapaxes(dyn.I_body, -1, -2), atol=1e-8)


def test_mass_matrix_symmetric_positive_definite(robot):
    q = _rand_state(robot, jax.random.PRNGKey(0))[0]
    M = robot.mass_matrix(q)
    assert jnp.allclose(M, M.T, atol=ATOL)
    assert jnp.linalg.eigvalsh(M).min() > 0.0


def test_forward_inverse_consistency(robot):
    q, qd, tau = _rand_state(robot, jax.random.PRNGKey(1))
    qdd = robot.forward_dynamics(q, qd, tau)
    tau_rt = robot.inverse_dynamics(q, qd, qdd)
    assert jnp.abs(tau_rt - tau).max() < 1e-4


def test_inverse_dynamics_decomposition(robot):
    """ID(q,qd,qdd) == M(q) qdd + bias(q,qd)."""
    q, qd, qdd = _rand_state(robot, jax.random.PRNGKey(2))
    tau = robot.inverse_dynamics(q, qd, qdd)
    bias = robot.inverse_dynamics(q, qd, jnp.zeros_like(qdd))
    M = robot.mass_matrix(q)
    assert jnp.abs(tau - (M @ qdd + bias)).max() < 1e-4


def test_batched_shapes(robot):
    n = robot.dynamics.num_dof
    q, qd, qdd = _rand_state(robot, jax.random.PRNGKey(3), batch=(4, 5))
    tau = robot.inverse_dynamics(q, qd, qdd)
    assert tau.shape == (4, 5, n)
    M = robot.mass_matrix(q)
    assert M.shape == (4, 5, n, n)
    # Batched result matches per-sample result. In f32 the batched path reorders
    # reductions, so use a relative tolerance rather than the tight ATOL.
    tau_single = robot.inverse_dynamics(q[0, 0], qd[0, 0], qdd[0, 0])
    assert jnp.allclose(tau[0, 0], tau_single, rtol=1e-2, atol=1e-3)


def test_gravity_zero_at_rest_without_gravity(robot):
    n = robot.dynamics.num_dof
    z = jnp.zeros(n)
    tau = dynamics.inverse_dynamics(robot, z, z, z, gravity=0.0)
    assert jnp.abs(tau).max() < ATOL


def test_damping_contribution(robot):
    n = robot.dynamics.num_dof
    z = jnp.zeros(n)
    qd = jnp.ones(n)
    tau = dynamics.inverse_dynamics(robot, z, qd, z, gravity=0.0)
    tau_nodamp_dyn = robot.dynamics
    # Coriolis at q=0 plus damping; subtracting the damping term must equal
    # the same computation with damping zeroed.
    import jax_dataclasses as jdc

    with jdc.copy_and_mutate(robot, validate=False) as robot_nd:
        with jdc.copy_and_mutate(tau_nodamp_dyn, validate=False) as dyn_nd:
            dyn_nd.damping = jnp.zeros(n)
        robot_nd.dynamics = dyn_nd
    tau_nd = dynamics.inverse_dynamics(robot_nd, z, qd, z, gravity=0.0)
    assert jnp.abs((tau - tau_nd) - robot.dynamics.damping * qd).max() < ATOL


def test_inverse_dynamics_gradient_finite_difference(robot):
    n = robot.dynamics.num_dof
    key = jax.random.PRNGKey(4)
    q, qd, qdd = jax.random.normal(key, (3, n))

    def f(q_):
        return dynamics.inverse_dynamics(robot, q_, qd, qdd).sum()

    g = jax.grad(f)(q)
    # f32 central difference: eps=1e-5 suffers subtractive cancellation, so use
    # a larger step and an f32-sized tolerance (truncation + roundoff).
    eps = 1e-2
    for i in range(n):
        dq = jnp.zeros(n).at[i].set(eps)
        fd = (f(q + dq) - f(q - dq)) / (2 * eps)
        assert abs(float(g[i]) - float(fd)) < 5e-2 * (1 + abs(float(g[i]))), (i, g[i], fd)


def test_forward_dynamics_differentiable(robot):
    n = robot.dynamics.num_dof
    key = jax.random.PRNGKey(5)
    q, qd, tau = jax.random.normal(key, (3, n))
    g = jax.grad(lambda t: dynamics.forward_dynamics(robot, q, qd, t).sum())(tau)
    Minv_rowsum = jnp.linalg.inv(robot.mass_matrix(q)).sum(axis=0)
    assert jnp.abs(g - Minv_rowsum).max() < 1e-4


def test_jacobian_linear_vs_autodiff(robot):
    """J_lin must equal d r / d q (frame-origin velocity = J_lin @ qd)."""
    n = robot.dynamics.num_dof
    q = jax.random.normal(jax.random.PRNGKey(6), (n,))
    J, r = dynamics.jacobian(robot, q)
    dr_dq = jax.jacfwd(lambda q_: dynamics.jacobian(robot, q_)[1])(q)
    assert jnp.abs(J[:, 3:, :] - dr_dq).max() < 1e-3  # f32 tolerance


def test_jacobian_angular_vs_autodiff(robot):
    """skew(J_ang[:, j]) == d R_wb / d q_j @ R_wb^T for each body."""
    from pyroffi.dynamics._dynamics_jax import _compute_X0, _compute_Xup

    dyn = robot.dynamics
    n = dyn.num_dof
    q = jax.random.normal(jax.random.PRNGKey(7), (n,))
    J, _ = dynamics.jacobian(robot, q)

    def rotations(q_):
        X0 = _compute_X0(dyn, _compute_Xup(dyn, q_))
        # E maps world -> body, so R_wb = E^T.
        return jnp.stack([X0[i][:3, :3].T for i in range(n)])

    R = rotations(q)
    dR = jax.jacfwd(rotations)(q)  # (n_body, 3, 3, n_dof)
    for i in range(n):
        for j in range(n):
            W = dR[i, :, :, j] @ R[i].T  # skew(d omega / d qd_j)
            w = jnp.array([W[2, 1], W[0, 2], W[1, 0]])
            assert jnp.abs(J[i, :3, j] - w).max() < 1e-3, (i, j)  # f32 tolerance


def test_fext_generalized_force_equivalence(robot):
    """ID(q,qd,qdd,f_ext) == ID(q,qd,qdd) - J^T f_ext (world-axis wrenches)."""
    n = robot.dynamics.num_dof
    key = jax.random.PRNGKey(8)
    q, qd, qdd = jax.random.normal(key, (3, n))
    f_ext = jax.random.normal(jax.random.PRNGKey(9), (n, 6))
    tau = dynamics.inverse_dynamics(robot, q, qd, qdd)
    tau_f = dynamics.inverse_dynamics(robot, q, qd, qdd, f_ext=f_ext)
    J, _ = dynamics.jacobian(robot, q)
    tau_ext = jnp.einsum("ijk,ij->k", J, f_ext)
    assert jnp.abs((tau - tau_f) - tau_ext).max() < 1e-4


def test_fext_forward_inverse_roundtrip(robot):
    n = robot.dynamics.num_dof
    key = jax.random.PRNGKey(10)
    q, qd, tau = jax.random.normal(key, (3, n))
    f_ext = jax.random.normal(jax.random.PRNGKey(11), (n, 6))
    qdd = dynamics.forward_dynamics(robot, q, qd, tau, f_ext=f_ext)
    tau_rt = dynamics.inverse_dynamics(robot, q, qd, qdd, f_ext=f_ext)
    assert jnp.abs(tau_rt - tau).max() < 1e-4


def test_fext_batched(robot):
    n = robot.dynamics.num_dof
    q, qd, qdd = _rand_state(robot, jax.random.PRNGKey(12), batch=(4,))
    f_ext = jax.random.normal(jax.random.PRNGKey(13), (4, n, 6))
    tau = dynamics.inverse_dynamics(robot, q, qd, qdd, f_ext=f_ext)
    assert tau.shape == (4, n)
    tau0 = dynamics.inverse_dynamics(robot, q[0], qd[0], qdd[0], f_ext=f_ext[0])
    # f32: batched vs single reorders reductions; with external wrenches on the
    # larger arms the drift is a few % of the torque magnitude, so relax further.
    assert jnp.allclose(tau[0], tau0, rtol=5e-2, atol=1e-3)


def test_step_semi_implicit_definition(robot):
    n = robot.dynamics.num_dof
    key = jax.random.PRNGKey(14)
    q, qd, tau = jax.random.normal(key, (3, n))
    dt = 1e-3
    q1, qd1 = dynamics.step(robot, q, qd, tau, dt)
    qdd = dynamics.forward_dynamics(robot, q, qd, tau)
    assert jnp.abs(qd1 - (qd + dt * qdd)).max() < ATOL
    assert jnp.abs(q1 - (q + dt * qd1)).max() < ATOL


def test_step_rk4_rollout_matches_fine_euler(robot):
    """A coarse RK4 rollout should track a much finer semi-implicit rollout."""
    n = robot.dynamics.num_dof
    q0 = 0.1 * jax.random.normal(jax.random.PRNGKey(15), (n,))
    qd0 = jnp.zeros(n)
    tau = jnp.zeros(n)
    dt, steps, refine = 5e-3, 20, 50

    def rollout(method, dt, steps):
        def body(carry, _):
            q, qd = carry
            return dynamics.step(robot, q, qd, tau, dt, method=method), None

        (q, qd), _ = jax.lax.scan(body, (q0, qd0), None, length=steps)
        return q, qd

    q_rk4, _ = rollout("rk4", dt, steps)
    q_ref, _ = rollout("semi_implicit", dt / refine, steps * refine)
    assert jnp.abs(q_rk4 - q_ref).max() < 1e-3


def test_step_jit_and_batch(robot):
    n = robot.dynamics.num_dof
    q, qd, tau = _rand_state(robot, jax.random.PRNGKey(16), batch=(4,))
    step = jax.jit(lambda q, qd: dynamics.step(robot, q, qd, tau, 1e-3))
    q1, qd1 = step(q, qd)
    assert q1.shape == qd1.shape == (4, n)


def test_step_substeps_default_matches_single_step(robot):
    """substeps=1 (the default) must reproduce the pre-existing behavior exactly."""
    n = robot.dynamics.num_dof
    q, qd, tau = _rand_state(robot, jax.random.PRNGKey(17), batch=())
    dt = 1e-3
    q1, qd1 = dynamics.step(robot, q, qd, tau, dt)
    q1_sub, qd1_sub = dynamics.step(robot, q, qd, tau, dt, substeps=1)
    assert jnp.array_equal(q1, q1_sub)
    assert jnp.array_equal(qd1, qd1_sub)


def test_step_substeps_equals_manual_subdivision(robot):
    """step(dt, substeps=k) must equal k applications of step(dt/k, substeps=1).

    This is what the `substeps` parameter is defined to do -- subdivide dt into
    k equal fixed-step updates. (The parameter exists so callers can subdivide a
    large dt for stability; note that subdivision alone does not rescue every
    stiff/ill-conditioned scenario -- forward-dynamics rollouts can still diverge
    for reasons unrelated to step size, e.g. a near-singular mass matrix.)
    """
    n = robot.dynamics.num_dof
    q = jnp.linspace(-0.2, 0.2, n)
    qd = jnp.zeros(n)
    tau = jnp.full(n, 0.5)
    dt, k = 2e-3, 4

    q_sub, qd_sub = dynamics.step(robot, q, qd, tau, dt, substeps=k)

    qc, qdc = q, qd
    for _ in range(k):
        qc, qdc = dynamics.step(robot, qc, qdc, tau, dt / k, substeps=1)

    assert jnp.abs(q_sub - qc).max() < ATOL
    assert jnp.abs(qd_sub - qdc).max() < ATOL


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
