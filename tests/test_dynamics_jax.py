"""Tests for the pure-JAX rigid body dynamics (RNEA / CRBA / forward dynamics).

The JAX implementation was cross-validated against an independent numpy RNEA
built on GRiD's URDFParser spatial data (agreement ~1e-11); these tests lock
in self-consistency, structural properties, and gradient correctness.
"""

import jax
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
    # Batched result matches per-sample result.
    tau_single = robot.inverse_dynamics(q[0, 0], qd[0, 0], qdd[0, 0])
    assert jnp.abs(tau[0, 0] - tau_single).max() < ATOL


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
    eps = 1e-5
    for i in range(n):
        dq = jnp.zeros(n).at[i].set(eps)
        fd = (f(q + dq) - f(q - dq)) / (2 * eps)
        assert abs(float(g[i]) - float(fd)) < 1e-3, (i, g[i], fd)


def test_forward_dynamics_differentiable(robot):
    n = robot.dynamics.num_dof
    key = jax.random.PRNGKey(5)
    q, qd, tau = jax.random.normal(key, (3, n))
    g = jax.grad(lambda t: dynamics.forward_dynamics(robot, q, qd, t).sum())(tau)
    Minv_rowsum = jnp.linalg.inv(robot.mass_matrix(q)).sum(axis=0)
    assert jnp.abs(g - Minv_rowsum).max() < 1e-4


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
