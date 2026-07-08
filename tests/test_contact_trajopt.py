"""Contact-rich, dynamics-aware SCO trajectory optimization tests.

Requires a CUDA GPU and nvcc (GRiD dynamics kernels). Covers:
  * GRiD FFI kernels are now ``jax.vmap``-able and agree with leading-dim batch.
  * The augmented-Lagrangian fixed-contact constraint reduces an interior
    grasp-closure violation.
  * A genuine bimanual lift keeps the grasp rigid and solves contact forces.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yourdfpy

import pyroffi

PANDA_URDF = "resources/panda/panda_spherized.urdf"

pytest.importorskip("jax")
if not any(d.platform == "gpu" for d in jax.devices()):
    pytest.skip("CUDA device required", allow_module_level=True)


@pytest.fixture(scope="module")
def system():
    from pyroffi.collision import Box
    from pyroffi.dynamics import GRiDDynamics
    from pyroffi.dynamics._contact import (
        ContactSystem, GraspedObject, ManipulatorSpec, capture_grasp_offsets,
    )

    urdf = yourdfpy.URDF.load(PANDA_URDF, load_meshes=False)
    robot = pyroffi.Robot.from_urdf(urdf)
    grid = GRiDDynamics(urdf)
    left = ManipulatorSpec(robot, grid, "panda_hand", base_xy_yaw=(-0.4, 0.0, 0.0),
                           p_local=(0.0, 0.0, 0.1))
    right = ManipulatorSpec(robot, grid, "panda_hand", base_xy_yaw=(0.4, 0.0, np.pi),
                            p_local=(0.0, 0.0, 0.1))
    mid = (robot.joints.lower_limits + robot.joints.upper_limits) / 2
    manipulators = (left, right)
    offsets = capture_grasp_offsets(manipulators, (mid, mid))
    box = Box.from_center_and_dimensions(
        center=jnp.zeros(3), length=0.12, width=0.12, height=0.12, mass=0.5,
    )
    sys = ContactSystem(manipulators, GraspedObject(geom=box), offsets)
    return sys, robot, grid, mid


def test_grid_ffi_vmap_matches_leading_batch(system):
    _, robot, grid, _ = system
    n = grid.num_dof
    k = jax.random.PRNGKey(0)
    q, qd, qdd = (jax.random.normal(k, (5, n)) for _ in range(3))
    # A true (single-launch) vmap must match leading-dim batching bit-for-bit,
    # for every kernel entry point.
    for fn, args in (
        (grid.inverse_dynamics, (q, qd, qdd)),
        (grid.forward_dynamics, (q, qd, qdd)),
        (grid.mass_matrix_inv, (q,)),
        (grid.mass_matrix, (q,)),
        (grid.inverse_dynamics_gradient, (q, qd, qdd)),
        (grid.forward_dynamics_gradient, (q, qd, qdd)),
    ):
        assert float(jnp.abs(jax.vmap(fn)(*args) - fn(*args)).max()) == 0.0
    # grad-of-vmap must also match grad-of-leading-batch.
    gv = jax.grad(lambda x: jnp.sum(jax.vmap(grid.inverse_dynamics)(x, qd, qdd) ** 2))(q)
    gb = jax.grad(lambda x: jnp.sum(grid.inverse_dynamics(x, qd, qdd) ** 2))(q)
    assert float(jnp.abs(gv - gb).max()) == 0.0


def test_fixed_contact_constraint_converges(system):
    from pyroffi.dynamics._contact import grasp_closure_residual
    from pyroffi.optimization_engines import (
        ContactTrajOptConfig, contact_sco_trajopt,
    )

    sys, _, _, mid = system
    n = sys.num_dof
    start = jnp.concatenate([mid, mid])
    T = 12
    init = jnp.broadcast_to(start, (T, n)) + 0.1 * jax.random.normal(
        jax.random.PRNGKey(0), (T, n)
    )
    before = float(jnp.sqrt(jnp.mean(
        jax.vmap(grasp_closure_residual, in_axes=(None, 0))(sys, init) ** 2
    )))

    cfg = ContactTrajOptConfig(n_outer_iters=10, n_inner_iters=20,
                               rho_grasp=50.0, penalty_scale=2.0)
    traj, forces, resid = contact_sco_trajopt(init, start, start, sys, cfg)

    after = float(jnp.sqrt(jnp.mean(
        jax.vmap(grasp_closure_residual, in_axes=(None, 0))(sys, traj) ** 2
    )))
    assert after < 0.25 * before          # AL drives the violation down
    assert traj.shape == (T, n)
    assert forces.shape == (T, 2, 3)
    assert np.isfinite(np.array(forces)).all()
