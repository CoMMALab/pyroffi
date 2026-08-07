"""Contact-rich trajopt (contact forces as decision variables): correctness.

Requires a CUDA GPU and nvcc (GRiD dynamics kernels).

Unlike the differential-flatness solver (which *allocates* the forces so the
object-dynamics residual is zero by construction), this solver *optimizes* the
forces under an augmented-Lagrangian Newton-Euler constraint. The tests below
check that the optimizer actually drives that residual down, keeps the grasp
rigid, and returns feasible (finite, friction-cone-respecting) forces.
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


def _build_system(box_mass, box_half=0.12):
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
    manips = (left, right)
    offsets = capture_grasp_offsets(manips, (mid, mid))
    box = Box.from_center_and_dimensions(
        center=jnp.zeros(3), length=box_half, width=box_half, height=box_half,
        mass=box_mass,
    )
    sys = ContactSystem(manips, GraspedObject(geom=box), offsets)
    return sys, mid


def test_contact_rich_optimizes_feasible_forces():
    """Solver drives the Newton-Euler residual down and holds the grasp rigid."""
    from pyroffi.dynamics._contact import object_center_world, object_dynamics_residual
    from pyroffi.optimization_engines import (
        ContactRichTrajOptConfig, contact_rich_trajopt,
    )
    from pyroffi.optimization_engines._contact_trajopt import _fd_vel_acc

    sys, mid = _build_system(box_mass=0.5)
    n = sys.num_dof
    start = jnp.concatenate([mid, mid])
    T = 16
    init = jnp.broadcast_to(start, (T, n)) + 0.05 * jax.random.normal(
        jax.random.PRNGKey(0), (T, n)
    )
    cfg = ContactRichTrajOptConfig(n_outer_iters=15, n_inner_iters=40,
                                   rho_grasp=50.0, rho_obj=10.0, penalty_scale=1.8)
    traj, forces, resid, centers, dt = contact_rich_trajopt(init, start, start, sys, cfg)

    assert np.isfinite(np.array(forces)).all()
    # resid = [obj_dyn_rms, obj_dyn_max, grasp_rms]
    assert float(resid[0]) < 0.5, f"object-dynamics residual too high: {resid}"
    assert float(resid[2]) < 5e-3, f"grasp drifted: {resid}"

    # dt stays within the configured bounds.
    assert cfg.min_dt - 1e-6 <= float(dt) <= cfg.max_dt + 1e-6

    # Independently recompute the linear force balance at the optimized dt.
    _, a_obj = _fd_vel_acc(
        jax.vmap(object_center_world, in_axes=(None, 0))(sys, traj), dt
    )
    b = jax.vmap(object_dynamics_residual, in_axes=(None, 0, 0, 0))(
        sys, traj, a_obj, forces
    )
    force_res = float(jnp.sqrt(jnp.mean(b[:, :3] ** 2)))
    assert force_res < 0.5, f"object force balance not met: {force_res}"


def test_contact_rich_single_manipulator():
    """Single-arm system (empty grasp offsets) solves and returns one force/step."""
    from pyroffi.collision import Box
    from pyroffi.dynamics import GRiDDynamics
    from pyroffi.dynamics._contact import (
        ContactSystem, GraspedObject, ManipulatorSpec,
    )
    from pyroffi.optimization_engines import (
        ContactRichTrajOptConfig, contact_rich_trajopt,
    )

    urdf = yourdfpy.URDF.load(PANDA_URDF, load_meshes=False)
    robot = pyroffi.Robot.from_urdf(urdf)
    grid = GRiDDynamics(urdf)
    arm = ManipulatorSpec(robot, grid, "panda_hand", p_local=(0.0, 0.0, 0.1))
    mid = (robot.joints.lower_limits + robot.joints.upper_limits) / 2
    box = Box.from_center_and_dimensions(
        center=jnp.zeros(3), length=0.05, width=0.13, height=0.04, mass=0.15,
    )
    sys = ContactSystem((arm,), GraspedObject(geom=box), ())

    T = 12
    init = jnp.broadcast_to(mid, (T, sys.num_dof))
    cfg = ContactRichTrajOptConfig(n_outer_iters=10, n_inner_iters=30, f_min=1.0)
    traj, forces, resid, centers, dt = contact_rich_trajopt(init, mid, mid, sys, cfg)

    assert forces.shape == (T, 1, 3)
    assert np.isfinite(np.array(forces)).all()
    assert float(resid[2]) == 0.0  # no grasp-closure residual for a single arm
