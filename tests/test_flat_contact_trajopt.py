"""Differential-flatness contact-*aware* trajopt: correctness + a discriminating
benchmark against the plain augmented-Lagrangian solver.

(For the genuinely contact-*rich* solver — contact forces as decision variables —
see ``test_contact_rich_trajopt.py``.)

Requires a CUDA GPU and nvcc (GRiD dynamics kernels).

The benchmark (``test_flat_beats_naive_and_al``) is built so that a *naive*
plan visibly FAILS — linearly interpolating the two arms' joint configs does
NOT preserve the relative gripper transform mid-trajectory, so the rigid grasp
drifts. This is the failure case the old benchmark lacked: it makes "did the
contact constraint actually get enforced?" an unambiguous, measurable question.
"""

import time

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


def _grasp_rms_max(sys, traj):
    from pyroffi.dynamics._contact import grasp_closure_residual
    g = jax.vmap(grasp_closure_residual, in_axes=(None, 0))(sys, traj)
    return float(jnp.sqrt(jnp.mean(g**2))), float(jnp.max(jnp.abs(g)))


def test_flat_keeps_grasp_rigid_and_dynamics_exact():
    """Flat solver holds the grasp rigid and satisfies object force balance."""
    from pyroffi.dynamics._contact import object_center_world, object_dynamics_residual
    from pyroffi.optimization_engines import (
        FlatContactTrajOptConfig, flat_contact_trajopt,
    )
    from pyroffi.optimization_engines._flat_contact_trajopt import _fd_vel_acc

    sys, mid = _build_system(box_mass=0.5)
    n = sys.num_dof
    start = jnp.concatenate([mid, mid])
    T = 16
    init = jnp.broadcast_to(start, (T, n)) + 0.05 * jax.random.normal(
        jax.random.PRNGKey(0), (T, n)
    )
    cfg = FlatContactTrajOptConfig(n_stages=5, n_inner_iters=50, w_track=200.0,
                                   track_scale=4.0, w_track_max=1e6)
    traj, forces, resid, centers, dt = flat_contact_trajopt(init, start, start, sys, cfg)

    # Grasp stays rigid by construction (tracked, not penalized after the fact).
    assert float(resid[0]) < 5e-3, f"grasp rms too high: {resid}"
    assert np.isfinite(np.array(forces)).all()

    # Allocated forces satisfy the object's linear (Newton) balance ~exactly.
    # Use the *optimized* timestep the solver allocated the forces at.
    _, a_obj = _fd_vel_acc(
        jax.vmap(object_center_world, in_axes=(None, 0))(sys, traj), dt
    )
    b = jax.vmap(object_dynamics_residual, in_axes=(None, 0, 0, 0))(
        sys, traj, a_obj, forces
    )
    force_res = float(jnp.sqrt(jnp.mean(b[:, :3] ** 2)))
    assert force_res < 0.2, f"object force balance not met: {force_res}"


def test_flat_beats_naive_and_al(capsys):
    """Discriminating benchmark: naive joint-interp drifts the grasp; the flat
    solver keeps it rigid, allocates feasible forces, and is far faster than the
    augmented-Lagrangian solver."""
    from pyroffi.dynamics._contact import (
        object_center_world, object_dynamics_residual,
    )
    from pyroffi.optimization_engines import (
        ContactTrajOptConfig, FlatContactTrajOptConfig,
        contact_sco_trajopt, flat_contact_trajopt,
    )
    from pyroffi.optimization_engines._flat_contact_trajopt import _fd_vel_acc

    BOX_MASS = 3.0  # heavy -> dynamics genuinely matter
    sys, mid = _build_system(box_mass=BOX_MASS)
    n = sys.num_dof
    T = 20
    start = jnp.concatenate([mid, mid])

    # Goal: same arms, box lifted. Translating BOTH gripper targets by the same
    # world vector preserves their relative transform, so the *endpoints* are a
    # valid rigid grasp -- only the interpolation in between can break it.
    goal = start  # (hold-and-squeeze; the discriminator is mid-trajectory drift)

    # --- Naive baseline: linear joint interpolation with a mid detour -------
    # A small opposing per-arm perturbation in the middle mimics an independent
    # kinematic plan that does not respect the rigid coupling.
    tvec = jnp.linspace(0.0, 1.0, T)[:, None]
    bump = jnp.sin(jnp.pi * tvec) * 0.15
    naive = start[None] * (1 - tvec) + goal[None] * tvec
    naive = naive.at[:, 0].add(bump[:, 0]).at[:, n // 2].add(-bump[:, 0])
    naive = naive.at[0].set(start).at[-1].set(goal)

    def dyn_force_res(traj, forces):
        _, a_obj = _fd_vel_acc(
            jax.vmap(object_center_world, in_axes=(None, 0))(sys, traj), 0.1
        )
        b = jax.vmap(object_dynamics_residual, in_axes=(None, 0, 0, 0))(
            sys, traj, a_obj, forces
        )
        return float(jnp.sqrt(jnp.mean(b[:, :3] ** 2)))

    # Naive forces: a static even weight split (what a kinematic planner assumes).
    share = BOX_MASS * sys.gravity / sys.num_manipulators
    naive_forces = jnp.tile(jnp.array([0.0, 0.0, share]), (T, sys.num_manipulators, 1))

    rows = []
    ng_rms, ng_max = _grasp_rms_max(sys, naive)
    rows.append(("naive-interp", ng_rms, ng_max, dyn_force_res(naive, naive_forces),
                 float("nan"), float("nan")))

    # --- Old augmented-Lagrangian solver -----------------------------------
    al_cfg = ContactTrajOptConfig(n_outer_iters=15, n_inner_iters=25, dt=0.1,
                                  rho_grasp=50.0, penalty_scale=1.8, tau_max=87.0)
    # Warm up with the EXACT config we time -- opt_cfg is a static JIT arg, so a
    # mismatched warmup config would force a full recompile inside the timed
    # region (this is what made the original example's "execution" look ~70s).
    jax.block_until_ready(contact_sco_trajopt(naive, start, goal, sys, al_cfg))
    t0 = time.perf_counter()
    al_traj, al_forces, _ = contact_sco_trajopt(naive, start, goal, sys, al_cfg)
    jax.block_until_ready(al_traj)
    al_solve = time.perf_counter() - t0
    ag_rms, ag_max = _grasp_rms_max(sys, al_traj)
    rows.append(("aug-lagrangian", ag_rms, ag_max, dyn_force_res(al_traj, al_forces),
                 al_solve, float("nan")))

    # --- New differential-flatness solver ----------------------------------
    flat_cfg = FlatContactTrajOptConfig(n_stages=5, n_inner_iters=50, dt=0.1,
                                        w_track=200.0, track_scale=4.0,
                                        w_track_max=1e6, tau_max=87.0)
    jax.block_until_ready(flat_contact_trajopt(naive, start, goal, sys, flat_cfg))
    t0 = time.perf_counter()
    fl_traj, fl_forces, fl_resid, _, _ = flat_contact_trajopt(naive, start, goal, sys, flat_cfg)
    jax.block_until_ready(fl_traj)
    fl_solve = time.perf_counter() - t0
    fg_rms, fg_max = _grasp_rms_max(sys, fl_traj)
    rows.append(("flat (ours)", fg_rms, fg_max, dyn_force_res(fl_traj, fl_forces),
                 fl_solve, float("nan")))

    header = f"\n=== Contact-rich trajopt benchmark (box mass {BOX_MASS} kg, T={T}) ==="
    cols = (f"{'method':16s} {'grasp_rms':>10s} {'grasp_max':>10s} "
            f"{'force_res':>10s} {'solve_s':>9s}")
    lines = [header, cols]
    for name, gr, gm, fr, st, _ in rows:
        st_s = "  -" if np.isnan(st) else f"{st:8.2f}"
        lines.append(f"{name:16s} {gr:10.5f} {gm:10.5f} {fr:10.4f} {st_s:>9s}")
    table = "\n".join(lines)
    with capsys.disabled():
        print(table)
    with open("/tmp/flat_bench_table.txt", "w") as fh:
        fh.write(table + "\n")

    naive_force_res = dyn_force_res(naive, naive_forces)
    flat_force_res = dyn_force_res(fl_traj, fl_forces)
    # Value proposition of the flatness reformulation:
    #  1. it removes the naive plan's rigid-grasp drift (grasp closure),
    assert fg_rms < 0.2 * ng_rms, "flat did not fix the naive grasp drift"
    #  2. its allocated forces satisfy object dynamics ~exactly (naive's static
    #     force guess does not once the object moves),
    assert flat_force_res < 0.2, "flat force balance broken"
    assert flat_force_res < naive_force_res, "flat should beat naive on dynamics"
    #  3. and it is far cheaper than the augmented-Lagrangian solver.
    assert fl_solve < al_solve, "flat should be faster than AL"
