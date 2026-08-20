"""Interactive IOC on the Panda: teleoperate demos, then recover the cost weights.

Drag the transform-controls gizmo to pose the end-effector; "Add waypoint" solves
IK for the current gizmo pose and appends it to the demo under construction;
"Finish demo" closes it out and resamples it into a fixed-length trajectory (the
representation `ioc.robot.problem` expects: endpoints clamped, interior
waypoints free).  After recording a few demos, "Run IOC" fits five methods
(kkt, cioc, fd, cmaes, implicit -- reusing `ioc.robot.e1_identifiability`'s
machinery) against the recorded trajectories.

There is no ground-truth theta* here -- these are teleoperated, not generated
from a known cost -- so the panel reports each method's recovered theta_hat and
a *reconstruction* error: EE-path RMSE between the demo and that method's
resolved x_hat(theta_hat), not regret against a true cost (there is none).

Usage:
    python examples/21_00_panda_ioc_teleop_viser.py
"""

from __future__ import annotations

import pathlib
import sys
import time

# `ioc` lives at the repo root, not on PYTHONPATH (unlike `pyroffi`, which is an
# installed src/ package) -- only reachable when run as `python -m` or with cwd
# on sys.path, neither of which holds for a plain `python examples/foo.py`.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroffi as pk
import viser
import yourdfpy
from viser.extras import ViserUrdf

from ioc import analytic
from ioc.inner import make_inner_solver
from ioc.robot import bases, problem as prob
from ioc.robot.e1_identifiability import make_dynamics_forward_solver
from pyroffi.optimization_engines._ls_ik import ls_ik_solve

RESOURCE_ROOT = pathlib.Path(__file__).resolve().parent.parent / "resources"
URDF_PATH = RESOURCE_ROOT / "panda" / "panda_spherized.urdf"
SRDF_PATH = RESOURCE_ROOT / "panda" / "panda.srdf"
MESH_DIR = RESOURCE_ROOT / "panda" / "meshes"

N_TIMESTEPS = 24
N_NEWTON = 60
METHODS = ("kkt", "cioc", "fd", "cmaes", "implicit")


def resample_waypoints(qs: list[np.ndarray], n_timesteps: int) -> np.ndarray:
    """Piecewise-linear resample of a waypoint sequence to n_timesteps rows.

    Teleop gives an irregular number of clicks; the IOC machinery needs a fixed
    grid with clamped endpoints, so this is the join between the two.
    """
    qs = np.stack(qs)
    if qs.shape[0] == 1:
        return np.repeat(qs, n_timesteps, axis=0)
    s_src = np.linspace(0.0, 1.0, qs.shape[0])
    s_dst = np.linspace(0.0, 1.0, n_timesteps)
    return np.stack([np.interp(s_dst, s_src, qs[:, j]) for j in range(qs.shape[1])], axis=-1)


def main() -> None:
    problem = prob.RobotProblem.load(str(URDF_PATH), str(SRDF_PATH), str(MESH_DIR), N_TIMESTEPS)
    urdf = yourdfpy.URDF.load(str(URDF_PATH), mesh_dir=str(MESH_DIR))

    rng = np.random.default_rng(0)
    scene0 = problem.sample_scenes(rng, 1)
    obs_center = np.asarray(scene0.obs_center[0])
    obs_radius = float(scene0.obs_radius[0, 0])
    q_start = np.asarray(scene0.q_start[0])
    q_goal = np.asarray(scene0.q_goal[0])

    residual_fn, names = bases.kinematic(problem, "k3")
    K = len(names)
    forward_solver = make_dynamics_forward_solver()

    def build(scales):
        return make_inner_solver(
            residual_fn, scales, n_iter=N_NEWTON, forward_solver=forward_solver,
        )

    server = viser.ViserServer()
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")
    urdf_vis.update_cfg(q_start)

    server.scene.add_icosphere(
        "/obstacle", radius=obs_radius, position=tuple(obs_center),
        color=(220, 60, 60),
    )
    ee_link_index = problem.ee_index

    target_handle = server.scene.add_transform_controls(
        "/ik_target", scale=0.2,
        position=tuple(np.asarray(problem.ee_positions(q_start))),
        wxyz=(0.0, 0.0, 1.0, 0.0),
    )

    demos: list[np.ndarray] = []  # each (n_timesteps, dof) resampled trajectory
    current_waypoints: list[np.ndarray] = [q_start.copy()]
    rng_key = jax.random.PRNGKey(0)
    ik_solution = q_start.copy()

    with server.gui.add_folder("Teleop"):
        n_wp_h = server.gui.add_number("Waypoints in current demo", initial_value=1, disabled=True)
        n_demo_h = server.gui.add_number("Demos recorded", initial_value=0, disabled=True)
        add_wp_btn = server.gui.add_button("Add waypoint")
        finish_btn = server.gui.add_button("Finish demo")
        reset_btn = server.gui.add_button("Reset current demo")

    with server.gui.add_folder("IOC"):
        run_btn = server.gui.add_button("Run IOC")
        status_h = server.gui.add_markdown("_no results yet_")

    @add_wp_btn.on_click
    def _(_) -> None:
        nonlocal ik_solution, rng_key
        target_pose = jaxlie.SE3.from_rotation_and_translation(
            rotation=jaxlie.SO3(wxyz=jnp.array(target_handle.wxyz)),
            translation=jnp.array(target_handle.position),
        )
        rng_key, subkey = jax.random.split(rng_key)
        ik_solution = np.asarray(ls_ik_solve(
            robot=problem.robot, target_link_indices=(ee_link_index,),
            target_poses=(target_pose,), rng_key=subkey,
            previous_cfg=jnp.asarray(ik_solution),
        ))
        current_waypoints.append(ik_solution.copy())
        urdf_vis.update_cfg(ik_solution)
        n_wp_h.value = len(current_waypoints)

    @reset_btn.on_click
    def _(_) -> None:
        current_waypoints.clear()
        current_waypoints.append(q_start.copy())
        n_wp_h.value = 1

    @finish_btn.on_click
    def _(_) -> None:
        if len(current_waypoints) < 2:
            return
        traj = resample_waypoints(current_waypoints, N_TIMESTEPS)
        demos.append(traj)
        n_demo_h.value = len(demos)
        current_waypoints.clear()
        current_waypoints.append(traj[-1].copy())
        n_wp_h.value = 1

    @run_btn.on_click
    def _(_) -> None:
        if len(demos) == 0:
            status_h.content = "_record at least one demo first_"
            return
        status_h.content = "_running..._"

        # One shared obstacle scene per demo, endpoints taken from the demo's
        # own clamped first/last row -- what "the demonstration" means here is
        # the teleoperated trajectory, not a synthetic solve under a theta*.
        scenes = prob.Scene(
            q_start=jnp.asarray(np.stack([d[0] for d in demos])),
            q_goal=jnp.asarray(np.stack([d[-1] for d in demos])),
            obs_center=jnp.asarray(np.tile(obs_center, (len(demos), 1))),
            obs_radius=jnp.asarray(np.full((len(demos), 1), obs_radius)),
        )
        demos_arr = jnp.asarray(np.stack(demos))
        x0s = problem.seeds(scenes)

        scales = problem.calibrate(residual_fn, scenes, jax.random.key(0))
        inner = build(scales)

        z0 = jnp.zeros(K)
        results = {}

        def reconstruction_rmse(z):
            theta = jax.nn.softmax(z)

            def one(scene, demo, x0):
                x_hat = inner.solve_implicit(x0, theta, scene)
                q_hat = problem.unpack(x_hat, scene)
                return jnp.sqrt(jnp.mean(jnp.sum(
                    (problem.ee_positions(q_hat) - problem.ee_positions(demo)) ** 2,
                    axis=-1,
                )))

            return float(jnp.mean(jax.vmap(one)(scenes, demos_arr, x0s)))

        t0 = time.perf_counter()
        z = analytic.kkt_fit(inner.grad_x, scenes, demos_arr, K)
        results["kkt"] = (z, time.perf_counter() - t0)

        t0 = time.perf_counter()
        z = analytic.cioc_fit(inner.grad_x, inner.gn_system, scenes, demos_arr, K)
        results["cioc"] = (z, time.perf_counter() - t0)

        from ioc import outer as outer_opt

        loss = prob.make_outer(problem, inner.solve_implicit, scenes, demos_arr, x0s)
        loss_j = jax.jit(loss)

        t0 = time.perf_counter()
        z, _ = outer_opt.adam(outer_opt.fd_grad_fn(loss_j, 1e-4), z0, lr=0.15, n_steps=25)
        results["fd"] = (z, time.perf_counter() - t0)

        t0 = time.perf_counter()
        z, _ = outer_opt.cma_es(loss_j, z0, n_gens=15, sigma0=0.5, seed=0)
        results["cmaes"] = (z, time.perf_counter() - t0)

        t0 = time.perf_counter()
        gf = jax.jit(jax.value_and_grad(loss))
        z, _ = outer_opt.adam(gf, z0, lr=0.15, n_steps=25)
        results["implicit"] = (z, time.perf_counter() - t0)

        lines = [f"**{len(demos)} demo(s), K={K} features ({', '.join(names)})**", "", "| method | " + " | ".join(names) + " | EE RMSE | wall |", "|---|" + "---|" * (K + 2)]
        for m in METHODS:
            z, wall = results[m]
            theta = np.asarray(jax.nn.softmax(z))
            rmse = reconstruction_rmse(z)
            lines.append(
                "| " + m + " | " + " | ".join(f"{t:.3f}" for t in theta)
                + f" | {rmse:.4f} | {wall:.1f}s |"
            )
        status_h.content = "\n".join(lines)
        print("\n".join(lines))

    print(f"Viser server: http://0.0.0.0:{server.get_port()}")
    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    main()
