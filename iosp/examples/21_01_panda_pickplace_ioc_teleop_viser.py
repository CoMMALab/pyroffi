"""Interactive multi-segment IOC: teleop a full pick-and-place task, recover
per-segment cost weights on the composed 4-phase planner (`iosp.pickplace`).

Same gizmo-driven waypoint recording UX as `examples/21_00_panda_ioc_teleop_viser
.py`, generalized from one free-form reach to the fixed skeleton
approach -> grasp -> transport -> place: four "record <phase>" buttons let the
human demonstrate the whole task in one continuous session, phase by phase
(chosen over one undifferentiated recording because the resample step needs to
know the phase boundaries to build a `iosp.pickplace.PickPlaceScene`-consistent
demo of exactly `T_TOTAL` waypoints with the right per-phase counts).

There is no ground-truth theta* for a human demo, so -- exactly as in 21_00 --
"Run IOC" reports each method's recovered per-segment theta_hat and a
reconstruction error (EE-path RMSE against the recorded demo), not regret.

Usage:
    python iosp/examples/21_01_panda_pickplace_ioc_teleop_viser.py
"""

from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import viser
import yourdfpy
from viser.extras import ViserUrdf

from ioc import outer as outer_opt
from ioc.inner import make_inner_solver
from ioc.robot.e1_identifiability import make_dynamics_forward_solver
from iosp import pickplace as pp
from pyroffi.optimization_engines._ls_ik import ls_ik_solve

RESOURCE_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent / "resources"
URDF_PATH = RESOURCE_ROOT / "panda" / "panda_spherized.urdf"
SRDF_PATH = RESOURCE_ROOT / "panda" / "panda.srdf"
MESH_DIR = RESOURCE_ROOT / "panda" / "meshes"

PHASES = ("approach", "grasp", "transport", "place")
METHODS = ("fd", "cmaes", "implicit")


def resample_waypoints(qs: list[np.ndarray], n: int) -> np.ndarray:
    """Piecewise-linear resample of one phase's clicks to n rows (see 21_00)."""
    qs = np.stack(qs)
    if qs.shape[0] == 1:
        return np.repeat(qs, n, axis=0)
    s_src = np.linspace(0.0, 1.0, qs.shape[0])
    s_dst = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(s_dst, s_src, qs[:, j]) for j in range(qs.shape[1])], axis=-1)


def main() -> None:
    problem = pp.PickPlaceProblem.load(str(URDF_PATH), str(SRDF_PATH), str(MESH_DIR))
    urdf = yourdfpy.URDF.load(str(URDF_PATH), mesh_dir=str(MESH_DIR))

    q_start = np.array([0.0, -0.6, 0.0, -2.2, 0.0, 1.6, 0.8])
    pick_pos = np.array([0.4, 0.25, 0.25])
    place_pos = np.array([0.4, -0.25, 0.25])
    obs_center = np.array([0.35, 0.0, 0.4])
    obs_radius = 0.06

    forward_solver = make_dynamics_forward_solver()

    server = viser.ViserServer()
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")
    urdf_vis.update_cfg(q_start)

    server.scene.add_icosphere(
        "/obstacle", radius=obs_radius, position=tuple(obs_center), color=(220, 60, 60),
    )
    server.scene.add_box("/pick_target", dimensions=(0.05, 0.05, 0.05),
                          position=tuple(pick_pos), color=(60, 180, 90))
    server.scene.add_box("/place_target", dimensions=(0.05, 0.05, 0.05),
                          position=tuple(place_pos), color=(60, 90, 220))

    ee_link_index = problem.problem.ee_index
    target_handle = server.scene.add_transform_controls(
        "/ik_target", scale=0.2,
        position=tuple(np.asarray(problem.ee_positions(q_start))),
        wxyz=(0.0, 0.0, 1.0, 0.0),
    )

    # phase -> list of recorded joint configs for that phase (in order).
    phase_demo: dict[str, list[np.ndarray]] = {p: [] for p in PHASES}
    current_waypoints: list[np.ndarray] = [q_start.copy()]
    current_phase_idx = 0
    rng_key = jax.random.PRNGKey(0)
    ik_solution = q_start.copy()

    with server.gui.add_folder("Teleop"):
        phase_h = server.gui.add_text("Current phase", initial_value=PHASES[0], disabled=True)
        n_wp_h = server.gui.add_number("Waypoints in current phase", initial_value=1, disabled=True)
        add_wp_btn = server.gui.add_button("Add waypoint")
        finish_btn = server.gui.add_button("Finish phase")
        reset_btn = server.gui.add_button("Reset current phase")

    with server.gui.add_folder("IOC"):
        run_btn = server.gui.add_button("Run IOC")
        status_h = server.gui.add_markdown("_record all 4 phases first_")

    def _add_wp(_) -> None:
        nonlocal ik_solution, rng_key
        target_pose = jaxlie.SE3.from_rotation_and_translation(
            rotation=jaxlie.SO3(wxyz=jnp.array(target_handle.wxyz)),
            translation=jnp.array(target_handle.position),
        )
        rng_key, subkey = jax.random.split(rng_key)
        ik_solution = np.asarray(ls_ik_solve(
            robot=problem.problem.robot, target_link_indices=(ee_link_index,),
            target_poses=(target_pose,), rng_key=subkey,
            previous_cfg=jnp.asarray(ik_solution),
        ))
        current_waypoints.append(ik_solution.copy())
        urdf_vis.update_cfg(ik_solution)
        n_wp_h.value = len(current_waypoints)

    def _reset(_) -> None:
        current_waypoints.clear()
        current_waypoints.append(ik_solution.copy())
        n_wp_h.value = 1

    def _finish(_) -> None:
        nonlocal current_phase_idx
        if len(current_waypoints) < 2:
            return
        phase = PHASES[current_phase_idx]
        n_target = {"approach": pp.N_APPROACH, "grasp": pp.N_GRASP,
                    "transport": pp.N_TRANSPORT, "place": pp.N_PLACE}[phase]
        phase_demo[phase] = list(resample_waypoints(current_waypoints, n_target))
        current_phase_idx = min(current_phase_idx + 1, len(PHASES) - 1)
        phase_h.value = PHASES[current_phase_idx]
        current_waypoints.clear()
        current_waypoints.append(phase_demo[phase][-1].copy())
        n_wp_h.value = 1
        if all(len(phase_demo[p]) > 0 for p in PHASES):
            status_h.content = "_all 4 phases recorded -- ready to run IOC_"

    add_wp_btn.on_click(_add_wp)
    reset_btn.on_click(_reset)
    finish_btn.on_click(_finish)

    @run_btn.on_click
    def _(_) -> None:
        if not all(len(phase_demo[p]) > 0 for p in PHASES):
            status_h.content = "_record all 4 phases first_"
            return
        status_h.content = "_running..._"

        demo_traj = np.concatenate([phase_demo[p] for p in PHASES], axis=0)
        assert demo_traj.shape[0] == pp.T_TOTAL, demo_traj.shape
        demo = jnp.asarray(demo_traj)

        scene = pp.PickPlaceScene(
            q_start=jnp.asarray(demo_traj[0]),
            pick_pos=jnp.asarray(pick_pos), place_pos=jnp.asarray(place_pos),
            obs_center=jnp.asarray(obs_center), obs_radius=jnp.asarray([obs_radius]),
        )
        scenes = jax.tree.map(lambda a: a[None], scene)
        demos = demo[None]
        x0s = problem.seeds(scenes)

        scales = problem.calibrate(scenes, jax.random.key(0))
        inner = make_inner_solver(problem.residual_fn, scales,
                                   forward_solver=forward_solver)

        def loss(z):
            theta = jax.nn.softmax(z)

            def one(sc, dm, x0):
                x_hat = inner.solve_implicit(x0, theta, sc)
                q_hat = problem.unpack(x_hat, sc)
                return jnp.mean(jnp.sum(
                    (problem.ee_positions(q_hat) - problem.ee_positions(dm)) ** 2, axis=-1))

            return jnp.mean(jax.vmap(one)(scenes, demos, x0s))

        loss_j = jax.jit(loss)
        z0 = jnp.zeros(pp.K)
        results = {}

        t0 = time.perf_counter()
        z, _ = outer_opt.adam(outer_opt.fd_grad_fn(loss_j, 1e-4), z0, lr=0.15, n_steps=20)
        results["fd"] = (z, time.perf_counter() - t0)

        t0 = time.perf_counter()
        z, _ = outer_opt.cma_es(loss_j, z0, n_gens=12, sigma0=0.5, seed=0)
        results["cmaes"] = (z, time.perf_counter() - t0)

        t0 = time.perf_counter()
        z, _ = outer_opt.adam(jax.jit(jax.value_and_grad(loss)), z0, lr=0.15, n_steps=20)
        results["implicit"] = (z, time.perf_counter() - t0)

        lines = [
            f"**pick-and-place demo, K={pp.K} features**", "",
            "| method | " + " | ".join(pp.THETA_NAMES) + " | EE RMSE | wall |",
            "|---|" + "---|" * (pp.K + 2),
        ]
        for m in METHODS:
            z, wall = results[m]
            theta = np.asarray(jax.nn.softmax(z))
            rmse = float(jnp.sqrt(loss(z)))
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
