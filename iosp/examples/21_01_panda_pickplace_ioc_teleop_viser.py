"""Interactive multi-segment IOC: teleop a full pick-and-place task, recover
IK-stage + trajopt-stage cost weights on the composed IK->trajopt planner
(`iosp.model.pickplace` -- the genuinely-composed design, not the retired flat-vector
one; see that module's docstring for the composition and why it matters).

Same gizmo-driven waypoint recording UX as `examples/21_00_panda_ioc_teleop_viser
.py`, generalized from one free-form reach to the fixed skeleton
approach -> grasp -> transport -> place: four "record <phase>" buttons let the
human demonstrate the whole task in one continuous session, phase by phase
(chosen over one undifferentiated recording because the resample step needs to
know the phase boundaries to build one fixed-length trajectory per segment,
matching `iosp.model.pickplace.PHASES`' per-phase waypoint counts).

There is no ground-truth theta* for a human demo, so -- exactly as in 21_00 --
"Run IOC" reports each method's recovered theta_hat (`iosp.model.pickplace.
THETA_IK_NAMES` + `THETA_TRAJOPT_NAMES`) and a reconstruction error (EE-path
RMSE against the recorded demo), not regret.  See `iosp/recovery_bench.py` for
the ground-truth (synthetic-theta*) version of this same recovery loop, used
there instead of FD agreement to validate the implicit adjoint -- FD disagrees
badly with the implicit gradient on this composed chain even after three
separate hard-branch-to-soft fixes in the trajopt forward solver (see
`iosp/pickplace.py`'s module docstring), so FD is not a meaningful cross-check
here either; this example still runs all three outer methods for interactive
comparison, but treat "implicit" as the one calibrated against ground truth.

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
from iosp.model import pickplace as pp
from pyroffi.optimization_engines._ls_ik import ls_ik_solve

THETA_NAMES = pp.THETA_IK_NAMES + pp.THETA_TRAJOPT_NAMES
K = pp.K_IK + pp.K_TRAJOPT
THETA_IK0 = jnp.array([0.05, 0.05], dtype=jnp.float32)  # seed for prob.seeds(...)

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

    forward_solver = pp.make_composed_forward_solver(n_iters=60)

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

    ee_link_index = problem.ee_index
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
            robot=problem.base.robot, target_link_indices=(ee_link_index,),
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

        # Demo path in EE space, built with the SAME per-phase boundary-row
        # dedup as `PickPlaceProblem.full_ee_path` (drop the repeated first
        # row of every phase after the first), so it is directly comparable to
        # a candidate theta's rollout without an indexing mismatch.
        demo_rows = []
        for i, p in enumerate(PHASES):
            q_phase = jnp.asarray(np.stack(phase_demo[p]))
            ee = problem.ee_positions(q_phase)
            demo_rows.append(ee[1:] if i > 0 else ee)
        demo_path = jnp.concatenate(demo_rows, axis=0)

        scene = pp.PickPlaceScene(
            q_start=jnp.asarray(phase_demo["approach"][0]),
            pick_pos=jnp.asarray(pick_pos), place_pos=jnp.asarray(place_pos),
            obs_center=jnp.asarray(obs_center), obs_radius=jnp.asarray([obs_radius]),
        )
        scenes = jax.tree.map(lambda a: a[None], scene)

        def split_trajopt(theta_trajopt):
            out, i = {}, 0
            for p in PHASES:
                n = len(pp.SEGMENT_FEATURES[p])
                out[p] = theta_trajopt[i:i + n]
                i += n
            return out

        x0_seed, phase_scenes_seed, _, _ = problem.seeds(scenes, THETA_IK0)
        inner_by_phase = {}
        for p in PHASES:
            residual_fn, _ = problem.make_segment_inner(p, forward_solver)
            scales = problem.calibrate_segment(p, residual_fn, phase_scenes_seed[p], jax.random.PRNGKey(0))
            inner_by_phase[p] = make_inner_solver(residual_fn, scales, forward_solver=forward_solver)

        def unpack_z(z):
            return z[:pp.K_IK], z[pp.K_IK:]

        def loss(z):
            theta_ik, z_trajopt = unpack_z(z)
            theta_trajopt_by_phase = split_trajopt(jax.nn.softmax(z_trajopt))
            x0, phase_scenes, _, _ = problem.seeds(scenes, theta_ik)
            _, _, xs, phase_scenes2 = problem.solve(theta_ik, theta_trajopt_by_phase, scenes, inner_by_phase, x0)
            path = problem.full_ee_path(scenes, xs, phase_scenes2, batch_index=0)
            return jnp.mean(jnp.sum((path - demo_path) ** 2, axis=-1))

        loss_j = jax.jit(loss)
        z0 = jnp.concatenate([THETA_IK0, jnp.zeros(pp.K_TRAJOPT, dtype=jnp.float32)])
        results = {}

        t0 = time.perf_counter()
        z, _ = outer_opt.adam(outer_opt.fd_grad_fn(loss_j, 1e-3), z0, lr=0.05, n_steps=10)
        results["fd"] = (z, time.perf_counter() - t0)

        t0 = time.perf_counter()
        z, _ = outer_opt.cma_es(loss_j, z0, n_gens=10, sigma0=0.3, seed=0)
        results["cmaes"] = (z, time.perf_counter() - t0)

        t0 = time.perf_counter()
        z, _ = outer_opt.adam(jax.jit(jax.value_and_grad(loss)), z0, lr=0.05, n_steps=10)
        results["implicit"] = (z, time.perf_counter() - t0)

        lines = [
            f"**pick-and-place demo, K={K} features (theta_ik + theta_trajopt)**", "",
            "| method | " + " | ".join(THETA_NAMES) + " | EE RMSE | wall |",
            "|---|" + "---|" * (K + 2),
        ]
        for m in METHODS:
            z, wall = results[m]
            theta_ik, z_trajopt = unpack_z(z)
            theta_trajopt = np.asarray(jax.nn.softmax(z_trajopt))
            theta_hat = np.concatenate([np.asarray(theta_ik), theta_trajopt])
            rmse = float(jnp.sqrt(loss(z)))
            lines.append(
                "| " + m + " | " + " | ".join(f"{t:.3f}" for t in theta_hat)
                + f" | {rmse:.4f} | {wall:.1f}s |"
            )
        status_h.content = "\n".join(lines)
        print("\n".join(lines))

    print(f"Viser server: http://0.0.0.0:{server.get_port()}")
    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    main()
