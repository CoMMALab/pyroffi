"""Sample Panda IK configurations across multiple cartesian box regions in one call.

This example exercises ``direct_sample_box_region_cuda`` (sample target points
uniformly inside each box, then solve IK to each one with the standard
``ls_ik_cuda`` LM kernel) and demonstrates the batched-box API: two boxes are
solved in a single call, with seeds round-robin'd across regions inside the
shared GPU launch.

Includes optional collision avoidance against a draggable sphere obstacle plus
self-collision pruning, both handled in-kernel by ``ls_ik_cuda``.

Prerequisite:
    bash src/pyroffi/cuda_kernels/build_ls_ik_cuda.sh
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import pyroffi as pk
import viser
import yourdfpy
from pyroffi.collision import RobotCollisionSpherized, Sphere
from pyroffi.optimization_engines import direct_sample_box_region_cuda
from pyroffi.optimization_engines._region_ik import _box_entropy
from viser.extras import ViserUrdf


def _box_corners_edges(box_min: np.ndarray, box_max: np.ndarray) -> np.ndarray:
    corners = np.array(
        [
            [box_min[0], box_min[1], box_min[2]],
            [box_max[0], box_min[1], box_min[2]],
            [box_max[0], box_max[1], box_min[2]],
            [box_min[0], box_max[1], box_min[2]],
            [box_min[0], box_min[1], box_max[2]],
            [box_max[0], box_min[1], box_max[2]],
            [box_max[0], box_max[1], box_max[2]],
            [box_min[0], box_max[1], box_max[2]],
        ],
        dtype=np.float32,
    )
    edges = np.array(
        [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ]
    )
    return corners[edges]


def _stats_markdown(
    per_box_counts: list[int],
    requested_samples: int,
    solve_ms: float,
    inside_ratios: list[float],
    err_means: list[float],
    entropies: list[float],
    max_entropy: float,
) -> str:
    lines = [f"**Solved in {solve_ms:.1f} ms across {len(per_box_counts)} boxes**", ""]
    for b, (cnt, ir, em, ent) in enumerate(
        zip(per_box_counts, inside_ratios, err_means, entropies)
    ):
        count_str = f"{cnt} / {requested_samples}" + (
            " (partial)" if cnt < requested_samples else ""
        )
        lines.append(
            f"- **Box {b}**: {count_str} samples, "
            f"inside {ir * 100.0:.2f}%, "
            f"err {em:.6f}, "
            f"ent {ent:.3f}/{max_entropy:.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--seeds-per-launch", type=int, default=2048)
    parser.add_argument("--restarts-per-target", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--pos-weight", type=float, default=50.0)
    parser.add_argument("--lambda-init", type=float, default=5e-3)
    parser.add_argument("--eps-pos", type=float, default=1e-4)
    parser.add_argument("--entropy-bins", type=int, default=10)
    parser.add_argument(
        "--target-entropy",
        type=float,
        default=None,
        help=(
            "Stop collecting samples once Shannon entropy of the EE-point "
            "distribution reaches this value (nats), per box."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    urdf = yourdfpy.URDF.load(
        "resources/panda/panda_spherized.urdf",
        mesh_dir="resources/panda/meshes",
    )
    robot = pk.Robot.from_urdf(urdf)
    robot_coll = RobotCollisionSpherized.from_urdf(urdf)

    target_link_name = "panda_hand"
    target_link_index = robot.links.names.index(target_link_name)

    fixed_joint_mask = jnp.zeros(
        (robot.joints.num_actuated_joints,), dtype=jnp.int32
    )

    sphere_radius = 0.06
    sphere_coll_template = Sphere.from_center_and_radius(
        np.array([0.0, 0.0, 0.0]), np.array([sphere_radius])
    )

    previous_cfg = (robot.joints.lower_limits + robot.joints.upper_limits) / 2.0

    # Two boxes, one on each side of the robot.
    default_centers = np.array(
        [[0.45, -0.25, 0.40], [0.45, 0.25, 0.40]], dtype=np.float32
    )
    default_dims = np.array(
        [[0.20, 0.20, 0.24], [0.20, 0.20, 0.24]], dtype=np.float32
    )
    n_boxes = default_centers.shape[0]

    call_kwargs = dict(
        robot=robot,
        target_link_index=target_link_index,
        previous_cfg=previous_cfg,
        num_samples=args.samples,
        seeds_per_launch=args.seeds_per_launch,
        restarts_per_target=args.restarts_per_target,
        max_iter=args.max_iter,
        pos_weight=args.pos_weight,
        lambda_init=args.lambda_init,
        eps_pos=args.eps_pos,
        fixed_joint_mask=fixed_joint_mask,
        memory_limit_gb=2.0,
        target_entropy=args.target_entropy,
        entropy_bins=args.entropy_bins,
        verbose=args.verbose,
    )

    max_entropy = float(np.log(args.entropy_bins**3))

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.add_grid("/ground", width=2.0, height=2.0, cell_size=0.1)

    urdf_vis = ViserUrdf(server, urdf, root_node_name="/panda")

    # Per-box sliders.
    box_sliders: list[dict] = []
    for b in range(n_boxes):
        folder = server.gui.add_folder(f"Box {b}")
        with folder:
            cx = server.gui.add_slider(
                f"Center X##{b}", min=0.15, max=0.85, step=0.01,
                initial_value=float(default_centers[b, 0]),
            )
            cy = server.gui.add_slider(
                f"Center Y##{b}", min=-0.6, max=0.6, step=0.01,
                initial_value=float(default_centers[b, 1]),
            )
            cz = server.gui.add_slider(
                f"Center Z##{b}", min=0.05, max=0.9, step=0.01,
                initial_value=float(default_centers[b, 2]),
            )
            sx = server.gui.add_slider(
                f"Size X##{b}", min=0.02, max=0.8, step=0.01,
                initial_value=float(default_dims[b, 0]),
            )
            sy = server.gui.add_slider(
                f"Size Y##{b}", min=0.02, max=0.8, step=0.01,
                initial_value=float(default_dims[b, 1]),
            )
            sz = server.gui.add_slider(
                f"Size Z##{b}", min=0.02, max=0.8, step=0.01,
                initial_value=float(default_dims[b, 2]),
            )
        box_sliders.append(dict(cx=cx, cy=cy, cz=cz, sx=sx, sy=sy, sz=sz))

    auto_resolve = server.gui.add_checkbox(
        "Auto Resolve On Box Change", initial_value=False
    )
    collision_enabled = server.gui.add_checkbox("Collision Avoidance", initial_value=True)
    collision_weight = server.gui.add_number(
        "Collision Weight", initial_value=1e3, step=1e2
    )
    collision_margin = server.gui.add_slider(
        "Collision Margin (m)", min=0.0, max=0.1, step=0.005, initial_value=0.02
    )
    resolve_btn = server.gui.add_button("Resolve Samples")
    status_md = server.gui.add_markdown("Preparing solver...")
    idx_slider = server.gui.add_slider(
        "Sample Index", min=0, max=0, step=1, initial_value=0
    )
    play = server.gui.add_checkbox("Play", initial_value=True)

    sphere_handle = server.scene.add_transform_controls(
        "/obstacle", scale=0.15, position=(0.55, 0.0, 0.55),
    )
    server.scene.add_mesh_trimesh(
        "/obstacle/mesh", mesh=sphere_coll_template.to_trimesh()
    )

    # Each row of cfgs_np corresponds to a (box, sample) pair flattened in row-major
    # order so the slider scrubs through every sample of every box.
    solve_nonce = 0
    needs_solve = True
    cfgs_np = np.zeros((n_boxes, 1, robot.joints.num_actuated_joints), dtype=np.float32)
    flat_cfgs_np = np.zeros((1, robot.joints.num_actuated_joints), dtype=np.float32)
    flat_box_idx = np.zeros((1,), dtype=np.int32)

    box_colors = np.array(
        [
            [255, 99, 132],   # box 0 — pink/red
            [99, 132, 255],   # box 1 — blue
            [99, 255, 132],   # box 2 — green
            [255, 200, 99],   # box 3 — orange
        ],
        dtype=np.uint8,
    )

    def _gui_boxes() -> tuple[np.ndarray, np.ndarray]:
        mins = np.zeros((n_boxes, 3), dtype=np.float32)
        maxs = np.zeros((n_boxes, 3), dtype=np.float32)
        for b, s in enumerate(box_sliders):
            center = np.array(
                [s["cx"].value, s["cy"].value, s["cz"].value], dtype=np.float32
            )
            dims = np.array(
                [s["sx"].value, s["sy"].value, s["sz"].value], dtype=np.float32
            )
            mins[b] = center - 0.5 * dims
            maxs[b] = center + 0.5 * dims
        return mins, maxs

    def _set_status(text: str) -> None:
        if hasattr(status_md, "content"):
            status_md.content = text
        elif hasattr(status_md, "markdown"):
            status_md.markdown = text
        else:
            status_md.value = text

    def _draw_boxes() -> None:
        mins, maxs = _gui_boxes()
        for b in range(n_boxes):
            color = box_colors[b % box_colors.shape[0]]
            server.scene.add_line_segments(
                f"/region/box_{b}",
                points=_box_corners_edges(mins[b], maxs[b]),
                colors=color,
                line_width=2.0,
            )

    def _solve_now() -> None:
        nonlocal solve_nonce, cfgs_np, flat_cfgs_np, flat_box_idx
        mins_np, maxs_np = _gui_boxes()
        box_min_jax = jnp.asarray(mins_np, dtype=jnp.float32)
        box_max_jax = jnp.asarray(maxs_np, dtype=jnp.float32)
        _set_status("Solving...")

        sphere_world = sphere_coll_template.transform_from_wxyz_position(
            wxyz=np.array(sphere_handle.wxyz),
            position=np.array(sphere_handle.position),
        )
        coll_kwargs = dict(
            collision_free=bool(collision_enabled.value),
            collision_checker=robot_coll,
            collision_world=sphere_world,
            collision_weight=float(collision_weight.value),
            collision_margin=float(collision_margin.value),
        )

        if solve_nonce == 0:
            rng_key_warmup = jax.random.PRNGKey(0)
            t0 = time.perf_counter()
            warm_cfgs, _, _, _ = direct_sample_box_region_cuda(
                rng_key=rng_key_warmup,
                box_min=box_min_jax,
                box_max=box_max_jax,
                **coll_kwargs,
                **call_kwargs,
            )
            warm_cfgs.block_until_ready()
            warmup_ms = (time.perf_counter() - t0) * 1000.0
            print(f"Warmup (JIT compile + run): {warmup_ms:.1f} ms")

        solve_nonce += 1
        rng_key_run = jax.random.PRNGKey(solve_nonce)
        t0 = time.perf_counter()
        cfgs, ee_points, target_points, errors = direct_sample_box_region_cuda(
            rng_key=rng_key_run,
            box_min=box_min_jax,
            box_max=box_max_jax,
            **coll_kwargs,
            **call_kwargs,
        )
        cfgs.block_until_ready()
        solve_ms = (time.perf_counter() - t0) * 1000.0
        print(f"Solve {solve_nonce}: {solve_ms:.1f} ms (n_boxes={n_boxes})")

        # Shapes: (n_boxes, n_samples_per_box, ...)
        cfgs_np = np.asarray(cfgs)
        ee_np = np.asarray(ee_points)
        tgt_np = np.asarray(target_points)
        err_np = np.asarray(errors)

        per_box_counts: list[int] = []
        inside_ratios: list[float] = []
        err_means: list[float] = []
        entropies: list[float] = []

        # Render each box's points in its own color, plus build the flat
        # cfg array used by the playback slider.
        flat_cfg_chunks: list[np.ndarray] = []
        flat_box_idx_chunks: list[np.ndarray] = []
        for b in range(n_boxes):
            ee_b = ee_np[b]
            tgt_b = tgt_np[b]
            err_b = err_np[b]
            cfg_b = cfgs_np[b]

            inside_b = np.all(
                (ee_b >= mins_np[b]) & (ee_b <= maxs_np[b]), axis=1
            )
            color = box_colors[b % box_colors.shape[0]]
            color_arr = np.broadcast_to(color, (ee_b.shape[0], 3)).astype(np.uint8)

            server.scene.add_point_cloud(
                f"/region/target_points_box_{b}",
                points=tgt_b,
                colors=color_arr,
                point_size=0.003,
                point_shape="sparkle",
            )
            server.scene.add_point_cloud(
                f"/region/ee_points_box_{b}",
                points=ee_b,
                colors=color_arr,
                point_size=0.004,
                point_shape="circle",
            )

            per_box_counts.append(int(cfg_b.shape[0]))
            inside_ratios.append(
                float(inside_b.mean()) if inside_b.size else 0.0
            )
            err_means.append(float(err_b.mean()) if err_b.size else 0.0)
            entropies.append(
                _box_entropy(ee_b, mins_np[b], maxs_np[b], args.entropy_bins)
            )

            flat_cfg_chunks.append(cfg_b)
            flat_box_idx_chunks.append(np.full((cfg_b.shape[0],), b, dtype=np.int32))

        flat_cfgs_np = np.concatenate(flat_cfg_chunks, axis=0)
        flat_box_idx = np.concatenate(flat_box_idx_chunks, axis=0)

        idx_slider.max = max(flat_cfgs_np.shape[0] - 1, 0)
        idx_slider.value = 0
        urdf_vis.update_cfg(flat_cfgs_np[0])
        _set_status(
            _stats_markdown(
                per_box_counts=per_box_counts,
                requested_samples=args.samples,
                solve_ms=solve_ms,
                inside_ratios=inside_ratios,
                err_means=err_means,
                entropies=entropies,
                max_entropy=max_entropy,
            )
        )

    def _request_solve(_: object | None = None) -> None:
        nonlocal needs_solve
        needs_solve = True

    resolve_btn.on_click(_request_solve)

    last_box_state = np.concatenate(
        [default_centers.reshape(-1), default_dims.reshape(-1)], axis=0
    )
    last_sphere_pos = np.array(sphere_handle.position, dtype=np.float32)
    _draw_boxes()

    print(f"Viewer running at http://{args.host}:{args.port}")

    while True:
        current_box_state = np.array(
            [
                v
                for s in box_sliders
                for v in (
                    s["cx"].value, s["cy"].value, s["cz"].value,
                    s["sx"].value, s["sy"].value, s["sz"].value,
                )
            ],
            dtype=np.float32,
        )
        if not np.allclose(current_box_state, last_box_state):
            last_box_state = current_box_state
            _draw_boxes()
            if auto_resolve.value:
                needs_solve = True

        current_sphere_pos = np.array(sphere_handle.position, dtype=np.float32)
        if not np.allclose(current_sphere_pos, last_sphere_pos):
            last_sphere_pos = current_sphere_pos
            if auto_resolve.value and collision_enabled.value:
                needs_solve = True

        if needs_solve:
            try:
                _solve_now()
            except Exception as exc:
                _set_status(f"Solve failed: `{exc}`")
            needs_solve = False

        if play.value and flat_cfgs_np.shape[0] > 0:
            idx_slider.value = (idx_slider.value + 1) % flat_cfgs_np.shape[0]
        urdf_vis.update_cfg(flat_cfgs_np[idx_slider.value])
        time.sleep(0.05)


if __name__ == "__main__":
    main()
