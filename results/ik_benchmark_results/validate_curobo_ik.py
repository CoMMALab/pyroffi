"""Validate CuRobo IK solutions using pyronot's collision checker.

Loads the bench_ik_curobo_*.pkl files (panda, fetch, baxter) and re-checks
each configuration against the bench_env_large.json obstacle scene using
pyronot's RobotCollisionSpherized with SRDF-masked self-collision pairs.

The goal is to detect whether CuRobo's collision checker has false negatives
(i.e., reports a configuration as collision-free when it is not).

For non-collision-free sections (sequential / batch), CuRobo never claimed
collision-free, so we just report the raw pyronot collision-free count without
a false-negative calculation.

DOF alignment:
  panda:  curobo=9 (7 arm + 2 fingers), pyronot=7 (arm only)  → keep first 7
  fetch:  curobo=8, pyronot=8                                  → direct
  baxter: curobo=14, pyronot=14                                → direct

Usage:
    # Print-only validation for all robots:
    python validate_curobo_ik.py

    # Visualize sequential configs for a specific robot (default: panda):
    python validate_curobo_ik.py --visualize
    python validate_curobo_ik.py --visualize --robot fetch --port 8080
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import trimesh
import yourdfpy

import pyronot as pk
from pyronot.collision import Box, RobotCollisionSpherized, Sphere, collide
from pyronot._robot_srdf_parser import read_disabled_collisions_from_srdf

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT     = pathlib.Path(__file__).resolve().parent.parent.parent
RESOURCE_ROOT = REPO_ROOT / "resources"
RESULTS_DIR   = pathlib.Path(__file__).resolve().parent

ENV_FILE = RESOURCE_ROOT / "bench_env_large.json"
CSV_FILE = RESULTS_DIR / "bench_ik_results_curobo.csv"

ROBOT_URDFS = {
    "panda":  RESOURCE_ROOT / "panda"  / "panda_spherized.urdf",
    "fetch":  RESOURCE_ROOT / "fetch"  / "fetch_spherized.urdf",
    "baxter": RESOURCE_ROOT / "baxter" / "baxter_spherized.urdf",
}

ROBOT_MESH_DIRS = {
    "panda":  RESOURCE_ROOT / "panda"  / "meshes",
    "fetch":  RESOURCE_ROOT / "fetch"  / "meshes",
    "baxter": RESOURCE_ROOT / "baxter" / "meshes",
}

ROBOT_SRDFS = {
    "panda":  RESOURCE_ROOT / "panda"  / "panda.srdf",
    "fetch":  RESOURCE_ROOT / "fetch"  / "fetch.srdf",
    "baxter": RESOURCE_ROOT / "baxter" / "baxter.srdf",
}

PKL_FILES = {
    "panda":  RESULTS_DIR / "bench_ik_curobo_panda.pkl",
    "fetch":  RESULTS_DIR / "bench_ik_curobo_fetch.pkl",
    "baxter": RESULTS_DIR / "bench_ik_curobo_baxter.pkl",
}

ROBOTS = ("panda", "fetch", "baxter")

# Sections where CuRobo did CF-IK (false-neg analysis applies).
CF_SECTIONS = {"sequential_collision_free", "batch_collision_free"}
# All sections (non-CF ones get world-collision rate only).
ALL_SECTIONS = (
    "sequential",
    "batch",
    "sequential_collision_free",
    "batch_collision_free",
)

# ---------------------------------------------------------------------------
# DOF alignment
# ---------------------------------------------------------------------------

def _align_configs(configs: np.ndarray, robot_name: str, pyronot_dof: int) -> np.ndarray:
    """Map curobo configs to pyronot's joint ordering/count."""
    curobo_dof = configs.shape[1]
    if curobo_dof == pyronot_dof:
        return configs
    if robot_name == "panda":
        assert curobo_dof == pyronot_dof + 2, (
            f"Unexpected panda DOF: curobo={curobo_dof}, pyronot={pyronot_dof}"
        )
        return configs[:, :pyronot_dof]
    raise ValueError(
        f"Unknown DOF mismatch for {robot_name}: curobo={curobo_dof}, pyronot={pyronot_dof}"
    )

# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _load_env_dict(env_file: pathlib.Path) -> dict:
    return json.loads(env_file.read_text())


def _env_dict_to_geoms(env: dict) -> list:
    obs_geoms: list = []
    for s in env.get("spheres", []):
        obs_geoms.append(
            Sphere.from_center_and_radius(
                np.array(s["center"], dtype=np.float32),
                np.array([s["radius"]], dtype=np.float32),
            )
        )
    for b in env.get("cuboids", []):
        d    = b["dims"]
        wxyz = b.get("wxyz", [1.0, 0.0, 0.0, 0.0])
        obs_geoms.append(
            Box.from_center_and_dimensions(
                np.array(b["center"], dtype=np.float32),
                float(d[0]), float(d[1]), float(d[2]),
                wxyz=np.array(wxyz, dtype=np.float32),
            )
        )
    return obs_geoms


def _disabled_pairs(srdf_path: pathlib.Path) -> tuple[tuple[str, str], ...]:
    if not srdf_path.exists():
        return ()
    pairs = read_disabled_collisions_from_srdf(srdf_path.as_posix())
    return tuple(
        (str(p["link1"]), str(p["link2"]))
        for p in pairs
        if p.get("link1") and p.get("link2")
    )


def _make_min_dist_checker(
    robot_coll: RobotCollisionSpherized,
    robot: pk.Robot,
    obs_geoms: list,
):
    """Return a vmapped fn: configs (N, n_dof) -> min_dist (N,)."""
    _coll_vs_world = jax.vmap(collide, in_axes=(-2, None), out_axes=-2)

    def _min_dist_single(cfg):
        coll_geom = robot_coll.at_config(robot, cfg)
        dists = [
            jnp.min(_coll_vs_world(coll_geom, obs.broadcast_to((1,))))
            for obs in obs_geoms
        ]
        if not dists:
            return jnp.inf
        return jnp.min(jnp.stack(dists))

    return jax.jit(jax.vmap(_min_dist_single))


def _parse_csv_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _update_source_csv(
    csv_path: pathlib.Path,
    validated_counts: dict[str, dict[str, int]],
) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames:
        raise ValueError(f"CSV has no header: {csv_path}")

    updates: list[tuple[int, str, str, int, str]] = []

    for robot_name, mode_to_count in validated_counts.items():
        for mode, pyronot_cf_count in mode_to_count.items():
            candidates: list[tuple[str, int]] = []
            for idx, row in enumerate(rows):
                solver_key = str(row.get("solver", "")).removesuffix("-BATCH")
                if (
                    str(row.get("robot", "")) == robot_name
                    and str(row.get("mode", "")) == mode
                    and solver_key == "CuRobo"
                    and _parse_csv_bool(str(row.get("collision_free", "")))
                ):
                    candidates.append((str(row.get("timestamp", "")), idx))

            if not candidates:
                print(
                    f"[WARN] No matching CSV row found for robot={robot_name}, "
                    f"mode={mode}, collision_free=True"
                )
                continue

            latest_timestamp, latest_idx = max(candidates, key=lambda x: x[0])
            rows[latest_idx]["coll_free_n"] = str(pyronot_cf_count)
            updates.append((latest_idx, robot_name, mode, pyronot_cf_count, latest_timestamp))

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nUpdated source CSV: {csv_path}")
    for _, robot_name, mode, count, ts in updates:
        print(
            f"  - robot={robot_name}, mode={mode}, timestamp={ts}: "
            f"coll_free_n={count}"
        )

# ---------------------------------------------------------------------------
# Validation (print table)
# ---------------------------------------------------------------------------

def run_validation(obs_geoms: list, modify_source: bool = False) -> None:
    col_w = [30, 5, 9, 9, 20]
    sep = " ".join("-" * w for w in col_w)
    validated_counts: dict[str, dict[str, int]] = {}

    for robot_name in ROBOTS:
        urdf_path = ROBOT_URDFS[robot_name]
        mesh_dir  = ROBOT_MESH_DIRS[robot_name]
        srdf_path = ROBOT_SRDFS[robot_name]
        pkl_path  = PKL_FILES[robot_name]

        print(f"{'='*65}")
        print(f"Robot: {robot_name}")
        urdf   = yourdfpy.URDF.load(str(urdf_path), mesh_dir=str(mesh_dir))
        robot  = pk.Robot.from_urdf(urdf)
        n_dof  = robot.joints.num_actuated_joints

        ignore_pairs = _disabled_pairs(srdf_path)
        robot_coll   = RobotCollisionSpherized.from_urdf(
            urdf,
            user_ignore_pairs=ignore_pairs,
        )
        print(f"  pyronot DOF: {n_dof}  |  SRDF pairs masked: {len(ignore_pairs)}")

        check_batch = _make_min_dist_checker(robot_coll, robot, obs_geoms)
        jax.block_until_ready(check_batch(jnp.zeros((1, n_dof), dtype=jnp.float32)))
        print(f"  Collision checker JIT ready.\n")

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        hdr = (
            f"{'Section':<{col_w[0]}} {'N':>{col_w[1]}} "
            f"{'CuRobo✓':>{col_w[2]}} {'Pyronot✓':>{col_w[3]}} "
            f"{'False-neg':>{col_w[4]}}"
        )
        print(hdr)
        print(sep)

        for section in ALL_SECTIONS:
            if section not in data:
                continue
            sec = data[section]

            configs_raw = np.array(sec["configs"])
            curobo_succ = np.array(sec["curobo_success"]).ravel().astype(bool)

            if configs_raw.ndim == 3:
                configs = configs_raw[:, 0, :]
            else:
                configs = configs_raw

            n_total = len(configs)
            configs_aligned = _align_configs(configs, robot_name, n_dof)

            min_dists    = np.array(check_batch(jnp.array(configs_aligned, dtype=jnp.float32)))
            pyronot_cf   = min_dists > 0.0
            n_pyronot_cf = int(pyronot_cf.sum())
            n_curobo_succ = int(curobo_succ.sum())

            is_cf_section = section in CF_SECTIONS
            if is_cf_section:
                csv_mode = "sequential" if section.startswith("sequential") else "batch"
                validated_counts.setdefault(robot_name, {})[csv_mode] = n_pyronot_cf
                false_neg_mask = curobo_succ & ~pyronot_cf
                n_false_neg    = int(false_neg_mask.sum())
                fn_rate        = n_false_neg / n_curobo_succ if n_curobo_succ > 0 else float("nan")
                fn_str = f"{n_false_neg}/{n_curobo_succ} ({fn_rate*100:.1f}%)"
            else:
                fn_str = "N/A (no CF solving)"

            print(
                f"{section:<{col_w[0]}} {n_total:>{col_w[1]}} "
                f"{n_curobo_succ:>{col_w[2]}} {n_pyronot_cf:>{col_w[3]}} "
                f"{fn_str:>{col_w[4]}}"
            )

        print()

    print("=" * 65)
    print("False-neg = CuRobo CF✓ but pyronot detects world collision.")
    print("Non-CF sections: CuRobo did not attempt collision avoidance.")
    print("min_dist threshold: 0.0 m (strict; > 0 = fully clear of obstacles).")

    if modify_source:
        _update_source_csv(CSV_FILE, validated_counts)

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def run_visualize(robot_name: str, obs_geoms: list, env_dict: dict, port: int) -> None:
    import viser
    from viser.extras import ViserUrdf

    urdf_path = ROBOT_URDFS[robot_name]
    mesh_dir  = ROBOT_MESH_DIRS[robot_name]
    srdf_path = ROBOT_SRDFS[robot_name]
    pkl_path  = PKL_FILES[robot_name]

    print(f"Loading robot: {robot_name}")
    urdf   = yourdfpy.URDF.load(str(urdf_path), mesh_dir=str(mesh_dir))
    robot  = pk.Robot.from_urdf(urdf)
    n_dof  = robot.joints.num_actuated_joints

    ignore_pairs = _disabled_pairs(srdf_path)
    robot_coll_spherized = RobotCollisionSpherized.from_urdf(
        urdf,
        user_ignore_pairs=ignore_pairs,
    )
    print(f"  pyronot DOF: {n_dof}  |  SRDF pairs masked: {len(ignore_pairs)}")

    check_batch = _make_min_dist_checker(robot_coll_spherized, robot, obs_geoms)
    jax.block_until_ready(check_batch(jnp.zeros((1, n_dof), dtype=jnp.float32)))

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Use the sequential section (10 configs).
    sec         = data["sequential"]
    configs_raw = np.array(sec["configs"])
    if configs_raw.ndim == 3:
        configs = configs_raw[:, 0, :]
    else:
        configs = configs_raw
    configs = _align_configs(configs, robot_name, n_dof)
    n_configs = len(configs)

    # Also grab the sequential_collision_free configs if available.
    sec_cf         = data.get("sequential_collision_free", {})
    configs_cf_raw = np.array(sec_cf.get("configs", np.zeros((0, 1, n_dof))))
    if configs_cf_raw.ndim == 3:
        configs_cf = configs_cf_raw[:, 0, :]
    else:
        configs_cf = configs_cf_raw
    if configs_cf.shape[0] > 0:
        configs_cf = _align_configs(configs_cf, robot_name, n_dof)

    # Pre-compute per-config min distances for all sequential configs.
    all_configs = configs
    has_cf = configs_cf.shape[0] > 0
    if has_cf:
        all_configs_cf = configs_cf

    min_dists_seq = np.array(
        check_batch(jnp.array(configs, dtype=jnp.float32))
    )
    min_dists_cf = (
        np.array(check_batch(jnp.array(configs_cf, dtype=jnp.float32)))
        if has_cf else np.zeros(0)
    )

    # ── Viser server ─────────────────────────────────────────────────────────
    server = viser.ViserServer(port=port)
    print(f"Viser server started. Open http://localhost:{port} in your browser.")

    server.scene.add_grid("/env/floor", width=4, height=4, cell_size=0.1)

    # Draw obstacles.
    for i, s in enumerate(env_dict.get("spheres", [])):
        name   = s.get("name", f"sphere_{i}")
        center = np.array(s["center"], dtype=np.float32)
        radius = float(s["radius"])
        obs    = Sphere.from_center_and_radius(center=center, radius=radius)
        server.scene.add_mesh_trimesh(f"/env/obstacles/{name}", obs.to_trimesh())

    for i, b in enumerate(env_dict.get("cuboids", [])):
        name   = b.get("name", f"cuboid_{i}")
        center = np.array(b["center"], dtype=np.float32)
        dims   = np.array(b["dims"], dtype=np.float32)
        wxyz   = np.array(b.get("wxyz", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32)
        obs    = Box.from_center_and_dimensions(
            center, float(dims[0]), float(dims[1]), float(dims[2]), wxyz=wxyz
        )
        server.scene.add_mesh_trimesh(f"/env/obstacles/{name}", obs.to_trimesh())

    # Robot visualizer.
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")

    # ── GUI ──────────────────────────────────────────────────────────────────
    with server.gui.add_folder("Configuration"):
        section_dropdown = server.gui.add_dropdown(
            "Section",
            options=["sequential", "sequential_collision_free"] if has_cf else ["sequential"],
            initial_value="sequential",
        )
        cfg_slider = server.gui.add_slider(
            "Config index", min=0, max=n_configs - 1, step=1, initial_value=0
        )
        status_label = server.gui.add_text("Collision status", initial_value="")
        dist_label   = server.gui.add_number(
            "Min obstacle dist (m)", initial_value=0.0, step=1e-4, disabled=True
        )
        show_collision_geom = server.gui.add_checkbox(
            "Show collision geometry", initial_value=True
        )

    state = {"section": "sequential", "idx": 0}

    def _update_robot() -> None:
        section = state["section"]
        idx     = state["idx"]

        if section == "sequential":
            cfg       = configs[idx]
            min_dist  = float(min_dists_seq[idx])
        else:
            cfg       = configs_cf[idx]
            min_dist  = float(min_dists_cf[idx])

        urdf_vis.update_cfg(cfg)
        if show_collision_geom.value:
            coll_mesh = robot_coll_spherized.at_config(
                robot, jnp.array(cfg, dtype=jnp.float32)
            ).to_trimesh()
            server.scene.add_mesh_trimesh("/robot/collision_geometry", coll_mesh)
        else:
            server.scene.add_mesh_trimesh("/robot/collision_geometry", trimesh.Trimesh())
        dist_label.value = round(min_dist, 6)
        if min_dist > 0.0:
            status_label.value = f"[{idx}] COLLISION-FREE (dist={min_dist:.4f} m)"
        else:
            status_label.value = f"[{idx}] IN COLLISION  (dist={min_dist:.4f} m)"

    @section_dropdown.on_update
    def _(_event) -> None:
        section = str(section_dropdown.value)
        state["section"] = section
        if section == "sequential":
            n = n_configs
        else:
            n = configs_cf.shape[0]
        cfg_slider.max   = n - 1
        cfg_slider.value = min(state["idx"], n - 1)
        state["idx"]     = int(cfg_slider.value)
        _update_robot()

    @cfg_slider.on_update
    def _(_event) -> None:
        state["idx"] = int(cfg_slider.value)
        _update_robot()

    @show_collision_geom.on_update
    def _(_event) -> None:
        _update_robot()

    # Initial render.
    _update_robot()

    print(f"Showing {n_configs} sequential configs for '{robot_name}'.")
    print("Use the 'Config index' slider to step through configurations.")
    print("Press Ctrl+C to exit.\n")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down.")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate CuRobo IK configs using pyronot's collision checker."
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Launch an interactive viser visualizer instead of printing the table.",
    )
    parser.add_argument(
        "--robot", choices=ROBOTS, default="panda",
        help="Robot to visualize (only used with --visualize). Default: panda.",
    )
    parser.add_argument(
        "--port", type=int, default=8080,
        help="Viser server port (only used with --visualize). Default: 8080.",
    )
    parser.add_argument(
        "--modify-source", action="store_true",
        help=(
            "Update bench_ik_results_curobo.csv coll_free_n values for the latest "
            "CuRobo collision_free=True rows using validated pyronot counts."
        ),
    )
    args = parser.parse_args()

    print(f"Loading environment: {ENV_FILE}")
    env_dict  = _load_env_dict(ENV_FILE)
    obs_geoms = _env_dict_to_geoms(env_dict)
    print(f"  {len(obs_geoms)} obstacle geometry objects loaded.\n")

    if args.visualize:
        if args.modify_source:
            parser.error("--modify-source cannot be used with --visualize.")
        run_visualize(args.robot, obs_geoms, env_dict, args.port)
    else:
        run_validation(obs_geoms, modify_source=args.modify_source)


if __name__ == "__main__":
    main()
