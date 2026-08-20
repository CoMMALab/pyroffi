"""cuRobo baseline child for the IK benchmark (runs in the ``curobo`` conda env).

This is the child process spawned by ``tests/bench_ik.py`` for the ``cuRobo``
solver.  It CANNOT import ``bench_ik`` (that module pulls in JAX/pyroffi at
import time, which are not installed in the curobo env), so it is
self-contained: the benchmark constants, the success thresholds, the timing
protocol, and the CSV row contract are all duplicated here.  Keep them in sync
with ``tests/bench_ik.py`` (see the ``_CSV_FIELDS`` and ``_write_csv`` blocks
there — the rows appended below must be byte-compatible with pyroffi's rows).

Protocol (mirrors pyroffi's bench and cuRobo's own ik_benchmark.py):

  * Targets come from the ``.npz`` sidecar that the pyroffi dispatcher writes
    during target generation (``bench_ik_targets_<robot>.npz``): key ``seq``
    is (32, 7) and ``batch`` is (256, 7), both wxyz_xyz float32.
  * Four rows, built and timed SEQUENTIALLY (one solver at a time, freed
    between blocks to bound VRAM):
      - sequential, no scene   (max_batch_size=1, 32 problems)
      - sequential, with scene (max_batch_size=1, 32 problems)
      - batch,      no scene   (max_batch_size=256, 256 problems)
      - batch,      with scene (max_batch_size=256, 256 problems)
  * Sequential: per problem — one correctness solve, then N_TIMED timed reps,
    each preceded by ``reset_seed()`` (fresh multistarts, same as pyroffi's
    fresh-seed timed reps).  Per-problem time = median of the reps.
  * Batch: one correctness solve over all targets, then N_TIMED timed reps
    with fresh seeds; effective per-problem time = median(rep)/n_targets.
    The timed loop is wrapped in the NVML monitor (same as pyroffi).
  * Warmup: 3 solves with ``exit_early=False`` (full optimizer path) so the
    worst-case CUDA graph is captured before any timed solve; correctness and
    timed solves run with ``exit_early=True`` (cuRobo's benchmark protocol).
  * cuRobo's native tolerances are 5 mm / 0.05 rad (its published protocol);
    success is still SCORED at pyroffi's 1 mm / 0.05 rad thresholds, applied
    to ``result.position_error`` / ``result.rotation_error`` (meters/radians,
    max over tool links).
  * ``coll_free_n`` is cuRobo's own ``result.feasible`` (self + world
    collision and joint-limit check), counted over the correctness solve.

Robot frames: cuRobo ships configs only for panda (franka.yml) and g1
(unitree_g1.yml).  The per-robot tool frame is narrowed in-memory BEFORE
``IKSolverCfg.create`` so goals and comparison happen in the SAME frame as
pyroffi's targets: panda → ``panda_hand`` (franka.yml's default), g1 →
``right_hand_palm_link`` (pyroffi's EE link; present in unitree_g1.yml's
kinematics even though the shipped config lists the 4 fingertip + ankle
frames).  No static offset is needed.

Usage (normally spawned by the dispatcher — see ``_run_solver_subprocess`` in
``tests/bench_ik.py``):

    python bench_ik_curobo.py --robot panda \
        --targets resources/bench_ik_targets_panda.npz \
        --outdir resources --env-file resources/bench_env_large.json
"""

from __future__ import annotations

import contextlib
import copy
import csv
import datetime
import json
import os
import pathlib
import threading
import xml.etree.ElementTree as ET

import curobo.runtime as runtime

# Must be set BEFORE importing the solver stack (same as cuRobo's own
# benchmark): disable torch.compile/jit so timings reflect eager kernels.
runtime.enable_torch_compile = False
runtime.enable_torch_jit = False

# Standard Library
import argparse

# Third Party
import numpy as np
import torch

# CuRobo
from curobo._src.geom.types import SceneCfg
from curobo._src.solver.solver_ik import IKSolver
from curobo._src.solver.solver_ik_cfg import IKSolverCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import GoalToolPose
from curobo._src.util.cuda_event_timer import CudaEventTimer
from curobo._src.util.logging import setup_curobo_logger
from curobo._src.util_file import (
    get_assets_path,
    get_robot_configs_path,
    join_path,
    load_yaml,
)

# Enable CUDA event timing for accurate GPU measurements (must come after the
# curobo imports, exactly like cuRobo's own ik_benchmark.py).
runtime.enable_cuda_event_timer = True

# Seeds / precision flags — identical to cuRobo's own ik_benchmark.py.
torch.manual_seed(2)
np.random.seed(2)

torch._dynamo.config.compiled_autograd = True
torch._dynamo.config.cache_size_limit = 64

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------------------------------------------------------------------------
# Constants — keep in sync with tests/bench_ik.py
# ---------------------------------------------------------------------------

N_TARGETS = 32          # sequential problems
N_TARGETS_BATCH = 256   # batch problems
N_WARMUP = 3            # full-optimizer warmup solves
N_TIMED = 5             # timed reps per problem
N_SEEDS = 32            # cuRobo LM/seed multistarts (uniform across robots)

POS_THR_M = 1e-3        # pyroffi success threshold (m)
ROT_THR_RAD = 0.05      # pyroffi success threshold (rad)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CSV_NAME = "bench_ik_results.csv"

# cuRobo robot config files (same mapping as _CUROBO_ROBOT_FILES in bench_ik.py)
# and the tool frame each must be narrowed to so that goals/comparison happen
# in pyroffi's EE frame (see the module docstring).
_ROBOT_FILES = {"panda": "franka.yml", "g1": "unitree_g1.yml"}
_TOOL_FRAMES = {"panda": ["panda_hand"], "g1": ["right_hand_palm_link"]}

# cspace joints that fall OUTSIDE the narrowed tool-frame tree and therefore
# must be explicitly locked (this cuRobo's KinematicsLoader validation raises
# on any cspace joint that is neither in the tree nor locked).  panda's finger
# locks ship in franka.yml itself; unitree_g1.yml ships NO lock_joints, so with
# the single-frame right_hand_palm_link goal the tree is base→waist→right arm
# →palm: both legs (sibling branches of the waist chain), the entire left arm
# + hand, and the right fingers (palm is the tool frame) all need locking.
# 0.0 is within every joint's URDF limits and is a neutral pose (locked joints
# do not affect the palm pose; they only participate in the collision
# variant's feasibility checks).
_LOCK_JOINTS = {
    "g1": {
        # left leg
        "left_hip_pitch_joint": 0.0, "left_hip_roll_joint": 0.0,
        "left_hip_yaw_joint": 0.0, "left_knee_joint": 0.0,
        "left_ankle_pitch_joint": 0.0, "left_ankle_roll_joint": 0.0,
        # right leg
        "right_hip_pitch_joint": 0.0, "right_hip_roll_joint": 0.0,
        "right_hip_yaw_joint": 0.0, "right_knee_joint": 0.0,
        "right_ankle_pitch_joint": 0.0, "right_ankle_roll_joint": 0.0,
        # left arm
        "left_shoulder_pitch_joint": 0.0, "left_shoulder_roll_joint": 0.0,
        "left_shoulder_yaw_joint": 0.0, "left_elbow_joint": 0.0,
        "left_wrist_roll_joint": 0.0, "left_wrist_pitch_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
        # left hand
        "left_hand_thumb_0_joint": 0.0, "left_hand_thumb_1_joint": 0.0,
        "left_hand_thumb_2_joint": 0.0, "left_hand_middle_0_joint": 0.0,
        "left_hand_middle_1_joint": 0.0, "left_hand_index_0_joint": 0.0,
        "left_hand_index_1_joint": 0.0,
        # right fingers (palm is the tool frame, so these are outside the tree)
        "right_hand_thumb_0_joint": 0.0, "right_hand_thumb_1_joint": 0.0,
        "right_hand_thumb_2_joint": 0.0, "right_hand_middle_0_joint": 0.0,
        "right_hand_middle_1_joint": 0.0, "right_hand_index_0_joint": 0.0,
        "right_hand_index_1_joint": 0.0,
    },
}

# Row contract — MUST match _CSV_FIELDS in tests/bench_ik.py.
_CSV_FIELDS = [
    "timestamp", "robot", "mode", "solver", "collision_free",
    "n_problems", "n_timed",
    "t_med_ms", "t_p95_ms",
    "pos_med_mm", "pos_p95_mm",
    "rot_med_rad", "rot_p95_rad",
    "success_n", "success_total",
    "coll_free_n",
    "peak_gpu_pct", "avg_gpu_pct", "peak_vram_mb",
]


# ---------------------------------------------------------------------------
# GPU monitoring (NVML) — duplicated from tests/bench_ik.py.  pynvml is
# optional here: without it the batch rows simply get empty GPU columns.
# ---------------------------------------------------------------------------

try:
    import pynvml as _pynvml

    _pynvml.nvmlInit()
    _cve = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    _cve_idx = int(_cve.split(",")[0].strip()) if _cve.strip() else 0
    _NVML_HANDLE = _pynvml.nvmlDeviceGetHandleByIndex(_cve_idx)
    _NVML_OK = True
except Exception:
    _NVML_HANDLE = None
    _NVML_OK = False


@contextlib.contextmanager
def _gpu_monitor(interval_s: float = 0.02):
    samples: dict[str, list[float]] = {"gpu_util": [], "vram_mb": []}
    stop_evt = threading.Event()

    def _sample() -> None:
        while not stop_evt.is_set():
            if _NVML_OK and _NVML_HANDLE is not None:
                util = _pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE)
                mem = _pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE)
                samples["gpu_util"].append(float(util.gpu))
                samples["vram_mb"].append(float(mem.used) / 1024 ** 2)
            stop_evt.wait(interval_s)

    t = threading.Thread(target=_sample, daemon=True)
    t.start()
    try:
        yield samples
    finally:
        stop_evt.set()
        t.join(timeout=1.0)


# ---------------------------------------------------------------------------
# Solver construction and solve loops
# ---------------------------------------------------------------------------

def _env_to_scene_dict(env: dict) -> dict | None:
    """Convert the env JSON to the SceneCfg dict schema that
    ``IKSolverCfg.create(scene_model=...)`` consumes:
    ``{"sphere": {name: {radius, pose}}, "cuboid": {name: {dims, pose}}}`` with
    ``pose = [x, y, z, qw, qx, qy, qz]`` (the format ``SceneCfg.create``
    passes straight to the ``Sphere``/``Cuboid`` dataclasses).

    Prefers the pre-built ``curobo_world_model`` key when present (written by
    bench_ik.py's ``_build_and_save_env``), else falls back to the raw
    ``spheres``/``cuboids`` lists — the SAME source pyroffi's ``_env_to_geoms``
    consumes — so both sides always collide against the same obstacles.  The
    floor plane has no SceneCfg primitive and is omitted, matching the
    pre-built key (which also excludes it).  Returns None if there are no
    obstacles at all.
    """
    d = env.get("curobo_world_model")
    if d is None:
        d = {}
        spheres = {
            s["name"]: {"radius": s["radius"], "pose": [*s["center"], 1, 0, 0, 0]}
            for s in env.get("spheres", [])
        }
        if spheres:
            d["sphere"] = spheres
        cuboids = {}
        for b in env.get("cuboids", []):
            wxyz = b.get("wxyz", [1.0, 0.0, 0.0, 0.0])
            cuboids[b["name"]] = {"dims": b["dims"], "pose": [*b["center"], *wxyz]}
        if cuboids:
            d["cuboid"] = cuboids
    if not d:
        return None
    return d


def _make_goal(pose7: np.ndarray, tool_frames: list[str], device: torch.device) -> GoalToolPose:
    """Build a GoalToolPose from (B, 7) wxyz_xyz numpy poses."""
    pos = torch.from_numpy(np.ascontiguousarray(pose7[:, 4:7])).to(
        device, dtype=torch.float32
    )
    quat = torch.from_numpy(np.ascontiguousarray(pose7[:, 0:4])).to(
        device, dtype=torch.float32
    )
    return GoalToolPose.from_poses(
        {tool_frames[0]: Pose(position=pos, quaternion=quat)},
        ordered_tool_frames=tool_frames,
    )


def _build_solver(
    robot_data: dict,
    world_dict: dict | None,
    max_batch_size: int,
    collision_free: bool,
) -> IKSolver:
    """Build one IKSolver.  *robot_data* is deep-copied because the
    no-scene variant mutates the collision model (matching cuRobo's own
    benchmark).  *world_dict* is the raw SceneCfg dict (see
    ``_env_to_scene_dict``); passing a dict routes through
    ``create_solver_core_cfg`` → ``SceneCfg.create`` exactly like a YAML
    path would."""
    rd = copy.deepcopy(robot_data)
    if not collision_free:
        # Drop the collision model entirely — this is cuRobo's "collision
        # free" benchmark configuration.  NOTE: unlike cuRobo's benchmark we
        # must NOT null lock_joints here — the finger joints are in cspace
        # but outside the panda_hand tree, so this cuRobo version's
        # KinematicsLoader validation requires them to stay locked.
        rd["kinematics"]["collision_link_names"] = None
    cfg = IKSolverCfg.create(
        robot=rd,
        optimizer_configs=["ik/lbfgs_ik.yml"],
        metrics_rollout="metrics_base.yml",
        transition_model="ik/transition_ik.yml",
        scene_model=world_dict if collision_free else None,
        self_collision_check=collision_free,
        device_cfg=DeviceCfg(),
        num_seeds=N_SEEDS,
        # cuRobo's native tolerances (its published protocol scores at 5 mm);
        # our rows still score success at the 1 mm / 0.05 rad thresholds.
        position_tolerance=0.005,
        orientation_tolerance=0.05,
        use_cuda_graph=True,
        optimizer_collision_activation_distance=0.0025,
        # seed_solver_num_seeds stays at the uniform 32: the g1-specific
        # 128-seed / 240-iter overrides in cuRobo's benchmark compensate the
        # multi-link (4 fingertip) goal, which we do not use.
        seed_solver_num_seeds=N_SEEDS,
        max_batch_size=max_batch_size,
    )
    return IKSolver(cfg)


def _run_sequential(solver: IKSolver, poses: np.ndarray, tool_frames: list[str],
                    device: torch.device) -> tuple[list[tuple[float, float, float]], int]:
    """One correctness solve + N_TIMED timed reps per problem.

    Returns ([(pos_err_m, rot_err_rad, time_ms), ...], n_feasible).
    """
    solver.config.exit_early = False
    goal0 = _make_goal(poses[:1], tool_frames, device)
    for _ in range(N_WARMUP):
        solver.reset_seed()
        solver.solve_pose(goal_tool_poses=goal0, seed_config=None)
    torch.cuda.empty_cache()

    solver.config.exit_early = True
    results: list[tuple[float, float, float]] = []
    n_feasible = 0
    for i in range(len(poses)):
        goal = _make_goal(poses[i : i + 1], tool_frames, device)
        solver.reset_seed()
        res = solver.solve_pose(goal_tool_poses=goal, seed_config=None)
        pos_err = float(res.position_error.view(-1)[0])
        rot_err = float(res.rotation_error.view(-1)[0])
        n_feasible += int(bool(res.feasible.view(-1)[0]))

        times: list[float] = []
        for _ in range(N_TIMED):
            solver.reset_seed()
            timer = CudaEventTimer().start()
            solver.solve_pose(goal_tool_poses=goal, seed_config=None)
            times.append(timer.stop() * 1e3)
        results.append((pos_err, rot_err, float(np.median(times))))
    return results, n_feasible


def _run_batch(solver: IKSolver, poses: np.ndarray, tool_frames: list[str],
               device: torch.device) -> tuple[np.ndarray, np.ndarray, float, int,
                                              tuple[float, float, float]]:
    """One correctness solve over the whole batch + N_TIMED timed reps.

    Returns (pos_errs_m, rot_errs_rad, effective_ms_per_problem, n_feasible,
    (peak_gpu_pct, avg_gpu_pct, peak_vram_mb)).
    """
    solver.config.exit_early = False
    goal = _make_goal(poses, tool_frames, device)
    for _ in range(N_WARMUP):
        solver.reset_seed()
        solver.solve_pose(goal_tool_poses=goal, seed_config=None)
    torch.cuda.empty_cache()

    solver.config.exit_early = True
    solver.reset_seed()
    res = solver.solve_pose(goal_tool_poses=goal, seed_config=None)
    pos_errs = res.position_error.view(-1).cpu().numpy().astype(np.float64)
    rot_errs = res.rotation_error.view(-1).cpu().numpy().astype(np.float64)
    n_feasible = int(res.feasible.view(-1).sum())

    times: list[float] = []
    with _gpu_monitor() as samples:
        for _ in range(N_TIMED):
            solver.reset_seed()
            timer = CudaEventTimer().start()
            solver.solve_pose(goal_tool_poses=goal, seed_config=None)
            times.append(timer.stop() * 1e3)
    effective_ms = float(np.median(times)) / len(poses)
    gpu = (
        float(np.max(samples["gpu_util"])) if samples["gpu_util"] else float("nan"),
        float(np.mean(samples["gpu_util"])) if samples["gpu_util"] else float("nan"),
        float(np.max(samples["vram_mb"])) if samples["vram_mb"] else float("nan"),
    )
    return pos_errs, rot_errs, effective_ms, n_feasible, gpu


# ---------------------------------------------------------------------------
# CSV rows — byte-compatible with _write_csv in tests/bench_ik.py
# ---------------------------------------------------------------------------

def _seq_row(ts: str, robot: str, collision_free: bool,
             results: list[tuple[float, float, float]],
             coll_free: int | None) -> dict:
    pos = np.array([r[0] * 1e3 for r in results])
    rot = np.array([r[1] for r in results])
    t = np.array([r[2] for r in results])
    solved = sum(r[0] < POS_THR_M and r[1] < ROT_THR_RAD for r in results)
    return {
        "timestamp":      ts,
        "robot":          robot,
        "mode":           "sequential",
        "solver":         "cuRobo",
        "collision_free": collision_free,
        "n_problems":     len(results),
        "n_timed":        N_TIMED,
        "t_med_ms":       round(float(np.median(t)),        6),
        "t_p95_ms":       round(float(np.percentile(t, 95)), 6),
        "pos_med_mm":     round(float(np.median(pos)),        6),
        "pos_p95_mm":     round(float(np.percentile(pos, 95)), 6),
        "rot_med_rad":    round(float(np.median(rot)),        6),
        "rot_p95_rad":    round(float(np.percentile(rot, 95)), 6),
        "success_n":      solved,
        "success_total":  len(results),
        "coll_free_n":    coll_free if coll_free is not None else "",
        "peak_gpu_pct":   "",
        "avg_gpu_pct":    "",
        "peak_vram_mb":   "",
    }


def _batch_row(ts: str, robot: str, collision_free: bool,
               pos_errs: np.ndarray, rot_errs: np.ndarray, time_ms: float,
               coll_free: int | str, gpu: tuple[float, float, float]) -> dict:
    pos = pos_errs * 1e3
    rot = rot_errs
    solved = int(np.sum((pos_errs < POS_THR_M) & (rot_errs < ROT_THR_RAD)))

    def _fmtf(v):
        return round(float(v), 6) if not np.isnan(v) else ""

    return {
        "timestamp":      ts,
        "robot":          robot,
        "mode":           "batch",
        "solver":         "cuRobo",
        "collision_free": collision_free,
        "n_problems":     len(pos),
        "n_timed":        N_TIMED,
        "t_med_ms":       round(time_ms, 6),
        "t_p95_ms":       "",
        "pos_med_mm":     round(float(np.median(pos)),         6),
        "pos_p95_mm":     round(float(np.percentile(pos, 95)), 6),
        "rot_med_rad":    round(float(np.median(rot)),         6),
        "rot_p95_rad":    round(float(np.percentile(rot, 95)), 6),
        "success_n":      solved,
        "success_total":  len(pos),
        "coll_free_n":    coll_free,
        "peak_gpu_pct":   _fmtf(gpu[0]),
        "avg_gpu_pct":    _fmtf(gpu[1]),
        "peak_vram_mb":   _fmtf(gpu[2]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="cuRobo IK benchmark child (see module docstring)"
    )
    parser.add_argument("--robot", required=True, choices=sorted(_ROBOT_FILES))
    parser.add_argument(
        "--targets",
        type=pathlib.Path,
        required=True,
        help="npz sidecar written by the pyroffi dispatcher (keys: seq, batch)",
    )
    parser.add_argument("--outdir", type=pathlib.Path, default=REPO_ROOT / "resources")
    parser.add_argument(
        "--env-file",
        type=pathlib.Path,
        default=REPO_ROOT / "resources" / "bench_env_large.json",
    )
    args = parser.parse_args()

    setup_curobo_logger("error")

    if not args.targets.exists():
        raise SystemExit(
            f"targets file not found: {args.targets}\n"
            f"  It is written by the pyroffi dispatcher during target generation "
            f"(see tests/bench_ik.py); run the dispatcher, not this child directly."
        )

    device = torch.device("cuda")
    data = np.load(args.targets)
    seq_poses = data["seq"].astype(np.float32)
    batch_poses = data["batch"].astype(np.float32)
    if seq_poses.shape != (N_TARGETS, 7):
        raise SystemExit(f"targets 'seq' shape is {seq_poses.shape}, expected ({N_TARGETS}, 7)")
    if batch_poses.shape != (N_TARGETS_BATCH, 7):
        raise SystemExit(
            f"targets 'batch' shape is {batch_poses.shape}, expected ({N_TARGETS_BATCH}, 7)"
        )

    env = json.loads(args.env_file.read_text())
    world_dict = _env_to_scene_dict(env)
    if world_dict is not None:
        # Fail fast on a schema mismatch (validates field names against the
        # Sphere/Cuboid dataclasses) before any solver is built.
        SceneCfg.create(world_dict)

    robot_data = load_yaml(join_path(get_robot_configs_path(), _ROBOT_FILES[args.robot]))
    if "kinematics" not in robot_data:
        # Newer robot YAMLs wrap the config under "robot_cfg" (franka.yml);
        # g1's ships kinematics at the top level (same unwrap as cuRobo's
        # own benchmark).
        robot_data = robot_data["robot_cfg"]
    tool_frames = _TOOL_FRAMES[args.robot]
    # Links live in the URDF, not the YAML — validate the tool frame is a
    # real link before building any solver (fails fast on a frame-name typo).
    kin = robot_data["kinematics"]
    urdf_path = join_path(get_assets_path(), kin["urdf_path"])
    urdf_links = {el.get("name") for _, el in ET.iterparse(urdf_path) if el.tag == "link"}
    urdf_links |= set(kin.get("extra_links") or {})
    if tool_frames[0] not in urdf_links:
        raise SystemExit(
            f"tool frame '{tool_frames[0]}' not found in {args.robot}'s URDF links "
            f"({pathlib.Path(urdf_path).name})"
        )
    # Narrow to pyroffi's EE frame (see module docstring) BEFORE create().
    kin["tool_frames"] = tool_frames
    # Lock the cspace joints the narrowed tree drops (see _LOCK_JOINTS).
    if args.robot in _LOCK_JOINTS:
        kin["lock_joints"] = _LOCK_JOINTS[args.robot]

    ts = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    print(f"cuRobo baseline: robot={args.robot} tool_frames={tool_frames}")
    print(f"  targets: seq={seq_poses.shape} batch={batch_poses.shape} "
          f"({args.targets.name})")
    n_obs = len(SceneCfg.create(world_dict).objects) if world_dict is not None else 0
    print(f"  scene: {args.env_file.name} ({n_obs} obstacles), num_seeds={N_SEEDS}")

    rows: list[dict] = []

    # 1) sequential, no scene
    print("\n[cuRobo] sequential, no scene ...")
    solver = _build_solver(robot_data, world_dict, 1, collision_free=False)
    seq_results, _ = _run_sequential(solver, seq_poses, tool_frames, device)
    del solver
    torch.cuda.empty_cache()
    rows.append(_seq_row(ts, args.robot, False, seq_results, None))

    # 2) sequential, with scene
    print("[cuRobo] sequential, with scene ...")
    solver = _build_solver(robot_data, world_dict, 1, collision_free=True)
    seq_coll_results, seq_coll_free = _run_sequential(solver, seq_poses, tool_frames, device)
    del solver
    torch.cuda.empty_cache()
    rows.append(_seq_row(ts, args.robot, True, seq_coll_results, seq_coll_free))

    # 3) batch, no scene
    print("[cuRobo] batch, no scene ...")
    solver = _build_solver(robot_data, world_dict, N_TARGETS_BATCH, collision_free=False)
    b_pos, b_rot, b_ms, _, b_gpu = _run_batch(solver, batch_poses, tool_frames, device)
    del solver
    torch.cuda.empty_cache()
    # "" (not 0): no collision check ran, so coll_free_n is not measured —
    # matches the sequential no-scene row and the pyroffi dispatcher rows.
    rows.append(_batch_row(ts, args.robot, False, b_pos, b_rot, b_ms, "", b_gpu))

    # 4) batch, with scene
    print("[cuRobo] batch, with scene ...")
    solver = _build_solver(robot_data, world_dict, N_TARGETS_BATCH, collision_free=True)
    bc_pos, bc_rot, bc_ms, bc_feasible, bc_gpu = _run_batch(
        solver, batch_poses, tool_frames, device
    )
    del solver
    torch.cuda.empty_cache()
    rows.append(_batch_row(ts, args.robot, True, bc_pos, bc_rot, bc_ms, bc_feasible, bc_gpu))

    # -- summary table -------------------------------------------------------
    print("\ncuRobo baseline results (scored at 1 mm / 0.05 rad):")
    hdr = (f"  {'mode':<12} {'scene':<7} {'t_med(ms)':>10} {'t_p95(ms)':>10} "
           f"{'pos_med(mm)':>11} {'rot_med(rad)':>12} {'success':>9} {'coll_free':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        print(
            f"  {r['mode']:<12} {'yes' if r['collision_free'] else 'no':<7} "
            f"{r['t_med_ms']:>10} {str(r['t_p95_ms']):>10} "
            f"{r['pos_med_mm']:>11} {r['rot_med_rad']:>12} "
            f"{str(r['success_n']) + '/' + str(r['success_total']):>9} "
            f"{str(r['coll_free_n']):>9}"
        )

    # -- append CSV rows ------------------------------------------------------
    csv_path = args.outdir / CSV_NAME
    args.outdir.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"\nappended {len(rows)} cuRobo rows to {csv_path}")


if __name__ == "__main__":
    main()
