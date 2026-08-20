"""Benchmark and correctness evaluation of IK solvers: pyroffi (this work) vs baselines.

Methods
    pyroffi (this work)
        HJCD-IK, LS-IK, SQP-IK, MPPI-IK, Analytic-IK — each with a JAX (differentiable,
        GPU) backend and a CUDA/FFI kernel backend.  Learned-IK (JAX, Flax MLP warm-start
        + LM refinement).
    Baselines
        cuRobo IK (NV LAb LBM-optimized IK, its own conda env, see below).
        PyRoKi-LS and PyRoKi-AnalyticJac (jaxls LM; "AnalyticJac" uses the analytic
        task-space Jacobian in the residual, not a closed-form solver).
        QuIK-CPU (Halley's-method CPU IK).

Sequential (per-problem) timing
    Every solver is evaluated one pose at a time to measure single-problem
    latency.  JAX/CUDA pyroffi solvers are timed with JIT device timers inside a
    fixed-count lax.scan (no Python dispatch in the timed loop); cuRobo is timed
    with CUDA events; PyRoKi and QuIK with wall clocks.

Batch timing
    Batch solvers are timed over N_TARGETS_BATCH targets at once to measure
    throughput.  The effective per-problem time is total_time / N_TARGETS_BATCH.

Correctness
    For each solver the median position / rotation errors across all target
    poses are reported, and success is counted against a single common
    threshold (POS_THR_M = 1 mm, ROT_THR_RAD = 0.05 rad) for every method.
    (cuRobo's solver internally targets its native 5 mm tolerance; the success
    column is nevertheless scored with the same 1 mm / 0.05 rad metric.)

Collision-free IK
    When COLLISION_FREE=True a static obstacle scene is created (or loaded from
    ENV_FILE) and each differentiable solver is re-run with the same soft
    collision penalty in its objective.  Results include a coll_free column
    showing how many solutions are actually collision-free (min signed distance
    > 0).  cuRobo uses its own world-collision-aware IK; its rows are scored
    with the same success thresholds and its coll_free_n comes from its own
    self/world-collision check.  Analytic-IK is excluded from the collision
    rows: its CUDA collision handling is candidate *selection* over a closed-form
    branch set, not the differentiable penalty that all other collision rows use,
    so including it would not be an apples-to-apples comparison.

    The scene is saved to ENV_FILE as JSON with a ``curobo_world_model`` key
    that can be fed directly into a cuRobo WorldConfig for fair comparison.

Learned-IK
    The Learned-IK solver requires a pre-trained Flax model.  Train one with:
        python train_learned_ik.py --robot panda
    The model is saved to resources/learned_ik/panda.pkl and loaded
    automatically.  If no model is found the learned-IK rows are skipped.

Usage:
    python tests/bench_ik.py

    Every solver is benchmarked in its OWN subprocess (one per robot x solver), so
    no solver's JAX preallocation, GLASS-tier cache, JIT/kernel compilation or
    allocator state can perturb another's timings. The top-level process is a thin
    dispatcher that never touches the GPU; the ``--robot``/``--solver`` flags mark
    the isolated child invocations and are not meant to be passed by hand.

    Fairness / reproducibility notes:
      * All target poses are generated from a fixed seed-0 RNG in the pyroffi env
        and every method is given the exact same targets.  For cuRobo the targets
        are dumped to an .npz sidecar and its child (which runs in a separate
        conda env) consumes that file, so nothing is re-sampled or re-derived.
      * All pyroffi and PyRoKi solvers use num_seeds = 32 random restarts.
        cuRobo is also given num_seeds = 32 (its own benchmark ships with 2/8);
        the seed budget is therefore documented per method.
      * Analytic-IK children run with JAX_ENABLE_X64=1 (set pre-import) because
        the closed-form solve and its CUDA FFI require float64; all other
        children run in the default (float32) dtype.
      * cuRobo runs in its own conda env (``curobo``, editable install of
        baselines/curobo).  The dispatcher locates the interpreter via the
        CUROBO_PYTHON env var, a sibling "curobo" conda env, or
        ``conda run -n curobo``.  cuRobo only ships robot configs for panda
        (franka.yml) and g1 (unitree_g1.yml); fetch/baxter cuRobo rows are
        skipped with a logged note.  The per-robot tool frame is narrowed
        in-memory before the solver is built (panda → panda_hand, g1 →
        right_hand_palm_link, which is present in unitree_g1.yml's kinematics),
        so cuRobo goals and the error comparison happen in the same frame as
        the pyroffi targets — no static offset is needed.
      * GPU monitoring (NVML) respects CUDA_VISIBLE_DEVICES: with
        CUDA_VISIBLE_DEVICES=k the monitor watches physical GPU k, so the
        reported util/VRAM rows always belong to the GPU the child actually ran
        on.
      * MPPI's L-BFGS refinement budget is identical (25 iterations) on the JAX
        and CUDA backends.

Prerequisites:
    1. A CUDA-capable GPU.
    2. CUDA libraries compiled:
           bash build_kernels/build_hjcd_ik_cuda.sh
           bash build_kernels/build_ls_ik_cuda.sh
           bash build_kernels/build_sqp_ik_cuda.sh
           bash build_kernels/build_mppi_ik_cuda.sh
    3. robot_descriptions installed:
           pip install robot_descriptions
    4. (Optional) Flax model for Learned-IK:
           pip install flax optax
           python train_learned_ik.py --robot panda
    5. (Optional) cuRobo baseline: a conda env ``curobo`` with an editable
       install of baselines/curobo (or CUROBO_PYTHON pointing at its python).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime
import functools
import json
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

# JAX reads platform selection during import/initialization.  Parse this flag
# before importing jax so ``--cpu-only`` can prevent GPU backend setup.
_CPU_ONLY = "--cpu-only" in sys.argv[1:]
if _CPU_ONLY:
    os.environ["JAX_PLATFORMS"] = "cpu"

# ``--no-jax`` skips all JAX-based solvers (HJCD/LS/SQP/MPPI-JAX, Learned-JAX,
# PyRoKi) but keeps CUDA/FFI kernel solvers and GPU monitoring enabled, so the
# FFI kernels can be benchmarked in isolation.  ``--cpu-only`` implies it.
_NO_JAX = _CPU_ONLY or "--no-jax" in sys.argv[1:]

# Isolation model (see main() / _run_solver_subprocess): each solver is
# benchmarked in its OWN subprocess, invoked with ``--solver LABEL``. A process
# WITHOUT ``--solver`` is the dispatcher: it only spawns per-solver children and
# never runs a solver itself, so it must not preallocate JAX's default 75% VRAM
# chunk — that chunk would then be unavailable to every child and is exactly the
# parent/child double-allocation that used to OOM the CUDA kernels. Children run
# one at a time and each get the full card with the default (preallocating)
# allocator, so their timings are not perturbed by any co-resident solver.
_IS_SOLVER_CHILD = "--solver" in sys.argv[1:]
if not _CPU_ONLY and not _IS_SOLVER_CHILD:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Analytic-IK (closed-form) children run in float64: the 7-DOF branch solve and
# its CUDA FFI kernel both require x64, while every other child keeps the
# default dtype so JAX/CUDA agreement is measured under the same precision.
# JAX reads JAX_ENABLE_X64 at import, so this must land before ``import jax``.
if "--solver" in sys.argv[1:]:
    _solver_idx = sys.argv.index("--solver")
    if _solver_idx + 1 < len(sys.argv) and sys.argv[_solver_idx + 1] in (
        "Analytic-JAX", "Analytic-CUDA",
    ):
        os.environ["JAX_ENABLE_X64"] = "1"

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroffi as pk
import yourdfpy

from pyroffi.collision import Box, RobotCollisionSpherized, Sphere, collide
from pyroffi._robot_srdf_parser import read_disabled_collisions_from_srdf

# Optional NVML for GPU monitoring (nvidia-ml-py / pynvml).
# NVML indices are PHYSICAL, so the handle must track CUDA_VISIBLE_DEVICES —
# otherwise a child pinned to GPU k would report GPU 0's util/VRAM.
try:
    if _CPU_ONLY:
        raise RuntimeError("CPU-only mode")
    import pynvml as _pynvml
    _pynvml.nvmlInit()
    _cve = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    _cve_idx = int(_cve.split(",")[0].strip()) if _cve.strip() else 0
    _NVML_HANDLE: object | None = _pynvml.nvmlDeviceGetHandleByIndex(_cve_idx)
    _NVML_OK = True
except Exception:
    _NVML_HANDLE = None
    _NVML_OK = False


@contextlib.contextmanager
def _gpu_monitor(interval_s: float = 0.02):
    """Sample GPU utilisation and VRAM in a background thread.

    Yields a dict; on exit the dict contains:
        ``gpu_util``  – list of utilisation % samples
        ``vram_mb``   – list of used VRAM (MiB) samples
    """
    samples: dict[str, list[float]] = {"gpu_util": [], "vram_mb": []}
    stop_evt = threading.Event()

    def _sample() -> None:
        while not stop_evt.is_set():
            if _NVML_OK and _NVML_HANDLE is not None:
                util = _pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE)
                mem  = _pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE)
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

from pyroffi.optimization_engines._hjcd_ik import hjcd_solve
from pyroffi.optimization_engines._ls_ik import ls_ik_solve
from pyroffi.optimization_engines._sqp_ik import sqp_ik_solve
from pyroffi.optimization_engines._mppi_ik import mppi_ik_solve

if not _CPU_ONLY:
    from pyroffi.optimization_engines._hjcd_ik import (
        hjcd_solve_cuda,
        hjcd_solve_cuda_batch,
    )
    from pyroffi.optimization_engines._ls_ik import (
        ls_ik_solve_cuda,
        ls_ik_solve_cuda_batch,
    )
    from pyroffi.optimization_engines._sqp_ik import (
        sqp_ik_solve_cuda,
        sqp_ik_solve_cuda_batch,
    )
    from pyroffi.optimization_engines._mppi_ik import (
        mppi_ik_solve_cuda,
        mppi_ik_solve_cuda_batch,
    )

# Analytic (closed-form) IK for the 7-DOF spherical-wrist family.  The JAX
# backend is pure JAX (available in CPU-only mode too); the CUDA/FFI backend
# needs a GPU and float64 (see the pre-import JAX_ENABLE_X64 flag above).
from pyroffi.kinematics._analytic_ik import (
    analytic_ik_solve,
    analytic_ik_solve_batched,
    build_geometry,
)

if not _CPU_ONLY:
    from pyroffi.optimization_engines._analytic_ik import (
        analytic_ik_solve_cuda,
        analytic_ik_solve_cuda_batch,
    )

# QuIK CPU (Halley's-method) IK backend (optional; needs cricket JIT + a
# DH-representable serial chain).  It always runs on the CPU, so it is timed
# here as the CPU alternative to the CUDA solvers.
try:
    from pyroffi.optimization_engines._quik_ik import QuIKSolver
    _QUIK_IMPORT_OK = True
except Exception:
    _QUIK_IMPORT_OK = False

# VAMP CPU collision checker + MPPI collision-free projection kernel (optional;
# needs cricket JIT).  Used to give QuIK a collision-aware mode: seeds are
# projected onto the collision-free manifold and solutions collision-filtered.
try:
    from pyroffi.collision import VAMPCPUCollisionChecker
    _VAMP_CPU_IMPORT_OK = True
except Exception:
    _VAMP_CPU_IMPORT_OK = False

# Learned-IK: imports only; model is loaded inside main() after the robot is known.
try:
    from pyroffi.optimization_engines._learned_ik import (
        get_default_model_path,
        load_learned_ik,
        make_learned_ik_solve,
    )
    _LEARNED_IK_IMPORT_OK = True
except Exception:
    _LEARNED_IK_IMPORT_OK = False

# PyRoKi IK solver (optional).
try:
    import pyroki as _pyroki
    import jaxls as _jaxls
    _PYROKI_AVAILABLE = True
except Exception:
    _PYROKI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROBOT_NAMES = ("panda", "fetch", "baxter", "g1")

# GLASS parallelism tier per robot, passed to the CUDA IK kernels via the
# PYROFFI_IK_TIER env var (see src/pyroffi/cuda_kernels/_tier_kernel.cuh).
# The tier is cached process-globally on first CUDA kernel launch; since every
# solver already runs in its own subprocess (see main()), the tier for a robot's
# CUDA-solver children is set from this table. g1 is omitted: at 43 DOF it
# exceeds TIER_CHOICE_MAX_N (32) and PYROFFI_TIER_DISPATCH forces Tier::Block
# regardless of this env var.
ROBOT_TIER = {
    "panda":  "thread",
    "fetch":  "warp",
    "baxter": "block",
}

# The four core IK methods, each with a JAX and a CUDA backend. Everything in the
# benchmark keys off these base solver labels (see _candidate_solvers / --solver).
_CORE_METHODS = ("HJCD", "LS", "SQP", "MPPI")

# Robots whose 7-DOF arm chain supports the closed-form (Analytic-IK) solve.
# build_geometry() validates the family at runtime; this set keeps the dispatcher
# from spawning children that would immediately skip.
_ANALYTIC_ROBOTS = frozenset(("panda", "fetch", "baxter"))

# cuRobo ships robot configs only for these two; the value is cuRobo's robot file
# name.  The cuRobo child narrows each config's tool_frames to pyroffi's EE link
# in-memory (panda_hand; right_hand_palm_link for g1, which is present in
# unitree_g1.yml's kinematics), so no static offset is needed.  fetch/baxter
# cuRobo rows are skipped with a logged note (see main()).
_CUROBO_ROBOT_FILES = {"panda": "franka.yml", "g1": "unitree_g1.yml"}


def _candidate_solvers(
    cpu_only: bool, no_jax: bool, robot_name: str | None = None,
) -> list[str]:
    """Base solver labels to benchmark, one subprocess each (see main()).

    This is the STATIC candidate set implied by the mode flags; the optional
    solvers (Learned/PyRoKi/QuIK/cuRobo) may still be unavailable at runtime, in
    which case that solver's child finds nothing to run and exits without writing
    rows.  When robot_name is given the set is further restricted to methods that
    support that robot: Analytic-IK only for 7-DOF spherical-wrist arms, cuRobo
    only where it ships a robot config.
    """
    labels: list[str] = []
    for m in _CORE_METHODS:
        if not no_jax:
            labels.append(f"{m}-JAX")
        if not cpu_only:
            labels.append(f"{m}-CUDA")
    _analytic_ok = robot_name is None or robot_name in _ANALYTIC_ROBOTS
    if not no_jax and _analytic_ok:
        labels.append("Analytic-JAX")
    if not cpu_only and _analytic_ok:
        labels.append("Analytic-CUDA")
    if not no_jax:
        labels += ["Learned-JAX", "PyRoKi-LS", "PyRoKi-AnalyticJac"]
    labels.append("QuIK-CPU")
    if robot_name is not None and robot_name in _CUROBO_ROBOT_FILES:
        labels.append("cuRobo")
    return labels


RESOURCE_ROOT = pathlib.Path(__file__).resolve().parent.parent / "resources"
ROBOT_URDFS = {
    "panda": RESOURCE_ROOT / "panda" / "panda_spherized.urdf",
    "fetch": RESOURCE_ROOT / "fetch" / "fetch_spherized.urdf",
    "baxter": RESOURCE_ROOT / "baxter" / "baxter_spherized.urdf",
    "g1": RESOURCE_ROOT / "g1_description" / "g1_29dof_with_hand_rev_1_0_spherized.urdf",
}

ROBOT_SRDFS = {
    "panda": RESOURCE_ROOT / "panda" / "panda.srdf",
    "fetch": RESOURCE_ROOT / "fetch" / "fetch.srdf",
    "baxter": RESOURCE_ROOT / "baxter" / "baxter.srdf",
    "g1": RESOURCE_ROOT / "g1_description" / "g1_29dof.srdf",
}

# Candidate EE links per robot. The first existing link in the loaded URDF is used.
ROBOT_TARGET_LINK_CANDIDATES = {
    "panda": ("panda_hand",),
    "fetch": ("gripper_link",),
    "baxter": ("right_hand",),
    "g1": ("right_hand_palm_link", "left_hand_palm_link"),
}

# Joints to keep fixed during IK (Panda finger joints only).
ROBOT_FIXED_JOINT_NAMES = {
    "panda": ("panda_finger_joint1", "panda_finger_joint2"),
    "fetch": (),
    "baxter": (),
    "g1": (),
}

N_TARGETS = 32    # number of random target poses to evaluate
N_TARGETS_BATCH = 256
N_WARMUP  = 3      # JIT / kernel warm-up calls (discarded from timing)
N_TIMED   = 5      # timed repetitions (sequential: per pose; batch: per full call)
N_DEVICE_REPEATS = 5  # repeats inside lax.scan per timed call (amortises dispatch overhead)

# HJCD-IK hyper-parameters.
IK_KWARGS_HJCD_JAX = dict(
    num_seeds          = 32,
    coarse_max_iter    = 20,
    lm_max_iter        = 40,
    lambda_init        = 1e-3,
    continuity_weight  = 0.0,
    limit_prior_weight = 1e-4,
    kick_scale         = 0.02,
)
IK_KWARGS_HJCD_CUDA = dict(**IK_KWARGS_HJCD_JAX)

# LS-IK hyper-parameters.
IK_KWARGS_LS_JAX = dict(
    num_seeds         = 32,
    max_iter          = 60,
    pos_weight        = 50.0,
    ori_weight        = 10.0,
    lambda_init       = 5e-3,
    continuity_weight = 0.0,
)
IK_KWARGS_LS_CUDA = dict(
    **IK_KWARGS_LS_JAX,
    eps_pos = 1e-8,
    eps_ori = 1e-8,
)

# SQP-IK hyper-parameters.
IK_KWARGS_SQP_JAX = dict(
    num_seeds         = 32,
    max_iter          = 60,
    n_inner_iters     = 2,
    pos_weight        = 50.0,
    ori_weight        = 10.0,
    lambda_init       = 5e-3,
    continuity_weight = 0.0,
)
IK_KWARGS_SQP_CUDA = dict(
    **IK_KWARGS_SQP_JAX,
    eps_pos = 1e-8,
    eps_ori = 1e-8,
)

# MPPI-IK hyper-parameters.
IK_KWARGS_MPPI_JAX = dict(
    num_seeds         = 32,
    n_particles       = 16,
    n_mppi_iters      = 5,
    n_lbfgs_iters     = 25,   # == CUDA backend (fairness: identical refinement budget)
    m_lbfgs           = 5,
    pos_weight        = 50.0,
    ori_weight        = 10.0,
    sigma             = 0.3,
    mppi_temperature  = 0.05,
    continuity_weight = 0.0,
)
IK_KWARGS_MPPI_CUDA = dict(
    num_seeds         = 32,
    n_particles       = 16,
    n_mppi_iters      = 5,
    n_lbfgs_iters     = 25,
    m_lbfgs           = 5,
    pos_weight        = 50.0,
    ori_weight        = 10.0,
    sigma             = 0.3,
    mppi_temperature  = 0.05,
    eps_pos           = 1e-8,
    eps_ori           = 1e-8,
    continuity_weight = 0.0,
)

# Learned-IK hyper-parameters.
# num_seeds:      half come from the MLP prediction ± noise; half are random.
# n_refine_iters: LM steps run on each seed after the MLP warm-start.
IK_KWARGS_LEARNED_JAX = dict(
    num_seeds         = 64,
    n_refine_iters    = 15,
    pos_weight        = 50.0,
    ori_weight        = 10.0,
    lambda_init       = 5e-3,
    continuity_weight = 0.0,
)

# PyRoKi hyper-parameters.
# num_seeds: random restarts vmapped in parallel (same as pyroffi solvers).
IK_KWARGS_PYROKI = dict(
    num_seeds    = 32,
    pos_weight   = 50.0,
    ori_weight   = 10.0,
    max_iter     = 100,
)

# Analytic (closed-form) IK: the 7-DOF branch solve is deterministic, so the only
# tuning knob is the q7 sample grid (JAX) / seed budget (CUDA); both use 32 to
# match every other method's num_seeds.
IK_KWARGS_ANALYTIC_JAX = dict(
    num_q7 = 32,
)
IK_KWARGS_ANALYTIC_CUDA = dict(
    num_seeds = 32,
)

# Success threshold.
POS_THR_M   = 1e-3
ROT_THR_RAD = 0.05

# ---------------------------------------------------------------------------
# Collision-free IK configuration
# ---------------------------------------------------------------------------

# Set to False to skip the collision-free IK section entirely.
COLLISION_FREE = True

# Where to persist the obstacle scene.  Reuse this file in curobo / pyroki
# benchmarks by loading the ``curobo_world_model`` key.
ENV_FILE = pathlib.Path(__file__).resolve().parent.parent / "resources" / "bench_env_large.json"

# CSV output path.  Results are appended (not overwritten) so multiple runs
# accumulate.  Set to None to disable CSV output.
CSV_FILE = pathlib.Path(__file__).resolve().parent.parent / "resources" / "bench_ik_results.csv"

# Smoothing radius [m] for the soft collision penalty (softplus approximation).
_COLL_EPS = 0.005

# Penalty weight applied to the collision cost inside the IK objective.
COLL_WEIGHT = 1e8

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SolveResult:
    cfg:     np.ndarray
    pos_err: float
    rot_err: float
    time_ms: float   # per-problem time in ms


@dataclass
class BatchResult:
    cfgs:          np.ndarray   # (N_TARGETS, n_act)
    pos_errs:      np.ndarray   # (N_TARGETS,) metres
    rot_errs:      np.ndarray   # (N_TARGETS,) radians
    time_ms:       float        # effective per-problem time (total / N_TARGETS)
    peak_gpu_util: float = float("nan")   # peak GPU utilisation (%)
    avg_gpu_util:  float = float("nan")   # mean GPU utilisation (%)
    peak_vram_mb:  float = float("nan")   # peak VRAM used (MiB)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pose_errors(
    robot: pk.Robot,
    cfg: jax.Array,
    target_link_index: int,
    target_pose: jaxlie.SE3,
) -> tuple[float, float]:
    Ts     = robot.forward_kinematics(cfg)
    actual = jaxlie.SE3(Ts[target_link_index])
    pos_err = float(jnp.linalg.norm(actual.translation() - target_pose.translation()))
    rot_err = float(jnp.linalg.norm(
        (target_pose.rotation().inverse() @ actual.rotation()).log()
    ))
    return pos_err, rot_err



def _run_solver_sequential(
    fn, robot, target_link_index, target_poses, fixed_joint_mask,
    rng_keys, previous_cfgs, kwargs, n_act, *, timer=None,
) -> list[SolveResult]:
    """Run a single-problem solver sequentially over all target poses.

    Timing uses a ``lax.scan``-compiled loop over ``N_DEVICE_REPEATS`` repeats
    to amortise Python / kernel-dispatch overhead; correctness uses a single
    un-scanned call to recover the actual solved configuration.

    If *timer* is provided it must already be JIT-compiled (i.e. warmed up
    during the benchmark's warmup phase).  When omitted the timer is built and
    warmed up on the first pose.
    """
    tli = (target_link_index,)

    _timer = timer
    _need_compile = _timer is None
    if _need_compile:
        _timer = _build_seq_ik_timer(fn, robot, tli, fixed_joint_mask, n_act, kwargs)

    results = []
    for i, target_pose in enumerate(target_poses):
        rng_key_i  = rng_keys[i]
        prev_cfg_i = previous_cfgs[i]

        # Single un-scanned call to get the solved cfg and compute errors.
        cfg = fn(
            robot,
            target_link_indices=tli,
            target_poses=(target_pose,),
            rng_key=rng_key_i,
            previous_cfg=prev_cfg_i,
            fixed_joint_mask=fixed_joint_mask,
            **kwargs,
        )
        jax.block_until_ready(cfg)
        pos_err, rot_err = _pose_errors(robot, cfg, target_link_index, target_pose)

        rng_keys_seq = jnp.stack(
            [jax.random.fold_in(rng_key_i, k) for k in range(N_DEVICE_REPEATS)]
        )
        if _need_compile:
            # Warm up the newly-built timer (only needed once; shapes are fixed).
            jax.block_until_ready(_timer(target_pose.wxyz_xyz, prev_cfg_i, rng_keys_seq))
            _need_compile = False
        t = _time_scan(_timer, target_pose.wxyz_xyz, prev_cfg_i, rng_keys_seq)

        results.append(SolveResult(np.array(cfg), pos_err, rot_err, t * 1e3))
    return results


def _run_solver_batch(
    fn, robot, target_link_index, target_poses_stacked, fixed_joint_mask,
    rng_key, previous_cfgs, kwargs, is_jax_batch: bool = False, *, timer=None,
) -> BatchResult:
    """Run a batch solver, timing with a ``lax.scan``-compiled loop.

    Correctness uses a single un-scanned call; timing uses ``N_DEVICE_REPEATS``
    scanned repeats per host dispatch to amortise Python / kernel-launch overhead.

    If *timer* is provided it must already be JIT-compiled (warmed up during the
    benchmark's warmup phase).  When omitted the timer is built and warmed up here.
    """
    tli = (target_link_index,)
    n_targets = len(target_poses_stacked.wxyz_xyz)
    target_poses_wxyz = target_poses_stacked.wxyz_xyz

    # Single un-scanned call for correctness / error evaluation.
    if is_jax_batch:
        cfgs_out = fn(robot, tli, target_poses_stacked, rng_key, previous_cfgs, fixed_joint_mask)
    else:
        cfgs_out = fn(robot, tli, target_poses_stacked, rng_key, previous_cfgs,
                      fixed_joint_mask=fixed_joint_mask, **kwargs)
    jax.block_until_ready(cfgs_out)
    cfgs_np = np.array(cfgs_out)  # (N_TARGETS, n_act)

    pos_errs = np.empty(n_targets)
    rot_errs = np.empty(n_targets)
    for i in range(n_targets):
        target_pose = jaxlie.SE3(target_poses_stacked.wxyz_xyz[i])
        pos_errs[i], rot_errs[i] = _pose_errors(
            robot, jnp.array(cfgs_np[i]), target_link_index, target_pose
        )

    _batch_timer = timer
    if _batch_timer is None:
        _batch_timer = _build_batch_ik_timer(fn, robot, tli, fixed_joint_mask, kwargs, is_jax_batch)

    if is_jax_batch:
        # rng_keys_seq: (N_DEVICE_REPEATS, N_TARGETS, 2)
        # Build repeats by folding repeat-id into each target's base key so
        # caller-supplied keys are respected.
        rng_keys_seq = _make_batched_rng_keys_seq(rng_key)
    else:
        # rng_keys_seq: (N_DEVICE_REPEATS, 2)
        rng_keys_seq = jnp.stack(
            [jax.random.fold_in(rng_key, k) for k in range(N_DEVICE_REPEATS)]
        )

    if timer is None:
        jax.block_until_ready(_batch_timer(target_poses_wxyz, previous_cfgs, rng_keys_seq))

    with _gpu_monitor() as gpu_samples:
        total_t = _time_scan(_batch_timer, target_poses_wxyz, previous_cfgs, rng_keys_seq)

    peak_gpu  = max(gpu_samples["gpu_util"],  default=float("nan"))
    avg_gpu   = float(np.mean(gpu_samples["gpu_util"])) if gpu_samples["gpu_util"] else float("nan")
    peak_vram = max(gpu_samples["vram_mb"],   default=float("nan"))

    effective_ms = total_t * 1e3 / n_targets
    return BatchResult(cfgs_np, pos_errs, rot_errs, effective_ms,
                       peak_gpu_util=peak_gpu, avg_gpu_util=avg_gpu, peak_vram_mb=peak_vram)


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

_COL_W = 20  # method name column width
_NUM_W = 10  # numeric column width

def _table_header(cols: list[str]) -> str:
    row = f"  {'Method':<{_COL_W}}"
    for c in cols:
        row += f"  {c:>{_NUM_W}}"
    return row

def _table_sep(n_cols: int) -> str:
    return "  " + "-" * (_COL_W + n_cols * (_NUM_W + 2))

def _table_row(label: str, vals: list[str]) -> str:
    row = f"  {label:<{_COL_W}}"
    for v in vals:
        row += f"  {v:>{_NUM_W}}"
    return row


def _seq_row(label: str, results: list[SolveResult]) -> tuple[str, dict]:
    pos    = np.array([r.pos_err * 1e3 for r in results])
    rot    = np.array([r.rot_err       for r in results])
    t      = np.array([r.time_ms       for r in results])
    solved = sum(r.pos_err < POS_THR_M and r.rot_err < ROT_THR_RAD for r in results)
    n      = len(results)
    vals = [
        f"{np.median(t):.3f}",
        f"{np.percentile(t, 95):.3f}",
        f"{np.median(pos):.4f}",
        f"{np.percentile(pos, 95):.4f}",
        f"{np.median(rot):.4f}",
        f"{np.percentile(rot, 95):.4f}",
        f"{solved}/{n}",
    ]
    return _table_row(label, vals), {"t_med": float(np.median(t))}


def _seq_row_coll(
    label: str, results: list[SolveResult], coll_free: int,
) -> tuple[str, dict]:
    """Like _seq_row but with an extra coll_free column."""
    pos    = np.array([r.pos_err * 1e3 for r in results])
    rot    = np.array([r.rot_err       for r in results])
    t      = np.array([r.time_ms       for r in results])
    solved = sum(r.pos_err < POS_THR_M and r.rot_err < ROT_THR_RAD for r in results)
    n      = len(results)
    vals = [
        f"{np.median(t):.3f}",
        f"{np.percentile(t, 95):.3f}",
        f"{np.median(pos):.4f}",
        f"{np.percentile(pos, 95):.4f}",
        f"{np.median(rot):.4f}",
        f"{np.percentile(rot, 95):.4f}",
        f"{solved}/{n}",
        f"{coll_free}/{n}",
    ]
    return _table_row(label, vals), {"t_med": float(np.median(t))}


def _batch_row(label: str, result: BatchResult) -> tuple[str, dict]:
    pos    = result.pos_errs * 1e3
    rot    = result.rot_errs
    solved = int(np.sum((result.pos_errs < POS_THR_M) & (result.rot_errs < ROT_THR_RAD)))
    n      = len(pos)

    def _fmt_pct(v: float) -> str:
        return f"{v:.0f}%" if not np.isnan(v) else "n/a"

    def _fmt_mb(v: float) -> str:
        return f"{v:.0f}" if not np.isnan(v) else "n/a"

    vals = [
        f"{result.time_ms:.3f}",
        f"{np.median(pos):.4f}",
        f"{np.percentile(pos, 95):.4f}",
        f"{np.median(rot):.4f}",
        f"{np.percentile(rot, 95):.4f}",
        f"{solved}/{n}",
        _fmt_pct(result.peak_gpu_util),
        _fmt_pct(result.avg_gpu_util),
        _fmt_mb(result.peak_vram_mb),
    ]
    return _table_row(label, vals), {}


def _batch_row_coll(
    label: str, result: BatchResult, coll_free: int,
) -> tuple[str, dict]:
    """Like _batch_row but with an extra coll_free column."""
    pos    = result.pos_errs * 1e3
    rot    = result.rot_errs
    solved = int(np.sum((result.pos_errs < POS_THR_M) & (result.rot_errs < ROT_THR_RAD)))
    n      = len(pos)

    def _fmt_pct(v: float) -> str:
        return f"{v:.0f}%" if not np.isnan(v) else "n/a"

    def _fmt_mb(v: float) -> str:
        return f"{v:.0f}" if not np.isnan(v) else "n/a"

    vals = [
        f"{result.time_ms:.3f}",
        f"{np.median(pos):.4f}",
        f"{np.percentile(pos, 95):.4f}",
        f"{np.median(rot):.4f}",
        f"{np.percentile(rot, 95):.4f}",
        f"{solved}/{n}",
        f"{coll_free}/{n}",
        _fmt_pct(result.peak_gpu_util),
        _fmt_pct(result.avg_gpu_util),
        _fmt_mb(result.peak_vram_mb),
    ]
    return _table_row(label, vals), {}


def _make_batched_jax_solver(base_fn, ik_kwargs):
    """Create a JITted batched JAX solver (vmap over targets)."""
    def _solve_batch(
        robot, target_link_indices, target_poses, rng_keys, previous_cfgs, fixed_joint_mask
    ):
        def _single(target_pose, rng_key, previous_cfg):
            return base_fn(
                robot=robot,
                target_link_indices=target_link_indices,
                target_poses=(target_pose,),
                rng_key=rng_key,
                previous_cfg=previous_cfg,
                fixed_joint_mask=fixed_joint_mask,
                **ik_kwargs,
            )
        return jax.vmap(_single, in_axes=(0, 0, 0))(target_poses, rng_keys, previous_cfgs)

    return jax.jit(_solve_batch, static_argnames=("target_link_indices",))


def _build_seq_ik_timer(fn, robot, tli, fixed_joint_mask, n_act, kwargs):
    """Return a JITted fn that runs sequential IK ``N_DEVICE_REPEATS`` times via lax.scan.

    Signature of the returned timer::

        timer(target_pose_wxyz_xyz, prev_cfg, rng_keys_seq) -> checksum

    where ``rng_keys_seq`` has shape ``(N_DEVICE_REPEATS, 2)``.
    """
    @jax.jit
    def _timer(target_pose_wxyz_xyz, prev_cfg, rng_keys_seq):
        target_pose = jaxlie.SE3(target_pose_wxyz_xyz)

        def body(carry, rng_key):
            out = fn(
                robot,
                target_link_indices=tli,
                target_poses=(target_pose,),
                rng_key=rng_key,
                previous_cfg=prev_cfg,
                fixed_joint_mask=fixed_joint_mask,
                **kwargs,
            )
            return carry + out.astype(jnp.float32), None

        checksum, _ = jax.lax.scan(
            body, jnp.zeros(n_act, dtype=jnp.float32), rng_keys_seq
        )
        return checksum

    return _timer


def _build_batch_ik_timer(fn, robot, tli, fixed_joint_mask, kwargs, is_jax_batch: bool):
    """Return a JITted timer for batch IK via lax.scan.

    For JAX batch solvers the signature is::

        fn(robot, tli, target_poses, rng_keys, prev_cfgs, fixed_joint_mask)

    and ``rng_keys_seq`` has shape ``(N_DEVICE_REPEATS, N, 2)``.

    For CUDA batch solvers the signature is::

        fn(robot, tli, target_poses, rng_key, prev_cfgs, fixed_joint_mask=…, **kwargs)

    and ``rng_keys_seq`` has shape ``(N_DEVICE_REPEATS, 2)``.
    """
    if is_jax_batch:
        @jax.jit
        def _timer(target_poses_wxyz_xyz, prev_cfgs, rng_keys_seq):
            target_poses = jaxlie.SE3(target_poses_wxyz_xyz)

            def body(carry, rng_keys_i):
                out = fn(robot, tli, target_poses, rng_keys_i, prev_cfgs, fixed_joint_mask)
                return carry + jnp.sum(out).astype(jnp.float32), None

            checksum, _ = jax.lax.scan(body, jnp.float32(0.0), rng_keys_seq)
            return checksum
    else:
        @jax.jit
        def _timer(target_poses_wxyz_xyz, prev_cfgs, rng_keys_seq):
            target_poses = jaxlie.SE3(target_poses_wxyz_xyz)

            def body(carry, rng_key):
                out = fn(
                    robot,
                    tli,
                    target_poses,
                    rng_key,
                    prev_cfgs,
                    fixed_joint_mask=fixed_joint_mask,
                    **kwargs,
                )
                return carry + jnp.sum(out).astype(jnp.float32), None

            checksum, _ = jax.lax.scan(body, jnp.float32(0.0), rng_keys_seq)
            return checksum

    return _timer


def _time_scan(timer_fn, *args, n: int = N_TIMED) -> float:
    """Run a scan-based timer *n* times and return median per-repeat wall-clock time (s)."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = timer_fn(*args)
        jax.block_until_ready(out)
        times.append((time.perf_counter() - t0) / N_DEVICE_REPEATS)
    return float(np.median(times))


def _make_batched_rng_keys_seq(base_rng_keys: jax.Array) -> jax.Array:
    """Expand per-target RNG keys into a (repeat, target, 2) sequence."""
    repeat_ids = jnp.arange(N_DEVICE_REPEATS, dtype=jnp.uint32)

    def _fold_target_key(key):
        return jax.vmap(lambda rid: jax.random.fold_in(key, rid))(repeat_ids)

    # (N_TARGETS, N_DEVICE_REPEATS, 2) -> (N_DEVICE_REPEATS, N_TARGETS, 2)
    keys_n_r = jax.vmap(_fold_target_key)(base_rng_keys)
    return jnp.swapaxes(keys_n_r, 0, 1)


# ---------------------------------------------------------------------------
# Analytic (closed-form) IK adapters
# ---------------------------------------------------------------------------
# The closed-form solve covers only the 7-DOF arm chain (the FIRST 7 actuated
# joints of every robot in this benchmark).  Joints past the chain (gripper
# fingers) keep their previous value, and fixed-joint-masked joints are pinned
# to their previous value — exactly what the other solvers do through
# fixed_joint_mask.

def _embed_7dof(q7, prev, fixed_joint_mask):
    """Embed a 7-DOF closed-form solution into the full actuated space."""
    if q7.shape[-1] != prev.shape[-1]:
        full = jnp.concatenate([q7, prev[..., 7:]], axis=-1)
    else:
        full = q7
    return jnp.where(fixed_joint_mask == 1, prev, full)


def _make_analytic_seq_fn(geometry, target_link_name):
    """Map the bench's generic seq convention onto analytic_ik_solve (JAX).

    The solve is deterministic, so rng_key is accepted and ignored.
    """
    def fn(robot, target_link_indices, target_poses, rng_key, previous_cfg,
           fixed_joint_mask=None, **kwargs):
        tp = target_poses[0] if isinstance(target_poses, (tuple, list)) else target_poses
        q7, _found = analytic_ik_solve(
            robot, target_link_name, tp,
            num_q7=int(kwargs.get("num_q7", 32)),
            previous_cfg=previous_cfg,      # sliced [:7] internally
            geometry=geometry,
        )
        return _embed_7dof(q7, previous_cfg, fixed_joint_mask)
    return fn


def _make_analytic_jax_batch_fn(geometry, target_link_name):
    """Map the bench's generic JAX batch convention onto analytic_ik_solve_batched.

    ``rng_keys`` is accepted and ignored (deterministic solve); the batched JAX
    path takes target POSE MATRICES, hence ``.as_matrix()``.
    """
    def fn(robot, tli, target_poses, rng_keys, previous_cfgs,
           fixed_joint_mask=None, **kwargs):
        q7, _found = analytic_ik_solve_batched(
            robot, target_link_name, target_poses.as_matrix(),
            num_q7=int(kwargs.get("num_q7", 32)),
            previous_cfg=previous_cfgs,     # sliced :7 internally
            geometry=geometry,
            backend="jax",                  # force JAX for the Analytic-JAX row
        )
        return _embed_7dof(q7, previous_cfgs, fixed_joint_mask)
    return fn


def _make_analytic_cuda_seq_fn():
    """Generic seq convention onto analytic_ik_solve_cuda (CUDA FFI, x64)."""
    def fn(robot, tli, target_poses, rng_key, previous_cfg,
           fixed_joint_mask=None, **kwargs):
        tp = target_poses[0] if isinstance(target_poses, (tuple, list)) else target_poses
        # The FFI kernel reshapes previous_cfg to (1, 7): exactly 7 columns.
        prev7 = None if previous_cfg is None else jnp.asarray(previous_cfg)[:7]
        q7 = analytic_ik_solve_cuda(
            robot, tli, tp, rng_key,
            previous_cfg=prev7,
            num_seeds=int(kwargs.get("num_seeds", 32)),
        )
        return _embed_7dof(q7, previous_cfg, fixed_joint_mask)
    return fn


def _make_analytic_cuda_batch_fn():
    """Generic CUDA batch convention onto analytic_ik_solve_cuda_batch."""
    def fn(robot, tli, target_poses, rng_key, previous_cfgs,
           fixed_joint_mask=None, **kwargs):
        q7 = analytic_ik_solve_cuda_batch(
            robot, tli, target_poses, rng_key,
            previous_cfgs=jnp.asarray(previous_cfgs)[:, :7],  # FFI: 7 cols
            num_seeds=int(kwargs.get("num_seeds", 32)),
        )
        return _embed_7dof(q7, previous_cfgs, fixed_joint_mask)
    return fn


# ---------------------------------------------------------------------------
# PyRoKi helpers
# ---------------------------------------------------------------------------

def _make_pyroki_solvers(
    pyroki_robot,
    target_link_index: int,
    kwargs: dict,
    *,
    collision_cost_fn=None,
    collision_weight: float = 0.0,
    pose_cost_factory=None,
):
    """Build JIT'd single-problem and batch PyRoKi IK solvers.

    ``pose_cost_factory`` selects the pose-cost factory: pyroki's autodiff
    ``pose_cost`` (default, "PyRoKi-LS") or ``pose_cost_analytic_jac``
    ("PyRoKi-AnalyticJac" — analytic task-space Jacobian, same LM solver).

    Returns:
        solve_fn(target_wxyz_xyz, seed_cfgs) -> cfg  (single target)
        batch_fn(target_wxyz_xyz_batch, seed_cfgs_batch) -> cfgs  (N targets)
    """
    if pose_cost_factory is None:
        pose_cost_factory = _pyroki.costs.pose_cost
    joint_var = pyroki_robot.joint_var_cls(0)
    tli_arr   = jnp.array(target_link_index, dtype=jnp.int32)
    pos_w     = float(kwargs.get("pos_weight", 50.0))
    ori_w     = float(kwargs.get("ori_weight", 10.0))
    max_iter  = int(kwargs.get("max_iter", 100))

    termination = _jaxls.TerminationConfig(
        max_iterations=max_iter,
        cost_tolerance=1e-7,
        gradient_tolerance=1e-6,
        parameter_tolerance=1e-7,
    )

    _coll_cost_factory = None
    if collision_cost_fn is not None and collision_weight > 0.0:
        coll_scale = float(np.sqrt(collision_weight))

        @_jaxls.Cost.factory(name="pyroki_collision_penalty")
        def _coll_cost(values, q_var):
            cfg = values[q_var]
            penalty = collision_cost_fn(cfg)
            return jnp.array([coll_scale * penalty], dtype=jnp.float32)

        _coll_cost_factory = _coll_cost

    @jax.jit
    def _solve_fn(target_wxyz_xyz, seed_cfgs):
        """Solve IK for a single target pose with multiple random seeds."""
        target = jaxlie.SE3(target_wxyz_xyz)

        def _one_seed(seed_cfg):
            costs = [
                pose_cost_factory(pyroki_robot, joint_var, target, tli_arr, pos_w, ori_w),
                _pyroki.costs.limit_cost(pyroki_robot, joint_var),
            ]
            if _coll_cost_factory is not None:
                costs.append(_coll_cost_factory(joint_var))

            analyzed = _jaxls.LeastSquaresProblem(
                costs=costs,
                variables=[joint_var],
            ).analyze()
            init_vals = _jaxls.VarValues.make([joint_var.with_value(seed_cfg)])
            return analyzed.solve(init_vals, verbose=False, termination=termination)[joint_var]

        all_cfgs = jax.vmap(_one_seed)(seed_cfgs)

        # Pick best seed by position error.
        def _pos_err(cfg):
            Ts     = pyroki_robot.forward_kinematics(cfg)
            actual = jaxlie.SE3(Ts[target_link_index])
            return jnp.linalg.norm(actual.translation() - target.translation())

        errs     = jax.vmap(_pos_err)(all_cfgs)
        best_idx = jnp.argmin(errs)
        return all_cfgs[best_idx]

    @jax.jit
    def _batch_fn(target_wxyz_xyz_batch, seed_cfgs_batch):
        """Solve IK for a batch of target poses (vmapped over targets)."""
        return jax.vmap(_solve_fn)(target_wxyz_xyz_batch, seed_cfgs_batch)

    return _solve_fn, _batch_fn


def _run_pyroki_sequential(
    solve_fn,
    robot: "pk.Robot",
    pyroki_robot,
    target_link_index: int,
    target_poses: list,
    lo: np.ndarray,
    hi: np.ndarray,
    n_act: int,
    num_seeds: int,
) -> list[SolveResult]:
    """Run PyRoKi IK sequentially, timing with plain wall-clock (no lax.scan)."""
    lo_arr = jnp.array(lo, dtype=jnp.float32)
    hi_arr = jnp.array(hi, dtype=jnp.float32)

    results: list[SolveResult] = []
    for i, target_pose in enumerate(target_poses):
        key = jax.random.PRNGKey(i + 1)
        seed_cfgs = jax.random.uniform(key, (num_seeds, n_act), minval=lo_arr, maxval=hi_arr)

        # Single solve for correctness.
        cfg = solve_fn(target_pose.wxyz_xyz, seed_cfgs)
        jax.block_until_ready(cfg)
        pos_err, rot_err = _pose_errors(robot, cfg, target_link_index, target_pose)

        # Timed repetitions (new seeds each time to avoid caching artefacts).
        times = []
        for j in range(N_TIMED):
            key_j   = jax.random.fold_in(key, j)
            seeds_j = jax.random.uniform(key_j, (num_seeds, n_act), minval=lo_arr, maxval=hi_arr)
            t0  = time.perf_counter()
            out = solve_fn(target_pose.wxyz_xyz, seeds_j)
            jax.block_until_ready(out)
            times.append(time.perf_counter() - t0)

        results.append(SolveResult(np.array(cfg), pos_err, rot_err, float(np.median(times)) * 1e3))
    return results


def _run_pyroki_batch(
    batch_fn,
    robot: "pk.Robot",
    pyroki_robot,
    target_link_index: int,
    target_poses_stacked: jaxlie.SE3,
    lo: np.ndarray,
    hi: np.ndarray,
    n_act: int,
    num_seeds: int,
) -> BatchResult:
    """Run PyRoKi batch IK, timing with plain wall-clock (no lax.scan)."""
    lo_arr   = jnp.array(lo, dtype=jnp.float32)
    hi_arr   = jnp.array(hi, dtype=jnp.float32)
    n_targets = len(target_poses_stacked.wxyz_xyz)

    key = jax.random.PRNGKey(0)
    # seed_cfgs_batch: (n_targets, num_seeds, n_act)
    seed_cfgs_batch = jax.random.uniform(
        key, (n_targets, num_seeds, n_act), minval=lo_arr, maxval=hi_arr
    )

    # Single batch call for correctness.
    cfgs_out = batch_fn(target_poses_stacked.wxyz_xyz, seed_cfgs_batch)
    jax.block_until_ready(cfgs_out)
    cfgs_np = np.array(cfgs_out)

    pos_errs = np.empty(n_targets)
    rot_errs = np.empty(n_targets)
    for i in range(n_targets):
        target_pose = jaxlie.SE3(target_poses_stacked.wxyz_xyz[i])
        pos_errs[i], rot_errs[i] = _pose_errors(
            robot, jnp.array(cfgs_np[i]), target_link_index, target_pose
        )

    # Timed repetitions.
    times = []
    with _gpu_monitor() as gpu_samples:
        for j in range(N_TIMED):
            key_j   = jax.random.fold_in(key, j)
            seeds_j = jax.random.uniform(
                key_j, (n_targets, num_seeds, n_act), minval=lo_arr, maxval=hi_arr
            )
            t0  = time.perf_counter()
            out = batch_fn(target_poses_stacked.wxyz_xyz, seeds_j)
            jax.block_until_ready(out)
            times.append(time.perf_counter() - t0)

    peak_gpu  = max(gpu_samples["gpu_util"], default=float("nan"))
    avg_gpu   = float(np.mean(gpu_samples["gpu_util"])) if gpu_samples["gpu_util"] else float("nan")
    peak_vram = max(gpu_samples["vram_mb"],  default=float("nan"))

    effective_ms = float(np.median(times)) * 1e3 / n_targets
    return BatchResult(cfgs_np, pos_errs, rot_errs, effective_ms,
                       peak_gpu_util=peak_gpu, avg_gpu_util=avg_gpu, peak_vram_mb=peak_vram)


def _quik_seed_helpers(quik, lo, hi, num_seeds, vamp_checker):
    """Shared seed-generation / solution-selection for the QuIK bench runners.

    Returns ``(make_seeds, pick_best)``:

    ``make_seeds(rng, n_batch)`` draws ``[n_batch, num_seeds, dof]`` chain-space
    seeds; when ``vamp_checker`` is given they are first pushed onto the
    collision-free manifold with the MPPI projection kernel
    (:meth:`VAMPCPUCollisionChecker.project_collision_free`), scattered through
    the robot's full actuated vector (non-chain joints at the limit midpoint).

    ``pick_best(q_actuated, err)`` selects the lowest-error solution per
    problem; with a checker, the lowest-error *collision-free* solution
    (falling back to lowest error when none is free).
    """
    order = quik.model.actuated_order
    lo_c = np.where(np.isfinite(lo[order]), lo[order], -np.pi)
    hi_c = np.where(np.isfinite(hi[order]), hi[order], np.pi)
    n_act = lo.shape[0]
    mid = np.where(np.isfinite(lo) & np.isfinite(hi), (lo + hi) / 2.0, 0.0)
    lo_f = np.where(np.isfinite(lo), lo, -np.pi).astype(np.float32)
    hi_f = np.where(np.isfinite(hi), hi, np.pi).astype(np.float32)

    def make_seeds(rng, n_batch: int) -> np.ndarray:
        seeds = rng.uniform(
            lo_c, hi_c, (n_batch, num_seeds, quik.dof)
        ).astype(np.float32)
        if vamp_checker is None:
            return seeds
        q_full = np.broadcast_to(
            mid.astype(np.float32), (n_batch, num_seeds, n_act)
        ).copy()
        q_full[..., order] = seeds
        qp, _ok = vamp_checker.project_collision_free(
            None, q_full.reshape(-1, n_act),
            lower=lo_f, upper=hi_f, seed=int(rng.integers(2**31)),
        )
        return np.asarray(qp).reshape(n_batch, num_seeds, n_act)[..., order]

    def pick_best(q_actuated: np.ndarray, err: np.ndarray) -> np.ndarray:
        # q_actuated: [n_batch, num_seeds, n_act]; err: [n_batch, num_seeds]
        if vamp_checker is not None:
            free = np.asarray(
                vamp_checker.check_collision_free(
                    None, q_actuated.reshape(-1, q_actuated.shape[-1])
                )
            ).reshape(err.shape)
            err = np.where(free, err, err + 1e6)  # free solutions always win
        best = np.argmin(err, axis=1)
        return q_actuated[np.arange(err.shape[0]), best]

    return make_seeds, pick_best


def _run_quik_sequential(
    quik: "QuIKSolver",
    robot: "pk.Robot",
    target_link_index: int,
    target_poses: list,
    lo: np.ndarray,
    hi: np.ndarray,
    num_seeds: int,
    algorithm: int = 0,
    vamp_checker=None,
) -> list[SolveResult]:
    """Run the QuIK CPU solver sequentially, timing with plain wall-clock.

    QuIK is a CPU C++ FFI kernel, so (like PyRoKi) it is timed with wall-clock
    rather than the JAX ``lax.scan`` device timer.  Seeds are drawn in the
    chain-joint subspace; the returned config is scattered back into the robot's
    full actuated-joint vector for a like-for-like FK error evaluation.

    With ``vamp_checker`` the run is collision-aware: seeds are MPPI-projected
    onto the collision-free manifold before solving, and the best
    collision-free solution is preferred (both inside the timed region).
    """
    make_seeds, pick_best = _quik_seed_helpers(quik, lo, hi, num_seeds, vamp_checker)

    def one(pose_mat, rng) -> jax.Array:
        seeds = make_seeds(rng, 1)[0]
        poses = jnp.broadcast_to(jnp.asarray(pose_mat, jnp.float32), (num_seeds, 4, 4))
        out = quik.solve_to_actuated(poses, jnp.asarray(seeds), algorithm=algorithm)
        q = np.asarray(out["q_actuated"])[None]
        err = np.asarray(out["error"])[None]
        return jnp.asarray(pick_best(q, err)[0])

    results: list[SolveResult] = []
    for i, target_pose in enumerate(target_poses):
        pose_mat = np.asarray(target_pose.as_matrix())
        rng = np.random.default_rng(i + 1)
        cfg = one(pose_mat, rng)
        jax.block_until_ready(cfg)
        pos_err, rot_err = _pose_errors(robot, cfg, target_link_index, target_pose)
        times = []
        for j in range(N_TIMED):
            rng_j = np.random.default_rng(1000 * (i + 1) + j)
            t0 = time.perf_counter()
            out = one(pose_mat, rng_j)
            jax.block_until_ready(out)
            times.append(time.perf_counter() - t0)
        results.append(SolveResult(np.array(cfg), pos_err, rot_err, float(np.median(times)) * 1e3))
    return results


def _run_quik_batch(
    quik: "QuIKSolver",
    robot: "pk.Robot",
    target_link_index: int,
    target_poses_stacked: jaxlie.SE3,
    lo: np.ndarray,
    hi: np.ndarray,
    num_seeds: int,
    algorithm: int = 0,
    vamp_checker=None,
) -> BatchResult:
    """Run the QuIK CPU solver over all targets in one fused FFI call (wall-clock).

    With ``vamp_checker`` the run is collision-aware (see
    :func:`_run_quik_sequential`).
    """
    make_seeds, pick_best = _quik_seed_helpers(quik, lo, hi, num_seeds, vamp_checker)
    n_targets = len(target_poses_stacked.wxyz_xyz)
    Ts = np.stack([
        np.asarray(jaxlie.SE3(target_poses_stacked.wxyz_xyz[i]).as_matrix())
        for i in range(n_targets)
    ]).astype(np.float32)

    def one(rng) -> np.ndarray:
        seeds = make_seeds(rng, n_targets)
        poses = np.repeat(Ts[:, None], num_seeds, axis=1)  # (n_targets, num_seeds, 4, 4)
        out = quik.solve_to_actuated(
            jnp.asarray(poses.reshape(-1, 4, 4)),
            jnp.asarray(seeds.reshape(-1, quik.dof)),
            algorithm=algorithm,
        )
        q = np.asarray(out["q_actuated"]).reshape(n_targets, num_seeds, -1)
        err = np.asarray(out["error"]).reshape(n_targets, num_seeds)
        return pick_best(q, err)

    cfgs_np = one(np.random.default_rng(0))
    pos_errs = np.empty(n_targets)
    rot_errs = np.empty(n_targets)
    for i in range(n_targets):
        tp = jaxlie.SE3(target_poses_stacked.wxyz_xyz[i])
        pos_errs[i], rot_errs[i] = _pose_errors(
            robot, jnp.array(cfgs_np[i]), target_link_index, tp
        )

    times = []
    for j in range(N_TIMED):
        rng_j = np.random.default_rng(j + 1)
        t0 = time.perf_counter()
        out = one(rng_j)
        jax.block_until_ready(jnp.asarray(out))
        times.append(time.perf_counter() - t0)

    effective_ms = float(np.median(times)) * 1e3 / n_targets
    return BatchResult(cfgs_np, pos_errs, rot_errs, effective_ms)


# ---------------------------------------------------------------------------
# Collision environment helpers
# ---------------------------------------------------------------------------

def _build_and_save_env(path: pathlib.Path, robot_name: str = "panda") -> dict:
    """Define a static obstacle scene, persist it as JSON, and return it.

    The JSON includes a ``curobo_world_model`` section (x,y,z,qw,qx,qy,qz
    pose convention) that can be loaded directly into a CuRobo WorldConfig::

        from curobo.geom.types import WorldConfig
        import json
        env = json.load(open("resources/bench_env.json"))
        world_cfg = WorldConfig.from_dict(env["curobo_world_model"])
    """
    env = {
        "description": (
            f"Static collision benchmark environment for {robot_name} robot. "
            "Load in curobo via WorldConfig.from_dict(env['curobo_world_model'])."
        ),
        # Floor half-space.
        "floor": {"point": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, 1.0]},
        # Sphere obstacles scattered through the Panda workspace.
        "spheres": [
            {"name": "center_obs", "center": [0.40,  0.00, 0.50], "radius": 0.10},
            {"name": "left_obs",   "center": [0.20,  0.40, 0.40], "radius": 0.08},
            {"name": "right_obs",  "center": [0.30, -0.30, 0.60], "radius": 0.08},
        ],
        # Box obstacles (table pedestal in front of the robot).
        "cuboids": [
            {
                "name":   "table",
                "center": [0.50, 0.00, 0.20],
                "dims":   [0.40, 0.60, 0.40],   # length × width × height
                "wxyz":   [1.0, 0.0, 0.0, 0.0],
            },
        ],
        # CuRobo-compatible world model (pose: x,y,z,qw,qx,qy,qz).
        "curobo_world_model": {
            "cuboid": {
                "table": {
                    "dims": [0.40, 0.60, 0.40],
                    "pose": [0.50, 0.00, 0.20, 1.0, 0.0, 0.0, 0.0],
                },
            },
            "sphere": {
                "center_obs": {"radius": 0.10, "pose": [0.40,  0.00, 0.50, 1, 0, 0, 0]},
                "left_obs":   {"radius": 0.08, "pose": [0.20,  0.40, 0.40, 1, 0, 0, 0]},
                "right_obs":  {"radius": 0.08, "pose": [0.30, -0.30, 0.60, 1, 0, 0, 0]},
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(env, indent=2))
    return env


def _validate_env_dict(env: dict, path: pathlib.Path) -> None:
    """Validate environment schema and raise ValueError with clear details."""
    if not isinstance(env, dict):
        raise ValueError(f"Environment JSON at {path} must be an object/dict.")

    spheres = env.get("spheres", [])
    if not isinstance(spheres, list):
        raise ValueError(f"Environment JSON at {path} key spheres must be a list")
    for i, sphere in enumerate(spheres):
        if not isinstance(sphere, dict):
            raise ValueError(f"Environment JSON at {path} spheres[{i}] must be an object")
        if "center" not in sphere or "radius" not in sphere:
            raise ValueError(f"Environment JSON at {path} spheres[{i}] must contain center and radius")
        if len(sphere["center"]) != 3:
            raise ValueError(f"Environment JSON at {path} spheres[{i}].center must be length-3")

    cuboids = env.get("cuboids", [])
    if not isinstance(cuboids, list):
        raise ValueError(f"Environment JSON at {path} key cuboids must be a list")
    for i, cuboid in enumerate(cuboids):
        if not isinstance(cuboid, dict):
            raise ValueError(f"Environment JSON at {path} cuboids[{i}] must be an object")
        if "center" not in cuboid or "dims" not in cuboid:
            raise ValueError(f"Environment JSON at {path} cuboids[{i}] must contain center and dims")
        if len(cuboid["center"]) != 3 or len(cuboid["dims"]) != 3:
            raise ValueError(f"Environment JSON at {path} cuboids[{i}].center and cuboids[{i}].dims must be length-3")
        if "wxyz" in cuboid and len(cuboid["wxyz"]) != 4:
            raise ValueError(f"Environment JSON at {path} cuboids[{i}].wxyz must be length-4")


def _env_to_geoms(env: dict):
    """Build pyroffi CollGeom objects from an env dict.

    Args:
        env: Environment dictionary with spheres and cuboids.

    Returns:
        obs_geoms: list of Sphere / Box
    """
    obs_geoms: list = []
    for s in env.get("spheres", []):
        obs_geoms.append(
            Sphere.from_center_and_radius(
                np.array(s["center"], dtype=np.float32),
                np.array([s["radius"]], dtype=np.float32),
            )
        )
    for b in env.get("cuboids", []):
        d = b["dims"]
        wxyz = b.get("wxyz", [1.0, 0.0, 0.0, 0.0])
        obs_geoms.append(
            Box.from_center_and_dimensions(
                np.array(b["center"], dtype=np.float32),
                float(d[0]), float(d[1]), float(d[2]),
                wxyz=np.array(wxyz, dtype=np.float32),
            )
        )
    return obs_geoms


def _default_env_file() -> pathlib.Path:
    """Return shared benchmark environment path used across all robots."""
    return ENV_FILE


def _default_srdf_for_robot(robot_name: str) -> pathlib.Path | None:
    """Resolve SRDF path for a robot, preferring explicit mapping then folder scan."""
    mapped = ROBOT_SRDFS.get(robot_name)
    if mapped is not None and mapped.exists():
        return mapped

    urdf_path = ROBOT_URDFS.get(robot_name)
    if urdf_path is None:
        return None

    srdf_candidates = sorted(urdf_path.parent.glob("*.srdf"))
    if len(srdf_candidates) == 1:
        return srdf_candidates[0]

    # If multiple SRDFs exist, prefer one whose stem prefixes the URDF stem.
    urdf_stem = urdf_path.stem
    for candidate in srdf_candidates:
        if urdf_stem.startswith(candidate.stem):
            return candidate

    return srdf_candidates[0] if srdf_candidates else None


def _disabled_pairs_from_srdf(srdf_path: pathlib.Path | None) -> tuple[tuple[str, str], ...]:
    """Load SRDF disabled collision pairs as RobotCollisionSpherized ignore tuples."""
    if srdf_path is None or not srdf_path.exists():
        return ()
    try:
        pairs = read_disabled_collisions_from_srdf(srdf_path.as_posix())
        return tuple(
            (str(p["link1"]), str(p["link2"]))
            for p in pairs
            if p.get("link1") and p.get("link2")
        )
    except Exception as exc:
        print(f"  Warning: failed to parse SRDF {srdf_path}: {exc}")
        return ()


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

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


def _write_csv(
    path: pathlib.Path,
    timestamp: str,
    robot_name: str,
    seq_results: dict[str, list[SolveResult]],
    batch_results: dict[str, BatchResult],
    seq_coll_results: dict[str, list[SolveResult]],
    batch_coll_results: dict[str, BatchResult],
    seq_coll_free: dict[str, int],
    batch_coll_free: dict[str, int],
) -> None:
    """Append all benchmark results to *path* as CSV rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()

    rows: list[dict] = []

    # -- Sequential (no collision) ------------------------------------------
    for solver, results in seq_results.items():
        pos = np.array([r.pos_err * 1e3 for r in results])
        rot = np.array([r.rot_err       for r in results])
        t   = np.array([r.time_ms       for r in results])
        solved = sum(r.pos_err < POS_THR_M and r.rot_err < ROT_THR_RAD for r in results)
        rows.append({
            "timestamp":      timestamp,
            "robot":          robot_name,
            "mode":           "sequential",
            "solver":         solver,
            "collision_free": False,
            "n_problems":     len(results),
            "n_timed":        N_TIMED,
            "t_med_ms":       round(float(np.median(t)),        6),
            "t_p95_ms":       round(float(np.percentile(t, 95)), 6),
            "pos_med_mm":     round(float(np.median(pos)),       6),
            "pos_p95_mm":     round(float(np.percentile(pos, 95)), 6),
            "rot_med_rad":    round(float(np.median(rot)),       6),
            "rot_p95_rad":    round(float(np.percentile(rot, 95)), 6),
            "success_n":      solved,
            "success_total":  len(results),
            "coll_free_n":    "",
            "peak_gpu_pct":   "",
            "avg_gpu_pct":    "",
            "peak_vram_mb":   "",
        })

    # -- Batch (no collision) ------------------------------------------------
    for solver, result in batch_results.items():
        pos    = result.pos_errs * 1e3
        rot    = result.rot_errs
        solved = int(np.sum((result.pos_errs < POS_THR_M) & (result.rot_errs < ROT_THR_RAD)))

        def _fmtf(v): return round(float(v), 6) if not np.isnan(v) else ""

        rows.append({
            "timestamp":      timestamp,
            "robot":          robot_name,
            "mode":           "batch",
            "solver":         solver,
            "collision_free": False,
            "n_problems":     len(pos),
            "n_timed":        N_TIMED,
            "t_med_ms":       round(result.time_ms, 6),
            "t_p95_ms":       "",
            "pos_med_mm":     round(float(np.median(pos)),        6),
            "pos_p95_mm":     round(float(np.percentile(pos, 95)), 6),
            "rot_med_rad":    round(float(np.median(rot)),         6),
            "rot_p95_rad":    round(float(np.percentile(rot, 95)), 6),
            "success_n":      solved,
            "success_total":  len(pos),
            "coll_free_n":    "",
            "peak_gpu_pct":   _fmtf(result.peak_gpu_util),
            "avg_gpu_pct":    _fmtf(result.avg_gpu_util),
            "peak_vram_mb":   _fmtf(result.peak_vram_mb),
        })

    # -- Sequential (collision-free) -----------------------------------------
    for solver, results in seq_coll_results.items():
        pos = np.array([r.pos_err * 1e3 for r in results])
        rot = np.array([r.rot_err       for r in results])
        t   = np.array([r.time_ms       for r in results])
        solved = sum(r.pos_err < POS_THR_M and r.rot_err < ROT_THR_RAD for r in results)
        rows.append({
            "timestamp":      timestamp,
            "robot":          robot_name,
            "mode":           "sequential",
            "solver":         solver,
            "collision_free": True,
            "n_problems":     len(results),
            "n_timed":        N_TIMED,
            "t_med_ms":       round(float(np.median(t)),          6),
            "t_p95_ms":       round(float(np.percentile(t, 95)),  6),
            "pos_med_mm":     round(float(np.median(pos)),         6),
            "pos_p95_mm":     round(float(np.percentile(pos, 95)), 6),
            "rot_med_rad":    round(float(np.median(rot)),         6),
            "rot_p95_rad":    round(float(np.percentile(rot, 95)), 6),
            "success_n":      solved,
            "success_total":  len(results),
            "coll_free_n":    seq_coll_free.get(solver, ""),
            "peak_gpu_pct":   "",
            "avg_gpu_pct":    "",
            "peak_vram_mb":   "",
        })

    # -- Batch (collision-free) ----------------------------------------------
    for solver, result in batch_coll_results.items():
        pos    = result.pos_errs * 1e3
        rot    = result.rot_errs
        solved = int(np.sum((result.pos_errs < POS_THR_M) & (result.rot_errs < ROT_THR_RAD)))

        def _fmtf(v): return round(float(v), 6) if not np.isnan(v) else ""  # noqa: F811

        rows.append({
            "timestamp":      timestamp,
            "robot":          robot_name,
            "mode":           "batch",
            "solver":         solver,
            "collision_free": True,
            "n_problems":     len(pos),
            "n_timed":        N_TIMED,
            "t_med_ms":       round(result.time_ms, 6),
            "t_p95_ms":       "",
            "pos_med_mm":     round(float(np.median(pos)),         6),
            "pos_p95_mm":     round(float(np.percentile(pos, 95)), 6),
            "rot_med_rad":    round(float(np.median(rot)),         6),
            "rot_p95_rad":    round(float(np.percentile(rot, 95)), 6),
            "success_n":      solved,
            "success_total":  len(pos),
            "coll_free_n":    batch_coll_free.get(solver, ""),
            "peak_gpu_pct":   _fmtf(result.peak_gpu_util),
            "avg_gpu_pct":    _fmtf(result.avg_gpu_util),
            "peak_vram_mb":   _fmtf(result.peak_vram_mb),
        })

    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve_target_link_name(robot_name: str, robot: pk.Robot) -> str:
    """Pick a valid end-effector link for the current robot."""
    candidates = ROBOT_TARGET_LINK_CANDIDATES.get(robot_name, ())
    for name in candidates:
        if name in robot.links.names:
            return name
    raise ValueError(
        f"No valid target link found for robot '{robot_name}'. "
        f"Tried {list(candidates)}"
    )


def _run_robot_benchmark(
    robot_name: str,
    csv_file: pathlib.Path | None,
    solver_filter: str | None = None,
    blocks: set[str] | None = None,
) -> None:  # noqa: C901
    # When ``solver_filter`` is set (the per-solver subprocess path, see main()),
    # only that one base solver is set up, warmed up, timed and written. ``_want``
    # normalises the ``-COLL`` / ``-BATCH`` variant suffixes back to the base label
    # so all four phases of the selected solver pass the filter and nothing else does.
    def _want(name: str) -> bool:
        if solver_filter is None:
            return True
        return name.replace("-COLL", "").replace("-BATCH", "") == solver_filter

    # ``blocks`` (from --blocks) restricts which of the four evaluation phases run:
    # {"seq","seq_coll","batch","batch_coll"}. None means all four.
    def _block_wanted(block: str) -> bool:
        return blocks is None or block in blocks

    print("=" * 80)
    _title_solver = solver_filter if solver_filter is not None else \
        "HJCD-IK, LS-IK, SQP-IK, and MPPI-IK"
    print(f"IK benchmark: {_title_solver}  (robot={robot_name}, "
          f"n_targets={N_TARGETS}, n_timed={N_TIMED})")
    gpu_mon_status = (
        "disabled (--cpu-only)"
        if _CPU_ONLY else
        "enabled (pynvml)"
        if _NVML_OK else
        "disabled (install nvidia-ml-py for GPU stats)"
    )
    if _CPU_ONLY:
        print("CPU-only mode: running only CPU-native solvers (QuIK); "
              "skipping all CUDA and JAX solvers; JAX_PLATFORMS=cpu")
    elif _NO_JAX:
        print("No-JAX mode: skipping all JAX-based solvers "
              "(HJCD/LS/SQP/MPPI-JAX, Learned-JAX, PyRoKi); "
              "CUDA/FFI kernel solvers and QuIK still run")
    print(f"GPU monitoring: {gpu_mon_status}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Load robot
    # ------------------------------------------------------------------
    print("\nLoading robot ...")
    urdf_path = ROBOT_URDFS[robot_name]
    if not urdf_path.exists():
        raise FileNotFoundError(f"Spherized URDF not found: {urdf_path}")
    mesh_dir = urdf_path.parent / "meshes"
    if mesh_dir.exists():
        urdf = yourdfpy.URDF.load(str(urdf_path), mesh_dir=str(mesh_dir))
    else:
        urdf = yourdfpy.URDF.load(str(urdf_path))
    robot = pk.Robot.from_urdf(urdf)
    n_act = robot.joints.num_actuated_joints
    target_link_name = _resolve_target_link_name(robot_name, robot)
    fixed_joint_names = ROBOT_FIXED_JOINT_NAMES.get(robot_name, ())
    target_link_index = robot.links.names.index(target_link_name)

    fixed_joint_mask = jnp.array(
        [name in fixed_joint_names for name in robot.joints.actuated_names],
        dtype=jnp.int32,
    )
    print(f"  URDF: {urdf_path}")
    print(f"  {n_act} actuated joints, target link: '{target_link_name}'")
    print(f"  Fixed joints: {[n for n in fixed_joint_names if n in robot.joints.actuated_names]}")

    lo = np.array(robot.joints.lower_limits)
    hi = np.array(robot.joints.upper_limits)
    mid_cfg = jnp.array((lo + hi) / 2, dtype=jnp.float32)

    # ------------------------------------------------------------------
    # PyRoKi setup (optional): two variants sharing one pyroki robot
    #   PyRoKi-LS          — autodiff pose residual (pyroki.costs.pose_cost)
    #   PyRoKi-AnalyticJac — analytic task-space Jacobian
    #                        (pyroki.costs.pose_cost_analytic_jac), same LM solver
    # ------------------------------------------------------------------
    _pyroki_ls       = None   # (solve_fn, batch_fn)
    _pyroki_analytic = None   # (solve_fn, batch_fn)
    _pyroki_ls_coll       = None
    _pyroki_analytic_coll = None

    if not (_want("PyRoKi-LS") or _want("PyRoKi-AnalyticJac")):
        pass  # not the selected solver; skip PyRoKi setup entirely
    elif _NO_JAX:
        print("\nPyRoKi disabled (--cpu-only/--no-jax: JAX-based solvers are skipped).")
    elif _PYROKI_AVAILABLE:
        print("\nSetting up PyRoKi solvers (LS + AnalyticJac) ...")
        _pyroki_robot = _pyroki.Robot.from_urdf(urdf)
        # Warm-up inputs shared by both variants.
        _num_seeds = IK_KWARGS_PYROKI["num_seeds"]
        _lo_j = jnp.array(lo, dtype=jnp.float32)
        _hi_j = jnp.array(hi, dtype=jnp.float32)
        _seeds0 = jax.random.uniform(
            jax.random.PRNGKey(0), (_num_seeds, n_act), minval=_lo_j, maxval=_hi_j,
        )
        # Placeholder target to warm up; it will be replaced in actual runs.
        _mid_wxyz_xyz = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5], dtype=jnp.float32)

        def _build_pyroki_variant(label, pose_cost_factory):
            """Build and warm up one PyRoKi variant; None if not selected."""
            if not _want(label):
                return None
            solve_fn, batch_fn = _make_pyroki_solvers(
                _pyroki_robot, target_link_index, IK_KWARGS_PYROKI,
                pose_cost_factory=pose_cost_factory,
            )
            if _block_wanted("seq") or _block_wanted("seq_coll"):
                print(f"  Warming up {label} (JIT compile) ...")
                for _ in range(N_WARMUP):
                    _w = solve_fn(_mid_wxyz_xyz, _seeds0)
                    jax.block_until_ready(_w)
                print(f"  {label} ready.")
            return solve_fn, batch_fn

        _pyroki_ls = _build_pyroki_variant("PyRoKi-LS", _pyroki.costs.pose_cost)
        _pyroki_analytic = _build_pyroki_variant(
            "PyRoKi-AnalyticJac", _pyroki.costs.pose_cost_analytic_jac,
        )
    elif not _PYROKI_AVAILABLE:
        print("\nPyRoKi unavailable (pip install git+https://github.com/chungmin99/pyroki.git).")

    # ------------------------------------------------------------------
    # QuIK CPU solver setup (optional)
    # ------------------------------------------------------------------
    _quik_solver = None
    _quik_num_seeds = 32
    if _QUIK_IMPORT_OK and _want("QuIK-CPU"):
        print("\nSetting up QuIK CPU solver (POE->DH + cricket JIT) ...")
        try:
            _quik_solver = QuIKSolver(robot, target_link_name)
            # Warm up (compile the FFI kernel; DH already extracted/validated).
            _order = _quik_solver.model.actuated_order
            _lo_c = np.where(np.isfinite(lo[_order]), lo[_order], -np.pi)
            _hi_c = np.where(np.isfinite(hi[_order]), hi[_order], np.pi)
            _seeds0 = np.random.default_rng(0).uniform(
                _lo_c, _hi_c, (_quik_num_seeds, _quik_solver.dof)
            ).astype(np.float32)
            _p0 = jnp.broadcast_to(jnp.eye(4, dtype=jnp.float32), (_quik_num_seeds, 4, 4))
            for _ in range(N_WARMUP):
                jax.block_until_ready(_quik_solver.solve(_p0, jnp.asarray(_seeds0))["q"])
            print(f"  QuIK ready (dof={_quik_solver.dof}, CPU).")
        except Exception as e:  # noqa: BLE001
            print(f"  QuIK unavailable for {robot_name}: {type(e).__name__}: {str(e)[:90]}")
            _quik_solver = None
    elif _want("QuIK-CPU"):
        print("\nQuIK unavailable (needs cricket JIT; build external/cricket).")

    # ------------------------------------------------------------------
    # Collision setup
    # ------------------------------------------------------------------
    robot_coll     = None
    _obs_geoms: list = []
    _collision_penalty = None
    _quik_vamp_checker = None
    coll_kwargs_jax  = {}
    coll_kwargs_cuda = {}
    coll_kwargs_ls_cuda_kernel = {}

    if COLLISION_FREE:
        print("\nSetting up collision environment ...")
        srdf_path = _default_srdf_for_robot(robot_name)
        ignore_pairs = _disabled_pairs_from_srdf(srdf_path)
        robot_coll = RobotCollisionSpherized.from_urdf(urdf, user_ignore_pairs=ignore_pairs)
        if srdf_path is not None and srdf_path.exists():
            print(f"  Using SRDF disabled pairs: {srdf_path} ({len(ignore_pairs)} pairs)")
        else:
            print("  SRDF disabled pairs: none")
        env_file = _default_env_file()
        if not env_file.exists():
            raise FileNotFoundError(
                f"Environment file not found: {env_file}. "
                "Create it once (for example by running this benchmark for panda) "
                "or point your workflow to an existing bench_env.json."
            )
        env_dict = json.loads(env_file.read_text())
        print(f"  Loaded environment from {env_file}")

        _validate_env_dict(env_dict, env_file)

        _obs_geoms = _env_to_geoms(env_dict)
        print(f"  Obstacles: {len(env_dict.get('spheres', []))} spheres"
              f" + {len(env_dict.get('cuboids', []))} cuboids")

        # All obstacles are captured in the closure — no dynamic arg needed.
        def _collision_penalty(cfg, robot_arg, _dummy):
            coll_geom = robot_coll.at_config(robot_arg, cfg)
            penalty   = jnp.zeros(())
            for obs in _obs_geoms:
                d = collide(coll_geom, obs.broadcast_to((1,)))
                penalty = penalty + jnp.sum(jax.nn.softplus(-d / _COLL_EPS) * _COLL_EPS)
            return penalty

        _dummy = jnp.zeros(())

        # JAX solvers use ``constraint_fns`` (tuple of callables).
        coll_kwargs_jax = dict(
            constraint_fns    = (_collision_penalty,),
            constraint_args   = (_dummy,),
            constraint_weights = jnp.array([COLL_WEIGHT]),
        )
        # CUDA wrappers use ``constraints`` (Sequence[Callable]).
        coll_kwargs_cuda = dict(
            constraints        = [_collision_penalty],
            constraint_args    = [_dummy],
            constraint_weights = [COLL_WEIGHT],
        )

        # LS-CUDA in-kernel collision path (no Python/JAX collision constraint fn).
        coll_kwargs_ls_cuda_kernel = dict(
            collision_free=True,
            collision_checker=robot_coll,
            collision_world=_obs_geoms,
            collision_weight=COLL_WEIGHT,
            collision_margin=_COLL_EPS,
            constraint_refine_iters=0,
        )

        # JIT a fast per-config collision checker for post-solve reporting.
        def _min_coll_dist_single(cfg):
            coll_geom = robot_coll.at_config(robot, cfg)
            dists = []
            for obs in _obs_geoms:
                dists.append(jnp.min(collide(coll_geom, obs.broadcast_to((1,)))))
            if not dists:
                return jnp.inf
            return jnp.min(jnp.stack(dists))

        _check_coll_jit        = jax.jit(_min_coll_dist_single)
        _check_coll_batch_jit  = jax.jit(jax.vmap(_min_coll_dist_single))

        # Collision-aware QuIK: VAMP CPU checker + MPPI seed projection.
        if _quik_solver is not None and _VAMP_CPU_IMPORT_OK:
            print("  Setting up collision-aware QuIK (VAMP MPPI seed projection) ...")
            try:
                _quik_vamp_checker = VAMPCPUCollisionChecker(
                    ROBOT_URDFS[robot_name], srdf_path=srdf_path
                )
                if _quik_vamp_checker.dimension != n_act:
                    raise RuntimeError(
                        f"VAMP dimension {_quik_vamp_checker.dimension} != "
                        f"robot actuated count {n_act}"
                    )
                _quik_vamp_checker.set_world_geoms(_obs_geoms)
                # Warm up the projection + check kernels.
                _warm = np.zeros((4, n_act), np.float32)
                jax.block_until_ready(
                    _quik_vamp_checker.project_collision_free(None, _warm)[0]
                )
                jax.block_until_ready(
                    _quik_vamp_checker.check_collision_free(None, _warm)
                )
                print("  Collision-aware QuIK ready.")
            except Exception as e:  # noqa: BLE001
                print(f"  Collision-aware QuIK unavailable: "
                      f"{type(e).__name__}: {str(e)[:90]}")
                _quik_vamp_checker = None

        def _make_pyroki_coll_variant(variant, label):
            """Collision-aware build of one PyRoKi variant; None if the base
            variant is absent."""
            if variant is None:
                return None
            print(f"  Setting up collision-aware {label} ...")

            def _pyroki_coll_cost(cfg):
                return _collision_penalty(cfg, robot, _dummy)

            return _make_pyroki_solvers(
                _pyroki_robot,
                target_link_index,
                IK_KWARGS_PYROKI,
                collision_cost_fn=_pyroki_coll_cost,
                collision_weight=COLL_WEIGHT,
            )

        _pyroki_ls_coll = _make_pyroki_coll_variant(_pyroki_ls, "PyRoKi-LS")
        _pyroki_analytic_coll = _make_pyroki_coll_variant(
            _pyroki_analytic, "PyRoKi-AnalyticJac",
        )

    # ------------------------------------------------------------------
    # Learned-IK: load pre-trained Flax model (optional)
    # ------------------------------------------------------------------
    _learned_ik_available = False
    _learned_ik_fn        = None
    _learned_ik_fn_batch  = None
    if not _want("Learned-JAX"):
        pass  # not the selected solver; skip Learned-IK setup entirely
    elif _NO_JAX:
        print("\nLearned-IK disabled (--cpu-only/--no-jax: JAX-based solvers are skipped).")
    elif _LEARNED_IK_IMPORT_OK:
        _model_path = get_default_model_path(robot_name)
        if _model_path.exists():
            print(f"\nLoaded Learned-IK model: {_model_path}")
            _model_data  = load_learned_ik(_model_path)
            _model_params = _model_data["params"]
            def _infer_ikflow_arch(params):
                if isinstance(params, dict) and "params" in params:
                    params = params["params"]
                if not isinstance(params, dict):
                    return 15, 1024
                nets = [k for k in params.keys() if k.startswith("nets_")]
                if not nets:
                    return 15, 1024
                n_layers = len(nets)
                hidden = 1024
                try:
                    hidden = int(params[nets[0]]["Dense_0"]["kernel"].shape[1])
                except Exception:
                    pass
                return n_layers, hidden

            _n_layers, _hidden = _infer_ikflow_arch(_model_params)
            _learned_base = make_learned_ik_solve(
                robot,
                latent_dim=_model_data.get("latent_dim", 15),
                n_layers=_n_layers,
                hidden=_hidden,
            )

            # Wrap so that model_params is baked in and the signature
            # matches the other single-problem solvers used in the benchmark.
            def _learned_ik_fn(
                robot, target_link_indices, target_poses,
                rng_key, previous_cfg, fixed_joint_mask=None, **kwargs,
            ):
                return _learned_base(
                    robot, target_link_indices, target_poses,
                    rng_key, previous_cfg,
                    model_params=_model_params,
                    fixed_joint_mask=fixed_joint_mask,
                    **kwargs,
                )

            _learned_ik_fn_batch = _make_batched_jax_solver(
                _learned_ik_fn, IK_KWARGS_LEARNED_JAX,
            )
            _learned_ik_available = True
        else:
            print(f"\nLearned-IK model not found at {_model_path}")
            print(f"  Run: python train_learned_ik.py --robot {robot_name}")
            print("  Learned-IK rows will be skipped in the benchmark.")
    else:
        print("\nLearned-IK unavailable (flax not installed).")

    rng_np  = np.random.default_rng(0)

    # ------------------------------------------------------------------
    # Generate target poses  (batch superset; sequential uses first N_TARGETS)
    # ------------------------------------------------------------------
    print(f"\nGenerating target poses (seq={N_TARGETS}, batch={N_TARGETS_BATCH}) ...")
    target_cfgs_np = rng_np.uniform(lo, hi, size=(N_TARGETS_BATCH, n_act)).astype(np.float32)
    all_target_poses: list[jaxlie.SE3] = []
    for i in range(N_TARGETS_BATCH):
        cfg_i = jnp.array(target_cfgs_np[i])
        Ts    = robot.forward_kinematics(cfg_i)
        all_target_poses.append(jaxlie.SE3(Ts[target_link_index]))

    # Sequential subset.
    target_poses = all_target_poses[:N_TARGETS]

    # Stack all batch poses into a single batched SE3 for the batch solvers.
    target_poses_stacked = jaxlie.SE3(
        jnp.stack([p.wxyz_xyz for p in all_target_poses])
    )  # (N_TARGETS_BATCH, 7)

    # Sidecar for the cuRobo child, which runs in a separate conda env and
    # cannot consume these jaxlie poses directly (see bench_ik_curobo.py).
    # The dump is deterministic (seed-0 targets), so every pyroffi child
    # rewrites the identical file; cuRobo is the LAST candidate solver per
    # robot, so by the time its child starts the file is guaranteed to exist.
    if robot_name in _CUROBO_ROBOT_FILES:
        out_dir = csv_file.parent if csv_file is not None else RESOURCE_ROOT
        targets_npz = out_dir / f"bench_ik_targets_{robot_name}.npz"
        np.savez_compressed(
            targets_npz,
            seq=np.stack([p.wxyz_xyz for p in target_poses]).astype(np.float32),
            batch=target_poses_stacked.wxyz_xyz.astype(np.float32),
        )
        print(f"  wrote cuRobo target sidecar: {targets_npz}")

    # Per-pose RNG keys and warm-start configs.
    rng_keys          = [jax.random.PRNGKey(i + 1) for i in range(N_TARGETS)]
    rng_keys_batch    = jnp.stack([jax.random.PRNGKey(i + 1) for i in range(N_TARGETS_BATCH)])
    previous_cfgs_seq   = [mid_cfg] * N_TARGETS
    previous_cfgs_batch = jnp.tile(mid_cfg[None], (N_TARGETS_BATCH, 1))  # (N_TARGETS_BATCH, n_act)

    # ------------------------------------------------------------------
    # JIT-compile / warm up all solvers
    # ------------------------------------------------------------------
    rng0 = jax.random.PRNGKey(0)

    jit_hjcd = jax.jit(
        functools.partial(hjcd_solve, **IK_KWARGS_HJCD_JAX),
        static_argnames=("target_link_indices", "num_seeds", "coarse_max_iter", "lm_max_iter"),
    )
    jit_ls = jax.jit(
        functools.partial(ls_ik_solve, **IK_KWARGS_LS_JAX),
        static_argnames=("target_link_indices", "num_seeds", "max_iter"),
    )
    jit_sqp = jax.jit(
        functools.partial(sqp_ik_solve, **IK_KWARGS_SQP_JAX),
        static_argnames=("target_link_indices", "num_seeds", "max_iter", "n_inner_iters"),
    )
    jit_hjcd_batch = _make_batched_jax_solver(hjcd_solve, IK_KWARGS_HJCD_JAX)
    jit_ls_batch   = _make_batched_jax_solver(ls_ik_solve, IK_KWARGS_LS_JAX)
    jit_sqp_batch  = _make_batched_jax_solver(sqp_ik_solve, IK_KWARGS_SQP_JAX)
    jit_mppi = jax.jit(
        functools.partial(mppi_ik_solve, **IK_KWARGS_MPPI_JAX),
        static_argnames=("target_link_indices", "num_seeds", "n_particles",
                         "n_mppi_iters", "n_lbfgs_iters", "m_lbfgs"),
    )
    jit_mppi_batch = _make_batched_jax_solver(mppi_ik_solve, IK_KWARGS_MPPI_JAX)

    # Collision-aware JAX sequential solvers.
    # ``constraint_fns`` must be in static_argnames (it's a tuple of callables).
    if COLLISION_FREE:
        jit_hjcd_coll = jax.jit(
            functools.partial(hjcd_solve, **IK_KWARGS_HJCD_JAX),
            static_argnames=(
                "target_link_indices", "num_seeds", "coarse_max_iter",
                "lm_max_iter", "constraint_fns",
            ),
        )
        jit_ls_coll = jax.jit(
            functools.partial(ls_ik_solve, **IK_KWARGS_LS_JAX),
            static_argnames=("target_link_indices", "num_seeds", "max_iter", "constraint_fns"),
        )
        jit_sqp_coll = jax.jit(
            functools.partial(sqp_ik_solve, **IK_KWARGS_SQP_JAX),
            static_argnames=(
                "target_link_indices", "num_seeds", "max_iter",
                "n_inner_iters", "constraint_fns",
            ),
        )
        jit_mppi_coll = jax.jit(
            functools.partial(mppi_ik_solve, **IK_KWARGS_MPPI_JAX),
            static_argnames=(
                "target_link_indices", "num_seeds", "n_particles",
                "n_mppi_iters", "n_lbfgs_iters", "m_lbfgs", "constraint_fns",
            ),
        )
        # Collision-aware batch JAX solvers (collision baked into ik_kwargs).
        jit_hjcd_coll_batch = _make_batched_jax_solver(
            hjcd_solve, {**IK_KWARGS_HJCD_JAX, **coll_kwargs_jax}
        )
        jit_ls_coll_batch = _make_batched_jax_solver(
            ls_ik_solve, {**IK_KWARGS_LS_JAX, **coll_kwargs_jax}
        )
        jit_sqp_coll_batch = _make_batched_jax_solver(
            sqp_ik_solve, {**IK_KWARGS_SQP_JAX, **coll_kwargs_jax}
        )
        jit_mppi_coll_batch = _make_batched_jax_solver(
            mppi_ik_solve, {**IK_KWARGS_MPPI_JAX, **coll_kwargs_jax}
        )

    warmup_seq = [] if _NO_JAX else [
        ("HJCD-JAX",   jit_hjcd,          {}),
        ("LS-JAX",     jit_ls,            {}),
        ("SQP-JAX",    jit_sqp,           {}),
        ("MPPI-JAX",   jit_mppi,          {}),
    ]
    if not _CPU_ONLY:
        warmup_seq += [
            ("HJCD-CUDA",  hjcd_solve_cuda,    IK_KWARGS_HJCD_CUDA),
            ("LS-CUDA",    ls_ik_solve_cuda,   IK_KWARGS_LS_CUDA),
            ("SQP-CUDA",   sqp_ik_solve_cuda,  IK_KWARGS_SQP_CUDA),
            ("MPPI-CUDA",  mppi_ik_solve_cuda, IK_KWARGS_MPPI_CUDA),
        ]
    if _learned_ik_available:
        warmup_seq.append(("Learned-JAX", _learned_ik_fn, IK_KWARGS_LEARNED_JAX))
    if COLLISION_FREE:
        if not _NO_JAX:
            warmup_seq += [
                ("HJCD-JAX-COLL",  jit_hjcd_coll,         coll_kwargs_jax),
                ("LS-JAX-COLL",    jit_ls_coll,            coll_kwargs_jax),
                ("SQP-JAX-COLL",   jit_sqp_coll,           coll_kwargs_jax),
                ("MPPI-JAX-COLL",  jit_mppi_coll,          coll_kwargs_jax),
            ]
        if not _CPU_ONLY:
            warmup_seq += [
                ("HJCD-CUDA-COLL", hjcd_solve_cuda,        {**IK_KWARGS_HJCD_CUDA, **coll_kwargs_cuda}),
                ("LS-CUDA-COLL",   ls_ik_solve_cuda,       {**IK_KWARGS_LS_CUDA, **coll_kwargs_ls_cuda_kernel}),
                ("SQP-CUDA-COLL",  sqp_ik_solve_cuda,      {**IK_KWARGS_SQP_CUDA, **coll_kwargs_cuda}),
                ("MPPI-CUDA-COLL", mppi_ik_solve_cuda,     {**IK_KWARGS_MPPI_CUDA, **coll_kwargs_cuda}),
            ]

    warmup_batch_jax = [] if _NO_JAX else [
        ("HJCD-JAX-BATCH",  jit_hjcd_batch,  {}),
        ("LS-JAX-BATCH",    jit_ls_batch,    {}),
        ("SQP-JAX-BATCH",   jit_sqp_batch,   {}),
        ("MPPI-JAX-BATCH",  jit_mppi_batch,  {}),
    ]
    if _learned_ik_available:
        warmup_batch_jax.append(("Learned-JAX-BATCH", _learned_ik_fn_batch, {}))
    if COLLISION_FREE and not _NO_JAX:
        warmup_batch_jax += [
            ("HJCD-JAX-COLL-BATCH", jit_hjcd_coll_batch, {}),
            ("LS-JAX-COLL-BATCH",   jit_ls_coll_batch,   {}),
            ("SQP-JAX-COLL-BATCH",  jit_sqp_coll_batch,  {}),
            ("MPPI-JAX-COLL-BATCH", jit_mppi_coll_batch, {}),
        ]

    warmup_batch_cuda = []
    if not _CPU_ONLY:
        warmup_batch_cuda += [
            ("LS-CUDA-BATCH",   ls_ik_solve_cuda_batch,   IK_KWARGS_LS_CUDA),
            ("HJCD-CUDA-BATCH", hjcd_solve_cuda_batch,     IK_KWARGS_HJCD_CUDA),
            ("SQP-CUDA-BATCH",  sqp_ik_solve_cuda_batch,  IK_KWARGS_SQP_CUDA),
            ("MPPI-CUDA-BATCH", mppi_ik_solve_cuda_batch, IK_KWARGS_MPPI_CUDA),
        ]
        if COLLISION_FREE:
            warmup_batch_cuda += [
                ("LS-CUDA-COLL-BATCH",   ls_ik_solve_cuda_batch,   {**IK_KWARGS_LS_CUDA,   **coll_kwargs_ls_cuda_kernel}),
                ("HJCD-CUDA-COLL-BATCH", hjcd_solve_cuda_batch,    {**IK_KWARGS_HJCD_CUDA, **coll_kwargs_cuda}),
                ("SQP-CUDA-COLL-BATCH",  sqp_ik_solve_cuda_batch,  {**IK_KWARGS_SQP_CUDA,  **coll_kwargs_cuda}),
                ("MPPI-CUDA-COLL-BATCH", mppi_ik_solve_cuda_batch, {**IK_KWARGS_MPPI_CUDA, **coll_kwargs_cuda}),
            ]

    # Per-solver isolation: keep only the selected solver's warmups so that only
    # its timers are built (and only its kernels are JIT/compiled in this process).
    warmup_seq        = [e for e in warmup_seq        if _want(e[0])]
    warmup_batch_jax  = [e for e in warmup_batch_jax  if _want(e[0])]
    warmup_batch_cuda = [e for e in warmup_batch_cuda if _want(e[0])]

    tli = (target_link_index,)

    # Pre-built rng sequences used to warm up the scan timers below.
    _wup_rng_seq = jnp.stack(
        [jax.random.fold_in(rng0, k) for k in range(N_DEVICE_REPEATS)]
    )  # (N_DEVICE_REPEATS, 2)
    _wup_rng_batch_jax = _make_batched_rng_keys_seq(rng_keys_batch)
    _wup_rng_batch_cuda = _wup_rng_seq  # (N_DEVICE_REPEATS, 2)

    # Dicts populated below; consumed when building the solver lists.
    seq_timers:   dict[str, object] = {}
    batch_timers: dict[str, object] = {}

    for name, fn, kwargs in warmup_seq:
        print(f"Warming up {name} ...")
        for _ in range(N_WARMUP):
            out = fn(robot=robot, target_link_indices=tli, target_poses=(target_poses[0],),
                     rng_key=rng0, previous_cfg=mid_cfg,
                     fixed_joint_mask=fixed_joint_mask, **kwargs)
            jax.block_until_ready(out)
        t = _build_seq_ik_timer(fn, robot, tli, fixed_joint_mask, n_act, kwargs)
        jax.block_until_ready(t(target_poses[0].wxyz_xyz, mid_cfg, _wup_rng_seq))
        seq_timers[name] = t

    for name, fn, kwargs in warmup_batch_jax:
        print(f"Warming up {name} ...")
        for _ in range(N_WARMUP):
            out = fn(robot, tli, target_poses_stacked, rng_keys_batch,
                     previous_cfgs_batch, fixed_joint_mask)
            jax.block_until_ready(out)
        t = _build_batch_ik_timer(fn, robot, tli, fixed_joint_mask, kwargs, is_jax_batch=True)
        jax.block_until_ready(
            t(target_poses_stacked.wxyz_xyz, previous_cfgs_batch, _wup_rng_batch_jax)
        )
        batch_timers[name] = t

    for name, fn, kwargs in warmup_batch_cuda:
        print(f"Warming up {name} ...")
        for _ in range(N_WARMUP):
            out = fn(robot=robot, target_link_indices=tli, target_poses=target_poses_stacked,
                     rng_key=rng0, previous_cfgs=previous_cfgs_batch,
                     fixed_joint_mask=fixed_joint_mask, **kwargs)
            jax.block_until_ready(out)
        t = _build_batch_ik_timer(fn, robot, tli, fixed_joint_mask, kwargs, is_jax_batch=False)
        jax.block_until_ready(
            t(target_poses_stacked.wxyz_xyz, previous_cfgs_batch, _wup_rng_batch_cuda)
        )
        batch_timers[name] = t

    for _label, _variant in (("PyRoKi-LS-BATCH", _pyroki_ls),
                            ("PyRoKi-AnalyticJac-BATCH", _pyroki_analytic)):
        if not (_block_wanted("batch") or _block_wanted("batch_coll")):
            continue  # skip the (large) vmapped batch warmup entirely
        if _variant is None:
            continue
        _num_seeds = IK_KWARGS_PYROKI["num_seeds"]
        _lo_j = jnp.array(lo, dtype=jnp.float32)
        _hi_j = jnp.array(hi, dtype=jnp.float32)
        print(f"Warming up {_label} ...")
        _batch_seeds = jax.random.uniform(
            jax.random.PRNGKey(0),
            (len(target_poses_stacked.wxyz_xyz), _num_seeds, n_act),
            minval=_lo_j, maxval=_hi_j,
        )
        for _ in range(N_WARMUP):
            out = _variant[1](target_poses_stacked.wxyz_xyz, _batch_seeds)
            jax.block_until_ready(out)

    # ------------------------------------------------------------------
    # Sequential evaluation (JAX + CUDA single-problem)
    # ------------------------------------------------------------------
    print(f"\n{'─'*80}")
    print("Sequential evaluation (per-problem latency) ...")
    print(f"{'─'*80}")

    seq_solvers = [] if _NO_JAX else [
        ("HJCD-JAX",  jit_hjcd,          {},                seq_timers.get("HJCD-JAX")),
        ("LS-JAX",    jit_ls,            {},                seq_timers.get("LS-JAX")),
        ("SQP-JAX",   jit_sqp,          {},                seq_timers.get("SQP-JAX")),
        ("MPPI-JAX",  jit_mppi,         {},                seq_timers.get("MPPI-JAX")),
    ]
    if not _CPU_ONLY:
        seq_solvers += [
            ("HJCD-CUDA", hjcd_solve_cuda,  IK_KWARGS_HJCD_CUDA,   seq_timers.get("HJCD-CUDA")),
            ("LS-CUDA",   ls_ik_solve_cuda,  IK_KWARGS_LS_CUDA, seq_timers.get("LS-CUDA")),
            ("SQP-CUDA",  sqp_ik_solve_cuda, IK_KWARGS_SQP_CUDA, seq_timers.get("SQP-CUDA")),
            ("MPPI-CUDA", mppi_ik_solve_cuda, IK_KWARGS_MPPI_CUDA, seq_timers.get("MPPI-CUDA")),
        ]
    if _learned_ik_available:
        seq_solvers.append(("Learned-JAX", _learned_ik_fn, IK_KWARGS_LEARNED_JAX,
                            seq_timers.get("Learned-JAX")))
    seq_solvers = [s for s in seq_solvers if _want(s[0])]

    seq_results: dict[str, list[SolveResult]] = {}

    for name, fn, kwargs, timer in seq_solvers:
        print(f"  Running {name} ...")
        seq_results[name] = _run_solver_sequential(
            fn, robot, target_link_index, target_poses,
            fixed_joint_mask, rng_keys, previous_cfgs_seq, kwargs, n_act, timer=timer,
        )

    for _label, _variant in (("PyRoKi-LS", _pyroki_ls),
                            ("PyRoKi-AnalyticJac", _pyroki_analytic)):
        if not _block_wanted("seq"):
            continue
        if _variant is None:
            continue
        print(f"  Running {_label} ...")
        seq_results[_label] = _run_pyroki_sequential(
            _variant[0], robot, _pyroki_robot, target_link_index,
            target_poses, lo, hi, n_act, IK_KWARGS_PYROKI["num_seeds"],
        )

    if _quik_solver is not None:
        print("  Running QuIK-CPU ...")
        seq_results["QuIK-CPU"] = _run_quik_sequential(
            _quik_solver, robot, target_link_index, target_poses,
            lo, hi, _quik_num_seeds,
        )

    # ------------------------------------------------------------------
    # Sequential evaluation — collision-free IK
    # ------------------------------------------------------------------
    seq_coll_results: dict[str, list[SolveResult]] = {}

    if COLLISION_FREE:
        print(f"\n{'─'*80}")
        print("Sequential evaluation — collision-free IK ...")
        print(f"{'─'*80}")

        seq_coll_solvers = [] if _NO_JAX else [
            ("HJCD-JAX",  jit_hjcd_coll,     coll_kwargs_jax,                              seq_timers.get("HJCD-JAX-COLL")),
            ("LS-JAX",    jit_ls_coll,        coll_kwargs_jax,                              seq_timers.get("LS-JAX-COLL")),
            ("SQP-JAX",   jit_sqp_coll,       coll_kwargs_jax,                              seq_timers.get("SQP-JAX-COLL")),
            ("MPPI-JAX",  jit_mppi_coll,      coll_kwargs_jax,                              seq_timers.get("MPPI-JAX-COLL")),
        ]
        if not _CPU_ONLY:
            seq_coll_solvers += [
                ("HJCD-CUDA", hjcd_solve_cuda,    {**IK_KWARGS_HJCD_CUDA, **coll_kwargs_cuda},  seq_timers.get("HJCD-CUDA-COLL")),
                ("LS-CUDA",   ls_ik_solve_cuda,   {**IK_KWARGS_LS_CUDA, **coll_kwargs_ls_cuda_kernel}, seq_timers.get("LS-CUDA-COLL")),
                ("SQP-CUDA",  sqp_ik_solve_cuda,  {**IK_KWARGS_SQP_CUDA, **coll_kwargs_cuda},   seq_timers.get("SQP-CUDA-COLL")),
                ("MPPI-CUDA", mppi_ik_solve_cuda, {**IK_KWARGS_MPPI_CUDA, **coll_kwargs_cuda},  seq_timers.get("MPPI-CUDA-COLL")),
            ]
        seq_coll_solvers = [s for s in seq_coll_solvers if _want(s[0])]

        for name, fn, kwargs, timer in seq_coll_solvers:
            print(f"  Running {name}-COLL ...")
            seq_coll_results[name] = _run_solver_sequential(
                fn, robot, target_link_index, target_poses,
                fixed_joint_mask, rng_keys, previous_cfgs_seq, kwargs, n_act, timer=timer,
            )

        for _label, _variant in (("PyRoKi-LS", _pyroki_ls_coll),
                                ("PyRoKi-AnalyticJac", _pyroki_analytic_coll)):
            if not _block_wanted("seq_coll"):
                continue
            if _variant is None:
                continue
            print(f"  Running {_label}-COLL ...")
            seq_coll_results[_label] = _run_pyroki_sequential(
                _variant[0], robot, _pyroki_robot, target_link_index,
                target_poses, lo, hi, n_act, IK_KWARGS_PYROKI["num_seeds"],
            )

        if _quik_vamp_checker is not None:
            print("  Running QuIK-CPU-COLL ...")
            seq_coll_results["QuIK-CPU"] = _run_quik_sequential(
                _quik_solver, robot, target_link_index, target_poses,
                lo, hi, _quik_num_seeds, vamp_checker=_quik_vamp_checker,
            )

    # ------------------------------------------------------------------
    # Batch evaluation (JAX + CUDA batch solvers)
    # ------------------------------------------------------------------
    print(f"\n{'─'*80}")
    print("Batch evaluation (all targets in one kernel launch) ...")
    print(f"{'─'*80}")

    batch_solvers = [] if _NO_JAX else [
        ("LS-JAX-BATCH",    jit_ls_batch,            {},                 rng_keys_batch, batch_timers.get("LS-JAX-BATCH")),
        ("HJCD-JAX-BATCH",  jit_hjcd_batch,           {},                 rng_keys_batch, batch_timers.get("HJCD-JAX-BATCH")),
        ("SQP-JAX-BATCH",   jit_sqp_batch,            {},                 rng_keys_batch, batch_timers.get("SQP-JAX-BATCH")),
        ("MPPI-JAX-BATCH",  jit_mppi_batch,           {},                 rng_keys_batch, batch_timers.get("MPPI-JAX-BATCH")),
    ]
    if not _CPU_ONLY:
        batch_solvers += [
            ("LS-CUDA-BATCH",   ls_ik_solve_cuda_batch,   IK_KWARGS_LS_CUDA,  rng0,           batch_timers.get("LS-CUDA-BATCH")),
            ("HJCD-CUDA-BATCH", hjcd_solve_cuda_batch,    IK_KWARGS_HJCD_CUDA, rng0,          batch_timers.get("HJCD-CUDA-BATCH")),
            ("SQP-CUDA-BATCH",  sqp_ik_solve_cuda_batch,  IK_KWARGS_SQP_CUDA,  rng0,         batch_timers.get("SQP-CUDA-BATCH")),
            ("MPPI-CUDA-BATCH", mppi_ik_solve_cuda_batch, IK_KWARGS_MPPI_CUDA, rng0,         batch_timers.get("MPPI-CUDA-BATCH")),
        ]
    if _learned_ik_available:
        batch_solvers.append(("Learned-JAX-BATCH", _learned_ik_fn_batch, {}, rng_keys_batch,
                              batch_timers.get("Learned-JAX-BATCH")))
    batch_solvers = [s for s in batch_solvers if _want(s[0])]

    batch_results: dict[str, BatchResult] = {}

    for name, fn, kwargs, rng, timer in batch_solvers:
        print(f"  Running {name} ...")
        batch_results[name] = _run_solver_batch(
            fn, robot, target_link_index, target_poses_stacked,
            fixed_joint_mask, rng, previous_cfgs_batch, kwargs,
            is_jax_batch=jnp.asarray(rng).ndim == 2, timer=timer,
        )

    for _label, _variant in (("PyRoKi-LS-BATCH", _pyroki_ls),
                            ("PyRoKi-AnalyticJac-BATCH", _pyroki_analytic)):
        if not _block_wanted("batch"):
            continue
        if _variant is None:
            continue
        print(f"  Running {_label} ...")
        batch_results[_label] = _run_pyroki_batch(
            _variant[1], robot, _pyroki_robot, target_link_index,
            target_poses_stacked, lo, hi, n_act, IK_KWARGS_PYROKI["num_seeds"],
        )

    if _quik_solver is not None:
        print("  Running QuIK-CPU-BATCH ...")
        batch_results["QuIK-CPU-BATCH"] = _run_quik_batch(
            _quik_solver, robot, target_link_index, target_poses_stacked,
            lo, hi, _quik_num_seeds,
        )

    # ------------------------------------------------------------------
    # Batch evaluation — collision-free IK
    # ------------------------------------------------------------------
    batch_coll_results: dict[str, BatchResult] = {}

    if COLLISION_FREE:
        print(f"\n{'─'*80}")
        print("Batch evaluation — collision-free IK ...")
        print(f"{'─'*80}")

        batch_coll_solvers = [] if _NO_JAX else [
            ("LS-JAX",    jit_ls_coll_batch,      {},                                          rng_keys_batch, batch_timers.get("LS-JAX-COLL-BATCH")),
            ("HJCD-JAX",  jit_hjcd_coll_batch,    {},                                          rng_keys_batch, batch_timers.get("HJCD-JAX-COLL-BATCH")),
            ("SQP-JAX",   jit_sqp_coll_batch,     {},                                          rng_keys_batch, batch_timers.get("SQP-JAX-COLL-BATCH")),
            ("MPPI-JAX",  jit_mppi_coll_batch,    {},                                          rng_keys_batch, batch_timers.get("MPPI-JAX-COLL-BATCH")),
        ]
        if not _CPU_ONLY:
            batch_coll_solvers += [
                ("LS-CUDA",   ls_ik_solve_cuda_batch,  {**IK_KWARGS_LS_CUDA,   **coll_kwargs_ls_cuda_kernel}, rng0, batch_timers.get("LS-CUDA-COLL-BATCH")),
                ("HJCD-CUDA", hjcd_solve_cuda_batch,   {**IK_KWARGS_HJCD_CUDA, **coll_kwargs_cuda}, rng0, batch_timers.get("HJCD-CUDA-COLL-BATCH")),
                ("SQP-CUDA",  sqp_ik_solve_cuda_batch, {**IK_KWARGS_SQP_CUDA,  **coll_kwargs_cuda}, rng0, batch_timers.get("SQP-CUDA-COLL-BATCH")),
                ("MPPI-CUDA", mppi_ik_solve_cuda_batch,{**IK_KWARGS_MPPI_CUDA, **coll_kwargs_cuda}, rng0, batch_timers.get("MPPI-CUDA-COLL-BATCH")),
            ]
        batch_coll_solvers = [s for s in batch_coll_solvers if _want(s[0])]

        for name, fn, kwargs, rng, timer in batch_coll_solvers:
            print(f"  Running {name}-COLL-BATCH ...")
            batch_coll_results[name] = _run_solver_batch(
                fn, robot, target_link_index, target_poses_stacked,
                fixed_joint_mask, rng, previous_cfgs_batch, kwargs,
                is_jax_batch=jnp.asarray(rng).ndim == 2, timer=timer,
            )

        for _label, _variant in (("PyRoKi-LS", _pyroki_ls_coll),
                                ("PyRoKi-AnalyticJac", _pyroki_analytic_coll)):
            if not _block_wanted("batch_coll"):
                continue
            if _variant is None:
                continue
            print(f"  Running {_label}-COLL-BATCH ...")
            batch_coll_results[_label] = _run_pyroki_batch(
                _variant[1], robot, _pyroki_robot, target_link_index,
                target_poses_stacked, lo, hi, n_act, IK_KWARGS_PYROKI["num_seeds"],
            )

        if _quik_vamp_checker is not None:
            print("  Running QuIK-CPU-COLL-BATCH ...")
            batch_coll_results["QuIK-CPU"] = _run_quik_batch(
                _quik_solver, robot, target_link_index, target_poses_stacked,
                lo, hi, _quik_num_seeds, vamp_checker=_quik_vamp_checker,
            )

    # ------------------------------------------------------------------
    # Results tables
    # ------------------------------------------------------------------
    seq_cols   = ["t_med(ms)", "t_p95(ms)", "pos_med(mm)", "pos_p95(mm)",
                  "rot_med(rad)", "rot_p95(rad)", "success"]
    batch_cols = ["ms/prob",   "pos_med(mm)", "pos_p95(mm)",
                  "rot_med(rad)", "rot_p95(rad)", "success",
                  "gpu_pk(%)", "gpu_avg(%)", "vram_pk(MB)"]

    seq_coll_cols   = seq_cols   + ["coll_free"]
    batch_coll_cols = ["ms/prob",   "pos_med(mm)", "pos_p95(mm)",
                       "rot_med(rad)", "rot_p95(rad)", "success", "coll_free",
                       "gpu_pk(%)", "gpu_avg(%)", "vram_pk(MB)"]

    _method_pairs = [
        ("HJCD-JAX", "HJCD-CUDA"),
        ("LS-JAX",   "LS-CUDA"),
        ("SQP-JAX",  "SQP-CUDA"),
        ("MPPI-JAX", "MPPI-CUDA"),
    ]

    def _method_order(jax_batch_suffix: str = "") -> list[str]:
        order = []
        for jax_label, cuda_label in _method_pairs:
            if not _NO_JAX:
                order.append(jax_label + jax_batch_suffix)
            if not _CPU_ONLY:
                order.append(cuda_label + jax_batch_suffix)
        return order

    seq_order = _method_order()
    if _learned_ik_available:
        seq_order.append("Learned-JAX")
    if _pyroki_ls is not None:
        seq_order.append("PyRoKi-LS")
    if _pyroki_analytic is not None:
        seq_order.append("PyRoKi-AnalyticJac")
    if "QuIK-CPU" in seq_results:
        seq_order.append("QuIK-CPU")

    batch_order = _method_order("-BATCH")
    if _learned_ik_available:
        batch_order.append("Learned-JAX-BATCH")
    if _pyroki_ls is not None:
        batch_order.append("PyRoKi-LS-BATCH")
    if _pyroki_analytic is not None:
        batch_order.append("PyRoKi-AnalyticJac-BATCH")
    if "QuIK-CPU-BATCH" in batch_results:
        batch_order.append("QuIK-CPU-BATCH")

    coll_seq_order = _method_order()
    if _pyroki_ls_coll is not None:
        coll_seq_order.append("PyRoKi-LS")
    if _pyroki_analytic_coll is not None:
        coll_seq_order.append("PyRoKi-AnalyticJac")
    if "QuIK-CPU" in seq_coll_results:
        coll_seq_order.append("QuIK-CPU")
    coll_batch_order = _method_order()
    if _pyroki_ls_coll is not None:
        coll_batch_order.append("PyRoKi-LS")
    if _pyroki_analytic_coll is not None:
        coll_batch_order.append("PyRoKi-AnalyticJac")
    if "QuIK-CPU" in batch_coll_results:
        coll_batch_order.append("QuIK-CPU")

    # In the per-solver isolation path only one solver's results exist; keep the
    # canonical ordering but drop labels this process did not run.
    seq_order        = [l for l in seq_order        if l in seq_results]
    batch_order      = [l for l in batch_order      if l in batch_results]
    coll_seq_order   = [l for l in coll_seq_order   if l in seq_coll_results]
    coll_batch_order = [l for l in coll_batch_order if l in batch_coll_results]

    print(f"\n{'='*80}")
    print(f"Sequential results — per-problem latency  (N={N_TARGETS}, timed={N_TIMED})")
    print(f"{'='*80}")
    print(_table_header(seq_cols))
    print(_table_sep(len(seq_cols)))
    for label in seq_order:
        row, _ = _seq_row(label, seq_results[label])
        print(row)

    print(f"\n{'='*80}")
    print(f"Batch results — effective per-problem time  (N={N_TARGETS_BATCH}, timed={N_TIMED})")
    print(f"{'='*80}")
    print(_table_header(batch_cols))
    print(_table_sep(len(batch_cols)))
    for label in batch_order:
        row, _ = _batch_row(label, batch_results[label])
        print(row)

    # NOTE: the JAX-vs-CUDA batch agreement check was removed with the switch to
    # per-solver isolation — a solver's JAX and CUDA backends now run in separate
    # processes, so no single process holds both error arrays to compare.

    if COLLISION_FREE:
        # Compute coll_free counts for sequential results.
        seq_coll_free: dict[str, int] = {}
        for name, results in seq_coll_results.items():
            count = sum(
                float(_check_coll_jit(jnp.array(r.cfg))) > 0
                for r in results
            )
            seq_coll_free[name] = count

        # Compute coll_free counts for batch results.
        batch_coll_free: dict[str, int] = {}
        for name, result in batch_coll_results.items():
            dists = np.array(_check_coll_batch_jit(jnp.array(result.cfgs)))
            batch_coll_free[name] = int(np.sum(dists > 0))

        print(f"\n{'='*80}")
        print(f"Sequential results — COLLISION-FREE IK  (N={N_TARGETS}, timed={N_TIMED})")
        print(f"  Scene: {env_file}")
        print(f"  Obstacles: {len(env_dict.get('spheres', []))} spheres"
              f" + {len(env_dict.get('cuboids', []))} cuboids")
        print(f"  coll_free: solutions with min signed dist > 0 (all links clear of all obstacles)")
        print(f"{'='*80}")
        print(_table_header(seq_coll_cols))
        print(_table_sep(len(seq_coll_cols)))
        for label in coll_seq_order:
            row, _ = _seq_row_coll(label, seq_coll_results[label], seq_coll_free[label])
            print(row)

        print(f"\n{'='*80}")
        print(f"Batch results — COLLISION-FREE IK  (N={N_TARGETS_BATCH}, timed={N_TIMED})")
        print(f"{'='*80}")
        print(_table_header(batch_coll_cols))
        print(_table_sep(len(batch_coll_cols)))
        for label in coll_batch_order:
            row, _ = _batch_row_coll(label, batch_coll_results[label], batch_coll_free[label])
            print(row)

    # ------------------------------------------------------------------
    # CSV output
    # ------------------------------------------------------------------
    if csv_file is not None:
        ts = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        _write_csv(
            csv_file, ts, robot_name,
            seq_results, batch_results,
            seq_coll_results, batch_coll_results,
            seq_coll_free if COLLISION_FREE else {},
            batch_coll_free if COLLISION_FREE else {},
        )
        print(f"\nResults appended to {csv_file}")

    print()


def _curobo_python_cmd() -> list[str] | None:
    """Locate a python interpreter with cuRobo installed.

    Precedence: the CUROBO_PYTHON env var, a sibling ``curobo`` conda env next
    to the active one (i.e. ``<envs>/curobo/bin/python``), then
    ``conda run -n curobo``.  Returns None if no candidate is found.
    """
    override = os.environ.get("CUROBO_PYTHON")
    if override:
        if pathlib.Path(override).is_file():
            return [override]
        print(f"  warning: CUROBO_PYTHON={override} not found; trying other candidates")
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        envs_dir = pathlib.Path(prefix).resolve().parent
    else:
        # Not inside a conda env: infer <envs>/ from sys.executable
        # (<envs>/<env>/bin/python).
        envs_dir = pathlib.Path(sys.executable).resolve().parent.parent.parent
    sibling = envs_dir / "curobo" / "bin" / "python"
    if sibling.is_file():
        return [str(sibling)]
    if shutil.which("conda") is not None:
        return ["conda", "run", "--no-capture-output", "-n", "curobo"]
    return None


def _run_solver_subprocess(
    robot_name: str, solver: str, csv_file: pathlib.Path, args: argparse.Namespace,
) -> None:
    """Re-invoke this script to benchmark ONE solver on ONE robot, in isolation.

    Each solver gets a fresh process so nothing it does — JAX preallocation, the
    process-global GLASS tier cache, JIT/kernel compilation, allocator pool state —
    can perturb another solver's timings. For CUDA solvers the robot's GLASS tier
    (ROBOT_TIER) is pinned via PYROFFI_IK_TIER, which is read once on first kernel
    launch; it is harmless (ignored) for JAX/CPU solvers.

    cuRobo is the exception: it runs in its OWN conda env via
    bench_ik_curobo.py, consuming the target sidecar this script wrote during
    target generation (see _run_robot_benchmark).
    """
    if solver == "cuRobo":
        prefix = _curobo_python_cmd()
        if prefix is None:
            print(
                f"\n=== Skipping {robot_name} / cuRobo: no cuRobo interpreter found ==="
                "\n  Set CUROBO_PYTHON to the env's python, create a sibling 'curobo'"
                "\n  conda env, or make 'conda' available — with an editable install"
                "\n  of baselines/curobo."
            )
            return
        targets_npz = csv_file.parent / f"bench_ik_targets_{robot_name}.npz"
        cmd = prefix + [
            str(pathlib.Path(__file__).with_name("bench_ik_curobo.py")),
            "--robot", robot_name,
            "--targets", str(targets_npz),
            "--outdir", str(csv_file.parent),
            "--env-file", str(ENV_FILE),
        ]
        print(f"\n=== Running {robot_name} / cuRobo in subprocess ({' '.join(prefix)}) ===")
        env = os.environ.copy()
        env.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
        subprocess.run(cmd, env=env, check=True)
        return

    tier = ROBOT_TIER.get(robot_name)
    tier_note = f", PYROFFI_IK_TIER={tier}" if tier is not None else ""
    print(f"\n=== Running {robot_name} / {solver} in subprocess{tier_note} ===")

    cmd = [sys.executable, __file__, "--robot", robot_name, "--solver", solver]
    if args.outdir is not None:
        cmd += ["--outdir", str(args.outdir)]
    if args.no_jax:
        cmd += ["--no-jax"]
    if args.cpu_only:
        cmd += ["--cpu-only"]

    env = os.environ.copy()
    if tier is not None:
        env["PYROFFI_IK_TIER"] = tier
    # Children benchmark a real solver, so they use JAX's default (preallocating)
    # allocator and get the whole card to themselves — clear the dispatcher's
    # no-preallocate override that env.copy() would otherwise inherit.
    env.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
    subprocess.run(cmd, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="IK benchmark with multi-robot support")
    parser.add_argument(
        "--disable-robot",
        action="append",
        choices=ROBOT_NAMES,
        default=[],
        metavar="ROBOT",
        help=(
            "Disable a robot benchmark (repeatable). "
            "Example: --disable-robot panda --disable-robot fetch"
        ),
    )
    parser.add_argument(
        "--outdir",
        type=pathlib.Path,
        default=None,
        help=(
            "Directory to write the CSV results file into. "
            "Defaults to the directory of CSV_FILE (resources/)."
        ),
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help=(
            "Run only CPU-native IK solvers (QuIK and other non-JAX integrations): "
            "skip all CUDA solvers AND all JAX-based solvers (HJCD/LS/SQP/MPPI-JAX, "
            "Learned-JAX, PyRoKi), force JAX_PLATFORMS=cpu, and disable GPU (pynvml) "
            "monitoring. This flag is parsed before jax is imported (see top of file)."
        ),
    )
    parser.add_argument(
        "--no-jax",
        action="store_true",
        help=(
            "Skip all JAX-based solvers (HJCD/LS/SQP/MPPI-JAX, Learned-JAX, "
            "PyRoKi) so only the CUDA/FFI kernel solvers (and QuIK) run, to "
            "benchmark the FFI kernels in isolation. Unlike --cpu-only, CUDA "
            "solvers and GPU (pynvml) monitoring stay enabled. This flag is "
            "parsed before jax is imported (see top of file)."
        ),
    )
    # The next two flags mark a per-solver child process (see _run_solver_subprocess).
    # A run WITHOUT --solver is the dispatcher: it spawns one child per (robot, solver).
    parser.add_argument(
        "--robot",
        choices=ROBOT_NAMES,
        default=None,
        help="Benchmark exactly this one robot (per-solver child process; internal).",
    )
    parser.add_argument(
        "--solver",
        default=None,
        metavar="LABEL",
        help=(
            "Benchmark exactly this one solver, e.g. HJCD-CUDA, LS-JAX, QuIK-CPU "
            "(per-solver child process; internal). Runs it in isolation and appends "
            "only its rows to the CSV."
        ),
    )
    parser.add_argument(
        "--blocks",
        default=None,
        metavar="LIST",
        help=(
            "Comma-separated subset of evaluation blocks to run: "
            "{seq,seq_coll,batch,batch_coll}. Default runs all four. "
            "Useful to skip the batch blocks (which build the large vmapped "
            "kernel) when only per-problem latency is wanted."
        ),
    )
    args = parser.parse_args()

    csv_file = (args.outdir / CSV_FILE.name) if args.outdir is not None else CSV_FILE

    # Child path: one solver, one robot, in this process (no further dispatch).
    if args.solver is not None:
        if args.robot is None:
            raise SystemExit("--solver requires --robot (internal child invocation).")
        _blocks = set(args.blocks.split(",")) if args.blocks else None
        _run_robot_benchmark(args.robot, csv_file, solver_filter=args.solver, blocks=_blocks)
        return

    # Dispatcher path: fan out one isolated subprocess per (robot, solver).
    disabled = set(args.disable_robot)
    selected = [name for name in ROBOT_NAMES if name not in disabled]
    if not selected:
        raise SystemExit("No robots selected. Re-enable at least one robot.")

    # --cpu-only implies --no-jax (mirrors _NO_JAX at the top of the file).
    solvers = _candidate_solvers(args.cpu_only, args.no_jax or args.cpu_only)
    print("Selected robots:", ", ".join(selected))
    print("Solvers (one isolated subprocess each):", ", ".join(solvers))
    for robot_name in selected:
        for solver in solvers:
            _run_solver_subprocess(robot_name, solver, csv_file, args)


if __name__ == "__main__":
    main()
