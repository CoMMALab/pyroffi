"""Region-constrained IK sampling via CUDA.

This module provides four region-IK samplers:

  brownian_motion_sample_box_region_cuda
    Two-phase per-seed strategy: Levenberg-Marquardt boundary reach, then
    null-space Brownian shuffling with periodic FK-check corrections.  Best
    for dense, well-distributed coverage of a single region.

  svgd_sample_box_region_cuda
    Particle transport via Stein Variational Gradient Descent with RBF-kernel
    repulsion.  Tends to spread particles more uniformly than random sampling
    and avoids boundary clustering.

  hit_and_run_sample_box_region_cuda
    Two-phase per-seed strategy: LM boundary reach, then Markov-chain
    hit-and-run Gaussian perturbations in joint space.  Lighter per-step cost
    than Brownian motion; good for exploration when the box is large.

  direct_sample_box_region_cuda
    Sample target points uniformly in cartesian space inside the box(es),
    then run multi-seed Levenberg-Marquardt IK directly to each sampled
    target.  No null-space exploration — coverage comes purely from the
    cartesian-space sampling of the targets.  Cheapest per sample when the
    box is reachable everywhere; uses the existing ``ls_ik_cuda`` kernel.

Batched box queries
-------------------
All samplers accept ``box_min`` / ``box_max`` either as

  * shape ``(3,)``         → single box; returns ``(num_samples, ...)``
  * shape ``(n_boxes, 3)`` → one box per row; returns ``(n_boxes, num_samples, ...)``

Seeds in each kernel launch are distributed round-robin across boxes so every
region is explored in parallel on the GPU.  Each box independently accumulates
``num_samples`` valid configurations before the call returns.

Example — two boxes in one call::

    cfgs, ee, tgt, err = direct_sample_box_region_cuda(
        robot, ee_link, rng_key, prev_cfg,
        box_min=jnp.array([[0.3, -0.2, 0.1], [0.3, 0.1, 0.3]]),
        box_max=jnp.array([[0.5,  0.0, 0.3], [0.5, 0.3, 0.5]]),
        num_samples=512,
    )
    # cfgs.shape == (2, 512, n_act)
    # ee.shape   == (2, 512, 3)

References:
  - Liu, Qiang, and Dilin Wang. "Stein variational gradient descent."
    Advances in neural information processing systems 30 (2016).
"""

from __future__ import annotations

import functools
import warnings
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float

from .._robot import Robot
from ._ls_ik import _prepare_ls_collision_buffers

_REGION_IK_MAX_TPB_BY_SMEM = 384  # MAX_JOINTS=64, MAX_ACT=16 build of CUDA kernel.
_HIT_AND_RUN_MAX_TPB_BY_SMEM = 384


# ── Shared helpers ───────────────────────────────────────────────────────────


def _validate_threads_per_block(
    threads_per_block: int, max_tpb: int = _REGION_IK_MAX_TPB_BY_SMEM
) -> None:
    if threads_per_block < 32 or threads_per_block > 1024 or threads_per_block % 32 != 0:
        raise ValueError("threads_per_block must be a multiple of 32 in [32, 1024].")
    if threads_per_block > max_tpb:
        raise ValueError(
            f"threads_per_block={threads_per_block} exceeds the shared-memory "
            f"budget for this kernel; use <= {max_tpb}."
        )


def _compute_ancestor_mask(robot: Robot, target_link_index: int) -> tuple[int, Array]:
    parent_joint_indices = np.asarray(robot.links.parent_joint_indices, dtype=np.int32)
    parent_idx = np.asarray(robot.joints.parent_indices, dtype=np.int32)
    n_joints = int(robot.joints.num_joints)

    target_jnt = int(parent_joint_indices[target_link_index])
    if target_jnt < 0:
        raise ValueError(
            f"Target link index {target_link_index} maps to root/base (no parent joint)."
        )

    mask = np.zeros((n_joints,), dtype=np.int32)
    j = target_jnt
    while j >= 0:
        mask[j] = 1
        j = int(parent_idx[j])

    return target_jnt, jnp.array(mask, dtype=jnp.int32)


def _seeds_per_launch_budget(
    n_act: int,
    desired: int,
    memory_limit_gb: float,
    restarts_per_target: int,
) -> int:
    bytes_per_seed = max(4096, 4 * (3 * n_act + n_act + 3 + 3 + 1 + 32))
    bytes_per_target = bytes_per_seed * max(1, int(restarts_per_target))
    budget = int((memory_limit_gb * (1024**3)) // bytes_per_target)
    budget = max(1, budget)
    return max(1, min(desired, budget))


def _box_entropy(
    ee_points: np.ndarray,
    box_min: np.ndarray,
    box_max: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Shannon entropy (nats) of the EE-point distribution discretized inside the box."""
    span = box_max - box_min
    normalized = (ee_points - box_min) / np.maximum(span, 1e-12)
    normalized = np.clip(normalized, 0.0, 1.0)
    hist, _ = np.histogramdd(normalized, bins=n_bins, range=[[0.0, 1.0]] * 3)
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist[hist > 0] / total
    return float(-np.sum(p * np.log(p)))


def _normalise_boxes(
    box_min: Array | np.ndarray,
    box_max: Array | np.ndarray,
) -> tuple[Array, Array, bool]:
    """Coerce ``(3,)`` or ``(n_boxes, 3)`` inputs to ``(n_boxes, 3)`` and validate."""
    box_min = jnp.asarray(box_min, dtype=jnp.float32)
    box_max = jnp.asarray(box_max, dtype=jnp.float32)
    batched_input = box_min.ndim == 2
    if not batched_input:
        box_min = box_min[None, :]
        box_max = box_max[None, :]
    if box_min.shape != box_max.shape or box_min.shape[-1] != 3:
        raise ValueError(
            f"box_min/box_max must have matching shape (3,) or (n_boxes, 3); "
            f"got {box_min.shape} and {box_max.shape}."
        )
    if not bool(jnp.all(box_max > box_min)):
        raise ValueError("box_max must be strictly greater than box_min for all axes.")
    return box_min, box_max, batched_input


def _generate_seeds(
    *,
    rng_key: Array,
    carry_cfg: Array,
    n_act: int,
    n_total: int,
    n_restarts: int,
    lower: Array,
    upper: Array,
    fixed_mask: Array,
) -> Array:
    """Generate ``n_total = n_batch * n_restarts`` seeds (1/4 warm, 3/4 random)."""
    n_warm = max(1, n_total // 4)
    n_rand = n_total - n_warm

    key_warm, key_rand = jax.random.split(rng_key)
    warm = jnp.clip(
        carry_cfg[None, :] + jax.random.normal(key_warm, (n_warm, n_act)) * 0.05,
        lower, upper,
    )
    rand = jax.random.uniform(key_rand, (n_rand, n_act), minval=lower, maxval=upper)
    seeds_flat = jnp.concatenate([warm, rand], axis=0)
    seeds_flat = jnp.where(fixed_mask[None, :], carry_cfg[None, :], seeds_flat)
    return seeds_flat.reshape(-1, n_restarts, n_act)


def _run_region_sampler_loop(
    *,
    n_act: int,
    lower: Array,
    upper: Array,
    fixed_mask: Array,
    previous_cfg: Array,
    rng_key: Array,
    box_min: Array,        # (n_boxes, 3)
    box_max: Array,        # (n_boxes, 3)
    batched_input: bool,
    num_samples: int,
    seeds_per_launch: int,
    restarts_per_target: int,
    max_batches: int | None,
    target_entropy: float | None,
    entropy_bins: int,
    verbose: bool,
    sample_init_points: bool,
    step_fn: Callable[..., tuple[Array, Array, Array, Array, Array]],
    increase_hint: str,
) -> tuple[Array, Array, Array, Array]:
    """Shared multi-region collection loop used by every region-IK sampler.

    ``step_fn`` is called as
        ``step_fn(seeds, init_points, box_mins_pp, box_maxs_pp, rng_seed)``
    and must return ``(best_cfgs, best_ee, best_targets, best_errs, inside)``
    each with leading axis ``n_problems = seeds.shape[0]``.  ``init_points``
    is ``None`` when ``sample_init_points=False`` (e.g. hit-and-run, which
    samples its own targets internally from the per-problem box bounds).
    """
    n_boxes = int(box_min.shape[0])
    box_min_np = np.asarray(box_min)
    box_max_np = np.asarray(box_max)

    cfg_chunks: list[list[Array]] = [[] for _ in range(n_boxes)]
    ee_chunks: list[list[Array]] = [[] for _ in range(n_boxes)]
    tgt_chunks: list[list[Array]] = [[] for _ in range(n_boxes)]
    err_chunks: list[list[Array]] = [[] for _ in range(n_boxes)]
    collected = [0] * n_boxes
    entropy_done = [False] * n_boxes

    if max_batches is None:
        max_batches = max(8, 8 * int(np.ceil(num_samples * n_boxes / seeds_per_launch)))

    carry_cfg = previous_cfg
    key = rng_key
    box_idx = jnp.arange(seeds_per_launch, dtype=jnp.int32) % n_boxes
    attempts = 0

    def _all_done() -> bool:
        if target_entropy is not None:
            return all(entropy_done)
        return all(c >= num_samples for c in collected)

    while not _all_done() and attempts < max_batches:
        attempts += 1
        n_batch = seeds_per_launch

        if verbose:
            import time as _time
            _t_batch = _time.perf_counter()

        key, key_seeds, key_pts, key_seed = jax.random.split(key, 4)

        seeds = _generate_seeds(
            rng_key=key_seeds,
            carry_cfg=carry_cfg,
            n_act=n_act,
            n_total=n_batch * restarts_per_target,
            n_restarts=restarts_per_target,
            lower=lower,
            upper=upper,
            fixed_mask=fixed_mask,
        )

        # Round-robin per-problem box bounds.
        box_mins_pp = box_min[box_idx]   # (n_batch, 3)
        box_maxs_pp = box_max[box_idx]   # (n_batch, 3)

        if sample_init_points:
            uniform = jax.random.uniform(
                key_pts, (n_batch, 3), minval=0.0, maxval=1.0, dtype=jnp.float32,
            )
            init_points_base = box_mins_pp + uniform * (box_maxs_pp - box_mins_pp)
            init_points = jnp.repeat(init_points_base[:, None, :], restarts_per_target, axis=1)
        else:
            init_points = None

        rng_seed_val = jax.random.randint(
            key_seed, (), minval=1, maxval=np.iinfo(np.int32).max, dtype=jnp.int32,
        )

        try:
            best_cfgs, best_ee, best_targets, best_errs, inside = step_fn(
                seeds=seeds,
                init_points=init_points,
                box_mins_pp=box_mins_pp,
                box_maxs_pp=box_maxs_pp,
                rng_seed=rng_seed_val,
            )
        except Exception as exc:  # pragma: no cover - runtime/environment dependent
            if "out of memory" in str(exc).lower() and seeds_per_launch > 1:
                seeds_per_launch = max(1, seeds_per_launch // 2)
                box_idx = jnp.arange(seeds_per_launch, dtype=jnp.int32) % n_boxes
                continue
            raise

        any_valid = False
        for b in range(n_boxes):
            box_b_mask = (box_idx == b) & inside
            valid_b = int(jnp.sum(box_b_mask))
            if valid_b > 0:
                cfg_chunks[b].append(best_cfgs[box_b_mask])
                ee_chunks[b].append(best_ee[box_b_mask])
                tgt_chunks[b].append(best_targets[box_b_mask])
                err_chunks[b].append(best_errs[box_b_mask])
                collected[b] += valid_b
                carry_cfg = best_cfgs[box_b_mask][-1]
                any_valid = True

        if not any_valid:
            carry_cfg = best_cfgs[int(jnp.argmin(best_errs))]

        if verbose:
            _batch_ms = (_time.perf_counter() - _t_batch) * 1000.0
            total_valid = sum(int(jnp.sum((box_idx == b) & inside)) for b in range(n_boxes))
            print(
                f"  batch {attempts:3d}: {total_valid:4d}/{n_batch} valid "
                f"({total_valid / n_batch * 100:.1f}%), "
                f"collected {collected}, "
                f"{_batch_ms:.1f} ms"
            )

        if target_entropy is not None:
            for b in range(n_boxes):
                if not entropy_done[b] and len(ee_chunks[b]) > 0:
                    ee_so_far = np.concatenate([np.asarray(c) for c in ee_chunks[b]], axis=0)
                    if _box_entropy(ee_so_far, box_min_np[b], box_max_np[b], entropy_bins) >= target_entropy:
                        entropy_done[b] = True

    if target_entropy is None:
        for b in range(n_boxes):
            if collected[b] < num_samples:
                warnings.warn(
                    f"Box {b}: unable to collect enough in-box IK samples. "
                    f"Collected {collected[b]}/{num_samples} after {attempts} batches. "
                    f"Try increasing {increase_hint} or widening the box.",
                    stacklevel=3,
                )
        if all(len(cc) == 0 for cc in cfg_chunks):
            raise RuntimeError("No valid in-box samples collected for any box.")

    if batched_input:
        # When max_batches caps short before every box collected num_samples,
        # truncate every box to the smallest collected count so the (n_boxes,
        # n_samples_per_box, ...) stack is rectangular.  Already-warned above.
        n_per_box = min(num_samples, *(c if c > 0 else 0 for c in collected))
        if n_per_box == 0:
            raise RuntimeError(
                "At least one box collected 0 in-box samples; cannot return a "
                "rectangular (n_boxes, ...) result."
            )

        def _stack_to(chunks: list[Array], n: int) -> Array:
            return jnp.concatenate(chunks, axis=0)[:n]

        cfg_all = jnp.stack([_stack_to(cfg_chunks[b], n_per_box) for b in range(n_boxes)])
        ee_all  = jnp.stack([_stack_to(ee_chunks[b],  n_per_box) for b in range(n_boxes)])
        tgt_all = jnp.stack([_stack_to(tgt_chunks[b], n_per_box) for b in range(n_boxes)])
        err_all = jnp.stack([_stack_to(err_chunks[b], n_per_box) for b in range(n_boxes)])
    else:
        def _stack(chunks: list[Array]) -> Array:
            return jnp.concatenate(chunks, axis=0)[:num_samples]

        cfg_all = _stack(cfg_chunks[0])
        ee_all  = _stack(ee_chunks[0])
        tgt_all = _stack(tgt_chunks[0])
        err_all = _stack(err_chunks[0])

    return cfg_all, ee_all, tgt_all, err_all


# ── Brownian-motion JIT step ─────────────────────────────────────────────────


@functools.partial(
    jax.jit,
    static_argnames=(
        "target_jnt",
        "max_iter",
        "pos_weight",
        "ori_weight",
        "lambda_init",
        "eps_pos",
        "noise_std",
        "n_brownian_steps",
        "fk_check_freq",
        "threads_per_block",
        "enable_collision",
        "collision_weight",
        "collision_margin",
    ),
)
def _brownian_motion_batch_select_jit(
    seeds: Array,
    init_points: Array,
    twists: Array,
    parent_tf: Array,
    parent_idx: Array,
    act_idx: Array,
    mimic_mul: Array,
    mimic_off: Array,
    mimic_act_idx: Array,
    topo_inv: Array,
    ancestor_mask: Array,
    box_mins: Array,
    box_maxs: Array,
    robot_spheres_local: Array,
    robot_sphere_joint_idx: Array,
    world_spheres: Array,
    world_capsules: Array,
    world_boxes: Array,
    world_halfspaces: Array,
    self_pair_i: Array,
    self_pair_j: Array,
    lower: Array,
    upper: Array,
    fixed_mask: Array,
    rng_seed: Array,
    *,
    target_jnt: int,
    max_iter: int,
    pos_weight: float,
    ori_weight: float,
    lambda_init: float,
    eps_pos: float,
    noise_std: float,
    n_brownian_steps: int,
    fk_check_freq: int,
    threads_per_block: int = 128,
    enable_collision: bool = False,
    collision_weight: float = 0.0,
    collision_margin: float = 0.02,
) -> tuple[Array, Array, Array, Array, Array]:
    from ..cuda_kernels._brownian_motion_ik_cuda import brownian_motion_ik_cuda

    cfgs, errs, ee_points, target_points = brownian_motion_ik_cuda(
        seeds=seeds,
        init_points=init_points,
        twists=twists,
        parent_tf=parent_tf,
        parent_idx=parent_idx,
        act_idx=act_idx,
        mimic_mul=mimic_mul,
        mimic_off=mimic_off,
        mimic_act_idx=mimic_act_idx,
        topo_inv=topo_inv,
        ancestor_mask=ancestor_mask,
        target_quat=jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        box_mins=box_mins,
        box_maxs=box_maxs,
        robot_spheres_local=robot_spheres_local,
        robot_sphere_joint_idx=robot_sphere_joint_idx,
        world_spheres=world_spheres,
        world_capsules=world_capsules,
        world_boxes=world_boxes,
        world_halfspaces=world_halfspaces,
        self_pair_i=self_pair_i,
        self_pair_j=self_pair_j,
        lower=lower,
        upper=upper,
        fixed_mask=fixed_mask,
        rng_seed=rng_seed,
        target_jnt=target_jnt,
        max_iter=max_iter,
        pos_weight=pos_weight,
        ori_weight=ori_weight,
        lambda_init=lambda_init,
        eps_pos=eps_pos,
        noise_std=noise_std,
        n_brownian_steps=n_brownian_steps,
        fk_check_freq=fk_check_freq,
        threads_per_block=threads_per_block,
        enable_collision=enable_collision,
        collision_weight=collision_weight,
        collision_margin=collision_margin,
    )

    best_idx = jnp.argmin(errs, axis=1)
    rows = jnp.arange(cfgs.shape[0])
    best_cfgs = cfgs[rows, best_idx]
    best_errs = errs[rows, best_idx]
    best_ee = ee_points[rows, best_idx]
    best_targets = target_points[rows, best_idx]
    inside = jnp.all((best_ee >= box_mins) & (best_ee <= box_maxs), axis=1)
    return best_cfgs, best_ee, best_targets, best_errs, inside


# ── SVGD JIT step ────────────────────────────────────────────────────────────


@functools.partial(
    jax.jit,
    static_argnames=(
        "target_jnt",
        "n_iters",
        "bandwidth",
        "step_size",
        "threads_per_block",
        "enable_collision",
        "collision_weight",
        "collision_margin",
    ),
)
def _svgd_region_batch_select_jit(
    seeds: Array,
    init_points: Array,
    twists: Array,
    parent_tf: Array,
    parent_idx: Array,
    act_idx: Array,
    mimic_mul: Array,
    mimic_off: Array,
    mimic_act_idx: Array,
    topo_inv: Array,
    ancestor_mask: Array,
    box_mins: Array,
    box_maxs: Array,
    robot_spheres_local: Array,
    robot_sphere_joint_idx: Array,
    world_spheres: Array,
    world_capsules: Array,
    world_boxes: Array,
    world_halfspaces: Array,
    self_pair_i: Array,
    self_pair_j: Array,
    lower: Array,
    upper: Array,
    fixed_mask: Array,
    rng_seed: Array,
    *,
    target_jnt: int,
    n_iters: int,
    bandwidth: float,
    step_size: float,
    threads_per_block: int = 128,
    enable_collision: bool = False,
    collision_weight: float = 0.0,
    collision_margin: float = 0.02,
) -> tuple[Array, Array, Array, Array, Array]:
    from ..cuda_kernels._svgd_region_ik_cuda import svgd_region_ik_cuda

    cfgs, errs, ee_points, target_points = svgd_region_ik_cuda(
        seeds=seeds,
        init_points=init_points,
        twists=twists,
        parent_tf=parent_tf,
        parent_idx=parent_idx,
        act_idx=act_idx,
        mimic_mul=mimic_mul,
        mimic_off=mimic_off,
        mimic_act_idx=mimic_act_idx,
        topo_inv=topo_inv,
        target_jnts=jnp.array([target_jnt], dtype=jnp.int32),
        ancestor_masks=ancestor_mask[None, :],
        robot_spheres_local=robot_spheres_local,
        robot_sphere_joint_idx=robot_sphere_joint_idx,
        world_spheres=world_spheres,
        world_capsules=world_capsules,
        world_boxes=world_boxes,
        world_halfspaces=world_halfspaces,
        self_pair_i=self_pair_i,
        self_pair_j=self_pair_j,
        lower=lower,
        upper=upper,
        fixed_mask=fixed_mask,
        n_iters=n_iters,
        bandwidth=bandwidth,
        step_size=step_size,
        enable_collision=enable_collision,
        collision_weight=collision_weight,
        collision_margin=collision_margin,
    )

    best_idx = jnp.argmin(errs, axis=1)
    rows = jnp.arange(cfgs.shape[0])
    best_cfgs = cfgs[rows, best_idx]
    best_errs = errs[rows, best_idx]
    best_ee = ee_points[rows, best_idx]
    best_targets = target_points[rows, best_idx]
    inside = jnp.all((best_ee >= box_mins) & (best_ee <= box_maxs), axis=1)
    return best_cfgs, best_ee, best_targets, best_errs, inside


# ── Hit-and-run JIT step ─────────────────────────────────────────────────────


@functools.partial(
    jax.jit,
    static_argnames=(
        "target_jnt",
        "max_iter",
        "n_iterations",
        "pos_weight",
        "ori_weight",
        "lambda_init",
        "eps_pos",
        "eps_ori",
        "noise_std",
        "threads_per_block",
        "enable_collision",
        "collision_weight",
        "collision_margin",
    ),
)
def _hit_and_run_batch_select_jit(
    seeds: Array,
    twists: Array,
    parent_tf: Array,
    parent_idx: Array,
    act_idx: Array,
    mimic_mul: Array,
    mimic_off: Array,
    mimic_act_idx: Array,
    topo_inv: Array,
    ancestor_mask: Array,
    box_mins: Array,
    box_maxs: Array,
    robot_spheres_local: Array,
    robot_sphere_joint_idx: Array,
    world_spheres: Array,
    world_capsules: Array,
    world_boxes: Array,
    world_halfspaces: Array,
    self_pair_i: Array,
    self_pair_j: Array,
    lower: Array,
    upper: Array,
    fixed_mask: Array,
    rng_seed: Array,
    *,
    target_jnt: int,
    max_iter: int,
    n_iterations: int,
    pos_weight: float,
    ori_weight: float,
    lambda_init: float,
    eps_pos: float,
    eps_ori: float,
    noise_std: float,
    threads_per_block: int = 128,
    enable_collision: bool = False,
    collision_weight: float = 0.0,
    collision_margin: float = 0.02,
) -> tuple[Array, Array, Array, Array, Array]:
    from ..cuda_kernels._hit_and_run_ik_cuda import hit_and_run_ik_cuda

    cfgs, errs, ee_points, target_points = hit_and_run_ik_cuda(
        seeds=seeds,
        twists=twists,
        parent_tf=parent_tf,
        parent_idx=parent_idx,
        act_idx=act_idx,
        mimic_mul=mimic_mul,
        mimic_off=mimic_off,
        mimic_act_idx=mimic_act_idx,
        topo_inv=topo_inv,
        ancestor_mask=ancestor_mask,
        box_mins=box_mins,
        box_maxs=box_maxs,
        robot_spheres_local=robot_spheres_local,
        robot_sphere_joint_idx=robot_sphere_joint_idx,
        world_spheres=world_spheres,
        world_capsules=world_capsules,
        world_boxes=world_boxes,
        world_halfspaces=world_halfspaces,
        self_pair_i=self_pair_i,
        self_pair_j=self_pair_j,
        lower=lower,
        upper=upper,
        fixed_mask=fixed_mask,
        rng_seed=rng_seed,
        target_jnt=target_jnt,
        max_iter=max_iter,
        n_iterations=n_iterations,
        pos_weight=pos_weight,
        ori_weight=ori_weight,
        lambda_init=lambda_init,
        eps_pos=eps_pos,
        eps_ori=eps_ori,
        noise_std=noise_std,
        threads_per_block=threads_per_block,
        enable_collision=enable_collision,
        collision_weight=collision_weight,
        collision_margin=collision_margin,
    )

    best_idx = jnp.argmin(errs, axis=1)
    rows = jnp.arange(cfgs.shape[0])
    best_cfgs = cfgs[rows, best_idx]
    best_errs = errs[rows, best_idx]
    best_ee = ee_points[rows, best_idx]
    best_targets = target_points[rows, best_idx]
    inside = jnp.all((best_ee >= box_mins) & (best_ee <= box_maxs), axis=1)
    return best_cfgs, best_ee, best_targets, best_errs, inside


# ── Direct (sample-then-solve) JIT step ──────────────────────────────────────


@functools.partial(
    jax.jit,
    static_argnames=(
        "target_jnt",
        "target_link_index",
        "max_iter",
        "pos_weight",
        "ori_weight",
        "lambda_init",
        "eps_pos",
        "enable_collision",
        "collision_weight",
        "collision_margin",
    ),
)
def _direct_region_batch_select_jit(
    robot: Robot,
    seeds: Array,
    init_points: Array,         # (n_problems, n_restarts, 3) — same target per restart
    ancestor_mask: Array,
    box_mins: Array,
    box_maxs: Array,
    fixed_mask: Array,
    robot_spheres_local: Array,
    robot_sphere_joint_idx: Array,
    world_spheres: Array,
    world_capsules: Array,
    world_boxes: Array,
    world_halfspaces: Array,
    self_pair_i: Array,
    self_pair_j: Array,
    *,
    target_jnt: int,
    target_link_index: int,
    max_iter: int,
    pos_weight: float,
    ori_weight: float,
    lambda_init: float,
    eps_pos: float,
    enable_collision: bool = False,
    collision_weight: float = 0.0,
    collision_margin: float = 0.02,
) -> tuple[Array, Array, Array, Array, Array]:
    """Sample-then-solve: each problem solves IK to a fixed cartesian target.

    Reuses ``ls_ik_cuda`` (the same kernel used by ``ls_ik_solve_cuda``)
    with one EE and identity target orientation.  EE positions are recovered
    with a JAX FK pass on the per-problem winners.

    ``ori_weight`` should be left small (default 0.1) so the orientation term
    is effectively a soft tie-breaker — the position residual dominates by a
    factor of ``(pos_weight / ori_weight)``.  Setting ``ori_weight = 0``
    triggers a degenerate trust-region update inside the kernel that prevents
    convergence; even ``1e-3`` is enough to recover sub-millimetre position
    accuracy.
    """
    from ..cuda_kernels._ls_ik_cuda import ls_ik_cuda

    n_problems = seeds.shape[0]
    targets_per_problem = init_points[:, 0, :]  # (n_problems, 3)

    quat = jnp.broadcast_to(
        jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32), (n_problems, 4)
    )
    target_T = jnp.concatenate([quat, targets_per_problem], axis=1)[:, None, :]  # (n_problems, 1, 7)

    cfgs, errs = ls_ik_cuda(
        seeds=seeds,
        twists=robot.joints.twists,
        parent_tf=robot.joints.parent_transforms,
        parent_idx=robot.joints.parent_indices,
        act_idx=robot.joints.actuated_indices,
        mimic_mul=robot.joints.mimic_multiplier,
        mimic_off=robot.joints.mimic_offset,
        mimic_act_idx=robot.joints.mimic_act_indices,
        topo_inv=robot.joints._topo_sort_inv,
        target_jnts=jnp.array([target_jnt], dtype=jnp.int32),
        ancestor_masks=ancestor_mask[None, :],
        target_T=target_T,
        robot_spheres_local=robot_spheres_local,
        robot_sphere_joint_idx=robot_sphere_joint_idx,
        world_spheres=world_spheres,
        world_capsules=world_capsules,
        world_boxes=world_boxes,
        world_halfspaces=world_halfspaces,
        self_pair_i=self_pair_i,
        self_pair_j=self_pair_j,
        lower=robot.joints.lower_limits,
        upper=robot.joints.upper_limits,
        fixed_mask=fixed_mask,
        max_iter=max_iter,
        pos_weight=pos_weight,
        ori_weight=ori_weight,
        lambda_init=lambda_init,
        eps_pos=eps_pos,
        eps_ori=1.0,
        enable_collision=enable_collision,
        collision_weight=collision_weight,
        collision_margin=collision_margin,
    )

    best_idx = jnp.argmin(errs, axis=1)
    rows = jnp.arange(n_problems)
    best_cfgs = cfgs[rows, best_idx]
    best_errs = errs[rows, best_idx]

    # Recover EE positions via JAX FK on the per-problem winners.
    link_poses = robot.forward_kinematics(best_cfgs)  # (n_problems, n_links, 7)
    best_ee = link_poses[:, target_link_index, 4:7]   # (n_problems, 3)

    inside = jnp.all((best_ee >= box_mins) & (best_ee <= box_maxs), axis=1)
    return best_cfgs, best_ee, targets_per_problem, best_errs, inside


# ── Public samplers ──────────────────────────────────────────────────────────


def brownian_motion_sample_box_region_cuda(
    robot: Robot,
    target_link_index: int,
    rng_key: Array,
    previous_cfg: Float[Array, "n_act"],
    box_min: Float[Array, "3"] | Float[Array, "n_boxes 3"],
    box_max: Float[Array, "3"] | Float[Array, "n_boxes 3"],
    *,
    num_samples: int = 4096,
    seeds_per_launch: int = 2048,
    restarts_per_target: int = 8,
    max_iter: int = 20,
    pos_weight: float = 50.0,
    ori_weight: float = 0.0,
    lambda_init: float = 5e-3,
    eps_pos: float = 1e-4,
    noise_std: float = 0.02,
    n_brownian_steps: int = 100,
    fk_check_freq: int = 5,
    threads_per_block: int = 128,
    fixed_joint_mask: Float[Array, "n_act"] | None = None,
    memory_limit_gb: float = 2.0,
    max_batches: int | None = None,
    target_entropy: float | None = None,
    entropy_bins: int = 10,
    verbose: bool = False,
    collision_free: bool = False,
    collision_checker: Any | None = None,
    collision_world: Any | None = None,
    collision_weight: float = 1e4,
    collision_margin: float = 0.02,
) -> (
    Tuple[
        Float[Array, "n_samples n_act"],
        Float[Array, "n_samples 3"],
        Float[Array, "n_samples 3"],
        Float[Array, "n_samples"],
    ]
    | Tuple[
        Float[Array, "n_boxes n_samples n_act"],
        Float[Array, "n_boxes n_samples 3"],
        Float[Array, "n_boxes n_samples 3"],
        Float[Array, "n_boxes n_samples"],
    ]
):
    """Sample IK configurations whose end-effectors lie inside one or more box regions.

    The CUDA kernel uses a two-phase strategy per seed:

      Phase 1 – LM Boundary Reach (``max_iter`` iterations):
        Levenberg-Marquardt IK toward a fixed target pre-sampled inside the
        box.

      Phase 2 – Null-Space Brownian Shuffle (``n_brownian_steps`` steps):
        Gaussian perturbations in joint space projected onto the null-space
        of the position Jacobian, with periodic FK-check corrections.

    **Batched boxes**: pass ``box_min``/``box_max`` with shape ``(n_boxes, 3)``
    to sample ``num_samples`` configurations for *each* box in a single call.
    """
    n_act = int(robot.joints.num_actuated_joints)
    lower = robot.joints.lower_limits
    upper = robot.joints.upper_limits
    fixed_mask = (
        jnp.zeros((n_act,), dtype=jnp.int32)
        if fixed_joint_mask is None
        else fixed_joint_mask.astype(jnp.int32)
    )

    target_jnt, ancestor_mask = _compute_ancestor_mask(robot, target_link_index)
    box_min, box_max, batched_input = _normalise_boxes(box_min, box_max)

    if restarts_per_target < 1:
        raise ValueError("restarts_per_target must be >= 1.")
    _validate_threads_per_block(int(threads_per_block))
    seeds_per_launch = _seeds_per_launch_budget(
        n_act, seeds_per_launch, memory_limit_gb, restarts_per_target
    )

    (
        robot_spheres_local,
        robot_sphere_joint_idx,
        world_spheres,
        world_capsules,
        world_boxes,
        world_halfspaces,
        self_pair_i,
        self_pair_j,
        kernel_collision_enabled,
    ) = _prepare_ls_collision_buffers(robot, collision_checker, collision_world)
    enable_collision = bool(collision_free and kernel_collision_enabled)

    def step_fn(*, seeds, init_points, box_mins_pp, box_maxs_pp, rng_seed):
        return _brownian_motion_batch_select_jit(
            seeds=seeds,
            init_points=init_points,
            twists=robot.joints.twists,
            parent_tf=robot.joints.parent_transforms,
            parent_idx=robot.joints.parent_indices,
            act_idx=robot.joints.actuated_indices,
            mimic_mul=robot.joints.mimic_multiplier,
            mimic_off=robot.joints.mimic_offset,
            mimic_act_idx=robot.joints.mimic_act_indices,
            topo_inv=robot.joints._topo_sort_inv,
            ancestor_mask=ancestor_mask,
            box_mins=box_mins_pp,
            box_maxs=box_maxs_pp,
            robot_spheres_local=robot_spheres_local,
            robot_sphere_joint_idx=robot_sphere_joint_idx,
            world_spheres=world_spheres,
            world_capsules=world_capsules,
            world_boxes=world_boxes,
            world_halfspaces=world_halfspaces,
            self_pair_i=self_pair_i,
            self_pair_j=self_pair_j,
            lower=lower,
            upper=upper,
            fixed_mask=fixed_mask,
            rng_seed=rng_seed,
            target_jnt=target_jnt,
            max_iter=max_iter,
            pos_weight=pos_weight,
            ori_weight=ori_weight,
            lambda_init=lambda_init,
            eps_pos=eps_pos,
            noise_std=noise_std,
            n_brownian_steps=n_brownian_steps,
            fk_check_freq=fk_check_freq,
            threads_per_block=threads_per_block,
            enable_collision=enable_collision,
            collision_weight=float(collision_weight),
            collision_margin=float(collision_margin),
        )

    return _run_region_sampler_loop(
        n_act=n_act,
        lower=lower,
        upper=upper,
        fixed_mask=fixed_mask,
        previous_cfg=previous_cfg,
        rng_key=rng_key,
        box_min=box_min,
        box_max=box_max,
        batched_input=batched_input,
        num_samples=num_samples,
        seeds_per_launch=seeds_per_launch,
        restarts_per_target=restarts_per_target,
        max_batches=max_batches,
        target_entropy=target_entropy,
        entropy_bins=entropy_bins,
        verbose=verbose,
        sample_init_points=True,
        step_fn=step_fn,
        increase_hint="max_iter/restarts_per_target",
    )


def svgd_sample_box_region_cuda(
    robot: Robot,
    target_link_index: int,
    rng_key: Array,
    previous_cfg: Float[Array, "n_act"],
    box_min: Float[Array, "3"] | Float[Array, "n_boxes 3"],
    box_max: Float[Array, "3"] | Float[Array, "n_boxes 3"],
    *,
    num_samples: int = 4096,
    seeds_per_launch: int = 2048,
    restarts_per_target: int = 8,
    n_iters: int = 50,
    bandwidth: float = 0.1,
    step_size: float = 0.05,
    threads_per_block: int = 128,
    fixed_joint_mask: Float[Array, "n_act"] | None = None,
    memory_limit_gb: float = 2.0,
    max_batches: int | None = None,
    target_entropy: float | None = None,
    entropy_bins: int = 10,
    verbose: bool = False,
    collision_free: bool = False,
    collision_checker: Any | None = None,
    collision_world: Any | None = None,
    collision_weight: float = 1e4,
    collision_margin: float = 0.02,
) -> (
    Tuple[
        Float[Array, "n_samples n_act"],
        Float[Array, "n_samples 3"],
        Float[Array, "n_samples 3"],
        Float[Array, "n_samples"],
    ]
    | Tuple[
        Float[Array, "n_boxes n_samples n_act"],
        Float[Array, "n_boxes n_samples 3"],
        Float[Array, "n_boxes n_samples 3"],
        Float[Array, "n_boxes n_samples"],
    ]
):
    """SVGD region IK with optional batched boxes (see module docstring)."""
    n_act = int(robot.joints.num_actuated_joints)
    lower = robot.joints.lower_limits
    upper = robot.joints.upper_limits
    fixed_mask = (
        jnp.zeros((n_act,), dtype=jnp.int32)
        if fixed_joint_mask is None
        else fixed_joint_mask.astype(jnp.int32)
    )

    target_jnt, ancestor_mask = _compute_ancestor_mask(robot, target_link_index)
    box_min, box_max, batched_input = _normalise_boxes(box_min, box_max)

    if restarts_per_target < 1:
        raise ValueError("restarts_per_target must be >= 1.")
    _validate_threads_per_block(int(threads_per_block))
    seeds_per_launch = _seeds_per_launch_budget(
        n_act, seeds_per_launch, memory_limit_gb, restarts_per_target
    )

    (
        robot_spheres_local,
        robot_sphere_joint_idx,
        world_spheres,
        world_capsules,
        world_boxes,
        world_halfspaces,
        self_pair_i,
        self_pair_j,
        kernel_collision_enabled,
    ) = _prepare_ls_collision_buffers(robot, collision_checker, collision_world)
    enable_collision = bool(collision_free and kernel_collision_enabled)

    def step_fn(*, seeds, init_points, box_mins_pp, box_maxs_pp, rng_seed):
        return _svgd_region_batch_select_jit(
            seeds=seeds,
            init_points=init_points,
            twists=robot.joints.twists,
            parent_tf=robot.joints.parent_transforms,
            parent_idx=robot.joints.parent_indices,
            act_idx=robot.joints.actuated_indices,
            mimic_mul=robot.joints.mimic_multiplier,
            mimic_off=robot.joints.mimic_offset,
            mimic_act_idx=robot.joints.mimic_act_indices,
            topo_inv=robot.joints._topo_sort_inv,
            ancestor_mask=ancestor_mask,
            box_mins=box_mins_pp,
            box_maxs=box_maxs_pp,
            robot_spheres_local=robot_spheres_local,
            robot_sphere_joint_idx=robot_sphere_joint_idx,
            world_spheres=world_spheres,
            world_capsules=world_capsules,
            world_boxes=world_boxes,
            world_halfspaces=world_halfspaces,
            self_pair_i=self_pair_i,
            self_pair_j=self_pair_j,
            lower=lower,
            upper=upper,
            fixed_mask=fixed_mask,
            rng_seed=rng_seed,
            target_jnt=target_jnt,
            n_iters=n_iters,
            bandwidth=bandwidth,
            step_size=step_size,
            threads_per_block=threads_per_block,
            enable_collision=enable_collision,
            collision_weight=float(collision_weight),
            collision_margin=float(collision_margin),
        )

    return _run_region_sampler_loop(
        n_act=n_act,
        lower=lower,
        upper=upper,
        fixed_mask=fixed_mask,
        previous_cfg=previous_cfg,
        rng_key=rng_key,
        box_min=box_min,
        box_max=box_max,
        batched_input=batched_input,
        num_samples=num_samples,
        seeds_per_launch=seeds_per_launch,
        restarts_per_target=restarts_per_target,
        max_batches=max_batches,
        target_entropy=target_entropy,
        entropy_bins=entropy_bins,
        verbose=verbose,
        sample_init_points=True,
        step_fn=step_fn,
        increase_hint="n_iters/restarts_per_target",
    )


def hit_and_run_sample_box_region_cuda(
    robot: Robot,
    target_link_index: int,
    rng_key: Array,
    previous_cfg: Float[Array, "n_act"],
    box_min: Float[Array, "3"] | Float[Array, "n_boxes 3"],
    box_max: Float[Array, "3"] | Float[Array, "n_boxes 3"],
    *,
    num_samples: int = 4096,
    seeds_per_launch: int = 2048,
    restarts_per_target: int = 8,
    max_iter: int = 20,
    n_iterations: int = 100,
    pos_weight: float = 50.0,
    ori_weight: float = 0.0,
    lambda_init: float = 5e-3,
    eps_pos: float = 1e-4,
    eps_ori: float = 1e-4,
    noise_std: float = 0.02,
    threads_per_block: int = 128,
    fixed_joint_mask: Float[Array, "n_act"] | None = None,
    memory_limit_gb: float = 2.0,
    max_batches: int | None = None,
    target_entropy: float | None = None,
    entropy_bins: int = 10,
    verbose: bool = False,
    collision_free: bool = False,
    collision_checker: Any | None = None,
    collision_world: Any | None = None,
    collision_weight: float = 1e4,
    collision_margin: float = 0.02,
) -> (
    Tuple[
        Float[Array, "n_samples n_act"],
        Float[Array, "n_samples 3"],
        Float[Array, "n_samples 3"],
        Float[Array, "n_samples"],
    ]
    | Tuple[
        Float[Array, "n_boxes n_samples n_act"],
        Float[Array, "n_boxes n_samples 3"],
        Float[Array, "n_boxes n_samples 3"],
        Float[Array, "n_boxes n_samples"],
    ]
):
    """Hit-and-run region IK with optional batched boxes (see module docstring)."""
    n_act = int(robot.joints.num_actuated_joints)
    lower = robot.joints.lower_limits
    upper = robot.joints.upper_limits
    fixed_mask = (
        jnp.zeros((n_act,), dtype=jnp.int32)
        if fixed_joint_mask is None
        else fixed_joint_mask.astype(jnp.int32)
    )

    target_jnt, ancestor_mask = _compute_ancestor_mask(robot, target_link_index)
    box_min, box_max, batched_input = _normalise_boxes(box_min, box_max)

    if restarts_per_target < 1:
        raise ValueError("restarts_per_target must be >= 1.")
    _validate_threads_per_block(int(threads_per_block), _HIT_AND_RUN_MAX_TPB_BY_SMEM)
    seeds_per_launch = _seeds_per_launch_budget(
        n_act, seeds_per_launch, memory_limit_gb, restarts_per_target
    )

    (
        robot_spheres_local,
        robot_sphere_joint_idx,
        world_spheres,
        world_capsules,
        world_boxes,
        world_halfspaces,
        self_pair_i,
        self_pair_j,
        kernel_collision_enabled,
    ) = _prepare_ls_collision_buffers(robot, collision_checker, collision_world)
    enable_collision = bool(collision_free and kernel_collision_enabled)

    def step_fn(*, seeds, init_points, box_mins_pp, box_maxs_pp, rng_seed):
        return _hit_and_run_batch_select_jit(
            seeds=seeds,
            twists=robot.joints.twists,
            parent_tf=robot.joints.parent_transforms,
            parent_idx=robot.joints.parent_indices,
            act_idx=robot.joints.actuated_indices,
            mimic_mul=robot.joints.mimic_multiplier,
            mimic_off=robot.joints.mimic_offset,
            mimic_act_idx=robot.joints.mimic_act_indices,
            topo_inv=robot.joints._topo_sort_inv,
            ancestor_mask=ancestor_mask,
            box_mins=box_mins_pp,
            box_maxs=box_maxs_pp,
            robot_spheres_local=robot_spheres_local,
            robot_sphere_joint_idx=robot_sphere_joint_idx,
            world_spheres=world_spheres,
            world_capsules=world_capsules,
            world_boxes=world_boxes,
            world_halfspaces=world_halfspaces,
            self_pair_i=self_pair_i,
            self_pair_j=self_pair_j,
            lower=lower,
            upper=upper,
            fixed_mask=fixed_mask,
            rng_seed=rng_seed,
            target_jnt=target_jnt,
            max_iter=max_iter,
            n_iterations=n_iterations,
            pos_weight=pos_weight,
            ori_weight=ori_weight,
            lambda_init=lambda_init,
            eps_pos=eps_pos,
            eps_ori=eps_ori,
            noise_std=noise_std,
            threads_per_block=threads_per_block,
            enable_collision=enable_collision,
            collision_weight=float(collision_weight),
            collision_margin=float(collision_margin),
        )

    return _run_region_sampler_loop(
        n_act=n_act,
        lower=lower,
        upper=upper,
        fixed_mask=fixed_mask,
        previous_cfg=previous_cfg,
        rng_key=rng_key,
        box_min=box_min,
        box_max=box_max,
        batched_input=batched_input,
        num_samples=num_samples,
        seeds_per_launch=seeds_per_launch,
        restarts_per_target=restarts_per_target,
        max_batches=max_batches,
        target_entropy=target_entropy,
        entropy_bins=entropy_bins,
        verbose=verbose,
        sample_init_points=False,
        step_fn=step_fn,
        increase_hint="max_iter/restarts_per_target",
    )


def direct_sample_box_region_cuda(
    robot: Robot,
    target_link_index: int,
    rng_key: Array,
    previous_cfg: Float[Array, "n_act"],
    box_min: Float[Array, "3"] | Float[Array, "n_boxes 3"],
    box_max: Float[Array, "3"] | Float[Array, "n_boxes 3"],
    *,
    num_samples: int = 4096,
    seeds_per_launch: int = 2048,
    restarts_per_target: int = 8,
    max_iter: int = 60,
    pos_weight: float = 50.0,
    ori_weight: float = 0.1,
    lambda_init: float = 5e-3,
    eps_pos: float = 1e-4,
    fixed_joint_mask: Float[Array, "n_act"] | None = None,
    memory_limit_gb: float = 2.0,
    max_batches: int | None = None,
    target_entropy: float | None = None,
    entropy_bins: int = 10,
    verbose: bool = False,
    collision_free: bool = False,
    collision_checker: Any | None = None,
    collision_world: Any | None = None,
    collision_weight: float = 1e4,
    collision_margin: float = 0.02,
) -> (
    Tuple[
        Float[Array, "n_samples n_act"],
        Float[Array, "n_samples 3"],
        Float[Array, "n_samples 3"],
        Float[Array, "n_samples"],
    ]
    | Tuple[
        Float[Array, "n_boxes n_samples n_act"],
        Float[Array, "n_boxes n_samples 3"],
        Float[Array, "n_boxes n_samples 3"],
        Float[Array, "n_boxes n_samples"],
    ]
):
    """Sample target points uniformly inside the box(es) and solve IK to each one.

    For each problem in a launch the loop draws a single cartesian target
    point uniformly inside the assigned box, then runs ``restarts_per_target``
    seeds through the standard ``ls_ik_cuda`` Levenberg-Marquardt kernel
    (position-only; ``ori_weight=0``).  The per-problem winner is the seed
    with the smallest weighted residual, and EE positions are recovered via
    a JAX FK pass on the winners.

    Compared with the brownian / SVGD / hit-and-run samplers, this sampler:

      * Has no null-space exploration phase — coverage of the box is
        determined purely by the cartesian-space sampling of the targets.
      * Reuses the same kernel as ``ls_ik_solve_cuda`` (no separate
        ``_region_ik`` shared library required).
      * Produces ``ee_points`` that hit each sampled target almost exactly,
        so the EE-point distribution matches the target distribution
        (uniform inside the box) within the LM convergence tolerance.

    **Batched boxes**: pass ``box_min``/``box_max`` with shape ``(n_boxes, 3)``
    to sample ``num_samples`` configurations for *each* box in a single call.
    Seeds within each launch are distributed round-robin across boxes so all
    regions are explored in parallel.

    Args:
        box_min: Box minimum corner(s).  Shape ``(3,)`` or ``(n_boxes, 3)``.
        box_max: Box maximum corner(s).  Matching shape to ``box_min``.
        num_samples: Total samples per box.
        seeds_per_launch: Targets per kernel launch (round-robin across boxes).
        restarts_per_target: LM seeds per target (winner is min-error seed).
        max_iter: LM iterations per seed.
        pos_weight: Weight on the position residual.
        ori_weight: Weight on the orientation residual.  Keep this small but
            non-zero (default 0.1).  ``ori_weight = 0`` triggers a degenerate
            update path in ``ls_ik_cuda`` that prevents convergence; ``0.1``
            with ``pos_weight=50`` lets position dominate by 500x while still
            keeping the kernel's trust region well-conditioned.
        lambda_init: Initial LM damping.
        eps_pos: Position convergence tolerance [m].
        target_entropy: Optional entropy-based early stop (per box).
        entropy_bins: Histogram bins per axis for entropy computation.
    """
    n_act = int(robot.joints.num_actuated_joints)
    fixed_mask = (
        jnp.zeros((n_act,), dtype=jnp.int32)
        if fixed_joint_mask is None
        else fixed_joint_mask.astype(jnp.int32)
    )

    target_jnt, ancestor_mask = _compute_ancestor_mask(robot, target_link_index)
    box_min, box_max, batched_input = _normalise_boxes(box_min, box_max)

    if restarts_per_target < 1:
        raise ValueError("restarts_per_target must be >= 1.")
    seeds_per_launch = _seeds_per_launch_budget(
        n_act, seeds_per_launch, memory_limit_gb, restarts_per_target
    )

    (
        robot_spheres_local,
        robot_sphere_joint_idx,
        world_spheres,
        world_capsules,
        world_boxes,
        world_halfspaces,
        self_pair_i,
        self_pair_j,
        kernel_collision_enabled,
    ) = _prepare_ls_collision_buffers(robot, collision_checker, collision_world)
    enable_collision = bool(collision_free and kernel_collision_enabled)

    def step_fn(*, seeds, init_points, box_mins_pp, box_maxs_pp, rng_seed):
        del rng_seed  # ls_ik_cuda is deterministic given seeds.
        return _direct_region_batch_select_jit(
            robot=robot,
            seeds=seeds,
            init_points=init_points,
            ancestor_mask=ancestor_mask,
            box_mins=box_mins_pp,
            box_maxs=box_maxs_pp,
            fixed_mask=fixed_mask,
            robot_spheres_local=robot_spheres_local,
            robot_sphere_joint_idx=robot_sphere_joint_idx,
            world_spheres=world_spheres,
            world_capsules=world_capsules,
            world_boxes=world_boxes,
            world_halfspaces=world_halfspaces,
            self_pair_i=self_pair_i,
            self_pair_j=self_pair_j,
            target_jnt=target_jnt,
            target_link_index=int(target_link_index),
            max_iter=max_iter,
            pos_weight=pos_weight,
            ori_weight=ori_weight,
            lambda_init=lambda_init,
            eps_pos=eps_pos,
            enable_collision=enable_collision,
            collision_weight=float(collision_weight),
            collision_margin=float(collision_margin),
        )

    return _run_region_sampler_loop(
        n_act=n_act,
        lower=robot.joints.lower_limits,
        upper=robot.joints.upper_limits,
        fixed_mask=fixed_mask,
        previous_cfg=previous_cfg,
        rng_key=rng_key,
        box_min=box_min,
        box_max=box_max,
        batched_input=batched_input,
        num_samples=num_samples,
        seeds_per_launch=seeds_per_launch,
        restarts_per_target=restarts_per_target,
        max_batches=max_batches,
        target_entropy=target_entropy,
        entropy_bins=entropy_bins,
        verbose=verbose,
        sample_init_points=True,
        step_fn=step_fn,
        increase_hint="max_iter/restarts_per_target",
    )
