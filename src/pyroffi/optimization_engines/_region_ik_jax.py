"""Pure-JAX fallback implementations of direct and SVGD region-constrained IK.

Drop-in alternatives to the CUDA-backed samplers in _region_ik.py that run on
any JAX device (CPU or GPU via XLA) without requiring compiled CUDA kernels.

  direct_sample_box_region_jax
    Same algorithm as direct_sample_box_region_cuda: sample cartesian targets
    uniformly inside each box, then run multi-seed Levenberg-Marquardt IK to
    each sampled target.  Uses the pure-JAX LM solver (_ls_ik_single) vmapped
    over seeds and problems.

  svgd_sample_box_region_jax
    Same algorithm as svgd_sample_box_region_cuda: transport particles via
    Stein Variational Gradient Descent with RBF-kernel repulsion.  Uses JAX
    autodiff to compute score functions instead of the CUDA kernel.

Both functions share the same public signature as their ``_cuda`` counterparts
and plug into the same ``_run_region_sampler_loop`` collection loop, so
``num_samples``, batched-box queries, and entropy-based stopping all work
identically.

Note on throughput
------------------
These JAX fallbacks trade CUDA-kernel throughput for portability.  On a modern
GPU the CUDA kernels are typically 5-20× faster.  On CPU the JAX versions still
produce correct results but will be slower still; decrease ``seeds_per_launch``
and ``num_samples`` accordingly.
"""

from __future__ import annotations

import functools
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
from jax import Array
from jaxtyping import Float

from .._robot import Robot
from ._ik_primitives import _ik_residual, _LS_ALPHAS
from ._region_ik import (
    _compute_ancestor_mask,
    _normalise_boxes,
    _run_region_sampler_loop,
    _seeds_per_launch_budget,
)


# ---------------------------------------------------------------------------
# Direct: per-problem LM IK step
# ---------------------------------------------------------------------------


@functools.partial(
    jax.jit,
    static_argnames=(
        "target_link_index",
        "max_iter",
        "pos_weight",
        "ori_weight",
        "lambda_init",
    ),
)
def _direct_lm_step_jax(
    seeds: Array,           # (n_problems, n_restarts, n_act)
    init_points: Array,     # (n_problems, n_restarts, 3)
    box_mins: Array,        # (n_problems, 3)
    box_maxs: Array,        # (n_problems, 3)
    lower: Array,           # (n_act,)
    upper: Array,           # (n_act,)
    fixed_mask: Array,      # (n_act,) int32
    robot: Robot,
    *,
    target_link_index: int,
    max_iter: int,
    pos_weight: float,
    ori_weight: float,
    lambda_init: float,
) -> tuple[Array, Array, Array, Array, Array]:
    """Run LM IK for each (problem, restart) pair and return per-problem winners.

    For each problem the cartesian target is ``init_points[i, 0, :]``; all
    restarts within a problem share the same target.  This mirrors the CUDA
    direct-region kernel exactly.
    """
    n_problems, n_restarts, n_act = seeds.shape
    targets = init_points[:, 0, :]              # (n_problems, 3)
    fixed_mask_bool = fixed_mask.astype(jnp.bool_)

    W = jnp.concatenate([
        jnp.full(3, pos_weight, dtype=jnp.float32),
        jnp.full(3, ori_weight, dtype=jnp.float32),
    ])

    def lm_single(cfg: Array, target_pos: Array) -> tuple[Array, Array]:
        """LM solve from one seed toward one target; returns (best_cfg, best_err)."""
        pose = jaxlie.SE3(jnp.concatenate([
            jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32), target_pos
        ]))

        def fused(q: Array) -> Array:
            f = _ik_residual(q, robot, target_link_index, pose)
            return f * W

        def lm_step(carry, _):
            c, lam, best_c, best_err = carry

            f, vjp_fn = jax.vjp(fused, c)
            J = jax.vmap(lambda g: vjp_fn(g)[0])(jnp.eye(6, dtype=f.dtype))  # (6, n_act)
            curr_err = jnp.dot(f, f)

            col_scale = jnp.linalg.norm(J, axis=0) + 1e-8
            Js = J / col_scale[None, :]
            A = Js.T @ Js + lam * jnp.eye(n_act, dtype=Js.dtype)
            rhs = -(Js.T @ f)
            p = jnp.linalg.solve(A, rhs)

            delta = p / col_scale
            delta = jnp.where(fixed_mask_bool, 0.0, delta)

            pos_err_r = jnp.linalg.norm(f[:3] / W[:3])
            ori_err_r = jnp.linalg.norm(f[3:] / W[3:])
            R = jnp.where(
                (pos_err_r > 1e-2) | (ori_err_r > 0.6), 0.38,
                jnp.where(
                    (pos_err_r > 1e-3) | (ori_err_r > 0.25), 0.22,
                    jnp.where((pos_err_r > 2e-4) | (ori_err_r > 0.08), 0.12, 0.05)
                )
            )
            delta_norm = jnp.linalg.norm(delta) + 1e-18
            delta = jnp.where(delta_norm > R, delta * R / delta_norm, delta)

            def eval_alpha(alpha):
                nc = jnp.clip(c + alpha * delta, lower, upper)
                nf = fused(nc)
                return jnp.dot(nf, nf)

            alpha_errs = jax.vmap(eval_alpha)(_LS_ALPHAS)
            best_ls = jnp.argmin(alpha_errs)
            new_err = alpha_errs[best_ls]
            new_c = jnp.clip(c + _LS_ALPHAS[best_ls] * delta, lower, upper).astype(c.dtype)

            improved = new_err < curr_err * (1.0 - 1e-4)
            c_out = jnp.where(improved, new_c, c)
            lam_out = jnp.clip(jnp.where(improved, lam * 0.5, lam * 3.0), 1e-10, 1e6)

            new_best_c = jnp.where(new_err < best_err, new_c, best_c)
            new_best_err = jnp.where(new_err < best_err, new_err, best_err)
            return (c_out, lam_out, new_best_c, new_best_err), None

        init_f = fused(cfg)
        init_err = jnp.dot(init_f, init_f)
        lam0 = jnp.asarray(lambda_init, dtype=cfg.dtype)
        (_, _, best_c, best_err), _ = jax.lax.scan(
            lm_step, (cfg, lam0, cfg, init_err), None, length=max_iter
        )
        return best_c, best_err

    def solve_problem(seeds_p: Array, target_p: Array) -> tuple[Array, Array, Array, Array, Array]:
        """Solve IK for all restarts of one problem; return the per-problem winner."""
        # seeds_p: (n_restarts, n_act), target_p: (3,)
        cfgs, errs = jax.vmap(lambda q: lm_single(q, target_p))(seeds_p)  # (n_r, n_act), (n_r,)
        best = jnp.argmin(errs)
        best_cfg = cfgs[best]
        best_err = errs[best]
        link_poses = robot.forward_kinematics(best_cfg)
        best_ee = link_poses[target_link_index, 4:7]
        return best_cfg, best_ee, target_p, best_err

    # vmap over problems
    best_cfgs, best_ees, best_targets, best_errs = jax.vmap(solve_problem)(seeds, targets)

    inside = jnp.all((best_ees >= box_mins) & (best_ees <= box_maxs), axis=1)
    return best_cfgs, best_ees, best_targets, best_errs, inside


# ---------------------------------------------------------------------------
# SVGD: per-problem particle transport step
# ---------------------------------------------------------------------------


@functools.partial(
    jax.jit,
    static_argnames=(
        "target_link_index",
        "n_iters",
        "pos_weight",
        "bandwidth",
        "step_size",
    ),
)
def _svgd_region_step_jax(
    seeds: Array,           # (n_problems, n_particles, n_act)
    init_points: Array,     # (n_problems, n_particles, 3)
    box_mins: Array,        # (n_problems, 3)
    box_maxs: Array,        # (n_problems, 3)
    lower: Array,           # (n_act,)
    upper: Array,           # (n_act,)
    fixed_mask: Array,      # (n_act,) int32
    robot: Robot,
    *,
    target_link_index: int,
    n_iters: int,
    pos_weight: float,
    bandwidth: float,
    step_size: float,
) -> tuple[Array, Array, Array, Array, Array]:
    """SVGD particle transport for each problem; return per-problem winners.

    Algorithm (Liu & Wang 2016):
      For each iteration:
        1. Compute score_i = -grad_q cost(q_i)   where cost = pos_weight * ||fk_pos(q) - t||^2
        2. K[i,j] = exp(-||q_i - q_j||^2 / (2*h^2))   (RBF kernel)
        3. phi(q_i) = (1/n) * sum_j [ K[i,j] * score_j   +   K[i,j]*(q_j - q_i)/h^2 ]
        4. q_i <- clip(q_i + step_size * phi(q_i), lower, upper)

    The repulsion term grad_{q_j} K(q_i, q_j) = K(q_i,q_j)*(q_j - q_i)/h^2
    ensures particles spread across the constraint manifold.

    After transport, the particle with the lowest position error is the winner.
    """
    n_problems, n_particles, n_act = seeds.shape
    targets = init_points[:, 0, :]          # (n_problems, 3)
    fixed_mask_bool = fixed_mask.astype(jnp.bool_)
    h2 = bandwidth ** 2

    def pos_cost_one(q: Array, target: Array) -> Array:
        link_poses = robot.forward_kinematics(q)
        ee = link_poses[target_link_index, 4:7]
        return pos_weight * jnp.sum((ee - target) ** 2)

    def svgd_problem(particles_p: Array, target_p: Array) -> tuple[Array, Array, Array, Array]:
        """SVGD for one problem; returns (best_cfg, best_ee, target, best_err)."""

        def step(particles: Array, _) -> tuple[Array, None]:
            # particles: (n_particles, n_act)

            # Score: -grad_q cost(q_i)
            scores = jax.vmap(
                lambda q: -jax.grad(pos_cost_one)(q, target_p)
            )(particles)   # (n_particles, n_act)

            # RBF kernel matrix  K[i,j] = exp(-||q_i - q_j||^2 / (2h^2))
            diff = particles[:, None, :] - particles[None, :, :]   # (n_p, n_p, n_act)
            sq_dist = jnp.sum(diff ** 2, axis=-1)                  # (n_p, n_p)
            K = jnp.exp(-sq_dist / (2.0 * h2))                     # (n_p, n_p)

            # phi[i] = (1/n) * sum_j K[i,j] * (score[j] + (q_j - q_i)/h^2)
            # Note: (q_j - q_i) = -diff[i,j], so repulsion = -diff / h^2
            repulsion = -diff / h2                                   # (n_p, n_p, n_act)
            # weighted: K[i,j] * (score[j] + repulsion[i,j])
            # score[j] broadcast: (n_p, n_act) -> (n_p, n_p, n_act)
            phi = jnp.mean(
                K[:, :, None] * (scores[None, :, :] + repulsion), axis=1
            )                                                        # (n_p, n_act)

            new_particles = jnp.clip(particles + step_size * phi, lower, upper)
            new_particles = jnp.where(fixed_mask_bool[None, :], particles, new_particles)
            return new_particles, None

        final_particles, _ = jax.lax.scan(step, particles_p, None, length=n_iters)

        # Pick winner: lowest position error
        def pos_err_one(q: Array) -> Array:
            link_poses = robot.forward_kinematics(q)
            ee = link_poses[target_link_index, 4:7]
            return pos_weight * jnp.sum((ee - target_p) ** 2)

        errs = jax.vmap(pos_err_one)(final_particles)    # (n_particles,)
        best = jnp.argmin(errs)
        best_cfg = final_particles[best]
        best_err = errs[best]
        link_poses = robot.forward_kinematics(best_cfg)
        best_ee = link_poses[target_link_index, 4:7]
        return best_cfg, best_ee, target_p, best_err

    best_cfgs, best_ees, best_targets, best_errs = jax.vmap(svgd_problem)(seeds, targets)

    inside = jnp.all((best_ees >= box_mins) & (best_ees <= box_maxs), axis=1)
    return best_cfgs, best_ees, best_targets, best_errs, inside


# ---------------------------------------------------------------------------
# Public samplers
# ---------------------------------------------------------------------------


def direct_sample_box_region_jax(
    robot: Robot,
    target_link_index: int,
    rng_key: Array,
    previous_cfg: Float[Array, "n_act"],
    box_min: Float[Array, "3"] | Float[Array, "n_boxes 3"],
    box_max: Float[Array, "3"] | Float[Array, "n_boxes 3"],
    *,
    num_samples: int = 512,
    seeds_per_launch: int = 256,
    restarts_per_target: int = 4,
    max_iter: int = 60,
    pos_weight: float = 50.0,
    ori_weight: float = 0.1,
    lambda_init: float = 5e-3,
    fixed_joint_mask: Float[Array, "n_act"] | None = None,
    memory_limit_gb: float = 2.0,
    max_batches: int | None = None,
    target_entropy: float | None = None,
    entropy_bins: int = 10,
    verbose: bool = False,
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
    """Sample IK configurations whose end-effectors lie inside box region(s).

    Pure-JAX fallback for :func:`direct_sample_box_region_cuda`.  Samples a
    cartesian target uniformly inside the box then runs multi-seed
    Levenberg-Marquardt IK to each target using the JAX LM solver.

    This function is a drop-in replacement: it accepts the same box shapes
    (``(3,)`` or ``(n_boxes, 3)``), returns the same output shapes, and
    supports the same entropy-based early stop.

    Differences from the CUDA variant:
      * No ``threads_per_block`` parameter (not relevant for XLA).
      * No collision-free option (collision avoidance requires the CUDA
        collision buffers; use the CUDA variant when collision is needed).
      * Lower default ``seeds_per_launch`` / ``num_samples`` to keep
        per-launch JIT-compiled arrays from being too large on CPU.

    Args:
        robot:               The robot model.
        target_link_index:   Index of the target link (end-effector).
        rng_key:             JAX PRNG key.
        previous_cfg:        Previous joint configuration (warm-start seed).
        box_min:             Box minimum corner(s), shape ``(3,)`` or ``(n_boxes, 3)``.
        box_max:             Box maximum corner(s), matching shape.
        num_samples:         Target samples per box.
        seeds_per_launch:    Targets attempted per kernel launch.
        restarts_per_target: LM seeds per target.
        max_iter:            LM iterations per seed.
        pos_weight:          Weight on position residual.
        ori_weight:          Weight on orientation residual (keep > 0).
        lambda_init:         Initial LM damping.
        fixed_joint_mask:    Int32 mask; 1 = joint must not move.
        memory_limit_gb:     Soft cap on seeds per launch.
        max_batches:         Maximum collection loop iterations.
        target_entropy:      Entropy-based early stop (nats), per box.
        entropy_bins:        Histogram bins per axis for entropy.
        verbose:             Print per-batch timing.

    Returns:
        ``(cfgs, ee_points, target_points, errors)`` each with leading shape
        ``(n_samples,)`` for a single box or ``(n_boxes, n_samples)`` for
        batched boxes.
    """
    n_act = int(robot.joints.num_actuated_joints)
    lower = robot.joints.lower_limits
    upper = robot.joints.upper_limits
    fixed_mask = (
        jnp.zeros((n_act,), dtype=jnp.int32)
        if fixed_joint_mask is None
        else fixed_joint_mask.astype(jnp.int32)
    )

    _compute_ancestor_mask(robot, target_link_index)   # validate link index
    box_min, box_max, batched_input = _normalise_boxes(box_min, box_max)

    if restarts_per_target < 1:
        raise ValueError("restarts_per_target must be >= 1.")
    seeds_per_launch = _seeds_per_launch_budget(
        n_act, seeds_per_launch, memory_limit_gb, restarts_per_target
    )

    def step_fn(*, seeds, init_points, box_mins_pp, box_maxs_pp, rng_seed):
        del rng_seed
        return _direct_lm_step_jax(
            seeds=seeds,
            init_points=init_points,
            box_mins=box_mins_pp,
            box_maxs=box_maxs_pp,
            lower=lower,
            upper=upper,
            fixed_mask=fixed_mask,
            robot=robot,
            target_link_index=int(target_link_index),
            max_iter=max_iter,
            pos_weight=pos_weight,
            ori_weight=ori_weight,
            lambda_init=lambda_init,
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


def svgd_sample_box_region_jax(
    robot: Robot,
    target_link_index: int,
    rng_key: Array,
    previous_cfg: Float[Array, "n_act"],
    box_min: Float[Array, "3"] | Float[Array, "n_boxes 3"],
    box_max: Float[Array, "3"] | Float[Array, "n_boxes 3"],
    *,
    num_samples: int = 256,
    seeds_per_launch: int = 128,
    restarts_per_target: int = 8,
    n_iters: int = 50,
    bandwidth: float = 0.1,
    step_size: float = 0.05,
    pos_weight: float = 50.0,
    fixed_joint_mask: Float[Array, "n_act"] | None = None,
    memory_limit_gb: float = 2.0,
    max_batches: int | None = None,
    target_entropy: float | None = None,
    entropy_bins: int = 10,
    verbose: bool = False,
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
    """SVGD region-IK sampling — pure-JAX fallback for :func:`svgd_sample_box_region_cuda`.

    Transports particles via Stein Variational Gradient Descent with an RBF
    kernel to cover the kinematic constraint manifold uniformly inside the
    box region(s).  JAX autodiff computes the score function (negative cost
    gradient) at each iteration so no compiled CUDA kernel is required.

    SVGD update rule (Liu & Wang 2016):

        phi(q_i) = (1/n) * sum_j [ K(q_i, q_j) * score(q_j)
                                  + grad_{q_j} K(q_i, q_j) ]

    where K is the RBF kernel with bandwidth ``h`` and
    ``score(q) = -grad_q cost(q)``, ``cost = pos_weight * ||fk(q) - t||^2``.

    Supports batched-box queries (``box_min``/``box_max`` shape ``(n_boxes, 3)``)
    identical to the CUDA variant.

    Args:
        robot:               The robot model.
        target_link_index:   Index of the target link (end-effector).
        rng_key:             JAX PRNG key.
        previous_cfg:        Previous joint configuration (warm-start seed).
        box_min:             Box minimum corner(s), shape ``(3,)`` or ``(n_boxes, 3)``.
        box_max:             Box maximum corner(s), matching shape.
        num_samples:         Target samples per box.
        seeds_per_launch:    Number of problems launched per collection batch.
        restarts_per_target: Particles per problem (SVGD population size).
        n_iters:             SVGD iterations per launch.
        bandwidth:           RBF kernel bandwidth ``h`` (joint-space units).
        step_size:           SVGD gradient step size.
        pos_weight:          Weight on position cost (scales the score function).
        fixed_joint_mask:    Int32 mask; 1 = joint must not move.
        memory_limit_gb:     Soft cap on seeds per launch.
        max_batches:         Maximum collection loop iterations.
        target_entropy:      Entropy-based early stop (nats), per box.
        entropy_bins:        Histogram bins per axis for entropy.
        verbose:             Print per-batch timing.

    Returns:
        ``(cfgs, ee_points, target_points, errors)`` each with leading shape
        ``(n_samples,)`` for a single box or ``(n_boxes, n_samples)`` for
        batched boxes.
    """
    n_act = int(robot.joints.num_actuated_joints)
    lower = robot.joints.lower_limits
    upper = robot.joints.upper_limits
    fixed_mask = (
        jnp.zeros((n_act,), dtype=jnp.int32)
        if fixed_joint_mask is None
        else fixed_joint_mask.astype(jnp.int32)
    )

    _compute_ancestor_mask(robot, target_link_index)   # validate link index
    box_min, box_max, batched_input = _normalise_boxes(box_min, box_max)

    if restarts_per_target < 1:
        raise ValueError("restarts_per_target must be >= 1.")
    seeds_per_launch = _seeds_per_launch_budget(
        n_act, seeds_per_launch, memory_limit_gb, restarts_per_target
    )

    def step_fn(*, seeds, init_points, box_mins_pp, box_maxs_pp, rng_seed):
        del rng_seed
        return _svgd_region_step_jax(
            seeds=seeds,
            init_points=init_points,
            box_mins=box_mins_pp,
            box_maxs=box_maxs_pp,
            lower=lower,
            upper=upper,
            fixed_mask=fixed_mask,
            robot=robot,
            target_link_index=int(target_link_index),
            n_iters=n_iters,
            pos_weight=pos_weight,
            bandwidth=bandwidth,
            step_size=step_size,
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
        increase_hint="n_iters/restarts_per_target/bandwidth",
    )
