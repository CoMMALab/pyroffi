"""Kernel-repulsion particle phase for the tetris CSP (fast, GPU-shaped).

An earlier version of this file implemented *true* SVGD (Liu & Wang 2016): drop
SPaSM's top-k pre-filter, keep the full particle set, and run many
kernel-coupled transport steps (score = -grad(cost), median-heuristic RBF) to a
threshold. It converged but was ~1000x slower than `spasm.tetris.solve` (4681 ms vs
4.49 ms on tetris-3): SVGD couples every particle every step, needs the whole
unfiltered batch, and its repulsion keeps the ensemble from collapsing onto the
single best minimum, so the outer resample-until-`cost<eps` `while_loop`
retried many times before any particle cleared the threshold.

This version keeps the *idea* SVGD contributes -- an RBF-kernel **repulsion**
term that maintains particle diversity so the ensemble doesn't collapse -- but
folds it into SPaSM's own fast pipeline instead of replacing it:

  1. `sample_particles` (imported from spasm.tetris.solve, unchanged): sample the big
     `sampling_batch`, keep the top-`opt_batch` by cost. This cheap pre-filter
     is exactly what makes SPaSM GPU-shaped and is what pure SVGD threw away.
  2. Linear-phase GD; when `use_repulsion` is set, each step adds a small
     median-heuristic RBF repulsion drift on top of SPaSM's own linear
     `opt_step` delta. Attraction is plain gradient descent (fast, decoupled
     per particle); only the *repulsion* is kernel-coupled -- O(P^2 * D) with
     P<=256, D<=32, which is negligible.
  3. Quadratic polish (reusing spasm.tetris.solve.opt_step in 'quadratic' mode,
     vmapped per particle) -- byte-identical to spasm.tetris.solve.project's polish.

**Repulsion is off by default** (`use_repulsion=False`) so this solver is
faithful to `spasm.tetris.solve.solve` for benchmarking: with it off, the linear step
reduces bit-for-bit to SPaSM's `opt_step` and the pipeline matches the source.
Pass `--use_repulsion` (or set `params.use_repulsion = True`) to re-enable the
kernel drift -- the intended base for the forthcoming EBM + SMC formulation.

Same tetris cost, same CLI (--num_blocks {3,5,8}, --bench), same jit +
while_loop-until-threshold outer structure as spasm.tetris.solve.solve.
"""
import copy
from functools import partial
import os
import sys
import argparse
import time

import jax
import jax.numpy as jnp


from spasm import backend
from spasm.tetris.env import Simulation
from spasm.tetris.solve import SpasmParams, cost, clip_particle, opt_step, sample_particles


class SvgdParams(SpasmParams):
    """SpasmParams + two extra knobs: `use_repulsion`, which toggles the
    RBF-repulsion drift on the linear GD phase, and `svgd_weight`, its strength.

    Repulsion is **off by default** so this solver is faithful to SPaSM's
    `spasm.tetris.solve.solve` for benchmarking: with `use_repulsion=False` the linear
    step reduces bit-for-bit to SPaSM's `opt_step` and the whole pipeline
    matches the source. Set `use_repulsion=True` to re-enable the
    diversity-preserving kernel drift (the future EBM + SMC formulation will
    build on this path). All other fields (sampling_batch, opt_batch, opt_steps,
    lr, quadratic_*, cost_thresh) keep their spasm.tetris.solve meaning."""

    def __init__(self):
        super().__init__()
        self.use_repulsion = False
        self.svgd_weight = 2e-3


def _rbf_repulsion(particles):
    """particles: (P, nb, 4) -> (P, nb, 4) repulsive drift.

    Median-heuristic RBF kernel K_ij = exp(-||qi-qj||^2 / 2h^2); repulsion on
    particle i is (1/P) sum_j grad_{qj} K_ij = (1/P) sum_j K_ij (qi-qj)/h^2,
    which pushes particles away from their neighbours (the diversity-preserving
    half of the SVGD update). Attraction is handled by plain GD in the caller,
    so this is the only particle-coupled term."""
    P = particles.shape[0]
    bshape = particles.shape[1:]
    D = bshape[0] * bshape[1]

    flat = particles.reshape(P, D)
    diff = flat[:, None, :] - flat[None, :, :]        # (P, P, D), qi - qj
    sq_dist = jnp.sum(diff ** 2, axis=-1)             # (P, P)
    h2 = jnp.median(sq_dist) / (2.0 * jnp.log(P + 1.0)) + 1e-6
    K = jnp.exp(-sq_dist / (2.0 * h2))                # (P, P)

    repulsion = jnp.mean(K[:, :, None] * diff / h2, axis=1)   # (P, D)
    return repulsion.reshape(P, *bshape)


def _linear_svgd_step(i, params: SvgdParams, sim, particles):
    """One linear-phase step over the whole (P, nb, 4) batch: SPaSM's linear
    `opt_step` gradient delta (per particle), plus a small RBF repulsion drift
    when `params.use_repulsion` is set.

    With `use_repulsion=False` (the default) this is bit-for-bit SPaSM's linear
    `opt_step` applied per particle -- no particle coupling at all."""
    J = jax.vmap(jax.grad(cost, argnums=2), in_axes=(None, None, 0))(params, sim, particles)

    lr = (1 - i / params.opt_steps) * params.lr
    delta = -J * 5e-3
    delta = delta.at[..., -1].mul(1.0e3)
    gd_update = delta * lr

    particles_new = particles + gd_update
    if params.use_repulsion:
        particles_new = particles_new + params.svgd_weight * _rbf_repulsion(particles)

    return jax.vmap(clip_particle, in_axes=(None, 0))(sim, particles_new)


def project_svgd(params: SvgdParams, sim, particles):
    """Linear phase (kernel-repulsed GD, particle-coupled) then quadratic
    polish (per-particle, reusing spasm.tetris.solve.opt_step)."""
    lin_step = lambda i, particles: _linear_svgd_step(i, params, sim, particles)
    particles = jax.lax.fori_loop(0, params.opt_steps, lin_step, particles)

    quad_params = copy.copy(params)
    quad_params.opt = 'quadratic'
    quad_step = lambda i, particle: opt_step(i, quad_params, sim, particle)

    def polish(particle):
        return jax.lax.fori_loop(0, quad_params.quadratic_opt_steps, quad_step, particle)

    return jax.vmap(polish)(particles)


@partial(jax.jit, static_argnames=['params', 'sim'])
def solve(params: SvgdParams, sim, key):
    cost_fun = jax.vmap(cost, in_axes=(None, None, 0))

    def solve_once(state):
        key, _, _ = state

        particles = sample_particles(params, sim, key)
        opt_particles = project_svgd(params, sim, particles)
        opt_error = cost_fun(params, sim, opt_particles)
        min_idx = jnp.argmin(opt_error)
        min_particle = opt_particles[min_idx]
        min_error = opt_error[min_idx]

        _, new_key = jax.random.split(key)
        return new_key, min_particle, min_error

    def cond(state):
        key, min_particle, min_error = state
        return min_error > params.cost_thresh

    _, min_particle, min_error = jax.lax.while_loop(
        cond, solve_once, (key, jnp.zeros((sim.num_blocks, 4)), jnp.inf))
    return min_particle


def make_params(num_blocks: int) -> SvgdParams:
    """Same per-block sampling/opt schedule as spasm.tetris.solve.__main__, plus the
    RBF-repulsion weight."""
    params = SvgdParams()
    if num_blocks == 3:
        params.sampling_batch = 512
        params.opt_batch = 64
        params.opt_steps = 25
    elif num_blocks == 5:
        params.sampling_batch = 4096
        params.opt_batch = 256
        params.opt_steps = 25
        params.cost_thresh = 0.42
    elif num_blocks == 8:
        params.sampling_batch = 2048 * 128
        params.opt_batch = 256
        params.opt_steps = 50
        params.cost_thresh = 0.66
    else:
        raise ValueError('num_blocks must be one of 3, 5, 8.')
    return params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_blocks', type=int, default=5, help="Number of blocks")
    parser.add_argument('--cached', action='store_true', help="view cached solution")
    parser.add_argument('--bench', action='store_true', help="Enable benchmarking")
    parser.add_argument('--use_repulsion', action='store_true',
                        help="Re-enable the RBF-kernel repulsion drift (off by default "
                             "so the solver stays faithful to spasm.tetris.solve)")
    args = parser.parse_args()

    backend.jax_cache_on()

    sim = Simulation(num_blocks=args.num_blocks)
    params = make_params(args.num_blocks)
    params.use_repulsion = args.use_repulsion
    key = jax.random.key(int(time.time()))

    if not args.cached:
        if args.bench:
            opt_particle = solve(params, sim, key)
            opt_particle.block_until_ready()

        start_time = time.perf_counter()
        for _ in range(1 if not args.bench else 10):
            opt_particle = solve(params, sim, key)
        opt_particle.block_until_ready()
        total_time = time.perf_counter() - start_time

        if args.bench:
            print(f"Total time: {total_time * 1000 / 10:.2f} ms")
        else:
            print(f"Total time: {total_time * 1000:.2f} ms")

        os.makedirs('saved', exist_ok=True)
        jnp.save('saved/tetris_svgd.npy', opt_particle)

    print("Loading...")
    opt_particle = jnp.load('saved/tetris_svgd.npy')
    print("Loaded!")

    cost_jit = jax.jit(cost, static_argnames=('params', 'sim'))
    print('Best error', cost_jit(params, sim, opt_particle))
