"""Tetris placement CSP solver (Algorithm 1). Ported from spasm/spasm/solve.py
verbatim in structure/hyperparameters; only the FK import moved to backend.py.
"""
import copy
from functools import partial
import os
import sys
import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np


from spasm import backend
from spasm.tetris.env import Simulation, _block_pose_to_spheres, block_pose_to_spheres


class SpasmParams:
    def __init__(self):
        self.sampling_batch = 512
        self.opt_batch = 64
        self.opt_steps = 15
        self.viewopt = False
        self.noopt = False

        self.lr = 0.6
        self.quadratic_lr = 6
        self.quadratic_opt_steps = 3

        self.cost_thresh = 0.44

        self.opt = 'linear'

        # Resample-move SMC (see solve_smc). These are inert for `solve`.
        self.smc_rounds = 8
        self.smc_T0 = 0.5          # initial (high) annealing temperature
        self.smc_Tf = 0.05         # final (low) annealing temperature
        self.smc_jitter = 0.01     # gaussian mutation scale on survivors (metres/rad)

    def _tree_flatten(self):
        children = None
        aux_data = {k: v for k, v in self.__dict__.items()}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))


def sphere_sphere_penetration(spheres1, spheres2, margin=0.010):
    """spheres1: (N,4), spheres2: (M,4) -> penetration (N,). No self-collision assumed."""
    assert spheres1.shape[1] == 4
    assert spheres2.shape[1] == 4

    spheres1 = spheres1[:, jnp.newaxis, :]
    spheres2 = spheres2[jnp.newaxis, :, :]

    dist = jnp.linalg.norm(spheres1[..., :3] - spheres2[..., :3], axis=-1)
    radii_sum = spheres1[..., 3] + spheres2[..., 3]

    penetration = jnp.maximum(-margin, radii_sum - dist) + margin
    return jnp.abs(penetration)


def sphere_wall_penetration(spheres, sim, rizz_aura=0.010):
    """spheres: (N,4) -> penetration (N,) against the (infinitely thick) goal walls."""
    goal_pose = sim.goal_position
    goal_dims = sim.goal_dims

    sphere_centers = spheres[..., :3]
    sphere_radii = spheres[..., 3]

    x_max_b = goal_pose[0] + goal_dims[0] / 2.0
    x_min_b = goal_pose[0] - goal_dims[0] / 2.0
    y_max_b = goal_pose[1] + goal_dims[1] / 2.0
    y_min_b = goal_pose[1] - goal_dims[1] / 2.0

    aura = rizz_aura * 2
    pen_x_pos = jnp.maximum(-aura, sphere_centers[..., 0] - x_max_b + sphere_radii) + aura
    pen_x_neg = jnp.maximum(-aura, x_min_b - sphere_centers[..., 0] + sphere_radii) + aura
    pen_y_pos = jnp.maximum(-aura, sphere_centers[..., 1] - y_max_b + sphere_radii) + aura
    pen_y_neg = jnp.maximum(-aura, y_min_b - sphere_centers[..., 1] + sphere_radii) + aura

    xy_penetration = jnp.stack([pen_x_pos, pen_x_neg, pen_y_pos, pen_y_neg])
    xy_penetration = jnp.max(xy_penetration, axis=0)

    wall_penetration = jnp.maximum(0, sim.block_z + 0.07 - spheres[..., 2] + sphere_radii)
    floor_penetration = jnp.maximum(0, 0.00 - spheres[..., 2] + sphere_radii)
    z_penetration = jnp.where(xy_penetration > rizz_aura + 1e-6, wall_penetration * 0.6, floor_penetration)

    return jnp.abs(xy_penetration) + jnp.abs(z_penetration)


def cost(params, sim, block_poses):
    """block_poses: (num_blocks, 4) -> scalar wall+sphere collision cost."""
    assert block_poses.shape == (sim.num_blocks, 4)

    sphere_poses = block_pose_to_spheres(sim, block_poses)

    sphere_penetration = 0
    block_i = []
    block_j = []
    for i in range(len(sphere_poses)):
        for j in range(len(sphere_poses)):
            if i == j:
                continue
            block_i.append(i)
            block_j.append(j)

    ssp_func = lambda i, j: sphere_sphere_penetration(sphere_poses[i], sphere_poses[j])
    ssp = jax.vmap(ssp_func)(jnp.array(block_i), jnp.array(block_j))
    sphere_penetration += ssp.sum() if params.opt == 'linear' else (ssp**2).sum()

    wall_penetration = sphere_wall_penetration(sphere_poses.reshape(-1, 4), sim)
    wall_penetration = wall_penetration.sum() if params.opt == 'linear' else (wall_penetration**2).sum()

    return wall_penetration * 3 + sphere_penetration * 0.5


def clip_particle(sim, block_poses):
    xy = block_poses[..., :2]
    yaw = block_poses[..., 3, None]
    half_dims = sim.goal_dims[:2] / 2.0

    xy_clipped = jnp.clip(xy - sim.goal_position[:2], -half_dims, half_dims) + sim.goal_position[:2]
    z_clipped = jnp.full((xy_clipped.shape[0], 1), sim.block_z)
    yaw = (yaw + jnp.pi) % (2 * jnp.pi) - jnp.pi

    return jnp.concatenate([xy_clipped, z_clipped, yaw], axis=-1)


def sample_particles(params: SpasmParams, sim, key):
    xy = jax.random.uniform(key, (params.sampling_batch, sim.num_blocks, 2),
                             minval=-sim.goal_dims[:2] / 2, maxval=sim.goal_dims[:2] / 2)
    xy += sim.goal_position[:2]
    zs = jnp.full((params.sampling_batch, sim.num_blocks, 1), sim.block_z)
    yaws = jax.random.uniform(key, (params.sampling_batch, sim.num_blocks, 1), minval=-jnp.pi, maxval=jnp.pi)

    block_poses = jnp.concatenate([xy, zs, yaws], axis=-1)

    errors = jax.vmap(cost, in_axes=(None, None, 0))(params, sim, block_poses)
    top_indices = jnp.argsort(errors)[:params.opt_batch]
    top_place_poses = block_poses[top_indices]

    return top_place_poses


def opt_step(i, params: SpasmParams, sim, particle):
    J = jax.grad(cost, argnums=2)(params, sim, particle)

    if params.opt == 'linear':
        lr = (1 - i / params.opt_steps) * params.lr
        delta = -J * 5e-3
        delta = delta.at[..., -1].mul(1.0e3)
        particle_new = particle + delta * lr
    else:
        lr = params.quadratic_lr
        delta = -J * 5e-3
        delta = delta.at[..., -1].mul(250)
        particle_new = particle + delta * lr

    particle_new = clip_particle(sim, particle_new)

    if params.viewopt:
        def callback(particle, i):
            print('viewing step', i)
            sim.set_state(particle)
            sim.render()
        jax.debug.callback(callback, particle_new, i)

    return particle_new


def project(params: SpasmParams, sim, particles):
    quad_opt_params = copy.copy(params)
    quad_opt_params.opt = 'quadratic'

    opt_stepp = lambda i, particle: opt_step(i, params, sim, particle)
    quad_opt_stepp = lambda i, particle: opt_step(i, quad_opt_params, sim, particle)

    def opt(particle):
        particle = jax.lax.fori_loop(0, params.opt_steps, opt_stepp, particle)
        particle = jax.lax.fori_loop(0, quad_opt_params.quadratic_opt_steps, quad_opt_stepp, particle)
        return particle

    return jax.vmap(opt)(particles)


@partial(jax.jit, static_argnames=['params', 'sim'])
def solve(params: SpasmParams, sim, key):
    cost_fun = jax.vmap(cost, in_axes=(None, None, 0))

    def solve_once(state):
        key, _, _ = state

        particles = sample_particles(params, sim, key)
        opt_particles = project(params, sim, particles)
        opt_error = cost_fun(params, sim, opt_particles)
        min_idx = jnp.argmin(opt_error)
        min_particle = opt_particles[min_idx]
        min_error = opt_error[min_idx]

        _, new_key = jax.random.split(key)
        return new_key, min_particle, min_error

    def cond(state):
        key, min_particle, min_error = state
        return min_error > params.cost_thresh

    _, min_particle, min_error = jax.lax.while_loop(cond, solve_once, (key, jnp.zeros((sim.num_blocks, 4)), jnp.inf))
    return min_particle


# ---------------------------------------------------------------------------
# Resample-move Sequential Monte Carlo (REPORT.md §6.2)
#
# The stock `solve` above uses a `while_loop` that throws away the ENTIRE
# particle batch whenever the single best particle fails `cost < thresh`. For
# large problems (tetris-8: sampling_batch = 262144) that discards enormous
# amounts of near-feasible work — e.g. a particle with 7 of 8 blocks placed —
# and re-samples from scratch. SMC instead keeps a persistent population and,
# each round, (1) *moves* particles with gradient descent, (2) *reweights* them
# by exp(-cost/T), (3) *resamples* survivors (systematic resampling), and
# (4) *mutates* them with a little jitter to restore diversity. Good partial
# solutions are preferentially propagated instead of destroyed. T is annealed
# from smc_T0 (broad, exploratory) to smc_Tf (sharp, exploitative), mirroring
# the linear->quadratic schedule of a single `project`.
# ---------------------------------------------------------------------------

def systematic_resample(key, weights):
    """Low-variance systematic resampling. weights: (P,) nonneg -> idx (P,)."""
    P = weights.shape[0]
    w = weights / (jnp.sum(weights) + 1e-30)
    positions = (jax.random.uniform(key) + jnp.arange(P)) / P
    cumsum = jnp.cumsum(w)
    cumsum = cumsum.at[-1].set(1.0)  # guard fp drift so the last bin catches 1.0
    return jnp.searchsorted(cumsum, positions)


@partial(jax.jit, static_argnames=['params', 'sim'])
def solve_smc(params: SpasmParams, sim, key):
    """Resample-move SMC placement solver. Drop-in for `solve`: returns the
    single lowest-cost block-pose particle found across all rounds."""
    cost_fun = jax.vmap(cost, in_axes=(None, None, 0))
    vclip = jax.vmap(clip_particle, in_axes=(None, 0))

    key, sk = jax.random.split(key)
    particles = sample_particles(params, sim, sk)  # (opt_batch, num_blocks, 4)

    def anneal_T(r):
        # geometric schedule from T0 -> Tf over smc_rounds
        frac = r / jnp.maximum(1, params.smc_rounds - 1)
        return params.smc_T0 * (params.smc_Tf / params.smc_T0) ** frac

    def round_step(r, state):
        key, particles, best_particle, best_cost = state

        # (1) move: gradient projection (linear + quadratic phases)
        particles = project(params, sim, particles)
        costs = cost_fun(params, sim, particles)

        # track the running elite so resampling can never lose the best solution
        idx_min = jnp.argmin(costs)
        improved = costs[idx_min] < best_cost
        best_particle = jnp.where(improved, particles[idx_min], best_particle)
        best_cost = jnp.where(improved, costs[idx_min], best_cost)

        # (2) reweight (tempered) and (3) systematic resample
        T = anneal_T(r)
        logw = -(costs - jnp.min(costs)) / T
        weights = jnp.exp(logw)
        key, rk, jk = jax.random.split(key, 3)
        idx = systematic_resample(rk, weights)
        particles = particles[idx]

        # (4) mutate survivors to restore diversity (elitism: slot 0 untouched)
        jitter = jax.random.normal(jk, particles.shape) * params.smc_jitter
        jitter = jitter.at[0].set(0.0)
        particles = vclip(sim, particles + jitter)

        return key, particles, best_particle, best_cost

    init = (key, particles, jnp.zeros((sim.num_blocks, 4)), jnp.inf)
    _, _, best_particle, _ = jax.lax.fori_loop(0, params.smc_rounds, round_step, init)
    return best_particle


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_blocks', type=int, default=5, help="Number of blocks")
    parser.add_argument('--noopt', action='store_true', help="Disable optimization")
    parser.add_argument('--cached', action='store_true', help="view cached solution")
    parser.add_argument('--viewopt', action='store_true', help="View opt steps")
    parser.add_argument('--bench', action='store_true', help="Enable benchmarking")
    parser.add_argument('--render', action='store_true', help="Render final result via viser (opens a server)")
    args = parser.parse_args()

    from spasm.util import jax_cache_on
    jax_cache_on()

    sim = Simulation(num_blocks=args.num_blocks)
    params = SpasmParams()
    key = jax.random.key(int(time.time()))

    if args.num_blocks == 3:
        params.sampling_batch = 512
        params.opt_batch = 64
        params.opt_steps = 25
    elif args.num_blocks == 5:
        params.sampling_batch = 4096
        params.opt_batch = 256
        params.opt_steps = 25
        params.cost_thresh = 0.42
    elif args.num_blocks == 8:
        params.sampling_batch = 2048 * 128
        params.opt_batch = 256
        params.opt_steps = 50
        params.cost_thresh = 0.66
    else:
        raise ValueError('num_blocks must be one of 3, 5, 10.')

    if args.noopt:
        params.noopt = True

    if args.viewopt:
        params.opt_batch = 1
        params.viewopt = True

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
        jnp.save('saved/tetris.npy', opt_particle)

    print("Loading...")
    opt_particle = jnp.load('saved/tetris.npy')
    print("Loaded!")

    cost_jit = jax.jit(cost, static_argnames=('params', 'sim'))
    print('Best error', cost_jit(params, sim, opt_particle))

    if args.render:
        sim.set_state(opt_particle)
        sim.render()
        time.sleep(2)
        sim.reset_state()
