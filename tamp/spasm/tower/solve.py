"""Tower-stacking placement CSP solver. Ported from spasm/spasm/tower_solve.py
verbatim in structure/hyperparameters/cost math; only the sim import moved to
spasm/tower_env.py (viser instead of meshcat/pinocchio).
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


from spasm.tower.env import TowerSimulation


class SpasmParams:
    def __init__(self):
        self.sampling_batch = 128
        self.opt_batch = 1
        self.opt_steps = 10
        self.viewopt = False
        self.noopt = False

        self.lr = 0.06

    def __repr__(self):
        attrs = ', '.join(f'{k}={v}' for k, v in self.__dict__.items())
        return f"SpasmParams({attrs})"

    def hash(self):
        return hash(tuple((k, v) for k, v in self.__dict__.items()))

    @staticmethod
    def flatten(params):
        children = tuple(params.__dict__.values())
        aux_data = None
        return children, aux_data

    @staticmethod
    def unflatten(aux_data, children):
        key_order = SpasmParams().__dict__.keys()
        params = SpasmParams()
        for k, v in zip(key_order, children):
            setattr(params, k, v)
        return params


def rotate(xy, yaw):
    assert xy.shape == (2,), f'Expected shape to be (2,), got {xy.shape}'
    x, y = xy[..., 0], xy[..., 1]
    return jnp.array([x * jnp.cos(yaw) - y * jnp.sin(yaw),
                       x * jnp.sin(yaw) + y * jnp.cos(yaw)])


def corners_rotated(sim, yaw):
    one_corner = sim.block_dims[:2] / 2.0
    rotated_corner = rotate(one_corner, yaw)
    corners = jnp.array([[rotated_corner[0], rotated_corner[1]],
                          [-rotated_corner[0], -rotated_corner[1]],
                          [rotated_corner[1], -rotated_corner[0]],
                          [-rotated_corner[1], rotated_corner[0]]])
    return corners


def block_block_penetration(sim, pose_a, pose_b, inflate=1.0):
    """Computes penetration cost between two blocks.
    Args:
        pose_a: The pose of the first block (4,) -> (x, y, z, yaw).
        pose_b: The pose of the second block (4,) -> (x, y, z, yaw).
    Returns:
        A scalar representing the penetration cost.
    """
    yaw_a = pose_a[3]
    yaw_b = pose_b[3]

    half_dims = sim.block_dims[:2] / 2.0 * inflate

    # Get the 4 corners of b in a's frame
    rel_pos = pose_b[:2] - pose_a[:2]
    b_in_a_frame = rotate(rel_pos, -yaw_a)
    b_corners = corners_rotated(sim, yaw_b) + b_in_a_frame

    # Check if any of b is inside a
    penetration_b_in_a = jnp.maximum(0, half_dims - jnp.abs(b_corners)).sum()

    # Get the 4 corners of a in b's frame
    rel_pos_ab = -rel_pos
    a_in_b_frame = rotate(rel_pos_ab, -yaw_b)
    a_corners = corners_rotated(sim, yaw_a) + a_in_b_frame

    # Check if any of a is inside b
    penetration_a_in_b = jnp.maximum(0, half_dims - jnp.abs(a_corners)).sum()

    b_lower = pose_b[2] - sim.block_height / 2.0 * inflate
    b_upper = pose_b[2] + sim.block_height / 2.0 * inflate
    z_touching = (b_lower < pose_a[2] + sim.block_dims[2] / 2.0) & (b_upper > pose_a[2] - sim.block_dims[2] / 2.0)

    return z_touching * (penetration_b_in_a + penetration_a_in_b)


def spheres_blocks_collision(sim, spheres_xyz, spheres_radii, block_states):
    """
    Computes the collision cost between a set of spheres and a set of blocks.

    This is a simplified check that treats blocks as spheres with a radius
    equal to their half-diagonal.

    Args:
        spheres_xyz: (num_spheres, 3).
        spheres_radii: (num_spheres,).
        block_states:(num_blocks, 4) -> (x, y, z, yaw).

    Returns:
        shape (num_blocks,) representing the penetration cost for each block.
    """
    assert spheres_xyz.shape[1] == 3  # (N, 3)
    assert spheres_radii.ndim == 1, spheres_radii.shape  # (N,)
    assert block_states.shape[1] == 4, f"block_states should be of shape (B, 4), got {block_states.shape}"

    block_diagonal = jnp.linalg.norm(sim.block_dims / 2)
    block_centers = block_states[:, :3]

    dists = jnp.linalg.norm(spheres_xyz[:, None, :] - block_centers[None, :, :], axis=-1)

    penetration = (spheres_radii[:, None] + block_diagonal) - dists
    penetration = jnp.maximum(0, penetration)

    return penetration.sum(axis=0)


def cost(sim: TowerSimulation, block_poses, init_poses):
    """
    Calculates the cost for a given arrangement of blocks.
    The cost is a combination of z-height correctness and tower stability.

    Args:
        sim: The simulation environment.
        block_poses: The poses of the blocks (num_blocks, 4) -> (x, y, z, yaw).
        init_poses: Initial poses of the blocks (num_blocks, 4).

    Returns:
        A scalar cost value.
    """
    assert block_poses.shape == (sim.num_blocks, 4), f"block_poses should be of shape ({sim.num_blocks}, 4), got {block_poses.shape}"

    # Iterate from the second block to the top
    # So x_sum[0] = CoM(N)
    #    x_sum[1] = CoM(N, N-1)
    #    x_sum[2] = CoM(N, N-1, N-2)
    #    x_sum[N-1] = CoM(N, N-1, ..., 1)
    # (N, 2)
    sum = jnp.cumsum(block_poses[:, :2], axis=0)

    def sum_from(i, j):
        """
        Mass sum from i to j inclusive (i < j)
        DO NOT LET i BE 0
        returns [x, y]
        """
        return sum[j] - sum[i - 1]

    def stability_cost_for(i, height):
        """Where i [0, N-1] is the block index from the bottom
        height is the num blocks in the stack"""
        com_above = sum_from(i + 1, height - 1) / (height - i - 1)

        base_xy = block_poses[i, :]

        offset = com_above - base_xy[:2]
        offset_rotated = rotate(offset, -base_xy[3])
        block_dims_strict = sim.block_dims[:2] / 2 * 0.1
        outofboundsness = jnp.maximum(0, jnp.abs(offset_rotated) - block_dims_strict)

        return jnp.sum(outofboundsness)

    # 1. Z-height cost
    target_z = jnp.arange(sim.num_blocks) * sim.block_height + sim.block_height / 2.0
    z_error = jnp.sum((block_poses[:, 2] - target_z) ** 2) * sim.z_error_mul

    # Distance from start cost
    transport_error = jnp.sum((block_poses[:, :2] - init_poses[:, :2]) ** 2)
    transport_error += jnp.sum((block_poses[:, 3] - init_poses[:, 3]) ** 2) * 0.1

    # 2. Stability cost for every tower height [2, N] and block index [0, height-2]
    indices = []
    heights = []
    for h in range(2, sim.num_blocks + 1):
        for i in range(h - 1):
            indices.append(i)
            heights.append(h)
    indices = jnp.array(indices)
    heights = jnp.array(heights)

    stability_cost = jax.vmap(stability_cost_for)(indices, heights)
    ops_in = jnp.exp(indices / sim.num_blocks) / jnp.e / 10
    stability_cost = (stability_cost * ops_in).sum()

    # 3. Block-block penetration cost
    block_i = []
    block_j = []
    for i in range(sim.num_blocks):
        for j in range(i + 1, sim.num_blocks):
            block_i.append(i)
            block_j.append(j)

    bbp = lambda i, j: block_block_penetration(sim, block_poses[i], block_poses[j])
    block_penetrations = jax.vmap(bbp)(jnp.array(block_i), jnp.array(block_j))
    block_penetrations = (block_penetrations ** 2).sum()

    # 4. Block-obstacle collision cost
    obs_pene = spheres_blocks_collision(sim, sim.obstacle_poses, sim.obstacle_radii, block_poses)
    obs_pene = jnp.abs(obs_pene).sum()

    return z_error * 11 + stability_cost * 7.0 + block_penetrations * 0.01 + obs_pene * 0.1


def clip_particle(sim: TowerSimulation, block_poses: jnp.ndarray):
    """Clips the block poses to be within the goal region and wraps yaw."""
    xy = block_poses[..., :2]
    z = block_poses[..., 2, None]
    yaw = block_poses[..., 3, None]

    half_dims = sim.goal_dims[:2] / 2.0
    goal_pos_xy = sim.goal_position[:2]

    xy_clipped = jnp.clip(xy - goal_pos_xy, -half_dims, half_dims) + goal_pos_xy

    yaw_wrapped = (yaw + jnp.pi) % (2 * jnp.pi) - jnp.pi

    return jnp.concatenate([xy_clipped, z, yaw_wrapped], axis=-1)


def sample_particles(params: SpasmParams, sim: TowerSimulation, key):
    """Samples a batch of initial block arrangements."""
    key_xy, key_yaw = jax.random.split(key)

    xyz = jax.random.uniform(key_xy, (params.sampling_batch, sim.num_blocks, 3),
                              minval=sim.goal_position - sim.goal_dims / 2,
                              maxval=sim.goal_position + sim.goal_dims / 2)

    yaws = jax.random.uniform(key_yaw, (params.sampling_batch, sim.num_blocks, 1),
                               minval=-jnp.pi, maxval=jnp.pi)

    # (N, num_blocks, 4)
    block_poses = jnp.concatenate([xyz, yaws], axis=-1)

    errors = jax.vmap(cost, in_axes=(None, 0, None))(sim, block_poses, sim.get_initial_state())
    top_indices = jnp.argsort(errors)[:params.opt_batch]
    top_block_poses = block_poses[top_indices]

    return top_block_poses


def opt_step(i, params: SpasmParams, sim: TowerSimulation, particle):
    """Performs a single optimization step."""
    error, J = jax.value_and_grad(cost, argnums=1)(sim, particle, sim.get_initial_state())

    lr = params.lr * (1 - i / params.opt_steps)
    delta = -J * lr

    particle_new = particle + delta
    particle_new = clip_particle(sim, particle_new)

    if params.viewopt:
        def callback(particle, i):
            print(f'viewing step {i}')
            sim.set_state(particle)
            sim.render()
            time.sleep(0.4)
        jax.debug.callback(callback, particle_new, i)

    return particle_new


def project(params: SpasmParams, sim: TowerSimulation, particles):
    """Runs the optimization loop for a batch of particles."""
    opt_stepper = lambda i, particle: opt_step(i, params, sim, particle)

    def opt(particle):
        particle = jax.lax.fori_loop(0, params.opt_steps, opt_stepper, particle)
        return particle

    return jax.vmap(opt)(particles)


@partial(jax.jit, static_argnames=['params', 'sim'])
def solve(params: SpasmParams, sim: TowerSimulation, key):
    """
    Main solver function. Samples initial states and optimizes them.
    """
    key, place_key = jax.random.split(key)
    particles = sample_particles(params, sim, place_key)

    opt_particles = project(params, sim, particles)

    return opt_particles


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_blocks', type=int, default=10, help="Number of blocks")
    parser.add_argument('--num_obs', type=int, default=10, help="Number of obstacles (0, 1, 10)")
    parser.add_argument('--noopt', action='store_true', help="Disable optimization")
    parser.add_argument('--cached', action='store_true', help="View cached solution")
    parser.add_argument('--viewopt', action='store_true', help="View opt steps")
    parser.add_argument('--bench', action='store_true', help="Enable benchmarking")
    parser.add_argument('--render', action='store_true', help="Render final result via viser (opens a server)")
    args = parser.parse_args()

    from spasm.util import jax_cache_on
    jax_cache_on()

    sim_key = jax.random.PRNGKey(int(time.time()))
    sim = TowerSimulation(num_blocks=args.num_blocks, num_obs=args.num_obs)
    if args.render:
        sim.render()
    params = SpasmParams()
    solve_key = jax.random.key(int(time.time()) + 1)

    if not args.cached:
        # Solve
        if args.bench:
            # Warmup
            opt_particles = solve(params, sim, solve_key)
            opt_particles.block_until_ready()

        start_time = time.perf_counter()
        for _ in range(1 if not args.bench else 10):
            opt_particles = solve(params, sim, solve_key)
        opt_particles.block_until_ready()
        total_time = time.perf_counter() - start_time

        if args.bench:
            print(f"Total time: {total_time * 1000 / 10:.2f} ms")
        else:
            print(f"Total time: {total_time * 1000:.2f} ms")

        # Sort
        opt_errors = jax.vmap(cost, in_axes=(None, 0, None))(sim, opt_particles, sim.get_initial_state())
        sorted_indices = jnp.argsort(opt_errors)
        opt_particles = opt_particles[sorted_indices]
        opt_errors = opt_errors[sorted_indices]

        os.makedirs('saved', exist_ok=True)
        jnp.savez('saved/tower.npz', opt_particles=opt_particles, opt_errors=opt_errors, init_state=sim.get_initial_state())

    print("Loading best solution...")
    data = jnp.load('saved/tower.npz')
    opt_particles = data['opt_particles']
    opt_errors = data['opt_errors']
    print("Loaded!")

    print('Top 10 errors:')
    for i in range(min(10, len(opt_errors))):
        print(f'    {i}: {opt_errors[i].item()}')

    if args.render:
        sim.set_state(opt_particles[0])
        sim.render()
        time.sleep(2)
        sim.reset_state()
