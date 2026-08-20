"""Tower trajopt stage. Ported from spasm/spasm/tower_traj.py; FK/get_ee_pose
now come from backend.py (pyroffi), same cost terms/schedules/hand-rolled GD
and boundary re-stitching.
"""
from functools import partial
import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np


from spasm import backend
from spasm.tower.env import TowerSimulation
from spasm.tower.solve import cost as tower_cost, block_block_penetration, spheres_blocks_collision
from spasm.conversions import interp, matrix_to_xyzyaw, q_traj_init


class TrajOptParams:
    def __init__(self):
        self.trajopt_steps = 40
        self.trajopt_lr = 0.02

        self.arm_collision_weight = 0.6
        self.block_collision_weight = 0.20
        self.orientation_weight = 0.60
        self.shortness_weight = 0.5
        self.tower_cost_weight = 2.5

        self.viewopt = False


def error_to_down(q):
    """Computes the error between the current robot hand pose and a target pose."""
    target_rot = jaxlie.SO3.from_z_radians(jnp.pi)

    current_pose_mat = backend.get_ee_pose(q)
    current_rot = jaxlie.SO3.from_matrix(current_pose_mat[:3, :3])

    error_rot = target_rot.inverse() @ current_rot
    axis_angle = error_rot.log()

    error_vec = axis_angle * 0.4
    return jnp.sum(jnp.abs(error_vec))


def block_traj_blocks_collision(sim, block_traj, blocks, scale):
    """
    Computes collision between a block trajectory and a collection of static blocks.
    block_traj: (T, 4)
    blocks: (num_blocks, 4)
    """
    assert block_traj.ndim == 2 and block_traj.shape[1] == 4, f'Expected (T, 4), got {block_traj.shape}'
    assert blocks.ndim == 2 and blocks.shape[1] == 4, f'Expected (num_blocks, 4), got {blocks.shape}'

    penetrations = jax.vmap(
        lambda traj_pose: jax.vmap(
            block_block_penetration, in_axes=(None, None, 0, None)
        )(sim, traj_pose, blocks, scale)
    )(block_traj)
    return penetrations.sum(axis=0)


def sweep_spheres_collision(spheres_xyz, spheres_radii, s2, s2r):
    assert spheres_xyz.shape[1] == 3, f'Expected (N, 3), got {spheres_xyz.shape}'
    assert spheres_radii.ndim == 1, f'Expected (N,), got {spheres_radii.shape}'
    assert s2.shape[1] == 3
    assert s2r.ndim == 1

    block_centers = s2
    dists = jnp.linalg.norm(spheres_xyz[:, None, :] - block_centers[None, :, :], axis=-1)
    dists = dists - (spheres_radii[:, None] + s2r[None, :]) - 10e-2
    dists = jnp.where(dists < 0, -dists, 0.0)
    return dists.sum()


def cost(params: TrajOptParams, sim: TowerSimulation, initial_state, q_trajs, i):
    """
    Computes the total cost for a batch of trajectories.
    q_trajs: (num_trajs, T, 7)
    """
    num_blocks = sim.num_blocks
    num_trajs = 2 * num_blocks - 1
    assert q_trajs.ndim == 3, f'Expected (num_trajs, T, 7), got {q_trajs.shape}'
    assert q_trajs.shape[0] == num_trajs
    assert q_trajs.shape[2] == 7, f'Expected (num_trajs, T, 7), got {q_trajs.shape}'
    assert initial_state.shape == (num_trajs, 7), f'Expected (num_trajs, 7), got {initial_state.shape}'

    q_to_block = lambda q: matrix_to_xyzyaw(backend.get_ee_pose(q))
    q_to_block_vmap = jax.vmap(jax.vmap(q_to_block))

    # --- Convert q to poses ---
    ee_poses_xyzyaw = q_to_block_vmap(q_trajs)
    initial_poses = ee_poses_xyzyaw[:, 0, :]
    final_poses = ee_poses_xyzyaw[:, -1, :]

    # --- Robot Arm Collision Cost ---
    def arm_collision_cost_fn(q_traj, traj_idx):
        """
        Computes collision cost for the arm of a single trajectory against other blocks and obstacles.
        - Trajectory `traj_idx` arm collides with initial blocks `j > block_idx`.
        - Trajectory `traj_idx` arm collides with final blocks `j < block_idx`.
        - Trajectory `traj_idx` arm collides with obstacles.
        """
        block_idx = traj_idx // 2

        def single_q_cost(q):
            spheres, radii = backend.fk(q)

            initial_mask = jnp.arange(num_blocks) > block_idx
            cost_initial = spheres_blocks_collision(sim, spheres, radii, initial_poses[::2])
            assert cost_initial.shape == initial_mask.shape, f"cost_initial shape should be {initial_mask.shape}, got {cost_initial.shape}"
            cost_initial = jnp.sum(cost_initial * initial_mask)

            final_mask = jnp.arange(num_blocks) < block_idx
            cost_final = spheres_blocks_collision(sim, spheres, radii, final_poses[::2])
            assert cost_final.shape == final_mask.shape, f"cost_final shape should be {final_mask.shape}, got {cost_final.shape}"
            cost_final = jnp.sum(cost_final * final_mask)

            cost_obstacles = sweep_spheres_collision(spheres, radii, sim.obstacle_poses, sim.obstacle_radii)

            cost_ground = jax.nn.relu(radii - spheres[:, 2] + 2e-2).sum() * 10

            return cost_initial + cost_final + cost_obstacles + cost_ground

        return jax.vmap(single_q_cost)(q_traj).sum()

    arm_collision_cost = jax.vmap(arm_collision_cost_fn, in_axes=(0, 0))(q_trajs, jnp.arange(num_trajs)).sum()
    arm_collision_cost *= params.arm_collision_weight

    # --- Robot Orientation Cost ---
    orientation_cost = (jax.vmap(jax.vmap(error_to_down))(q_trajs[:, [0, -1], :])).sum() * params.orientation_weight

    # --- Held Block Collision Cost ---
    def held_block_collision_cost_fn(ee_poses, block_idx):
        """
        Computes collision cost for the held block.
        - The final pose of the held block for trajectory `block_idx` collides with final poses of blocks `j < block_idx`.
        - The trajectory of the held block for trajectory `block_idx` collides with initial poses of blocks `j > block_idx`.
        """
        assert ee_poses.ndim == 2 and ee_poses.shape[1] == 4, f'Expected (T, 4), got {ee_poses.shape}'

        initial_mask = jnp.arange(num_blocks) > block_idx
        traj_block_collision = block_traj_blocks_collision(sim, ee_poses, initial_poses[::2], 1.0)
        assert traj_block_collision.shape == initial_mask.shape, f"traj_block_collision shape should be {initial_mask.shape}, got {traj_block_collision.shape}"
        traj_block_collision = jnp.sum(traj_block_collision * initial_mask)

        final_mask = jnp.arange(num_blocks) < block_idx
        final_block_collision = block_traj_blocks_collision(sim, ee_poses, final_poses[::2], 1.2)
        assert final_block_collision.shape == final_mask.shape, f"final_block_collision shape should be {final_mask.shape}, got {final_block_collision.shape}"
        final_block_collision = jnp.sum(final_block_collision * final_mask)

        cost_ground = jax.nn.relu(-ee_poses[:, 2] + 2e-2 + sim.block_height).sum() * 10

        return final_block_collision + traj_block_collision + cost_ground

    held_block_collision_cost = jax.vmap(held_block_collision_cost_fn, in_axes=(0, 0))(
        ee_poses_xyzyaw[::2], jnp.arange(num_blocks)
    ).sum()
    held_block_collision_cost *= params.block_collision_weight

    # --- Trajectory Shortness Cost ---
    shortness_cost = jnp.linalg.norm(q_trajs[:, 1:, :] - q_trajs[:, :-1, :], axis=-1).sum()
    shortness_cost += jnp.linalg.norm(q_trajs[:, 0, :] - initial_state, axis=-1).sum()

    second_last_pose = jax.lax.stop_gradient(q_trajs[:, -2, :])
    last_pose = q_trajs[:, -1, :]
    diff = jnp.linalg.norm(last_pose - second_last_pose, axis=-1).sum()

    shortness_cost *= params.shortness_weight

    # --- Final Tower Cost ---
    tower_cost_val = tower_cost(sim, final_poses[::2], initial_state[::2]) * params.tower_cost_weight

    if params.viewopt:
        jax.debug.print("Arm collision cost: {:.2f}, Orientation cost: {:.2f}, Held block collision cost: {:.2f}, Shortness cost: {:.2f}, Tower cost: {:.2f}",
                         arm_collision_cost, orientation_cost, held_block_collision_cost, shortness_cost, tower_cost_val)

    shortness_schedule = jnp.where((i > 0) & (i < 20), 1 - i / 20, 0.0)
    total_cost = 1.6 * tower_cost_val + arm_collision_cost + orientation_cost + held_block_collision_cost + shortness_cost * shortness_schedule
    return total_cost


@partial(jax.jit, static_argnames=('params', 'sim'))
def opt(params: TrajOptParams, sim: TowerSimulation, initial_state, final_state):
    """
    Optimizes a single trajectory segment. Only the interpolated points are modified.
    initial_state: (num_blocks, 4)
    final_state: (num_blocks, 4)
    """
    T = 10
    num_blocks = sim.num_blocks
    num_trajs = 2 * num_blocks - 1

    q_trajs_init = q_traj_init(initial_state, final_state, T)
    assert q_trajs_init.shape == (num_trajs, T + 2, 7), f'Expected ({num_trajs}, {T + 2}, 7), got {q_trajs_init.shape}'

    def schedule_lr(init_lr, step, total_steps):
        return (1.0 - step / total_steps) * init_lr

    def opt_step(i, q_trajs):
        lr = schedule_lr(params.trajopt_lr, i, params.trajopt_steps)
        grad = jax.grad(cost, argnums=3)(params, sim, q_trajs[:, 0, :], q_trajs[:, 1:, :], i)
        q_trajs = q_trajs.at[:, 1:, :].add(grad * -lr)

        # Set the return trajectories' start and end
        return_starts = q_trajs[::2, -1, :][:-1]
        q_trajs = q_trajs.at[1::2, 0, :].set(return_starts)

        return_ends = q_trajs[::2, 0, :][1:]
        q_trajs = q_trajs.at[1::2, -1, :].set(return_ends)

        if params.viewopt:
            def callback(q_trajs):
                final_poses = jax.vmap(lambda q: matrix_to_xyzyaw(backend.get_ee_pose(q)))(q_trajs[::2, -1, :])
                sim.set_state(final_poses)
                sim.draw_trajs(q_trajs)
                sim.render()
            jax.debug.callback(callback, q_trajs)

        return q_trajs

    opt_q_traj = jax.lax.fori_loop(0, params.trajopt_steps, opt_step, q_trajs_init)
    return opt_q_traj


if __name__ == '__main__':
    from spasm.util import jax_cache_on
    jax_cache_on()

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_blocks', type=int, default=10, help="Number of blocks (must match saved/tower.npz)")
    parser.add_argument('--num_obs', type=int, default=10, help="Number of obstacles (0, 1, 10)")
    parser.add_argument('--bench', action='store_true', help='Time benchmark.')
    parser.add_argument('--viewopt', action='store_true', help='Enable view optimization (opens a viser server).')
    parser.add_argument('--render', action='store_true', help='Render trajectory playback (opens a viser server).')
    args = parser.parse_args()

    try:
        data = jnp.load('saved/tower.npz')
        solutions = data['opt_particles']  # (num_solutions, num_blocks, 4)
        init_state = data['init_state']    # (num_blocks, 4)
    except FileNotFoundError:
        print("Could not find 'saved/tower.npz'. Please run 'spasm/tower_solve.py' first.")
        exit()

    params = TrajOptParams()
    params.viewopt = args.viewopt
    sim = TowerSimulation(num_blocks=init_state.shape[0], num_obs=args.num_obs)
    sim.z_error_mul = 5.0
    sim.set_state(solutions[0])
    if args.viewopt or args.render:
        sim.render()

    if sim.num_obs == 0:
        params.trajopt_steps = 10
    if sim.num_obs == 1:
        params.trajopt_steps = 30

    initial_cost_trajs = q_traj_init(init_state, solutions[0], 10)
    initial_cost = cost(params, sim, initial_cost_trajs[:, 0, :], initial_cost_trajs[:, 1:, :], 0)
    print(f"Initial cost: {float(initial_cost):.4f}")

    # Warm up
    if args.bench:
        opt_traj = opt(params, sim, init_state, solutions[0])
        opt_traj.block_until_ready()

    begin = time.perf_counter()
    for _ in range(10 if args.bench else 1):
        q_opt_traj = opt(params, sim, init_state, solutions[0])
        q_opt_traj.block_until_ready()
    end = time.perf_counter()

    if args.bench:
        print(f"Average optimization time: {(end - begin) * 1000 / 10:.2f} ms")
    else:
        print(f"Time: {(end - begin) * 1000:.2f} ms")

    final_cost = cost(params, sim, q_opt_traj[:, 0, :], q_opt_traj[:, 1:, :], params.trajopt_steps)
    print(f"Final cost: {float(final_cost):.4f}")

    os.makedirs('saved', exist_ok=True)
    jnp.save('saved/tower_traj.npy', q_opt_traj)
    print("Saved trajectory to 'saved/tower_traj.npy'")

    if args.render:
        q_to_block = lambda q: matrix_to_xyzyaw(backend.get_ee_pose(q))
        q_to_block_jit = jax.jit(q_to_block)

        sim.set_state(init_state)
        sim.render()

        for traj_idx in range(q_opt_traj.shape[0]):
            block_idx = traj_idx // 2
            is_pick_place = (traj_idx % 2 == 0)

            q_interp = interp(q_opt_traj[traj_idx], 0.03)

            for time_idx in range(q_interp.shape[0]):
                ee = backend.get_ee_pose(q_interp[time_idx])
                if is_pick_place:
                    sim.block_poses_matrix[block_idx] = np.asarray(ee)
                sim.set_robot_pose(q_interp[time_idx])
                sim.render()

            if is_pick_place:
                sim.set_one_state(block_idx, q_to_block_jit(q_opt_traj[traj_idx, -1]))
                sim.render()
                time.sleep(0.5)
