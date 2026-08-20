"""Tetris trajopt stage. Ported from spasm/spasm/tetris_traj.py; FK/get_ee_pose
now come from backend.py (pyroffi), same cost terms/schedules/hand-rolled GD.
"""
import argparse
import os
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import jaxlie


from spasm import backend
from spasm.conversions import (grasp_to_q, interp, matrix_to_xyzyaw, q_traj_init, yaw_to_quat_xyz)
from spasm.tetris.solve import cost as tetris_cost
from spasm.tetris.solve import sphere_sphere_penetration, sphere_wall_penetration
from spasm.tetris.env import Simulation, _block_pose_to_spheres


class TrajOptParams:
    def __init__(self):
        self.trajopt_steps = 20
        self.trajopt_lr = 0.003

        self.arm_collision_weight = 0.005
        self.block_collision_weight = 1.20
        self.orientation_weight = 0.05
        self.shortness_weight = 0.50
        self.tetris_cost_weight = 0.003

        self.viewopt = False
        self.opt = 'linear'


def error_to_down(q):
    """Error between current robot hand pose and pointing straight down."""
    target_rot = jaxlie.SO3.from_z_radians(jnp.pi)

    current_pose_mat = backend.get_ee_pose(q)
    current_rot = jaxlie.SO3.from_matrix(current_pose_mat[:3, :3])

    error_rot = target_rot.inverse() @ current_rot
    axis_angle = error_rot.log()

    error_vec = axis_angle * 0.4
    return jnp.sum(jnp.abs(error_vec))


schleem = 5e-2


def block_traj_collision(sim, block_spheres_traj, other_blocks_spheres):
    def single_step_collision(block_spheres):
        penetrations = jax.vmap(
            lambda other_spheres: sphere_sphere_penetration(block_spheres, other_spheres, schleem).sum()
        )(other_blocks_spheres)
        return penetrations.sum()

    return jax.vmap(single_step_collision)(block_spheres_traj).sum()


def cost(params: TrajOptParams, sim: Simulation, initial_state_q, q_trajs):
    """q_trajs: (num_trajs, T, 7), initial_state_q: (num_trajs, 7) -> scalar."""
    num_blocks = sim.num_blocks
    num_trajs = 2 * num_blocks - 1

    assert q_trajs.ndim == 3
    assert q_trajs.shape[0] == num_trajs
    assert q_trajs.shape[2] == 7
    assert initial_state_q.shape == (num_trajs, 7)

    q_to_block = lambda q: matrix_to_xyzyaw(backend.get_ee_pose(q))
    q_to_block_vmap = jax.vmap(jax.vmap(q_to_block))

    ee_poses_xyzyaw = q_to_block_vmap(q_trajs[::2])
    initial_poses = jax.vmap(q_to_block)(q_trajs[::2, 0, :])
    final_poses = ee_poses_xyzyaw[:, -1, :]

    def arm_collision_cost_fn(q_traj, traj_idx):
        block_idx = traj_idx // 2

        def single_q_cost(q):
            spheres, radii = backend.fk(q)
            arm_spheres = jnp.hstack([spheres, radii[:, None]])

            initial_mask = jnp.arange(num_blocks) > block_idx
            # NOTE: the original SPaSM referenced a module-global `initial_state`
            # only defined under `__main__` (see REPORT.md §2). Derive it from the
            # sim's original block poses so `cost` is self-contained and callable
            # from the in-memory pipeline, not just the trajopt __main__ script.
            initial_state_poses = jnp.asarray(sim.block_poses_original)
            initial_spheres = jax.vmap(_block_pose_to_spheres, in_axes=(0, 0))(sim.block_spheres, initial_state_poses)
            cost_initial = jax.vmap(lambda s: sphere_sphere_penetration(arm_spheres, s, 2e-2).sum())(initial_spheres)
            cost_initial = jnp.sum(cost_initial * initial_mask)

            final_mask = jnp.arange(num_blocks) < block_idx
            final_spheres = jax.vmap(_block_pose_to_spheres, in_axes=(0, 0))(sim.block_spheres, final_poses)
            cost_final = jax.vmap(lambda s: sphere_sphere_penetration(arm_spheres, s, 2e-2).sum())(final_spheres)
            cost_final = jnp.sum(cost_final * final_mask)

            cost_ground = jax.nn.relu(-arm_spheres[:, 2] + arm_spheres[:, 3] + 6e-2).sum()
            cost_walls = sphere_wall_penetration(arm_spheres, sim, 0).sum()

            return cost_initial + cost_final + cost_walls + cost_ground
        return jax.vmap(single_q_cost)(q_traj).sum()

    arm_collision_cost = jax.vmap(arm_collision_cost_fn, in_axes=(0, 0))(q_trajs, jnp.arange(num_trajs)).sum()
    arm_collision_cost *= params.arm_collision_weight

    orientation_cost = (jax.vmap(jax.vmap(error_to_down))(q_trajs[:, [0, -1], :])).sum() * params.orientation_weight

    def held_block_collision_cost_fn(ee_poses, block_idx):
        block_spheres = sim.block_spheres[block_idx]
        block_spheres_traj = jax.vmap(_block_pose_to_spheres, in_axes=(None, 0))(block_spheres, ee_poses)

        initial_mask = jnp.arange(num_blocks) > block_idx
        initial_spheres = jax.vmap(_block_pose_to_spheres, in_axes=(0, 0))(sim.block_spheres, initial_poses)

        traj_block_collision = jax.vmap(lambda s: block_traj_collision(sim, block_spheres_traj, s[None, ...]))(initial_spheres)
        traj_block_collision = jnp.sum(traj_block_collision * initial_mask)

        final_mask = jnp.arange(num_blocks) < block_idx
        final_spheres = jax.vmap(_block_pose_to_spheres, in_axes=(0, 0))(sim.block_spheres, final_poses)
        final_block_collision = jax.vmap(lambda s: sphere_sphere_penetration(block_spheres_traj[-1], s, schleem).sum())(final_spheres)
        final_block_collision = jnp.sum(final_block_collision * final_mask)

        wall_collision = sphere_wall_penetration(block_spheres_traj.reshape(-1, 4), sim).sum()
        ground_collision = jax.nn.relu(-block_spheres_traj[..., 2] + block_spheres_traj[..., 3] + 6e-2).sum() * 10

        return traj_block_collision + final_block_collision + wall_collision + ground_collision

    held_block_collision_cost = jax.vmap(held_block_collision_cost_fn, in_axes=(0, 0))(
        ee_poses_xyzyaw, jnp.arange(num_blocks)
    ).sum()
    held_block_collision_cost *= params.block_collision_weight

    shortness_cost = jnp.linalg.norm(q_trajs[:, 1:, :] - q_trajs[:, :-1, :], axis=-1).sum()
    shortness_cost += jnp.linalg.norm(q_trajs[:, 0, :] - initial_state_q, axis=-1).sum()
    shortness_cost *= params.shortness_weight

    tetris_cost_val = tetris_cost(params, sim, final_poses) * params.tetris_cost_weight

    if params.viewopt:
        jax.debug.print('Arm collision cost: {:.4f}, Orientation cost: {:.4f}, Held block collision cost: {:.4f}, Shortness cost: {:.4f}, Tetris cost: {:.4f}',
                         arm_collision_cost, orientation_cost, held_block_collision_cost, shortness_cost, tetris_cost_val)

    total_cost = arm_collision_cost + orientation_cost + held_block_collision_cost + shortness_cost + tetris_cost_val
    return total_cost


@partial(jax.jit, static_argnames=('params', 'sim'))
def opt(params: TrajOptParams, sim: Simulation, initial_state, final_state):
    T = 8
    num_blocks = sim.num_blocks
    num_trajs = 2 * num_blocks - 1

    q_trajs_init = q_traj_init(initial_state, final_state, T)
    assert q_trajs_init.shape == (num_trajs, T + 2, 7)

    def schedule_lr(init_lr, step, total_steps):
        return (1.0 - step / total_steps) * init_lr

    def opt_step(i, q_trajs):
        lr = schedule_lr(params.trajopt_lr, i, params.trajopt_steps)
        grad = jax.grad(cost, argnums=3)(params, sim, q_trajs[:, 0, :], q_trajs[:, 1:, :])

        grad = grad.at[:, :, 1].multiply(2)
        grad = grad.at[:, :, 3].multiply(2)

        if params.viewopt:
            jax.debug.print('grad {}', grad)

        grad = jnp.nan_to_num(grad, nan=0.0)
        q_trajs = q_trajs.at[:, 1:-1, :].add(-params.trajopt_lr * grad[:, :-1, :])

        return_starts = q_trajs[::2, -1, :][:-1]
        q_trajs = q_trajs.at[1::2, 0, :].set(return_starts)

        return_ends = q_trajs[::2, 0, :][1:]
        q_trajs = q_trajs.at[1::2, -1, :].set(return_ends)

        if params.viewopt:
            def callback(i, q_trajs):
                if i % 2 != 0:
                    return
                final_poses = jax.vmap(lambda q: matrix_to_xyzyaw(backend.get_ee_pose(q)))(q_trajs[::2, -1, :])
                sim.set_state(final_poses)
                sim.draw_trajs(q_trajs)
                sim.render()
                print('Render step')
            jax.debug.callback(callback, i, q_trajs)

        return q_trajs

    opt_q_traj = jax.lax.fori_loop(0, params.trajopt_steps, opt_step, q_trajs_init)
    return opt_q_traj


if __name__ == '__main__':
    from spasm.util import jax_cache_on
    jax_cache_on()

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_blocks', type=int, default=5, help="Number of blocks")
    parser.add_argument('--bench', action='store_true', help='Time benchmark.')
    parser.add_argument('--viewopt', action='store_true', help='Enable view optimization (opens a viser server).')
    parser.add_argument('--render', action='store_true', help='Render trajectory playback (opens a viser server).')
    args = parser.parse_args()

    try:
        solutions = jnp.load('saved/tetris.npy')
    except FileNotFoundError:
        print("Could not find 'saved/tetris.npy'. Please run 'spasm/solve.py' first.")
        exit()

    params = TrajOptParams()
    params.viewopt = args.viewopt
    sim = Simulation(num_blocks=solutions.shape[0])
    if args.viewopt or args.render:
        sim.render()

    initial_state = jnp.array(sim.block_poses_original)
    final_state = solutions

    _q_trajs_init = q_traj_init(initial_state, final_state, 8)
    initial_cost = cost(params, sim, _q_trajs_init[:, 0, :], _q_trajs_init[:, 1:, :])
    print(f"Initial cost: {float(initial_cost):.4f}")

    if args.bench:
        opt_traj = opt(params, sim, initial_state, final_state)
        opt_traj.block_until_ready()

    begin = time.perf_counter()
    for _ in range(10 if args.bench else 1):
        q_opt_traj = opt(params, sim, initial_state, final_state)
        q_opt_traj.block_until_ready()
    end = time.perf_counter()

    if args.bench:
        print(f"Time: {(end - begin) * 1000 / 10:.2f} ms")
    else:
        print(f"Time: {(end - begin) * 1000:.2f} ms")

    final_cost = cost(params, sim, q_opt_traj[:, 0, :], q_opt_traj[:, 1:, :])
    print(f"Final cost: {float(final_cost):.4f}")

    jnp.save('saved/tetris_traj.npy', q_opt_traj)
    print("Saved trajectory to 'saved/tetris_traj.npy'")

    if args.render:
        q_to_block = lambda q: matrix_to_xyzyaw(backend.get_ee_pose(q))
        q_to_block_jit = jax.jit(q_to_block)

        sim.set_state(initial_state)
        sim.render()

        for traj_idx in range(q_opt_traj.shape[0]):
            block_idx = traj_idx // 2
            is_pick_place = (traj_idx % 2 == 0)

            q_interp = interp(q_opt_traj[traj_idx], 0.3)

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
