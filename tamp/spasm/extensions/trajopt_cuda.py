"""Tetris trajopt stage using pyroffi's CUDA SCO trajopt engine
(TrajoptMotionGenerator(..., use_cuda=True), backed by
pyroffi/src/pyroffi/cuda_kernels/trajopt/_sco_trajopt_cuda.py).

This is a sibling of extensions/trajopt_pyroffi.py, reusing its helpers by
import rather than copying: segment construction (_grasp_se3,
_obstacle_spheres), the shared evaluation metric (spasm.tetris.traj.cost),
and the run()-level structure. The only thing that changes is the trajopt
engine itself: use_cuda=True instead of the default JAX SCO path.

Why this exists as a separate script rather than a flag on trajopt_pyroffi.py:
the CUDA SCO kernel does FK + collision entirely inside the CUDA kernel
(see _sco_trajopt_cuda.py's docstring: "FK + finite-difference collision
Jacobians + inner L-BFGS entirely on-device"), so it does NOT go through
pyroffi's `RobotCollisionSpherized.compute_world_collision_distance` at all
-- it sidesteps the exact vmap/flatten-order bug that forced
trajopt_pyroffi.py to run with WORLD_COLLISION_DISABLED=True
(world_geoms=()). So trajopt_cuda.py can and does pass real world_geoms
(the other-block obstacle spheres + floor) through to the optimizer, unlike
its JAX-SCO sibling. This is the honest point of comparison: does CUDA's
"world collision actually works" capability translate into a better
trajectory than the JAX-SCO path (which optimizes smoothness/limits only)
and than spasm.tetris.traj.py's hand-rolled GD (which does optimize
collision, in JAX)?

Same masking convention, same EE_LINK ("panda_hand", since
panda_spherized.urdf has no panda_grasptarget -- see trajopt_pyroffi.py's
docstring for the same caveat, unchanged here), same evaluation metric
(spasm.tetris.traj.cost) for apples-to-apples comparison against both
spasm/tetris_traj.py and extensions/trajopt_pyroffi.py.
"""
import os
import sys
import time
import argparse

import jax
import jax.numpy as jnp


from spasm import backend
import spasm.tetris.traj as spasm_tetris_traj
from spasm.tetris.env import Simulation
from spasm.tetris.traj import TrajOptParams

from pyroffi.motion_generators import TrajoptMotionGenerator
from pyroffi.optimization_engines import ScoTrajOptConfig

# Reuse trajopt_pyroffi.py's helpers verbatim (not copied): grasp-pose
# construction, obstacle-sphere construction, and its loaded planner robot /
# floor geometry (identical URDF, identical EE link convention).
import spasm.extensions.trajopt_pyroffi as tp
from spasm.extensions.trajopt_pyroffi import (
    PLANNER_ROBOT, PLANNER_ROBOT_COLL, FLOOR, EE_LINK,
    _grasp_se3, _obstacle_spheres,
)


def plan_segment_cuda(start_pose_se3, goal_pose_se3, world_geoms, key, n_timesteps=10):
    """Same shape/role as trajopt_pyroffi.plan_segment, but dispatches to
    pyroffi's CUDA SCO kernel and passes real world_geoms through (see
    module docstring for why this is possible here but not in the JAX-SCO
    sibling)."""
    motion_gen = TrajoptMotionGenerator(
        robot=PLANNER_ROBOT,
        robot_coll=PLANNER_ROBOT_COLL,
        world_geoms=world_geoms,
        ee_link_name=EE_LINK,
        n_timesteps=n_timesteps,
        n_batch=8,
        seed_mode='linear_js',
        cartesian_spline_mode='linear',
        use_cuda=True,
        trajopt_cfg=ScoTrajOptConfig(
            n_outer_iters=15,
            n_inner_iters=40,
            w_smooth=5.0,
            w_collision=5.0,
            w_collision_max=50.0,
            collision_margin=0.02,
        ),
    )
    best_traj, costs, _, _, _ = motion_gen.generate(start_pose_se3, goal_pose_se3, key)
    return best_traj[:, :7]


def run(num_blocks, bench=False):
    sim = Simulation(num_blocks=num_blocks)
    initial_state = jnp.array(sim.block_poses_original)

    try:
        final_state = jnp.load('saved/tetris.npy')
    except FileNotFoundError:
        raise SystemExit("Could not find 'saved/tetris.npy'. Run spasm/solve.py first.")
    assert final_state.shape == (num_blocks, 4), \
        f"saved/tetris.npy has {final_state.shape[0]} blocks, expected {num_blocks}"

    num_trajs = 2 * num_blocks - 1
    T = 10  # matches trajopt_pyroffi.py / spasm.tetris.traj's T+2=10

    key = jax.random.PRNGKey(0)
    key, *seg_keys = jax.random.split(key, num_trajs + 1)

    t0 = time.perf_counter()

    pick_place_qs = [None] * num_blocks
    for block_idx in range(num_blocks):
        start_se3 = _grasp_se3(initial_state[block_idx])
        goal_se3 = _grasp_se3(final_state[block_idx])
        world = _obstacle_spheres(sim, block_idx, initial_state, final_state)
        q_traj = plan_segment_cuda(start_se3, goal_se3, (world, FLOOR), seg_keys[2 * block_idx], T)
        pick_place_qs[block_idx] = q_traj

    return_qs = [None] * (num_blocks - 1)
    for i in range(num_blocks - 1):
        start_se3 = _grasp_se3(final_state[i])
        goal_se3 = _grasp_se3(initial_state[i + 1])
        world = _obstacle_spheres(sim, i, initial_state, final_state)
        q_traj = plan_segment_cuda(start_se3, goal_se3, (world, FLOOR), seg_keys[2 * i + 1], T)
        return_qs[i] = q_traj

    q_trajs = jnp.zeros((num_trajs, T, 7))
    for i in range(num_blocks):
        q_trajs = q_trajs.at[2 * i].set(pick_place_qs[i])
    for i in range(num_blocks - 1):
        q_trajs = q_trajs.at[2 * i + 1].set(return_qs[i])

    wall_time = time.perf_counter() - t0

    # Same shared metric as trajopt_pyroffi.py and spasm/tetris_traj.py.
    spasm_tetris_traj.initial_state = initial_state
    params = TrajOptParams()
    initial_state_q_full = q_trajs[:, 0, :]
    metric = float(spasm_tetris_traj.cost(params, sim, initial_state_q_full, q_trajs[:, 1:, :]))

    traj_len = float(jnp.sum(jnp.linalg.norm(q_trajs[:, 1:, :] - q_trajs[:, :-1, :], axis=-1)))

    return q_trajs, wall_time, metric, traj_len, sim


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_blocks', type=int, default=3)
    args = parser.parse_args()

    q_trajs, wall_time, metric, traj_len, sim = run(args.num_blocks)
    print(f"pyroffi-trajopt-cuda: wall time {wall_time*1000:.1f} ms, "
          f"collision-cost metric {metric:.4f}, traj length (sum joint-dist) {traj_len:.4f}")

    os.makedirs('saved', exist_ok=True)
    jnp.save('saved/tetris_traj_cuda.npy', q_trajs)
    print("Saved trajectory to 'saved/tetris_traj_cuda.npy'")
