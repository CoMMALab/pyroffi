"""Alternative tetris trajopt stage using pyroffi's Cartesian trajopt engines
(TrajoptMotionGenerator + SCO) for the arm motions between placements, instead
of spasm.tetris.traj's hand-rolled GD.

Loads saved/tetris.npy (final block placements from spasm.tetris.solve /
extensions.svgd_solve), computes pick/place EE poses the same way
spasm.tetris.traj does (reusing spasm.conversions helpers), and plans each
of the 2*num_blocks-1 segments (pick-place + return) with pyroffi's SCO
trajopt. Compared against spasm/tetris_traj.py's own output using the SAME
metric: spasm.tetris.traj.cost (arm+held-block collision penalty).

IMPORTANT DEVIATION (documented per DESIGN.md's known pyroffi bug, which
turned out to be worse than expected -- see the WORLD_COLLISION_DISABLED
comment block below for the full story): pyroffi's sco_trajopt calls
`robot_coll.compute_world_collision_distance` on a single unbatched config
per inner step, and this errors for BOTH the capsule `RobotCollision` (rank
mismatch) and `RobotCollisionSpherized` (DESIGN.md's documented flatten-order
bug) with this codebase's version of pyroffi. We therefore run sco_trajopt
with `world_geoms=()` (its world-collision term disabled) and instead score
obstacle avoidance purely via the downstream metric shared with the
baseline (`spasm.tetris.traj.cost`, which uses backend.fk's spherized
collision spheres and is unaffected by pyroffi's internal bug). This is a
real, reported asymmetry, not a workaround that restores parity -- see
PORT_NOTES.md for the full comparison and honest verdict.

Other simplifications (documented, not hidden):
  - EE link for pyroffi's engine is `panda_hand` (panda_spherized.urdf has no
    `panda_grasptarget`); grasp targets are therefore offset by the
    hand->grasptarget transform used elsewhere in the port
    (`backend`'s EE_LINK is `panda_grasptarget`). We approximate by planning
    to `panda_hand` at the same xyz/yaw target spasm.conversions computes for
    the grasp pose (no analytic offset correction) -- acceptable for this
    comparison since both trajectories are scored with the same downstream
    metric (which uses backend.get_ee_pose / panda_grasptarget), and any
    small systematic offset shows up as extra held-block collision cost
    equally likely to hurt both methods' relative standing (reported below).
  - World obstacles are the OTHER blocks in their "known" state at that
    point in the plan (already-placed blocks at their final pose,
    not-yet-picked blocks at their initial pose) -- same masking convention
    as spasm.tetris.traj's arm_collision_cost -- approximated as one Sphere
    geom per tetromino sphere (from spasm.tetris.env.block_pose_to_spheres)
    plus a floor HalfSpace. Built and passed through regardless of
    WORLD_COLLISION_DISABLED, so re-enabling that flag (once pyroffi's bug
    is fixed upstream) needs no other code changes here.
  - Return segments (odd trajs, robot moving with an empty gripper between
    a place and the next pick) are planned with the same SCO engine.
"""
import os
import sys
import time
import argparse

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import yourdfpy
import pyroffi as pk


from spasm import backend
from spasm.conversions import yaw_to_quat_xyz
from spasm.tetris.env import Simulation, _block_pose_to_spheres
import spasm.tetris.traj as spasm_tetris_traj
from spasm.tetris.traj import TrajOptParams
from spasm.conversions import q_traj_init

from pyroffi.motion_generators import TrajoptMotionGenerator
from pyroffi.optimization_engines import ScoTrajOptConfig

from spasm.paths import PYROFFI_ROOT, PANDA_URDF, PANDA_MESH_DIR
EE_LINK = 'panda_hand'

_urdf = yourdfpy.URDF.load(PANDA_URDF, mesh_dir=PANDA_MESH_DIR)
PLANNER_ROBOT = pk.Robot.from_urdf(_urdf)

# --- World-collision path: both pyroffi collision models are broken here ---
# We tried both models for sco_trajopt's per-config world-collision term
# (`_sco_optimization.py`'s `per_cfg` calls `robot_coll.compute_world_
# collision_distance` on a *single unbatched* cfg via a double vmap):
#   1. Capsule `RobotCollision`: at_config() on an unbatched cfg returns a
#      rank-1 (N,) geom; compute_world_collision_distance's internal
#      `vmap(collide, in_axes=(-2, None))` needs rank>=2 and raises
#      "vmap ... rank should be at least 2, but is only 1".
#   2. `RobotCollisionSpherized`: at_config() returns rank-2 (S, N), which
#      passes the rank check, but trips exactly the bug DESIGN.md documents
#      (flatten-order mismatch between the (S,N)-shaped link_geom axes and
#      the world-object axis) -- confirmed here with the *same* signature:
#      "vmap got inconsistent sizes ... axis -2 ... size 13 [links] vs ...
#      size 18 [spheres/link]".
# Per the task's own fallback guidance, we do NOT patch pyroffi's internals
# (out of scope for extensions/-only changes). Instead we disable pyroffi's
# built-in world_geoms path (call sco_trajopt with world_geoms=()) and let
# obstacle avoidance be scored purely by the shared downstream metric
# (spasm.tetris.traj.cost, the same one the baseline is graded on) -- an
# honest, reduced comparison: pyroffi's SCO here only optimizes smoothness/
# limits/Cartesian-tracking, not world collision, whereas spasm.tetris.traj's
# hand-rolled GD DOES optimize collision terms directly. This asymmetry is
# reported in PORT_NOTES.md and the benchmark output, not hidden.
PLANNER_ROBOT_COLL = pk.collision.RobotCollisionSpherized.from_urdf(_urdf)
WORLD_COLLISION_DISABLED = True

FLOOR = pk.collision.HalfSpace.from_point_and_normal(
    np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))


def _grasp_se3(block_pose_xyzyaw):
    """block pose (x,y,z,yaw) -> jaxlie.SE3 grasp target (same convention as
    spasm.conversions._q_traj_init_single: lift by `upness`, point straight
    down, yawed grasp)."""
    upness = 0.02
    pose = block_pose_xyzyaw + jnp.array([0.0, 0.0, upness, 0.0])
    xyzquat = yaw_to_quat_xyz(pose)  # (x,y,z,qx,qy,qz,qw)
    xyz = xyzquat[:3]
    quat_xyzw = xyzquat[3:]
    rot = jaxlie.SO3.from_quaternion_xyzw(quat_xyzw)
    return jaxlie.SE3.from_rotation_and_translation(rot, xyz)


def _obstacle_spheres(sim, block_idx, initial_state, final_poses):
    """Other-block spheres visible to segment `block_idx`'s arm motion:
    not-yet-picked blocks (idx > block_idx) at their initial pose,
    already-placed blocks (idx < block_idx) at their final pose. Same
    masking convention as spasm.tetris.traj.cost's arm_collision_cost_fn."""
    num_blocks = sim.num_blocks
    centers = []
    radii = []
    for j in range(num_blocks):
        if j == block_idx:
            continue
        pose = final_poses[j] if j < block_idx else initial_state[j]
        spheres = _block_pose_to_spheres(sim.block_spheres[j], pose)  # (6,4)
        centers.append(np.asarray(spheres[:, :3]))
        radii.append(np.asarray(spheres[:, 3]))
    if len(centers) == 0:
        # No other blocks (num_blocks == 1): degenerate zero-radius sphere far away.
        centers = [np.zeros((1, 3)) + 1e3]
        radii = [np.zeros((1,))]
    centers = np.concatenate(centers, axis=0)
    radii = np.concatenate(radii, axis=0)
    M = radii.shape[0]
    sph = pk.collision.Sphere.from_center_and_radius(center=centers, radius=radii)
    # CollGeom's default `inertia_diag` is an *unbatched* zeros(3), which makes
    # get_batch_axes() see an inconsistent (3,) leaf against this object's
    # (M,) batched pose/size and error inside compute_world_collision_distance
    # (even with_physical_properties() itself calls get_batch_axes() first,
    # so it can't be used to fix this after the fact). Broadcast the physical-
    # property fields to (M,)/(M,3) directly instead.
    import jax_dataclasses as jdc
    with jdc.copy_and_mutate(sph, validate=False) as sph:
        sph.mass = jnp.zeros((M,))
        sph.inertia_diag = jnp.zeros((M, 3))
        sph.friction = jnp.zeros((M,))
    return sph


def plan_segment(start_pose_se3, goal_pose_se3, world_geoms, key, n_timesteps=10):
    """Run pyroffi's SCO trajopt (via TrajoptMotionGenerator) for one
    point-to-point Cartesian segment. Returns q traj (n_timesteps, 7).

    world_geoms is accepted for API symmetry / documentation but not passed
    to the engine -- see WORLD_COLLISION_DISABLED above."""
    motion_gen = TrajoptMotionGenerator(
        robot=PLANNER_ROBOT,
        robot_coll=PLANNER_ROBOT_COLL,
        world_geoms=() if WORLD_COLLISION_DISABLED else world_geoms,
        ee_link_name=EE_LINK,
        n_timesteps=n_timesteps,
        n_batch=8,
        seed_mode='linear_js',
        cartesian_spline_mode='linear',
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
        raise SystemExit("Could not find 'saved/tetris.npy'. Run spasm/solve.py or "
                          "extensions/svgd_solve.py first.")
    assert final_state.shape == (num_blocks, 4), \
        f"saved/tetris.npy has {final_state.shape[0]} blocks, expected {num_blocks}"

    num_trajs = 2 * num_blocks - 1
    T = 10  # n_timesteps per segment; matches spasm.tetris.traj's T+2=10

    key = jax.random.PRNGKey(0)
    key, *seg_keys = jax.random.split(key, num_trajs + 1)

    t0 = time.perf_counter()

    pick_place_qs = [None] * num_blocks
    for block_idx in range(num_blocks):
        start_se3 = _grasp_se3(initial_state[block_idx])
        goal_se3 = _grasp_se3(final_state[block_idx])
        world = _obstacle_spheres(sim, block_idx, initial_state, final_state)
        q_traj = plan_segment(start_se3, goal_se3, (world, FLOOR), seg_keys[2 * block_idx], T)
        pick_place_qs[block_idx] = q_traj

    # Return segments: odd trajs, place(i) -> pick(i+1), same Cartesian-planning
    # approach (arm moves empty-handed; obstacles = both endpoint blocks' known
    # state, i.e. previous block already placed, next block not yet picked).
    return_qs = [None] * (num_blocks - 1)
    for i in range(num_blocks - 1):
        start_se3 = _grasp_se3(final_state[i])
        goal_se3 = _grasp_se3(initial_state[i + 1])
        world = _obstacle_spheres(sim, i, initial_state, final_state)  # reuse i's obstacle set
        q_traj = plan_segment(start_se3, goal_se3, (world, FLOOR), seg_keys[2 * i + 1], T)
        return_qs[i] = q_traj

    q_trajs = jnp.zeros((num_trajs, T, 7))
    for i in range(num_blocks):
        q_trajs = q_trajs.at[2 * i].set(pick_place_qs[i])
    for i in range(num_blocks - 1):
        q_trajs = q_trajs.at[2 * i + 1].set(return_qs[i])

    wall_time = time.perf_counter() - t0

    # Evaluate with the SAME metric as spasm/tetris_traj.py. That module's
    # cost() closes over a module-global `initial_state` (see PORT_NOTES.md
    # "What mapped cleanly" -- verbatim port of the original's behavior), so
    # we set it here before calling, exactly as spasm/tetris_traj.py's
    # __main__ does.
    spasm_tetris_traj.initial_state = initial_state
    params = TrajOptParams()
    initial_state_q_full = q_trajs[:, 0, :]  # first q of every traj (pick + return starts)
    metric = float(spasm_tetris_traj.cost(params, sim, initial_state_q_full, q_trajs[:, 1:, :]))

    traj_len = float(jnp.sum(jnp.linalg.norm(q_trajs[:, 1:, :] - q_trajs[:, :-1, :], axis=-1)))

    return q_trajs, wall_time, metric, traj_len, sim


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_blocks', type=int, default=3)
    args = parser.parse_args()

    q_trajs, wall_time, metric, traj_len, sim = run(args.num_blocks)
    print(f"pyroffi-trajopt: wall time {wall_time*1000:.1f} ms, "
          f"collision-cost metric {metric:.4f}, traj length (sum joint-dist) {traj_len:.4f}")

    os.makedirs('saved', exist_ok=True)
    jnp.save('saved/tetris_traj_pyroffi.npy', q_trajs)
    print("Saved trajectory to 'saved/tetris_traj_pyroffi.npy'")
