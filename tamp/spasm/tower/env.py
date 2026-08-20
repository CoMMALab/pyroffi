"""Tower simulation environment. Ported from spasm/spasm/tower_env.py;
meshcat -> viser (lazy server, only created on first render()); pinocchio
robot rendering (visualization-only in the original) is dropped -- the
robot is drawn from backend.fk() collision spheres, same as tetris_env.py.
"""
import os
import random
import sys
from typing import List, Literal

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np


from spasm import backend
from spasm.conversions import grasp_to_q, gray_hex, rgb_to_hex, yaw_to_quat_xyz


def gray_hexr():
    return [210, 210, 210]


class TowerSimulation:
    def __init__(self, num_blocks: Literal[1, 3, 5, 8, 10], num_obs: int = 10):
        """
        Initializes the simulation environment with blocks for stacking.

        Args:
            num_blocks: The number of blocks to include in the environment.
            num_obs: The number of floating red spheres to include as obstacles.
        """
        self.num_blocks = num_blocks
        self.num_obs = num_obs
        assert num_obs in [0, 1, 10]

        # Custom costs
        self.z_error_mul = 1.0

        self.block_dims = jnp.array([0.06, 0.06, 0.06])
        self.block_height = self.block_dims[2]

        # Lazy viser server: created only on first render() call.
        self._server = None

        self.table_dims = [1.1, 1.5, 0.02]
        self.table_pose = [0.15, 0.0, -0.011]
        self.table_color = [255, 255, 255]

        # Deepness, width, height
        self.goal_dims = jnp.array([0.6, 1.0, 1.0])
        self.goal_position = jnp.array([0.3, 0.0, 0.5])
        self.goal_color = [186, 255, 201, 0.2]

        key = jax.random.PRNGKey(21)

        # Spawn all blocks on the left side of the table
        block_poses = [[0.4 - (i - 5) * 0.12, 0.30, self.block_height / 2.0, 0.0] for i in range(num_blocks // 2, num_blocks)] + \
                      [[0.4 - i * 0.12, 0.5, self.block_height / 2.0, 0.0] for i in range(num_blocks // 2)]
        block_poses = jnp.array(block_poses)

        self.block_poses = block_poses
        self.block_poses_original = self.block_poses.copy()
        self.block_colors = [0xfd3f52, 0xff6b6b, 0xfd7e03, 0xffbc16, 0xa9e507, 0x65d73d, 0x38c188, 0x0cd4ae, 0x02ccd0, 0x31b5e7]
        random.seed(42)
        random.shuffle(self.block_colors)
        self.block_colors = self.block_colors[:num_blocks]

        self.block_poses_matrix = {}

        self.q = TowerSimulation.get_neutral_pose()
        self.qmins, self.qmaxes = backend.get_joint_limits()

        # Add obstacles

        self.obstacle_color = [255, 255, 255]
        obs_key, _ = jax.random.split(key)
        self.obstacle_poses = jnp.array([
            [0.20, 0.5, 0.6],
            [0.15, 0.05, 0.48],  # over the hill
            [0.5, 0.55, 0.4],  # high
            [0.0, -0.4, 0.3],  # low
            [0.0, 0.2, 0.2],
            [0.0, -0.1, 0.9],
            [0.1, -0.2, 0.2],
            [0.2, -0.5, 0.1],
            [0.3, -0.5 - 100, 0.5],
            [0.25, -0.4, 0.0]])

        self.obstacle_radii = jnp.array([
            0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3,
        ])

        if self.num_obs == 0:
            self.obstacle_poses = jnp.zeros((0, 3))
            self.obstacle_radii = jnp.zeros((0,))
        elif self.num_obs == 1:
            self.obstacle_poses = self.obstacle_poses[-1:]
            self.obstacle_radii = self.obstacle_radii[-1:]

    def _ensure_server(self):
        """Lazily create the viser server; never called on the solve path."""
        if self._server is None:
            import viser
            self._server = viser.ViserServer()
            self._server.scene.add_grid("/ground", width=2, height=2)

    def set_robot_pose(self, q: jnp.ndarray):
        """Sets the robot's joint configuration.

        Args:
            q: Joint configuration for the robot. Shape (7,) or (9,).
        """
        if q.shape == (7,):
            q = jnp.pad(q, (0, 2))
        assert q.shape == (9,), f"q should be of shape (9,), got {q.shape}"
        self.q = q

    @staticmethod
    def get_neutral_pose():
        return jnp.array([0., -jnp.pi / 4, 0., -2 * jnp.pi / 4, 0., jnp.pi / 2, jnp.pi / 4, 0., 0.])

    def set_state(self, block_poses: jnp.ndarray):
        """Sets the state of the blocks.

        Args:
            block_poses: An array of poses for each block. Shape (num_blocks, 4).
        """
        assert block_poses.shape == (self.num_blocks, 4), f"block_poses should be of shape ({self.num_blocks}, 4), " \
                                                            f"got {block_poses.shape}"
        self.block_poses = block_poses.copy()

    def set_one_state(self, block_idx: int, block_pose: jnp.ndarray):
        """Sets the state of a single block.

        Args:
            block_idx: Index of the block to set.
            block_pose: Pose for the block. Shape (4,).
        """
        assert 0 <= block_idx < self.num_blocks, f"block_idx should be in [0, {self.num_blocks}), got {block_idx}"
        assert block_pose.shape == (4,), f"block_pose should be of shape (4,), got {block_pose.shape}"
        self.block_poses = self.block_poses.at[block_idx].set(block_pose)

    def reset_state(self):
        """Resets the state of the robot and blocks to the original configuration."""
        self.block_poses = self.block_poses_original.copy()
        self.q = TowerSimulation.get_neutral_pose()

    def get_initial_state(self):
        """Returns the initial state of the blocks. (num_blocks, 4)"""
        return self.block_poses_original.copy()

    def step(self):
        """An empty step function."""
        pass

    def draw_trajs(self, q_trajs: jnp.ndarray):
        """Draws the trajectories of the blocks as lines and spheres at the waypoints.
        trajs: (num_blocks * 2 - 1, T, 7)
        """
        from spasm.conversions import interp

        self._ensure_server()
        server = self._server
        for i in range(self.num_blocks * 2 - 1):
            points = []
            for t in range(q_trajs.shape[1]):
                pose = q_trajs[i, t]
                pose = jnp.pad(pose, (0, 2))
                ee_pose = backend.get_ee_pose(pose[:7])[0:3, 3]
                points.append(ee_pose)

            q_traj_interp = interp(q_trajs[i], dist_per_step=0.05)
            interp_points = []
            for t in range(q_traj_interp.shape[0]):
                pose = q_traj_interp[t]
                pose = jnp.pad(pose, (0, 2))
                ee_pose = backend.get_ee_pose(pose[:7])[0:3, 3]
                interp_points.append(ee_pose)

            points_np = np.array(interp_points)
            server.scene.add_spline_catmull_rom(
                f"/traj/line_{i}", positions=points_np,
                color=self.block_colors[i // 2])

            for t, position in enumerate(points):
                server.scene.add_icosphere(f"/traj/sphere_{i}_{t}", radius=0.005,
                                            color=self.block_colors[i // 2],
                                            position=tuple(float(v) for v in position))

    def render(self):
        """Renders the environment in viser. Starts the viser server on first call."""
        self._ensure_server()
        server = self._server

        server.scene.add_box("/table", *self.table_dims, color=self.table_color,
                              position=tuple(float(v) for v in self.table_pose[:3]))

        assert np.isfinite(np.array(self.block_poses)).all(), f"Invalid block_poses: {self.block_poses}"

        for i, (pose, color) in enumerate(zip(self.block_poses, self.block_colors)):
            if i in self.block_poses_matrix:
                transform_matrix = np.asarray(self.block_poses_matrix.pop(i))
                assert transform_matrix.shape == (4, 4)
                position = tuple(float(v) for v in transform_matrix[:3, 3])
                wxyz = tuple(float(v) for v in jaxlie.SO3.from_matrix(transform_matrix[:3, :3]).wxyz)
            else:
                pose_7d = yaw_to_quat_xyz(pose)
                position = tuple(float(v) for v in pose_7d[:3])
                wxyz = tuple(float(v) for v in jaxlie.SO3.from_quaternion_xyzw(pose_7d[3:]).wxyz)

            r, g, b = (color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF
            server.scene.add_box(f"/block_{i}", *[float(d) for d in self.block_dims],
                                  color=(r, g, b), wxyz=wxyz, position=position)

        # Render obstacles
        for i, (pos, r) in enumerate(zip(self.obstacle_poses, self.obstacle_radii)):
            opacity = 0.5 if i == 9 else 1.0
            server.scene.add_icosphere(f"/obstacle_{i}", radius=float(r), color=tuple(self.obstacle_color),
                                        opacity=opacity, position=tuple(float(v) for v in pos))

        # Render robot as collision spheres (backend.fk); pinocchio path from
        # the original (visualization-only) is dropped.
        robot_pos, robot_radii = backend.fk(self.q[:7])
        robot_pos_np = np.asarray(robot_pos)
        robot_radii_np = np.asarray(robot_radii)
        for j in range(robot_pos_np.shape[0]):
            server.scene.add_icosphere(f"/robot_{j}", radius=float(robot_radii_np[j]),
                                        color=(200, 200, 200), position=tuple(robot_pos_np[j]))


if __name__ == '__main__':
    from spasm.util import jax_cache_on
    jax_cache_on()

    sim = TowerSimulation(num_blocks=10, num_obs=1)
    sim.render()
