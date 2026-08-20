"""Tetris simulation environment. Ported from spasm/spasm/tetris_env.py;
meshcat -> viser, and viser server creation is lazy (only on first render()
call) so headless solve/traj runs never open a server/port.
"""
import os
import sys
import time
from typing import List, Literal

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np


from spasm import backend
from spasm.conversions import grasp_to_q, gray_hex, rgb_to_hex, yaw_to_quat_xyz

unit_quat = [1.0, 0.0, 0.0, 0.0]


def create_tetris_spheres(shape: str, sph_radius: float) -> jnp.ndarray:
    _shape_coords = {
        "L": jnp.array([(0, 0, 0), (0, 1, 0), (0, -1, 0), (1, -1, 0)]),
        "O": jnp.array([(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)]),
    }
    coords = _shape_coords[shape]
    num_coords = coords.shape[0]

    spheres = jnp.zeros((num_coords + 2, 4))
    spheres = spheres.at[:num_coords, :3].set(coords * sph_radius * 2)
    spheres = spheres.at[:num_coords, 3].set(sph_radius)

    stick_spheres = jnp.array([
        [0.0, 0.0, -sph_radius * 1.25, sph_radius / 2],
        [0.0, 0.0, -sph_radius * 2, sph_radius / 2]
    ])
    spheres = spheres.at[num_coords:, :].set(stick_spheres)

    z_offset = -spheres[-1, 2]
    spheres = spheres.at[:, 2].add(z_offset)
    return spheres


def create_walls(goal_pose: List[float], goal_dims: List[float], wall_height: float, wall_thickness: float) -> jnp.ndarray:
    cx, cy, cz = goal_pose
    cdx, cdy, _ = goal_dims

    w1_x1 = cx - cdx / 2; w1_y1 = cy + cdy / 2; w1_z1 = cz
    w1_x2 = cx + cdx / 2; w1_y2 = cy + cdy / 2 + wall_thickness; w1_z2 = cz + wall_height

    w2_x1 = cx - cdx / 2; w2_y1 = cy - cdy / 2 - wall_thickness; w2_z1 = cz
    w2_x2 = cx + cdx / 2; w2_y2 = cy - cdy / 2; w2_z2 = cz + wall_height

    w3_x1 = cx - cdx / 2 - wall_thickness; w3_y1 = cy - cdy / 2; w3_z1 = cz
    w3_x2 = cx - cdx / 2; w3_y2 = cy + cdy / 2; w3_z2 = cz + wall_height

    w4_x1 = cx + cdx / 2; w4_y1 = cy - cdy / 2; w4_z1 = cz
    w4_x2 = cx + cdx / 2 + wall_thickness; w4_y2 = cy + cdy / 2; w4_z2 = cz + wall_height

    walls = jnp.array([
        [w1_x1, w1_y1, w1_z1, w1_x2, w1_y2, w1_z2],
        [w2_x1, w2_y1, w2_z1, w2_x2, w2_y2, w2_z2],
        [w3_x1, w3_y1, w3_z1, w3_x2, w3_y2, w3_z2],
        [w4_x1, w4_y1, w4_z1, w4_x2, w4_y2, w4_z2],
    ])
    return walls


def _block_pose_to_spheres(spheres, pose):
    """spheres: (6,4), pose: (4,) [x y z yaw] -> transformed spheres (6,4)."""
    assert spheres.shape == (6, 4), f"spheres should be of shape (6, 4), got {spheres.shape}"
    assert pose.shape == (4,), f"pose should be of shape (4,), got {pose.shape}"

    pose = yaw_to_quat_xyz(pose)
    sphere_pos = spheres[:, :3]
    sphere_r = spheres[:, 3, None]

    pos = pose[:3]
    rot = jaxlie.SO3.from_quaternion_xyzw(pose[3:])

    trans_position = rot.apply(sphere_pos) + pos
    return jnp.concatenate([trans_position, sphere_r], axis=-1)


def block_pose_to_spheres(sim, block_poses):
    assert block_poses.shape == (sim.num_blocks, 4), f"block_poses should be of shape ({sim.num_blocks}, 4), got {block_poses.shape}"
    return jax.vmap(_block_pose_to_spheres, in_axes=(0, 0))(sim.block_spheres, block_poses)


class Simulation:
    def __init__(self, num_blocks: Literal[1, 3, 5, 10, 20]):
        self.num_blocks = num_blocks

        sph_radius: float = 0.03
        wall_height: float = 0.045
        wall_thickness: float = 0.015

        # Lazy viser server: created only on first render() call.
        self._server = None

        self.table_dims = [0.8, 1.5, 0.02]
        self.table_pose = [0.30, 0.0, -0.011, *unit_quat]
        self.table_color = [255, 255, 255]

        L_sphs = create_tetris_spheres("L", sph_radius)
        O_sphs = create_tetris_spheres("O", sph_radius)
        L_block_z = (L_sphs[:, 2] + L_sphs[:, 3]).max() - (L_sphs[:, 2] - L_sphs[:, 3]).min() - 1e-2
        O_block_z = (O_sphs[:, 2] + O_sphs[:, 3]).max() - (O_sphs[:, 2] - O_sphs[:, 3]).min() - 1e-2

        block_poses = [
            [0.50, 0.35, O_block_z, 0],
            [0.15, -0.6, L_block_z, 0],
            [0.00, 0.6, L_block_z, 0],
            [0.15, 0.6, L_block_z, 0],
            [0.00, -0.6, L_block_z, 0],
            [0.50, -0.3, O_block_z, 0],
            [0.50, -0.1, O_block_z, 0],
            [0.50, 0.1, O_block_z, 0],
        ]

        assert jnp.isclose(L_block_z, O_block_z).all(), "Block z offsets should be the same."
        self.block_z = L_block_z

        all_block_spheres = [O_sphs, L_sphs, O_sphs, O_sphs, O_sphs, L_sphs, L_sphs, L_sphs]
        self.block_colors = [0xe81416, 0xffa500, 0xfaeb36, 0x79c314, 0x487de7, 0x87369d, 0x5eb40d, 0xffa500]

        def hex_to_rgb(h):
            return ((h >> 16) & 0xFF, (h >> 8) & 0xFF, h & 0xFF)

        self.block_colors = [hex_to_rgb(c) if isinstance(c, int) else c for c in self.block_colors]

        pastel_factor = 0.6
        self.block_colors = [
            [int(v * pastel_factor + 255 * (1 - pastel_factor)) for v in c]
            for c in self.block_colors
        ]

        indices = list(range(self.num_blocks))
        if self.num_blocks == 2:
            indices = [1, 2]

        self.block_spheres = jnp.array([all_block_spheres[i] for i in indices])
        self.num_blocks = self.block_spheres.shape[0]
        self.num_spheres = self.block_spheres.shape[1]
        self.block_poses = [jnp.array(block_poses[i]) for i in indices]
        self.block_poses_original = self.block_poses.copy()

        self.block_poses_matrix = {}

        diameter = sph_radius * 2

        match self.num_blocks:
            case 1:
                goal_wideness = 2; goal_tallness = 2
            case 3:
                goal_wideness = 6; goal_tallness = 2
            case 5:
                goal_wideness = 10; goal_tallness = 2
            case 8:
                goal_wideness = 16; goal_tallness = 2
            case _:
                raise ValueError("num_blocks must be one of 1, 3, 5, 9.")

        buffer = sph_radius * 1.0
        goal_wideness = goal_wideness * diameter + buffer
        goal_tallness = goal_tallness * diameter + buffer

        self.goal_dims = jnp.array([goal_tallness, goal_wideness, 0.01])
        self.goal_position = jnp.array([0.3, 0.0, -0.005])
        self.goal_color = [255, 255, 255]

        self.goal_walls = create_walls(self.goal_position, self.goal_dims, wall_height, wall_thickness)
        self.wall_color = [255, 255, 255]

        self.q = Simulation.get_neutral_pose()
        self.qmins, self.qmaxes = backend.get_joint_limits()

    def _ensure_server(self):
        """Lazily create the viser server; never called on the solve path."""
        if self._server is None:
            import viser
            self._server = viser.ViserServer()
            self._server.scene.add_grid("/ground", width=2, height=2)

    def set_robot_pose(self, q: jnp.ndarray):
        if q.shape[0] == 7:
            q = jnp.pad(q, (0, 2))
        assert q.shape == (9,), f"q should be of shape (9,), got {q.shape}"
        self.q = q

    @staticmethod
    def get_neutral_pose():
        return jnp.array([0., -jnp.pi / 4, 0., -2 * jnp.pi / 4, 0., jnp.pi / 2, jnp.pi / 4, 0., 0.])

    def set_state(self, block_poses: jnp.ndarray):
        assert block_poses.shape == (self.num_blocks, 4), f"block_poses should be of shape ({self.num_blocks}, 4), got {block_poses.shape}"
        self.block_poses = block_poses

    def set_one_state(self, block_idx: int, block_pose: jnp.ndarray):
        assert 0 <= block_idx < self.num_blocks
        assert block_pose.shape == (4,)
        self.block_poses = self.block_poses.at[block_idx].set(block_pose.copy())

    def reset_state(self):
        self.block_poses = self.block_poses_original
        self.q = Simulation.get_neutral_pose()

    def step(self):
        pass

    def render(self):
        """Renders the environment in viser. Starts the viser server on first call."""
        import viser.transforms as vtf

        self._ensure_server()
        server = self._server

        server.scene.add_box("/table", *self.table_dims, color=self.table_color,
                              position=tuple(float(v) for v in self.table_pose[:3]))

        goal_dims = [float(v) for v in self.goal_dims]
        goal_position = [float(v) for v in self.goal_position]
        server.scene.add_box("/goal", *goal_dims, color=self.goal_color, position=tuple(goal_position))

        for i, wall_aabb in enumerate(self.goal_walls):
            x1, y1, z1, x2, y2, z2 = [float(v) for v in wall_aabb]
            dims = [x2 - x1, y2 - y1, z2 - z1]
            pose = (float((x1 + x2) / 2), float((y1 + y2) / 2), float((z1 + z2) / 2))
            server.scene.add_box(f"/wall_{i}", *dims, color=self.wall_color, position=pose)

        assert np.isfinite(self.block_poses).all(), f"Invalid block_poses: {self.block_poses}"

        for i in range(self.num_blocks):
            if i in self.block_poses_matrix:
                transform_matrix = self.block_poses_matrix.pop(i)
                assert transform_matrix.shape == (4, 4)
                spheres_h = jnp.pad(self.block_spheres[i, :, :3], ((0, 0), (0, 1)), constant_values=1.0)
                transformed_spheres_h = spheres_h @ transform_matrix.T
                transformed_pos = transformed_spheres_h[:, :3]
                transformed_spheres = jnp.concatenate([transformed_pos, self.block_spheres[i, :, 3, None]], axis=-1)
            else:
                transformed_spheres = _block_pose_to_spheres(self.block_spheres[i], self.block_poses[i])

            spheres_np = np.asarray(transformed_spheres)
            for j, sphere_data in enumerate(spheres_np):
                sphere_data = [float(v) for v in sphere_data]
                sphere_pos, sphere_radius = sphere_data[:3], sphere_data[3]
                server.scene.add_icosphere(f"/block_{i}_{j}", radius=sphere_radius,
                                            color=self.block_colors[i], position=tuple(sphere_pos))

        robot_pos, robot_radii = backend.fk(self.q[:7])
        robot_pos_np = np.asarray(robot_pos)
        robot_radii_np = np.asarray(robot_radii)
        for j in range(robot_pos_np.shape[0]):
            server.scene.add_icosphere(f"/robot_{j}", radius=float(robot_radii_np[j]),
                                        color=(200, 200, 200), position=tuple(robot_pos_np[j]))

    def animate_trajectory(self, traj):
        goto_trajs, goto_traj_qs, place_trajs, place_traj_qs = traj
        N = self.num_blocks

        assert goto_traj_qs.shape[0] == N
        assert place_traj_qs.shape[0] == N

        def animate_segment(block_poses, q_poses, holding_idx=None):
            for block_pose, q in zip(block_poses, q_poses):
                if holding_idx is not None:
                    q = q.at[-2:].set(0.06)
                else:
                    q = q.at[-2:].set(0.0)
                self.set_robot_pose(q)
                if holding_idx is not None:
                    self.block_poses[holding_idx] = block_pose
                self.render()

        self.reset_state()
        for block_idx in range(N):
            animate_segment(goto_trajs[block_idx], goto_traj_qs[block_idx], None)
            animate_segment(place_trajs[block_idx], place_traj_qs[block_idx], block_idx)

    def draw_trajs(self, q_trajs: jnp.ndarray):
        self._ensure_server()
        server = self._server
        assert q_trajs.shape[0] == self.num_blocks * 2 - 1
        for i in range(self.num_blocks * 2 - 1):
            points = []
            for t in range(q_trajs.shape[1]):
                pose = q_trajs[i, t]
                pose = jnp.pad(pose, (0, 2))
                ee_pose = backend.get_ee_pose(pose[:7])[0:3, 3]
                points.append(ee_pose)

            points_np = np.array(points)
            server.scene.add_spline_catmull_rom(
                f"/traj/line_{i}", positions=points_np,
                color=self.block_colors[i // 2])

            for t, position in enumerate(points):
                server.scene.add_icosphere(f"/traj/sphere_{i}_{t}", radius=0.005,
                                            color=self.block_colors[i // 2],
                                            position=tuple(float(v) for v in position))


if __name__ == '__main__':
    from spasm.util import jax_cache_on
    jax_cache_on()

    sim = Simulation(num_blocks=5)

    solutions = jnp.load('saved/tetris.npy')
    sim.set_state(solutions)

    sim.render()

    grasp_to_qf = jax.jit(grasp_to_q)
