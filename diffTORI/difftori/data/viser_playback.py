"""Interactive viser playback of the demonstration dataset.

Serves a browser view of the Panda replaying demonstrations from the zarr, with
the scene's obstacle drawn at its true radius and the demonstrated end-effector
path traced as a spline.  Nothing is re-solved; every quantity is reconstructed
from ``data/state`` (see ``difftori.data.visualize``), so what you see is what
the policy is trained on.

    PYTHONPATH=diffTORI python -m difftori.data.viser_playback
    PYTHONPATH=diffTORI python -m difftori.data.viser_playback --port 8081 \\
        --n-episodes 20

Then open the printed URL.  Controls: pick an episode, play/pause, scrub the
waypoint, and toggle the collision spheres the teacher's clearance term is
actually computed from -- the end-effector is not one of them, which is why a
path can look clear while a link is not.

Joint order is mapped **by name** between pyroffi and the URDF rather than
assumed positional, following the same contract as
``pyroffi.toolbox._exchange``: a viewer that silently reorders joints draws a
pose that is wrong in a way nobody notices.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from difftori.data.dataset import ReplayBuffer
from difftori.data.visualize import _resolve, sphere_clearance, unpack_episodes

DOF = 7


def _joint_permutation(pyroffi_names, viser_names) -> np.ndarray:
    """Index map so ``q[perm]`` is ordered the way ``ViserUrdf`` expects."""
    missing = set(viser_names) - set(pyroffi_names)
    if missing:
        raise RuntimeError(
            f"URDF actuated joints absent from the pyroffi model: {sorted(missing)}")
    lookup = {n: i for i, n in enumerate(pyroffi_names)}
    return np.array([lookup[n] for n in viser_names], dtype=int)


def main(
    data: str = "diffTORI/data/panda_reach_expert.zarr",
    urdf_path: str = "resources/panda/panda_spherized.urdf",
    srdf_path: str = "resources/panda/panda.srdf",
    mesh_dir: str = "resources/panda/meshes",
    n_episodes: int = 12,
    port: int = 8080,
    fps: float = 6.0,
):
    """Serve the dataset; blocks until interrupted."""
    import viser
    import yourdfpy
    from viser.extras import ViserUrdf

    from ioc.robot import problem as prob

    buf = ReplayBuffer.load(_resolve(data))
    n_timesteps = int(buf.meta.get("n_timesteps", 16))
    problem = prob.RobotProblem.load(_resolve(urdf_path), _resolve(srdf_path),
                                     _resolve(mesh_dir), n_timesteps)
    eps = unpack_episodes(buf, problem, n_episodes)
    print(f"{buf.meta.get('task')}: showing {len(eps)}/{buf.n_episodes} episodes")

    server = viser.ViserServer(port=port)
    urdf = yourdfpy.URDF.load(_resolve(urdf_path))
    viser_urdf = ViserUrdf(server, urdf, root_node_name="/robot")
    perm = _joint_permutation(problem.robot.joints.actuated_names,
                              tuple(viser_urdf.get_actuated_joint_names()))

    server.scene.add_grid("/grid", width=2.0, height=2.0)
    obstacle = server.scene.add_icosphere(
        "/obstacle", radius=float(eps[0]["obs_radius"]), color=(220, 90, 90),
        opacity=0.55)
    ee_path = server.scene.add_spline_catmull_rom(
        "/ee_path", positions=eps[0]["ee"], color=(60, 130, 240), line_width=3.0)
    spheres = server.scene.add_point_cloud(
        "/collision_spheres", points=np.zeros((1, 3)), colors=(160, 160, 200),
        point_size=0.02, visible=False)

    with server.gui.add_folder("Episode"):
        gui_ep = server.gui.add_slider("episode", 0, len(eps) - 1, 1, 0)
        gui_info = server.gui.add_text("min clearance", "", disabled=True)
    with server.gui.add_folder("Playback"):
        gui_play = server.gui.add_checkbox("play", True)
        gui_t = server.gui.add_slider("waypoint", 0, len(eps[0]["q"]) - 1, 1, 0)
        gui_fps = server.gui.add_slider("fps", 1.0, 30.0, 1.0, fps)
    gui_spheres = server.gui.add_checkbox("show collision spheres", False)

    def show_episode(i: int) -> None:
        ep = eps[i]
        obstacle.radius = float(ep["obs_radius"])
        obstacle.position = tuple(np.asarray(ep["obs_center"], dtype=float))
        ee_path.positions = np.asarray(ep["ee"], dtype=np.float32)
        gui_t.max = len(ep["q"]) - 1
        gui_t.value = min(gui_t.value, gui_t.max)
        worst = float(ep["clearance"].min())
        gui_info.value = (f"{worst:+.3f} m"
                          + ("  (penetrating)" if worst < 0 else ""))

    def show_pose(i: int, t: int) -> None:
        q = np.asarray(eps[i]["q"][t], dtype=float)
        viser_urdf.update_cfg(q[perm])
        if gui_spheres.value:
            coll = problem.robot_coll.at_config(problem.robot, q[None, :])
            pts = np.asarray(coll.pose.translation()).reshape(-1, 3)
            spheres.points = pts.astype(np.float32)
            spheres.visible = True
        else:
            spheres.visible = False

    gui_ep.on_update(lambda _: (show_episode(gui_ep.value),
                                show_pose(gui_ep.value, gui_t.value)))
    gui_t.on_update(lambda _: show_pose(gui_ep.value, gui_t.value))
    gui_spheres.on_update(lambda _: show_pose(gui_ep.value, gui_t.value))

    show_episode(0)
    show_pose(0, 0)
    print(f"viser: http://localhost:{port}   (Ctrl-C to stop)")

    try:
        while True:
            time.sleep(1.0 / max(gui_fps.value, 1e-3))
            if not gui_play.value:
                continue
            t = gui_t.value + 1
            if t > gui_t.max:
                t = 0
                gui_ep.value = (gui_ep.value + 1) % len(eps)
            gui_t.value = t   # fires on_update, which redraws the pose
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
