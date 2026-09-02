"""The `multistart_behavior*.gif` recording, replayed in a viser web viewer.

Same npz contract as `iosp.viz.behavior3d` (path_hist / cand_hist / demo /
obstacles / waypoints / rmse_hist), rendered interactively instead of baked to
a gif: an outer-step slider, per-scene toggles, and candidates shown or hidden.
The gif's rotating camera exists to disambiguate depth in a static projection;
here you just orbit it yourself.

`--joints` additionally plays the ARM back through the fitted trajectory at
whichever outer step the slider is on, with the manipulated object attached to
the gripper for exactly the rows the task skeleton says it is held
(`pickplace.PHASE_SPAN`: grasped at the end of `grasp`, released at the end of
`transport`), and resting at the pick / place target otherwise.

That needs `q_hist` in the npz -- the configurations behind the EE paths.  The
arm is redundant, so q is NOT recoverable from a recorded EE path; a recording
made before `iosp.record.multistart` started saving `q_hist` therefore cannot
drive the robot, and this viewer says so and falls back to paths-only rather
than inventing a q by post-hoc IK (which would show a different arm motion
from the one that was actually fitted).  Re-record to get it.

Usage:
    python -m iosp.viz.multistart_viser scratch/viz/multistart_behavior_q.npz --joints
"""

from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
import viser

SCENE_COLORS = [(0x3b, 0x7d, 0xd8), (0xd9, 0x53, 0x4f), (0x2e, 0x8b, 0x57)]
DEMO_COLOR = (0x22, 0x22, 0x22)
CAND_COLOR = (0xa0, 0xa0, 0xa0)


def _polyline(server, name, pts, color, width):
    pts = np.asarray(pts, np.float32)
    segs = np.stack([pts[:-1], pts[1:]], axis=1)          # (T-1, 2, 3)
    return server.scene.add_line_segments(
        name, points=segs, colors=np.tile(np.asarray(color, np.uint8), (len(segs), 2, 1)),
        line_width=width)


def _load_urdf():
    import yourdfpy
    root = pathlib.Path(__file__).resolve().parents[2] / "resources" / "panda"
    return yourdfpy.URDF.load(str(root / "panda_spherized.urdf"), load_meshes=True,
                              build_scene_graph=True, mesh_dir=str(root / "meshes"))


def main(npz_path, obs_z=None, show_robot=False, joints=False, port=8080):
    d = np.load(npz_path, allow_pickle=True)
    P = d["path_hist"]                                    # (F, B, T, 3)
    demo = d["demo"]                                      # (B, T, 3)
    C = d["cand_hist"] if "cand_hist" in d.files else None
    rmse = d["rmse_hist"] if "rmse_hist" in d.files else None
    obs = d["obstacles"] if "obstacles" in d.files else None
    way = d["waypoints"] if "waypoints" in d.files else None
    label = str(d["label"]) if "label" in d.files else ""
    Q = d["q_hist"] if "q_hist" in d.files else None          # (F, C, B, T, dof)
    if joints and Q is None:
        print(f"[multistart_viser] {npz_path} has no `q_hist` -- it predates the "
              "recorder saving joint paths.  Robot playback disabled (q cannot be "
              "recovered from EE paths on a redundant arm); re-record with "
              "`python -m iosp.record.multistart ...` to get it.")
        joints = False
    winner = int(d["winner"]) if "winner" in d.files else None
    F, B, T, dim = P.shape
    if dim != 3:
        raise ValueError(f"{npz_path} holds {dim}-D paths; this viewer is 3D-only")
    if obs_z is None:
        from iosp.config import OBS_CENTER
        obs_z = float(OBS_CENTER[2])

    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("+z")

    urdf_vis = None
    if show_robot or joints:
        from viser.extras import ViserUrdf
        from iosp.config import Q_START
        urdf_vis = ViserUrdf(server, _load_urdf(), root_node_name="/robot")
        urdf_vis.update_cfg(np.asarray(Q_START))

    # The manipulated object: a box that sits at the pick target, rides the
    # gripper through `transport`, then sits at the place target.
    obj = None
    if joints:
        obj = server.scene.add_box("/object", dimensions=(0.05, 0.05, 0.05),
                                   color=(0.95, 0.55, 0.15))

    # static scene furniture: obstacles + waypoints, per context
    for b in range(B):
        if obs is not None:
            for j, (ox, oy, r) in enumerate(np.asarray(obs[b])):
                server.scene.add_icosphere(f"/scene{b}/obs{j}", radius=float(r),
                                           position=(float(ox), float(oy), obs_z),
                                           color=(0.6, 0.6, 0.65), opacity=0.55)
        if way is not None:
            for j, w in enumerate(np.asarray(way[b])):
                server.scene.add_icosphere(f"/scene{b}/way{j}", radius=0.012,
                                           position=tuple(float(v) for v in w),
                                           color=(1.0, 0.75, 0.1))
        _polyline(server, f"/scene{b}/demo", demo[b], DEMO_COLOR, 4.0)

    # mutable per-frame geometry
    fit_h: list = [None] * B
    cand_h: dict = {}

    def draw(f: int, show_cands: bool, ctx: list[bool]) -> None:
        for b in range(B):
            if fit_h[b] is not None:
                fit_h[b].remove()
                fit_h[b] = None
            for k in list(cand_h):
                if k[0] == b:
                    cand_h.pop(k).remove()
            if not ctx[b]:
                continue
            if show_cands and C is not None:
                for c in range(C.shape[1]):
                    if winner is not None and c == winner:
                        continue
                    cand_h[(b, c)] = _polyline(
                        server, f"/scene{b}/cand{c}", C[f, c, b], CAND_COLOR, 1.0)
            fit_h[b] = _polyline(server, f"/scene{b}/fit", P[f, b],
                                 SCENE_COLORS[b % len(SCENE_COLORS)], 5.0)

    # EE index and grasp/release rows, straight from the task skeleton.
    grasp_row = rel_row = None
    if joints:
        from iosp.model.pickplace import PHASE_SPAN
        grasp_row = PHASE_SPAN["grasp"][1] - 1                 # object attached here
        rel_row = PHASE_SPAN["transport"][1] - 1               # released here

    def object_pos(b: int, t: int, ee: np.ndarray) -> np.ndarray:
        """Where the box is at path row `t` of context `b`."""
        w = np.asarray(way[b]) if way is not None else None
        if t < grasp_row:
            return w[0] if w is not None else ee
        if t <= rel_row:
            return ee                                          # carried by the gripper
        return w[1] if w is not None else ee

    with server.gui.add_folder("Recording"):
        server.gui.add_markdown(f"**{label}**" if label else "_(no label)_")
        step = server.gui.add_slider("Outer step", 0, F - 1, 1, F - 1)
        play = server.gui.add_checkbox("Play", False)
        cands = server.gui.add_checkbox("Show candidates", C is not None)
        ctx_boxes = [server.gui.add_checkbox(f"Context {b}"
                                             + (" (train)" if b == 0 else " (held out)"), True)
                     for b in range(B)]
        stat = server.gui.add_markdown("")
    if joints:
        with server.gui.add_folder("Robot"):
            cand_sel = server.gui.add_slider("Candidate", 0, Q.shape[1] - 1, 1,
                                             winner if winner is not None else 0)
            ctx_sel = server.gui.add_slider("Context", 0, B - 1, 1, 0)
            t_sel = server.gui.add_slider("Path row", 0, T - 1, 1, 0)
            play_q = server.gui.add_checkbox("Play trajectory", True)
            rob_stat = server.gui.add_markdown("")

    def refresh(_=None) -> None:
        f = int(step.value)
        draw(f, cands.value, [c.value for c in ctx_boxes])
        stat.content = (f"step {f}/{F - 1}"
                        + (f" &nbsp; held-out RMSE **{float(rmse[f]):.4f}**" if rmse is not None else ""))

    def refresh_robot(_=None) -> None:
        if not joints:
            return
        f, c, b, t = int(step.value), int(cand_sel.value), int(ctx_sel.value), int(t_sel.value)
        q = np.asarray(Q[f, c, b, t])
        urdf_vis.update_cfg(q)
        ee = np.asarray(C[f, c, b, t] if C is not None else P[f, b, t])
        obj.position = tuple(float(v) for v in object_pos(b, t, ee))
        held = grasp_row <= t <= rel_row
        rob_stat.content = (f"outer step **{f}** &nbsp; candidate **{c}**"
                            + (" (winner)" if c == winner else "")
                            + f" &nbsp; ctx **{b}** &nbsp; row **{t}**/{T - 1}"
                            + f" &nbsp; object **{'held' if held else 'at rest'}**")

    step.on_update(refresh)
    cands.on_update(refresh)
    for c in ctx_boxes:
        c.on_update(refresh)
    if joints:
        for h in (step, cand_sel, ctx_sel, t_sel):
            h.on_update(refresh_robot)
    refresh()
    refresh_robot()

    print(f"[multistart_viser] {npz_path}: {F} outer steps, {B} contexts, "
          f"{0 if C is None else C.shape[1]} candidates"
          + ("  + joint playback (q_hist)" if joints else
             "  (robot drawn STATIC at Q_START; no joint playback)" if show_robot else ""))
    print(f"Viser server: http://0.0.0.0:{server.get_port()}")
    while True:
        if play.value:
            step.value = (int(step.value) + 1) % F
        if joints and play_q.value:
            t_sel.value = (int(t_sel.value) + 1) % T
        time.sleep(1 / 6)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="?", default="scratch/viz/multistart_behavior.npz")
    ap.add_argument("--obs-z", type=float, default=None)
    ap.add_argument("--robot", action="store_true",
                    help="draw the Panda at config.Q_START for spatial context (static)")
    ap.add_argument("--joints", action="store_true",
                    help="play the arm through the fitted trajectory at the selected "
                         "outer step, with the object picked and placed (needs q_hist)")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    main(a.npz, obs_z=a.obs_z, show_robot=a.robot, joints=a.joints, port=a.port)
