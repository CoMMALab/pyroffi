"""Demonstration vs. reconstruction, played back as ROLLOUTS in viser.

`iosp.viz.multistart_viser` shows the *fit*: the outer optimization walking the
candidate population toward the demo, one outer step per slider tick.  This
module shows the two things that fit is between -- the demonstrator's
trajectory and the trajectory the recovered cost reproduces -- played back as
motion, side by side, on the full TAMP scene rather than as free-space curves.

Both rollouts are drawn in their own copy of the environment (table, obstacle,
pick/place targets, the manipulated box), offset along -y/+y so they can be
watched together.  The box is not decoration: the task skeleton says exactly
which rows of the path hold it (`pickplace.PHASE_SPAN`: grasped at the end of
`grasp`, released at the end of `transport`), so a reconstruction that gets the
arm roughly right but the grasp row wrong is visible here and invisible in an
EE-path plot.

Data comes from a `iosp.record.multistart` npz, which must carry `q_hist` /
`demo_joint` -- the configurations.  The arm is redundant, so q is NOT
recoverable from a recorded EE path; a recording without them cannot drive this
viewer and post-hoc IK would show a different arm motion from the one that was
actually fitted.  Re-record to get them.

By default the reconstruction is the WINNING candidate at the LAST outer step
-- the converged fit, selected on training loss, exactly as the paper reports
it.  `--step` / `--candidate` override that for inspecting a loser or an
intermediate step.

Usage:
    python -m iosp.viz.rollout_viser scratch/viz/multistart_behavior_q.npz
"""

from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
import viser

DEMO_COLOR = (0x22, 0x22, 0x22)
RECON_COLOR = (0x3b, 0x7d, 0xd8)
OBJ_COLOR = (0.95, 0.55, 0.15)


def _polyline(server, name, pts, color, width):
    pts = np.asarray(pts, np.float32)
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    return server.scene.add_line_segments(
        name, points=segs,
        colors=np.tile(np.asarray(color, np.uint8), (len(segs), 2, 1)),
        line_width=width)


def _load_urdf():
    import yourdfpy
    root = pathlib.Path(__file__).resolve().parents[2] / "resources" / "panda"
    return yourdfpy.URDF.load(str(root / "panda_spherized.urdf"), load_meshes=True,
                              build_scene_graph=True, mesh_dir=str(root / "meshes"))


class Rollout:
    """One arm + its own copy of the scene, at a lateral offset."""

    def __init__(self, server, name, offset, q, ee, obs, way, obs_z,
                 color, grasp_row, rel_row, obj_size):
        from viser.extras import ViserUrdf
        self.server, self.name, self.offset = server, name, np.asarray(offset, float)
        self.q, self.ee = q, ee                      # (T, dof), (T, 3)
        self.way = None if way is None else np.asarray(way, float)
        self.grasp_row, self.rel_row = grasp_row, rel_row
        self.color = color

        server.scene.add_frame(f"/{name}", show_axes=False, position=self.offset)
        self.urdf = ViserUrdf(server, _load_urdf(), root_node_name=f"/{name}/robot")

        # table: the arm's workspace, so the scene reads as a scene
        server.scene.add_box(f"/{name}/table", dimensions=(0.9, 1.2, 0.02),
                             position=(0.45, 0.0, -0.01), color=(0.72, 0.68, 0.62))
        if obs is not None:
            for j, (ox, oy, r) in enumerate(np.asarray(obs)):
                server.scene.add_icosphere(
                    f"/{name}/obs{j}", radius=float(r),
                    position=(float(ox), float(oy), obs_z),
                    color=(0.6, 0.6, 0.65), opacity=0.55)
        if self.way is not None:
            for j, w in enumerate(self.way):          # pick / place targets
                server.scene.add_icosphere(
                    f"/{name}/target{j}", radius=0.02, position=tuple(w),
                    color=(1.0, 0.75, 0.1), opacity=0.5)
        _polyline(server, f"/{name}/path", ee, color, 4.0)
        self.obj = server.scene.add_box(f"/{name}/object",
                                        dimensions=(obj_size,) * 3, color=OBJ_COLOR)

        # Carry offset: the gripper stops a standoff short of the object, so
        # riding the box on the raw EE position would teleport it at pickup.
        # Pin the offset at the grasp row and the carry is continuous.
        pick = self.way[0] if self.way is not None else ee[grasp_row]
        self._carry = pick - ee[grasp_row]

    def object_pos(self, t):
        if t < self.grasp_row:
            return self.way[0] if self.way is not None else self.ee[t]
        if t <= self.rel_row:
            return self.ee[t] + self._carry              # held by the gripper
        return self.way[1] if self.way is not None else self.ee[t]

    def set_row(self, t):
        self.urdf.update_cfg(np.asarray(self.q[t]))
        self.obj.position = tuple(float(v) for v in self.object_pos(t))

    def redraw_path(self):
        _polyline(self.server, f"/{self.name}/path", self.ee, self.color, 4.0)

    def set_data(self, q, ee):
        self.q, self.ee = q, ee
        pick = self.way[0] if self.way is not None else ee[self.grasp_row]
        self._carry = pick - ee[self.grasp_row]
        self.redraw_path()


def main(npz_path, step=None, cand=None, ctx=0, obs_z=None, spread=0.75,
         obj_size=0.05, fps=6.0, port=8080):
    d = np.load(npz_path, allow_pickle=True)
    if "q_hist" not in d.files or "demo_joint" not in d.files:
        raise SystemExit(
            f"{npz_path} has no `q_hist`/`demo_joint` -- it predates the recorder "
            "saving joint paths.  q is not recoverable from EE paths on a redundant "
            "arm; re-record with `python -m iosp.record.multistart ...`.")
    Q = d["q_hist"]                                   # (F, C, B, T, dof)
    C_ee = d["cand_hist"]                              # (F, C, B, T, 3)
    demo_q, demo_ee = d["demo_joint"], d["demo"]       # (B, T, dof), (B, T, 3)
    obs = d["obstacles"] if "obstacles" in d.files else None
    way = d["waypoints"] if "waypoints" in d.files else None
    label = str(d["label"]) if "label" in d.files else ""
    winner = int(d["winner"]) if "winner" in d.files else 0
    rmse_fit = d["ee_train_hist"] if "ee_train_hist" in d.files else None
    rmse_held = d["ee_held_hist"] if "ee_held_hist" in d.files else None
    F, nC, B, T, _ = C_ee.shape
    step = F - 1 if step is None else step
    cand = winner if cand is None else cand
    if obs_z is None:
        from iosp.config import OBS_CENTER
        obs_z = float(OBS_CENTER[2])

    from iosp.model.pickplace import PHASE_SPAN, PHASES
    grasp_row = PHASE_SPAN["grasp"][1] - 1
    rel_row = PHASE_SPAN["transport"][1] - 1

    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("+z")

    demo = Rollout(server, "demo", (0.0, +spread, 0.0), demo_q[ctx], demo_ee[ctx],
                   None if obs is None else obs[ctx], None if way is None else way[ctx],
                   obs_z, DEMO_COLOR, grasp_row, rel_row, obj_size)
    recon = Rollout(server, "recon", (0.0, -spread, 0.0), Q[step, cand, ctx],
                    C_ee[step, cand, ctx],
                    None if obs is None else obs[ctx], None if way is None else way[ctx],
                    obs_z, RECON_COLOR, grasp_row, rel_row, obj_size)
    server.scene.add_label("/demo/label", "demonstration", position=(0.0, 0.0, 0.9))
    server.scene.add_label("/recon/label", "reconstruction", position=(0.0, 0.0, 0.9))

    def phase_of(t):
        return next((p for p in PHASES
                     if PHASE_SPAN[p][0] <= t < PHASE_SPAN[p][1]), PHASES[-1])

    with server.gui.add_folder("Playback"):
        server.gui.add_markdown(f"**{label}**" if label else "_(no label)_")
        row = server.gui.add_slider("Path row", 0, T - 1, 1, 0)
        play = server.gui.add_checkbox("Play", True)
        speed = server.gui.add_slider("Steps / sec", 1.0, 30.0, 1.0, fps)
        info = server.gui.add_markdown("")
    with server.gui.add_folder("What is reconstructed"):
        ctx_sel = server.gui.add_slider("Context (0 fit, 1 held out)", 0, B - 1, 1, ctx)
        step_sel = server.gui.add_slider("Outer step", 0, F - 1, 1, step)
        cand_sel = server.gui.add_slider("Candidate", 0, nC - 1, 1, cand)
        fit_info = server.gui.add_markdown("")

    def reload(_=None):
        b, f, c = int(ctx_sel.value), int(step_sel.value), int(cand_sel.value)
        demo.set_data(demo_q[b], demo_ee[b])
        recon.set_data(Q[f, c, b], C_ee[f, c, b])
        e = float(np.sqrt(np.mean(np.sum(
            (np.asarray(C_ee[f, c, b]) - np.asarray(demo_ee[b])) ** 2, -1))))
        rec = rmse_fit if b == 0 else rmse_held
        fit_info.content = (
            f"context **{b}** ({'fit' if b == 0 else 'held out'}) &nbsp; "
            f"outer step **{f}**/{F - 1} &nbsp; candidate **{c}**"
            + (" (winner)" if c == winner else "")
            + f"<br>EE RMSE vs demo **{e:.5f}** m"
            + (f" &nbsp; _(recorded {float(rec[f, c]):.5f})_" if rec is not None else ""))
        set_row()

    def set_row(_=None):
        t = int(row.value)
        demo.set_row(t)
        recon.set_row(t)
        held = grasp_row <= t <= rel_row
        gap = float(np.linalg.norm(np.asarray(recon.ee[t]) - np.asarray(demo.ee[t])))
        info.content = (f"row **{t}**/{T - 1} &nbsp; phase **{phase_of(t)}** &nbsp; "
                        f"object **{'held' if held else 'at rest'}**"
                        f"<br>EE gap at this row **{gap:.4f}** m")

    row.on_update(set_row)
    for h in (ctx_sel, step_sel, cand_sel):
        h.on_update(reload)
    reload()

    print(f"[rollout_viser] {npz_path}: {T} rows, {B} contexts, {nC} candidates, "
          f"{F} outer steps; showing step {step}, candidate {cand} "
          f"(winner={winner}), context {ctx}")
    print(f"Viser server: http://0.0.0.0:{server.get_port()}")
    while True:
        if play.value:
            row.value = (int(row.value) + 1) % T
        time.sleep(1.0 / float(speed.value))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="?",
                    default="scratch/viz/multistart_behavior_q.npz")
    ap.add_argument("--step", type=int, default=None,
                    help="outer step to reconstruct from (default: last)")
    ap.add_argument("--candidate", type=int, default=None,
                    help="candidate index (default: the training-loss winner)")
    ap.add_argument("--context", type=int, default=0,
                    help="0 = fit scene A, 1 = held-out scene B")
    ap.add_argument("--spread", type=float, default=0.75,
                    help="lateral offset between the two rollouts, metres")
    ap.add_argument("--obj-size", type=float, default=0.05)
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--obs-z", type=float, default=None)
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    main(a.npz, step=a.step, cand=a.candidate, ctx=a.context, obs_z=a.obs_z,
         spread=a.spread, obj_size=a.obj_size, fps=a.fps, port=a.port)
