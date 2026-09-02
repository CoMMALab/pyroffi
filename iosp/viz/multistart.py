"""Talk/paper figure for the multistart fit: the candidate population converging
in two scenes, and what that costs on the scene the loss never saw.

Three panels, deliberately: the fit scene, the held-out scene, and every
candidate's held-out error.  Candidates are coloured by IK BRANCH rather than by
index, because branch is what separates them (measured 0.02 / 0.5 / 2.9 held-out
across the three, with the cost starts clustered tightly inside each), so a
viewer sees "one branch works, two do not" without being told.

The figure is built to be TALKED OVER: short labels, no annotations explaining
what a presenter can say out loud, and nothing in it that needs a caption to
parse.

A fourth panel -- per-waypoint distance to the demonstration -- is available via
`detail=True`.  It is off by default but worth knowing about: the selected fit's
path moves ~0.085 m against a trajectory spanning ~1.1 m, so the geometry panels
genuinely cannot resolve its convergence (RMSE 0.0498 -> 0.0036), and that panel
is the only place it is visible without exaggerating the path, which would make
the figure lie to make a point.

Required npz keys: `cand_hist (F, C, M, T, 3)`, `demo (M, T, 3)`, `winner`, `S`,
`ee_held_hist (F, C)`, `ee_train_hist (F, C)`.  Optional `obstacles`,
`waypoints`, `space`, `scene_b_scale`.
"""

import os

import numpy as np

# One hue per IK branch.  Blue/orange/purple stay separable under deuteranopia
# and protanopia; red/green would not.
BRANCH_COLORS = ["#1f77b4", "#e8820c", "#8250c4", "#2ca02c", "#d62728"]
DEMO_COLOR = "#111111"


def _sphere(ax, cx, cy, cz, r, n=16):
    u = np.linspace(0, 2 * np.pi, 2 * n)
    v = np.linspace(0, np.pi, n)
    ax.plot_wireframe(cx + r * np.outer(np.cos(u), np.sin(v)),
                      cy + r * np.outer(np.sin(u), np.sin(v)),
                      cz + r * np.outer(np.ones_like(u), np.cos(v)),
                      color="#8a8a8a", alpha=0.20, lw=0.5, zorder=1)


def render(npz_path, out_path, fps=8, dpi=120, obs_z=None, elev=20.0,
           azim0=-62.0, spin=40.0, hold=8, detail=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    d = np.load(npz_path, allow_pickle=True)
    cand, demo = d["cand_hist"], d["demo"]
    win, S = int(d["winner"]), int(d["S"])
    held, train = d["ee_held_hist"], d["ee_train_hist"]
    obs = d["obstacles"] if "obstacles" in d else None
    way = d["waypoints"] if "waypoints" in d else None
    F, C, M, T, _ = cand.shape
    branch_of = [c // S for c in range(C)]
    win_color = BRANCH_COLORS[branch_of[win] % len(BRANCH_COLORS)]
    if obs_z is None:
        from iosp.config import OBS_CENTER
        obs_z = float(OBS_CENTER[2])

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.size": 12, "legend.frameon": False, "axes.edgecolor": "#cfcfd6",
        "xtick.color": "#666666", "ytick.color": "#666666",
        "axes.labelcolor": "#444444", "text.color": "#222222",
    })
    n_ax = 4 if detail else 3
    fig = plt.figure(figsize=(5.4 * n_ax + 0.6, 5.8))
    gs = fig.add_gridspec(1, n_ax,
                          width_ratios=[1.1, 1.1, 0.95] + ([0.85] if detail else []),
                          left=0.015, right=0.985, top=0.84, bottom=0.13,
                          wspace=0.22)
    ax3 = [fig.add_subplot(gs[0, i], projection="3d") for i in (0, 1)]
    ax_l = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[0, 3]) if detail else None

    cand_lines, win_lines = {}, []
    for m, ax in enumerate(ax3):
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor("#fcfcfe")
            axis.pane.set_edgecolor("#e6e6ec")
            axis._axinfo["grid"].update(color="#ebebf1", linewidth=0.6)
        if obs is not None:
            for o in obs[m]:
                _sphere(ax, float(o[0]), float(o[1]), obs_z, float(o[2]))
        for c in range(C):
            b = branch_of[c]
            # First NON-winner candidate of a branch carries its legend entry;
            # keying on `c % S == 0` drops branch 0 whenever the winner is that
            # branch's first candidate.
            first = next(k for k in range(C) if branch_of[k] == b and k != win)
            ln, = ax.plot(cand[0, c, m, :, 0], cand[0, c, m, :, 1],
                          cand[0, c, m, :, 2], "-",
                          color=BRANCH_COLORS[b % len(BRANCH_COLORS)], lw=1.1,
                          alpha=0.40, zorder=2,
                          label=f"branch {b}" if (c == first and m == 0) else None)
            cand_lines.setdefault(m, []).append(ln)
        ax.plot(demo[m, :, 0], demo[m, :, 1], demo[m, :, 2], "--",
                color=DEMO_COLOR, lw=2.4, zorder=4,
                label="demonstration" if m == 0 else None)
        # White casing under the winner so it reads against both the pale
        # candidates and the dark demo without needing a fourth hue.
        glow, = ax.plot(cand[0, win, m, :, 0], cand[0, win, m, :, 1],
                        cand[0, win, m, :, 2], "-", color="white", lw=5.2,
                        alpha=0.9, zorder=5, solid_capstyle="round")
        wl, = ax.plot(cand[0, win, m, :, 0], cand[0, win, m, :, 1],
                      cand[0, win, m, :, 2], "-o", color=win_color, lw=2.8,
                      ms=3.0, zorder=6, label="selected fit" if m == 0 else None)
        win_lines.append((glow, wl))
        if way is not None:
            ax.plot(way[m, :, 0], way[m, :, 1], way[m, :, 2], "*",
                    color="#f0b429", ms=17, lw=0, zorder=7,
                    markeredgecolor="#7a5312", markeredgewidth=0.7,
                    label="waypoints" if m == 0 else None)

        # Frame set by the winner + demo + obstacle.  A candidate that flies off
        # is meant to leave the frame; letting it set the scale would shrink the
        # motion being judged to nothing.
        pts = np.concatenate([cand[:, win, m].reshape(-1, 3), demo[m]], 0)
        lo, hi = pts.min(0), pts.max(0)
        if obs is not None:
            for o in obs[m]:
                ctr = np.array([o[0], o[1], obs_z], float)
                lo = np.minimum(lo, ctr - float(o[2]))
                hi = np.maximum(hi, ctr + float(o[2]))
        mid, half = (lo + hi) / 2, (hi - lo).max() / 2 * 1.12
        ax.set_xlim(mid[0] - half, mid[0] + half)
        ax.set_ylim(mid[1] - half, mid[1] + half)
        ax.set_zlim(mid[2] - half, mid[2] + half)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        for setter in (ax.set_xlabel, ax.set_ylabel, ax.set_zlabel):
            setter("")            # axes are metres on a robot; say it out loud
        ax.tick_params(pad=-2, labelsize=8, colors="#999999")
        ax.set_title("Fit scene" if m == 0 else "Held-out scene",
                     pad=-6, fontsize=15, color="#222222")
        ax.view_init(elev=elev, azim=azim0)
        if m == 0:
            ax.legend(loc="upper left", fontsize=10.5, borderpad=0.2,
                      handlelength=1.5, labelspacing=0.35,
                      bbox_to_anchor=(-0.02, 0.99))

    for c in range(C):
        ax_l.plot(held[:, c], color=BRANCH_COLORS[branch_of[c] % len(BRANCH_COLORS)],
                  lw=2.6 if c == win else 1.1,
                  alpha=1.0 if c == win else 0.35, zorder=4 if c == win else 2)
    ax_l.set_yscale("log")
    ax_l.set_xlabel("outer step", fontsize=11)
    ax_l.set_ylabel("held-out error [m]", fontsize=11)
    ax_l.set_title("Held-out error", pad=10, fontsize=15, color="#222222")
    ax_l.grid(True, which="both", alpha=0.16, lw=0.6)
    for sp in ("top", "right"):
        ax_l.spines[sp].set_visible(False)
    mark, = ax_l.plot([0], [held[0, win]], "o", color=win_color, ms=9,
                      mec="white", mew=1.5, zorder=6)

    det = []
    if detail:
        err = np.linalg.norm(cand[:, win] - demo[None], axis=-1)
        # Interior waypoints set the limits: the endpoints are clamped to
        # q_start/q_goal, so their error is 0 by construction and would pin a
        # log axis to its floor for a value that was never fitted.
        ax_d.set_yscale("log")
        ax_d.set_xlim(0, T - 1)
        ax_d.set_ylim(max(err[:, :, 1:-1].min() * 0.6, 1e-4), err.max() * 1.6)
        ax_d.set_xlabel("waypoint", fontsize=11)
        ax_d.set_ylabel("distance to demo [m]", fontsize=11)
        ax_d.set_title("Selected fit, per waypoint", pad=10, fontsize=15)
        ax_d.grid(True, which="both", alpha=0.16, lw=0.6)
        for sp in ("top", "right"):
            ax_d.spines[sp].set_visible(False)
        for m in range(M):
            ax_d.plot(err[0, m], "-" if m == 0 else "--", color="#cdcdd6",
                      lw=1.2, zorder=1)
        det = [ax_d.plot(err[0, m], "-" if m == 0 else "--", lw=2.4, zorder=3,
                         color=win_color if m == 0 else "#d1495b",
                         label="fit" if m == 0 else "held out")[0]
               for m in range(M)]
        ax_d.legend(loc="lower right", fontsize=10)

    fig.text(0.5, 0.975, "One demonstration, many candidate cost functions, "
             "one selection", ha="center", va="top", fontsize=17,
             color="#111111")
    step = fig.text(0.5, 0.028, "", ha="center", va="bottom", fontsize=12.5,
                    color="#777777")

    def update(t):
        for m in range(M):
            for c, ln in enumerate(cand_lines[m]):
                ln.set_data(cand[t, c, m, :, 0], cand[t, c, m, :, 1])
                ln.set_3d_properties(cand[t, c, m, :, 2])
            for ln in win_lines[m]:
                ln.set_data(cand[t, win, m, :, 0], cand[t, win, m, :, 1])
                ln.set_3d_properties(cand[t, win, m, :, 2])
            ax3[m].view_init(elev=elev, azim=azim0 + spin * t / max(F - 1, 1))
        mark.set_data([t], [held[t, win]])
        for m, ln in enumerate(det):
            ln.set_ydata(np.linalg.norm(cand[t, win, m] - demo[m], axis=-1))
        step.set_text(f"step {t} / {F-1}      fit {train[t, win]*100:.2f} cm"
                      f"      held out {held[t, win]*100:.2f} cm")
        return ()

    # Hold the converged state so the last frame is readable when it loops.
    anim = FuncAnimation(fig, update, frames=list(range(F)) + [F - 1] * hold,
                         blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"[viz_multistart] wrote {out_path} "
          f"({os.path.getsize(out_path)/1e6:.1f} MB, {F + hold} frames @ {fps}fps)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="?", default="iosp/data/viz/multistart_behavior.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--azim0", type=float, default=-62.0)
    ap.add_argument("--spin", type=float, default=40.0)
    ap.add_argument("--detail", action="store_true",
                    help="add the per-waypoint panel (see module docstring)")
    a = ap.parse_args()
    render(a.npz, a.out or a.npz.replace(".npz", "_talk.gif"), fps=a.fps,
           dpi=a.dpi, elev=a.elev, azim0=a.azim0, spin=a.spin, detail=a.detail)
