"""Fig 2: multistart convergence — 3-panel PDF for IEEE T-RO.

Panel (a): per-candidate held-out RMSE convergence (log scale), colored by
IK branch, winner highlighted.
Panel (b): winner's 3D EE path vs demonstration on the held-out scene.
Panel (c): per-branch best held-out RMSE bar chart.

    python -m iosp.viz.fig_multistart path/to/multistart.npz
    python -m iosp.viz.fig_multistart path/to/multistart.npz --out fig2.pdf
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COL2 = 7.16
BRANCH_COLORS = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7"]
BRANCH_MARKERS = ["o", "s", "^", "D", "v"]
BRANCH_LS = ["-", "--", "-.", (0, (3, 1, 1, 1)), ":"]
DEMO_COLOR = "#222222"


def _set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.5,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.3,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _sphere_wireframe(ax, cx, cy, cz, r, n=12):
    u = np.linspace(0, 2 * np.pi, 2 * n)
    v = np.linspace(0, np.pi, n)
    ax.plot_wireframe(
        cx + r * np.outer(np.cos(u), np.sin(v)),
        cy + r * np.outer(np.sin(u), np.sin(v)),
        cz + r * np.outer(np.ones_like(u), np.cos(v)),
        color="#b0b0b0", alpha=0.25, lw=0.4, zorder=1)


def render(input_path, out_path):
    _set_style()

    d = np.load(input_path, allow_pickle=True)
    cand = d["cand_hist"]           # (F, C, M, T, 3)
    demo = d["demo"]                # (M, T, 3)
    win = int(d["winner"])
    S = int(d["S"])
    B = int(d["B"])
    ee_held = d["ee_held_hist"]     # (F, C)
    F, C = ee_held.shape

    obs = d["obstacles"] if "obstacles" in d else None
    way = d["waypoints"] if "waypoints" in d else None

    branch_of = [c // S for c in range(C)]

    fig = plt.figure(figsize=(COL2, 2.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.1, 0.65],
                          left=0.06, right=0.97, top=0.88, bottom=0.18,
                          wspace=0.32)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], projection="3d")
    ax_c = fig.add_subplot(gs[0, 2])

    # --- panel (a): convergence curves ---
    seen_branches = set()
    for c in range(C):
        b = branch_of[c]
        is_winner = (c == win)
        lbl = f"branch {b}" if b not in seen_branches else None
        seen_branches.add(b)
        ax_a.plot(
            ee_held[:, c],
            color=BRANCH_COLORS[b % len(BRANCH_COLORS)],
            ls=BRANCH_LS[b % len(BRANCH_LS)],
            lw=2.0 if is_winner else 0.7,
            alpha=1.0 if is_winner else 0.35,
            zorder=5 if is_winner else 2,
            label=lbl)
    ax_a.plot([], [], color=BRANCH_COLORS[branch_of[win] % len(BRANCH_COLORS)],
              lw=2.0, label="selected")
    ax_a.set_yscale("log")
    ax_a.set_xlabel("outer step")
    ax_a.set_ylabel("held-out EE RMSE (m)")
    ax_a.legend(loc="upper right", fontsize=6, ncol=1, handlelength=1.5)
    ax_a.grid(True, which="both", alpha=0.2, lw=0.4)
    for sp in ("top", "right"):
        ax_a.spines[sp].set_visible(False)
    ax_a.text(-0.02, 1.04, "(a)", transform=ax_a.transAxes, fontsize=9,
              fontweight="bold", va="bottom", ha="left")

    # --- panel (b): 3D EE path (held-out scene, index 1) ---
    m_held = 1 if demo.shape[0] > 1 else 0
    for axis in (ax_b.xaxis, ax_b.yaxis, ax_b.zaxis):
        axis.pane.set_facecolor("#fafafa")
        axis.pane.set_edgecolor("#e0e0e0")
        axis._axinfo["grid"].update(color="#ebebeb", linewidth=0.4)

    if obs is not None and m_held < obs.shape[0]:
        from iosp.config import OBS_CENTER
        obs_z = float(OBS_CENTER[2])
        for o in obs[m_held]:
            _sphere_wireframe(ax_b, float(o[0]), float(o[1]), obs_z, float(o[2]))

    ax_b.plot(demo[m_held, :, 0], demo[m_held, :, 1], demo[m_held, :, 2],
              "--", color=DEMO_COLOR, lw=1.8, zorder=4, label="demo")
    win_color = BRANCH_COLORS[branch_of[win] % len(BRANCH_COLORS)]
    # White casing for legibility
    ax_b.plot(cand[-1, win, m_held, :, 0], cand[-1, win, m_held, :, 1],
              cand[-1, win, m_held, :, 2], "-", color="white", lw=3.5,
              alpha=0.9, zorder=5, solid_capstyle="round")
    ax_b.plot(cand[-1, win, m_held, :, 0], cand[-1, win, m_held, :, 1],
              cand[-1, win, m_held, :, 2], "-o", color=win_color, lw=1.6,
              ms=2.2, zorder=6, label="fit")

    if way is not None and m_held < way.shape[0]:
        ax_b.plot(way[m_held, :, 0], way[m_held, :, 1], way[m_held, :, 2],
                  "*", color="#f0b429", ms=10, lw=0, zorder=7,
                  markeredgecolor="#7a5312", markeredgewidth=0.5)

    ax_b.tick_params(pad=-3, labelsize=6, colors="#999999")
    for s in (ax_b.set_xlabel, ax_b.set_ylabel, ax_b.set_zlabel):
        s("")
    try:
        ax_b.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax_b.legend(loc="upper left", fontsize=6, handlelength=1.3)
    ax_b.view_init(elev=22, azim=-55)
    ax_b.text2D(-0.04, 1.02, "(b)", transform=ax_b.transAxes, fontsize=9,
                fontweight="bold", va="bottom", ha="left")

    # --- panel (c): per-branch best held-out bar ---
    branch_best = []
    for b in range(B):
        cands_b = [c for c in range(C) if branch_of[c] == b]
        branch_best.append(min(ee_held[-1, c] for c in cands_b))
    y_pos = np.arange(B)
    bars = ax_c.barh(y_pos, branch_best,
                     color=[BRANCH_COLORS[b % len(BRANCH_COLORS)] for b in range(B)],
                     height=0.6, edgecolor="white", linewidth=0.4)
    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels([f"br {b}" for b in range(B)])
    ax_c.set_xlabel("best held-out\nRMSE (m)")
    ax_c.invert_yaxis()
    for sp in ("top", "right"):
        ax_c.spines[sp].set_visible(False)
    ax_c.grid(True, axis="x", alpha=0.2, lw=0.4)
    ax_c.text(-0.02, 1.04, "(c)", transform=ax_c.transAxes, fontsize=9,
              fontweight="bold", va="bottom", ha="left")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path)
    png_path = out_path.rsplit(".", 1)[0] + ".png" if out_path.endswith(".pdf") else None
    if png_path:
        fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"[fig_multistart] wrote {out_path}" +
          (f" + {png_path}" if png_path else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="?", default="iosp/data/results/multistart/joint_s0.npz")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join("iosp", "figures", "fig2_multistart.pdf")
    render(a.npz, out)
