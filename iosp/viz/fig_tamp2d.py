"""Fig 1: 2D TAMP sanity check — 2-panel PDF for IEEE T-RO.

Panel (a): 2D environment with obstacles, waypoints, and recovered paths
overlaid at several outer steps (increasingly bold), demo as dashed black.
Panel (b): Held-out RMSE convergence (log y-axis).

    python -m iosp.viz.fig_tamp2d path/to/tamp2d_fit.npz
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

COL2 = 7.16
FIT_COLOR = "#0072B2"
DEMO_COLOR = "#222222"
OBS_COLOR = "#d0d0d0"
WAY_COLOR = "#f0b429"


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
        "grid.linewidth": 0.4,
        "grid.alpha": 0.3,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def render(input_path, out_path):
    _set_style()

    d = np.load(input_path, allow_pickle=True)
    path_hist = d["path_hist"]   # (F, B, T, 2) or (F, B, T, 3)
    demo = d["demo"]             # (B, T, 2) or (B, T, 3)
    rmse = d["rmse_hist"]        # (F,)
    obs = d["obstacles"] if "obstacles" in d else None    # (B, n_obs, 3)
    way = d["waypoints"] if "waypoints" in d else None    # (B, n_way, 2)
    F, B_all, T, dim = path_hist.shape

    show_steps = [0, min(10, F - 1), min(20, F - 1), F - 1]
    show_steps = sorted(set(show_steps))

    n_ctx = min(B_all, 3)

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.6),
                             gridspec_kw=dict(width_ratios=[1.3, 1.0],
                                              wspace=0.30))
    ax_env, ax_rmse = axes

    # --- panel (a): environment + path overlay ---
    for ctx in range(n_ctx):
        if obs is not None and ctx < obs.shape[0]:
            for o in obs[ctx]:
                circ = Circle((float(o[0]), float(o[1])), float(o[2]),
                              fc=OBS_COLOR, ec="#aaaaaa", lw=0.5, zorder=1,
                              alpha=0.6)
                ax_env.add_patch(circ)

        if way is not None and ctx < way.shape[0]:
            ax_env.plot(way[ctx, :, 0], way[ctx, :, 1], "*",
                        color=WAY_COLOR, ms=9, mec="#7a5312", mew=0.5,
                        zorder=8, label="waypoints" if ctx == 0 else None)

        ax_env.plot(demo[ctx, :, 0], demo[ctx, :, 1], "--",
                    color=DEMO_COLOR, lw=1.8, zorder=5,
                    label="demo" if ctx == 0 else None)

        for i, step in enumerate(show_steps):
            frac = (i + 1) / len(show_steps)
            lw = 0.5 + 1.5 * frac
            alpha = 0.25 + 0.75 * frac
            ax_env.plot(path_hist[step, ctx, :, 0], path_hist[step, ctx, :, 1],
                        "-", color=FIT_COLOR, lw=lw, alpha=alpha, zorder=4,
                        label=f"step {step}" if ctx == 0 else None)

    ax_env.set_xlabel("x (m)")
    ax_env.set_ylabel("y (m)")
    ax_env.set_aspect("equal", adjustable="datalim")
    ax_env.legend(loc="best", fontsize=6, ncol=2, handlelength=1.3)
    for sp in ("top", "right"):
        ax_env.spines[sp].set_visible(False)
    ax_env.text(-0.02, 1.04, "(a)", transform=ax_env.transAxes, fontsize=9,
                fontweight="bold", va="bottom", ha="left")

    # --- panel (b): RMSE convergence ---
    ax_rmse.plot(rmse, "-o", color=FIT_COLOR, lw=1.4, ms=2.5, mew=0,
                 zorder=3, label="held-out RMSE")
    ax_rmse.set_yscale("log")
    ax_rmse.set_xlabel("outer step")
    ax_rmse.set_ylabel("held-out RMSE (m)")
    ax_rmse.grid(True, which="both", alpha=0.2, lw=0.4)
    for sp in ("top", "right"):
        ax_rmse.spines[sp].set_visible(False)
    ax_rmse.text(-0.02, 1.04, "(b)", transform=ax_rmse.transAxes, fontsize=9,
                 fontweight="bold", va="bottom", ha="left")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path)
    png_path = out_path.rsplit(".", 1)[0] + ".png" if out_path.endswith(".pdf") else None
    if png_path:
        fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"[fig_tamp2d] wrote {out_path}" +
          (f" + {png_path}" if png_path else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="?", default="iosp/data/results/tamp2d/tamp2d_fit.npz")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join("iosp", "figures", "fig1_tamp2d.pdf")
    render(a.npz, out)
