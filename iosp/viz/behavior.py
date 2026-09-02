"""Behavioural convergence animation: the trajectory walking onto the demo.

The earlier animation (`iosp.viz.fit_animation`) plotted cost WEIGHTS against
their ground truth.  On a rank-deficient problem that panel is close to
meaningless -- the weights drift freely along null directions while the
behaviour is reproduced exactly, so the bars can miss their targets entirely
while the fit is doing its job (measured on the Panda: the two IK standoffs
land on top of ground truth, the seven trajopt weights scramble).

This renders the thing that is actually being claimed instead:

  left    the trajectory in GEOMETRIC space, one line per context, converging
          onto the demonstration.  Obstacles are drawn as discs and the task
          skeleton's waypoints as stars, so "did it keep the skeleton" and
          "did it clear the obstacle" are both directly visible.
  right   the BEHAVIOURAL loss -- held-out RMSE against the demonstrations --
          on a log axis, with a marker at the current step.

Problem-agnostic: it reads an .npz and knows nothing about pick-and-place or
the 2D TAMP benchmark.  `path_hist` may be 2D or 3D; a 3D end-effector path is
projected onto x-y (stated on the axis label, not silently).

Required npz keys
-----------------
  path_hist   (n_frames, B, T, dim)  predicted paths per outer step
  demo        (B, T, dim)            the demonstrations being fitted
  rmse_hist   (n_frames,)            held-out behavioural RMSE per step
Optional
  cand_hist   (n_frames, C, B, T, dim)  ALL multistart candidates, drawn as thin
              grey lines behind `path_hist` (which is then the selected winner)
  loss_hist   (n_frames,)            training loss, drawn as a second line
  obstacles   (B, n_obs, 3)          x, y, radius
  waypoints   (B, n_way, dim)        task-skeleton anchors
  label       str
"""

import os

import numpy as np


def render(npz_path, out_path, fps=6, dpi=110, max_ctx=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    d = np.load(npz_path, allow_pickle=True)
    P = d["path_hist"]          # (F, B, T, dim)
    demo = d["demo"]            # (B, T, dim)
    rmse = d["rmse_hist"]
    loss = d["loss_hist"] if "loss_hist" in d else None
    cand = d["cand_hist"] if "cand_hist" in d else None
    obs = d["obstacles"] if "obstacles" in d else None
    way = d["waypoints"] if "waypoints" in d else None
    label = str(d["label"]) if "label" in d else ""
    F, B, T, dim = P.shape
    B = min(B, max_ctx)
    proj = dim > 2  # 3D EE path -> x-y, said out loud on the axis label

    plt.rcParams.update({"figure.facecolor": "white", "axes.grid": True,
                         "grid.alpha": 0.25, "font.size": 9})
    fig, axes = plt.subplots(1, B + 1, figsize=(4.2 * (B + 1), 4.4))
    axes = np.atleast_1d(axes)
    ax_l = axes[-1]

    colors = ["#3b7dd8", "#d9534f", "#2e8b57"]
    lines = []
    cand_lines = {}
    for b in range(B):
        ax = axes[b]
        if obs is not None:
            for o in obs[b]:
                ax.add_patch(plt.Circle((o[0], o[1]), o[2], color="#999999",
                                        alpha=0.45, zorder=1))
        # Losing candidates FIRST and thin, so the winner and the demo stay
        # readable on top of them; one labelled entry, not C legend rows.
        if cand is not None:
            for c in range(cand.shape[1]):
                cl, = ax.plot(cand[0, c, b, :, 0], cand[0, c, b, :, 1], "-",
                              color="#b0b0b0", lw=0.9, alpha=0.75, zorder=2,
                              label="other candidates" if (c == 0 and b == 0) else None)
                cand_lines.setdefault(b, []).append(cl)
        ax.plot(demo[b, :, 0], demo[b, :, 1], "--", color="#444", lw=1.8,
                zorder=3, label="demonstration")
        ln, = ax.plot(P[0, b, :, 0], P[0, b, :, 1], "-o", color=colors[b % 3],
                      ms=2.5, lw=1.6, zorder=4,
                      label="selected fit" if cand is not None else "current fit")
        lines.append(ln)
        if way is not None:
            ax.plot(way[b, :, 0], way[b, :, 1], "*", color="#e8a33d", ms=15,
                    zorder=5, markeredgecolor="#7a5312", label="skeleton waypoint")
        ax.plot([demo[b, 0, 0]], [demo[b, 0, 1]], "s", color="#222", ms=6, zorder=5)
        if cand is not None:
            # Clip the view to the DEMO+winner extent (padded), not the whole
            # candidate population: one diverged candidate would otherwise zoom
            # the interesting motion down to a few pixels.  Losers that leave
            # the box are meant to leave it -- that is the finding.
            pts = np.concatenate([P[:, b].reshape(-1, 2 if dim == 2 else dim)[:, :2],
                                  demo[b][:, :2]], axis=0)
            lo, hi = pts.min(0), pts.max(0)
            pad = 0.25 * max(hi - lo).item()
            ax.set_xlim(lo[0] - pad, hi[0] + pad)
            ax.set_ylim(lo[1] - pad, hi[1] + pad)
            ax.set_aspect("equal", adjustable="box")
        else:
            ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x" + ("  (3D path projected to x-y)" if proj else ""))
        ax.set_ylabel("y")
        ax.set_title(f"context {b}")
        if b == 0:
            ax.legend(loc="best", fontsize=7)

    ax_l.plot(rmse, color="#3b7dd8", lw=1.6, zorder=3, label="held-out RMSE")
    if loss is not None:
        ax_l.plot(np.sqrt(np.maximum(loss, 0)), color="#999", lw=1.2, ls=":",
                  zorder=2, label="train RMSE")
    best = int(np.argmin(rmse))
    ax_l.axvline(best, color="#2e8b57", ls="--", lw=1.2, zorder=1,
                 label=f"best (step {best}, {rmse[best]:.4f})")
    ax_l.set_yscale("log")
    ax_l.set_xlabel("outer step")
    ax_l.set_ylabel("behavioural RMSE")
    ax_l.set_title("behavioural loss (the criterion)")
    ax_l.legend(loc="best", fontsize=7)
    marker, = ax_l.plot([0], [rmse[0]], "o", color="#d9534f", ms=7, zorder=6)

    sup = fig.suptitle("")
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    def update(t):
        for b, ln in enumerate(lines):
            ln.set_data(P[t, b, :, 0], P[t, b, :, 1])
        for b, lst in cand_lines.items():
            for c, cl in enumerate(lst):
                cl.set_data(cand[t, c, b, :, 0], cand[t, c, b, :, 1])
        marker.set_data([t], [rmse[t]])
        sup.set_text(f"{label}   step {t}/{F-1}   held-out RMSE={rmse[t]:.5f}"
                     + ("   <- best" if t == best else ""))
        return (*lines, *[cl for lst in cand_lines.values() for cl in lst],
                marker, sup)

    anim = FuncAnimation(fig, update, frames=F, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"[render] wrote {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB, "
          f"{F} frames @ {fps}fps)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--max-ctx", type=int, default=3)
    a = ap.parse_args()
    render(a.npz, a.out or a.npz.replace(".npz", ".gif"), fps=a.fps, max_ctx=a.max_ctx)
