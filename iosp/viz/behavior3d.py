"""`viz_behavior` in three dimensions: the EE path converging onto the demo,
drawn in the space it actually lives in.

Why a separate renderer rather than a flag on `viz_behavior`
------------------------------------------------------------
The 2D renderer projects a 3D end-effector path onto x-y and says so on the
axis label.  That projection is not cosmetic on this task: the obstacle sits at
z = 0.4 and `transport` lifts over it, so a large part of the clearance
behaviour -- the part `transport.clearance` is supposed to explain -- happens
along the ONE axis the projection discards.  A path that appears to cut through
the obstacle disc in x-y may be clearing it comfortably in z, and the reverse.
This draws all three axes so that reading is unambiguous.

It shares `viz_behavior`'s npz contract exactly, so a recording made for one
renders in the other:

  path_hist   (n_frames, B, T, 3)   predicted EE paths per outer step
  demo        (B, T, 3)             the demonstrations being fitted
  rmse_hist   (n_frames,)           held-out behavioural RMSE per step
Optional
  cand_hist   (n_frames, C, B, T, 3)  ALL multistart candidates, thin and grey
              behind `path_hist` (which is then the SELECTED winner)
  loss_hist, obstacles (B, n_obs, 3 = x, y, radius -- z taken from `obs_z`),
  waypoints (B, n_way, 3), label

`obstacles` carries (x, y, radius) because that is what the 2D renderer needs;
the sphere's z centre is not in the file, so it is passed in (`obs_z`,
defaulting to `recovery_bench.OBS_CENTER[2]`) rather than guessed silently.

The camera rotates slowly through the animation.  That is not decoration: a
static 3D projection is genuinely ambiguous about depth, and a moving one is
not.  `--elev/--azim0/--spin` control it; `--spin 0` gives a fixed camera.
"""

import os

import numpy as np


def _sphere(ax, cx, cy, cz, r, n=18):
    """Wireframe sphere -- transparent, so the path stays readable through it.

    A solid surface would occlude exactly the thing being judged (does the path
    pass through or around), and matplotlib's painter's algorithm gets the
    depth order wrong for intersecting 3D artists anyway, so a solid sphere
    would additionally be drawn WRONG rather than merely opaque.
    """
    u = np.linspace(0, 2 * np.pi, 2 * n)
    v = np.linspace(0, np.pi, n)
    x = cx + r * np.outer(np.cos(u), np.sin(v))
    y = cy + r * np.outer(np.sin(u), np.sin(v))
    z = cz + r * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color="#999999", alpha=0.30, lw=0.5, zorder=1)


def render(npz_path, out_path, fps=6, dpi=110, max_ctx=2, obs_z=None,
           elev=22.0, azim0=-60.0, spin=50.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers '3d')

    d = np.load(npz_path, allow_pickle=True)
    P = d["path_hist"]
    demo = d["demo"]
    rmse = d["rmse_hist"]
    loss = d["loss_hist"] if "loss_hist" in d else None
    cand = d["cand_hist"] if "cand_hist" in d else None
    obs = d["obstacles"] if "obstacles" in d else None
    way = d["waypoints"] if "waypoints" in d else None
    label = str(d["label"]) if "label" in d else ""
    F, B, T, dim = P.shape
    if dim != 3:
        raise ValueError(f"{npz_path} holds {dim}-D paths; this renderer is 3D-only "
                         "(use iosp.viz.behavior for 2D)")
    B = min(B, max_ctx)
    if obs_z is None:
        from iosp.config import OBS_CENTER
        obs_z = float(OBS_CENTER[2])

    plt.rcParams.update({"figure.facecolor": "white", "font.size": 9})
    fig = plt.figure(figsize=(4.6 * B + 4.4, 4.8))
    axes3 = [fig.add_subplot(1, B + 1, b + 1, projection="3d") for b in range(B)]
    ax_l = fig.add_subplot(1, B + 1, B + 1)

    colors = ["#3b7dd8", "#d9534f", "#2e8b57"]
    lines = []
    cand_lines = {}
    for b, ax in enumerate(axes3):
        if obs is not None:
            for o in obs[b]:
                _sphere(ax, float(o[0]), float(o[1]), obs_z, float(o[2]))
        if cand is not None:
            for c in range(cand.shape[1]):
                cl, = ax.plot(cand[0, c, b, :, 0], cand[0, c, b, :, 1],
                              cand[0, c, b, :, 2], "-", color="#b0b0b0", lw=0.9,
                              alpha=0.75, zorder=2,
                              label="other candidates" if (c == 0 and b == 0) else None)
                cand_lines.setdefault(b, []).append(cl)
        ax.plot(demo[b, :, 0], demo[b, :, 1], demo[b, :, 2], "--", color="#444",
                lw=1.8, zorder=3, label="demonstration")
        ln, = ax.plot(P[0, b, :, 0], P[0, b, :, 1], P[0, b, :, 2], "-o",
                      color=colors[b % 3], ms=2.2, lw=1.6, zorder=4,
                      label="selected fit" if cand is not None else "current fit")
        lines.append(ln)
        if way is not None:
            ax.plot(way[b, :, 0], way[b, :, 1], way[b, :, 2], "*", color="#e8a33d",
                    ms=14, lw=0, zorder=5, markeredgecolor="#7a5312",
                    label="skeleton waypoint")
        ax.plot([demo[b, 0, 0]], [demo[b, 0, 1]], [demo[b, 0, 2]], "s",
                color="#222", ms=6, lw=0, zorder=5)

        # One cubic box around EVERY frame of BOTH the fit and the demo, so the
        # path is never rescaled mid-animation -- an axis that grows as the fit
        # moves makes convergence look like motion that isn't there.
        pts = np.concatenate([P[:, b].reshape(-1, 3), demo[b]], axis=0)
        lo, hi = pts.min(0), pts.max(0)
        if obs is not None:
            for o in obs[b]:
                c = np.array([o[0], o[1], obs_z], dtype=float)
                lo = np.minimum(lo, c - float(o[2]))
                hi = np.maximum(hi, c + float(o[2]))
            mid, half = (lo + hi) / 2, (hi - lo).max() / 2 * 1.10
        else:
            mid, half = (lo + hi) / 2, (hi - lo).max() / 2 * 1.10
        ax.set_xlim(mid[0] - half, mid[0] + half)
        ax.set_ylim(mid[1] - half, mid[1] + half)
        ax.set_zlim(mid[2] - half, mid[2] + half)
        try:
            ax.set_box_aspect((1, 1, 1))  # equal aspect: matplotlib 3.3+
        except Exception:
            pass
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
        ax.set_title(f"context {b}" + ("  (fit)" if b == 0 else "  (held out)"))
        ax.view_init(elev=elev, azim=azim0)
        if b == 0:
            ax.legend(loc="upper left", fontsize=7)

    ax_l.grid(True, alpha=0.25)
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
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    def update(t):
        for b, ln in enumerate(lines):
            # Line3D takes x/y in set_data and z separately -- set_data with
            # three arrays silently keeps the OLD z.
            ln.set_data(P[t, b, :, 0], P[t, b, :, 1])
            ln.set_3d_properties(P[t, b, :, 2])
        for b, lst in cand_lines.items():
            for c, cl in enumerate(lst):
                cl.set_data(cand[t, c, b, :, 0], cand[t, c, b, :, 1])
                cl.set_3d_properties(cand[t, c, b, :, 2])
        for ax in axes3:
            ax.view_init(elev=elev, azim=azim0 + spin * t / max(F - 1, 1))
        marker.set_data([t], [rmse[t]])
        sup.set_text(f"{label}   step {t}/{F-1}   held-out RMSE={rmse[t]:.5f}"
                     + ("   <- best" if t == best else ""))
        return (*lines, *[cl for lst in cand_lines.values() for cl in lst],
                marker, sup)

    anim = FuncAnimation(fig, update, frames=F, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"[render3d] wrote {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB, "
          f"{F} frames @ {fps}fps)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--max-ctx", type=int, default=2)
    ap.add_argument("--obs-z", type=float, default=None)
    ap.add_argument("--elev", type=float, default=22.0)
    ap.add_argument("--azim0", type=float, default=-60.0)
    ap.add_argument("--spin", type=float, default=50.0,
                    help="total azimuth sweep in degrees; 0 = fixed camera")
    a = ap.parse_args()
    render(a.npz, a.out or a.npz.replace(".npz", "_3d.gif"), fps=a.fps, dpi=a.dpi,
           max_ctx=a.max_ctx, obs_z=a.obs_z, elev=a.elev, azim0=a.azim0, spin=a.spin)
