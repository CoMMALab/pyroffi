"""Visualize the demonstration dataset: EE paths, joint traces, action stats.

Reads the zarr the generator wrote -- it does **not** re-solve anything, so what
you see is exactly what the policy is trained on.

Everything is reconstructed from ``data/state`` alone, using its documented
layout ``[q(7), dq(7), q_goal - q(7), obs_center - p_ee(q)(3), obs_radius(1)]``:

    q            = state[:, :7]
    q_goal       = state[:, 14:21] + q          (constant within an episode)
    obs_center   = state[:, 21:24] + p_ee(q)    (likewise)

The stored rows stop one waypoint short of the goal (the last waypoint has no
action following it), so the goal configuration is appended back on to draw the
full demonstrated trajectory.

Usage (from the repository root):

    PYTHONPATH=diffTORI python -m difftori.data.visualize --n-contexts 6

For interactive 3D playback see `difftori.data.viser_playback`.

Headless matplotlib; writes PNGs next to the dataset by default.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

import jax
import numpy as np

from difftori.data.dataset import ReplayBuffer

DOF = 7


def _resolve(path: str) -> str:
    """Interpret a relative resource path against the repo root, not the cwd."""
    p = Path(path)
    return str(p) if p.exists() or p.is_absolute() else str(_ROOT / p)


def _sphere_surface(center, radius, n=14):
    u, v = np.linspace(0, 2 * np.pi, n), np.linspace(0, np.pi, n)
    return (center[0] + radius * np.outer(np.cos(u), np.sin(v)),
            center[1] + radius * np.outer(np.sin(u), np.sin(v)),
            center[2] + radius * np.outer(np.ones_like(u), np.cos(v)))


def sphere_clearance(problem, q, obs_center, obs_radius):
    """Min distance from ANY robot collision sphere to the obstacle surface.

    This is the quantity the teacher's collision feature is built from
    (`ioc.robot.problem.clearance_residual`, minus its soft-min smoothing), and
    therefore the honest check on whether a demonstration is actually clear.
    The end-effector *frame origin* is not a collision sphere, so an EE-to-
    obstacle distance can go slightly negative on a perfectly valid trajectory
    and says nothing on its own.
    """
    coll = problem.robot_coll.at_config(problem.robot, q)
    d = (np.linalg.norm(np.asarray(coll.pose.translation()) - obs_center,
                        axis=-1)
         - np.asarray(coll.radius) - obs_radius)
    return d.reshape(d.shape[0], -1).min(axis=-1)


def unpack_episodes(buf: ReplayBuffer, problem, n_contexts: int | None = None):
    """Reconstruct per-episode ``(q, ee, q_goal, obs_center, obs_radius)``.

    Returns lists indexed by episode; ``q`` has the goal waypoint appended, so
    it is one row longer than the stored rows for that episode.
    """
    n_ep = buf.n_episodes if n_contexts is None else min(n_contexts, buf.n_episodes)
    out = []
    for i in range(n_ep):
        lo, hi = buf.episode_bounds(i)
        s = buf.state[lo:hi]
        q = s[:, :DOF]
        p_ee = np.asarray(problem.ee_positions(q))
        q_goal = s[0, 2 * DOF:3 * DOF] + q[0]
        obs_center = s[0, 3 * DOF:3 * DOF + 3] + p_ee[0]
        obs_radius = float(s[0, -1])
        q_full = np.concatenate([q, q_goal[None, :]], axis=0)
        ee_full = np.asarray(problem.ee_positions(q_full))
        out.append(dict(q=q_full, ee=ee_full, q_goal=q_goal,
                        obs_center=obs_center, obs_radius=obs_radius,
                        clearance=sphere_clearance(problem, q_full, obs_center,
                                                   obs_radius),
                        action=buf.action[lo:hi]))
    return out


def plot_ee_paths(eps, out_path: str):
    n = len(eps)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(4.6 * ncols, 4.4 * nrows))
    for i, ep in enumerate(eps):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        p = ep["ee"]
        ax.plot(p[:, 0], p[:, 1], p[:, 2], "-o", ms=3, lw=2, color="C0")
        ax.scatter(*p[0], color="green", s=45, label="start")
        ax.scatter(*p[-1], color="red", s=45, label="goal")
        sx, sy, sz = _sphere_surface(ep["obs_center"], ep["obs_radius"])
        ax.plot_surface(sx, sy, sz, color="gray", alpha=0.35, linewidth=0)
        ax.set_title(f"episode {i}\nmin sphere clearance "
                     f"{ep['clearance'].min():.3f} m", fontsize=9)
        ax.set_xlabel("x"), ax.set_ylabel("y"), ax.set_zlabel("z")
        if i == 0:
            ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Demonstrated end-effector paths (teacher: pyroffi "
                 "dynamics_trajopt, dynamic basis)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_joint_traces(eps, out_path: str):
    n = len(eps)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.2 * nrows),
                             squeeze=False)
    for i, ep in enumerate(eps):
        ax = axes[i // ncols][i % ncols]
        for j in range(DOF):
            ax.plot(ep["q"][:, j], color=f"C{j}", lw=1.3,
                    label=f"q{j}" if i == 0 else None)
        ax.set_title(f"episode {i}", fontsize=9)
        ax.set_xlabel("waypoint"), ax.set_ylabel("q [rad]")
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    if n:
        axes[0][0].legend(fontsize=6, ncol=2)
    fig.suptitle("Joint trajectories")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_dataset_summary(buf: ReplayBuffer, eps, out_path: str):
    """Dataset-level view: what the policy is actually asked to regress."""
    scale = buf.meta.get("action_scale", 1.0)
    # `data/action` is stored already divided by `action_scale`, so it lives in
    # [-1, 1]; multiply back to read it in radians.
    action_rad = buf.action * scale
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0][0]
    for j in range(DOF):
        ax.hist(action_rad[:, j], bins=60, histtype="step", label=f"q{j}")
    ax.axvline(scale, color="k", ls="--", lw=1)
    ax.axvline(-scale, color="k", ls="--", lw=1,
               label=f"±action_scale ({scale:.3f} rad)")
    ax.set_title("Per-joint action (joint delta) distribution")
    ax.set_xlabel("delta q [rad]"), ax.set_ylabel("count")
    ax.legend(fontsize=6, ncol=2)

    ax = axes[0][1]
    lens = [len(ep["action"]) for ep in eps]
    T = min(lens) if lens else 0
    mags = np.stack([np.linalg.norm(ep["action"][:T] * scale, axis=-1)
                     for ep in eps])
    ax.plot(mags.T, color="C0", alpha=0.25, lw=1)
    ax.plot(mags.mean(axis=0), color="C3", lw=2.5, label="mean")
    ax.set_title("Action magnitude along the trajectory")
    ax.set_xlabel("waypoint"), ax.set_ylabel("||delta q|| [rad]")
    ax.legend(fontsize=7)

    ax = axes[1][0]
    worst = np.array([ep["clearance"].min() for ep in eps])
    for ep in eps:
        ax.plot(ep["clearance"], color="C0", alpha=0.35, lw=1)
    ax.axhline(0.0, color="k", ls="--", lw=1, label="obstacle surface")
    ax.axhline(0.05, color="C3", ls=":", lw=1,
               label="CLEARANCE_MARGIN (cost turns on)")
    ax.set_title(f"Min robot-sphere clearance to the obstacle\n"
                 f"{(worst < 0).sum()}/{len(eps)} episodes penetrate")
    ax.set_xlabel("waypoint"), ax.set_ylabel("distance [m]")
    ax.legend(fontsize=7)

    ax = fig.add_subplot(2, 2, 4, projection="3d")
    axes[1][1].axis("off")
    for ep in eps:
        p = ep["ee"]
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color="C0", alpha=0.4, lw=1)
        ax.scatter(*ep["obs_center"], color="gray", s=8 + 400 * ep["obs_radius"],
                   alpha=0.25)
    ax.set_title("All EE paths and obstacles", fontsize=10)

    meta = buf.meta
    fig.suptitle(f"{meta.get('task', 'dataset')} — {buf.n_episodes} episodes, "
                 f"{len(buf.state)} rows, discard rate "
                 f"{meta.get('discard_rate', float('nan')):.1%}, max "
                 f"stationarity {meta.get('max_stationarity_kept', float('nan')):.1e}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main(
    data: str = "diffTORI/data/panda_reach_expert.zarr",
    n_contexts: int = 6,
    urdf_path: str = "resources/panda/panda_spherized.urdf",
    srdf_path: str = "resources/panda/panda.srdf",
    mesh_dir: str = "resources/panda/meshes",
    out_prefix: str = "diffTORI/figures/dataset",
    summary_episodes: int = 60,
):
    """`--n-contexts` controls the per-episode panels; `--summary-episodes` the
    number pooled into the dataset-level figure."""
    from ioc.robot import problem as prob

    buf = ReplayBuffer.load(_resolve(data))
    print(f"{buf.meta.get('task')}: {buf.n_episodes} episodes, "
          f"{len(buf.state)} rows")

    n_timesteps = int(buf.meta.get("n_timesteps", 16))
    problem = prob.RobotProblem.load(_resolve(urdf_path), _resolve(srdf_path),
                                     _resolve(mesh_dir), n_timesteps)

    eps = unpack_episodes(buf, problem, n_contexts)

    # Resolve the output prefix against the repo root too.  Resolving inputs
    # there but not outputs is worse than resolving neither: the script keeps
    # working from any cwd and quietly writes the figures somewhere else, so
    # you sit looking at a stale PNG wondering why an edit had no effect.
    out_prefix = str(_ROOT / out_prefix) if not Path(out_prefix).is_absolute() \
        else out_prefix
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    plot_ee_paths(eps, f"{out_prefix}_ee.png")
    plot_joint_traces(eps, f"{out_prefix}_joints.png")
    plot_dataset_summary(buf, unpack_episodes(buf, problem, summary_episodes),
                         f"{out_prefix}_summary.png")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
