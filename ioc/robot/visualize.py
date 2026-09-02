"""Visualize 7-DoF IOC trajectories: EE paths + obstacles, and joint traces.

Solves the same demo-generation pipeline `e1_identifiability.run_trial` uses
(sample scenes -> calibrate -> solve at theta*), optionally also solves a
second theta (e.g. a recovered theta_hat) on the SAME scenes so demo vs.
recovered trajectories can be compared directly. Headless matplotlib (Agg),
writes one PNG.

Usage:
    XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=0 \
        python -m ioc.robot.visualize --n-contexts 4
    # compare against a theta you already recovered:
    python -m ioc.robot.visualize --theta-hat 0.42 0.35 0.23
    # or recover one here, then plot its reconstruction against ground truth:
    python -m ioc.robot.visualize --fit --n-contexts 4

`ioc` is not an installed package (`pyproject.toml` ships `pyroffi` only), so it
imports only with the repo root on `sys.path`, and running this from any other
working directory raised `ModuleNotFoundError: No module named 'ioc'`.  Two
things are needed to make the script location-independent, both below:

- the `sys.path` bootstrap, which fixes the `python .../ioc/robot/visualize.py`
  form from any cwd (it cannot fix `python -m ioc.robot.visualize`, since `-m`
  resolves the module before any file-level code runs -- run that form from the
  repo root, or `pip install -e .` with `ioc` added to the packages list);
- `_resolve`, since the default URDF/SRDF/mesh paths are relative to the repo
  root and would otherwise miss even once the import succeeds.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from ioc import outer as outer_opt
from ioc.inner import make_inner_solver
from ioc.robot import bases, problem as prob
from ioc.robot.e1_identifiability import make_dynamics_forward_solver
from pyroffi.optimization_engines import DynamicsTrajOptConfig


def _resolve(path):
    """Interpret a relative resource path against the repo root, not the cwd.

    An existing path is returned untouched, so an explicit absolute or
    cwd-relative override still wins.
    """
    p = Path(path)
    if p.exists() or p.is_absolute():
        return str(p)
    return str(_ROOT / p)


def _sphere_surface(center, radius, n=12):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def main(
    urdf_path: str = "resources/panda/panda_spherized.urdf",
    srdf_path: str = "resources/panda/panda.srdf",
    mesh_dir: str = "resources/panda/meshes",
    n_contexts: int = 4,
    n_timesteps: int = 24,
    n_newton: int = 40,
    seed: int = 0,
    theta_star: tuple[float, ...] = (0.5, 0.3, 0.2),
    theta_hat: tuple[float, ...] | None = None,
    fit: bool = False,
    n_outer_steps: int = 60,
    lr: float = 0.15,
    out: str = "ioc_trajectories.png",
):
    """`--theta-hat` plots a theta you supply; `--fit` recovers one here by
    running the outer fit on these demonstrations and plots THAT, so the panels
    show an actual reconstruction rather than a hand-picked weight vector."""
    if not jax.config.jax_enable_x64:
        print("WARNING: x64 is OFF; run with JAX_ENABLE_X64=1 for a real solve.")

    problem = prob.RobotProblem.load(_resolve(urdf_path), _resolve(srdf_path),
                                     _resolve(mesh_dir), n_timesteps)
    residual_fn, names = bases.kinematic(problem, "k3")
    theta_star = jnp.asarray(theta_star)

    rng = np.random.default_rng(seed)
    scenes = problem.sample_scenes(rng, n_contexts)
    scales = problem.calibrate(residual_fn, scenes, jax.random.key(seed))

    forward_solver = make_dynamics_forward_solver(DynamicsTrajOptConfig(n_iters=n_newton))
    inner = make_inner_solver(residual_fn, scales, forward_solver=forward_solver)

    x0s, x_star, demos = prob.make_demos(
        problem, inner.solve_implicit, scenes, theta_star, rng, demo_noise=0.0
    )
    ee_demo = jax.vmap(problem.ee_positions)(demos)  # (n, T, 3)

    if fit:
        if theta_hat is not None:
            raise SystemExit("pass either --theta-hat or --fit, not both")
        loss_and_grad = jax.jit(jax.value_and_grad(
            prob.make_outer(problem, inner.solve_implicit, scenes, demos, x0s)))
        z0 = jnp.asarray(rng.normal(scale=0.5, size=len(names)))
        z_hat, _ = outer_opt.adam(loss_and_grad, z0, lr=lr, n_steps=n_outer_steps)
        theta_hat = tuple(float(v) for v in jax.nn.softmax(z_hat))
        print(f"recovered theta_hat = {np.asarray(theta_hat)}  (names: {list(names)})")

    ee_hat = None
    if theta_hat is not None:
        theta_hat_arr = jnp.asarray(theta_hat)
        q_hat = jax.vmap(lambda x0, s: inner.solve_implicit(x0, theta_hat_arr, s))(x0s, scenes)
        q_hat = jax.vmap(problem.unpack)(q_hat, scenes)
        ee_hat = jax.vmap(problem.ee_positions)(q_hat)

    ee_demo = np.asarray(ee_demo)
    demos = np.asarray(demos)
    err = None
    if ee_hat is not None:
        ee_hat = np.asarray(ee_hat)
        q_hat_np = np.asarray(q_hat)
        # per-context, per-timestep EE displacement: the reconstruction error
        # the outer loss is actually built from (`prob.make_outer`).
        err = np.linalg.norm(ee_hat - ee_demo, axis=-1)      # (n, T)
        rmse = np.sqrt((err ** 2).mean(axis=1))              # (n,)
        print(f"EE reconstruction RMSE per context: "
              f"{np.array2string(rmse, precision=4)}  mean={rmse.mean():.4f} m")

    n = n_contexts
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols

    # -- figure 1: EE paths in 3D, one panel per context ------------------
    fig = plt.figure(figsize=(4.2 * ncols, 4.2 * nrows))
    for i in range(n):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        p = ee_demo[i]
        ax.plot(p[:, 0], p[:, 1], p[:, 2], "-o", ms=2, color="C0", lw=2,
                label="ground truth (theta*)")
        ax.scatter(*p[0], color="green", s=40, label="start")
        ax.scatter(*p[-1], color="red", s=40, label="goal")
        if ee_hat is not None:
            ph = ee_hat[i]
            ax.plot(ph[:, 0], ph[:, 1], ph[:, 2], "--o", ms=2, color="C3", lw=2,
                    label="reconstructed (theta_hat)")
            # residual whiskers: the two paths are time-aligned, so joining
            # corresponding samples shows WHERE the reconstruction drifts
            # rather than only how much.
            for a, b in zip(p, ph):
                ax.plot(*zip(a, b), color="gray", lw=0.5, alpha=0.6)
        center = np.asarray(scenes.obs_center[i])
        radius = float(scenes.obs_radius[i, 0])
        sx, sy, sz = _sphere_surface(center, radius)
        ax.plot_surface(sx, sy, sz, color="gray", alpha=0.3, linewidth=0)
        title = f"context {i}"
        if err is not None:
            title += f"\nRMSE {rmse[i]:.4f} m  max {err[i].max():.4f} m"
        ax.set_title(title, fontsize=9)
        if i == 0:
            ax.legend(fontsize=7, loc="upper left")
    if theta_hat is not None:
        fig.suptitle(f"EE paths: ground truth theta*={np.asarray(theta_star)} vs "
                     f"reconstruction theta_hat={np.round(np.asarray(theta_hat), 3)}")
    else:
        fig.suptitle("7-DoF EE paths at theta* with anchored obstacle")
    fig.tight_layout()
    ee_path = out.replace(".png", "_ee.png")
    fig.savefig(ee_path, dpi=150)
    plt.close(fig)

    # -- figure 2: per-joint angle traces, one panel per context ----------
    dof = problem.dof
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
    t = np.arange(n_timesteps)
    for i in range(n):
        ax = axes[i // ncols][i % ncols]
        for j in range(dof):
            ax.plot(t, demos[i, :, j], color=f"C{j % 10}", lw=1.2,
                     label=f"q{j}" if i == 0 else None)
            if theta_hat is not None:
                ax.plot(t, q_hat_np[i, :, j], color=f"C{j % 10}", lw=1.2, ls="--")
        ax.set_title(f"context {i}")
        ax.set_xlabel("timestep")
        ax.set_ylabel("q [rad]")
    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")
    if theta_hat is not None:
        fig.suptitle("Joint traces: solid = theta* demo, dashed = theta_hat")
    else:
        fig.suptitle("Joint traces (theta* demo)")
    axes[0][0].legend(fontsize=6, ncol=2, loc="upper right")
    fig.tight_layout()
    joints_path = out.replace(".png", "_joints.png")
    fig.savefig(joints_path, dpi=150)
    plt.close(fig)

    print(f"wrote {ee_path}")
    print(f"wrote {joints_path}")

    # -- figure 3: reconstruction error over time -------------------------
    # Separated from the 3D panels because magnitude is unreadable in a
    # projection: a path that looks coincident there can still carry the bulk
    # of the outer loss in a few timesteps near the obstacle.
    if err is not None:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for i in range(n):
            ax.plot(t, err[i], lw=1.4, label=f"context {i} (RMSE {rmse[i]:.4f})")
        ax.axhline(rmse.mean(), color="k", ls=":", lw=1,
                   label=f"mean RMSE {rmse.mean():.4f}")
        ax.set_xlabel("timestep")
        ax.set_ylabel("|EE_hat - EE_demo| [m]")
        ax.set_title("EE reconstruction error vs. ground truth")
        ax.legend(fontsize=7)
        fig.tight_layout()
        err_path = out.replace(".png", "_error.png")
        fig.savefig(err_path, dpi=150)
        plt.close(fig)
        print(f"wrote {err_path}")


if __name__ == "__main__":
    tyro.cli(main)
