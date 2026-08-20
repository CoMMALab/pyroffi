"""Render the 2D IOC benchmarks: scaling curves, recovered reward fields, paths.

Reads the JSON produced by `ioc.bench2d.run` for the scaling curves, and re-runs
one short fit per benchmark to draw trajectories (cheap in 2D).  This is the
screen-quality pass; `ioc.plots` renders the paper figures from `ioc/data`.

    python -m ioc.analysis.bench2d_quicklook
"""

import glob
import json

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tyro

from ioc import outer as outer_opt
from ioc.bench2d import problems as b
from ioc.bench2d.run import build_solver

COLORS = {"implicit": "C0", "unrolled": "C1", "fd": "C2", "cmaes": "C3",
          "cioc": "C4", "kkt": "C5", "random": "0.6"}


def solves_to(trace, target):
    return next((s for s, l in trace if l < target), None)


# ---------------------------------------------------------------------------
# (a) scaling curve
# ---------------------------------------------------------------------------


def plot_scaling(out):
    files = sorted(glob.glob("bench2d_seg_K*.json"),
                   key=lambda f: int(f.split("K")[-1].split(".")[0]))
    if not files:
        print("no bench2d_seg_K*.json; skipping scaling plot")
        return
    Ks, data = [], {m: [] for m in ["implicit", "fd", "cmaes"]}
    for f in files:
        d = json.load(open(f))
        r = d["results"]
        Ks.append(d["K"])
        for m in data:
            vals = [solves_to(r[s][m]["trace"], 1e-2) for s in r]
            ok = [v for v in vals if v]
            data[m].append(np.median(ok) if len(ok) >= 2 else np.nan)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for m in data:
        ax[0].plot(Ks, data[m], "o-", color=COLORS[m], label=m)
    ax[0].set_xlabel("number of cost parameters $K$")
    ax[0].set_ylabel("forward solves to reach $L<10^{-2}$")
    ax[0].set_yscale("log")
    ax[0].set_title("Cost of a fit vs cost dimension")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    base = np.array(data["implicit"], dtype=float)
    for m in ["fd", "cmaes"]:
        ax[1].plot(Ks, np.array(data[m], dtype=float) / base, "o-",
                   color=COLORS[m], label=f"{m} / implicit")
    ax[1].axhline(1.0, color="k", lw=0.8, ls="--")
    ax[1].set_xlabel("number of cost parameters $K$")
    ax[1].set_ylabel("solve-count ratio vs implicit")
    ax[1].set_title("Advantage widens with $K$")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# (b) trajectories and recovered reward field
# ---------------------------------------------------------------------------


def fit_and_draw(benchmark, ax_traj, ax_field=None, k_bumps=6, M=6, T=30,
                 n_iter=80, budget=4000, bump_width=0.9, seed=0, lr=0.1):
    res_fn, names, d = b.BENCHMARKS[benchmark]
    cfg = b.default_cfg(benchmark, bump_width=bump_width)
    names = b.benchmark_names(benchmark, k_bumps, cfg)
    K = len(names)
    rng = np.random.default_rng(seed)
    theta_star = np.maximum(rng.dirichlet(np.full(K, 0.7)), 0.01)
    theta_star = jnp.asarray(theta_star / theta_star.sum())
    ctxs = b.sample_contexts(rng, M, benchmark, T, d, k_bumps, cfg)
    scales = b.calibrate(res_fn, ctxs, T, d, cfg, jax.random.key(seed), K)
    si = build_solver(res_fn, scales, T, d, cfg, n_iter, 1e-2, 3, 1e-9).solve_implicit
    x0 = jax.vmap(lambda c: b.seed_path(c, T, d, cfg))(ctxs)
    xstar = jax.vmap(lambda x, c: si(x, theta_star, c))(x0, ctxs)
    demos = jax.vmap(lambda x, c: b.unpack(x, c, T, d))(xstar, ctxs)

    def loss(z):
        t = jax.nn.softmax(z)

        def one(c, dm, xx):
            s = si(xx, t, c)
            p = b.unpack(s, c, T, d)[:, :2]
            return jnp.mean(jnp.sum((p - dm[:, :2]) ** 2, axis=-1))

        return jnp.mean(jax.vmap(one)(ctxs, demos, x0))

    gf = jax.jit(jax.value_and_grad(loss))
    z, _ = outer_opt.adam(gf, jnp.zeros(K), lr=lr, budget_solves=budget,
                          solves_per_step=M, trace_best=True)
    th = jax.nn.softmax(z)
    xhat = jax.vmap(lambda x, c: si(x, th, c))(x0, ctxs)

    for i in range(M):
        c = jax.tree.map(lambda a: a[i], ctxs)
        dm = np.asarray(demos[i])[:, :2]
        ph = np.asarray(b.unpack(xhat[i], c, T, d))[:, :2]
        ax_traj.plot(dm[:, 0], dm[:, 1], "-", color="k", lw=2, alpha=0.55,
                     label="demonstration" if i == 0 else None)
        ax_traj.plot(ph[:, 0], ph[:, 1], "--", color="C0", lw=1.8,
                     label="recovered cost" if i == 0 else None)
        if benchmark in ("unicycle", "segments"):
            for ox, oy, orr in np.asarray(c.obstacles):
                ax_traj.add_patch(plt.Circle((ox, oy), orr, color="0.75", zorder=0))
    if benchmark == "racing":
        R, W = cfg["track_radius"], cfg["track_halfwidth"]
        for rad, st in ((R - W, "-"), (R + W, "-")):
            a = np.linspace(0, 2 * np.pi, 200)
            ax_traj.plot(rad * np.cos(a), rad * np.sin(a), st, color="0.8", lw=1, zorder=0)
    ax_traj.set_aspect("equal")
    ax_traj.set_title(f"{benchmark}: demo vs recovered")
    ax_traj.legend(fontsize=7)

    if ax_field is not None and benchmark == "field":
        c0 = jax.tree.map(lambda a: a[0], ctxs)
        cen = np.asarray(c0.centers)
        wid = np.asarray(c0.widths)
        gx_, gy_ = np.meshgrid(np.linspace(-2.4, 2.4, 160), np.linspace(-1.8, 1.8, 120))
        pts = np.stack([gx_.ravel(), gy_.ravel()], -1)

        def render(weights):
            f = np.zeros(pts.shape[0])
            for k in range(cen.shape[0]):
                d2 = ((pts - cen[k]) ** 2).sum(-1)
                f += weights[k + 2] * np.exp(-d2 / (2 * wid[k] ** 2))
            return f.reshape(gx_.shape)

        tstar, that = np.asarray(theta_star), np.asarray(th)
        F1, F2 = render(tstar), render(that)
        vm = max(abs(F1).max(), abs(F2).max())
        ax_field[0].pcolormesh(gx_, gy_, F1, cmap="RdBu_r", vmin=-vm, vmax=vm)
        ax_field[1].pcolormesh(gx_, gy_, F2, cmap="RdBu_r", vmin=-vm, vmax=vm)
        for a_, t_ in zip(ax_field, ["true reward field", "recovered field"]):
            a_.set_aspect("equal")
            a_.set_title(t_, fontsize=9)
            a_.scatter(cen[:, 0], cen[:, 1], s=8, c="k")
        err = float(np.abs(tstar - that).sum())
        ax_field[1].set_xlabel(f"$\\|\\hat\\theta-\\theta^*\\|_1$ = {err:.3f}", fontsize=8)


def main(out_prefix: str = "ioc2d"):
    plot_scaling(f"{out_prefix}_scaling.png")

    fig = plt.figure(figsize=(13, 3.6))
    axf = fig.add_subplot(1, 4, 1)
    axr = fig.add_subplot(1, 4, 2)
    axu = fig.add_subplot(1, 4, 3)
    fit_and_draw("field", axf, k_bumps=6, bump_width=0.9)
    fit_and_draw("racing", axr)
    fit_and_draw("unicycle", axu)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_paths.png", dpi=160)
    print(f"wrote {out_prefix}_paths.png")

    fig2, ax2 = plt.subplots(1, 2, figsize=(8, 3.2))
    fit_and_draw("field", plt.figure().gca(), ax_field=ax2, k_bumps=6, bump_width=0.9)
    fig2.tight_layout()
    fig2.savefig(f"{out_prefix}_field.png", dpi=160)
    print(f"wrote {out_prefix}_field.png")


if __name__ == "__main__":
    tyro.cli(main)
