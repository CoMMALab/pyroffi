"""Animate the outer fit: Path A's named cost weights and Path B's RKHS cost.

Two stages, deliberately separable, because the fit is expensive (Path B's
`jit_loss_a` compile alone measured 3545s) and the rendering is not:

    record   -- re-run the wide fit, saving every iterate to an .npz
    render   -- turn a saved .npz into a .gif, no GPU, no solves

so you can re-style the figure as many times as you like off one fit.

    python -m iosp.viz.fit_animation record --path a --out iosp/data/viz/fitA.npz
    python -m iosp.viz.fit_animation render iosp/data/viz/fitA.npz --out fitA.gif
    python -m iosp.viz.fit_animation both   --path a --out iosp/data/viz/fitA.npz

WHAT IS SHOWN

Path A (known cost library, K=9): the 9 named weights as bars against their
ground truth, since every coordinate has a meaning worth reading off.

Path B (unknown cost, RKHS, K=6+M): the weights are M kernel coefficients and
mean nothing individually, so the bar chart is replaced by the object that
does have meaning -- the induced transport cost itself,

    C_w(u) = sum_j theta_j * phi_j(u)^2,

rendered over a 2D slice of the descriptor space `u = [q, dq, (d_link)]`.
That surface is what Path B is actually learning; the coefficient stem plot is
kept alongside it only to show the weights settling.

The slice is evaluated directly from the same `_rff_residual_fn` feature map
the study fits, so no inner solves are needed to draw it -- which is why the
render stage is GPU-free.

CAVEAT worth keeping attached to any figure this produces: `ioc.outer.adam`
returns its BEST iterate, and these fits do not converge monotonically -- they
bottom out and then climb (Study 1 measured 0.13128 at step 9 rising to
0.27478 by step 40, and Study 2's regime (a) spiked to 1622.97 mid-run). The
animation shows the true trajectory, so the final frame is NOT the reported
number. The best-so-far frame is marked.
"""

import argparse
import os
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# stage 1: record
# ---------------------------------------------------------------------------

def record_fit(which="a", n_steps=None, lr=None, M=256, ls=10.0,
               mode="geom", form="lin", seed=0):
    """Re-run the wide fit, keeping every iterate.

    `ioc.outer.adam` returns only the best `z` and a loss trace, so this
    reimplements its loop -- same `optax.adamw(lr)`, same order of operations
    -- rather than changing shared solver code for a plotting script.  The
    recorded trajectory is therefore the one the study actually takes; if
    `ioc.outer.adam` changes, this must be re-checked against it.
    """
    import jax.numpy as jnp
    import optax
    import iosp.fit.parametric as st

    n_steps = st.N_STEPS if n_steps is None else n_steps
    lr = st.LR if lr is None else lr

    t0 = time.perf_counter()
    built_a, built_b = st.build_same_demo(M=M, ls=ls, mode=mode, form=form, seed=seed)
    built = built_a if which == "a" else built_b
    print(f"[record] built {built['label']} in {time.perf_counter() - t0:.0f}s", flush=True)

    gf, K = built["gf"], built["K"]
    u = jnp.zeros(K, dtype=jnp.float32)
    opt = optax.adamw(lr, weight_decay=0.0)
    opt_state = opt.init(u)

    u_hist, loss_hist = [np.asarray(u)], []
    for t in range(n_steps):
        val, g = gf(u)
        loss_hist.append(float(val))
        updates, opt_state = opt.update(g, opt_state, u)
        u = optax.apply_updates(u, updates)
        u_hist.append(np.asarray(u))
        print(f"[record] step {t + 1:3d}/{n_steps}  loss={float(val):.6f}", flush=True)
    # loss at the final iterate too, so len(loss) == len(u_hist)
    loss_hist.append(float(gf(u)[0]))

    u_hist = np.stack(u_hist)
    theta_hist = np.stack([built["theta_of"](jnp.asarray(uu)) for uu in u_hist])

    out = dict(
        which=which, label=built["label"], K=K, n_ik=built["n_ik"],
        names=np.array(built["names"], dtype=object),
        u_hist=u_hist, theta_hist=theta_hist,
        loss_hist=np.asarray(loss_hist),
        M=M, ls=ls, mode=mode, form=form, seed=seed,
    )
    if built["theta_star"] is not None:
        out["theta_star"] = np.asarray(built["theta_star"])
    return out


def _save(rec, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(path, **rec)
    print(f"[record] wrote {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# the RKHS cost surface (render-side, no solves)
# ---------------------------------------------------------------------------

def rkhs_cost_grid(theta_transport, M, ls, mode, form, seed, dof=7,
                   n_grid=60, span=None, axes=(0, 7)):
    """`C_w` over a 2D slice of descriptor space, all other coordinates at 0.

    Rebuilds the SAME feature map as `_rff_residual_fn` -- same PRNGKey
    (`seed + 7`), same `Omega`/`b`/`amp`, same `form` -- but applied to a grid
    of descriptors rather than to solved trajectory states, so it needs no
    inner solve.  `axes` picks which two descriptor coordinates to sweep;
    default is (q_0, dq_0).

    `span` defaults to `3 * ls`, NOT a fixed window: `Omega ~ N(0, 1) / ls`,
    so at the retuned `ls=10` the feature phases `u @ Omega.T` barely move
    across a +/-2 box and the surface renders flat.  The window has to scale
    with the lengthscale to show the structure the kernel actually has.
    """
    import jax
    import jax.numpy as jnp

    span = 3.0 * ls if span is None else span

    dim = 2 * dof + (13 if mode == "geom" else 0)
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed + 7))
    Omega = np.asarray(jax.random.normal(k1, (M, dim), dtype=jnp.float32)) / ls
    b = np.asarray(jax.random.uniform(k2, (M,), dtype=jnp.float32)) * 2.0 * np.pi
    amp = np.sqrt(2.0 / M)

    g = np.linspace(-span, span, n_grid)
    A, B = np.meshgrid(g, g, indexing="ij")
    U = np.zeros((n_grid * n_grid, dim), dtype=np.float64)
    U[:, axes[0]] = A.ravel()
    U[:, axes[1]] = B.ravel()

    z = U @ Omega.T + b
    ph = amp * np.cos(z) if form == "sq" else np.sqrt(2.0) * np.sin(0.5 * z)
    C = (ph ** 2) @ np.asarray(theta_transport, dtype=np.float64)
    return g, C.reshape(n_grid, n_grid)


def _transport_slice(theta_full, n_ik, M):
    """The transport block of a `theta_of(u)` vector.

    `theta_of` returns [theta_ik, softmax(all trajopt logits)] and
    `_split_trajopt_m` lays the trajopt block out in `pp.PHASES` order, so
    transport starts after approach's and grasp's features.
    """
    import iosp.model.pickplace as pp
    off = n_ik
    for p in pp.PHASES:
        n = M if p == "transport" else len(pp.SEGMENT_FEATURES[p])
        if p == "transport":
            return theta_full[off:off + n]
        off += n
    raise AssertionError("no transport phase")


# ---------------------------------------------------------------------------
# stage 2: render
# ---------------------------------------------------------------------------

def render(npz_path, out_path, fps=6, n_grid=60, dpi=110):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    d = np.load(npz_path, allow_pickle=True)
    which = str(d["which"])
    label = str(d["label"])
    theta_hist = d["theta_hist"]
    loss = d["loss_hist"]
    names = list(d["names"])
    n_frames = len(theta_hist)
    best_t = int(np.argmin(loss))

    plt.rcParams.update({"figure.facecolor": "white", "axes.grid": True,
                         "grid.alpha": 0.25, "font.size": 9})

    if which == "a":
        fig, (ax_w, ax_l) = plt.subplots(1, 2, figsize=(11, 4.2))
        star = d["theta_star"] if "theta_star" in d else None
        idx = np.arange(len(names))
        bars = ax_w.bar(idx, theta_hist[0], color="#3b7dd8", zorder=3)
        if star is not None:
            ax_w.bar(idx, star, facecolor="none", edgecolor="#d9534f",
                     linewidth=1.6, zorder=4, label="ground truth")
            ax_w.legend(loc="upper right", fontsize=8)
        ax_w.set_xticks(idx)
        ax_w.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        hi = float(max(theta_hist.max(), star.max() if star is not None else 0))
        ax_w.set_ylim(min(0.0, float(theta_hist.min())) - 0.02, hi * 1.15 + 0.02)
        ax_w.set_ylabel("weight")
        ax_w.set_title("named cost weights")
        artists = list(bars)
    else:
        fig = plt.figure(figsize=(13, 4.4))
        ax_c = fig.add_subplot(1, 3, 1)
        ax_k = fig.add_subplot(1, 3, 2)
        ax_l = fig.add_subplot(1, 3, 3)
        M, ls = int(d["M"]), float(d["ls"])
        mode, form, seed = str(d["mode"]), str(d["form"]), int(d["seed"])
        n_ik = int(d["n_ik"])

        tr_hist = np.stack([_transport_slice(th, n_ik, M) for th in theta_hist])
        grids = [rkhs_cost_grid(tr, M, ls, mode, form, seed, n_grid=n_grid)[1]
                 for tr in tr_hist]
        g = rkhs_cost_grid(tr_hist[0], M, ls, mode, form, seed, n_grid=n_grid)[0]
        vmin = float(min(gr.min() for gr in grids))
        vmax = float(max(gr.max() for gr in grids))

        im = ax_c.imshow(grids[0].T, origin="lower", cmap="viridis",
                         extent=[g[0], g[-1], g[0], g[-1]], vmin=vmin, vmax=vmax,
                         aspect="auto")
        fig.colorbar(im, ax=ax_c, fraction=0.046)
        ax_c.set_xlabel("descriptor $q_0$")
        ax_c.set_ylabel("descriptor $\\dot q_0$")
        ax_c.set_title("learned RKHS transport cost $C_w$")
        ax_c.grid(False)

        stem, = ax_k.plot(np.arange(M), tr_hist[0], lw=0.8, color="#3b7dd8")
        ax_k.set_xlim(0, M - 1)
        ax_k.set_ylim(float(tr_hist.min()) * 1.1 - 1e-6, float(tr_hist.max()) * 1.15 + 1e-6)
        ax_k.set_xlabel("kernel feature $j$")
        ax_k.set_ylabel(r"$\theta_j$")
        ax_k.set_title(f"kernel coefficients (M={M})")
        artists = [im, stem]

    ax_l.plot(loss, color="#444", lw=1.4, zorder=2)
    ax_l.axvline(best_t, color="#2e8b57", ls="--", lw=1.2, zorder=1,
                 label=f"best iterate (step {best_t}, {loss[best_t]:.4f})")
    ax_l.set_yscale("log")
    ax_l.set_xlabel("outer step")
    ax_l.set_ylabel("fit loss")
    ax_l.set_title("outer loss (best iterate is what gets reported)")
    ax_l.legend(loc="upper right", fontsize=8)
    marker, = ax_l.plot([0], [loss[0]], "o", color="#d9534f", ms=7, zorder=5)

    sup = fig.suptitle("")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    def update(t):
        if which == "a":
            for bar, h in zip(artists, theta_hist[t]):
                bar.set_height(float(h))
        else:
            artists[0].set_data(grids[t].T)
            artists[1].set_ydata(tr_hist[t])
        marker.set_data([t], [loss[t]])
        flag = "  <- best" if t == best_t else ""
        sup.set_text(f"{label}   step {t}/{n_frames - 1}   loss={loss[t]:.5f}{flag}")
        return (*artists, marker, sup)

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"[render] wrote {out_path}  ({os.path.getsize(out_path) / 1e6:.1f} MB, "
          f"{n_frames} frames @ {fps}fps)")


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("record", "render", "both"))
    ap.add_argument("npz", nargs="?", default=None,
                    help="render: the .npz to read")
    ap.add_argument("--path", choices=("a", "b"), default="a")
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--M", type=int, default=256)
    ap.add_argument("--ls", type=float, default=10.0)
    ap.add_argument("--mode", default="geom")
    ap.add_argument("--form", default="lin")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--grid", type=int, default=60)
    a = ap.parse_args(argv)

    if a.stage in ("record", "both"):
        npz = a.out if (a.stage == "record" and a.out) else (
            a.npz or f"iosp/data/viz/fit_{a.path}.npz")
        rec = record_fit(which=a.path, n_steps=a.steps, lr=a.lr, M=a.M, ls=a.ls,
                         mode=a.mode, form=a.form, seed=a.seed)
        _save(rec, npz)
    else:
        npz = a.npz
        if npz is None:
            ap.error("render needs an .npz path")

    if a.stage in ("render", "both"):
        out = a.out if a.stage == "render" and a.out else f"iosp/data/viz/fit_{a.path}.gif"
        if a.stage == "both" and a.out:
            out = a.out.replace(".npz", ".gif")
        render(npz, out, fps=a.fps, n_grid=a.grid)


if __name__ == "__main__":
    sys.exit(main())
