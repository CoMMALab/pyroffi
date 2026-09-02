"""Eigenparameter recovery under rank deficiency — 2-panel PDF for IEEE T-RO.

Panel (a): eigenvalue spectrum of the feature-gradient Gram matrix G (log
scale, trace-normalised, descending).  The 95%-cumulative-trace rule splits
the K eigendirections into an identifiable block (top-k) and a structurally
near-null block; the split is drawn as a divider, so the rank deficiency is
visible rather than asserted.

Panel (b): the recovery error ||theta_hat - theta*|| decomposed onto those
two orthogonal subspaces.  The point of the figure is that the raw norm is
dominated by the null component -- the part of theta the demonstrations
structurally cannot see -- while the identifiable component is small.  Raw
L2 param error therefore understates recovery; the identifiable projection
is the honest number.

    python -m iosp.viz.fig_eigenspectrum <eigen_projection.npz>
    python -m iosp.viz.fig_eigenspectrum <npz> --out fig_eigenspectrum.pdf
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COL2 = 7.16

C_TOP = "#0072B2"    # identifiable
C_NULL = "#D55E00"   # near-null
C_RAW = "#666666"


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


def _panel_label(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=9,
            va="bottom", ha="left")


def render(input_path, out_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)
    d = np.load(input_path, allow_pickle=True)

    ev = np.asarray(d["eigvals"], float)
    desc = np.argsort(ev)[::-1]
    ev_desc = ev[desc]
    k = int(d["k"])
    K = len(ev_desc)
    names = list(d["order"]) if "order" in d else [str(i + 1) for i in range(K)]
    eigvecs = np.asarray(d["eigvecs"], float)
    names_desc = [names[int(np.argmax(np.abs(eigvecs[:, desc[i]])))] for i in range(K)]

    raw_err = float(d["raw_err"])
    top_err = float(d["top_err"])
    null_err = float(d["null_err"])

    _set_style()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(COL2 + 0.5, 2.8),
                                     gridspec_kw={"width_ratios": [1.5, 1.0]})

    # --- (a) Gram spectrum ------------------------------------------------
    x = np.arange(K)
    floor = max(ev_desc[ev_desc > 0].min() * 1e-1, 1e-12) if np.any(ev_desc > 0) else 1e-12
    heights = np.clip(ev_desc, floor, None)
    colors = [C_TOP if i < k else C_NULL for i in range(K)]
    ax_a.bar(x, heights, color=colors, width=0.72, edgecolor="none", zorder=3)
    ax_a.set_yscale("log")
    ax_a.set_ylim(bottom=floor)
    ax_a.axvline(k - 0.5, color="#333333", lw=0.8, ls="--", zorder=4)
    ax_a.set_xlabel(r"eigendirection of $G$ (descending)")
    ax_a.set_ylabel("eigenvalue (trace-norm.)")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(names_desc, rotation=45, ha="right", fontsize=6)
    ax_a.grid(True, axis="y", which="both", alpha=0.2, lw=0.4, zorder=0)
    for sp in ("top", "right"):
        ax_a.spines[sp].set_visible(False)
    ax_a.plot([], [], color=C_TOP, lw=4,
              label=f"identifiable (top-{k}, 95% trace)")
    ax_a.plot([], [], color=C_NULL, lw=4,
              label=f"near-null ({K - k}-dim)")
    ax_a.legend(loc="upper right", handlelength=1.2)
    _panel_label(ax_a, "(a)")

    # --- (b) error decomposition -------------------------------------------
    labels = ["raw\n" + r"$\|\Delta\theta\|$",
              f"identifiable\n(top-{k})",
              f"near-null\n({K - k}-dim)"]
    vals = [raw_err, top_err, null_err]
    bar_colors = [C_RAW, C_TOP, C_NULL]
    xb = np.arange(3)
    ax_b.bar(xb, vals, color=bar_colors, width=0.6, edgecolor="none", zorder=3)
    for xi, v in zip(xb, vals):
        ax_b.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels(labels)
    ax_b.set_ylabel(r"recovery error $\|\hat\theta-\theta^\star\|_2$")
    ax_b.set_ylim(0, max(vals) * 1.25)
    ax_b.grid(True, axis="y", alpha=0.2, lw=0.4, zorder=0)
    for sp in ("top", "right"):
        ax_b.spines[sp].set_visible(False)
    _panel_label(ax_b, "(b)")

    fig.tight_layout(pad=0.8, w_pad=2.0, rect=(0.02, 0, 1, 1))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    stem = os.path.splitext(out_path)[0]
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}")
    plt.close(fig)
    print(f"[fig_eigenspectrum] k={k}/{K}  raw={raw_err:.4f} "
          f"top={top_err:.4f} null={null_err:.4f} -> {stem}.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", default="iosp/figures/fig_eigenspectrum.pdf")
    a = ap.parse_args()
    render(a.input, a.out)
