"""Fig 3: stage-2 ablation — 2-panel PDF for IEEE T-RO.

Panel (a): per-seed held-out RMSE, paired connected dots for "full 3-stage"
vs "refine only (no stage 2)".
Panel (b): mean +/- std summary bars for each condition, plus oracle/baseline
reference lines.

Reads `.npz` files from a directory. Each file contains:
  test_rmse_full, test_rmse_no_stage2, baseline_uniform, oracle, seed.

    python -m iosp.viz.fig_ablation iosp/data/results/ablation/
"""

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COL2 = 7.16
C_FULL = "#0072B2"
C_ABLATE = "#D55E00"
C_ORACLE = "#009E73"
C_BASE = "#999999"


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


def _seed_from_path(path):
    m = re.search(r"s(?:eed)?(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0


def render(input_path, out_path):
    """input_path: directory of ablation npz files. out_path: pdf file."""
    _set_style()

    files = sorted(glob.glob(os.path.join(input_path, "*.npz")))
    if not files:
        print(f"[fig_ablation] no npz files in {input_path}")
        return

    data = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        entry = dict(seed=_seed_from_path(f))
        if "test_rmse_full" in d:
            entry["full"] = float(d["test_rmse_full"])
            entry["no_s2"] = float(d["test_rmse_no_stage2"])
        elif "fit" in d:
            fit = d["fit"].item() if hasattr(d["fit"], "item") else d["fit"]
            entry["full"] = float(fit["test_rmse"])
            ns2 = d["no_stage2"].item() if hasattr(d["no_stage2"], "item") else d["no_stage2"]
            entry["no_s2"] = float(ns2["test_rmse"])
        else:
            continue
        if "baseline_uniform" in d:
            entry["baseline"] = float(d["baseline_uniform"])
        if "oracle" in d:
            entry["oracle"] = float(d["oracle"])
        data.append(entry)

    if not data:
        print(f"[fig_ablation] no valid ablation data in {input_path}")
        return

    data.sort(key=lambda x: x["seed"])
    n = len(data)
    full_vals = np.array([d["full"] for d in data])
    no_s2_vals = np.array([d["no_s2"] for d in data])

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(COL2, 2.4),
                                      gridspec_kw=dict(width_ratios=[1.2, 0.8],
                                                       wspace=0.35))

    # --- panel (a): paired per-seed comparison ---
    x = np.arange(n)
    for i in range(n):
        ax_a.plot([0, 1], [full_vals[i], no_s2_vals[i]], "-", color="#888888",
                  lw=0.7, alpha=0.5, zorder=1)
    ax_a.scatter(np.zeros(n), full_vals, color=C_FULL, s=30, zorder=3,
                 marker="o", edgecolors="white", linewidths=0.5,
                 label="Full 3-stage")
    ax_a.scatter(np.ones(n), no_s2_vals, color=C_ABLATE, s=30, zorder=3,
                 marker="s", edgecolors="white", linewidths=0.5,
                 label="Refine only")
    ax_a.set_xticks([0, 1])
    ax_a.set_xticklabels(["Full\n3-stage", "Refine\nonly"])
    ax_a.set_ylabel("held-out EE RMSE (m)")
    ax_a.set_xlim(-0.4, 1.4)
    ax_a.legend(loc="upper right", fontsize=6)
    for sp in ("top", "right"):
        ax_a.spines[sp].set_visible(False)
    ax_a.grid(True, axis="y", alpha=0.2, lw=0.4)
    ax_a.text(-0.02, 1.04, "(a)", transform=ax_a.transAxes, fontsize=9,
              fontweight="bold", va="bottom", ha="left")

    # --- panel (b): summary bars ---
    conditions = ["Full 3-stage", "Refine only"]
    means = [full_vals.mean(), no_s2_vals.mean()]
    stds = [full_vals.std(ddof=1) if n > 1 else 0, no_s2_vals.std(ddof=1) if n > 1 else 0]
    colors = [C_FULL, C_ABLATE]
    x_bar = np.arange(len(conditions))
    bars = ax_b.bar(x_bar, means, yerr=stds, width=0.5, color=colors,
                    edgecolor="white", linewidth=0.5, capsize=3, zorder=3,
                    error_kw=dict(lw=0.8, capthick=0.8))

    baselines = [d.get("baseline") for d in data if "baseline" in d]
    oracles = [d.get("oracle") for d in data if "oracle" in d]
    if baselines:
        bl = np.mean(baselines)
        ax_b.axhline(bl, color=C_BASE, ls=":", lw=0.8, zorder=1, label="baseline")
    if oracles:
        orc = np.mean(oracles)
        ax_b.axhline(orc, color=C_ORACLE, ls="--", lw=0.8, zorder=1, label="oracle")

    ax_b.set_xticks(x_bar)
    ax_b.set_xticklabels(conditions, fontsize=7)
    ax_b.set_ylabel("mean held-out\nRMSE (m)")
    for sp in ("top", "right"):
        ax_b.spines[sp].set_visible(False)
    ax_b.legend(loc="upper right", fontsize=6)
    ax_b.grid(True, axis="y", alpha=0.2, lw=0.4)
    ax_b.text(-0.02, 1.04, "(b)", transform=ax_b.transAxes, fontsize=9,
              fontweight="bold", va="bottom", ha="left")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path)
    png_path = out_path.rsplit(".", 1)[0] + ".png" if out_path.endswith(".pdf") else None
    if png_path:
        fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"[fig_ablation] wrote {out_path}" +
          (f" + {png_path}" if png_path else ""))

    # Print summary
    print(f"\n  Full 3-stage:  {full_vals.mean():.5f} +/- {full_vals.std(ddof=1) if n > 1 else 0:.5f}")
    print(f"  Refine only:   {no_s2_vals.mean():.5f} +/- {no_s2_vals.std(ddof=1) if n > 1 else 0:.5f}")
    rel = abs(means[0] - means[1]) / max(means[1], 1e-30)
    print(f"  Relative diff: {100 * (1 - means[0]/means[1]):+.1f}%"
          f" ({'load-bearing' if rel > 0.05 else 'washes out'})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default="iosp/data/results/ablation/")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join("iosp", "figures", "fig3_ablation.pdf")
    render(a.results_dir, out)
