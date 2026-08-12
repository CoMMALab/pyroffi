"""Aggregate the E1 demonstration-noise sweep (e1_sigma*.json).

This is E1's headline result.  With noiseless demonstrations the demonstration
is an exact stationary point of the true cost, so KKT/feature-matching recovers
theta* exactly using zero forward solves and differentiating through the
optimizer buys nothing.  The value of the differentiable method is a robustness
claim: it should degrade gracefully as demonstrations become suboptimal, where
feature matching -- which assumes stationarity -- should not.
"""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tyro

METHODS = ["implicit", "fd", "cmaes", "kkt", "unrolled", "random"]
COLORS = dict(zip(METHODS, ["C0", "C2", "C3", "C4", "C1", "0.6"]))


def main(pattern: str = "e1_sigma*.json", out_prefix: str = "e1_noise"):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no files matching {pattern}")

    rows = {}
    for path in files:
        with open(path) as f:
            data = json.load(f)
        sigma = data["demo_noise"]
        res = data["results"]
        rows[sigma] = res
        print(f"{os.path.basename(path)}: sigma={sigma}  {len(res)} trials")

    sigmas = sorted(rows)

    def gather(sigma, method, field):
        return [t["methods"][method][field] for t in rows[sigma].values()
                if method in t["methods"]]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, field, label, logy in zip(
        axes,
        ["theta_l1", "regret", "ee_rmse"],
        [r"$\|\hat\theta-\theta^*\|_1$", "cost regret", "EE RMSE [m]"],
        [False, True, True],
    ):
        for m in METHODS:
            med, lo, hi = [], [], []
            for s in sigmas:
                v = gather(s, m, field)
                if not v:
                    med.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
                med.append(np.median(v))
                lo.append(np.percentile(v, 25))
                hi.append(np.percentile(v, 75))
            ax.plot(sigmas, med, "o-", color=COLORS[m], label=m)
            ax.fill_between(sigmas, lo, hi, color=COLORS[m], alpha=0.15)
        ax.set_xlabel(r"demonstration noise $\sigma$ [rad]")
        ax.set_ylabel(label)
        if logy:
            ax.set_yscale("log")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle(
        "E1: recovery vs demonstration suboptimality (median, IQR over seeds)"
    )
    fig.tight_layout()
    fig.savefig(f"{out_prefix}.png", dpi=160)

    print(f"\n{'sigma':>7s} " + " ".join(f"{m:>16s}" for m in METHODS))
    print("  (theta L1 median (IQR))")
    for s in sigmas:
        cells = []
        for m in METHODS:
            v = gather(s, m, "theta_l1")
            if not v:
                cells.append("--")
                continue
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            cells.append(f"{med:6.3f}({q3 - q1:5.3f})")
        print(f"{s:>7.3f} " + " ".join(f"{c:>16s}" for c in cells))

    print(f"\n{'sigma':>7s} " + " ".join(f"{m:>14s}" for m in METHODS))
    print("  (cost regret, median)")
    for s in sigmas:
        cells = []
        for m in METHODS:
            v = gather(s, m, "regret")
            cells.append(f"{np.median(v):.3e}" if v else "--")
        print(f"{s:>7.3f} " + " ".join(f"{c:>14s}" for c in cells))

    # Solve counts are identical across sigma; report once.
    print("\nforward solves (median):")
    for m in METHODS:
        v = gather(sigmas[0], m, "n_solves")
        if v:
            print(f"  {m:>9s}: {np.median(v):.0f}")
    print(f"\nwrote {out_prefix}.png")


if __name__ == "__main__":
    tyro.cli(main)
