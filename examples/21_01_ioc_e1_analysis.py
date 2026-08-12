"""Aggregate and plot E1 results (see 21_00_ioc_e1_synthetic.py).

Produces the two figures E1 is meant to support:
  (a) weight-recovery error vs number of demonstration contexts M, per method
  (b) accuracy vs number of forward trajopt solves -- the sample-efficiency claim
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tyro

METHODS = ["implicit", "unrolled", "fd", "cmaes", "kkt", "random"]
COLORS = dict(zip(METHODS, ["C0", "C1", "C2", "C3", "C4", "0.6"]))


def main(results: str = "e1_results.json", out_prefix: str = "e1"):
    with open(results) as f:
        data = json.load(f)
    res = data["results"]
    keys = sorted({(int(k.split("_")[0][1:])) for k in res})
    seeds = sorted({int(k.split("_s")[1]) for k in res})

    def gather(m, method, field):
        return [
            res[f"M{m}_s{s}"]["methods"][method][field]
            for s in seeds
            if f"M{m}_s{s}" in res
        ]

    # -- (a) recovery vs M -------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, field, label in zip(
        axes,
        ["theta_l1", "regret", "ee_rmse"],
        [r"$\|\hat\theta-\theta^*\|_1$", "cost regret", "EE RMSE [m]"],
    ):
        for method in METHODS:
            med = [np.median(gather(m, method, field)) for m in keys]
            lo = [np.percentile(gather(m, method, field), 25) for m in keys]
            hi = [np.percentile(gather(m, method, field), 75) for m in keys]
            ax.plot(keys, med, "o-", color=COLORS[method], label=method)
            ax.fill_between(keys, lo, hi, color=COLORS[method], alpha=0.15)
        ax.set_xscale("log")
        ax.set_xlabel("demonstration contexts $M$")
        ax.set_ylabel(label)
        if field != "theta_l1":
            ax.set_yscale("log")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("E1: parameter recovery vs demonstration diversity (median, IQR over seeds)")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_recovery.png", dpi=160)

    # -- (b) loss vs forward solves ---------------------------------------
    m_max = keys[-1]
    fig, ax = plt.subplots(figsize=(5, 3.8))
    for method in ["implicit", "unrolled", "fd", "cmaes"]:
        hs = [
            res[f"M{m_max}_s{s}"]["methods"][method]["history"]
            for s in seeds
            if f"M{m_max}_s{s}" in res
        ]
        hs = [h for h in hs if h]
        if not hs:
            continue
        n = min(len(h) for h in hs)
        solves = np.array([p[0] for p in hs[0][:n]])
        vals = np.median(np.array([[p[1] for p in h[:n]] for h in hs]), axis=0)
        ax.plot(solves, vals, color=COLORS[method], label=method)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("forward trajopt solves")
    ax.set_ylabel("outer loss")
    ax.set_title(f"Sample efficiency (M={m_max})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_solves.png", dpi=160)

    # -- identifiability + gradient-fidelity table -------------------------
    print(f"{'M':>4s} {'lam_min(G)':>12s} {'cond(G)':>12s} {'inner|grad|':>12s} {'cos(imp,fd)':>12s}")
    for m in keys:
        c = [res[f"M{m}_s{s}"]["certificate"] for s in seeds if f"M{m}_s{s}" in res]
        lam = np.median([x["gram_lambda_min"] for x in c])
        cond = np.median([x["gram_cond"] for x in c])
        gn = np.median([x.get("inner_grad_norm_med", np.nan) for x in c])
        cs = [x["gradients"]["fd_eps_0.001"]["cos"] for x in c if "gradients" in x]
        cos = np.median(cs) if cs else np.nan
        print(f"{m:>4d} {lam:12.3e} {cond:12.3e} {gn:12.3e} {cos:12.6f}")

    print(f"\nwrote {out_prefix}_recovery.png, {out_prefix}_solves.png")


if __name__ == "__main__":
    tyro.cli(main)
