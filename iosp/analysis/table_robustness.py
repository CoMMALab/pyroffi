"""Table 1: multi-seed robustness — LaTeX table with booktabs for IEEE T-RO.

Reads all multi-seed `.npz` files from a results directory, aggregates by
loss space (joint vs EE), and reports: mean held-out RMSE, std, joint-wins
count, candidate spread, and dominant branch.

    python -m iosp.analysis.table_robustness iosp/data/results/multistart/
    python -m iosp.analysis.table_robustness iosp/data/results/multistart/ --out table1.tex
"""

import argparse
import glob
import os
import re

import numpy as np


def _seed_from_path(path):
    m = re.search(r"s(?:eed)?(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0


def _load_runs(results_dir):
    """Load all multistart npz files, keyed by (space, seed)."""
    runs = {}
    for f in sorted(glob.glob(os.path.join(results_dir, "*.npz"))):
        bn = os.path.basename(f)
        if "spectrum" in bn:
            continue
        d = np.load(f, allow_pickle=True)
        if "ee_held_hist" not in d:
            continue
        space = str(d["space"]) if "space" in d else (
            "joint" if "joint" in bn else "ee")
        seed = _seed_from_path(f)
        S = int(d["S"]) if "S" in d else 3
        B = int(d["B"]) if "B" in d else 3
        winner = int(d["winner"])
        ee_held = d["ee_held_hist"]   # (F, C)
        runs[(space, seed)] = dict(
            ee_held_final=ee_held[-1, winner],
            spread=float(ee_held[-1].max() / ee_held[-1].min()),
            best_branch=winner // S,
            B=B, S=S, winner=winner)
    return runs


def render(input_path, out_path):
    """input_path: directory of multistart npz files. out_path: .tex file."""
    runs = _load_runs(input_path)
    if not runs:
        print(f"[table_robustness] no valid npz files in {input_path}")
        return

    seeds_by_space = {}
    for (space, seed), v in runs.items():
        seeds_by_space.setdefault(space, {})[seed] = v

    all_seeds = sorted(set(s for _, s in runs))
    paired_seeds = [s for s in all_seeds
                    if s in seeds_by_space.get("joint", {})
                    and s in seeds_by_space.get("ee", {})]

    joint_wins = sum(
        1 for s in paired_seeds
        if seeds_by_space["joint"][s]["ee_held_final"] <
           seeds_by_space["ee"][s]["ee_held_final"])

    rows = {}
    for space in ("joint", "ee"):
        if space not in seeds_by_space:
            continue
        vals = [v["ee_held_final"] for v in seeds_by_space[space].values()]
        spreads = [v["spread"] for v in seeds_by_space[space].values()]
        branches = [v["best_branch"] for v in seeds_by_space[space].values()]
        from collections import Counter
        dom_branch = Counter(branches).most_common(1)[0][0] if branches else -1
        rows[space] = dict(
            mean=np.mean(vals), std=np.std(vals, ddof=1) if len(vals) > 1 else 0.0,
            n=len(vals), spread_mean=np.mean(spreads),
            dom_branch=dom_branch)

    # Print to stdout
    print(f"\n{'Space':>8} {'Mean RMSE':>10} {'Std':>8} {'N':>4} "
          f"{'Spread':>8} {'Dom. br.':>9}")
    for space in ("joint", "ee"):
        if space not in rows:
            continue
        r = rows[space]
        print(f"{space:>8} {r['mean']:10.5f} {r['std']:8.5f} {r['n']:4d} "
              f"{r['spread_mean']:8.1f}x {r['dom_branch']:9d}")
    if paired_seeds:
        print(f"\nJoint wins on {joint_wins}/{len(paired_seeds)} paired seeds")

    # Write LaTeX
    lines = [
        r"\begin{table}",
        r"  \centering",
        r"  \caption{Multi-seed robustness: joint-space vs.\ end-effector loss. "
        + f"Joint wins on {joint_wins}/{len(paired_seeds)} paired seeds." + "}",
        r"  \label{tab:robustness}",
        r"  \begin{tabular}{lccccc}",
        r"    \toprule",
        r"    Loss space & Mean RMSE & Std & Seeds & Spread & Dom.\ branch \\",
        r"    \midrule",
    ]
    for space in ("joint", "ee"):
        if space not in rows:
            continue
        r = rows[space]
        label = "Joint" if space == "joint" else "End-eff."
        lines.append(
            f"    {label} & {r['mean']:.5f} & {r['std']:.5f} & "
            f"{r['n']} & {r['spread_mean']:.1f}$\\times$ & {r['dom_branch']} \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    tex = "\n".join(lines) + "\n"

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(tex)
    print(f"[table_robustness] wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default="iosp/data/results/multistart/")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join("iosp", "figures", "table1_robustness.tex")
    render(a.results_dir, out)
