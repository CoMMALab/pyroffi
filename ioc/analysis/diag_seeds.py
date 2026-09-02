"""Collapse the per-(axis, seed) diagnostic JSONs into mean +/- std tables.

Reports `e_test` / `regret` / eigen-projected weight error ahead of raw
`param_err`: the goal for the single-segment work is BEHAVIORAL matching, and
with a ~26:1 Gram a weight error along a low-lambda direction is one the
demonstrations never constrained, so raw param_err is a diagnostic rather than
a score.

Usage:
    python scratch/aggregate_diag_seeds.py
    python scratch/aggregate_diag_seeds.py --axes method_agreement
"""

import argparse
import json
import pathlib
import statistics

OUTDIR = pathlib.Path(__file__).resolve().parents[1] / "scratch" / "logs" / "multiseed"

# ordered so the behavioural metrics are read first
PREFERRED = ["e_test", "e_demo", "e_global", "regret", "param_err_eig", "param_err"]


def stat(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return None
    if len(vals) == 1:
        return f"{vals[0]:.4g} (n=1)"
    m, s = statistics.mean(vals), statistics.stdev(vals)
    return f"{m:.4g} +/- {s:.2g} (n={len(vals)}, cv={s / (abs(m) + 1e-30):.0%})"


def collect(axis):
    rows = {}
    for f in sorted(OUTDIR.glob(f"{axis}__seed*.json")):
        seed = int(f.stem.split("seed")[-1])
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            print(f"  {f.name}: unreadable (job likely still running)")
            continue
        r = d.get(axis, d)
        if isinstance(r, dict) and "error" in r:
            print(f"  seed {seed}: FAILED -- {r['error']}")
            continue
        rows[seed] = r
    return rows


def flatten(obj, prefix=""):
    """Pull scalar metrics out of the nested per-axis result dicts, keyed by a
    path, so different axes can share one summarizer."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "trace":
                continue  # long, and not a metric
            out.update(flatten(v, f"{prefix}{k}." if prefix or True else k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}{i}."))
    elif isinstance(obj, (int, float)):
        out[prefix.rstrip(".")] = obj
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axes", default="")
    args = ap.parse_args()

    axes = ([a.strip() for a in args.axes.split(",") if a.strip()]
            or sorted({f.stem.split("__")[0] for f in OUTDIR.glob("*__seed*.json")}))
    if not axes:
        print(f"no per-seed results in {OUTDIR}")
        return

    for axis in axes:
        print(f"\n=== {axis} ===")
        rows = collect(axis)
        if not rows:
            print("  no completed seeds")
            continue
        flat = {s: flatten(r) for s, r in rows.items()}
        keys = sorted({k for f in flat.values() for k in f})
        # behavioural metrics first, then everything else
        keys.sort(key=lambda k: (min([i for i, p in enumerate(PREFERRED)
                                      if p in k] or [len(PREFERRED)]), k))
        print(f"  seeds: {sorted(rows)}")
        for k in keys:
            s = stat([flat[sd].get(k) for sd in flat])
            if s:
                print(f"    {k:<62s} {s}")


if __name__ == "__main__":
    main()
