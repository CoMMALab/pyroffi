"""Post-process resources/bench_dynamics_results.csv for easier reading.

Dedupes rows by (robot, solver, op, batch) keeping the LAST occurrence
(most recent rerun wins), then writes processed/summary tables to an
output directory. The original CSV is never modified.

Usage:
    python tests/process_benchmark_dynamics.py
    python tests/process_benchmark_dynamics.py --csv path/to/results.csv --outdir path/to/out
"""

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "resources" / "bench_dynamics_results.csv"
DEFAULT_OUTDIR = REPO_ROOT / "resources" / "bench_dynamics_processed"

DEDUP_KEYS = ["robot", "solver", "op", "batch"]


def dedup_keep_last(df):
    return df.drop_duplicates(subset=DEDUP_KEYS, keep="last").reset_index(drop=True)


def build_pivot(df, value_col):
    return (
        df.pivot_table(
            index=["robot", "op", "batch"],
            columns="solver",
            values=value_col,
            aggfunc="first",
        )
        .sort_index()
    )


BACKEND_SUMMARY_BATCH = 256


def build_backend_summary(df, batch=BACKEND_SUMMARY_BATCH):
    return (
        df[df["batch"] == batch]
        .groupby(["robot", "solver", "op", "backend"], as_index=False)
        .agg(
            n=("t_med_ms", "size"),
            t_med_ms=("t_med_ms", "median"),
            t_p95_ms=("t_p95_ms", "median"),
            rel_err_max=("rel_err_max", "max"),
            peak_vram_mb=("peak_vram_mb", "max"),
        )
        .sort_values(["op", "robot", "solver", "backend"])
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                     help=f"input CSV (default: {DEFAULT_CSV})")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR,
                     help=f"output directory for processed tables (default: {DEFAULT_OUTDIR})")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    n_before = len(df)
    deduped = dedup_keep_last(df)
    n_after = len(deduped)

    args.outdir.mkdir(parents=True, exist_ok=True)

    deduped_path = args.outdir / "deduped.csv"
    deduped.to_csv(deduped_path, index=False)

    t_med_pivot = build_pivot(deduped, "t_med_ms")
    t_med_path = args.outdir / "summary_t_med_ms.csv"
    t_med_pivot.to_csv(t_med_path)

    rel_err_pivot = build_pivot(deduped, "rel_err_max")
    rel_err_path = args.outdir / "summary_rel_err_max.csv"
    rel_err_pivot.to_csv(rel_err_path)

    backend_summary = build_backend_summary(deduped)
    backend_path = args.outdir / "summary_by_backend.csv"
    backend_summary.to_csv(backend_path, index=False)

    print(f"Input:  {args.csv} ({n_before} rows)")
    print(f"Deduped by {DEDUP_KEYS} (keep last): {n_after} rows "
          f"({n_before - n_after} duplicate(s) dropped)")
    print(f"Original CSV left untouched.\n")
    print(f"Wrote:")
    print(f"  {deduped_path}")
    print(f"  {t_med_path}         (t_med_ms pivoted: robot/op/batch x solver)")
    print(f"  {rel_err_path}     (rel_err_max pivoted: robot/op/batch x solver)")
    print(f"  {backend_path}      (median/p95/rel_err per robot/solver/op/backend, batch={BACKEND_SUMMARY_BATCH} only)")


if __name__ == "__main__":
    main()
