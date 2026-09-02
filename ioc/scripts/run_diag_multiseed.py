"""Fan the diagnostic suite out over (axis, seed) pairs and all free GPUs.

Every number the suite has produced so far is single-seed, against an inner
solver with ~40% run-to-run residual spread, so none of them carry an error
bar.  This runs each axis at several seeds and distributes the work over the
idle GPUs, one worker process per device, so the variance is characterized
rather than assumed.

Each (axis, seed) writes its own JSON.  That is deliberate: the suite driver's
single end-of-run write is what previously discarded completed axes when a
later one crashed, and per-job files also make the run RESUMABLE -- a job whose
output already exists is skipped, so an interrupted run picks up where it left
off rather than repeating hours of solves.

Usage
-----
    python scratch/run_diag_multiseed.py --seeds 6 --gpus 0,1,2,3
    python scratch/run_diag_multiseed.py --axes generalization --seeds 3
    python scratch/run_diag_multiseed.py --dry-run          # print the plan
"""

import argparse
import itertools
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "scratch" / "logs" / "multiseed"

# generalization and identifiability_conditioning first: they are the two axes
# that actually decide the robustness question and both have zero data so far.
DEFAULT_AXES = [
    "generalization",
    "identifiability_conditioning",
    "method_agreement",
    "demo_quality",
    "demo_diversity",
    "basis_size",
    "identifiable_refit",
]


def free_gpus(max_mem_mib=1000):
    """Indices of GPUs with little memory in use.  These boxes are shared, so
    a busy device is skipped rather than piled onto."""
    q = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True)
    out = []
    for line in q.stdout.strip().splitlines():
        idx, mem, util = (int(v) for v in line.split(","))
        if mem <= max_mem_mib:
            out.append(idx)
        else:
            print(f"  skipping GPU {idx}: {mem} MiB in use, {util}% util")
    return out


def jobs_for(axes, seeds):
    return [(a, s) for a, s in itertools.product(axes, range(seeds))]


def run_worker(gpu, jobs, extra_args):
    """One background shell per GPU, running its jobs sequentially."""
    lines = []
    for axis, seed in jobs:
        out = OUTDIR / f"{axis}__seed{seed}.json"
        log = OUTDIR / f"{axis}__seed{seed}.log"
        if out.exists():
            lines.append(f'echo "SKIP {axis} seed{seed} (exists)"')
            continue
        cmd = (f'echo "START {axis} seed{seed} on gpu{gpu} $(date +%H:%M:%S)" && '
               f'python -m ioc.diagnostics --which {axis} --seed {seed} '
               f'--out {out} {extra_args} > {log} 2>&1 '
               f'&& echo "OK   {axis} seed{seed} $(date +%H:%M:%S)" '
               f'|| echo "FAIL {axis} seed{seed} (see {log})"')
        lines.append(cmd)
    script = "\n".join(lines) if lines else 'echo "nothing to do"'

    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),
               XLA_PYTHON_CLIENT_PREALLOCATE="false",
               JAX_ENABLE_X64="1")
    worker_log = open(OUTDIR / f"worker_gpu{gpu}.log", "w")
    return subprocess.Popen(["bash", "-c", script], cwd=ROOT, env=env,
                            stdout=worker_log, stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axes", default=",".join(DEFAULT_AXES))
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--gpus", default="", help="comma list; default = auto-detect free")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--extra", default="", help="extra flags passed to ioc.diagnostics")
    args = ap.parse_args()

    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    OUTDIR.mkdir(parents=True, exist_ok=True)

    gpus = ([int(g) for g in args.gpus.split(",") if g.strip()]
            if args.gpus else free_gpus())
    if not gpus:
        sys.exit("no free GPUs -- all devices busy; not launching")

    jobs = jobs_for(axes, args.seeds)
    pending = [j for j in jobs if not (OUTDIR / f"{j[0]}__seed{j[1]}.json").exists()]
    # round-robin so each GPU gets a mix of axes rather than all of the slowest
    shards = {g: pending[i::len(gpus)] for i, g in enumerate(gpus)}

    print(f"{len(jobs)} jobs ({len(axes)} axes x {args.seeds} seeds), "
          f"{len(pending)} pending, over GPUs {gpus}")
    for g, sh in shards.items():
        print(f"  gpu{g}: {len(sh)} jobs -> {[f'{a}/s{s}' for a, s in sh]}")
    if args.dry_run:
        return

    procs = {g: run_worker(g, sh, args.extra) for g, sh in shards.items()}
    print(f"launched {len(procs)} workers; per-job JSON in {OUTDIR}")
    t0 = time.perf_counter()
    for g, p in procs.items():
        p.wait()
        print(f"  gpu{g} worker done rc={p.returncode} "
              f"({time.perf_counter() - t0:.0f}s elapsed)")
    print("all workers finished")


if __name__ == "__main__":
    main()
