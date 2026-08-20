#!/usr/bin/env bash
# Parallel motion-validation benchmark across geometric backends.
#
# Each backend needs a different conda environment -- pyroffi/pybullet want JAX
# and numpy>=2, cuRobo pins numpy<2 and has no JAX -- so they cannot share an
# interpreter. This script runs each in its own process and appends to one CSV.
#
# The skeletons are generated ONCE and passed to every backend as a .npz. That
# is the basis of the whole comparison: identical work, so any difference is
# throughput rather than luck. Regenerating per backend would silently invalidate
# every number here.
#
# Usage:
#   bash benchmarks/run_parallel_bench.sh                    # defaults
#   BATCHES="1 8 32 128 512" REPS=5 bash benchmarks/run_parallel_bench.sh
#   SKIP_CUROBO=1 bash benchmarks/run_parallel_bench.sh
set -uo pipefail          # NOT -e: one backend failing should not lose the rest

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAMP_ROOT="$(cd "${HERE}/.." && pwd)"
RESULTS="${HERE}/results"
mkdir -p "${RESULTS}"

BATCHES="${BATCHES:-1 8 32 128}"
REPS="${REPS:-3}"
OBJECTS="${OBJECTS:-3}"
SEED="${SEED:-0}"
N_SKELETONS="${N_SKELETONS:-}"
SKIP_CUROBO="${SKIP_CUROBO:-0}"

PYROFFI_ENV="${PYROFFI_TAMP_ENV:-pyroffi-tamp}"
CUROBO_ENV="${CUROBO_ENV:-curobo}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

STAMP="$(date +%Y%m%d_%H%M%S)"
SKELETONS="${RESULTS}/skeletons_${OBJECTS}obj_seed${SEED}.npz"
CSV="${RESULTS}/parallel_${STAMP}.csv"
SCRIPT="${HERE}/bench_parallel_validation.py"

# Largest batch requested determines how many skeletons are needed.
if [ -z "${N_SKELETONS}" ]; then
  N_SKELETONS=$(echo ${BATCHES} | tr ' ' '\n' | sort -n | tail -1)
fi

run_backend() {
  local env="$1" backend="$2" extra="${3:-}"
  echo "==> ${backend} (env: ${env})"
  CUDA_VISIBLE_DEVICES="${GPU}" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH="${TAMP_ROOT}" \
    conda run --no-capture-output -n "${env}" python "${SCRIPT}" \
      --backend "${backend}" \
      --skeletons "${SKELETONS}" \
      --batch-sizes ${BATCHES} \
      --reps "${REPS}" \
      --csv "${CSV}" ${extra} \
    2>&1 | grep -vE "WARNING|INFO|Unable to resolve|Can't find|^b3|not found \[" \
    || echo "    ${backend}: FAILED (continuing)"
}

# --- 1. Generate the shared skeletons, once ------------------------------- #
if [ ! -f "${SKELETONS}" ]; then
  echo "==> generating ${N_SKELETONS} skeletons (${OBJECTS} objects, seed ${SEED})"
  CUDA_VISIBLE_DEVICES="${GPU}" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH="${TAMP_ROOT}" \
    conda run --no-capture-output -n "${PYROFFI_ENV}" python "${SCRIPT}" \
      --backend pyroffi --skeletons "${SKELETONS}" \
      --generate "${N_SKELETONS}" --objects "${OBJECTS}" --seed "${SEED}" \
    2>&1 | grep -vE "WARNING|INFO|Unable to resolve|Can't find"
else
  echo "==> reusing ${SKELETONS}"
fi

# --- 2. Each backend in its own environment -------------------------------- #
run_backend "${PYROFFI_ENV}" pyroffi "--torque"
run_backend "${PYROFFI_ENV}" pyroffi-serial
run_backend "${PYROFFI_ENV}" pybullet
if [ "${SKIP_CUROBO}" != "1" ]; then
  run_backend "${CUROBO_ENV}" curobo
fi

# --- 3. Aggregate ---------------------------------------------------------- #
echo
echo "==> aggregate (median over ${REPS} reps)"
python - "${CSV}" <<'PY'
import csv, sys
from collections import defaultdict
from statistics import median

rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    print("no rows"); raise SystemExit

by = defaultdict(list)
for r in rows:
    by[(r["backend"], int(r["batch"]))].append(r)

backends = sorted({b for b, _ in by})
batches = sorted({n for _, n in by})

print()
print("plans/sec (median), and speedup vs the same backend at batch 1")
head = f"| {'backend':<16} |" + "".join(f" B={n:<9} |" for n in batches)
print(head); print("|" + "-" * (len(head) - 2) + "|")
for b in backends:
    base = None
    cells = ""
    for n in batches:
        rs = by.get((b, n))
        if not rs:
            cells += f" {'-':<11} |"; continue
        tp = median(float(r["plans_per_s"]) for r in rs)
        base = base or tp
        cells += f" {tp:8.0f} ({tp/base:.1f}x)|"
    print(f"| {b:<16} |{cells}")

print()
print("validity agreement across backends (should be identical)")
for n in batches:
    verdicts = {b: {int(r["n_valid"]) for r in by.get((b, n), [])} for b in backends}
    vals = {b: (v.pop() if len(v) == 1 else v) for b, v in verdicts.items() if v}
    agree = len(set(map(str, vals.values()))) <= 1
    print(f"  B={n:<5} " + "  ".join(f"{b}={v}" for b, v in vals.items())
          + ("   OK" if agree else "   <-- DISAGREE"))
print(f"\nCSV: {sys.argv[1]}")
PY
