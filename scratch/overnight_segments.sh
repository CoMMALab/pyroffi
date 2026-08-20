#!/bin/bash
# Conditional overnight rerun of ioc bench2d_scale (segments K=5,9,17,33).
# Waits on the running probe (PID 933040, S=4/K=9 config) to resolve, then
# only proceeds with the full sweep if the DEMO_N_ITER=2500 bump actually
# cleared screening. Uses GPU 2 (free at launch time; GPU1 is the main
# collect job on bench2d_regime, GPU0 has an unrelated external process).

set -u
cd /home/sadmin/Work/pyroffi
source activate pyroffi

PROBE_LOG=/home/sadmin/Work/pyroffi/scratch/logs/probe_seg_K5.log
OUT_LOG=/home/sadmin/Work/pyroffi/scratch/logs/overnight_segments.log
PROBE_PID=933040

exec > "$OUT_LOG" 2>&1

echo "[$(date -Is)] waiting on probe PID $PROBE_PID to finish..."
while kill -0 "$PROBE_PID" 2>/dev/null; do
    sleep 30
done
echo "[$(date -Is)] probe process ended"

if grep -q "not local optima" "$PROBE_LOG"; then
    echo "[$(date -Is)] PROBE FAILED: screening still rejects demos at DEMO_N_ITER=2500."
    grep "not local optima" "$PROBE_LOG"
    echo "[$(date -Is)] Not proceeding with the full segments rerun -- treat as evidence"
    echo "the benchmark needs re-evaluation, not another iteration bump."
    exit 1
fi

if ! grep -q "wrote /tmp/probe_seg_K5.json" "$PROBE_LOG"; then
    echo "[$(date -Is)] PROBE INCONCLUSIVE: process ended without a clear pass or fail signal."
    echo "--- tail of probe log ---"
    tail -n 40 "$PROBE_LOG"
    exit 1
fi

echo "[$(date -Is)] PROBE PASSED: screening cleared at DEMO_N_ITER=2500. Proceeding with full rerun."

FAIL=0
for S in 2 4 8 16; do
    K=$((2 * S + 1))
    N_CTX=$((2 * S + 1 > 8 ? 2 * S + 1 : 8))
    BUDGET=$((2000 * (2 * S + 1)))
    OUT="/home/sadmin/Work/pyroffi/ioc/data/bench2d/bench2d_seg_K${K}.json"
    echo "[$(date -Is)] === segments K=$K (S=$S, n-contexts=$N_CTX, budget=$BUDGET) ==="
    CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 \
        python -u -m ioc.bench2d.run --benchmark segments --k-segments "$S" \
            --n-contexts "$N_CTX" --n-seeds 3 --n-timesteps 30 --n-iter 800 \
            --budget "$BUDGET" --out "$OUT"
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[$(date -Is)] K=$K FAILED (exit $rc)"
        FAIL=1
    else
        echo "[$(date -Is)] K=$K wrote $OUT"
    fi
done

if [ $FAIL -eq 0 ]; then
    echo "[$(date -Is)] all four segments K values collected -- rebuilding fig1_scaling"
    CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 \
        python -m ioc.plots --only scaling
else
    echo "[$(date -Is)] at least one K value failed; not rebuilding fig1_scaling"
fi

echo "[$(date -Is)] overnight_segments.sh done, FAIL=$FAIL"
