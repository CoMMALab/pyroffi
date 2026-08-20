#!/bin/bash
# bench2d_scale sweep for the fixed `segments` benchmark (per-segment clearance,
# K=3S) now that the forward solver is pyroffi's stock dynamics_trajopt
# (early-stopping L-BFGS, grad_tol=1e-9) instead of the retired internal
# Gauss-Newton loop. DEMO_N_ITER["segments"]=6000 confirmed still sufficient
# post-swap (subagent verification). Budget kept modest (not the original
# 2000*K) since early-stopping now does the real work of not wasting solves.
set -u
cd /home/sadmin/Work/pyroffi
source activate pyroffi

OUT_LOG=/home/sadmin/Work/pyroffi/scratch/logs/segments_scale_v3.log
exec > "$OUT_LOG" 2>&1

BUDGET=800
FAIL=0
for S in 2 4 8 16; do
    K=$((3 * S))
    N_CTX=$((3 * S > 8 ? 3 * S : 8))
    OUT="/home/sadmin/Work/pyroffi/ioc/data/bench2d/bench2d_seg_K${K}.json"
    echo "[$(date -Is)] === segments K=$K (S=$S, n-contexts=$N_CTX, budget=$BUDGET) ==="
    CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 \
        python -u -m ioc.bench2d.run --benchmark segments --k-segments "$S" \
            --n-contexts "$N_CTX" --n-seeds 3 --n-timesteps 30 --n-iter 200 \
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

echo "[$(date -Is)] run_segments_scale_v3.sh done, FAIL=$FAIL"
