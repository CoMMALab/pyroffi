#!/bin/bash
# bench2d_scale sweep for `segments` (per-segment clearance, K=3S), post
# open-loop-engine retirement of the internal Gauss-Newton solver: `implicit`
# now runs on dynamics_trajopt(early_stop=True), `unrolled` on
# dynamics_trajopt(early_stop=False, unroll_tail=...) -- both stock pyroffi.
#
# Budget schedule: v3's flat budget=800 starved fd/cmaes everywhere (fig1
# showed fd absent at every K). A 5-min-capped profile on K=6 found fd/cmaes
# actually cross the target at budget=2000 (fd@448 solves, cmaes@216); a flat
# budget=3000 timed out before one seed finished. Since unrolled cannot
# early-stop, wall time is roughly budget-proportional, so this uses a flat
# 2.5x scale-up of v3's budget across all K (2000) rather than the earlier
# cubic extrapolation (2000/5600/21000/84000, ~3h) -- that guess was informed
# by only one profiled point and risked being intractable again at K=48. If
# fd/cmaes still don't cross the target at K=24/48 under this budget, that's
# itself the reported result (censored, per fig_scaling's >=2/3-seed rule),
# not a failure to fix.
set -u
cd /home/sadmin/Work/pyroffi
source activate pyroffi

OUT_LOG=/home/sadmin/Work/pyroffi/scratch/logs/segments_scale_v4.log
exec > "$OUT_LOG" 2>&1

BUDGET=2000
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

echo "[$(date -Is)] run_segments_scale_v4.sh done, FAIL=$FAIL"
