#!/bin/bash
# Regenerate fig5_regime's underlying data (bench2d_regime_bw*_R*.json) under
# the current stock-engine, open-loop code. Supersedes PID 1030542, which was
# killed after running >27h: it started before the solver-swap/open-loop
# changes (process env is fixed at start, Python doesn't hot-reload), so it
# was running the OLD internal Gauss-Newton solver at collect.py's original
# n_iter=800/budget=32000 -- and would have overwritten the current (already
# stale, Aug 14) data file with output from an architecture the paper no
# longer uses, without ever being close to finishing.
#
# Tractability retune (measured on GPU2, single-seed probes, this session):
#   n_iter=800 -> 200: matches the cut already applied to bench2d_scale/segments.
#   budget=32000 -> 2000: at n_iter=200, wall time is ~linear in budget
#   (~0.06s/budget-unit/seed, restarts don't add extra cost -- `n_restarts`
#   divides available outer steps, budget currency stays ~constant per
#   the module's own accounting). budget=32000 at n_iter=200 was still
#   extrapolated at >10h across all 4 (bw,R) configs x 5 seeds; budget=2000
#   was chosen as the largest budget that keeps the full stage under an hour.
set -u
cd /home/sadmin/Work/pyroffi
source activate pyroffi

OUT_LOG=/home/sadmin/Work/pyroffi/scratch/logs/regime_v2.log
exec > "$OUT_LOG" 2>&1

BUDGET=2000
FAIL=0
for bw in 0.45 0.90; do
    for R in 1 4; do
        OUT="/home/sadmin/Work/pyroffi/ioc/data/bench2d/bench2d_regime_bw${bw}_R${R}.json"
        echo "[$(date -Is)] === regime bw=$bw R=$R budget=$BUDGET ==="
        CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 \
            python -u -m ioc.bench2d.run --benchmark field --k-bumps 6 --n-contexts 8 \
                --n-seeds 5 --n-timesteps 30 --n-iter 200 --budget "$BUDGET" \
                --bump-width "$bw" --n-restarts "$R" --out "$OUT"
        rc=$?
        if [ $rc -ne 0 ]; then
            echo "[$(date -Is)] bw=$bw R=$R FAILED (exit $rc)"
            FAIL=1
        else
            echo "[$(date -Is)] bw=$bw R=$R wrote $OUT"
        fi
    done
done

if [ $FAIL -eq 0 ]; then
    echo "[$(date -Is)] all four regime configs collected -- rebuilding fig5_regime"
    CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 \
        python -m ioc.plots --only regime
else
    echo "[$(date -Is)] at least one config failed; not rebuilding fig5_regime"
fi

echo "[$(date -Is)] run_regime_v2.sh done, FAIL=$FAIL"
