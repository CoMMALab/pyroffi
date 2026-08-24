#!/bin/bash
# bench2d_scale sweep for `segments`, budget retune #2: v4's flat budget=2000
# left cmaes censored (0/3 seeds) at K=24/48 and fd censored almost everywhere
# -- fig1_scaling effectively showed only implicit+unrolled at higher K.
#
# Calibration probes this session (K=24, K=48, budget=6000, 1 seed):
#   cmaes stalled just short of the L<1e-2 target (0.0186 @ 19 gens on K=24,
#   0.019 @ 8 gens on K=48) -- a modest bump should clear it.
#   fd's per-step cost is M*(K+1) solves; at K=48 it only completed 2 outer
#   steps in budget=6000 (loss 0.082, nowhere near target). Reaching
#   convergence there would need a budget in the hundreds of thousands --
#   not tractable, and not really a bug to fix: fd's cost scaling with K is
#   the paper's own point about why it doesn't scale. This schedule is tuned
#   for cmaes to reliably cross at every K; fd gets whatever headroom comes
#   along for free but is not expected to fully converge at K=48.
#
# Budget schedule (vs v4's flat 2000), with ~2-3x safety margin over the
# probed crossing point at each K:
#   K=6: 4000   K=12: 6000   K=24: 14000   K=48: 20000
# Estimated wall time (~0.03s/budget-unit/seed, measured on K=24/K=48
# probes): ~66 min total across all four K, 3 seeds each.
set -u
cd /home/sadmin/Work/pyroffi
source activate pyroffi

OUT_LOG=/home/sadmin/Work/pyroffi/scratch/logs/segments_scale_v5.log
exec > "$OUT_LOG" 2>&1

declare -A BUDGETS=( [2]=4000 [4]=6000 [8]=14000 [16]=20000 )
FAIL=0
for S in 2 4 8 16; do
    K=$((3 * S))
    N_CTX=$((3 * S > 8 ? 3 * S : 8))
    BUDGET=${BUDGETS[$S]}
    OUT="/home/sadmin/Work/pyroffi/ioc/data/bench2d/bench2d_seg_K${K}.json"
    echo "[$(date -Is)] === segments K=$K (S=$S, n-contexts=$N_CTX, budget=$BUDGET) ==="
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 \
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
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 \
        python -m ioc.plots --only scaling
else
    echo "[$(date -Is)] at least one K value failed; not rebuilding fig1_scaling"
fi

echo "[$(date -Is)] run_segments_scale_v5.sh done, FAIL=$FAIL"
