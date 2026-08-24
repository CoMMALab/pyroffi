#!/bin/bash
# Add all 6 baseline methods (implicit, unrolled, fd, cmaes, kkt, cioc) to
# fig3_recovery, fig3b_recovery_highnoise, and fig5_regime (the noise sweep
# figure made this session), per explicit request. Cost check (single-trial
# probes, this session): fig3-style call ~170s, fig5 per-trial ~175s x 25
# trials (5 sigma x 5 seeds) ~= 73min. All live-computed, no separate data
# collection stage.
set -u
cd /home/sadmin/Work/pyroffi
source activate pyroffi

OUT_LOG=/home/sadmin/Work/pyroffi/scratch/logs/all_baselines.log
exec > "$OUT_LOG" 2>&1

echo "[$(date -Is)] all_baselines start"
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 \
    python -m ioc.plots --only recovery
echo "[$(date -Is)] recovery (fig3) done"
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 \
    python -m ioc.plots --only recovery_highnoise
echo "[$(date -Is)] recovery_highnoise (fig3b) done"
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 \
    python -m ioc.plots --only regime
echo "[$(date -Is)] regime (fig5) done"
echo "[$(date -Is)] all_baselines fully done"
