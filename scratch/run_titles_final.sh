#!/bin/bash
# Final title-only pass for the two figures that recompute live (no persisted
# data cache): fig3/fig3b (~3 min each) and fig5_noise_field (~75 min, same
# cost as its data-collection run since there's no way to add a title without
# recomputing). Everything else was cheap and already re-rendered directly.
set -u
cd /home/sadmin/Work/pyroffi
source activate pyroffi

OUT_LOG=/home/sadmin/Work/pyroffi/scratch/logs/titles_final.log
exec > "$OUT_LOG" 2>&1

echo "[$(date -Is)] titles_final start"
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 \
    python -m ioc.plots --only recovery
echo "[$(date -Is)] fig3 done"
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 \
    python -m ioc.plots --only recovery_highnoise
echo "[$(date -Is)] fig3b done"
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 \
    python -m ioc.plots --only regime
echo "[$(date -Is)] fig5 done"
echo "[$(date -Is)] titles_final fully done"
