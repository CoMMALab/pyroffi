#!/bin/bash
# Recollect e1_noise (fig4_noise_robot's data) with n_restarts=7 i.i.d. jitter
# multistart, matching fig5_noise_field's restart count so both noise-sweep
# figures are controlled for inner-problem multimodality with the same
# restart budget. Single-seed probe (this session): ~270s/seed at
# n_restarts=7 vs the earlier ~150s/seed with no restarts. Estimated
# 4 sigma x 5 seeds x 270s =~ 90 min.
set -u
cd /home/sadmin/Work/pyroffi
source activate pyroffi

OUT_LOG=/home/sadmin/Work/pyroffi/scratch/logs/e1_noise_restarts.log
exec > "$OUT_LOG" 2>&1

echo "[$(date -Is)] e1_noise (n_restarts=7) start"
python -m ioc.collect --stages e1_noise --gpu 1
rc=$?
echo "[$(date -Is)] e1_noise done rc=$rc"

if [ $rc -eq 0 ]; then
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 \
        python -m ioc.plots --only noise
fi
echo "[$(date -Is)] e1_noise_restarts fully done"
