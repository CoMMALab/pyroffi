#!/bin/bash
# Regenerate fig4_noise's underlying data (e1_sigma*.json) under the current
# stock-engine code. Superseded run: old data was Aug 13, predates the
# solver-swap-to-dynamics_trajopt change entirely.
set -u
cd /home/sadmin/Work/pyroffi
source activate pyroffi

OUT_LOG=/home/sadmin/Work/pyroffi/scratch/logs/e1_noise_v2.log
exec > "$OUT_LOG" 2>&1

echo "[$(date -Is)] e1_noise_v2 start"
python -m ioc.collect --stages e1_noise --gpu 1
rc=$?
echo "[$(date -Is)] e1_noise_v2 done rc=$rc"

if [ $rc -eq 0 ]; then
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 \
        python -m ioc.plots --only noise
fi
echo "[$(date -Is)] e1_noise_v2 fully done"
