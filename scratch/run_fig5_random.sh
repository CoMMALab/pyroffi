#!/bin/bash
# Rerun fig5_noise_field with "random" baseline added. Already uses
# n_restarts=7 topo_restarts=True (unchanged), so this run also satisfies
# the "recollect with the same restart parameters" ask for this figure --
# no separate restart-only pass needed, unlike fig4 which never had restarts.
set -u
cd /home/sadmin/Work/pyroffi
source activate pyroffi

OUT_LOG=/home/sadmin/Work/pyroffi/scratch/logs/fig5_random.log
exec > "$OUT_LOG" 2>&1

echo "[$(date -Is)] fig5_noise_field (with random) start"
CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 \
    python -m ioc.plots --only regime
echo "[$(date -Is)] fig5_noise_field done rc=$?"
