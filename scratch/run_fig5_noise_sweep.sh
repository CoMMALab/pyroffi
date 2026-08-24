#!/bin/bash
# fig5_regime redesign: distribution of recovery error (theta_l1, regret) vs
# demonstration noise sigma, on the same identifiability-friendly multimodal
# field fig_recovery/fig_recovery_highnoise use (hand-placed bump layout,
# topo_restarts=True structured multistart -- the i.i.d.-jitter default was
# measured stuck in outer-loss local minima on this field regardless of
# budget). Retires the old bump-width x restart-count matrix, which was
# artifact-prone (budget-starved at high n_restarts) and didn't isolate the
# noise effect cleanly. Runs live inside ioc.plots, no separate collect stage.
set -u
cd /home/sadmin/Work/pyroffi
source activate pyroffi

OUT_LOG=/home/sadmin/Work/pyroffi/scratch/logs/fig5_noise_sweep.log
exec > "$OUT_LOG" 2>&1

echo "[$(date -Is)] fig5 noise sweep start"
CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 \
    python -m ioc.plots --only regime
echo "[$(date -Is)] fig5 noise sweep done rc=$?"
