#!/bin/bash
# Run a queue of experiments on a single GPU, sequentially.
# Usage: gpu_queue.sh <gpu_id> <config:run> [<config:run> ...]
set -euo pipefail
cd /home/sadmin/Work/pyroffi

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate pyroffi

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_disable_hlo_passes=fusion"
export PYTHONUNBUFFERED=1

GPU=$1; shift
LOG_DIR=iosp/data/logs
mkdir -p "$LOG_DIR"

for spec in "$@"; do
    config="${spec%%:*}"
    run_name="${spec##*:}"
    logf="$LOG_DIR/${config%.yaml}_${run_name}.log"
    echo "[$(date +%H:%M:%S)] GPU=$GPU START $config/$run_name"
    CUDA_VISIBLE_DEVICES=$GPU python -m iosp.run_experiment \
        "iosp/experiments/configs/$config" --run "$run_name" --gpu "$GPU" \
        > "$logf" 2>&1 || echo "[$(date +%H:%M:%S)] GPU=$GPU FAILED $config/$run_name"
    echo "[$(date +%H:%M:%S)] GPU=$GPU DONE  $config/$run_name"
done
echo "[$(date +%H:%M:%S)] GPU=$GPU queue finished"
