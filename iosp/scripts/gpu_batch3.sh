#!/bin/bash
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
    echo "[$(date +%H:%M:%S)] GPU=$GPU START $config/$run_name" | tee -a "$LOG_DIR/gpu${GPU}_batch3.log"
    CUDA_VISIBLE_DEVICES=$GPU python -m iosp.run_experiment \
        "iosp/experiments/configs/$config" --run "$run_name" --gpu "$GPU" \
        > "$logf" 2>&1 || true
    echo "[$(date +%H:%M:%S)] GPU=$GPU DONE  $config/$run_name exit=$?" | tee -a "$LOG_DIR/gpu${GPU}_batch3.log"
done
echo "[$(date +%H:%M:%S)] GPU=$GPU ALL DONE" | tee -a "$LOG_DIR/gpu${GPU}_batch3.log"
