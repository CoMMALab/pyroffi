#!/bin/bash
# Run all remaining iosp experiments across 4 GPUs in parallel.
# Usage: bash iosp/scripts/run_all_remaining.sh
set -uo pipefail
cd /home/sadmin/Work/pyroffi

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate pyroffi

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_disable_hlo_passes=fusion"
export PYTHONUNBUFFERED=1

LOG_DIR=iosp/data/logs
mkdir -p "$LOG_DIR"

run_queue() {
    local GPU=$1; shift
    for spec in "$@"; do
        local config="${spec%%:*}"
        local run_name="${spec##*:}"
        local logf="$LOG_DIR/${config%.yaml}_${run_name}.log"
        echo "[$(date +%H:%M:%S)] GPU=$GPU START $config/$run_name"
        CUDA_VISIBLE_DEVICES=$GPU python -m iosp.run_experiment \
            "iosp/experiments/configs/$config" --run "$run_name" --gpu "$GPU" \
            > "$logf" 2>&1 || true
        echo "[$(date +%H:%M:%S)] GPU=$GPU DONE  $config/$run_name exit=$?"
    done
    echo "[$(date +%H:%M:%S)] GPU=$GPU ALL DONE"
}

run_queue 0 \
    multistart_robustness.yaml:joint_seed4 \
    stage2_ablation.yaml:seed0 \
    stage2_ablation.yaml:seed1 &

run_queue 1 \
    tamp2d_sanity.yaml:seed0 \
    tamp2d_sanity.yaml:seed1 \
    tamp2d_sanity.yaml:seed2 &

run_queue 2 \
    tetris_iosp.yaml:blocks3_seed0 \
    tetris_iosp.yaml:blocks3_seed1 \
    tetris_iosp.yaml:blocks3_seed2 \
    tetris_iosp.yaml:blocks5_seed0 \
    tetris_iosp.yaml:blocks5_seed1 \
    tetris_iosp.yaml:blocks5_seed2 \
    stage2_ablation.yaml:seed2 &

run_queue 3 \
    tower_iosp.yaml:blocks3_seed0 \
    tower_iosp.yaml:blocks3_seed1 \
    tower_iosp.yaml:blocks3_seed2 \
    tower_iosp.yaml:blocks5_seed0 \
    tower_iosp.yaml:blocks5_seed1 \
    tower_iosp.yaml:blocks5_seed2 \
    stage2_ablation.yaml:seed3 \
    stage2_ablation.yaml:seed4 &

wait
echo "=== ALL GPUS ALL DONE ==="
