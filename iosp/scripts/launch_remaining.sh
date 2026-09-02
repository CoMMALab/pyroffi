#!/bin/bash
set -euo pipefail
cd /home/sadmin/Work/pyroffi

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate pyroffi

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_disable_hlo_passes=fusion"
export PYTHONUNBUFFERED=1

LOG_DIR=iosp/data/logs
mkdir -p "$LOG_DIR"

run() {
    local gpu=$1 config=$2 name=$3
    echo "[$(date +%H:%M:%S)] START $config/$name on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python -m iosp.run_experiment \
        "iosp/experiments/configs/$config" --run "$name" --gpu "$gpu" \
        >> "$LOG_DIR/${config%.yaml}_${name}.log" 2>&1
    local rc=$?
    echo "[$(date +%H:%M:%S)] DONE  $config/$name exit=$rc"
    return $rc
}

# Track background PIDs per GPU
declare -A GPU_PID

wait_gpu() {
    local gpu=$1
    if [[ -n "${GPU_PID[$gpu]:-}" ]]; then
        wait "${GPU_PID[$gpu]}" || true
    fi
}

assign_gpu() {
    # Wait for any GPU to free up, return its index
    while true; do
        for g in 0 1 2 3; do
            if [[ -z "${GPU_PID[$g]:-}" ]] || ! kill -0 "${GPU_PID[$g]}" 2>/dev/null; then
                wait "${GPU_PID[$g]:-}" 2>/dev/null || true
                echo "$g"
                return
            fi
        done
        sleep 5
    done
}

launch() {
    local config=$1 name=$2
    local gpu
    gpu=$(assign_gpu)
    run "$gpu" "$config" "$name" &
    GPU_PID[$gpu]=$!
}

echo "=== Launching remaining experiments $(date) ==="

# Paper-critical: multistart remaining (7 runs)
launch multistart_robustness.yaml joint_seed3
launch multistart_robustness.yaml joint_seed4
# ee seeds shelved
# ee seeds shelved — joint space is the canonical method

# Paper-critical: tamp2d (3 runs)
launch tamp2d_sanity.yaml seed0
launch tamp2d_sanity.yaml seed1
launch tamp2d_sanity.yaml seed2

# Paper-critical: ablation (5 runs)
launch stage2_ablation.yaml seed0
launch stage2_ablation.yaml seed1
launch stage2_ablation.yaml seed2
launch stage2_ablation.yaml seed3
launch stage2_ablation.yaml seed4

# Tetris retry with shape fix (6 runs)
launch tetris_iosp.yaml blocks3_seed0
launch tetris_iosp.yaml blocks3_seed1
launch tetris_iosp.yaml blocks3_seed2
launch tetris_iosp.yaml blocks5_seed0
launch tetris_iosp.yaml blocks5_seed1
launch tetris_iosp.yaml blocks5_seed2

# Tower (6 runs)
launch tower_iosp.yaml blocks3_seed0
launch tower_iosp.yaml blocks3_seed1
launch tower_iosp.yaml blocks3_seed2
launch tower_iosp.yaml blocks5_seed0
launch tower_iosp.yaml blocks5_seed1
launch tower_iosp.yaml blocks5_seed2

# Wait for everything to finish
for g in 0 1 2 3; do
    wait_gpu "$g"
done

echo "=== All experiments finished $(date) ==="

# Count successes and failures
total=0
ok=0
for f in "$LOG_DIR"/*.log; do
    if [[ -f "$f" ]]; then
        total=$((total+1))
        if grep -q "exit: 0" "$f" 2>/dev/null; then
            ok=$((ok+1))
        fi
    fi
done
echo "Results: $ok/$total succeeded"
