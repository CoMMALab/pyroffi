#!/bin/bash
set -euo pipefail

# IOSP experiment suite — distributes configs across free GPUs.
#
# Usage:
#   bash iosp/scripts/run_all.sh                    # auto-select GPUs
#   bash iosp/scripts/run_all.sh --gpu 2             # pin to GPU 2
#   bash iosp/scripts/run_all.sh --parallel           # distribute across free GPUs
#   bash iosp/scripts/run_all.sh --dry-run            # print without executing
#   bash iosp/scripts/run_all.sh --figures-only        # render from existing data

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IOSP_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$IOSP_DIR")"
CONFIGS_DIR="$IOSP_DIR/experiments/configs"
LOG_DIR="$IOSP_DIR/data/logs"

GPU_PIN=""
PARALLEL=false
DRY_RUN=""
FIGURES_ONLY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU_PIN="$2"; shift 2 ;;
        --parallel) PARALLEL=true; shift ;;
        --dry-run) DRY_RUN="--dry-run"; shift ;;
        --figures-only) FIGURES_ONLY="--figures-only"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"

cd "$REPO_DIR"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

get_free_gpus() {
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
        | awk -F',' '$2 < 100 {print $1}' | tr -d ' '
}

CONFIGS=("$CONFIGS_DIR"/*.yaml)
NUM_CONFIGS=${#CONFIGS[@]}

if [ "$NUM_CONFIGS" -eq 0 ]; then
    echo "No configs found in $CONFIGS_DIR"
    exit 1
fi

echo "============================================="
echo "IOSP Experiment Suite"
echo "  configs: $NUM_CONFIGS"
echo "  log dir: $LOG_DIR"
echo "============================================="

if [ "$PARALLEL" = true ] && [ -z "$FIGURES_ONLY" ]; then
    FREE_GPUS=($(get_free_gpus))
    NUM_GPUS=${#FREE_GPUS[@]}
    if [ "$NUM_GPUS" -eq 0 ]; then
        echo "ERROR: no free GPUs"
        exit 1
    fi
    echo "Free GPUs: ${FREE_GPUS[*]}"
    echo "Distributing $NUM_CONFIGS configs across $NUM_GPUS GPUs"

    PIDS=()
    for i in "${!CONFIGS[@]}"; do
        cfg="${CONFIGS[$i]}"
        gpu_idx=$((i % NUM_GPUS))
        gpu="${FREE_GPUS[$gpu_idx]}"
        name="$(basename "$cfg" .yaml)"
        log="$LOG_DIR/${name}.log"

        echo "  $name -> GPU $gpu (log: $log)"
        python -m iosp.run_experiment "$cfg" --gpu "$gpu" $DRY_RUN 2>&1 | tee "$log" &
        PIDS+=($!)
    done

    FAILURES=0
    for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
            FAILURES=$((FAILURES + 1))
        fi
    done

    if [ "$FAILURES" -gt 0 ]; then
        echo "WARNING: $FAILURES configs had failures"
    fi
else
    GPU_ARG=""
    if [ -n "$GPU_PIN" ]; then
        GPU_ARG="--gpu $GPU_PIN"
    fi

    for cfg in "${CONFIGS[@]}"; do
        name="$(basename "$cfg" .yaml)"
        log="$LOG_DIR/${name}.log"
        echo ""
        echo "--- $name ---"
        python -m iosp.run_experiment "$cfg" $GPU_ARG $DRY_RUN $FIGURES_ONLY 2>&1 | tee "$log"
    done
fi

echo ""
echo "--- Rendering all figures ---"
python -m iosp.make_figures
echo ""
echo "Done."
