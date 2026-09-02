#!/bin/bash
set -euo pipefail

# Overnight data collection: distribute experiments across GPUs 0, 2, 3.
# GPU 1 is occupied (100% util as of launch).
#
# GPU 0: multistart_robustness (10 runs, ~20h — the longest job)
# GPU 2: tamp2d_sanity + stage2_ablation (~12h combined)
# GPU 3: tetris + tower + identifiability (~8h combined)
#
# Usage: bash iosp/scripts/launch_overnight.sh

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

LOG_DIR=iosp/data/logs
mkdir -p "$LOG_DIR" iosp/data/results/{multistart,tamp2d,ablation,tetris,tower,identifiability} iosp/figures

echo "============================================="
echo "IOSP overnight launch — $(date)"
echo "  GPU 0: multistart_robustness"
echo "  GPU 2: tamp2d_sanity + stage2_ablation"
echo "  GPU 3: tetris + tower + identifiability"
echo "============================================="

# --- GPU 0: multistart (the big one) ---
(
    python -m iosp.run_experiment iosp/experiments/configs/multistart_robustness.yaml --gpu 0 \
        2>&1 | tee "$LOG_DIR/multistart_robustness.log"
) &
PID_GPU0=$!
echo "GPU 0 launched (PID $PID_GPU0): multistart_robustness"

# --- GPU 2: tamp2d then ablation (sequential — both need compile cache) ---
(
    python -m iosp.run_experiment iosp/experiments/configs/tamp2d_sanity.yaml --gpu 2 \
        2>&1 | tee "$LOG_DIR/tamp2d_sanity.log"
    python -m iosp.run_experiment iosp/experiments/configs/stage2_ablation.yaml --gpu 2 \
        2>&1 | tee "$LOG_DIR/stage2_ablation.log"
) &
PID_GPU2=$!
echo "GPU 2 launched (PID $PID_GPU2): tamp2d + ablation"

# --- GPU 3: tetris, tower, identifiability (sequential) ---
(
    python -m iosp.run_experiment iosp/experiments/configs/tetris_iosp.yaml --gpu 3 \
        2>&1 | tee "$LOG_DIR/tetris_iosp.log"
    python -m iosp.run_experiment iosp/experiments/configs/tower_iosp.yaml --gpu 3 \
        2>&1 | tee "$LOG_DIR/tower_iosp.log"
    python -m iosp.run_experiment iosp/experiments/configs/identifiability_spectrum.yaml --gpu 3 \
        2>&1 | tee "$LOG_DIR/identifiability_spectrum.log"
) &
PID_GPU3=$!
echo "GPU 3 launched (PID $PID_GPU3): tetris + tower + identifiability"

echo ""
echo "All 3 GPU jobs launched. Logs: $LOG_DIR/"
echo "Monitor: tail -f $LOG_DIR/*.log"
echo ""
echo "Waiting for all jobs..."

FAILURES=0
for pid in $PID_GPU0 $PID_GPU2 $PID_GPU3; do
    if ! wait "$pid"; then
        FAILURES=$((FAILURES + 1))
        echo "WARNING: PID $pid exited with error"
    fi
done

echo ""
echo "--- All jobs complete. Rendering figures ---"
python -m iosp.make_figures
echo ""
if [ "$FAILURES" -gt 0 ]; then
    echo "WARNING: $FAILURES GPU job(s) had failures. Check logs."
else
    echo "SUCCESS: all experiments complete, figures rendered."
fi
echo "Done at $(date)"
