#!/bin/bash
# Cross-domain aggregate table (EXPERIMENT_PLAN.md v2, deliverable 2):
# pick-place, tetris packing and block stacking, run sequentially on ONE GPU.
#
# Sequential by design: two JAX processes on one device OOM, and these
# configs share an XLA compilation cache -- the first tetris/tower run pays a
# ~15 min compile that later seeds at the same shape then hit for free.
#
#   bash iosp/scripts/run_domains.sh <gpu-idx>
set -uo pipefail
GPU="${1:?usage: run_domains.sh <gpu-idx>}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p iosp/data/logs
for cfg in tetris_iosp tower_iosp pickplace_iosp; do
    echo "=== $cfg on GPU $GPU ==="
    python -u -m iosp.run_experiment "iosp/experiments/configs/$cfg.yaml" \
        --gpu "$GPU" 2>&1 | tee "iosp/data/logs/domains_$cfg.log"
done
echo "=== all domains done ==="
