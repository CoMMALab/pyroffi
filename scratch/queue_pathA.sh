#!/usr/bin/env bash
# Wait for the 2D TAMP run to finish, then record path-A reconstruction data on
# the GPU it frees.  Path A's build alone is ~50 min, so this waits for an
# EXCLUSIVELY idle device rather than sharing with anyone's training job.
set -u
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate pyroffi
cd /home/sadmin/Work/pyroffi
while pgrep -f "study5_tamp2d_spasm" >/dev/null; do sleep 60; done
pick() { for i in 0 1 2 3; do
  a=$(nvidia-smi -i "$i" --query-compute-apps=pid --format=csv,noheader | wc -l)
  m=$(nvidia-smi -i "$i" --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$a" -eq 0 ] && [ "$m" -lt 200 ] && { echo "$i"; return; }; done; }
g=""; while [ -z "$g" ]; do g=$(pick); [ -z "$g" ] && sleep 120; done
echo "=== [$(date +%H:%M:%S)] path-A record -> GPU $g" >&2
CUDA_VISIBLE_DEVICES="$g" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -u -m iosp.record_pathA_behavior --steps 40 \
    --out scratch/viz/pathA_behavior.npz
