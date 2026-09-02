#!/usr/bin/env bash
# Retry the 2D TAMP study AFTER the path-A recording finishes.  Chained rather
# than independently polling: two pollers would race for the same freed GPU.
set -u
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate pyroffi
cd /home/sadmin/Work/pyroffi
sleep 300                                   # let the path-A queue claim its GPU first
while pgrep -f "record_pathA_behavior" >/dev/null; do sleep 60; done
pick() { for i in 0 1 2 3; do
  a=$(nvidia-smi -i "$i" --query-compute-apps=pid --format=csv,noheader | wc -l)
  m=$(nvidia-smi -i "$i" --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$a" -eq 0 ] && [ "$m" -lt 200 ] && { echo "$i"; return; }; done; }
g=""; while [ -z "$g" ]; do g=$(pick); [ -z "$g" ] && sleep 120; done
echo "=== [$(date +%H:%M:%S)] study5 retry -> GPU $g" >&2
CUDA_VISIBLE_DEVICES="$g" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -u -m iosp.study5_tamp2d_spasm --n-ctx 6 --n-steps 40 --n-starts 4 \
    --record scratch/viz/tamp2d_fit.npz
