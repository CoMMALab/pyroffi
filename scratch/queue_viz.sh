#!/usr/bin/env bash
# Record + render the fit animation for Path A then Path B, each held until a
# GPU is EXCLUSIVELY idle (no compute apps, < 200 MiB).  Deliberately stricter
# than "some free memory": Path B compiles for ~3545s, and the Path A leg
# already raced a difftori run onto GPU 3 once.  These are visualisations --
# they yield to anyone's real training job rather than share a device with it.
set -u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pyroffi
cd /home/sadmin/Work/pyroffi

pick_idle_gpu() {
  for i in 0 1 2 3; do
    apps=$(nvidia-smi -i "$i" --query-compute-apps=pid --format=csv,noheader | wc -l)
    mem=$(nvidia-smi -i "$i" --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ "$apps" -eq 0 ] && [ "$mem" -lt 200 ]; then echo "$i"; return; fi
  done
}

for path in a b; do
  gpu=""
  while [ -z "$gpu" ]; do
    gpu=$(pick_idle_gpu)
    [ -z "$gpu" ] && sleep 120
  done
  echo "=== [$(date +%H:%M:%S)] path $path -> GPU $gpu (exclusively idle)" >&2
  CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python -u -m iosp.viz_fit_animation both \
      --path "$path" --out "scratch/viz/fit_${path}.npz" \
    || echo "=== path $path FAILED (exit $?)" >&2
done
echo "=== [$(date +%H:%M:%S)] queue done" >&2
