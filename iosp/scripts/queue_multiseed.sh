#!/usr/bin/env bash
# Multi-seed sweep: every number quoted so far is seed 0 alone.
#
# What `seed` actually varies (checked, not assumed):
#   * `PickPlaceProblem.calibrate_segment(..., PRNGKey(seed))` -- the per-feature
#     scale constants, i.e. the conditioning of the outer parameterization
#   * `branch_refs(..., PRNGKey(seed + 11))` -- which IK branches the candidates
#     sit on (reference 0 is always q_start, so branch 0 is common to all seeds)
#   * `make_candidates(..., seed)` -- the cost-parameter starts
# It does NOT vary theta_star, the scenes, or the demo.  So this measures the
# variance of the METHOD (calibration + basin coverage + initialization), which
# is the thing the robustness claim is about.  Demo/scene variance is a separate
# axis and is not what this sweep answers.
#
# Stage A (headline): the joint-vs-EE comparison, PAIRED -- both spaces at the
# same seed share calibration and starts, so the per-seed difference is the
# statistic, not two independent means.  Spectrum is skipped (--no-spectrum):
# it is a separate claim, already measured once per space, and costs ~700s.
#
# Stage B: the 9-candidate multistart under the 2x-harder held-out scene.
#
# Jobs are pinned to whichever GPUs are free at launch and run one at a time per
# GPU.  GPU 1 is deliberately not probed here if busy -- these boxes are shared.
set -u
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate pyroffi
cd /home/sadmin/Work/pyroffi
export PYTHONPATH=/home/sadmin/Work/pyroffi XLA_PYTHON_CLIENT_PREALLOCATE=false
LOGDIR=${LOGDIR:-scratch/logs/multiseed_iosp}
OUTDIR=${OUTDIR:-scratch/viz/multiseed}
mkdir -p "$LOGDIR" "$OUTDIR"

SEEDS_A="1 2 3 4"     # seed 0 already recorded for both spaces
SEEDS_B="1 2"         # seed 0 already recorded for the multistart

free_gpus() {
  for i in 0 1 2 3; do
    a=$(nvidia-smi -i "$i" --query-compute-apps=pid --format=csv,noheader | wc -l)
    m=$(nvidia-smi -i "$i" --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$a" -eq 0 ] && [ "$m" -lt 500 ] && echo "$i"
  done
}

JOBS=()
for s in $SEEDS_A; do
  for sp in joint ee; do
    JOBS+=("A|$s|$sp")
  done
done
for s in $SEEDS_B; do JOBS+=("B|$s|joint"); done

run_job() {
  local gpu=$1 spec=$2
  IFS='|' read -r stage seed space <<< "$spec"
  if [ "$stage" = "A" ]; then
    local log=""$LOGDIR"/A_${space}_seed${seed}.log"
    CUDA_VISIBLE_DEVICES="$gpu" python -u -m iosp.experiments.e7_loss_space \
      --space "$space" --seed "$seed" --no-spectrum \
      --out ""$OUTDIR"/A_${space}_seed${seed}.npz" > "$log" 2>&1
  else
    local log=""$LOGDIR"/B_seed${seed}.log"
    CUDA_VISIBLE_DEVICES="$gpu" python -u -m iosp.record.multistart \
      --space joint --seed "$seed" --n-branches 3 --n-starts 3 --steps 40 \
      --scene-b-scale 2.0 --no-render \
      --out ""$OUTDIR"/B_seed${seed}.npz" > "$log" 2>&1
  fi
  echo "[$(date +%H:%M:%S)] done gpu=$gpu $spec (exit $?)" >&2
}

# One worker per free GPU, each pulling from the shared job list by index.
mapfile -t GPUS < <(free_gpus)
[ "${#GPUS[@]}" -eq 0 ] && { echo "no free GPU; not launching" >&2; exit 1; }
echo "[$(date +%H:%M:%S)] ${#JOBS[@]} jobs over GPUs: ${GPUS[*]}" >&2

n=${#GPUS[@]}
for w in "${!GPUS[@]}"; do
  (
    i=$w
    while [ "$i" -lt "${#JOBS[@]}" ]; do
      echo "[$(date +%H:%M:%S)] gpu=${GPUS[$w]} start ${JOBS[$i]}" >&2
      run_job "${GPUS[$w]}" "${JOBS[$i]}"
      i=$(( i + n ))
    done
  ) &
done
wait
echo "[$(date +%H:%M:%S)] ALL MULTISEED JOBS COMPLETE" >&2
