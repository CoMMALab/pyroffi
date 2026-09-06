#!/usr/bin/env bash
# Safety-bounded IOSP figure-data pipeline.
#
# Each (config, run) is executed as its OWN transient `systemd --user` unit with:
#   - MemoryMax + MemorySwapMax : an OOM is a cgroup-local kill of just that run
#   - RuntimeMaxSec             : a compile-hang / blow-up is killed on a timeout
# The unit is launched with `systemd-run --user --wait`, so this driver blocks
# until the run finishes and learns its exit status.  ANY failure (OOM, timeout,
# nonzero exit, or missing output) is recorded in $FAILLOG and the pipeline
# CONTINUES to the next run -- blow-ups become recorded failures, never a crash
# that takes down the box or sibling work.  See [[iosp-e10-host-ram-oom]].
#
# The driver itself is meant to run as a transient unit too; the per-run units it
# spawns are independent siblings at the user-manager level, so their MemoryMax is
# independent of the driver's.
set -u

LOGDIR=/home/sadmin/Work/pyroffi/iosp/data/logs
DRLOG="$LOGDIR/pipeline.log"
DONE="$LOGDIR/pipeline.done"
FAILLOG="$LOGDIR/pipeline.failures"
OKLOG="$LOGDIR/pipeline.ok"
rm -f "$DONE"; : > "$FAILLOG"; : > "$OKLOG"
cd /home/sadmin/Work/pyroffi

MEM=${MEM:-100G}
SWAP=${SWAP:-6G}
TIMEOUT=${TIMEOUT:-2400}          # 40 min/run: enough to compile+fit, kills hangs
GPU=${GPU:-0}
# Re-enable XLA fusion (the breaking commit disabled it).  Empty string lets the
# per-config XLA_FLAGS stand; "FUSION_ON" strips the disable-fusion flag.
FUSION=${FUSION:-FUSION_ON}

run_one () {  # $1=tag  $2=config  $3=run  $4=out_path
  local tag=$1 cfg=$2 run=$3 out=$4
  local unit="iosp_r_${tag}"
  echo ">>> [$(date -Is)] $tag  ($cfg :: $run)  mem=$MEM timeout=${TIMEOUT}s fusion=$FUSION"
  systemctl --user reset-failed "$unit" 2>/dev/null

  local xla=""
  [ "$FUSION" = "FUSION_ON" ] && xla="--setenv=XLA_FLAGS= "

  # --wait blocks until the unit exits; --pipe streams its output into our log.
  systemd-run --user --wait --collect --pipe --unit="$unit" \
    -p MemoryMax="$MEM" -p MemorySwapMax="$SWAP" \
    -p RuntimeMaxSec="$TIMEOUT" \
    --setenv=CUDA_VISIBLE_DEVICES="$GPU" \
    --setenv=XLA_PYTHON_CLIENT_PREALLOCATE=false \
    $xla \
    /home/sadmin/miniconda3/envs/pyroffi-tamp/bin/python -u \
      -m iosp.run_experiment "$cfg" --run "$run" --gpu "$GPU"
  local rc=$?

  if [ $rc -eq 0 ] && [ -f "$out" ]; then
    echo "OK   $tag  ($run)  rc=$rc" | tee -a "$OKLOG"
  else
    local reason="rc=$rc"
    [ ! -f "$out" ] && reason="$reason no-output"
    # RuntimeMaxSec kill shows as timeout in the unit result
    systemctl --user show "$unit" -p Result --value 2>/dev/null | grep -q timeout && reason="$reason TIMEOUT"
    systemctl --user show "$unit" -p Result --value 2>/dev/null | grep -q oom && reason="$reason OOM"
    echo "FAIL $tag  ($run)  $reason" | tee -a "$FAILLOG"
  fi
  systemctl --user reset-failed "$unit" 2>/dev/null
}

R=iosp/data/results
{
  echo "=== bounded pipeline launch $(date -Is) (pid $$) MEM=$MEM TIMEOUT=$TIMEOUT FUSION=$FUSION ==="

  for s in 0 1 2 3 4; do
    run_one "pp$s" iosp/experiments/configs/pickplace_iosp.yaml "pickplace_seed$s" "$R/pickplace/pickplace_seed$s.json"
  done
  for b in 3 5; do for s in 0 1 2; do
    run_one "tet_b${b}s${s}" iosp/experiments/configs/tetris_iosp.yaml "blocks${b}_seed$s" "$R/tetris/blocks${b}_seed$s.npz"
  done; done
  for b in 3 5; do for s in 0 1 2; do
    run_one "tow_b${b}s${s}" iosp/experiments/configs/tower_iosp.yaml "blocks${b}_seed$s" "$R/tower/blocks${b}_seed$s.npz"
  done; done
  for s in 0 1 2 3; do
    run_one "ms$s" iosp/experiments/configs/multistart_robustness.yaml "joint_seed$s" "$R/multistart/joint_seed$s.npz"
  done
  run_one "eigen" iosp/experiments/configs/identifiability_spectrum.yaml "eigen_projection" "$R/identifiability/eigen_projection.npz"

  echo "=== pipeline COMPLETE $(date -Is) ==="
  echo "  OK:     $(wc -l < "$OKLOG") runs"
  echo "  FAILED: $(wc -l < "$FAILLOG") runs"
  cat "$FAILLOG"
  echo "0" > "$DONE"
} >> "$DRLOG" 2>&1
