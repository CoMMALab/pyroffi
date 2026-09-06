#!/usr/bin/env bash
# Detached, session-isolated launcher for one or more IOSP YAML configs on a
# single GPU, run sequentially (one stream per GPU, no on-device contention).
#
# Usage: run_config_detached.sh <tag> <gpu> <spec> [<spec> ...]
#   <tag>   short name for log/sentinel files
#   <gpu>   CUDA device index
#   <spec>  "config.yaml=run1,run2,..."  (runs the whole config if "=..." omitted)
#
# Invoked via setsid by the caller, so the job runs in its own session/process
# group; a SIGTERM or quiet death cannot propagate to the Claude session.
# Each named run is a separate invocation so one failure doesn't abort the rest.
set -u

TAG=$1; GPU=$2; shift 2
SPECS=("$@")

LOGDIR=/home/sadmin/Work/pyroffi/iosp/data/logs
LOG="$LOGDIR/regen_${TAG}.log"
DONE="$LOGDIR/regen_${TAG}.done"
PIDFILE="$LOGDIR/regen_${TAG}.pid"
rm -f "$DONE"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pyroffi-tamp
cd /home/sadmin/Work/pyroffi

echo "PID $$" > "$PIDFILE"
{
  echo "=== [$TAG] launch $(date -Is) GPU $GPU (pid $$, sid $(ps -o sid= -p $$)) specs=${SPECS[*]} ==="
  overall=0
  for spec in "${SPECS[@]}"; do
    cfg="${spec%%=*}"
    runs="${spec#*=}"; [ "$runs" = "$spec" ] && runs=""
    if [ -z "$runs" ]; then
      echo "--- [$TAG] $cfg (all runs) $(date -Is) ---"
      CUDA_VISIBLE_DEVICES="$GPU" XLA_PYTHON_CLIENT_PREALLOCATE=false \
        python -u -m iosp.run_experiment "$cfg" --gpu "$GPU"
      rc=$?; [ $rc -ne 0 ] && overall=$rc
    else
      IFS=',' read -ra RN <<< "$runs"
      for r in "${RN[@]}"; do
        echo "--- [$TAG] $cfg :: $r $(date -Is) ---"
        CUDA_VISIBLE_DEVICES="$GPU" XLA_PYTHON_CLIENT_PREALLOCATE=false \
          python -u -m iosp.run_experiment "$cfg" --run "$r" --gpu "$GPU"
        rc=$?
        echo "--- [$TAG] $cfg :: $r exit $rc ---"
        [ $rc -ne 0 ] && overall=$rc
      done
    fi
  done
  echo "=== [$TAG] ALL DONE exit $overall at $(date -Is) ==="
  echo "$overall" > "$DONE"
} >> "$LOG" 2>&1
