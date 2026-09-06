#!/usr/bin/env bash
# Detached, session-isolated launcher for the E10 method comparison.
#
# Meant to be run as a transient `systemd --user` service so it lives in its own
# cgroup: a per-unit MemoryMax then makes an out-of-memory condition a
# cgroup-local kill of THIS job, instead of a global OOM that takes down the
# whole box (and the Claude session) — which is what kept happening when several
# ~50 GB JAX jobs ran at once on the 113 GB host.
set -u

LOG=/home/sadmin/Work/pyroffi/iosp/data/logs/e10_full_run.log
DONE=/home/sadmin/Work/pyroffi/iosp/data/logs/e10_full_run.done
PIDFILE=/home/sadmin/Work/pyroffi/iosp/data/logs/e10_full_run.pid
GPU=${CUDA_VISIBLE_DEVICES:-0}

rm -f "$DONE"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pyroffi-tamp
cd /home/sadmin/Work/pyroffi

export CUDA_VISIBLE_DEVICES="$GPU"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "PID $$" > "$PIDFILE"
{
  echo "=== launch $(date -Is) on GPU $GPU (pid $$, sid $(ps -o sid= -p $$)) ==="
  python -u -m iosp.experiments.e10_method_comparison \
      --methods implicit,fd,cmaes,unrolled
  rc=$?
  echo "=== exit code $rc at $(date -Is) ==="
  echo "$rc" > "$DONE"
} >> "$LOG" 2>&1
