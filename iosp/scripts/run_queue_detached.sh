#!/usr/bin/env bash
# SERIAL queue for the memory-heavy IOSP jobs.  Measured host-RAM peaks are
# ~50-60 GB per job on a 113 GB box, so two do NOT safely co-run (tetris was
# cgroup-OOM-killed at a 54 GB cap).  This queue therefore runs the remaining
# jobs strictly one at a time; the enclosing systemd unit's MemoryMax bounds
# each so a runaway is a cgroup-local kill, never a global OOM that would take
# down the box and the Claude session.
#
# pickplace runs as its own unit (already launched); this queue picks up the
# rest once pickplace finishes.
set -u

LOGDIR=/home/sadmin/Work/pyroffi/iosp/data/logs
QLOG="$LOGDIR/queue.log"
QDONE="$LOGDIR/queue.done"
rm -f "$QDONE"
cd /home/sadmin/Work/pyroffi

wait_done () {  # $1 = sentinel tag ; waits for <tag>.done
  local f="$LOGDIR/$1.done"
  while [ ! -f "$f" ]; do sleep 20; done
  echo "$(date -Is) queue: $1 finished (exit $(cat "$f"))"
}

{
  echo "=== queue launch $(date -Is) (pid $$) ==="
  echo "waiting for pickplace to finish (it runs as its own unit)..."
  wait_done regen_pickplace

  echo "=== [serial] tetris + identifiability $(date -Is) ==="
  CUDA_VISIBLE_DEVICES=0 bash iosp/scripts/run_config_detached.sh tetris_ident 0 \
    "iosp/experiments/configs/tetris_iosp.yaml=blocks3_seed0,blocks3_seed1,blocks3_seed2,blocks5_seed0,blocks5_seed1,blocks5_seed2" \
    "iosp/experiments/configs/identifiability_spectrum.yaml=eigen_projection"

  echo "=== [serial] E10 method comparison $(date -Is) ==="
  CUDA_VISIBLE_DEVICES=0 bash iosp/scripts/run_e10_detached.sh

  echo "=== [serial] tower + multistart $(date -Is) ==="
  CUDA_VISIBLE_DEVICES=0 bash iosp/scripts/run_config_detached.sh tower_multi 0 \
    "iosp/experiments/configs/tower_iosp.yaml=blocks3_seed0,blocks3_seed1,blocks3_seed2,blocks5_seed0,blocks5_seed1,blocks5_seed2" \
    "iosp/experiments/configs/multistart_robustness.yaml=joint_seed0,joint_seed1,joint_seed2,joint_seed3"

  echo "=== queue ALL DONE $(date -Is) ==="
  echo "0" > "$QDONE"
} >> "$QLOG" 2>&1
