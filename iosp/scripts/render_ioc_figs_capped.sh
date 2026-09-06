#!/usr/bin/env bash
# Render the inline-built IOC figures one per capped process, serially, so a
# host-RAM OOM kills only that figure's scope (not the session) and JAX caches
# are released between figures. Each figure computes -> saves data/figdata/<name>.npz
# -> renders from disk. Cheap figures (ambiguity, environments) first, then the
# heavy multi-method fits (recovery, recovery_highnoise, regime).
set -u
cd /home/sadmin/Work/pyroffi
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pyroffi-tamp
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg

MEMCAP="${MEMCAP:-55G}"
FIGS="${FIGS:-ambiguity environments recovery recovery_highnoise regime}"

for fig in $FIGS; do
  echo "=================== $fig  ($(date +%H:%M:%S)) ==================="
  systemd-run --user --scope -p MemoryMax="$MEMCAP" -p MemorySwapMax=0 \
    python -m ioc.plots --only "$fig"
  rc=$?
  echo ">>> $fig exit=$rc"
  if [ $rc -ne 0 ]; then
    echo ">>> $fig FAILED (rc=$rc) -- continuing to next figure"
  fi
done
echo "=================== ALL DONE ($(date +%H:%M:%S)) ==================="
ls -la ioc/data/figdata/
