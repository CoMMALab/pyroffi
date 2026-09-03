#!/bin/bash
set -euo pipefail
# Reproduce all IOC data. Wraps `python -m ioc.collect --stages all`.
#
# Usage:
#   bash ioc/scripts/run_all.sh              # default GPU 0
#   bash ioc/scripts/run_all.sh --gpu 2      # pin to GPU 2
#   bash ioc/scripts/run_all.sh --dry-run    # print without executing
#   bash ioc/scripts/run_all.sh --stages e1_noise,e2_scaling

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

GPU=0
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

export CUDA_VISIBLE_DEVICES="$GPU"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=1
export PYTHONUNBUFFERED=1

cd "$REPO_DIR"

if [[ ${#EXTRA_ARGS[@]} -eq 0 ]]; then
  python -m ioc.collect --stages all
else
  python -m ioc.collect "${EXTRA_ARGS[@]}"
fi
