#!/usr/bin/env bash
# Build all CUDA kernels.
#
# Usage (from repo root):
#   bash src/pyronot/cuda_kernels/build_all.sh
#   bash src/pyronot/cuda_kernels/build_all.sh --debug
#
# Override GPU arch for all kernels:
#   GPU_ARCH=-arch=sm_80 bash src/pyronot/cuda_kernels/build_all.sh

set -euo pipefail

DEBUG_FLAG=""
for arg in "$@"; do
  case "$arg" in
    --debug) DEBUG_FLAG="--debug" ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/build_fk_cuda.sh" ${DEBUG_FLAG}
bash "${SCRIPT_DIR}/build_collision_cuda.sh" ${DEBUG_FLAG}
bash "${SCRIPT_DIR}/build_hjcd_ik_cuda.sh" ${DEBUG_FLAG}
bash "${SCRIPT_DIR}/build_ls_ik_cuda.sh" ${DEBUG_FLAG}
bash "${SCRIPT_DIR}/build_mppi_ik_cuda.sh" ${DEBUG_FLAG}
bash "${SCRIPT_DIR}/build_sqp_ik_cuda.sh" ${DEBUG_FLAG}
bash "${SCRIPT_DIR}/build_sco_trajopt_cuda.sh" ${DEBUG_FLAG}
bash "${SCRIPT_DIR}/build_ls_trajopt_cuda.sh" ${DEBUG_FLAG}
bash "${SCRIPT_DIR}/build_chomp_trajopt_cuda.sh" ${DEBUG_FLAG}
bash "${SCRIPT_DIR}/build_stomp_trajopt_cuda.sh" ${DEBUG_FLAG}

echo "All CUDA kernels built successfully."
