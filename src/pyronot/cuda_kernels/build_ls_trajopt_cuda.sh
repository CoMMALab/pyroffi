#!/usr/bin/env bash
# Build _ls_trajopt_cuda_lib.so from _ls_trajopt_cuda_kernel.cu.
#
# Usage (from repo root):
#   bash src/pyronot/cuda_kernels/build_ls_trajopt_cuda.sh
#   bash src/pyronot/cuda_kernels/build_ls_trajopt_cuda.sh --debug

set -euo pipefail

DEBUG=0
for arg in "$@"; do
  case "$arg" in
    --debug) DEBUG=1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SCRIPT_DIR}/_ls_trajopt_cuda_kernel.cu"
OUT="${SCRIPT_DIR}/_ls_trajopt_cuda_lib.so"

JAXLIB_INC="$(python3 -c \
  "import os, jaxlib; print(os.path.join(os.path.dirname(jaxlib.__file__), 'include'))")"

if [ ! -f "${JAXLIB_INC}/xla/ffi/api/ffi.h" ]; then
  echo "ERROR: xla/ffi/api/ffi.h not found under ${JAXLIB_INC}"
  echo "Make sure jaxlib >= 0.4.14 is installed in your Python environment."
  exit 1
fi

GPU_ARCH="${GPU_ARCH:--arch=native}"

NVCC_OPT="-O3"
if [ "${DEBUG}" -eq 1 ]; then
  NVCC_OPT="-O0 -G -lineinfo"
  echo "Building in DEBUG mode (with -G for Nsight Compute)..."
fi

nvcc \
  ${NVCC_OPT} \
  -std=c++17 \
  ${GPU_ARCH} \
  --shared \
  --compiler-options "-fPIC" \
  -I"${SCRIPT_DIR}" \
  -I"${JAXLIB_INC}" \
  -o "${OUT}" \
  "${SRC}"

echo "Built: ${OUT}"
