#!/usr/bin/env bash
# Build _collision_cuda_lib.so from _collision_cuda_kernel.cu.
#
# Usage (from repo root):
#   bash src/pyronot/cuda_kernels/build_collision_cuda.sh
#   bash src/pyronot/cuda_kernels/build_collision_cuda.sh --debug
#
# Requirements:
#   - nvcc (CUDA toolkit)
#   - jaxlib >= 0.4.14 installed in the active Python environment
#     (provides the xla/ffi/api/ffi.h headers)
#
# Optional env vars:
#   GPU_ARCH   override the target architecture, e.g. GPU_ARCH=-arch=sm_80

set -euo pipefail

DEBUG=0
for arg in "$@"; do
  case "$arg" in
    --debug) DEBUG=1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SCRIPT_DIR}/_collision_cuda_kernel.cu"
OUT="${SCRIPT_DIR}/_collision_cuda_lib.so"

# Locate the jaxlib include directory that ships xla/ffi/api/ffi.h.
JAXLIB_INC="$(python3 -c \
  "import os, jaxlib; print(os.path.join(os.path.dirname(jaxlib.__file__), 'include'))")"

if [ ! -f "${JAXLIB_INC}/xla/ffi/api/ffi.h" ]; then
  echo "ERROR: xla/ffi/api/ffi.h not found under ${JAXLIB_INC}"
  echo "Make sure jaxlib >= 0.4.14 is installed in your Python environment."
  exit 1
fi

# GPU architecture flag.
# -arch=native (CUDA 11.6+) targets the installed GPU automatically.
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
  -I"${JAXLIB_INC}" \
  -o "${OUT}" \
  "${SRC}"

echo "Built: ${OUT}"
