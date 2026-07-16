#!/usr/bin/env bash
# Build _stomp_trajopt_cuda_lib.so from _stomp_trajopt_cuda_kernel.cu.
#
# Usage (from repo root):
#   bash build_kernels/build_stomp_trajopt_cuda.sh
#   bash build_kernels/build_stomp_trajopt_cuda.sh --debug
#
# Requirements:
#   - nvcc (CUDA toolkit)
#   - jaxlib >= 0.4.14 installed in the active Python environment
#     (provides the xla/ffi/api/ffi.h headers)

set -euo pipefail

# Build parameters (--max-joints / --max-act / --debug) + guardrails live in one
# place so the 15 kernel builds cannot drift apart. Defaults are applied there and
# ALWAYS passed as -D, so a .so never depends on a header fallback for its capacity.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_build_params.sh"
parse_build_params "$@"
KERNELS_DIR="$(cd "${SCRIPT_DIR}/../src/pyroffi/cuda_kernels" && pwd)"
# GLASS: header-only block/warp/thread linear algebra (external/GLASS/glass.cuh).
GLASS_INC="$(cd "${SCRIPT_DIR}/../external/GLASS" && pwd)"
SRC="${KERNELS_DIR}/trajopt/_stomp_trajopt_cuda_kernel.cu"
OUT="${KERNELS_DIR}/trajopt/_stomp_trajopt_cuda_lib.so"

# Locate the jaxlib include directory that ships xla/ffi/api/ffi.h.
JAXLIB_INC="$(python -c \
  "import os, jaxlib; print(os.path.join(os.path.dirname(jaxlib.__file__), 'include'))")"

if [ ! -f "${JAXLIB_INC}/xla/ffi/api/ffi.h" ]; then
  echo "ERROR: xla/ffi/api/ffi.h not found under ${JAXLIB_INC}"
  echo "Make sure jaxlib >= 0.4.14 is installed in your Python environment."
  exit 1
fi

# GPU architecture flag.
# -arch=native (CUDA 11.6+) targets the installed GPU automatically.
# Override for a specific arch: GPU_ARCH=-arch=sm_80 bash build_stomp_trajopt_cuda.sh
GPU_ARCH="${GPU_ARCH:--arch=native}"

NVCC_OPT="-O3"
if [ "${DEBUG}" -eq 1 ]; then
  NVCC_OPT="-O0 -G -lineinfo"
  echo "Building in DEBUG mode (with -G for Nsight Compute)..."
fi

nvcc \
  ${NVCC_OPT} \
  -std=c++17 \
  ${BUILD_PARAM_FLAGS} \
  ${GPU_ARCH} \
  --shared \
  --compiler-options "-fPIC" \
  -I"${JAXLIB_INC}" \
  -I"${KERNELS_DIR}" \
  -I"${GLASS_INC}" \
  -o "${OUT}" \
  "${SRC}"

echo "Built: ${OUT}"
