#!/usr/bin/env bash
# Build _svgd_region_ik_cuda_lib.so from _svgd_region_ik_cuda_kernel.cu.
#
# Usage (from repo root):
#   bash build_kernels/build_svgd_region_ik_cuda.sh

set -euo pipefail

# Build parameters (--max-joints / --max-act / --debug) + guardrails live in one
# place so the kernel builds cannot drift apart. Defaults are applied there and
# ALWAYS passed as -D, so a .so never depends on a header fallback for its capacity.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_build_params.sh"
parse_build_params "$@"
KERNELS_DIR="$(cd "${SCRIPT_DIR}/../src/pyroffi/cuda_kernels" && pwd)"
SRC="${KERNELS_DIR}/region_ik/_svgd_region_ik_cuda_kernel.cu"
OUT="${KERNELS_DIR}/region_ik/_svgd_region_ik_cuda_lib.so"

JAXLIB_INC="$(python -c "
import os, jaxlib
print(os.path.join(os.path.dirname(jaxlib.__file__), 'include'))
")"

if [ ! -f "${JAXLIB_INC}/xla/ffi/api/ffi.h" ]; then
  echo "ERROR: xla/ffi/api/ffi.h not found under ${JAXLIB_INC}"
  echo "Make sure jaxlib >= 0.4.14 is installed in your Python environment."
  exit 1
fi

GPU_ARCH="${GPU_ARCH:--arch=native}"

nvcc \
  -O3 \
  -std=c++17 \
  ${BUILD_PARAM_FLAGS} \
  ${GPU_ARCH} \
  --shared \
  --compiler-options "-fPIC" \
  -I"${KERNELS_DIR}" \
  -I"${JAXLIB_INC}" \
  -I"${KERNELS_DIR}" \
  -o "${OUT}" \
  "${SRC}"

echo "Built: ${OUT}"
