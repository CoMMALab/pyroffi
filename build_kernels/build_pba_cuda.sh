#!/usr/bin/env bash
# Build _pba_cuda_lib.so from _pba_cuda_kernel.cu (PBA+ 3D Euclidean
# Distance Transform, faithful-V2 ESDF port Piece 2 -- the CUDA
# edt_solver="pba" alternative to pyroffi.collision._esdf's pure-JAX
# edt_solver="jfa" baseline).
#
# Usage (from repo root):
#   bash build_kernels/build_pba_cuda.sh
#   bash build_kernels/build_pba_cuda.sh --debug
#
# Requirements:
#   - nvcc (CUDA toolkit)
#   - jaxlib >= 0.4.14 installed in the active Python environment
#     (provides the xla/ffi/api/ffi.h headers)
#
# Note: if your default `nvcc` rejects the system g++ as "unsupported" (CUDA
# 12.8's nvcc caps out below gcc 15), put a newer CUDA toolkit's bin/ (e.g.
# CUDA 13.3, which accepts gcc 13) earlier on PATH before running this script
# -- same as build_collision_cuda.sh.
#
# Optional env vars:
#   GPU_ARCH   override the target architecture, e.g. GPU_ARCH=-arch=sm_80

set -euo pipefail

# --max-joints / --max-act are accepted (so build_all.sh can forward one
# resolved pair to every kernel) but unused here: PBA's distance-transform
# kernels don't size anything from robot DOF. parse_build_params has
# already validated them.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_build_params.sh"
parse_build_params "$@"

KERNELS_DIR="$(cd "${SCRIPT_DIR}/../src/pyroffi/cuda_kernels" && pwd)"
SRC="${KERNELS_DIR}/esdf/_pba_cuda_kernel.cu"
OUT="${KERNELS_DIR}/esdf/_pba_cuda_lib.so"

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
  -I"${KERNELS_DIR}/esdf" \
  -o "${OUT}" \
  "${SRC}"

echo "Built: ${OUT}"
