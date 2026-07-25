#!/usr/bin/env bash
# Build _sqp_ik_cuda_lib.so from _sqp_ik_cuda_kernel.cu.
#
# Usage (from repo root):
#   bash build_kernels/build_sqp_ik_cuda.sh
#   bash build_kernels/build_sqp_ik_cuda.sh --debug
#   bash build_kernels/build_sqp_ik_cuda.sh --max-act=24   # e.g. panda_allegro (23 DOF)
#
# Requirements:
#   - nvcc (CUDA toolkit)
#   - jaxlib >= 0.4.14 installed in the active Python environment
#     (provides the xla/ffi/api/ffi.h headers)

set -euo pipefail

DEBUG=0
MAX_JOINTS_OVERRIDE=""
MAX_ACT_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --debug)
      DEBUG=1
      shift
      ;;
    --max-joints)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --max-joints requires an integer value"
        exit 1
      fi
      MAX_JOINTS_OVERRIDE="$2"
      shift 2
      ;;
    --max-joints=*)
      MAX_JOINTS_OVERRIDE="${1#*=}"
      shift
      ;;
    --max-act)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --max-act requires an integer value"
        exit 1
      fi
      MAX_ACT_OVERRIDE="$2"
      shift 2
      ;;
    --max-act=*)
      MAX_ACT_OVERRIDE="${1#*=}"
      shift
      ;;
    *)
      echo "ERROR: Unknown argument: $1"
      exit 1
      ;;
  esac
done

MAX_JOINTS_FLAG=""
if [[ -n "${MAX_JOINTS_OVERRIDE}" ]]; then
  if ! [[ "${MAX_JOINTS_OVERRIDE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --max-joints must be a positive integer, got '${MAX_JOINTS_OVERRIDE}'"
    exit 1
  fi
  MAX_JOINTS_FLAG="-DMAX_JOINTS=${MAX_JOINTS_OVERRIDE}"
  echo "Overriding MAX_JOINTS=${MAX_JOINTS_OVERRIDE}"
fi

# MAX_ACT bounds the per-thread cfg/Jacobian/Hessian stack arrays, so it must be
# >= the robot's actuated-joint count or the kernel writes out of bounds
# (CUDA_ERROR_ILLEGAL_ADDRESS). Default 16 fits e.g. panda (7); panda_allegro
# needs 23 -> build with --max-act=24. Cost: H_s/A_init are MAX_ACT^2 doubles per
# thread, so raising it grows local memory quadratically and lowers occupancy.
MAX_ACT_FLAG=""
if [[ -n "${MAX_ACT_OVERRIDE}" ]]; then
  if ! [[ "${MAX_ACT_OVERRIDE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --max-act must be a positive integer, got '${MAX_ACT_OVERRIDE}'"
    exit 1
  fi
  MAX_ACT_FLAG="-DMAX_ACT=${MAX_ACT_OVERRIDE}"
  echo "Overriding MAX_ACT=${MAX_ACT_OVERRIDE}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNELS_DIR="$(cd "${SCRIPT_DIR}/../src/pyroffi/cuda_kernels" && pwd)"
SRC="${KERNELS_DIR}/ik/_sqp_ik_cuda_kernel.cu"
OUT="${KERNELS_DIR}/ik/_sqp_ik_cuda_lib.so"

JAXLIB_INC="$(python -c \
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
  ${MAX_JOINTS_FLAG} \
  ${MAX_ACT_FLAG} \
  ${GPU_ARCH} \
  --shared \
  --compiler-options "-fPIC" \
  -I"${KERNELS_DIR}" \
  -I"${JAXLIB_INC}" \
  -I"${KERNELS_DIR}" \
  -o "${OUT}" \
  "${SRC}"

echo "Built: ${OUT}"
