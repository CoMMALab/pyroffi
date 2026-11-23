#!/usr/bin/env bash
# Build pyroffi's own CUDA kernels: the world-side collision checkers, the ESDF builder and
# roboGPU (skipped without the OptiX SDK). The motion kernels (IK, trajopt, robot-side collision, dynamics, least squares)
# live in external/GRiM and compile per robot on first use.
#
# Usage (from repo root):
#   bash build_kernels/build_all.sh
#   bash build_kernels/build_all.sh --debug
#
# Override GPU arch for all kernels:
#   GPU_ARCH=-arch=sm_80 bash build_kernels/build_all.sh

set -euo pipefail

# Parse + validate ONCE here, then forward the resolved flags to every kernel.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_build_params.sh"
parse_build_params "$@"
BUILD_ARGS=("${BUILD_PARAM_ARGS[@]}")

echo "Building pyroffi's world-side collision kernels (motion kernels build per robot through GRiM)"

bash "${SCRIPT_DIR}/build_collision_cuda.sh" "${BUILD_ARGS[@]}"
bash "${SCRIPT_DIR}/build_collision_binary_cuda.sh" "${BUILD_ARGS[@]}"
bash "${SCRIPT_DIR}/build_pba_cuda.sh" "${BUILD_ARGS[@]}"
# roboGPU is optional and needs the NVIDIA OptiX SDK (a manual download). Without it, skip it
# here; roboGPU then raises when used. mantis needs no step: `import grim` finds it in GRiM.
rc=0
bash "${SCRIPT_DIR}/build_robogpu_collision.sh" "${BUILD_ARGS[@]}" || rc=$?
if [[ ${rc} -eq 3 ]]; then
  echo "SKIPPED roboGPU (no OptiX SDK). Set OPTIX_SDK and rerun build_robogpu_collision.sh to add it."
elif [[ ${rc} -ne 0 ]]; then
  exit "${rc}"
fi

echo "All CUDA kernels built successfully."
