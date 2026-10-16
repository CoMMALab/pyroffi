#!/usr/bin/env bash
# Build pyroffi's own CUDA kernels: the world-side collision checkers, the ESDF builder and
# roboGPU. The motion kernels (IK, trajopt, robot-side collision, dynamics, least squares)
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
bash "${SCRIPT_DIR}/build_robogpu_collision.sh" "${BUILD_ARGS[@]}"
bash "${SCRIPT_DIR}/build_cricket.sh"

echo "All CUDA kernels built successfully."
