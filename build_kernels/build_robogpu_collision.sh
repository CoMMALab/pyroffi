#!/usr/bin/env bash
# Build the RoboGPU OptiX sphere-octree collision checker.
#
# Produces two files in src/pyroffi/cuda_kernels/:
#   _robogpu_optix_programs.ptx  — OptiX device programs (ray gen / intersection
#                                   / any-hit / miss), loaded at runtime.
#   _robogpu_collision_lib.so    — Host library with FK CUDA kernel + OptiX
#                                   pipeline management + XLA FFI handler.
#
# Usage (from repo root):
#   bash build_kernels/build_robogpu_collision.sh
#   bash build_kernels/build_robogpu_collision.sh --debug
#   bash build_kernels/build_robogpu_collision.sh --max-joints 128
#
# Requirements:
#   - nvcc  (CUDA toolkit >= 11.2 for cudaMallocAsync)
#   - NVIDIA OptiX SDK 7.x  (set OPTIX_SDK or install to a standard path)
#   - jaxlib >= 0.4.14  (provides xla/ffi/api/ffi.h headers)
#
# Optional environment variables:
#   OPTIX_SDK   Path to the OptiX SDK root (contains include/optix.h).
#               If unset, common paths are searched automatically.
#   GPU_ARCH    nvcc architecture flag, e.g. -arch=sm_86 (default: -arch=native).
#               Must be sm_50 or newer (OptiX 7 requirement).

set -euo pipefail

DEBUG=0
MAX_JOINTS_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --debug)
      DEBUG=1; shift ;;
    --max-joints)
      [[ $# -lt 2 ]] && { echo "ERROR: --max-joints requires a value"; exit 1; }
      MAX_JOINTS_OVERRIDE="$2"; shift 2 ;;
    --max-joints=*)
      MAX_JOINTS_OVERRIDE="${1#*=}"; shift ;;
    *)
      echo "ERROR: Unknown argument: $1"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNELS_DIR="$(cd "${SCRIPT_DIR}/../src/pyroffi/cuda_kernels" && pwd)"

DEVICE_SRC="${KERNELS_DIR}/collision/_robogpu_optix_programs.cu"
HOST_SRC="${KERNELS_DIR}/collision/_robogpu_collision_host.cu"
PTX_OUT="${KERNELS_DIR}/collision/_robogpu_optix_programs.ptx"
SO_OUT="${KERNELS_DIR}/collision/_robogpu_collision_lib.so"

# ── Locate jaxlib XLA FFI headers ──────────────────────────────────────────

JAXLIB_INC="$(python -c \
  "import os, jaxlib; print(os.path.join(os.path.dirname(jaxlib.__file__), 'include'))")"

if [ ! -f "${JAXLIB_INC}/xla/ffi/api/ffi.h" ]; then
  echo "ERROR: xla/ffi/api/ffi.h not found under ${JAXLIB_INC}"
  echo "Make sure jaxlib >= 0.4.14 is installed."
  exit 1
fi

# ── Locate OptiX SDK ────────────────────────────────────────────────────────

find_optix_sdk() {
  # 1. Explicit env var
  if [[ -n "${OPTIX_SDK:-}" && -f "${OPTIX_SDK}/include/optix.h" ]]; then
    echo "${OPTIX_SDK}"; return 0
  fi
  # 2. Common install paths
  for p in \
      /usr/local/optix \
      /opt/NVIDIA-OptiX-SDK* \
      "${HOME}/NVIDIA-OptiX-SDK"* \
      "${SCRIPT_DIR}/../NVIDIA-OptiX-SDK"* \
      /usr/local/cuda \
  ; do
    # glob expansion in bash 'for' already handles wildcards
    if [[ -f "${p}/include/optix.h" ]]; then
      echo "${p}"; return 0
    fi
  done
  # 3. Scan PATH for optixNamespace.h peer
  local pp
  for pp in $(tr ':' '\n' <<< "${PATH:-}"); do
    local candidate
    for candidate in "${pp}/../include" "${pp}/../../include"; do
      if [[ -f "${candidate}/optix.h" ]]; then
        realpath "${candidate}/.." 2>/dev/null && return 0
      fi
    done
  done
  return 1
}

OPTIX_ROOT=""
if OPTIX_ROOT="$(find_optix_sdk)"; then
  echo "OptiX SDK found: ${OPTIX_ROOT}"
else
  echo "ERROR: NVIDIA OptiX SDK 7.x not found."
  echo "Install it and set OPTIX_SDK=/path/to/optix or pass it to PATH."
  echo "Download: https://developer.nvidia.com/designworks/optix/download"
  exit 1
fi

OPTIX_INC="${OPTIX_ROOT}/include"

# ── GPU architecture flag ───────────────────────────────────────────────────

GPU_ARCH="${GPU_ARCH:--arch=native}"

# ── Build flags ─────────────────────────────────────────────────────────────

if [ "${DEBUG}" -eq 1 ]; then
  NVCC_OPT="-O0 -G -lineinfo"
  PTX_OPT="-O0 -G"
  echo "Building in DEBUG mode (-G / lineinfo)..."
else
  NVCC_OPT="-O3"
  PTX_OPT="-O3"
fi

EXTRA_DEFS=""
if [[ -n "${MAX_JOINTS_OVERRIDE}" ]]; then
  EXTRA_DEFS="-DRGB_MAX_JOINTS=${MAX_JOINTS_OVERRIDE} -DRGB_MAX_LINKS=${MAX_JOINTS_OVERRIDE}"
  echo "Custom bounds: RGB_MAX_JOINTS/LINKS=${MAX_JOINTS_OVERRIDE}"
fi

# ── Step 1: Compile OptiX device programs to PTX ────────────────────────────
# The PTX is loaded at runtime by the host library via optixModuleCreate.

echo ""
echo "Step 1: Compiling OptiX device programs → PTX"
echo "  Source : ${DEVICE_SRC}"
echo "  Output : ${PTX_OUT}"

nvcc \
  ${PTX_OPT} \
  -std=c++17 \
  ${GPU_ARCH} \
  --ptx \
  -I"${OPTIX_INC}" \
  -o "${PTX_OUT}" \
  "${DEVICE_SRC}"

echo "  OK: ${PTX_OUT}"

# ── Step 2: Compile host code → shared library ──────────────────────────────
# The host code contains: FK + sphere-transform CUDA kernel, OptiX pipeline
# management (optix_stubs.h), BVH build/cache, and the XLA FFI handler.

echo ""
echo "Step 2: Compiling host library → .so"
echo "  Source : ${HOST_SRC}"
echo "  Output : ${SO_OUT}"

nvcc \
  ${NVCC_OPT} \
  -std=c++17 \
  ${GPU_ARCH} \
  --shared \
  --compiler-options "-fPIC" \
  -I"${JAXLIB_INC}" \
  -I"${OPTIX_INC}" \
  -I"${KERNELS_DIR}" \
  ${EXTRA_DEFS} \
  -ldl \
  -o "${SO_OUT}" \
  "${HOST_SRC}"

echo "  OK: ${SO_OUT}"

echo ""
echo "Build complete."
echo "  PTX : ${PTX_OUT}"
echo "  SO  : ${SO_OUT}"
echo ""
echo "Both files must reside in the same directory at runtime"
echo "(the host library locates the PTX file using dladdr)."
