#!/usr/bin/env bash
# Build + install cricket with the Python extension AND the runtime JIT enabled,
# so pyroffi's VAMPCPUCollisionChecker can JIT-compile robot-specialised VAMP
# collision checkers at runtime.
#
# This mirrors the exact, verified setup (conda-forge deps + scikit-build-core).
# Run it inside the target conda env, e.g.:
#
#   conda activate pyroffi
#   bash build_kernels/build_cricket_jit.sh
#
# Dependencies are taken from cricket/environment.yaml.  The JIT additionally
# needs a `clang` binary on PATH at *runtime* (the JIT driver shells out to it to
# discover system headers) — clangdev provides it inside the env.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRICKET_DIR="${REPO_ROOT}/external/cricket"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: activate the target conda env first (conda activate pyroffi)." >&2
  exit 1
fi

# 1. Install build/runtime dependencies (no-op if already present).
conda install -c conda-forge --solver=libmamba -y \
  pinocchio cppad eigen cgal-cpp nlohmann_json fmt \
  llvmdev clangdev lld cxx-compiler ninja pkg-config patch \
  nanobind scikit-build-core

if ! command -v clang >/dev/null 2>&1; then
  echo "ERROR: clang still not on PATH after install." >&2
  exit 1
fi

# 2. Build + install the cricket Python extension (JIT + Python both ON).
export CMAKE_PREFIX_PATH="${CONDA_PREFIX}:${CMAKE_PREFIX_PATH:-}"
export CMAKE_ARGS="-DCMAKE_PREFIX_PATH=${CONDA_PREFIX} -DCRICKET_BUILD_JIT=ON -DCRICKET_BUILD_PYTHON=ON"
pip install -e "${CRICKET_DIR}" --no-build-isolation

echo
echo "Done. Verify with:"
echo "  python -c 'from cricket import _core_ext as e; print(e.jit.JitSession)'"
echo "  python -m pytest tests/test_vamp_cpu_collision.py -s"
