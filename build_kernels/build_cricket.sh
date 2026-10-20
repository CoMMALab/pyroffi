#!/usr/bin/env bash
# Install crocket, the trace compiler GRiM uses for its traced builds (traced=True kinematics,
# and the traced dynamics tiers behind ContractDynamics); everything else in pyroffi runs
# without it. crocket is pure Python and lives in GRiM at external/crocket.
#
# Run it inside the target env, e.g.:
#
#   conda activate pyroffi
#   bash build_kernels/build_cricket.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GRIM_DIR="${REPO_ROOT}/external/GRiM"
CROCKET_DIR="${GRIM_DIR}/external/crocket"
if [[ ! -f "${CROCKET_DIR}/pyproject.toml" ]]; then
  git -C "${REPO_ROOT}" submodule update --init --recursive external/GRiM
fi

pip install -e "${CROCKET_DIR}"

echo
echo "Done. Verify with:"
echo "  python -m pytest tests/test_ik_traced.py -q"
