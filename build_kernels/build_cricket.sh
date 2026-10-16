#!/usr/bin/env bash
# Install cricket2, the trace compiler GRiM uses for its traced builds (traced=True kinematics,
# and the traced dynamics tiers behind ContractDynamics); everything else in pyroffi runs
# without it. cricket2 is pure Python and lives in GRiM at external/cricket2.
#
# Run it inside the target env, e.g.:
#
#   conda activate pyroffi
#   bash build_kernels/build_cricket.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GRIM_DIR="${REPO_ROOT}/external/GRiM"
CRICKET2_DIR="${GRIM_DIR}/external/cricket2"
if [[ ! -f "${CRICKET2_DIR}/pyproject.toml" ]]; then
  git -C "${REPO_ROOT}" submodule update --init --recursive external/GRiM
fi

pip install -e "${CRICKET2_DIR}"

echo
echo "Done. Verify with:"
echo "  python -m pytest tests/test_ik_traced.py -q"
