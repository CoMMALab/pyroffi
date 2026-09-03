#!/bin/bash
set -euo pipefail
# Render all IOSP paper figures from collected data.
#
# Usage:
#   bash iosp/scripts/run_figures.sh                # all figures
#   bash iosp/scripts/run_figures.sh fig2_multistart # one figure

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IOSP_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$IOSP_DIR")"
cd "$REPO_DIR"

ONLY=""
if [[ $# -gt 0 ]]; then
    ONLY="--only $1"
fi

python -m iosp.make_figures $ONLY
