#!/bin/bash
set -euo pipefail
# Render all IOC paper figures from collected data.
#
# Usage:
#   bash ioc/scripts/run_figures.sh                     # all figures
#   bash ioc/scripts/run_figures.sh scaling              # one figure
#   bash ioc/scripts/run_figures.sh scaling,recovery     # selected figures

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$REPO_DIR"

if [[ $# -gt 0 ]]; then
  python -m ioc.plots --only "$1"
else
  python -m ioc.plots
fi
