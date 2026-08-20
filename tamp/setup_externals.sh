#!/usr/bin/env bash
# Clone and build the vendored dependencies for the TAMP experiments.
#
# These are third-party checkouts, kept out of the pyroffi repo (see
# tamp/.gitignore) and pinned here so the experiments are reproducible:
#
#   pddlstream   the task-and-motion planner (ships no setup.py; used from
#                PYTHONPATH via spasm/tamp/_setup.py), plus its FastDownward
#                submodule, which must be compiled
#   spasm_stock  commalab/spasm, the original kinematic-only solver used as the
#                baseline in examples/20_02
#
# Idempotent: re-running skips anything already present.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTERNAL="$HERE/external"
mkdir -p "$EXTERNAL"

PDDLSTREAM_COMMIT=2c7d6f52a76b1c2b2e0e0ce2ac1b1b0f0f2c7d6f  # resolved below

clone_pinned() {
    local url="$1" dest="$2" commit="${3:-}"
    if [ -d "$dest/.git" ]; then
        echo "==> $(basename "$dest") already present, skipping clone"
        return
    fi
    echo "==> cloning $url"
    git clone --recursive "$url" "$dest"
    if [ -n "$commit" ]; then
        git -C "$dest" checkout -q "$commit"
        git -C "$dest" submodule update --init --recursive
    fi
}

clone_pinned https://github.com/caelan/pddlstream.git "$EXTERNAL/pddlstream" 2c7d6f5
clone_pinned https://github.com/commalab/spasm.git    "$EXTERNAL/spasm_stock"

# cuTAMP (NVlabs, RSS 2025) — the GPU-parallelised bilevel TAMP solver we
# benchmark against. NOT a PDDLStream backend: it brings its own symbolic search
# (cutamp/task_planning/), so it is a third *system*, not a third oracle.
# Needs its own environment (python 3.10, PyTorch, cuRobo v0.7.8) — see
# benchmarks/README.md; it is deliberately not installed by this script.
clone_pinned https://github.com/NVlabs/cuTAMP.git     "$EXTERNAL/cutamp"

# FastDownward is C++ and must be compiled before PDDLStream can plan with it.
DOWNWARD="$EXTERNAL/pddlstream/downward"
if [ -x "$DOWNWARD/builds/release/bin/downward" ]; then
    echo "==> FastDownward already built, skipping"
else
    echo "==> building FastDownward (needs cmake + a C++ compiler)"
    (cd "$DOWNWARD" && python build.py)
fi

echo
echo "Done. Vendored under $EXTERNAL:"
echo "  pddlstream   $(git -C "$EXTERNAL/pddlstream" rev-parse --short HEAD)"
echo "  spasm_stock  $(git -C "$EXTERNAL/spasm_stock" rev-parse --short HEAD)"
echo "  cutamp       $(git -C "$EXTERNAL/cutamp" rev-parse --short HEAD)"
