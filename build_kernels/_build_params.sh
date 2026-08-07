# shellcheck shell=bash
# Shared build-parameter handling for the pyroffi CUDA kernels — sourced, not executed.
#
# SINGLE SOURCE OF TRUTH for the capacity limits' DEFAULTS. The C++ side's
# src/pyroffi/cuda_kernels/_build_params.cuh carries #ifndef fallbacks with the same
# numbers (for ad-hoc nvcc runs) plus the static_assert guardrails; tests/test_build_params.py
# asserts the two files agree, so a default changed in one place cannot silently drift.
#
# These limits size fixed per-thread arrays in the kernels. A robot whose DOF exceeds the
# COMPILED value would run off the end of them (UB, not a crash), so the values are also
# exported from each .so — see pyroffi_max_act() / pyroffi_max_joints() — and validated
# Python-side before any launch.
#
# Usage from a build script:
#     SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#     source "${SCRIPT_DIR}/_build_params.sh"
#     parse_build_params "$@"        # consumes --max-joints/--max-act/--debug, sets the flags
#     ... nvcc ${BUILD_PARAM_FLAGS} ...
#
# Environment overrides (lower precedence than the CLI flags):
#     PYROFFI_MAX_JOINTS=96 PYROFFI_MAX_ACT=24 bash build_kernels/build_all.sh

# ── Defaults ─────────────────────────────────────────────────────────────────
# Keep in sync with _build_params.cuh's #ifndef fallbacks.
PYROFFI_MAX_JOINTS_DEFAULT=64
# 24, not 16, so panda_allegro (23 actuated DOF) builds out of the box; see the
# matching note in _build_params.cuh.
PYROFFI_MAX_ACT_DEFAULT=24

# ── Guardrail bounds ─────────────────────────────────────────────────────────
# Mirrored from _build_params.cuh's static_asserts. Enforced here too so the user
# gets a one-line error instead of a wall of nvcc template output.
PYROFFI_MAX_ACT_CEILING=64       # block tier's shared double A[N*N] vs the 48KB budget
PYROFFI_MAX_JOINTS_CEILING=256   # per-thread T_world[MAX_JOINTS*7] + shared model

_bp_die() { echo "ERROR: $*" >&2; exit 1; }

_bp_require_posint() {
  # _bp_require_posint <flag-name> <value>
  [[ "$2" =~ ^[1-9][0-9]*$ ]] || _bp_die "$1 must be a positive integer, got '$2'"
}

# parse_build_params "$@"
#
# Consumes --debug, --max-joints[= ]N, --max-act[= ]N. Leaves any unrecognized
# argument as an error (matching the previous per-script behaviour). Sets:
#   DEBUG               0|1
#   PYROFFI_MAX_JOINTS  resolved int
#   PYROFFI_MAX_ACT     resolved int
#   BUILD_PARAM_FLAGS   "-DMAX_JOINTS=<n> -DMAX_ACT=<n>"   (ALWAYS set, never empty)
#   BUILD_PARAM_ARGS    array to forward to sub-scripts (for build_all.sh)
parse_build_params() {
  DEBUG=0
  local max_joints="${PYROFFI_MAX_JOINTS:-}"
  local max_act="${PYROFFI_MAX_ACT:-}"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --debug)          DEBUG=1; shift ;;
      --max-joints)     [[ $# -ge 2 ]] || _bp_die "--max-joints requires an integer value"
                        max_joints="$2"; shift 2 ;;
      --max-joints=*)   max_joints="${1#*=}"; shift ;;
      --max-act)        [[ $# -ge 2 ]] || _bp_die "--max-act requires an integer value"
                        max_act="$2"; shift 2 ;;
      --max-act=*)      max_act="${1#*=}"; shift ;;
      *)                _bp_die "Unknown argument: $1" ;;
    esac
  done

  # Resolve: CLI > env > default. Always end up with a concrete number so the -D is
  # explicit — the .so should never depend on a header fallback for its capacity.
  PYROFFI_MAX_JOINTS="${max_joints:-${PYROFFI_MAX_JOINTS_DEFAULT}}"
  PYROFFI_MAX_ACT="${max_act:-${PYROFFI_MAX_ACT_DEFAULT}}"

  _bp_require_posint "--max-joints" "${PYROFFI_MAX_JOINTS}"
  _bp_require_posint "--max-act"    "${PYROFFI_MAX_ACT}"

  # Guardrails (mirrored from _build_params.cuh so the failure is legible).
  #
  # The CEILING is checked before the MAX_ACT<=MAX_JOINTS relation on purpose: with the
  # default MAX_JOINTS=64 both trip at --max-act 65, and the relation's message ("must be
  # <= MAX_JOINTS") would send you to raise MAX_JOINTS — which cannot help, because the
  # ceiling is 64 regardless. Report the binding limit, not the first one hit.
  if (( PYROFFI_MAX_ACT > PYROFFI_MAX_ACT_CEILING )); then
    _bp_die "MAX_ACT (${PYROFFI_MAX_ACT}) exceeds ${PYROFFI_MAX_ACT_CEILING}: above 32 DOF only the block tier fits, and its shared double A[N*N] (=$((PYROFFI_MAX_ACT*PYROFFI_MAX_ACT*8)) bytes/block here, plus ~6.8KB fixed) must stay inside the 48KB static shared budget. ${PYROFFI_MAX_ACT_CEILING} is the last size that does. Move the solve to dynamic shared memory before raising this."
  fi
  if (( PYROFFI_MAX_ACT > PYROFFI_MAX_JOINTS )); then
    _bp_die "MAX_ACT (${PYROFFI_MAX_ACT}) must be <= MAX_JOINTS (${PYROFFI_MAX_JOINTS}): every actuated DOF is a joint. Are --max-act / --max-joints swapped?"
  fi
  if (( PYROFFI_MAX_JOINTS > PYROFFI_MAX_JOINTS_CEILING )); then
    _bp_die "MAX_JOINTS (${PYROFFI_MAX_JOINTS}) exceeds ${PYROFFI_MAX_JOINTS_CEILING}: per-thread T_world[MAX_JOINTS*7] and the shared robot model exceed the block's shared-memory budget."
  fi

  BUILD_PARAM_FLAGS="-DMAX_JOINTS=${PYROFFI_MAX_JOINTS} -DMAX_ACT=${PYROFFI_MAX_ACT}"
  BUILD_PARAM_ARGS=(--max-joints "${PYROFFI_MAX_JOINTS}" --max-act "${PYROFFI_MAX_ACT}")
  if (( DEBUG )); then BUILD_PARAM_ARGS+=(--debug); fi

  if [[ "${PYROFFI_MAX_JOINTS}" != "${PYROFFI_MAX_JOINTS_DEFAULT}" || "${PYROFFI_MAX_ACT}" != "${PYROFFI_MAX_ACT_DEFAULT}" ]]; then
    echo "Build params: MAX_JOINTS=${PYROFFI_MAX_JOINTS} MAX_ACT=${PYROFFI_MAX_ACT} (non-default — rebuild ALL kernels to keep the .so set consistent)"
  fi
}
