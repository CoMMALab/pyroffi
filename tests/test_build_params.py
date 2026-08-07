"""The kernel capacity limits (MAX_JOINTS / MAX_ACT) must agree across four places.

The limits size fixed per-thread arrays in the CUDA kernels, which do no bounds
checking — a robot exceeding them corrupts thread-local state silently rather than
crashing. Four things therefore have to stay in lockstep:

  1. build_kernels/_build_params.sh   — the defaults the build scripts apply
  2. src/pyroffi/cuda_kernels/_build_params.cuh — the #ifndef fallbacks + guardrails
  3. every built .so                  — what it ACTUALLY compiled with (exported)
  4. Python-side hardcoded bounds     — e.g. _REGION_IK_MAX_TPB_BY_SMEM

These tests pin 1==2 (pure text, no GPU) and, when libraries are present, that each
.so self-reports and that the capacity check rejects oversized robots.
"""

from __future__ import annotations

import ctypes
import importlib.util
import pathlib
import re
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SH = REPO / "build_kernels" / "_build_params.sh"
CUH = REPO / "src" / "pyroffi" / "cuda_kernels" / "_build_params.cuh"
KERNELS = REPO / "src" / "pyroffi" / "cuda_kernels"


def _load_bp():
    """Import _build_params.py directly (the pyroffi package pulls in jax)."""
    spec = importlib.util.spec_from_file_location("_bp", KERNELS / "_build_params.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sh_default(name: str) -> int:
    m = re.search(rf"^{name}=(\d+)$", SH.read_text(), re.M)
    assert m, f"{name} not found in {SH.name}"
    return int(m.group(1))


def _cuh_default(name: str) -> int:
    m = re.search(rf"^#define {name} (\d+)$", CUH.read_text(), re.M)
    assert m, f"#define {name} not found in {CUH.name}"
    return int(m.group(1))


# ── 1 == 2: the two default sources agree (no GPU needed) ────────────────────

@pytest.mark.parametrize("sh_name,cuh_name", [
    ("PYROFFI_MAX_JOINTS_DEFAULT", "MAX_JOINTS"),
    ("PYROFFI_MAX_ACT_DEFAULT", "MAX_ACT"),
])
def test_shell_and_header_defaults_agree(sh_name, cuh_name):
    assert _sh_default(sh_name) == _cuh_default(cuh_name), (
        f"{SH.name}'s {sh_name} and {CUH.name}'s #define {cuh_name} have drifted. "
        f"The shell value is what every .so is actually built with; the header value "
        f"is the fallback for ad-hoc nvcc runs. They must match."
    )


def test_ceilings_mirror_the_static_asserts():
    """The shell's early-exit bounds must match the header's static_asserts."""
    sh = SH.read_text()
    cuh = CUH.read_text()
    # NB: these lines carry trailing comments, so no `$` anchor.
    sh_act = int(re.search(r"^PYROFFI_MAX_ACT_CEILING=(\d+)", sh, re.M).group(1))
    sh_jnt = int(re.search(r"^PYROFFI_MAX_JOINTS_CEILING=(\d+)", sh, re.M).group(1))
    assert f"MAX_ACT <= {sh_act}" in cuh, (
        f"_build_params.sh caps MAX_ACT at {sh_act} but the header's static_assert "
        f"does not. The shell check exists only to give a readable error before nvcc; "
        f"the static_assert is the real guard and they must agree."
    )
    assert f"MAX_JOINTS <= {sh_jnt}" in cuh, (
        f"_build_params.sh caps MAX_JOINTS at {sh_jnt} but the header's static_assert "
        f"does not."
    )
    assert _load_bp().MAX_ACT_CEILING == sh_act, (
        f"_build_params.py's MAX_ACT_CEILING and _build_params.sh's "
        f"PYROFFI_MAX_ACT_CEILING={sh_act} have drifted. Python uses it to decide "
        f"whether an oversized robot is a rebuild away or unsupported outright; if it "
        f"is too high it recommends a --max-act the build script will reject."
    )


def test_solve_bucket_ceiling_matches_max_act_ceiling():
    """The largest solve bucket must equal MAX_ACT's ceiling.

    A MAX_ACT above the last bucket means solve_bucket() returns 0 for a legal robot
    and NO kernel launches (the dispatch's `default`), so the FFI would hand back an
    uninitialized buffer. _glass_solve.cuh static_asserts this too; this catches it
    without a GPU.
    """
    glass = (KERNELS / "_glass_solve.cuh").read_text()
    buckets = re.search(r"#define PYROFFI_SOLVE_N_BUCKETS\(X\)(.+)", glass).group(1)
    largest = max(int(n) for n in re.findall(r"X\((\d+)\)", buckets))
    assert largest == _load_bp().MAX_ACT_CEILING, (
        f"largest solve bucket ({largest}) != MAX_ACT ceiling "
        f"({_load_bp().MAX_ACT_CEILING}). Add a bucket or lower the ceiling."
    )
    assert f"#define PYROFFI_SOLVE_MAX_N {largest}" in glass, (
        f"PYROFFI_SOLVE_MAX_N must be the largest bucket ({largest}); the FFI error "
        f"messages stringify it."
    )


def test_tier_choice_ceiling_agrees_between_python_and_cuda():
    """Python's TIER_CHOICE_MAX_ACT must mirror _glass_solve.cuh's TIER_CHOICE_MAX_N.

    Above it the kernels are locked to the block tier and PYROFFI_IK_TIER is ignored;
    the tooling describes that behaviour from the Python constant.
    """
    glass = (KERNELS / "_glass_solve.cuh").read_text()
    cuda_val = int(re.search(r"constexpr int TIER_CHOICE_MAX_N = (\d+);", glass).group(1))
    assert cuda_val == _load_bp().TIER_CHOICE_MAX_ACT


# ── shell guardrails actually reject bad input ───────────────────────────────

@pytest.mark.parametrize("args,expect", [
    (["--max-act", "32", "--max-joints", "8"], "must be <= MAX_JOINTS"),
    (["--max-act", "65"], "exceeds 64"),
    (["--max-joints", "512"], "exceeds 256"),
    (["--max-act", "0"], "positive integer"),
    (["--max-act", "7.5"], "positive integer"),
    (["--max-act"], "requires an integer value"),
])
def test_shell_guardrails_reject(args, expect):
    r = subprocess.run(
        ["bash", "-c", f'source "{SH}"; parse_build_params "$@"', "--", *args],
        capture_output=True, text=True,
    )
    assert r.returncode != 0, f"expected rejection of {args}, got success"
    assert expect in r.stderr, f"expected {expect!r} in stderr, got: {r.stderr!r}"


@pytest.mark.parametrize("max_act", [33, 48, 64])
def test_shell_accepts_high_dof_builds(max_act):
    """Above 32 DOF is a supported (block-tier-only) build, not an error.

    Pins the lifted ceiling: --max-act 48 is what a 43-DOF G1 needs, and it used to be
    rejected outright while _build_params.py's error message recommended it.
    """
    r = subprocess.run(
        ["bash", "-c", f'source "{SH}"; parse_build_params "$@"', "--",
         "--max-act", str(max_act)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"--max-act {max_act} should build, got: {r.stderr!r}"


def test_max_act_ceiling_beats_the_max_joints_relation():
    """--max-act 65 must cite the 64 ceiling, not 'must be <= MAX_JOINTS'.

    Both trip at the default MAX_JOINTS=64, but only one is actionable: raising
    MAX_JOINTS does not buy a higher MAX_ACT.
    """
    r = subprocess.run(
        ["bash", "-c", f'source "{SH}"; parse_build_params "$@"', "--", "--max-act", "65"],
        capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert "exceeds 64" in r.stderr, f"expected the ceiling message, got: {r.stderr!r}"


def test_shell_defaults_produce_explicit_flags():
    """Defaults must be passed as -D, never left to the header fallback."""
    r = subprocess.run(
        ["bash", "-c", f'source "{SH}"; parse_build_params; echo "$BUILD_PARAM_FLAGS"'],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    out = r.stdout.strip()
    assert f"-DMAX_JOINTS={_sh_default('PYROFFI_MAX_JOINTS_DEFAULT')}" in out
    assert f"-DMAX_ACT={_sh_default('PYROFFI_MAX_ACT_DEFAULT')}" in out


# ── BuildParams.check(): rebuildable vs unsupported ─────────────────────────

def test_check_recommends_a_buildable_max_act():
    """The recommended --max-act must be one the build script accepts.

    Regression: the message used to recommend `--max-act 48` for a 43-DOF G1 while
    _build_params.sh hard-rejected anything over 32 — advice impossible to follow.
    """
    bp = _load_bp()
    with pytest.raises(ValueError) as e:
        bp.BuildParams(max_joints=64, max_act=16, lib_name="x.so").check(n_act=43)
    m = re.search(r"--max-act (\d+)", str(e.value))
    assert m, f"no --max-act recommendation in: {e.value}"
    rec = int(m.group(1))
    assert 43 <= rec <= bp.MAX_ACT_CEILING, f"recommended --max-act {rec} is unusable"
    r = subprocess.run(
        ["bash", "-c", f'source "{SH}"; parse_build_params "$@"', "--",
         "--max-act", str(rec)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"recommended --max-act {rec} is rejected: {r.stderr!r}"


def test_check_rejects_past_the_ceiling_without_suggesting_a_rebuild():
    """Past MAX_ACT_CEILING there is no build that works — don't imply there is."""
    bp = _load_bp()
    n_act = bp.MAX_ACT_CEILING + 1
    with pytest.raises(ValueError) as e:
        bp.BuildParams(max_joints=64, max_act=64, lib_name="x.so").check(n_act=n_act)
    msg = str(e.value)
    assert "NOT a rebuild away" in msg
    # Must not hand over a runnable rebuild command: no --max-act value would work, and
    # the build script would reject every one of them. Saying "no --max-act value
    # supports it" is fine; `build_all.sh --max-act N` is not.
    assert not re.search(r"build_all\.sh\s+--max-act", msg), (
        f"a >{bp.MAX_ACT_CEILING} DOF robot cannot be fixed by any rebuild; the message "
        f"must not recommend one. Got: {msg}"
    )


def test_check_accepts_high_dof_on_a_high_dof_build():
    """A 43-DOF G1 must pass on a --max-act 48 build (block tier)."""
    bp = _load_bp()
    bp.BuildParams(max_joints=64, max_act=48, lib_name="x.so").check(n_act=43, n_joints=52)


# ── 3: built .so self-reports, and the check rejects oversized robots ────────

_LIBS = sorted(KERNELS.glob("*/*_lib.so"))
# Only kernels that include _build_params.cuh export the accessors; collision/robogpu
# do not use the limits at all.
_EXPECT_EXPORT = {"_ls_ik_cuda_lib.so", "_sqp_ik_cuda_lib.so", "_hjcd_ik_cuda_lib.so",
                  "_mppi_ik_cuda_lib.so", "_hit_and_run_ik_cuda_lib.so",
                  "_brownian_motion_ik_cuda_lib.so", "_svgd_region_ik_cuda_lib.so",
                  "_chomp_trajopt_cuda_lib.so", "_ls_trajopt_cuda_lib.so",
                  "_sco_trajopt_cuda_lib.so", "_stomp_trajopt_cuda_lib.so",
                  "_fk_cuda_lib.so"}


@pytest.mark.parametrize("lib", [p for p in _LIBS if p.name in _EXPECT_EXPORT],
                         ids=lambda p: p.name)
def test_so_reports_its_build_params(lib):
    bp = _load_bp()
    p = bp.read_build_params(ctypes.CDLL(str(lib)), lib.name)
    assert p.max_joints == _sh_default("PYROFFI_MAX_JOINTS_DEFAULT"), (
        f"{lib.name} was built with MAX_JOINTS={p.max_joints}, but the current default "
        f"is {_sh_default('PYROFFI_MAX_JOINTS_DEFAULT')}. Stale .so — rebuild."
    )
    assert p.max_act == _sh_default("PYROFFI_MAX_ACT_DEFAULT")


def test_capacity_check_rejects_oversized_robot():
    bp = _load_bp()
    p = bp.BuildParams(max_joints=64, max_act=16, lib_name="_test_lib.so")
    p.check(n_joints=64, n_act=16, kernel="k")          # exactly at capacity: fine
    with pytest.raises(ValueError, match="MAX_ACT=16"):
        p.check(n_joints=8, n_act=17, kernel="k")
    with pytest.raises(ValueError, match="MAX_JOINTS=64"):
        p.check(n_joints=65, n_act=8, kernel="k")


def test_missing_accessors_is_an_error_not_an_assumption():
    """A library without the exports must raise, never silently assume defaults."""
    bp = _load_bp()

    class _NoExports:
        def __getattr__(self, name):
            raise AttributeError(name)

    with pytest.raises(RuntimeError, match="does not export"):
        bp.read_build_params(_NoExports(), "_old_lib.so")
