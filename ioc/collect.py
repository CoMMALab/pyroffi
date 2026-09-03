"""Data collection for the IOC study: reproduces every result in `data/`.

Each stage records the exact configuration used for the numbers reported, so a
result can be regenerated without reconstructing flags from memory.  Stages are
independent and selectable; all of them write JSON into `data/`.

    python -m ioc.collect --stages bench2d_scale        # one stage
    python -m ioc.collect --list                        # show the plan
    python -m ioc.collect --stages all --dry-run        # print commands

Every stage runs on a single GPU (device 0) and is sequential by design: two
JAX processes sharing a device caused OOM failures that silently killed earlier
sweeps.  Total runtime for `all` is on the order of a day.
"""

import dataclasses
import os
import shlex
import subprocess
import sys

import tyro

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")

E1 = "ioc.robot.e1_identifiability"
E2 = "ioc.robot.e2_scaling"
E3 = "ioc.robot.e3_dynamics"
B2D = "ioc.bench2d.run"


@dataclasses.dataclass
class Run:
    module: str
    args: list[str]
    out: str  # path relative to data/
    note: str
    env: dict[str, str] = dataclasses.field(default_factory=dict)


def _e1(extra, out, note):
    return Run(E1, ["--n-timesteps", "16", "--n-newton", "60", "--n-contexts", "5",
                    "--n-outer-steps", "30"] + extra, out, note)


def _b2d(extra, out, note):
    return Run(B2D, extra, out, note)


STAGES: dict[str, list[Run]] = {
    # ---- robot experiments -------------------------------------------------
    "e1_noise": [
        # n_restarts=7 matches fig5_noise_field's structured-restart count, so
        # both noise-sweep figures are controlled for inner-problem
        # multimodality with the same restart budget (i.i.d. jitter here --
        # the 7-DOF robot can't use the 2D-only structured detour restarts).
        _e1(["--n-seeds", "5", "--demo-noise", s, "--n-restarts", "7"],
            f"robot/e1_sigma{t}.json", f"E1 demonstration-noise sweep, sigma={s}")
        for s, t in (("0.0", "00"), ("0.01", "001"), ("0.02", "002"), ("0.05", "005"))
    ],
    "e1_collinear": [
        _e1(["--n-seeds", "3", "--demo-noise", "0.02", "--collinear-control"],
            "robot/e1_collinear.json",
            "E1 non-identifiability control: two duplicated features")
    ],
    "e2_scaling": [
        Run(E2, ["--bases", b, "--n-timesteps", "16", "--n-newton", "60",
                 "--n-contexts", "5", "--n-outer-steps", "30", "--n-seeds", "3",
                 "--demo-noise", "0.02"], f"robot/e2b_{b}.json",
            f"E2 cost-dimension scaling on the robot, basis {b}")
        for b in ("k3", "k9", "k16")
    ],
    "e3_dynamics": [
        Run(E3, ["--n-timesteps", "16", "--n-contexts", "5", "--n-seeds", "5",
                 "--n-newton", "60", "--n-outer-steps", "30", "--demo-noise", "0.02",
                 "--payload-kg", "2.0", "--torque-backend", "grid"],
            "robot/e3_results.json",
            "E3 dynamics: GRiD forward solve, exact GRiD-analytic adjoint"),
        # End-to-end float32 (x64 off): the implicit adjoint's exact Hessian
        # now comes from GRiD's own idsva_so second-order kernel (see
        # ioc.inner / pyroffi.dynamics._grid_dynamics), so this no longer
        # needs the Gauss-Newton fallback -- it is the same code path as
        # e3_dynamics above, just run with x64 off end-to-end.
        Run(E3, ["--n-timesteps", "16", "--n-contexts", "5", "--n-seeds", "3",
                 "--n-newton", "60", "--n-outer-steps", "30", "--demo-noise", "0.02",
                 "--payload-kg", "2.0", "--torque-backend", "grid",
                 "--adjoint-ridge", "1e-6",
                 "--conv-tol", "1e-3", "--no-check-grads"], "robot/e3_float32.json",
            "E3 end-to-end float32 (conv_tol loosened: 1e-5 is unreachable there)",
            env={"JAX_ENABLE_X64": "0"}),
    ],
    # ---- 2D benchmarks -----------------------------------------------------
    # `--n-iter` is the *learner's* inner solver.  It is set so that inner-solver
    # truncation contributes well under the sigma=0.02 demonstration-noise floor
    # (8.0e-4 on the outer loss); at the previous value of 80 truncation alone
    # contributed 4.1x that floor on `field` and 1.8x on `unicycle`, so the fits
    # were being driven by solver error rather than by the demonstrations.  The
    # demonstrations themselves are solved to convergence separately -- see
    # `bench2d.run.DEMO_N_ITER` and the stationarity screen.
    "bench2d_main": [
        _b2d(["--benchmark", b, "--n-contexts", "8", "--n-seeds", "5",
              "--n-timesteps", "30", "--n-iter", "800", "--budget", "8000",
              "--k-bumps", "6"], f"bench2d/bench2d_{b}.json",
             f"2D benchmark {b}, matched 8000-solve budget")
        for b in ("field", "racing", "unicycle")
    ],
    "bench2d_scale": [
        _b2d(["--benchmark", "segments", "--k-segments", str(S),
              "--n-contexts", str(max(3 * S, 8)), "--n-seeds", "3",
              "--n-timesteps", "30", "--n-iter", "800",
              "--budget", str(2000 * (3 * S))],
             f"bench2d/bench2d_seg_K{3 * S}.json",
             f"Cost-dimension sweep, K={3 * S}. `segments` scales K through "
             "time-segmented quadratics (effort/smooth/clearance all per-segment) "
             "so landscape geometry is unchanged; the `field` benchmark cannot do "
             "this (K = number of bumps, so K and multimodality move together).")
        for S in (2, 4, 8, 16)
    ],
    "bench2d_regime": [
        _b2d(["--benchmark", "field", "--k-bumps", "6", "--n-contexts", "8",
              "--n-seeds", "5", "--n-timesteps", "30", "--n-iter", "800",
              "--budget", "32000", "--bump-width", bw, "--n-restarts", str(R)],
             f"bench2d/bench2d_regime_bw{bw}_R{R}.json",
             f"Regime study: bump width {bw} (multimodality), {R} inner restarts")
        for bw in ("0.45", "0.90") for R in (1, 4)
    ],
}


def main(
    stages: tuple[str, ...] = ("all",),
    dry_run: bool = False,
    list_only: bool = False,
    gpu: int = 0,
    python: str = sys.executable,
):
    names = list(STAGES) if "all" in stages else list(stages)
    unknown = [n for n in names if n not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; available: {list(STAGES)}")

    if list_only:
        for n in names:
            print(f"\n[{n}]")
            for r in STAGES[n]:
                print(f"  {r.out:42s} {r.note}")
        return

    base_env = dict(os.environ)
    base_env.update({
        "PYTHONUNBUFFERED": "1",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "JAX_ENABLE_X64": "1",
        "CUDA_VISIBLE_DEVICES": str(gpu),
    })
    os.makedirs(DATA, exist_ok=True)

    failures = []
    for n in names:
        print(f"\n=== stage {n} ===", flush=True)
        for r in STAGES[n]:
            out = os.path.join(DATA, r.out)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            cmd = [python, "-u", "-m", r.module, *r.args, "--out", out]
            print("  " + " ".join(shlex.quote(c) for c in cmd), flush=True)
            if dry_run:
                continue
            env = dict(base_env)
            env.update(r.env)
            rc = subprocess.call(cmd, cwd=ROOT, env=env)
            if rc != 0:
                print(f"  FAILED ({rc}): {r.out}", flush=True)
                failures.append(r.out)

    if failures:
        print(f"\n{len(failures)} run(s) failed: {failures}")
        raise SystemExit(1)
    print("\nall requested stages complete")


if __name__ == "__main__":
    tyro.cli(main)
