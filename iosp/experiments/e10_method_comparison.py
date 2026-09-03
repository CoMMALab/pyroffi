"""E10 method comparison: CMA-ES, unrolled autodiff, implicit diff, and FD
on the 10 human teleop episodes, with 8/2 fit/held-out split.

Saves structured results (JSON + NPZ) for later visualization and tables.

Usage:
    python -m iosp.experiments.e10_method_comparison [--compile-timeout 1800]
"""
import argparse
import dataclasses
import json
import os
import pathlib
import signal
import time
import traceback

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from iosp import config
config.enable_compilation_cache()

from ioc import identifiability as ident
from ioc import outer as outer_opt
from iosp.fit.teleop import build_teleop, measure_standoffs, z_prior, TCP_OFFSET_M
from iosp.fit.parametric import _build_inner, screen_stationarity
from iosp.fit.params import z_scale
from iosp.model import fr3, pickplace as pp
from iosp.model.pickplace import split_trajopt as _split_trajopt
from ioc.robot.problem import Scene

OUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "results" / "e10_methods"
N_OUTER_STEPS = 40
LR = 0.05
FD_EPS = 1e-3
CMA_BUDGET_SOLVES = 500


def _metrics(built, u):
    return dict(
        joint_rmse_fit=built["rmse_a"](u),
        joint_rmse_gen=built["rmse_b"](u),
        ee_rmse_fit=built["ee_rmse_a"](u),
        ee_rmse_gen=built["ee_rmse_b"](u),
        loss=float(built["gf"](u)[0]),
    )


def _theta_dict(built, u):
    theta = built["theta_of"](u)
    return {n: float(v) for n, v in zip(built["names"], theta)}


def _gram(built, u):
    t0 = time.perf_counter()
    eigvals, eigvecs = ident.sensitivity_spectrum(built["jac_fn"], u)
    retained, discarded, r = ident.select_rank(eigvals, rule="gap")
    t_gram = time.perf_counter() - t0
    return dict(
        eigvals=eigvals.tolist(),
        rank=int(r),
        retained=[int(i) for i in retained],
        discarded=[int(i) for i in discarded],
        wall_gram_s=t_gram,
    )


def run_implicit(built, u0):
    print("\n=== Implicit diff ===", flush=True)
    gf = built["gf"]

    # Warm up (first call triggers XLA compile)
    t_compile_start = time.perf_counter()
    _ = gf(u0)
    t_compile = time.perf_counter() - t_compile_start
    print(f"  compile: {t_compile:.1f}s", flush=True)

    t0 = time.perf_counter()
    u_hat, trace = ident.wide_fit(gf, u0, lr=LR, n_steps=N_OUTER_STEPS)
    t_infer = time.perf_counter() - t0
    print(f"  fit: {t_infer:.1f}s  loss {trace[0][1]:.4f} → {trace[-1][1]:.4f}", flush=True)

    return dict(
        method="implicit",
        u_hat=np.asarray(u_hat),
        trace=trace,
        wall_compile_s=t_compile,
        wall_infer_s=t_infer,
        wall_total_s=t_compile + t_infer,
    )


def run_fd(built, u0):
    print("\n=== Finite differences ===", flush=True)
    K = built["K"]
    loss_fn = lambda u: built["gf"](u)[0]

    fd_gf = outer_opt.fd_grad_fn(loss_fn, FD_EPS, batched=False)

    t_compile_start = time.perf_counter()
    _ = fd_gf(u0)
    t_compile = time.perf_counter() - t_compile_start
    print(f"  compile: {t_compile:.1f}s", flush=True)

    t0 = time.perf_counter()
    u_hat, trace = outer_opt.adam(fd_gf, u0, lr=LR, n_steps=N_OUTER_STEPS,
                                  solves_per_step=K + 1)
    t_infer = time.perf_counter() - t0
    print(f"  fit: {t_infer:.1f}s  loss {trace[0][1]:.4f} → {trace[-1][1]:.4f}", flush=True)

    return dict(
        method="fd",
        u_hat=np.asarray(u_hat),
        trace=trace,
        wall_compile_s=t_compile,
        wall_infer_s=t_infer,
        wall_total_s=t_compile + t_infer,
    )


def run_cmaes(built, u0):
    print("\n=== CMA-ES ===", flush=True)
    loss_fn = jax.jit(lambda u: built["gf"](u)[0])

    t_compile_start = time.perf_counter()
    _ = loss_fn(u0)
    t_compile = time.perf_counter() - t_compile_start
    print(f"  compile: {t_compile:.1f}s", flush=True)

    t0 = time.perf_counter()
    u_hat, trace = outer_opt.cma_es(
        lambda u: float(loss_fn(u)),
        np.asarray(u0),
        sigma0=0.5,
        budget_solves=CMA_BUDGET_SOLVES,
        seed=0,
        batched_eval=False,
    )
    t_infer = time.perf_counter() - t0
    u_hat = jnp.asarray(u_hat, dtype=jnp.float32)
    print(f"  fit: {t_infer:.1f}s  loss {trace[0][1]:.4f} → {trace[-1][1]:.4f}", flush=True)

    return dict(
        method="cmaes",
        u_hat=np.asarray(u_hat),
        trace=trace,
        wall_compile_s=t_compile,
        wall_infer_s=t_infer,
        wall_total_s=t_compile + t_infer,
    )


def run_unrolled(built, u0, compile_timeout=1800):
    """Unrolled autodiff through the last `unroll_tail` solver iterations.

    Requires rebuilding the inner solvers with a differentiable forward solver.
    If XLA compilation exceeds `compile_timeout` seconds, returns inf metrics.
    """
    print("\n=== Unrolled autodiff ===", flush=True)
    print(f"  compile timeout: {compile_timeout}s", flush=True)

    try:
        from pyroffi.optimization_engines import DynamicsTrajOptConfig, dynamics_trajopt
    except ImportError:
        print("  SKIP: pyroffi optimization_engines not available", flush=True)
        return _failed_result("unrolled", "import_error")

    # Build a differentiable forward solver with unroll_tail > 0
    unroll_tail = 10
    opt_cfg = DynamicsTrajOptConfig(
        n_iters=60, early_stop=False, unroll_tail=unroll_tail,
        soft_line_search=True, soft_curvature_gate=True,
    )
    unrolled_fwd = lambda x0, cost_fn: dynamics_trajopt(x0, cost_fn, opt_cfg)

    # Rebuild inner solvers with the unrolled forward solver
    prob = built["prob"]
    scenes = built["scenes"]
    fit_idx = built["fit_idx"]
    fit_scenes = jax.tree.map(lambda a: a[fit_idx], scenes)
    K = built["K"]
    S = z_scale(K, pp.K_IK)
    standoffs = built["standoff_prior"]
    P = z_prior(K, pp.K_IK, standoffs)
    z_of = lambda u: P + S * u

    # Build inner solvers: stock forward solver for x*, unrolled for differentiation.
    # _build_inner handles per-phase Scene construction and calibration correctly.
    stock_fwd = pp.make_composed_forward_solver(n_iters=60)
    theta_ik_init = z_of(jnp.zeros(K))[:pp.K_IK]
    x0_cal, phase_scenes_cal, _, _ = prob.seeds(fit_scenes, theta_ik_init)

    from ioc.inner import make_inner_solver
    inner_unrolled = {}
    for phase in pp.PHASES:
        residual_fn, _ = prob.make_segment_inner(phase, stock_fwd)
        cal_scales = prob.calibrate_segment(phase, residual_fn,
                                            phase_scenes_cal[phase],
                                            jax.random.PRNGKey(0))
        inner_unrolled[phase] = make_inner_solver(
            residual_fn, cal_scales,
            forward_solver=stock_fwd,
            unrolled_forward_solver=unrolled_fwd,
        )

    demo = built["demo_paths"]

    def _rollout_unrolled(u):
        z = z_of(u)
        theta_ik, z_traj = z[:pp.K_IK], z[pp.K_IK:]
        x0, phase_sc, _, _ = prob.seeds(scenes, theta_ik)
        by_phase = _split_trajopt(jax.nn.softmax(z_traj))
        xs = {}
        for phase in pp.PHASES:
            xs[phase] = jax.vmap(inner_unrolled[phase].solve_unrolled,
                                  in_axes=(0, None, 0))(
                x0[phase], by_phase[phase], phase_sc[phase])
        return xs, phase_sc

    def loss_unrolled(u):
        xs, phase_sc = _rollout_unrolled(u)
        paths = prob.full_joint_paths(scenes, xs, phase_sc)
        return jnp.mean(jnp.sum((paths[fit_idx] - demo[fit_idx]) ** 2, axis=-1))

    gf_unrolled = jax.jit(jax.value_and_grad(loss_unrolled))

    # Try to compile with a timeout
    class CompileTimeout(Exception):
        pass

    def _alarm_handler(signum, frame):
        raise CompileTimeout()

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(compile_timeout)
    try:
        t_compile_start = time.perf_counter()
        _ = gf_unrolled(u0)
        t_compile = time.perf_counter() - t_compile_start
        signal.alarm(0)
    except CompileTimeout:
        signal.signal(signal.SIGALRM, old_handler)
        print(f"  TIMEOUT: XLA compile exceeded {compile_timeout}s", flush=True)
        return _failed_result("unrolled", "compile_timeout")
    except Exception as e:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        print(f"  ERROR: {e}", flush=True)
        traceback.print_exc()
        return _failed_result("unrolled", f"error: {e}")
    finally:
        signal.signal(signal.SIGALRM, old_handler)

    print(f"  compile: {t_compile:.1f}s", flush=True)

    t0 = time.perf_counter()
    u_hat, trace = ident.wide_fit(gf_unrolled, u0, lr=LR, n_steps=N_OUTER_STEPS)
    t_infer = time.perf_counter() - t0
    print(f"  fit: {t_infer:.1f}s  loss {trace[0][1]:.4f} → {trace[-1][1]:.4f}", flush=True)

    return dict(
        method="unrolled",
        u_hat=np.asarray(u_hat),
        trace=trace,
        wall_compile_s=t_compile,
        wall_infer_s=t_infer,
        wall_total_s=t_compile + t_infer,
    )


def _failed_result(method, reason):
    return dict(
        method=method,
        u_hat=None,
        trace=[],
        wall_compile_s=float("inf"),
        wall_infer_s=float("inf"),
        wall_total_s=float("inf"),
        failure=reason,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-timeout", type=int, default=1800,
                        help="max seconds for unrolled XLA compile (default 1800)")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--methods", type=str, default="implicit,fd,cmaes,unrolled",
                        help="comma-separated list of methods to run")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Building teleop forward map...", flush=True)
    built = build_teleop(n_iters=60, space="joint")
    K = built["K"]
    u0 = jnp.zeros(K, dtype=jnp.float32)

    print(f"\n{len(built['episodes'])} episodes: {built['n_fit']} fit / "
          f"{len(built['gen_idx'])} held out")

    # Baseline (random init)
    init_metrics = _metrics(built, u0)
    print(f"\nBaseline (u=0): loss={init_metrics['loss']:.4f}  "
          f"ee_fit={init_metrics['ee_rmse_fit']:.4f}m  "
          f"ee_gen={init_metrics['ee_rmse_gen']:.4f}m")

    # Run methods
    results = {}
    run_methods = [m.strip() for m in args.methods.split(",")]
    runners = {
        "implicit": lambda: run_implicit(built, u0),
        "fd": lambda: run_fd(built, u0),
        "cmaes": lambda: run_cmaes(built, u0),
        "unrolled": lambda: run_unrolled(built, u0, compile_timeout=args.compile_timeout),
    }

    for name in run_methods:
        if name not in runners:
            print(f"Unknown method {name}, skipping", flush=True)
            continue
        r = runners[name]()
        if r["u_hat"] is not None:
            r["metrics"] = _metrics(built, jnp.asarray(r["u_hat"]))
            r["theta"] = _theta_dict(built, jnp.asarray(r["u_hat"]))
        else:
            r["metrics"] = {k: float("inf") for k in init_metrics}
            r["theta"] = {}
        results[name] = r

    # Gram eigendecomposition at each fitted point
    print("\n=== Gram eigendecomposition ===", flush=True)
    for name, r in results.items():
        if r["u_hat"] is not None:
            print(f"  {name}...", flush=True)
            r["gram"] = _gram(built, jnp.asarray(r["u_hat"]))
            print(f"    rank={r['gram']['rank']}  "
                  f"top3 eigvals={r['gram']['eigvals'][:3]}", flush=True)
        else:
            r["gram"] = None

    # Save paths for visualization
    print("\n=== Saving paths ===", flush=True)
    path_data = {"demo": np.asarray(built["ee_demo_paths"])}
    for name, r in results.items():
        if r["u_hat"] is not None:
            path_data[name] = np.asarray(built["ee_paths_fn"](jnp.asarray(r["u_hat"])))
    np.savez_compressed(out / "paths.npz", **path_data)

    # Save joint paths too
    joint_path_data = {"demo": np.asarray(built["demo_paths"])}
    for name, r in results.items():
        if r["u_hat"] is not None:
            joint_path_data[name] = np.asarray(built["paths_fn"](jnp.asarray(r["u_hat"])))
    np.savez_compressed(out / "joint_paths.npz", **joint_path_data)

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Method':<12} {'Loss':>8} {'EE fit':>8} {'EE gen':>8} "
          f"{'J fit':>8} {'J gen':>8} {'Compile':>10} {'Infer':>10} {'Total':>10}")
    print("-" * 80)
    print(f"{'init':<12} {init_metrics['loss']:8.4f} "
          f"{init_metrics['ee_rmse_fit']:8.4f} {init_metrics['ee_rmse_gen']:8.4f} "
          f"{init_metrics['joint_rmse_fit']:8.4f} {init_metrics['joint_rmse_gen']:8.4f} "
          f"{'--':>10} {'--':>10} {'--':>10}")
    for name in run_methods:
        r = results[name]
        m = r["metrics"]
        def _fmt_time(t):
            return f"{t:.1f}s" if np.isfinite(t) else "inf"
        print(f"{name:<12} {m['loss']:8.4f} "
              f"{m['ee_rmse_fit']:8.4f} {m['ee_rmse_gen']:8.4f} "
              f"{m['joint_rmse_fit']:8.4f} {m['joint_rmse_gen']:8.4f} "
              f"{_fmt_time(r['wall_compile_s']):>10} "
              f"{_fmt_time(r['wall_infer_s']):>10} "
              f"{_fmt_time(r['wall_total_s']):>10}")
    print("=" * 80)

    # Save JSON summary
    summary = {
        "init_metrics": init_metrics,
        "episodes": built["episodes"],
        "n_fit": int(built["n_fit"]),
        "K": K,
        "n_outer_steps": N_OUTER_STEPS,
        "lr": LR,
        "fd_eps": FD_EPS,
        "cma_budget_solves": CMA_BUDGET_SOLVES,
        "names": built["names"],
        "methods": {},
    }
    for name, r in results.items():
        entry = {
            "metrics": r["metrics"],
            "theta": r.get("theta", {}),
            "wall_compile_s": r["wall_compile_s"],
            "wall_infer_s": r["wall_infer_s"],
            "wall_total_s": r["wall_total_s"],
            "trace": r["trace"],
            "gram": r.get("gram"),
        }
        if "failure" in r:
            entry["failure"] = r["failure"]
        summary["methods"][name] = entry
    def _json_default(o):
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return None

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    # Save u_hat arrays
    u_hats = {}
    for name, r in results.items():
        if r["u_hat"] is not None:
            u_hats[name] = r["u_hat"]
    np.savez(out / "u_hats.npz", **u_hats)

    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
