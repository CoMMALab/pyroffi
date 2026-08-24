"""Fit every method on a 2D benchmark under a matched forward-solve budget.

The point of doing this in 2D is that a solve costs milliseconds, so the
comparison can be run the way it should be: every method gets the *same* number
of forward trajectory-optimization solves and is scored on what it did with
them.  That is the only fair currency, because the per-step costs differ by
construction (1 solve for a gradient step, K+1 for finite differences, a
population for CMA-ES), and it is what turns "the adjoint is cheaper" into a
measurable statement.

Each method returns a `(solves, best_loss)` trace, from which the headline
number -- solves to reach a target outer loss -- is read off directly.

    python -m ioc.bench2d.run --benchmark field --k-bumps 8
"""

import functools
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from ioc import analytic, outer as outer_opt
from ioc.inner import make_inner_solver
from ioc.bench2d import bases2d, problems as pb, robot2d
from pyroffi.optimization_engines import DynamicsTrajOptConfig, dynamics_trajopt


def make_dynamics_forward_solver(opt_cfg=DynamicsTrajOptConfig(n_iters=400)):
    """Wrap pyroffi's dynamics-aware L-BFGS engine as an `inner.forward_solver`."""

    def forward_solver(x0, cost_fn):
        return dynamics_trajopt(x0, cost_fn, opt_cfg)

    return forward_solver


def build_solver(res_fn, scales, T, d, cfg, n_iter, damping, unroll_tail, ridge,
                 n_restarts=1, topo_restarts=False, dynamics=False):
    """Adapt a benchmark's `res_fn(x, ctx, T, cfg)` to the shared inner solver.

    `topo_restarts` swaps the default i.i.d.-jitter multistart for the
    structured lateral-detour seeding in `pb.make_topo_seed_fn` -- only
    meaningful for `n_restarts > 1` on a 2D (d=2) benchmark.

    `dynamics=True` drives the benchmark through a synthetic GRiD-backed robot
    (`ioc.bench2d.robot2d`) instead of pure point-mass/unicycle kinematics:
    `res_fn`'s residuals are wrapped with `bases2d.dynamic` to append an RNEA
    torque feature.  The forward solve always runs on pyroffi's dynamics_trajopt
    L-BFGS engine, regardless of `dynamics` -- that flag only controls the
    torque-feature augmentation (and, via `adjoint_hessian`, whether the
    once-differentiable GRiD RNEA forces Gauss-Newton curvature). `unrolled`
    uses the same engine in its fixed-iteration, reverse-mode-differentiable
    form (`early_stop=False`) instead of `implicit`'s early-stopping form; the
    retired internal Gauss-Newton solver is gone entirely.
    """
    del damping  # no longer meaningful: the internal GN loop it damped is gone
    if dynamics:
        problem = robot2d.Robot2DProblem.load(d)
        residual_fn = bases2d.dynamic(problem, T, cfg, res_fn)
    else:
        residual_fn = functools.partial(_residuals, res_fn, T, cfg)
    forward_solver = make_dynamics_forward_solver(DynamicsTrajOptConfig(
        n_iters=n_iter, grad_tol=1e-9))
    unrolled_forward_solver = make_dynamics_forward_solver(DynamicsTrajOptConfig(
        n_iters=n_iter, early_stop=False, unroll_tail=unroll_tail))
    return make_inner_solver(
        residual_fn,
        scales,
        adjoint_ridge=ridge,
        n_restarts=n_restarts,
        restart_seed_fn=pb.make_topo_seed_fn(T, d) if topo_restarts else None,
        forward_solver=forward_solver,
        unrolled_forward_solver=unrolled_forward_solver,
        # The GRiD FFI supports one level of differentiation -- jax.hessian
        # raises on it (see ioc.robot.bases.dynamic) -- so the adjoint must use
        # the Gauss-Newton curvature whenever the torque feature is present.
        adjoint_hessian="gn" if dynamics else "true",
    )


def _calibrate_dynamics(residual_fn, ctxs, T, d, cfg, key, K, n_probe=12, jitter=0.25):
    """Like `pb.calibrate`, but for a `residual_fn(x, ctx)` already wrapped by
    `bases2d.dynamic` (K includes the appended torque feature)."""
    keys = jax.random.split(key, n_probe)

    def raw(ctx, k):
        x0 = pb.seed_path(ctx, T, d, cfg)
        x = x0 + jitter * jax.random.normal(k, x0.shape)
        rs = residual_fn(x, ctx)
        return jnp.stack([jnp.sum(r**2) for r in rs])

    vals = jax.vmap(jax.vmap(raw, in_axes=(None, 0)), in_axes=(0, None))(ctxs, keys)
    return jnp.maximum(jnp.mean(jnp.abs(vals.reshape(-1, K)), axis=0), 1e-8)


def _residuals(res_fn, T, cfg, x, ctx):
    return res_fn(x, ctx, T, cfg)


DEMO_STATIONARITY_TOL = 1e-6

# Inner iterations needed to drive the *demonstrations* to that tolerance.  These
# differ by an order of magnitude between benchmarks, which is exactly why a
# single shared default silently produced non-optimal demonstrations.  Only the
# one-off demonstration solve pays this; the bilevel fit uses `--n-iter`.
DEMO_N_ITER = {"racing": 400, "field": 2500, "unicycle": 4000, "segments": 6000}

# Extra headroom for the dynamics_trajopt L-BFGS forward solve (used whenever
# `dynamics=True`): its landscape is a harder RNEA-augmented cost than the
# kinematic default, so it benefits from a higher demo iteration count even
# though `dynamics_trajopt` now correctly returns the point with the smallest
# measured gradient rather than the lowest-cost line-search trial (see
# `pyroffi.optimization_engines._dynamics_trajopt`) -- that bug, not slow
# convergence, was the original cause of demos failing `_screen_demos`.
DEMO_N_ITER_DYNAMICS = {"field": 6000}


def relative_stationarity(features, grad_x, x, theta, ctx, K):
    """||sum_k th_k grad phi_k|| / sum_k th_k ||grad phi_k||, in [0, 1].

    The scale-free version of `inner.stationarity`.  The raw gradient norm is
    not comparable across benchmarks -- it carries the units of the cost, which
    whitening sets per benchmark -- so an absolute threshold is either vacuous
    on one benchmark or unreachable on another.  This ratio asks the question
    that actually matters: how completely do the weighted feature gradients
    cancel?  0 is an exact KKT point; 1 means they do not cancel at all.
    """
    e = jnp.eye(K)
    gs = jnp.stack([grad_x(x, e[k], ctx) for k in range(K)])
    num = jnp.linalg.norm(theta @ gs)
    return num / (jnp.sum(theta * jnp.linalg.norm(gs, axis=-1)) + 1e-300)


def _screen_demos(rel, benchmark, seed, n_iter):
    """Refuse to build a dataset on demonstrations that are not local optima.

    Every method here is scored on recovering the theta that *generated* the
    demonstrations, and that ground truth only means anything if the
    demonstration really is a minimizer of the corresponding cost.  Inverse KKT
    assumes it exactly (it fits grad_x J = 0 at the demonstration), so an
    unconverged demonstration silently handicaps one baseline and biases the
    weights recovered by all the others.

    This was not hypothetical.  At the shipped iteration counts the
    demonstrations sat at relative stationarity 1.1e-3 (`racing`), 2.0e-3
    (`field`) and 2.5e-2 (`unicycle`) -- the unicycle demonstrations were 2.5%
    away from being optimal for the cost they were labelled with.  The
    diagnostic was already being computed and stored as metadata, and simply
    never looked at; hence a hard check rather than a printed warning.
    """
    worst = float(jnp.max(rel))
    if not np.isfinite(worst) or worst > DEMO_STATIONARITY_TOL:
        n_bad = int(jnp.sum(rel > DEMO_STATIONARITY_TOL))
        raise RuntimeError(
            f"[{benchmark} seed {seed}] {n_bad}/{rel.shape[0]} demonstrations are "
            f"not local optima: max relative stationarity = {worst:.2e} > "
            f"{DEMO_STATIONARITY_TOL:.0e} after {n_iter} inner iterations. "
            "The recovered costs would not be valid; raise the demonstration "
            "iteration count (ioc.bench2d.run.DEMO_N_ITER)."
        )


def run(benchmark, n_contexts, n_seeds, T, n_iter, budget, k_bumps, demo_noise,
        damping, unroll_tail, ridge, fd_eps, lr, cfg, out, n_restarts=1,
        dynamics=False, topo_restarts=False):
    res_fn, _, d = pb.BENCHMARKS[benchmark]
    names = pb.benchmark_names(benchmark, k_bumps, cfg)
    if dynamics:
        # Demonstrated end to end on one benchmark; the pattern (wrap res_fn
        # with bases2d.dynamic, drive the forward solve through
        # dynamics_trajopt) generalizes trivially to the others once proven.
        assert benchmark == "field", "--dynamics is demonstrated on 'field' only"
        names = names + ("torque",)
    K = len(names)
    n_bump = k_bumps if benchmark == "field" else 1
    print(f"[{benchmark}] K={K} d={d} T={T} M={n_contexts} budget={budget} solves"
          + (" [dynamics]" if dynamics else ""))

    all_res = {}
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        theta_star = np.maximum(rng.dirichlet(np.full(K, 0.7)), 0.01)
        theta_star = jnp.asarray(theta_star / theta_star.sum())

        ctxs = pb.sample_contexts(rng, n_contexts, benchmark, T, d, n_bump, cfg)
        if dynamics:
            scales = _calibrate_dynamics(
                bases2d.dynamic(robot2d.Robot2DProblem.load(d), T, cfg, res_fn),
                ctxs, T, d, cfg, jax.random.key(seed), K,
            )
        else:
            scales = pb.calibrate(res_fn, ctxs, T, d, cfg, jax.random.key(seed), K)
        inner = build_solver(res_fn, scales, T, d, cfg, n_iter, damping, unroll_tail,
                             ridge, n_restarts, topo_restarts=topo_restarts,
                             dynamics=dynamics)

        x0s = jax.vmap(lambda c: pb.seed_path(c, T, d, cfg))(ctxs)

        # Demonstrations come from a *converged* solve, not from the same
        # truncated solver the learner uses.  This is both more correct (the
        # ground-truth theta labels a genuine local optimum) and more faithful
        # to the setting: the demonstrator is an expert, the learner's forward
        # model is an approximation of it.  Solved once per seed, so the high
        # iteration count costs nothing against the fitting budget.
        demo_counts = DEMO_N_ITER_DYNAMICS if dynamics else DEMO_N_ITER
        demo_iter = max(n_iter, demo_counts.get(benchmark, n_iter))
        demo_solver = build_solver(res_fn, scales, T, d, cfg, demo_iter, damping,
                                   unroll_tail, ridge, n_restarts, dynamics=dynamics)
        x_star = jax.vmap(
            lambda x, c: demo_solver.solve_implicit(x, theta_star, c))(x0s, ctxs)
        rel = jax.vmap(lambda x, c: relative_stationarity(
            demo_solver.features, demo_solver.grad_x, x, theta_star, c, K))(x_star, ctxs)
        conv = float(jnp.median(rel))
        _screen_demos(rel, benchmark, seed, demo_iter)
        demos = jax.vmap(lambda x, c: pb.unpack(x, c, T, d))(x_star, ctxs)
        if demo_noise > 0:
            nz = jnp.asarray(rng.normal(scale=demo_noise, size=demos.shape))
            demos = demos + nz.at[:, 0].set(0.0).at[:, -1].set(0.0)

        ref = jax.vmap(lambda x, c: inner.cost(x, theta_star, c))(x_star, ctxs)

        def outer_loss(solver):
            def loss(z):
                th = jax.nn.softmax(z)

                def one(c, dm, x0):
                    p = pb.unpack(solver(x0, th, c), c, T, d)[:, :2]
                    return jnp.mean(jnp.sum((p - dm[:, :2]) ** 2, axis=-1))

                return jnp.mean(jax.vmap(one)(ctxs, demos, x0s))

            return loss

        def score(z):
            """Weight error and regret under the *true* cost, plus the fitted paths."""
            th = jax.nn.softmax(z)
            xh = jax.vmap(lambda x, c: inner.solve_implicit(x, th, c))(x0s, ctxs)
            c = jax.vmap(lambda x, cc: inner.cost(x, theta_star, cc))(xh, ctxs)
            return (float(jnp.sum(jnp.abs(th - theta_star))),
                    float(jnp.mean(c - ref)), th, xh)

        # Every method is charged the same way: one "solve" is one context, times
        # the number of inner restarts, which restarts are charged for too.
        per_solve = n_contexts * n_restarts
        z0 = jnp.asarray(rng.normal(scale=0.5, size=K))
        R = {}

        def record(name, z, trace, wall, keep_theta=False):
            l1, rg, th, _ = score(z)
            R[name] = dict(l1=l1, regret=rg, wall=wall, trace=trace)
            if keep_theta:
                R[name]["theta"] = [float(v) for v in th]

        li = jax.jit(outer_loss(inner.solve_implicit))
        gi = jax.jit(jax.value_and_grad(outer_loss(inner.solve_implicit)))
        gi(z0)[0].block_until_ready()
        t0 = time.perf_counter()
        z, tr = outer_opt.adam(gi, z0, lr=lr, budget_solves=budget,
                               solves_per_step=per_solve, trace_best=True)
        record("implicit", z, tr, time.perf_counter() - t0, keep_theta=True)

        gu = jax.jit(jax.value_and_grad(outer_loss(inner.solve_unrolled)))
        gu(z0)[0].block_until_ready()
        t0 = time.perf_counter()
        z, tr = outer_opt.adam(gu, z0, lr=lr, budget_solves=budget,
                               solves_per_step=per_solve, trace_best=True)
        record("unrolled", z, tr, time.perf_counter() - t0)

        t0 = time.perf_counter()
        z, tr = outer_opt.adam(outer_opt.fd_grad_fn(li, fd_eps), z0, lr=lr,
                               budget_solves=budget,
                               solves_per_step=(K + 1) * per_solve, trace_best=True)
        record("fd", z, tr, time.perf_counter() - t0)

        t0 = time.perf_counter()
        z, tr = outer_opt.cma_es(li, z0, budget_solves=budget,
                                 solves_per_eval=per_solve, seed=seed, trace_best=True)
        record("cmaes", z, tr, time.perf_counter() - t0)

        t0 = time.perf_counter()
        z = analytic.kkt_fit(inner.grad_x, ctxs, demos, K, n_steps=600)
        record("kkt", z, [], time.perf_counter() - t0)

        t0 = time.perf_counter()
        z = analytic.cioc_fit(inner.grad_x, inner.gn_system, ctxs, demos, K,
                              n_steps=600)
        record("cioc", z, [], time.perf_counter() - t0, keep_theta=True)

        record("random", z0, [], 0.0)

        R["_meta"] = dict(K=K, demo_relative_stationarity=conv, demo_n_iter=demo_iter,
                          theta_star=[float(v) for v in theta_star])
        all_res[f"s{seed}"] = R
        print(f"  seed {seed}: " + "  ".join(
            f"{k}={R[k]['regret']:.2e}"
            for k in ["implicit", "fd", "cmaes", "kkt", "cioc", "random"]))

    with open(out, "w") as f:
        json.dump({"benchmark": benchmark, "K": K, "M": n_contexts,
                   "budget": budget, "demo_noise": demo_noise,
                   "results": all_res}, f, indent=2)
    print(f"wrote {out}")

    print(f"\n{'method':>10s} {'theta L1':>10s} {'regret':>12s} {'wall':>8s}")
    for m in ["implicit", "unrolled", "fd", "cmaes", "kkt", "cioc", "random"]:
        l1 = np.median([all_res[f"s{s}"][m]["l1"] for s in range(n_seeds)])
        rg = np.median([all_res[f"s{s}"][m]["regret"] for s in range(n_seeds)])
        w = np.median([all_res[f"s{s}"][m]["wall"] for s in range(n_seeds)])
        print(f"{m:>10s} {l1:>10.3f} {rg:>12.3e} {w:>7.1f}s")
    return all_res


def main(
    benchmark: str = "field",
    n_contexts: int = 8,
    n_seeds: int = 3,
    n_timesteps: int = 30,
    n_iter: int = 60,
    budget: int = 4000,
    k_bumps: int = 6,
    demo_noise: float = 0.02,
    damping: float = 1e-2,
    unroll_tail: int = 3,
    ridge: float = 1e-9,
    fd_eps: float = 1e-4,
    lr: float = 0.1,
    track_radius: float = 1.5,
    track_halfwidth: float = 0.42,
    n_obstacles: int = -1,  # -1: the benchmark's own default
    clearance: float = 0.20,
    nonhol_weight: float = 1.0,
    bump_width: float = 0.45,
    n_restarts: int = 1,
    topo_restarts: bool = False,
    k_segments: int = 2,
    dynamics: bool = False,
    out: str = "",
):
    """`dynamics` drives the benchmark through a synthetic GRiD-backed robot
    (`ioc.bench2d.robot2d`) instead of pure kinematics, appending an RNEA
    torque feature and running the forward solve on pyroffi's dynamics_trajopt
    L-BFGS engine.  Demonstrated on `field` only; see `run`'s docstring.

    `topo_restarts` swaps the default i.i.d.-jitter `n_restarts>1` multistart
    for `pb.make_topo_seed_fn`'s structured lateral-detour seeds (2D
    benchmarks only) -- the variant `fig_recovery` validated as actually
    escaping basins on a multimodal field, unlike plain jitter (see
    `ioc.inner.InnerSolver.solve`'s docstring on why jitter mostly doesn't).
    """
    cfg = pb.default_cfg(
        benchmark, track_radius=track_radius, track_halfwidth=track_halfwidth,
        clearance=clearance, nonhol_weight=nonhol_weight,
        bump_width=bump_width, k_segments=k_segments,
    )
    if n_obstacles > 0:
        cfg["n_obstacles"] = n_obstacles
    run(benchmark, n_contexts, n_seeds, n_timesteps, n_iter, budget, k_bumps,
        demo_noise, damping, unroll_tail, ridge, fd_eps, lr, cfg,
        out or f"bench2d_{benchmark}.json", n_restarts, dynamics,
        topo_restarts=topo_restarts)


if __name__ == "__main__":
    tyro.cli(main)
