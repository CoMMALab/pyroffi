"""Ablation: does seeding the implicit fit's z0 from Inverse-KKT help?

`ioc.analytic.kkt_fit` costs zero forward solves and is exact at sigma=0, so
seeding the gradient method's starting point with it is free relative to the
budget everything else spends.  The question is whether that head start
survives (a) demonstration noise, where KKT itself degrades below random past
sigma=0.05 (see `ioc.analytic.kkt_fit`'s docstring), and (b) landscape
multimodality, where a stationarity-only fit has no way to know which basin
the demonstration actually lives in.

Isolates one variable against `ioc.bench2d.run`: implicit only (no fd/cmaes),
two inits for z0 -- the existing `rng.normal(scale=0.5)` draw vs.
`analytic.kkt_fit`'s solution -- everything else (contexts, demos, budget,
n_restarts) identical between the two arms so the trace comparison is
apples-to-apples.

Grid: `bump_width` in {0.45, 0.90} is the field benchmark's existing
multimodality regime knob (`ioc.bench2d.problems._sample_bumps`); crossed with
`n_restarts` in {1, 4} (structured topo multistart, see
`ioc.bench2d.problems.make_topo_seed_fn`) to separate "bad landscape" from "bad
seed," and `demo_noise` in {0, 0.01, 0.02, 0.05} to catch the KKT crossover.

Defaults here (n_iter=150, budget=2000) are trimmed for iteration speed, not for
headline numbers: with budget_solves outer steps each paying for a full
n_iter-iteration forward Newton solve, wall time scales as
budget/per_solve * n_iter, dominating everything else in the grid (demo
generation is a one-off per seed at the higher DEMO_N_ITER count and is cheap
by comparison).  Pass --n-iter 800 --budget 8000 to reproduce the
`bench2d_main`-grade settings once a regime looks interesting.

    XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=<idx> \
        python -u -m ioc.bench2d.run_kkt_seed
"""

import itertools
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from ioc import analytic, outer as outer_opt
from ioc.bench2d import bases2d, problems as pb, robot2d
from ioc.bench2d.run import (
    build_solver, relative_stationarity, _screen_demos, DEMO_N_ITER,
    DEMO_N_ITER_DYNAMICS, _calibrate_dynamics,
)


def run_one(benchmark, n_contexts, n_seeds, T, n_iter, budget, k_bumps, demo_noise,
            damping, unroll_tail, ridge, fd_eps, lr, cfg, n_restarts, topo_restarts,
            dynamics=False):
    assert not dynamics or benchmark == "field", "--dynamics is demonstrated on 'field' only"
    res_fn, _, d = pb.BENCHMARKS[benchmark]
    names = pb.benchmark_names(benchmark, k_bumps, cfg)
    if dynamics:
        names = names + ("torque",)
    K = len(names)
    n_bump = k_bumps if benchmark == "field" else 1
    per_solve = n_contexts * n_restarts

    seeds_out = {}
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        theta_star = np.maximum(rng.dirichlet(np.full(K, 0.7)), 0.01)
        theta_star = jnp.asarray(theta_star / theta_star.sum())

        ctxs = pb.sample_contexts(rng, n_contexts, benchmark, T, d, n_bump, cfg)
        if dynamics:
            dyn_residual_fn = bases2d.dynamic(robot2d.Robot2DProblem.load(d), T, cfg, res_fn)
            scales = _calibrate_dynamics(
                dyn_residual_fn, ctxs, T, d, cfg, jax.random.key(seed), K)
        else:
            scales = pb.calibrate(res_fn, ctxs, T, d, cfg, jax.random.key(seed), K)
        inner = build_solver(res_fn, scales, T, d, cfg, n_iter, damping, unroll_tail,
                             ridge, n_restarts, topo_restarts=topo_restarts,
                             dynamics=dynamics)

        x0s = jax.vmap(lambda c: pb.seed_path(c, T, d, cfg))(ctxs)

        demo_counts = DEMO_N_ITER_DYNAMICS if dynamics else DEMO_N_ITER
        demo_iter = max(n_iter, demo_counts.get(benchmark, n_iter))
        demo_solver = build_solver(res_fn, scales, T, d, cfg, demo_iter,
                                   damping, unroll_tail, ridge, n_restarts,
                                   topo_restarts=topo_restarts, dynamics=dynamics)
        x_star = jax.vmap(
            lambda x, c: demo_solver.solve_implicit(x, theta_star, c))(x0s, ctxs)
        rel = jax.vmap(lambda x, c: relative_stationarity(
            demo_solver.features, demo_solver.grad_x, x, theta_star, c, K))(x_star, ctxs)
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
            th = jax.nn.softmax(z)
            xh = jax.vmap(lambda x, c: inner.solve_implicit(x, th, c))(x0s, ctxs)
            c = jax.vmap(lambda x, cc: inner.cost(x, theta_star, cc))(xh, ctxs)
            return float(jnp.sum(jnp.abs(th - theta_star))), float(jnp.mean(c - ref))

        z0_random = jnp.asarray(rng.normal(scale=0.5, size=K))
        z0_kkt = analytic.kkt_fit(inner.grad_x, ctxs, demos, K, n_steps=600)

        gi = jax.jit(jax.value_and_grad(outer_loss(inner.solve_implicit)))
        gi(z0_random)[0].block_until_ready()

        arm_results = {}
        for init_name, z0 in (("random", z0_random), ("kkt_seed", z0_kkt)):
            t0 = time.perf_counter()
            z, tr = outer_opt.adam(gi, z0, lr=lr, budget_solves=budget,
                                   solves_per_step=per_solve, trace_best=True)
            wall = time.perf_counter() - t0
            l1, regret = score(z)
            arm_results[init_name] = dict(
                l1=l1, regret=regret, wall=wall, trace=tr,
                z0_l1=float(jnp.sum(jnp.abs(jax.nn.softmax(z0) - theta_star))),
            )

        seeds_out[f"s{seed}"] = dict(
            arms=arm_results,
            demo_relative_stationarity=float(jnp.median(rel)),
            theta_star=[float(v) for v in theta_star],
        )
        r, k = arm_results["random"], arm_results["kkt_seed"]
        print(f"    seed={seed} random: l1={r['l1']:.3f} regret={r['regret']:.2e}  "
              f"kkt_seed: l1={k['l1']:.3f} regret={k['regret']:.2e} "
              f"(kkt z0 l1={k['z0_l1']:.3f})")

    return dict(K=K, results=seeds_out)


def main(
    n_contexts: int = 8,
    n_seeds: int = 5,
    n_timesteps: int = 30,
    n_iter: int = 150,
    budget: int = 2000,
    k_bumps: int = 6,
    damping: float = 1e-2,
    unroll_tail: int = 3,
    ridge: float = 1e-9,
    fd_eps: float = 1e-4,
    lr: float = 0.1,
    bump_widths: tuple[float, ...] = (0.45, 0.90),
    restart_counts: tuple[int, ...] = (1, 4),
    demo_noises: tuple[float, ...] = (0.0, 0.01, 0.02, 0.05),
    dynamics: bool = False,
    out: str = "bench2d_kkt_seed.json",
):
    """`dynamics=True` routes the forward solve through pyroffi's dynamics_trajopt
    L-BFGS engine (`ioc.bench2d.run.make_dynamics_forward_solver`) instead of the
    internal damped Gauss-Newton loop, appending an RNEA torque feature -- the
    same swap `ioc.bench2d.run --dynamics` demonstrates on `field`.
    """
    all_res = {}
    for bw, R, sigma in itertools.product(bump_widths, restart_counts, demo_noises):
        tag = f"bw{bw}_R{R}_sigma{sigma}"
        print(f"[field{'+dynamics' if dynamics else ''}] {tag}")
        cfg = pb.default_cfg("field", bump_width=bw)
        all_res[tag] = run_one(
            "field", n_contexts, n_seeds, n_timesteps, n_iter, budget, k_bumps,
            sigma, damping, unroll_tail, ridge, fd_eps, lr, cfg, R,
            topo_restarts=(R > 1), dynamics=dynamics,
        )

    with open(out, "w") as f:
        json.dump(all_res, f, indent=2)
    print(f"\nwrote {out}")

    print(f"\n{'regime':>22s} {'random L1':>10s} {'kkt L1':>10s} "
          f"{'random rg':>11s} {'kkt rg':>11s}")
    for tag, d in all_res.items():
        seeds = d["results"]
        rl1 = np.median([s["arms"]["random"]["l1"] for s in seeds.values()])
        kl1 = np.median([s["arms"]["kkt_seed"]["l1"] for s in seeds.values()])
        rrg = np.median([s["arms"]["random"]["regret"] for s in seeds.values()])
        krg = np.median([s["arms"]["kkt_seed"]["regret"] for s in seeds.values()])
        print(f"{tag:>22s} {rl1:>10.3f} {kl1:>10.3f} {rrg:>11.2e} {krg:>11.2e}")


if __name__ == "__main__":
    tyro.cli(main)
