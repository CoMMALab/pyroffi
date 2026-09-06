"""E2: what a fit costs as the cost dimension K grows, on the robot.

The per-outer-step solve count is the thing that separates the methods:

    implicit / unrolled   1 solve per context      independent of K
    fd                    K+1 solves per context   linear in K
    cmaes                 ~4 + 3 ln K per context  per generation

so the claim is that differentiating the solver makes the *cost* of a fit
independent of the number of cost parameters, while leaving its *quality*
unchanged.  Both halves matter: a faster method that recovers a worse cost has
not won anything, so every run reports weight error and regret next to the solve
count.

K is grown with the k3/k9/k16 bases from `ioc.robot.bases`, which add per-joint
weights.  The trajectory dimension and the landscape geometry are unchanged as K
grows, so the sweep isolates cost dimension rather than problem difficulty --
see `ioc.bench2d.problems.segment_residuals` for the same argument in 2D, and
for why the Gaussian-field benchmark cannot play this role.

    XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=0 \
        python -m ioc.robot.e2_scaling
"""

import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from ioc import analytic, outer as outer_opt
from ioc.inner import make_inner_solver
from ioc.robot import bases as cost_bases, problem as prob
from pyroffi.optimization_engines import DynamicsTrajOptConfig, dynamics_trajopt


def make_dynamics_forward_solver(opt_cfg=DynamicsTrajOptConfig()):
    """Wrap pyroffi's dynamics-aware L-BFGS engine as an `inner.forward_solver`."""

    def forward_solver(x0, cost_fn):
        return dynamics_trajopt(x0, cost_fn, opt_cfg)

    return forward_solver


def make_theta_star(K, rng, alpha=0.6, min_weight=0.01):
    """A spread-out point of the simplex, drawn Dirichlet(alpha).

    Sampling `uniform(0.5, 1.5)` and normalizing concentrates near the uniform
    simplex point as K grows, and a random initialization is near-uniform too --
    so the `random` baseline scores artificially well on ||theta_hat - theta*||_1
    and the metric stops discriminating at large K.  A Dirichlet with alpha < 1
    keeps the weights genuinely spread, so L1 measures recovery rather than
    proximity to the simplex centre.  `min_weight` avoids exactly-zero weights,
    whose features would be unidentifiable by construction.
    """
    w = np.maximum(rng.dirichlet(np.full(K, alpha)), min_weight)
    return jnp.asarray(w / w.sum())


def run_basis(
    problem, basis, seed, n_contexts, demo_noise, n_newton, damping, n_outer_steps,
    lr, fd_eps, n_unroll_tail, adjoint_ridge, conv_tol, run_baselines=True,
):
    residual_fn, names = cost_bases.kinematic(problem, basis)
    K = len(names)
    rng = np.random.default_rng(seed)
    theta_star = make_theta_star(K, rng)

    del damping  # no longer meaningful: the internal GN loop it damped is gone
    forward_solver = make_dynamics_forward_solver(
        DynamicsTrajOptConfig(n_iters=n_newton, grad_tol=min(conv_tol, 1e-6)))
    unrolled_forward_solver = make_dynamics_forward_solver(
        DynamicsTrajOptConfig(n_iters=n_newton, early_stop=False, unroll_tail=n_unroll_tail))

    def build(scales):
        return make_inner_solver(
            residual_fn, scales, adjoint_ridge=adjoint_ridge,
            forward_solver=forward_solver,
            unrolled_forward_solver=unrolled_forward_solver,
        )

    pool = problem.sample_scenes(rng, n_contexts * 3)
    scales = problem.calibrate(residual_fn, pool, jax.random.key(seed))
    scenes, _, _ = prob.screen_scenes(
        problem, pool, build(scales).stationarity, theta_star, conv_tol, n_contexts
    )
    inner = build(scales)

    x0s, _, demos = prob.make_demos(
        problem, inner.solve_implicit, scenes, theta_star, rng, demo_noise
    )

    z0 = jnp.asarray(rng.normal(scale=0.5, size=K))
    results = {}

    def record(name, z, wall, solves):
        m = prob.evaluate(
            problem, z, jax.jit(inner.solve_implicit), inner.cost, scenes, demos,
            x0s, theta_star,
        )
        m.update({"wall_s": wall, "n_solves": solves, "K": K})
        m.pop("theta_hat", None)
        results[name] = m
        print(f"    {basis:4s} K={K:2d} {name:9s} l1={m['theta_l1']:.3f} "
              f"cos={m['theta_cos']:.4f} regret={m['regret']:.3e} "
              f"solves={solves} {wall:.1f}s")

    loss_i = prob.make_outer(problem, inner.solve_implicit, scenes, demos, x0s)
    gf = jax.jit(jax.value_and_grad(loss_i))
    gf(z0)[0].block_until_ready()
    t0 = time.perf_counter()
    z, _ = outer_opt.adam(
        gf, z0, lr=lr, n_steps=n_outer_steps, solves_per_step=n_contexts
    )
    record("implicit", z, time.perf_counter() - t0, n_outer_steps * n_contexts)

    if run_baselines:
        # FD and CMA-ES answer the *cost-scaling* question; they are irrelevant
        # to identifiability, and at large K x M they dominate runtime (FD alone
        # is K+1 solves per step).  Skip them when only recovery versus
        # demonstration count is being measured.
        loss_j = jax.jit(loss_i)
        loss_j(z0).block_until_ready()
        t0 = time.perf_counter()
        z, _ = outer_opt.adam(
            outer_opt.fd_grad_fn(loss_j, fd_eps), z0, lr=lr, n_steps=n_outer_steps,
            solves_per_step=(K + 1) * n_contexts,
        )
        record("fd", z, time.perf_counter() - t0,
               n_outer_steps * (K + 1) * n_contexts)

        t0 = time.perf_counter()
        z, hist = outer_opt.cma_es(
            loss_j, z0, n_gens=max(1, n_outer_steps // 4), sigma0=0.5, seed=seed
        )
        record("cmaes", z, time.perf_counter() - t0, hist[-1][0] * n_contexts)

    t0 = time.perf_counter()
    z = analytic.kkt_fit(inner.grad_x, scenes, demos, K)
    record("kkt", z, time.perf_counter() - t0, 0)

    t0 = time.perf_counter()
    z = analytic.eiv_fit(inner.grad_x, scenes, demos, K)
    record("eiv", z, time.perf_counter() - t0, 0)

    record("random", z0, 0.0, 0)
    return results


def main(
    urdf_path: str = "resources/panda/panda_spherized.urdf",
    srdf_path: str = "resources/panda/panda.srdf",
    mesh_dir: str = "resources/panda/meshes",
    bases: tuple[str, ...] = ("k3", "k9", "k16"),
    n_timesteps: int = 16,
    n_contexts: int = 10,
    n_seeds: int = 3,
    n_newton: int = 100,
    damping: float = 1e-2,
    n_outer_steps: int = 30,
    lr: float = 0.15,
    fd_eps: float = 1e-4,
    demo_noise: float = 0.02,
    n_unroll_tail: int = 4,
    adjoint_ridge: float = 1e-9,
    conv_tol: float = 1e-5,
    run_baselines: bool = True,
    out: str = "e2_results.json",
):
    problem = prob.RobotProblem.load(urdf_path, srdf_path, mesh_dir, n_timesteps)
    print(f"jax devices: {jax.devices()}  T={n_timesteps}  noise={demo_noise}")

    all_results = {}
    for basis in bases:
        for seed in range(n_seeds):
            print(f"[{basis} seed={seed}]")
            all_results[f"{basis}_s{seed}"] = run_basis(
                problem, basis, seed, n_contexts, demo_noise, n_newton, damping,
                n_outer_steps, lr, fd_eps, n_unroll_tail, adjoint_ridge, conv_tol,
                run_baselines,
            )

    with open(out, "w") as f:
        json.dump({"demo_noise": demo_noise, "results": all_results}, f, indent=2)
    print(f"\nwrote {out}")

    print("\n=== solves to convergence, median over seeds ===")
    shown = [m for m in ["implicit", "fd", "cmaes", "kkt", "eiv"]
             if m in all_results[f"{bases[0]}_s0"]]
    print(f"{'basis':>6s} {'K':>3s} " + " ".join(f"{m:>12s}" for m in shown))
    for basis in bases:
        K = all_results[f"{basis}_s0"]["implicit"]["K"]
        row = []
        for m in shown:
            v = [all_results[f"{basis}_s{s}"][m]["n_solves"] for s in range(n_seeds)]
            l1 = [all_results[f"{basis}_s{s}"][m]["theta_l1"] for s in range(n_seeds)]
            row.append(f"{np.median(v):6.0f}/{np.median(l1):.2f}")
        print(f"{basis:>6s} {K:>3d} " + " ".join(f"{r:>12s}" for r in row))
    print("(cells are median forward solves / median theta L1 error)")


if __name__ == "__main__":
    tyro.cli(main)
