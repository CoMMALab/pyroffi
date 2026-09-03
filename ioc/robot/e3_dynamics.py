"""E3: the price of a misspecified cost basis.

Demonstrations are generated under a *dynamic* cost: kinematic terms plus an
RNEA torque feature with a payload, so the demonstrated motion depends on mass,
inertia and gravity rather than geometry alone.  Two fits are then compared:

  full        the well-specified basis, which includes the torque feature
  kinematic   a misspecified basis that cannot express torque at all

The kinematic fit is scored honestly: its recovered weights are used to re-solve
with the *kinematic* cost, and that trajectory is then evaluated under the true
dynamic cost.  A kinematic method optimizes what it can express, and pays for
what it cannot -- the regret gap between the two fits is that price, and the
`random` bookend says whether the misspecified fit is even worth doing.

This experiment also exercises the implicit adjoint on the GRiD CUDA
inverse-dynamics FFI (`--torque-backend grid`, float32). `jax.hessian` runs
straight through it: `GRiDDynamics`'s analytic-gradient kernels carry their
own `custom_jvp` built from GRiD's `idsva_so` second-order kernel (see
`pyroffi.dynamics._grid_dynamics`), so the implicit adjoint's exact-Hessian
curvature (`ioc.inner`'s `adjoint_hessian="jax"`, the only mode) needs no
float64-JAX twin and no Gauss-Newton fallback -- same code path as the
pure-JAX torque backend.

    XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 \
        python -m ioc.robot.e3_dynamics
"""

import dataclasses
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from ioc import metrics, outer as outer_opt
from ioc.inner import make_inner_solver
from ioc.robot import bases, problem as prob
from pyroffi.optimization_engines import DynamicsTrajOptConfig, dynamics_trajopt


def make_dynamics_forward_solver(opt_cfg=DynamicsTrajOptConfig()):
    """Wrap pyroffi's dynamics-aware L-BFGS engine as an `inner.forward_solver`.

    `dynamics_trajopt(x0, cost_fn, opt_cfg)` already matches the
    `Callable[[x0, cost_fn], x]` signature `make_inner_solver` expects; this
    just partially applies `opt_cfg` so the result drops straight in.
    """

    def forward_solver(x0, cost_fn):
        return dynamics_trajopt(x0, cost_fn, opt_cfg)

    return forward_solver


def run_trial(
    problem, theta_star_full, n_contexts, seed, demo_noise, payload_kg, n_newton,
    n_outer_steps, lr, n_unroll_tail, adjoint_ridge, conv_tol, check_grads,
    torque_backend, forward_solver, unrolled_forward_solver,
):
    rng = np.random.default_rng(seed)
    payload = bases.make_payload(problem, payload_kg)
    res_full, full_names = bases.dynamic(problem, payload, torque_backend)
    res_kin, kin_names = bases.kinematic(problem, "k3")
    res_jax, _ = bases.dynamic(problem, payload, "jax")

    def build_full(scales):
        return make_inner_solver(
            res_full, scales, adjoint_ridge=adjoint_ridge,
            forward_solver=forward_solver,
            unrolled_forward_solver=unrolled_forward_solver,
        )

    # --- well-specified (dynamic) problem: generates the demonstrations -------
    pool = problem.sample_scenes(rng, n_contexts * 4)
    scales_full = problem.calibrate(res_full, pool, jax.random.key(seed))
    scenes, discard, gn_kept = prob.screen_scenes(
        problem, pool, build_full(scales_full).stationarity, theta_star_full,
        conv_tol, n_contexts,
    )
    inner_full = build_full(scales_full)

    x0s, x_star, demos = prob.make_demos(
        problem, inner_full.solve_implicit, scenes, theta_star_full, rng, demo_noise
    )

    out = {"scene_discard_rate": float(discard),
           "inner_grad_norm_med": float(np.median(gn_kept))}

    if check_grads:
        loss_i = jax.jit(
            prob.make_outer(problem, inner_full.solve_implicit, scenes, demos, x0s)
        )
        z_probe = jnp.asarray(rng.normal(scale=0.5, size=len(full_names)))
        gi = jax.grad(loss_i)(z_probe)

        # FD is a noise diagnostic here, not a reference; see the module docstring.
        gfd = outer_opt.fd_grad_fn(loss_i, 1e-4)(z_probe)[1]
        out["grad_cos_vs_fd"] = metrics.cosine(gi, gfd)

        scales_ref = problem.calibrate(res_jax, scenes, jax.random.key(seed))
        inner_ref = make_inner_solver(
            res_jax, scales_ref, adjoint_ridge=adjoint_ridge,
            forward_solver=forward_solver,
            unrolled_forward_solver=unrolled_forward_solver,
        )
        loss_ref = jax.jit(
            prob.make_outer(problem, inner_ref.solve_implicit, scenes, demos, x0s)
        )
        out["grad_cos_vs_float64_ref"] = metrics.cosine(gi, jax.grad(loss_ref)(z_probe))
        print(f"    grad check: cos_vs_float64_ref={out['grad_cos_vs_float64_ref']:.6f}"
              f"  cos_vs_fd={out['grad_cos_vs_fd']:.6f} (FD noise-limited)")

    # --- scoring under the true dynamic cost ---------------------------------
    ref = jax.vmap(lambda s, x: inner_full.cost(x, theta_star_full, s))(scenes, x_star)

    def true_regret(x_hat):
        c = jax.vmap(lambda s, x: inner_full.cost(x, theta_star_full, s))(scenes, x_hat)
        return float(jnp.mean(c - ref))

    def ee_err(x_hat):
        def one(scene, x, demo):
            p = problem.ee_positions(problem.unpack(x, scene))
            pd = problem.ee_positions(demo)
            return jnp.sqrt(jnp.mean(jnp.sum((p - pd) ** 2, axis=-1)))

        return float(jnp.mean(jax.vmap(one)(scenes, x_hat, demos)))

    def fit(inner, names, tag):
        loss = prob.make_outer(problem, inner.solve_implicit, scenes, demos, x0s)
        gf = jax.jit(jax.value_and_grad(loss))
        z0 = jnp.asarray(rng.normal(scale=0.5, size=len(names)))
        gf(z0)[0].block_until_ready()
        t0 = time.perf_counter()
        z, _ = outer_opt.adam(
            gf, z0, lr=lr, n_steps=n_outer_steps, solves_per_step=n_contexts
        )
        wall = time.perf_counter() - t0
        theta = jax.nn.softmax(z)
        x_hat = jax.vmap(lambda s, x: inner.solve_implicit(x, theta, s))(scenes, x0s)
        r = {
            "theta_hat": [float(t) for t in theta],
            "regret": true_regret(x_hat),
            "ee_rmse": ee_err(x_hat),
            "wall_s": wall,
            "K": len(names),
        }
        if tag == "full":
            r["theta_l1"] = float(jnp.sum(jnp.abs(theta - theta_star_full)))
        return r

    results = {"full": fit(inner_full, full_names, "full")}

    # --- fit B: misspecified kinematic basis ----------------------------------
    scales_kin = problem.calibrate(res_kin, scenes, jax.random.key(seed))
    inner_kin = make_inner_solver(
        res_kin, scales_kin, adjoint_ridge=adjoint_ridge,
        forward_solver=forward_solver,
        unrolled_forward_solver=unrolled_forward_solver,
    )
    results["kinematic"] = fit(inner_kin, kin_names, "kinematic")

    # --- bookends -------------------------------------------------------------
    results["oracle"] = {"regret": true_regret(x_star), "ee_rmse": ee_err(x_star)}
    z_rand = jnp.asarray(rng.normal(scale=0.5, size=len(full_names)))
    th_rand = jax.nn.softmax(z_rand)
    x_rand = jax.vmap(
        lambda s, x: inner_full.solve_implicit(x, th_rand, s)
    )(scenes, x0s)
    results["random"] = {
        "theta_l1": float(jnp.sum(jnp.abs(th_rand - theta_star_full))),
        "regret": true_regret(x_rand),
        "ee_rmse": ee_err(x_rand),
    }

    for k in ["full", "kinematic", "oracle", "random"]:
        r = results[k]
        print(f"    {k:10s} regret={r['regret']:.3e} ee_rmse={r['ee_rmse']:.4f}"
              + (f" l1={r['theta_l1']:.3f}" if "theta_l1" in r else ""))

    out["methods"] = results
    return out


def main(
    urdf_path: str = "resources/panda/panda_spherized.urdf",
    srdf_path: str = "resources/panda/panda.srdf",
    mesh_dir: str = "resources/panda/meshes",
    n_timesteps: int = 16,
    n_contexts: int = 5,
    n_seeds: int = 5,
    n_newton: int = 60,
    damping: float = 1e-2,
    n_outer_steps: int = 30,
    lr: float = 0.15,
    demo_noise: float = 0.02,
    payload_kg: float = 2.0,
    torque_weight: float = 0.4,
    n_unroll_tail: int = 2,
    adjoint_ridge: float = 1e-9,
    conv_tol: float = 1e-5,
    check_grads: bool = True,
    torque_backend: str = "grid",
    dynamics_n_iters: int = 200,
    dynamics_m_lbfgs: int = 8,
    out: str = "e3_results.json",
):
    """The forward solve that finds x* always runs on pyroffi's contact-free
    `dynamics_trajopt` L-BFGS engine (see `ioc.inner`); the implicit adjoint is
    unaffected by that choice.
    """
    problem = prob.RobotProblem.load(urdf_path, srdf_path, mesh_dir, n_timesteps)
    print(f"jax devices: {jax.devices()}  T={n_timesteps}  payload={payload_kg}kg")

    # Torque must carry real weight or the dynamic and kinematic problems
    # coincide and the experiment measures nothing.
    rest = (1.0 - torque_weight) * np.array([0.5, 0.3, 0.2])
    theta_star = jnp.asarray(np.concatenate([rest, [torque_weight]]))
    print(f"theta* = {np.asarray(theta_star).round(3)}  {bases.DYNAMIC_NAMES}")

    dyn_opt_cfg = DynamicsTrajOptConfig(n_iters=dynamics_n_iters, m_lbfgs=dynamics_m_lbfgs)
    forward_solver = make_dynamics_forward_solver(dyn_opt_cfg)
    unrolled_forward_solver = make_dynamics_forward_solver(dataclasses.replace(
        dyn_opt_cfg, early_stop=False, unroll_tail=n_unroll_tail))

    all_results = {}
    for seed in range(n_seeds):
        print(f"[seed={seed}]")
        all_results[f"s{seed}"] = run_trial(
            problem, theta_star, n_contexts, seed, demo_noise, payload_kg,
            n_newton, n_outer_steps, lr, n_unroll_tail, adjoint_ridge,
            conv_tol, check_grads, torque_backend,
            forward_solver=forward_solver,
            unrolled_forward_solver=unrolled_forward_solver,
        )

    with open(out, "w") as f:
        json.dump({
            "theta_star": [float(t) for t in theta_star],
            "feature_names": list(bases.DYNAMIC_NAMES),
            "payload_kg": payload_kg,
            "demo_noise": demo_noise,
            "results": all_results,
        }, f, indent=2)
    print(f"\nwrote {out}")

    print("\n=== regret under the true dynamic cost (median over seeds) ===")
    for k in ["oracle", "full", "kinematic", "random"]:
        v = [all_results[f"s{s}"]["methods"][k]["regret"] for s in range(n_seeds)]
        e = [all_results[f"s{s}"]["methods"][k]["ee_rmse"] for s in range(n_seeds)]
        print(f"  {k:10s} regret={np.median(v):.3e}  ee_rmse={np.median(e):.4f}")
    print("\nThe full-vs-kinematic regret gap is the price of ignoring dynamics.")


if __name__ == "__main__":
    tyro.cli(main)
