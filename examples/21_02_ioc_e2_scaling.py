"""E2: how IOC cost scales with the number of cost parameters K.

The claim this experiment measures: the implicit adjoint needs one forward
trajectory-optimization solve per outer step regardless of K, while finite
differences need K+1 and derivative-free search needs a population that grows
with K.  So the advantage of differentiating through the optimizer is not a
constant factor -- it widens as the cost basis gets richer.

Cost bases (all linear in theta, all whitened, theta on the simplex):
  k3  : effort, collision, smooth                       (E1's basis)
  k9  : per-joint effort (7), collision, smooth
  k16 : per-joint effort (7), per-joint smooth (7), collision, posture

Reuses the E1 module wholesale -- same inner Gauss-Newton solver, same verified
implicit adjoint, same metrics -- by substituting the residual function and K.

    XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=0 \
        python examples/21_02_ioc_e2_scaling.py
"""

import importlib.util
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import tyro
import yourdfpy

import pyroffi as pk

_SPEC = importlib.util.spec_from_file_location(
    "e1", os.path.join(os.path.dirname(__file__), "21_00_ioc_e1_synthetic.py")
)
e1 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(e1)


# ---------------------------------------------------------------------------
# Cost bases of varying dimension
# ---------------------------------------------------------------------------


def make_basis(basis: str, dof: int, robot, robot_coll):
    """Return (residual_fn, feature_names) for the requested cost basis."""

    def clearance_residual(q, scene):
        coll = robot_coll.at_config(robot, q)
        d = (
            jnp.linalg.norm(coll.pose.translation() - scene.obs_center, axis=-1)
            - coll.radius
            - scene.obs_radius[0]
        )
        d = d.reshape(d.shape[0], -1)
        d_min = -e1.SOFTMIN_TAU * jax.scipy.special.logsumexp(
            -d / e1.SOFTMIN_TAU, axis=-1
        )
        return jax.nn.softplus(e1.SOFTNESS * (e1.CLEARANCE_MARGIN - d_min)) / e1.SOFTNESS

    def residuals(x_flat, scene, robot_, robot_coll_, dof_, collinear):
        q = e1.unpack(x_flat, scene, dof_)
        dq = q[1:] - q[:-1]
        ddq = q[2:] - 2.0 * q[1:-1] + q[:-2]
        r_coll = clearance_residual(q, scene)

        if basis == "k3":
            out = [dq.reshape(-1), r_coll, ddq.reshape(-1)]
        elif basis == "k9":
            # One effort weight per joint: the cost basis grows while the
            # trajectory dimension stays fixed, which is the regime where the
            # per-step solve count is the whole story.
            out = [dq[:, j] for j in range(dof_)] + [r_coll, ddq.reshape(-1)]
        elif basis == "k16":
            nominal = 0.5 * (scene.q_start + scene.q_goal)
            r_posture = (q - nominal).reshape(-1)
            out = (
                [dq[:, j] for j in range(dof_)]
                + [ddq[:, j] for j in range(dof_)]
                + [r_coll, r_posture]
            )
        else:
            raise ValueError(basis)
        return tuple(out)

    if basis == "k3":
        names = ["effort", "collision", "smooth"]
    elif basis == "k9":
        names = [f"effort_j{j}" for j in range(dof)] + ["collision", "smooth"]
    else:
        names = (
            [f"effort_j{j}" for j in range(dof)]
            + [f"smooth_j{j}" for j in range(dof)]
            + ["collision", "posture"]
        )
    return residuals, names


def make_theta_star(K, rng):
    """A non-degenerate point of the simplex: no vanishing and no dominant weight."""
    w = rng.uniform(0.5, 1.5, size=K)
    return jnp.asarray(w / w.sum())


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_basis(
    basis, robot, robot_coll, dof, ee_index, q_start, q_goal,
    n_contexts, seed, demo_noise, n_newton, damping, n_outer_steps, lr, fd_eps,
    n_unroll_tail, adjoint_ridge, conv_tol,
):
    residual_fn, names = make_basis(basis, dof, robot, robot_coll)
    K = len(names)

    # Substitute the basis into the E1 module: `features`, `make_inner` and the
    # metrics all read these globals, so the verified solver and adjoint are
    # reused unchanged.
    e1.residuals = residual_fn
    e1.K = K
    e1.FEATURE_NAMES = tuple(names)

    rng = np.random.default_rng(seed)
    theta_star = make_theta_star(K, rng)

    pool = e1.sample_scenes(rng, n_contexts * 3, q_start, q_goal, robot, robot_coll, dof)
    scales = e1.calibrate_scales(
        pool, robot, robot_coll, dof, False, jax.random.key(seed)
    )
    _, _, _, _, ign = e1.make_inner(
        robot, robot_coll, dof, scales, False, n_newton, damping,
        n_unroll_tail, adjoint_ridge,
    )
    x0_pool = jax.vmap(lambda s: e1.straight_line_seed(s, dof))(pool)
    gn = np.asarray(jax.vmap(lambda x, s: ign(x, theta_star, s))(x0_pool, pool))
    keep = np.flatnonzero(gn < conv_tol)[:n_contexts]
    if len(keep) < n_contexts:
        raise RuntimeError(f"{basis}: only {len(keep)}/{len(gn)} scenes converged")
    scenes = jax.tree.map(lambda a: a[keep], pool)

    cost, grad_x, solve_unrolled, solve_implicit, _ = e1.make_inner(
        robot, robot_coll, dof, scales, False, n_newton, damping,
        n_unroll_tail, adjoint_ridge,
    )
    seeds_x0 = jax.vmap(lambda s: e1.straight_line_seed(s, dof))(scenes)
    x_demo = jax.vmap(lambda x0, s: solve_implicit(x0, theta_star, s))(seeds_x0, scenes)
    demos = jax.vmap(lambda x, s: e1.unpack(x, s, dof))(x_demo, scenes)
    if demo_noise > 0:
        noise = jnp.asarray(rng.normal(scale=demo_noise, size=demos.shape))
        demos = demos + noise.at[:, 0].set(0.0).at[:, -1].set(0.0)

    z0 = jnp.asarray(rng.normal(scale=0.5, size=K))
    results = {}

    def record(name, z, wall, solves):
        m = e1.evaluate(
            z, jax.jit(solve_implicit), cost, robot, dof, ee_index,
            scenes, demos, seeds_x0, theta_star,
        )
        m.update({"wall_s": wall, "n_solves": solves, "K": K})
        m.pop("theta_hat", None)
        results[name] = m
        print(
            f"    {basis:4s} K={K:2d} {name:9s} l1={m['theta_l1']:.3f} "
            f"cos={m['theta_cos']:.4f} regret={m['regret']:.3e} "
            f"solves={solves} {wall:.1f}s"
        )

    loss_i = e1.make_outer(
        robot, robot_coll, dof, ee_index, solve_implicit, scenes, demos, seeds_x0
    )
    gf = jax.jit(jax.value_and_grad(loss_i))
    gf(z0)[0].block_until_ready()
    t0 = time.perf_counter()
    z, _ = e1.adam(gf, z0, n_outer_steps, lr, n_contexts)
    record("implicit", z, time.perf_counter() - t0, n_outer_steps * n_contexts)

    loss_j = jax.jit(loss_i)
    loss_j(z0).block_until_ready()
    t0 = time.perf_counter()
    z, _ = e1.adam(
        e1.fd_grad_fn(loss_j, fd_eps), z0, n_outer_steps, lr, (K + 1) * n_contexts
    )
    record("fd", z, time.perf_counter() - t0, n_outer_steps * (K + 1) * n_contexts)

    t0 = time.perf_counter()
    z, hist = e1.cma_es(loss_j, z0, max(1, n_outer_steps // 4), 0.5, seed)
    record("cmaes", z, time.perf_counter() - t0, hist[-1][0] * n_contexts)

    t0 = time.perf_counter()
    z, _ = e1.kkt_fit(grad_x, scenes, demos, dof)
    record("kkt", z, time.perf_counter() - t0, 0)

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
    out: str = "e2_results.json",
):
    e1.N_TIMESTEPS = n_timesteps
    print(f"jax devices: {jax.devices()}  T={n_timesteps}  noise={demo_noise}")

    urdf = yourdfpy.URDF.load(urdf_path, mesh_dir=mesh_dir)
    robot = pk.Robot.from_urdf(urdf)
    robot_coll = pk.collision.RobotCollisionSpherized.from_urdf(
        urdf, srdf_path=srdf_path
    )
    dof = robot.joints.num_actuated_joints
    ee_index = robot.links.names.index(e1.EE_LINK)
    q_start = np.array([0.0, -0.6, 0.0, -2.2, 0.0, 1.6, 0.8])[:dof]
    q_goal = np.array([0.9, -0.2, 0.0, -1.8, 0.0, 1.7, 0.8])[:dof]

    all_results = {}
    for basis in bases:
        for seed in range(n_seeds):
            print(f"[{basis} seed={seed}]")
            all_results[f"{basis}_s{seed}"] = run_basis(
                basis, robot, robot_coll, dof, ee_index, q_start, q_goal,
                n_contexts, seed, demo_noise, n_newton, damping, n_outer_steps,
                lr, fd_eps, n_unroll_tail, adjoint_ridge, conv_tol,
            )

    with open(out, "w") as f:
        json.dump({"demo_noise": demo_noise, "results": all_results}, f, indent=2)
    print(f"\nwrote {out}")

    print("\n=== solves to convergence, median over seeds ===")
    print(f"{'basis':>6s} {'K':>3s} " + " ".join(f"{m:>12s}" for m in
                                                 ["implicit", "fd", "cmaes", "kkt"]))
    for basis in bases:
        ks = [all_results[f"{basis}_s{s}"]["implicit"]["K"] for s in range(n_seeds)]
        row = []
        for m in ["implicit", "fd", "cmaes", "kkt"]:
            v = [all_results[f"{basis}_s{s}"][m]["n_solves"] for s in range(n_seeds)]
            l1 = [all_results[f"{basis}_s{s}"][m]["theta_l1"] for s in range(n_seeds)]
            row.append(f"{np.median(v):6.0f}/{np.median(l1):.2f}")
        print(f"{basis:>6s} {ks[0]:>3d} " + " ".join(f"{r:>12s}" for r in row))
    print("(cells are median forward solves / median theta L1 error)")


if __name__ == "__main__":
    tyro.cli(main)
