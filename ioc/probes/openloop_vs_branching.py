"""Is the outer loss's roughness caused by the inner solver's BRANCHES?

`smoothness_vs_budget.py` showed roughness/|g| falls 153 -> 18 as the iteration
cap rises, then plateaus: the cap was one cause, not the only one.  The engine
exposes three discrete, theta-dependent decisions that a plateau is consistent
with -- `early_stop` (stop index is an integer function of theta),
`soft_line_search=False` (hard accept/reject branch), `soft_curvature_gate=False`
(hard L-BFGS skip branch).  Each flips at some theta, and a flip moves x* by a
finite amount for an infinitesimal change in theta.

This runs the SAME roughness probe under four configurations to attribute the
residual roughness to a specific branch rather than assuming which one matters.
If open-loop (all branches soft/off) collapses roughness and drives the FD-vs-
adjoint cosine toward +1, the hash is branching and the fix costs no memory --
no unrolling required.  If roughness survives with every branch removed, the
cause is elsewhere and unrolled differentiation is the next thing to try.
"""
import dataclasses
import jax, jax.numpy as jnp, numpy as np
from ioc import metrics, outer as outer_opt
from ioc.diagnostics import SuiteConfig
from ioc.inner import make_inner_solver
from ioc.robot import bases, problem as prob


def main():
    """Runs the probe.  Called from `__main__` only: this module used to
    execute its whole solve at IMPORT time, so it could not be imported,
    listed or checked without launching GPU work."""
    def forward_solver(n_iters, **kw):
        from pyroffi.optimization_engines import DynamicsTrajOptConfig, dynamics_trajopt
        cfg = DynamicsTrajOptConfig(n_iters=n_iters, **kw)
        return lambda x0, cost_fn: dynamics_trajopt(x0, cost_fn, cfg)


    base = SuiteConfig()
    theta_star = jnp.asarray([0.5, 0.3, 0.2])
    problem = prob.RobotProblem.load(base.urdf_path, base.srdf_path, base.mesh_dir, base.n_timesteps)
    residual_fn, names = bases.kinematic(problem, "k3")
    K = len(names)
    rng = np.random.default_rng(base.seed)
    scenes = problem.sample_scenes(rng, 10)
    x0s = problem.seeds(scenes)
    z0 = jnp.asarray(np.random.default_rng(123).normal(scale=0.5, size=K))

    CONFIGS = {
        "baseline (all hard, early_stop)": dict(early_stop=True),
        "no early_stop only":              dict(early_stop=False),
        "soft line search + gate":         dict(early_stop=True,  soft_line_search=True,
                                                soft_curvature_gate=True),
        "OPEN LOOP (fixed N, all soft)":   dict(early_stop=False, soft_line_search=True,
                                                soft_curvature_gate=True),
    }

    print(f"{'config':34}{'stat_max':>11}{'g0':>13}{'roughness':>12}"
          f"{'rough/|g0|':>12}{'FD cos':>9}")
    for label, kw in CONFIGS.items():
        fs = forward_solver(base.n_newton, grad_tol=1e-6, **kw)
        inner = make_inner_solver(residual_fn, problem.calibrate(residual_fn, scenes,
                                  jax.random.key(base.seed)), forward_solver=fs,
                                  adjoint_ridge=base.adjoint_ridge)
        _, _, demos = prob.make_demos(problem, inner.solve_implicit, scenes, theta_star, rng, 0.0)
        loss = jax.jit(prob.make_outer(problem, inner.solve_implicit, scenes, demos, x0s))
        g = jax.jit(jax.grad(loss))(z0)

        th = jax.nn.softmax(z0)
        st = jax.vmap(lambda x0, s: inner.stationarity(x0, th, s))(x0s, scenes)

        deltas = np.linspace(-1e-3, 1e-3, 21)
        vals = np.array([float(loss(z0.at[0].add(float(d)))) for d in deltas])
        rough = float(np.mean(np.abs(np.diff(vals) / np.diff(deltas))))
        # best FD cosine over a step-size ladder (per ioc.inner's float floor note)
        cos = max(metrics.cosine(g, outer_opt.fd_grad_fn(loss, e)(z0)[1])
                  for e in (1e-2, 1e-3, 1e-4))
        print(f"{label:34}{float(jnp.max(st)):>11.2e}{float(g[0]):>13.4e}"
              f"{rough:>12.4e}{rough/(abs(float(g[0]))+1e-30):>12.2f}{cos:>9.4f}")


if __name__ == "__main__":
    main()
