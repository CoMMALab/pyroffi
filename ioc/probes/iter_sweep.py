"""Where does fixed-iteration (early_stop=False) stop paying off?

`openloop_vs_branching.py` attributed the outer loss's roughness to the inner
solver's discrete branches, and found `early_stop=False` alone the best
configuration: it cut roughness 23x AND gave the best FD-vs-adjoint cosine,
where the soft line-search/curvature-gate variants cut roughness further but
degraded the cosine (softening moves the fixed point the adjoint linearizes
about).  This sweeps the iteration count at that setting to find the budget
past which stationarity, the adjoint's own value, roughness, and FD agreement
all stop improving -- i.e. the smallest N worth paying for.
"""
import jax, jax.numpy as jnp, numpy as np
from ioc import metrics, outer as outer_opt
from ioc.diagnostics import SuiteConfig
from ioc.inner import make_inner_solver
from ioc.robot import bases, problem as prob


def main():
    """Runs the probe.  Called from `__main__` only: this module used to
    execute its whole solve at IMPORT time, so it could not be imported,
    listed or checked without launching GPU work."""
    def fsolve(n, **kw):
        from pyroffi.optimization_engines import DynamicsTrajOptConfig, dynamics_trajopt
        c = DynamicsTrajOptConfig(n_iters=n, **kw)
        return lambda x0, f: dynamics_trajopt(x0, f, c)


    b = SuiteConfig()
    ts = jnp.asarray([0.5, 0.3, 0.2])
    p = prob.RobotProblem.load(b.urdf_path, b.srdf_path, b.mesh_dir, b.n_timesteps)
    rf, nm = bases.kinematic(p, "k3")
    rng = np.random.default_rng(b.seed)
    sc = p.sample_scenes(rng, 10)
    x0s = p.seeds(sc)
    z0 = jnp.asarray(np.random.default_rng(123).normal(scale=0.5, size=3))

    print(f"{'n_iters':>8}{'stat_max':>11}{'stat_med':>11}{'g0':>13}"
          f"{'roughness':>12}{'rough/|g0|':>12}{'FD cos':>9}", flush=True)
    for n in (200, 600, 1500, 3000, 6000):
        fs = fsolve(n, grad_tol=1e-6, early_stop=False)
        inn = make_inner_solver(rf, p.calibrate(rf, sc, jax.random.key(b.seed)),
                                forward_solver=fs, adjoint_ridge=b.adjoint_ridge)
        _, _, dm = prob.make_demos(p, inn.solve_implicit, sc, ts, rng, 0.0)
        L = jax.jit(prob.make_outer(p, inn.solve_implicit, sc, dm, x0s))
        g = jax.jit(jax.grad(L))(z0)
        th = jax.nn.softmax(z0)
        st = jax.vmap(lambda a, s: inn.stationarity(a, th, s))(x0s, sc)
        d = np.linspace(-1e-3, 1e-3, 21)
        v = np.array([float(L(z0.at[0].add(float(t)))) for t in d])
        r = float(np.mean(np.abs(np.diff(v) / np.diff(d))))
        cos = max(metrics.cosine(g, outer_opt.fd_grad_fn(L, e)(z0)[1])
                  for e in (1e-2, 1e-3, 1e-4))
        print(f"{n:>8}{float(jnp.max(st)):>11.2e}{float(jnp.median(st)):>11.2e}"
              f"{float(g[0]):>13.4e}{r:>12.4e}{r/(abs(float(g[0]))+1e-30):>12.2f}"
              f"{cos:>9.4f}", flush=True)


if __name__ == "__main__":
    main()
