"""Does tightening the inner solve restore a differentiable outer loss?

`loss_profile.py` found the outer loss is DETERMINISTIC (repeat calls bit-identical)
yet oscillates over +/-25% of its own value at every probe span from 1e-2 down to
1e-4, with no trend, while the adjoint reports a slope 3 orders of magnitude
smaller than the observed local swings.  Deterministic + non-shrinking difference
quotients + uniform inner stationarity ~6e-4 => x*(z) is not being resolved as a
function of z at all: the forward solver hits its ITERATION CAP (asked for
grad_tol=1e-6, delivers 6e-4), so where it stops wobbles with z and the outer
loss reads out that wobble instead of the cost's actual dependence on theta.

If that is right, raising the cap must (a) drive stationarity down and (b) make
the profile's divided differences converge on the adjoint's value.  If the
profile stays hashy even at a tight solve, the mechanism is something else and
this hypothesis is dead.  `roughness` is the mean |divided difference|, which a
smooth function holds near |g[0]| and a hashy one inflates by orders.
"""
import jax, jax.numpy as jnp, numpy as np
from ioc.diagnostics import SuiteConfig, build_inner
from ioc.robot import bases, problem as prob
import dataclasses

theta_star = jnp.asarray([0.5, 0.3, 0.2])
base = SuiteConfig()
problem = prob.RobotProblem.load(base.urdf_path, base.srdf_path, base.mesh_dir, base.n_timesteps)
residual_fn, names = bases.kinematic(problem, "k3")
rng = np.random.default_rng(base.seed)
scenes = problem.sample_scenes(rng, 10)
x0s = problem.seeds(scenes)
z0 = jnp.asarray(np.random.default_rng(123).normal(scale=0.5, size=3))

print(f"{'n_newton':>9}{'stat_max':>12}{'stat_med':>12}{'adjoint g0':>14}"
      f"{'roughness':>13}{'rough/|g0|':>12}")
for n_iters in (80, 200, 600, 1500):
    cfg = dataclasses.replace(base, n_newton=n_iters)
    inner, _ = build_inner(problem, residual_fn, scenes, cfg)
    _, _, demos = prob.make_demos(problem, inner.solve_implicit, scenes, theta_star, rng, 0.0)
    loss = jax.jit(prob.make_outer(problem, inner.solve_implicit, scenes, demos, x0s))
    g0 = float(jax.jit(jax.grad(loss))(z0)[0])

    th = jax.nn.softmax(z0)
    st = jax.vmap(lambda x0, s: inner.stationarity(x0, th, s))(x0s, scenes)

    deltas = np.linspace(-1e-3, 1e-3, 21)
    vals = np.array([float(loss(z0.at[0].add(float(d)))) for d in deltas])
    dd = np.diff(vals) / np.diff(deltas)
    rough = float(np.mean(np.abs(dd)))
    print(f"{n_iters:>9}{float(jnp.max(st)):>12.3e}{float(jnp.median(st)):>12.3e}"
          f"{g0:>14.4e}{rough:>13.4e}{rough/(abs(g0)+1e-30):>12.2f}")
