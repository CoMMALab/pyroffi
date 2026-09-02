"""Is the outer loss discontinuous in z, and if so, why?

`grad_verdict.py` found: solver noise floor EXACTLY zero (loss is deterministic),
yet the forward-FD numerator |loss(z0+eps*e0) - loss(z0)| stays pinned near
3.5e-5 as eps sweeps 1e-3 -> 1e-5 instead of shrinking proportionally to eps.
A deterministic function whose difference quotient does not shrink with eps has
a JUMP, not noise.  This walks a fine grid through z0 along coordinate 0 to see
the shape of it directly, and reads out the forward solver's terminating
iteration count alongside, since an early-stopping solver whose stop index k(z)
is an integer function of z produces exactly this signature.
"""
import jax, jax.numpy as jnp, numpy as np
from ioc.diagnostics import SuiteConfig, build_inner
from ioc.robot import bases, problem as prob

cfg = SuiteConfig(n_newton=80)
theta_star = jnp.asarray([0.5, 0.3, 0.2])
problem = prob.RobotProblem.load(cfg.urdf_path, cfg.srdf_path, cfg.mesh_dir, cfg.n_timesteps)
residual_fn, names = bases.kinematic(problem, "k3")
rng = np.random.default_rng(cfg.seed)
scenes = problem.sample_scenes(rng, 10)
x0s = problem.seeds(scenes)
inner, _ = build_inner(problem, residual_fn, scenes, cfg)
_, _, demos = prob.make_demos(problem, inner.solve_implicit, scenes, theta_star, rng, 0.0)
loss = jax.jit(prob.make_outer(problem, inner.solve_implicit, scenes, demos, x0s))
z0 = jnp.asarray(rng.normal(scale=0.5, size=3))
g = jax.jit(jax.grad(loss))(z0)

print("profile of loss along coordinate 0 through z0")
print("slope column = successive divided difference; a smooth function holds it")
print("roughly constant, a jump makes it explode at one grid point.\n")
print(f"{'delta':>12}{'loss':>18}{'d(loss)/d(delta)':>20}")
for span in (1e-2, 1e-3, 1e-4):
    print(f"  --- span +/-{span:.0e} ---")
    deltas = np.linspace(-span, span, 21)
    vals = [float(loss(z0.at[0].add(float(d)))) for d in deltas]
    for i, (d, v) in enumerate(zip(deltas, vals)):
        s = "" if i == 0 else f"{(vals[i]-vals[i-1])/(deltas[i]-deltas[i-1]):>20.6e}"
        print(f"{d:>12.3e}{v:>18.10f}{s}")
    print(f"  adjoint g[0] = {float(g[0]):.6e}")

# Does the stationarity of the inner solve vary across the same sweep?  A solve
# that terminates at a different quality on either side of the jump is the
# mechanism; one that is uniformly converged rules it out.
print("\ninner-solve stationarity across the same sweep (max over scenes):")
for d in (-1e-3, -1e-4, 0.0, 1e-4, 1e-3):
    z = z0.at[0].add(d)
    th = jax.nn.softmax(z)
    st = jax.vmap(lambda x0, s: inner.stationarity(x0, th, s))(x0s, scenes)
    print(f"  delta={d:>10.1e}   stat_max={float(jnp.max(st)):.4e}   stat_med={float(jnp.median(st)):.4e}")
