"""Decide whether ioc's implicit adjoint or the FD estimate is the wrong one.

`ioc.diagnostics.test_optimizer_correctness` reports cos(adjoint, FD) negative
at every eps with the forward solve converged.  That is consistent with EITHER
a broken adjoint or an FD estimate destroyed by solver noise, and cosine alone
cannot separate them.  Three checks that can:

  1. DESCENT TEST -- evaluate loss(z0 - t*g) for a ladder of t.  Needs no second
     gradient estimate at all: if g is a true gradient, some small t decreases
     the loss, full stop.  This is the arbiter.
  2. CENTRAL differences on the SCALAR directional derivative along g, which is
     O(eps^2) accurate and far less noise-prone than the K-vector forward FD
     the suite uses.  Compare to <g,g> = |g|^2, the value it must reproduce.
  3. SOLVER-NOISE FLOOR -- re-evaluate loss(z0) repeatedly under a solver that
     should be deterministic, and perturb by eps in one coordinate, to size the
     noise against the FD numerator.  If noise >> numerator, FD is meaningless
     regardless of what the adjoint does.
"""
import jax, jax.numpy as jnp, numpy as np
from ioc import outer as outer_opt
from ioc.diagnostics import SuiteConfig, build_inner
from ioc.robot import bases, problem as prob

cfg = SuiteConfig(n_newton=80)
theta_star = jnp.asarray([0.5, 0.3, 0.2])
problem = prob.RobotProblem.load(cfg.urdf_path, cfg.srdf_path, cfg.mesh_dir, cfg.n_timesteps)
residual_fn, names = bases.kinematic(problem, "k3")
K = len(names)
rng = np.random.default_rng(cfg.seed)

scenes = problem.sample_scenes(rng, 10)
x0s = problem.seeds(scenes)
inner, _ = build_inner(problem, residual_fn, scenes, cfg)
_, _, demos = prob.make_demos(problem, inner.solve_implicit, scenes, theta_star, rng, 0.0)

loss = jax.jit(prob.make_outer(problem, inner.solve_implicit, scenes, demos, x0s))
z0 = jnp.asarray(rng.normal(scale=0.5, size=K))
g = jax.jit(jax.grad(loss))(z0)
L0 = float(loss(z0))
gn2 = float(jnp.dot(g, g))
print(f"loss(z0) = {L0:.8f}   |g|^2 = {gn2:.6e}   g = {np.asarray(g)}")

print("\n[1] DESCENT TEST along -g   (predicted drop = t*|g|^2 for small t)")
print(f"{'t':>10}{'loss(z0-t*g)':>18}{'actual drop':>16}{'predicted':>14}{'ratio':>9}")
for t in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
    Lt = float(loss(z0 - t * g))
    drop, pred = L0 - Lt, t * gn2
    print(f"{t:>10.0e}{Lt:>18.8f}{drop:>16.3e}{pred:>14.3e}{drop/pred:>9.3f}")

print("\n[2] CENTRAL difference of the directional derivative along g")
print(f"    must converge to <g,g> = {gn2:.6e}")
u = g / jnp.linalg.norm(g)
exact = float(jnp.dot(g, u))
print(f"{'eps':>10}{'central D_u':>16}{'exact <g,u>':>16}{'rel_err':>12}")
for eps in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
    D = (float(loss(z0 + eps * u)) - float(loss(z0 - eps * u))) / (2 * eps)
    print(f"{eps:>10.0e}{D:>16.6e}{exact:>16.6e}{abs(D-exact)/abs(exact):>12.3e}")

print("\n[3] SOLVER NOISE FLOOR")
reps = [float(loss(z0)) for _ in range(5)]
spread = max(reps) - min(reps)
print(f"    loss(z0) over 5 identical calls: spread = {spread:.3e}")
for eps in (1e-2, 1e-3, 1e-4, 1e-5):
    dz = jnp.zeros(K).at[0].set(eps)
    num = abs(float(loss(z0 + dz)) - L0)
    print(f"    eps={eps:.0e}  |FD numerator| = {num:.3e}   noise/numerator = "
          f"{spread/(num+1e-30):.3e}" + ("   <-- FD IS NOISE" if spread > 0.1*num else ""))
