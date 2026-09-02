"""Why does a wide fit stall above a loss it provably can reach?

On the composed model a fit with ALL K weights free plateaus at ee-RMSE 0.0244
while theta_star reproduces the demo exactly (0).  With the composed gradient now
measuring cos = 0.9996 against central differences, "the derivative is wrong" no
longer explains it, so this asks the question directly on the SINGLE segment,
where the demo is generated from a known theta_star in a well-specified basis and
the achievable floor is therefore known to be ~0.

Five probes, each ruling something specific in or out:

  [A] loss(z_star) -- the floor itself.  If this is NOT ~0, the demo is not
      reproducible even at the truth and nothing downstream is an optimizer
      failure at all; it would be a solver/demo-pipeline inconsistency.
  [B] Adam from the standard start, and from several random starts.  Spread
      across starts separates "one bad basin" from "a floor every start hits".
  [C] CMA-ES from the same budget.  Gradient-FREE: if it reaches materially
      lower loss than Adam, the gradient (or the step rule) is the problem; if
      it plateaus at the same value, the landscape is.
  [D] Loss along the segment from z_hat to z_star.  A monotone decrease means
      Adam simply stopped early; a barrier means the two are in different
      basins; a flat stretch means the direction is behaviourally inert.
  [E] Adam restarted FROM z_star.  If it stays, z_star is a fixed point of the
      optimizer and the floor is real; if it walks away and the loss RISES, the
      outer objective's minimum is not at the truth -- an objective-design
      problem, not an optimization one.
"""
import jax, jax.numpy as jnp, numpy as np

from ioc import outer as outer_opt
from ioc.diagnostics import SuiteConfig, build_inner, score
from ioc.robot import bases, problem as prob

cfg = SuiteConfig()
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
gf = jax.jit(jax.value_and_grad(loss))
z_star = jnp.log(theta_star)  # softmax(log p) == p


def rmse(z):
    return score(problem, inner, z, scenes, demos, x0s, theta_star)["ee_rmse"]


print(f"[A] loss(z_star) = {float(loss(z_star)):.6e}   ee_rmse = {rmse(z_star):.6e}")
print(f"    loss(zeros)  = {float(loss(jnp.zeros(K))):.6e}   ee_rmse = {rmse(jnp.zeros(K)):.6e}")

print("\n[B] Adam from several starts")
results = []
starts = [("zeros", jnp.zeros(K))] + [
    (f"rand{s}", jnp.asarray(np.random.default_rng(100 + s).normal(scale=0.5, size=K)))
    for s in range(4)
]
for label, z0 in starts:
    zh, _ = outer_opt.adam(gf, z0, lr=cfg.lr, n_steps=cfg.n_outer_steps)
    results.append((label, zh))
    print(f"    {label:8s} loss {float(loss(z0)):.4e} -> {float(loss(zh)):.4e}   "
          f"ee_rmse {rmse(zh):.5f}   theta={np.asarray(jax.nn.softmax(zh))}")

# Adam spends ~1 forward solve per gradient, so cfg.n_outer_steps solves total.
# CMA-ES gets 10x that -- deliberately generous: if a gradient-free search with
# an order more budget cannot beat Adam's plateau, the plateau is the landscape,
# not the gradient.
print(f"\n[C] CMA-ES (gradient-free), {10 * cfg.n_outer_steps} solves "
      f"vs Adam's ~{cfg.n_outer_steps}")
try:
    out = outer_opt.cma_es(loss, jnp.zeros(K), budget_solves=10 * cfg.n_outer_steps,
                           seed=cfg.seed)
    z_cma = out[0] if isinstance(out, tuple) else out
    print(f"    cma      loss -> {float(loss(z_cma)):.4e}   ee_rmse {rmse(z_cma):.5f}   "
          f"theta={np.asarray(jax.nn.softmax(z_cma))}")
except Exception as e:  # signature drift shouldn't kill the rest of the probes
    print(f"    cma_es unavailable/failed: {type(e).__name__}: {e}")

z_hat = results[0][1]
print("\n[D] loss along z_hat -> z_star")
for t in np.linspace(0, 1, 11):
    z = (1 - t) * z_hat + t * z_star
    print(f"    t={t:4.2f}  loss={float(loss(z)):.6e}  ee_rmse={rmse(z):.6f}")

print("\n[E] Adam restarted FROM z_star")
z_from_star, _ = outer_opt.adam(gf, z_star, lr=cfg.lr, n_steps=cfg.n_outer_steps)
print(f"    loss {float(loss(z_star)):.6e} -> {float(loss(z_from_star)):.6e}   "
      f"ee_rmse {rmse(z_from_star):.6f}")
print(f"    theta moved {np.asarray(jax.nn.softmax(z_star))} -> "
      f"{np.asarray(jax.nn.softmax(z_from_star))}")
