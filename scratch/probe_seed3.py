"""Why does generalization seed 3 fail when seeds 0-2 do not?

`test_generalization` draws z0 from `cfg.seed` alone (`diagnostics.fit`), so a
seed fixes ONE initialization reused across every `n_contexts` row -- seed 3's
five bad rows are five re-runs of one bad start, not five independent failures.
Seed 3's z0 puts `collision` at 0.065 against a true 0.3, which is the specific
thing this probe tests: with the collision weight near zero the demonstrated
obstacle avoidance is nearly absent from the initial solve, so the feature that
most needs recovering is also the one least excited at the start.

Records, for each seed, the z trajectory, the loss trace, the gradient norm at
z0, and the final theta -- enough to separate "stuck in a flat region" from
"converged to the wrong basin".
"""

import json
import pathlib
import sys

# `ioc` is not an installed package, and sys.path[0] is scratch/ for a script
# run by path -- same cause as the visualize.py import failure.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

from ioc import diagnostics as D
from ioc.robot import bases, problem as prob

SEEDS = [int(s) for s in (sys.argv[1:] or ["2", "3"])]
N_FIT = 5
THETA_STAR = jnp.asarray([0.5, 0.3, 0.2])

cfg = D.SuiteConfig()
problem = D.prob.RobotProblem.load(cfg.urdf_path, cfg.srdf_path, cfg.mesh_dir,
                                   cfg.n_timesteps)
residual_fn, names = bases.kinematic(problem, "k3")
print(f"features: {list(names)}  theta*={np.asarray(THETA_STAR)}", flush=True)

out = {}
for seed in SEEDS:
    scfg = D.SuiteConfig(seed=seed)
    rng = np.random.default_rng(seed)

    # mirror test_generalization's scene pipeline exactly
    pool = problem.sample_scenes(rng, (20 + 15) * 4)
    probe_inner, _ = D.build_inner(problem, residual_fn, pool, scfg)
    scenes_all, discard, _ = prob.screen_scenes(
        problem, pool, probe_inner.stationarity, THETA_STAR, scfg.conv_tol, 20 + 15)
    scenes_fit = jax.tree.map(lambda a: a[:N_FIT], scenes_all)
    x0s = problem.seeds(scenes_fit)
    inner, _ = D.build_inner(problem, residual_fn, scenes_fit, scfg)
    _, _, demos = prob.make_demos(problem, inner.solve_implicit, scenes_fit,
                                  THETA_STAR, rng, 0.0)

    loss = prob.make_outer(problem, inner.solve_implicit, scenes_fit, demos, x0s)
    lg = jax.jit(jax.value_and_grad(loss))

    z = jnp.asarray(np.random.default_rng(seed).normal(scale=0.5, size=3))
    z0 = z
    v0, g0 = lg(z0)

    # Adam by hand so the z trajectory can be recorded
    import optax
    opt = optax.adamw(scfg.lr)
    st = opt.init(z)
    zs, losses, gnorms = [], [], []
    for t in range(scfg.n_outer_steps):
        v, g = lg(z)
        zs.append([float(x) for x in z])
        losses.append(float(v))
        gnorms.append(float(jnp.linalg.norm(g)))
        u, st = opt.update(g, st, z)
        z = optax.apply_updates(z, u)

    theta0 = np.asarray(jax.nn.softmax(z0))
    theta_hat = np.asarray(jax.nn.softmax(z))
    gram = D.gram_certificate(inner, scenes_fit, demos, 3)
    m = D.score(problem, inner, z, scenes_fit, demos, x0s, THETA_STAR)

    print(f"\n=== seed {seed} ===")
    print(f"  z0        = {np.round(np.asarray(z0), 4)}")
    print(f"  theta0    = {np.round(theta0, 4)}   (collision={theta0[1]:.4f})")
    print(f"  theta_hat = {np.round(theta_hat, 4)}")
    print(f"  |grad| at z0 = {float(jnp.linalg.norm(g0)):.4e}   loss@z0 = {float(v0):.4e}")
    print(f"  loss: {losses[0]:.4e} -> {losses[-1]:.4e}  ({losses[-1]/losses[0]:.3f}x)")
    print(f"  |grad|: first={gnorms[0]:.3e}  median={np.median(gnorms):.3e}  last={gnorms[-1]:.3e}")
    print(f"  param_err={m['theta_l1']:.4f}  e_demo={m['ee_rmse']:.4e}")
    print(f"  gram: eigvals={np.round(gram['eigvals'], 5)}  cond={gram['cond']:.3g}"
          f"  eff_rank={gram['effective_rank']:.3f}")
    print(f"  discard_rate={discard:.3f}", flush=True)

    out[seed] = dict(z0=[float(x) for x in z0], theta0=theta0.tolist(),
                     theta_hat=theta_hat.tolist(), losses=losses,
                     gnorms=gnorms, zs=zs, gram=gram, discard_rate=float(discard),
                     param_err=m["theta_l1"], e_demo=m["ee_rmse"])

with open("scratch/logs/probe_seed3.json", "w") as f:
    json.dump(out, f, indent=2, default=float)
print("\nwrote scratch/logs/probe_seed3.json")
