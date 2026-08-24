"""Held-out generalization check: does fit-demo RMSE actually indicate correct
cost recovery on the composed pick-and-place model, or is it aliased?

Motivation
----------
`iosp/recovery_bench.py` scores recovery by reconstruction RMSE against the
SAME demo theta_hat was fit to.  That is optimistic by construction whenever
multiple distinct theta vectors can reproduce one fixed trajectory about
equally well -- and this codebase already has direct evidence that happens
here: `recovery_bench.run()` shows implicit and CMA-ES landing at visibly
different theta_hat (different theta_ik, different per-segment trajopt
weights) while reaching similar fit RMSE on the same demo.  Low fit RMSE
therefore does not by itself prove theta_hat recovered anything resembling
theta_star's actual cost -- it only proves theta_hat's rollout reproduces the
one trajectory it was fit to, which many aliased theta vectors could also do.

This script tests that directly: fit theta_hat on one scene (A), then compare
theta_star's and theta_hat's rollouts on a SECOND scene (B) theta_hat never
saw during fitting.  If the two rollouts stay close on B, theta_hat captured
something that generalizes.  If they diverge sharply relative to the fit-RMSE
scale, that is direct evidence of aliasing/overfitting to scene A.

Scenes
------
Scene A -- `recovery_bench.THETA_IK_STAR`/`Q_START`/`PICK_POS`/`PLACE_POS`/
`OBS_CENTER`/`OBS_RADIUS` (the same fitting scene `recovery_bench.run()`
uses).  theta_hat is fit here with 12 implicit-adjoint Adam steps, exactly as
in `recovery_bench.run()`'s "implicit" arm (CMA-ES is not re-run here --
this script isolates the fitted theta_hat's OWN generalization, not a
method comparison).

Scene B (held out, never touched during fitting) -- a shifted/rotated variant
of scene A, same qualitative structure (still has to route near the same
obstacle, still one pick-and-place task): `q_start` offset by
`[0.15, -0.1, 0.0, 0.1, 0.0, -0.1, 0.0]` rad on top of scene A's `Q_START`,
`pick_pos` offset by `[0.05, 0.08, -0.03]` m, `place_pos` offset by
`[-0.05, -0.06, 0.04]` m, same obstacle.

MEASURED result (float32, `soft_line_search`+`soft_curvature_gate`+
`early_stop=False` forward solver, seed 0; see `run()` below to reproduce):

    theta_ik_star      = [0.06, 0.04]
    theta_ik_hat        = [0.0436, 0.0167]
    theta_trajopt_star = [0.1036, 0.2817, 0.0629, 0.1036, 0.2817, 0.0629, 0.1036]
    theta_trajopt_hat   = [0.1254, 0.1716, 0.1586, 0.1214, 0.1217, 0.1428, 0.1586]

    fit RMSE on scene A (the demo actually fit)          = 0.0335
    generalization RMSE on scene B (theta_star vs theta_hat rollout) = 0.5649
    ratio (gen / fit)                                     = 16.9x

    named feature comparison on scene B (theta_star | theta_hat):
        approach.smooth_sumsq       0.00151 | 0.00118
        approach.clearance_max      0.00090 | 0.00090
        grasp.smooth_sumsq          ~0      | ~0
        transport.smooth_sumsq      0.38310 | 0.97664   (2.5x worse under theta_hat)
        transport.smooth_max        0.16020 | 0.34222   (2.1x worse under theta_hat)
        transport.clearance_max     0.01438 | 0.01186
        transport.upright_sumsq     3.31695 | 3.40515
        transport.upright_max       1.00011 | 0.99999   (see caveat below --
                                                           NOT saturation)
        place.smooth_sumsq          ~0      | ~0

    clearance-hinge max (0 = never inside margin; both scenes stayed safe):
        approach:  theta_star=0.00090  theta_hat=0.00090
        transport: theta_star=0.01438  theta_hat=0.01186

Conclusion (not something this script re-derives automatically -- recorded
here as the read at measurement time): a ~17x gap between fit RMSE and
held-out generalization RMSE is large, not "roughly comparable, some noise" --
this is direct evidence that low fit-demo RMSE is a MISLEADING indicator of
true cost recovery on this composed model: theta_hat reproduces the one demo
it was fit to well, but its rollout diverges substantially from theta_star's
on a held-out scene.  The biggest behavioral divergence is in
`transport.smooth` (~2.5x worse under theta_hat) -- clearance stays safe and
nearly identical under both theta, so the disagreement is concentrated in
trajectory smoothness during transport, not obstacle avoidance.  Any future
claim built on "low RMSE on the fitted demo" for this composed model should
be paired with this caveat, or ideally validated against a multi-demo fit
(see `recovery_bench.sweep_demo_count`, which shows fit-vs-recovery quality
weakly improving with more demonstrations -- consistent with under-constraint
from a single demo being at least part of this story).

Caveat on the upright feature, SUPERSEDED -- corrected here rather than left
to contradict the newer result: `transport.upright_max` sitting at ~1.0 for
BOTH theta_star and theta_hat's rollouts on scene B was originally read here
as possible saturation (a near-zero-gradient, weak-signal problem).  It is
NOT: `iosp/identifiability_check.py`'s Gram-matrix certificate measures
`transport.upright`'s gradient at 1.43, LARGER than `transport.smooth`'s own
(0.64) -- a strong gradient, not a saturated one.  The real mechanism is
near-exact collinearity: `cos(transport.smooth, transport.upright) = -0.9999`
at this demo, meaning the two features push the transport trajectory in
nearly the same direction, so the outer loop cannot tell how much of the
observed shape is due to one weight vs. the other -- which is the actual
reason recovery for those two weights is unreliable and doesn't generalize.
See `iosp/identifiability_check.py` for the full certificate (Gram spectrum,
cosine similarities, and the broader finding that 6 of 9 feature directions
are near-unidentifiable from this one demo, for a mix of collinearity and
genuine near-zero-gradient reasons).

Usage:
    CUDA_VISIBLE_DEVICES=<idx> XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        python -m iosp.generalization_check
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from ioc.inner import make_inner_solver
from ioc import outer as outer_opt
from iosp import pickplace as pp
from iosp.recovery_bench import (
    MESH_DIR,
    OBS_CENTER,
    OBS_RADIUS,
    PICK_POS,
    PLACE_POS,
    Q_START,
    SRDF_PATH,
    THETA_IK_STAR,
    URDF_PATH,
    Z_TRAJOPT_STAR,
    _split_trajopt,
    _unpack_z,
)

N_STEPS_FIT = 12  # same budget as recovery_bench.run()'s implicit arm

# Scene B: held out, never used in fitting.  Shifted/rotated variant of scene
# A -- same qualitative structure (routes near the same obstacle, same
# pick-and-place task), different enough that it is a genuine generalization
# probe rather than a near-duplicate of scene A.
SCENE_B_Q_START_OFFSET = jnp.array([0.15, -0.1, 0.0, 0.1, 0.0, -0.1, 0.0], dtype=jnp.float32)
SCENE_B_PICK_OFFSET = jnp.array([0.05, 0.08, -0.03], dtype=jnp.float32)
SCENE_B_PLACE_OFFSET = jnp.array([-0.05, -0.06, 0.04], dtype=jnp.float32)


def _phase_features(prob, xs, phase_scenes, phase):
    residual_fn = prob.segment_residual_fn(phase)
    rs = residual_fn(xs[phase][0], jax.tree.map(lambda a: a[0], phase_scenes[phase]))
    names = pp.SEGMENT_FEATURES[phase]
    out = {}
    for name, r in zip(names, rs):
        out[f"{phase}.{name}_sumsq"] = float(jnp.sum(r**2))
        out[f"{phase}.{name}_max"] = float(jnp.max(jnp.abs(r)))
    return out


def _clearance_hinge_max(prob, xs, phase_scenes, phase):
    if "clearance" not in pp.SEGMENT_FEATURES[phase]:
        return None
    problem = prob.seg[phase]
    sc = jax.tree.map(lambda a: a[0], phase_scenes[phase])
    q = problem.unpack(xs[phase][0], sc)
    return float(jnp.max(problem.clearance_residual(q, sc)))


def run(n_steps_fit=N_STEPS_FIT, seed=0):
    prob = pp.PickPlaceProblem.load(str(URDF_PATH), str(SRDF_PATH), str(MESH_DIR))
    forward_solver = pp.make_composed_forward_solver(n_iters=60)

    # -- scene A: fit theta_hat here, exactly as recovery_bench.run() does --
    scene_A = pp.PickPlaceScene(
        q_start=Q_START, pick_pos=PICK_POS, place_pos=PLACE_POS,
        obs_center=OBS_CENTER, obs_radius=OBS_RADIUS,
    )
    scenes_A = jax.tree.map(lambda a: a[None], scene_A)

    theta_trajopt_star = jax.nn.softmax(Z_TRAJOPT_STAR)
    x0_star, phase_scenes_star, _, _ = prob.seeds(scenes_A, THETA_IK_STAR)

    inner_by_phase = {}
    for p in pp.PHASES:
        residual_fn, _ = prob.make_segment_inner(p, forward_solver)
        scales = prob.calibrate_segment(p, residual_fn, phase_scenes_star[p], jax.random.PRNGKey(seed))
        inner_by_phase[p] = make_inner_solver(residual_fn, scales, forward_solver=forward_solver)

    _, _, xs_gt_A, phase_scenes_gt_A = prob.solve(
        THETA_IK_STAR, _split_trajopt(theta_trajopt_star), scenes_A, inner_by_phase, x0_star)
    demo_path_A = prob.full_ee_path(scenes_A, xs_gt_A, phase_scenes_gt_A, batch_index=0)

    def loss_A(z):
        theta_ik, z_trajopt = _unpack_z(z)
        theta_trajopt_by_phase = _split_trajopt(jax.nn.softmax(z_trajopt))
        x0, phase_scenes, _, _ = prob.seeds(scenes_A, theta_ik)
        _, _, xs, phase_scenes2 = prob.solve(theta_ik, theta_trajopt_by_phase, scenes_A, inner_by_phase, x0)
        path = prob.full_ee_path(scenes_A, xs, phase_scenes2, batch_index=0)
        return jnp.mean(jnp.sum((path - demo_path_A) ** 2, axis=-1))

    z0 = jnp.concatenate([jnp.zeros(pp.K_IK, dtype=jnp.float32), jnp.zeros(pp.K_TRAJOPT, dtype=jnp.float32)])
    gf = jax.jit(jax.value_and_grad(loss_A))
    t0 = time.perf_counter()
    jax.block_until_ready(gf(z0))
    compile_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    z_hat, _ = outer_opt.adam(gf, z0, lr=0.05, n_steps=n_steps_fit)
    fit_time = time.perf_counter() - t0

    theta_ik_hat, z_trajopt_hat = _unpack_z(z_hat)
    theta_trajopt_hat = jax.nn.softmax(z_trajopt_hat)
    fit_rmse_A = float(jnp.sqrt(loss_A(z_hat)))

    # -- scene B: held out, never used above --
    scene_B = pp.PickPlaceScene(
        q_start=Q_START + SCENE_B_Q_START_OFFSET,
        pick_pos=PICK_POS + SCENE_B_PICK_OFFSET,
        place_pos=PLACE_POS + SCENE_B_PLACE_OFFSET,
        obs_center=OBS_CENTER, obs_radius=OBS_RADIUS,
    )
    scenes_B = jax.tree.map(lambda a: a[None], scene_B)

    def rollout(theta_ik, theta_trajopt):
        theta_trajopt_by_phase = _split_trajopt(theta_trajopt)
        x0, phase_scenes, _, _ = prob.seeds(scenes_B, theta_ik)
        _, _, xs, phase_scenes2 = prob.solve(theta_ik, theta_trajopt_by_phase, scenes_B, inner_by_phase, x0)
        path = prob.full_ee_path(scenes_B, xs, phase_scenes2, batch_index=0)
        return xs, phase_scenes2, path

    xs_star_B, ps_star_B, path_star_B = rollout(THETA_IK_STAR, theta_trajopt_star)
    xs_hat_B, ps_hat_B, path_hat_B = rollout(theta_ik_hat, theta_trajopt_hat)
    jax.block_until_ready((path_star_B, path_hat_B))

    gen_rmse = float(jnp.sqrt(jnp.mean(jnp.sum((path_star_B - path_hat_B) ** 2, axis=-1))))

    features = {}
    for phase in pp.PHASES:
        features[phase] = dict(
            theta_star=_phase_features(prob, xs_star_B, ps_star_B, phase),
            theta_hat=_phase_features(prob, xs_hat_B, ps_hat_B, phase),
        )
    clearance = {}
    for phase in pp.PHASES:
        m_star = _clearance_hinge_max(prob, xs_star_B, ps_star_B, phase)
        m_hat = _clearance_hinge_max(prob, xs_hat_B, ps_hat_B, phase)
        if m_star is not None:
            clearance[phase] = dict(theta_star=m_star, theta_hat=m_hat)

    return dict(
        theta_ik_star=np.asarray(THETA_IK_STAR), theta_ik_hat=np.asarray(theta_ik_hat),
        theta_trajopt_star=np.asarray(theta_trajopt_star), theta_trajopt_hat=np.asarray(theta_trajopt_hat),
        fit_rmse_A=fit_rmse_A, gen_rmse_B=gen_rmse, ratio=gen_rmse / max(fit_rmse_A, 1e-12),
        compile_s=compile_time, fit_s=fit_time,
        features=features, clearance=clearance,
    )


def _print_report(r):
    print("theta_ik_star", r["theta_ik_star"], "theta_ik_hat", r["theta_ik_hat"])
    print("theta_trajopt_star", r["theta_trajopt_star"])
    print("theta_trajopt_hat ", r["theta_trajopt_hat"])
    print()
    print(f"fit RMSE on scene A:            {r['fit_rmse_A']:.4f}")
    print(f"generalization RMSE on scene B: {r['gen_rmse_B']:.4f}")
    print(f"ratio (gen / fit):              {r['ratio']:.1f}x")
    print(f"(compile {r['compile_s']:.1f}s, fit steady-state {r['fit_s']:.1f}s)")
    print()
    print(f"{'feature':28s} {'theta_star':>12s} {'theta_hat':>12s}")
    for phase, fp in r["features"].items():
        for k in fp["theta_star"]:
            print(f"{k:28s} {fp['theta_star'][k]:12.5f} {fp['theta_hat'][k]:12.5f}")
    print()
    print("clearance-hinge max (0 = never inside margin, larger = worse violation)")
    for phase, c in r["clearance"].items():
        print(f"{phase:12s} theta_star={c['theta_star']:.5f} theta_hat={c['theta_hat']:.5f}")


if __name__ == "__main__":
    _print_report(run())
