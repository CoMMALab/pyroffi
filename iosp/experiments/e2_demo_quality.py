"""E2 — Demo-quality ablation: does recovery track CURATION, not COUNT?

On E1's K=3 setup, compares three demo regimes (matched N=3):
    (a) single     1 demo ("clear" scene only)
    (b) jittered   3 demos, small random perturbations of one scene family
    (c) curated    3 demos = E1's CURATED_SCENES, designed to excite different features

Reuses `e1_minimal_identifiable.run_canonical` for the full procedure; only the
scene construction varies.

Usage:
    CUDA_VISIBLE_DEVICES=<idx> XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        python -m iosp.experiments.e2_demo_quality
"""

import jax
import jax.numpy as jnp
import numpy as np

from iosp.model import pickplace as pp
from iosp.config import MESH_DIR, Q_START, SRDF_PATH, URDF_PATH
from iosp.experiments.e1_minimal_identifiable import (
    CURATED_SCENES,
    HELD_OUT_SCENES,
    run_canonical,
)

N_REGIME = 3  # matches Study 1's CURATED_SCENES count


def _jitter_scene_spec(base_spec, rng, pos_jitter=0.02, obs_jitter=0.01):
    """One small random perturbation of a scene spec -- same jitter STYLE as
    `recovery_bench.sample_pickplace_scenes`, applied to a single base scene
    (here: "clear") rather than to the shared nominal scene, since regime (b)
    is specifically "N jittered copies of one motion", not "N independent
    random scenes"."""
    return dict(
        q_start=base_spec["q_start"] + jnp.asarray(rng.normal(scale=0.03, size=7), dtype=jnp.float32),
        pick_pos=base_spec["pick_pos"] + jnp.asarray(rng.normal(scale=pos_jitter, size=3), dtype=jnp.float32),
        place_pos=base_spec["place_pos"] + jnp.asarray(rng.normal(scale=pos_jitter, size=3), dtype=jnp.float32),
        obs_center=base_spec["obs_center"] + jnp.asarray(rng.normal(scale=obs_jitter, size=3), dtype=jnp.float32),
        obs_radius=base_spec["obs_radius"],
    )


def build_regimes(seed=0):
    rng = np.random.default_rng(seed)
    single = {"single": CURATED_SCENES["clear"]}
    jittered = {
        f"jitter{i}": _jitter_scene_spec(CURATED_SCENES["clear"], rng)
        for i in range(N_REGIME)
    }
    curated = {name: CURATED_SCENES[name] for name in ("clear", "smooth", "shape")}
    return {"(a) single": single, "(b) jittered": jittered, "(c) curated": curated}


def run_ablation(seed=0, n_steps=40, n_starts=16):
    prob = pp.PickPlaceProblem.load(str(URDF_PATH), str(SRDF_PATH), str(MESH_DIR))
    forward_solver = pp.make_composed_forward_solver(n_iters=60)

    regimes = build_regimes(seed=seed)
    results = {}
    for regime_name, scene_specs in regimes.items():
        fit_scene_spec = next(iter(scene_specs.values()))  # first/primary scene of the regime
        # Study 2 must use the SAME procedure Study 1 was validated with, or
        # the two are not comparable and the ablation just re-measures the
        # degenerate corner.  The defaults here were `alpha0_mode="grid"`,
        # `nonneg_theta=False`, `n_steps=12` -- the configuration that
        # produced the degenerate alpha_hat twice.  G1 established
        # multistart + nonneg as the procedure that actually moves.
        r = run_canonical(
            fit_scene_spec, held_out_specs=HELD_OUT_SCENES,
            certificate_scene_specs=scene_specs,
            prob=prob, forward_solver=forward_solver, seed=seed, n_steps=n_steps,
            alpha0_mode="multistart", nonneg_theta=True, n_starts=n_starts,
        )
        print(f"G1 VERDICT [{regime_name}]: "
              f"{'DEGENERATE (not a result)' if r['degenerate'] else 'non-degenerate'}"
              f"  move_rel={r['move_rel']:.3e}  loss_reduction={r['loss_reduction']:.2f}x",
              flush=True)
        results[regime_name] = r
        # Reuse the SAME prob/forward_solver object across regimes (returned
        # by run_canonical) -- avoids reloading the URDF/collision model each
        # time; does not by itself avoid recompiling `gf` when a regime's
        # selected rank k differs from the previous one's (a new alpha
        # dimension is a new shape).
        prob, forward_solver = r["prob"], r["forward_solver"]
    return results


def _print_ablation(results):
    print(f"{'regime':14s} {'eigvals':40s} {'k':>3s} {'param_err_nofit':>16s} {'param_err_hat':>14s} {'fit_rmse':>10s} {'fit?':>12s}")
    for name, r in results.items():
        eig_str = np.array2string(r["eigvals"], precision=3, suppress_small=True)
        flag = "DEGENERATE" if r.get("degenerate") else "ok"
        print(f"{name:14s} {eig_str:40s} {r['k']:3d} {r['param_err_nofit']:16.4f} "
              f"{r['param_err_hat']:14.4f} {r['fit_rmse']:10.4f} {flag:>12s}")
    print()
    print("held-out generalization RMSE (fitted vs no-fit baseline), per regime:")
    for name, r in results.items():
        print(f"  {name}")
        for gen_name in r["gen_rmse_hat"]:
            print(f"    [{gen_name}]  fitted={r['gen_rmse_hat'][gen_name]:.4f}   "
                  f"no-fit={r['gen_rmse_nofit'][gen_name]:.4f}")
    print()
    # The claim to check, not assume: does (c) beat (b) at matched N?
    #
    # An arm whose fit is DEGENERATE (alpha_hat never left its seed) has no
    # business in this comparison: its "fitted" RMSE is the seed's RMSE, so
    # comparing it against a real fit measures seed luck, not demo quality.
    # MEASURED 2026-08-28: (b) jittered came back degenerate with gen RMSE
    # 0.4647, which is BETTER than (c) curated's honest 0.4739 -- so the old
    # unconditional print concluded "no clear winner (comparable)" off a
    # number that was never a fit.  Refuse the comparison instead.
    rb, rc = results["(b) jittered"], results["(c) curated"]
    for gen_name in next(iter(results.values()))["gen_rmse_hat"]:
        b = rb["gen_rmse_hat"][gen_name]
        c = rc["gen_rmse_hat"][gen_name]
        bad = [n for n, r in (("(b) jittered", rb), ("(c) curated", rc))
               if r.get("degenerate")]
        if bad:
            print(f"[{gen_name}] jittered={b:.4f} vs curated={c:.4f}  ->  "
                  f"NO VERDICT: {', '.join(bad)} degenerate (alpha_hat == its "
                  f"seed), so this is not a comparison of two fits")
            continue
        verdict = "curated CLEARLY beats jittered" if c < 0.8 * b else (
            "jittered CLEARLY beats curated" if b < 0.8 * c else "no clear winner (comparable)")
        print(f"[{gen_name}] jittered={b:.4f} vs curated={c:.4f}  ->  {verdict}")


if __name__ == "__main__":
    results = run_ablation()
    _print_ablation(results)
