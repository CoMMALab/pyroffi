"""Study 2 -- demo-quality ablation: does recovery track CURATION, not COUNT?

Motivation
----------
Study 1 (`iosp/study1_minimal_identifiable.py`) established a canonical
inference procedure (whitened multi-demo Gram certificate -> 95%-cumulative-
trace rank selection -> zero-prior subspace fit -> held-out generalization
vs. a no-fit baseline) on a minimal, deliberately-curated 3-feature pick-
and-place setup.  This script asks a different question on the SAME setup:
holding the feature set and scene family fixed, does recovery quality track
demo CURATION quality or demo COUNT?  If curated data at a given N clearly
beats randomly-jittered data at the SAME N, that's evidence bad performance
is a data problem, not a method problem -- with real implications for
learning-based approaches that rely on naive/random data collection.

Reuse discipline: this script does NOT reimplement the certificate, rank
selection, zero-prior fit, or generalization test -- it imports and calls
`iosp.experiments.e1_minimal_identifiable.run_canonical` (and its dependencies)
directly, for every regime.  The only new code here is the demo-regime
SCENE CONSTRUCTION (single scene / random jitter / curated) and the
comparison table across regimes.

Demo regimes (matched N=3, i.e. Study 1's own `CURATED_SCENES` count)
-----------------------------------------------------------------------
    (a) single       1 demo (the "clear" scene alone) -- today's baseline,
                      rerun on the new `line_dev` feature set (not the old
                      `upright` one) for a fair comparison.
    (b) jittered      3 demos, all small random perturbations of ONE scene
                      family ("clear"'s obstacle-intrusion geometry) --
                      mirrors `recovery_bench.sample_pickplace_scenes`'s
                      jitter style: redundant/non-decoupling BY CONSTRUCTION,
                      since all 3 demos excite roughly the same directions.
    (c) curated       3 demos = Study 1's own `CURATED_SCENES` ("clear",
                      "smooth", "shape") -- deliberately designed to load
                      onto different features.

Scope decision, stated rather than silently assumed: the CERTIFICATE in each
regime is accumulated over that regime's full N-demo set (so rank selection
and the identifiable subspace genuinely reflect the regime's data
richness), but the FIT itself (the implicit-adjoint outer loop) is run
against ONE representative scene from that regime (the first/primary one),
not a jointly-batched N-demo fit.  This isolates the question this study is
actually asking (does the regime's data give a better-conditioned
certificate/identifiable-subspace, hence better recovery+generalization)
without the added engineering and compile cost of a batched multi-demo fit
loss (which `recovery_bench.sweep_demo_count` already explores separately,
for a different question -- does MORE data of any kind help).  Flagged here
as a real scope choice, not hidden.

MEASURED result: filled in after the run; see `if __name__ == "__main__"`'s
printed report.

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
