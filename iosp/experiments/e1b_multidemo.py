"""Diagnostic, NOT canonical: tests two hypotheses for why the corrected
`run_canonical` (see `study1_minimal_identifiable.py`, alpha0-selection bug
now fixed) produces a fit that is WORSE than the alpha=0 no-fit baseline on
both param_err and held-out generalization RMSE, despite the certificate
selecting a clean rank-2 subspace.

H1 (primary): certificate/fit mismatch. `select_rank`'s Gram matrix is
POOLED (averaged) over all 3 `CURATED_SCENES` demos, but `run_canonical`
only ever fits `alpha` against ONE demo (`fit_scene_spec`, always "clear" in
the two reported runs). Per-demo pairwise cosines printed by
`run_certificate` show real correlation on "clear" ALONE (smooth vs
clearance cos=-0.69, smooth vs line_dev cos=-0.65) even though the pooled
certificate looks clean rank-2 -- i.e. the certificate's identifiability
claim may not hold for a single demo, which is exactly the failure mode this
script tests by fitting against all 3 demos jointly (matching what the
certificate actually measures) instead of one.

H2 (secondary): optimizer step size/budget. n_steps=12, lr=0.05 may
overshoot a stiff/non-convex loss surface. Tested here by rerunning the
ORIGINAL single-demo fit with a smaller lr and more steps, to see whether
that alone recovers a sane fit without touching the multi-demo question.

This script does NOT modify `study1_minimal_identifiable.py`'s canonical
procedure -- it is a standalone diagnostic that imports and reuses its
pieces (`_setup`, `_solve_all`, `theta_from_alpha_zero_prior`,
`run_certificate`, `select_rank`) rather than duplicating them.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from ioc import outer as outer_opt
from iosp.model import pickplace as pp
from iosp.experiments.e1_minimal_identifiable import (
    CURATED_SCENES,
    HELD_OUT_SCENES,
    URDF_PATH,
    SRDF_PATH,
    MESH_DIR,
    THETA_IK_STAR,
    Z_TRAJOPT_STAR,
    _TRANSPORT_IDX,
    _make_scene,
    _setup,
    _solve_all,
    run_certificate,
    select_rank,
    theta_from_alpha_zero_prior,
)


def _fit_loss_single_demo(prob, forward_solver, inner_by_phase, scenes, x0_star,
                           theta_trajopt_star, demo_path, eigvecs, selected_idx):
    def loss(alpha):
        theta_transport = theta_from_alpha_zero_prior(alpha, eigvecs, selected_idx)
        theta_trajopt = theta_trajopt_star.at[_TRANSPORT_IDX].set(theta_transport)
        xs, ps = _solve_all(prob, scenes, inner_by_phase, theta_trajopt, x0_star)
        path = prob.full_ee_path(scenes, xs, ps, batch_index=0)
        return jnp.mean(jnp.sum((path - demo_path) ** 2, axis=-1))
    return loss


def _gen_rmse(prob, inner_by_phase, held_out_specs, theta_trajopt_star, theta_trajopt_hat, theta_trajopt_nofit):
    gen_rmse_hat, gen_rmse_nofit = {}, {}
    for gen_name, spec in held_out_specs.items():
        gen_scene = _make_scene(spec)
        scenes_gen = jax.tree.map(lambda a: a[None], gen_scene)
        x0_gen, _, _, _ = prob.seeds(scenes_gen, THETA_IK_STAR)

        def rollout(theta_trajopt):
            xs, ps = _solve_all(prob, scenes_gen, inner_by_phase, theta_trajopt, x0_gen)
            return prob.full_ee_path(scenes_gen, xs, ps, batch_index=0)

        path_star_gen = rollout(theta_trajopt_star)
        path_hat_gen = rollout(theta_trajopt_hat)
        path_nofit_gen = rollout(theta_trajopt_nofit)
        gen_rmse_hat[gen_name] = float(jnp.sqrt(jnp.mean(jnp.sum((path_star_gen - path_hat_gen) ** 2, axis=-1))))
        gen_rmse_nofit[gen_name] = float(jnp.sqrt(jnp.mean(jnp.sum((path_star_gen - path_nofit_gen) ** 2, axis=-1))))
    return gen_rmse_hat, gen_rmse_nofit


def _make_scene_batch(scene_specs, order):
    """Stack a {name: spec} dict (same shape as CURATED_SCENES) into ONE
    `PickPlaceScene` with a leading batch dim of len(order) -- as opposed to
    N separate batch-size-1 scenes, this is what lets `PickPlaceProblem.solve`
    (which already `jax.vmap`s `inner.solve_implicit` over its `scenes`
    argument -- see `pickplace.py:402`) process all N demos in ONE traced
    call instead of N separately-compiled closures fused into one jit."""
    def stack(key):
        return jnp.stack([jnp.asarray(scene_specs[name][key]) for name in order], axis=0)
    return pp.PickPlaceScene(
        q_start=stack("q_start"), pick_pos=stack("pick_pos"), place_pos=stack("place_pos"),
        obs_center=stack("obs_center"), obs_radius=stack("obs_radius"),
    )


def _setup_batch(prob, forward_solver, scenes, seed=0):
    """Batched analogue of `study1_minimal_identifiable._setup`: identical
    body, but takes an ALREADY-batched `scenes` (leading dim N, from
    `_make_scene_batch`) instead of a single unbatched scene it would
    otherwise wrap with `[None]` (batch=1). `calibrate_segment` and
    `prob.seeds` are already generic over the batch dim (see their
    docstrings/vmap calls in pickplace.py), so this is the SAME setup code
    path, just called once over N demos instead of N times over 1."""
    from ioc.inner import make_inner_solver

    theta_trajopt_star = jax.nn.softmax(Z_TRAJOPT_STAR)
    x0_star, phase_scenes_star, _, _ = prob.seeds(scenes, THETA_IK_STAR)

    inner_by_phase, residual_fn_by_phase, scales_by_phase = {}, {}, {}
    for p in pp.PHASES:
        if p == "transport":
            from iosp.experiments.e1_minimal_identifiable import _transport_residual_fn
            residual_fn = _transport_residual_fn(prob.seg[p])
        else:
            residual_fn, _ = prob.make_segment_inner(p, forward_solver)
        residual_fn_by_phase[p] = residual_fn
        scales = prob.calibrate_segment(p, residual_fn, phase_scenes_star[p], jax.random.PRNGKey(seed), jitter=0.3)
        scales_by_phase[p] = scales
        inner_by_phase[p] = make_inner_solver(residual_fn, scales, forward_solver=forward_solver)
    return theta_trajopt_star, x0_star, inner_by_phase, residual_fn_by_phase, scales_by_phase


def run_multi_demo_fit_vmap(theta_transport_star_override=None, seed=0, n_steps=12, lr=0.05,
                             prob=None, forward_solver=None, cert=None,
                             scene_order=("clear", "smooth", "shape")):
    """H1 test, VMAP-BATCHED version: fits alpha against the average loss
    over all N=len(scene_order) CURATED_SCENES demos using ONE traced/jitted
    forward+backward solve over a batch-N `PickPlaceScene`, instead of
    `run_multi_demo_fit`'s N separately-closured forward solves fused into
    one jit (which is what XLA's own "very slow compile" warning was flagging
    -- see HANDOFF.md). Numerically this should reproduce `run_multi_demo_fit`
    (same math: mean squared path error over the same N demos, same alpha0
    selection, same `theta_from_alpha_zero_prior`) -- it exists to fix
    COMPILE cost, not to test a different hypothesis than H1 already did."""
    if prob is None:
        prob = pp.PickPlaceProblem.load(str(URDF_PATH), str(SRDF_PATH), str(MESH_DIR))
    if forward_solver is None:
        forward_solver = pp.make_composed_forward_solver(n_iters=60)
    if cert is None:
        cert = run_certificate(seed=seed, whiten=True, scene_specs=CURATED_SCENES,
                                prob=prob, forward_solver=forward_solver)
    k, selected_idx = select_rank(cert["eigvals"], 0.95)
    N = len(scene_order)

    scenes = _make_scene_batch(CURATED_SCENES, scene_order)
    theta_trajopt_star, x0_star, inner_by_phase, _, _ = _setup_batch(prob, forward_solver, scenes, seed=seed)
    theta_transport_star = (
        theta_transport_star_override if theta_transport_star_override is not None
        else theta_trajopt_star[_TRANSPORT_IDX]
    )
    theta_trajopt_star = theta_trajopt_star.at[_TRANSPORT_IDX].set(theta_transport_star)

    xs_gt, ps_gt = _solve_all(prob, scenes, inner_by_phase, theta_trajopt_star, x0_star)
    demo_paths = jnp.stack(
        [prob.full_ee_path(scenes, xs_gt, ps_gt, batch_index=i) for i in range(N)], axis=0)

    def loss(alpha):
        theta_transport = theta_from_alpha_zero_prior(alpha, cert["eigvecs"], selected_idx)
        theta_trajopt = theta_trajopt_star.at[_TRANSPORT_IDX].set(theta_transport)
        # SINGLE `_solve_all` call over the batch-N scenes -- `prob.solve`
        # vmaps `inner.solve_implicit` internally (pickplace.py:402), so this
        # is one traced forward+backward solve covering all N demos, not N
        # of them fused into one jit.
        xs, ps = _solve_all(prob, scenes, inner_by_phase, theta_trajopt, x0_star)
        paths = jnp.stack([prob.full_ee_path(scenes, xs, ps, batch_index=i) for i in range(N)], axis=0)
        return jnp.mean(jnp.sum((paths - demo_paths) ** 2, axis=-1))

    gf = jax.jit(jax.value_and_grad(loss))
    ALPHA0_CANDIDATES = (0.001, 0.05, 0.2)
    theta_mags = {}
    for cand in ALPHA0_CANDIDATES:
        theta_mags[cand] = float(jnp.linalg.norm(theta_from_alpha_zero_prior(
            cand * jnp.ones(k, dtype=jnp.float32), cert["eigvecs"], selected_idx)))
    alpha0_scale = min(theta_mags, key=lambda c: abs(theta_mags[c] - 0.2))
    alpha0 = alpha0_scale * jnp.ones(k, dtype=jnp.float32)
    print(f"  [multi-demo vmap] selected alpha0_scale={alpha0_scale}")

    t0 = time.perf_counter()
    _, g_probe0 = gf(jnp.zeros(k, dtype=jnp.float32))
    compile_s = time.perf_counter() - t0
    print(f"  [multi-demo vmap] first-call (compile+solve) time: {compile_s:.1f}s")

    t0 = time.perf_counter()
    alpha_hat, _ = outer_opt.adam(gf, alpha0, lr=lr, n_steps=n_steps)
    fit_s = time.perf_counter() - t0
    print(f"  [multi-demo vmap] {n_steps} Adam steps (post-compile): {fit_s:.2f}s")

    theta_transport_hat = theta_from_alpha_zero_prior(alpha_hat, cert["eigvecs"], selected_idx)
    theta_trajopt_hat = theta_trajopt_star.at[_TRANSPORT_IDX].set(theta_transport_hat)
    theta_trajopt_nofit = theta_trajopt_star.at[_TRANSPORT_IDX].set(jnp.zeros(3, dtype=jnp.float32))

    param_err_hat = float(jnp.linalg.norm(theta_transport_hat - theta_transport_star))
    param_err_nofit = float(jnp.linalg.norm(theta_transport_star))
    fit_rmse = float(jnp.sqrt(loss(alpha_hat)))
    fit_rmse_nofit = float(jnp.sqrt(loss(jnp.zeros(k, dtype=jnp.float32))))

    ref_inner_by_phase = inner_by_phase  # shared across the batch; any one phase-scene works for gen rollout
    gen_rmse_hat, gen_rmse_nofit = _gen_rmse(
        prob, ref_inner_by_phase, HELD_OUT_SCENES, theta_trajopt_star, theta_trajopt_hat, theta_trajopt_nofit)

    print(f"theta_transport_star = {np.asarray(theta_transport_star)}")
    print(f"theta_transport_hat  = {np.asarray(theta_transport_hat)}  (alpha_hat={np.asarray(alpha_hat)})")
    print(f"param_err no-fit: {param_err_nofit:.4f}   param_err fitted: {param_err_hat:.4f}")
    print(f"fit RMSE (avg over {N} demos), fitted: {fit_rmse:.4f}   no-fit (alpha=0): {fit_rmse_nofit:.4f}")
    for gen_name in gen_rmse_hat:
        print(f"  [{gen_name}] gen_rmse fitted={gen_rmse_hat[gen_name]:.4f}  no-fit={gen_rmse_nofit[gen_name]:.4f}")
    return dict(param_err_hat=param_err_hat, param_err_nofit=param_err_nofit,
                fit_rmse=fit_rmse, fit_rmse_nofit=fit_rmse_nofit,
                gen_rmse_hat=gen_rmse_hat, gen_rmse_nofit=gen_rmse_nofit,
                theta_transport_hat=np.asarray(theta_transport_hat),
                compile_s=compile_s, fit_s=fit_s,
                prob=prob, forward_solver=forward_solver, cert=cert)


def run_multi_demo_fit(theta_transport_star_override=None, seed=0, n_steps=12, lr=0.05,
                        prob=None, forward_solver=None, cert=None):
    """H1 test (ORIGINAL, non-batched -- kept for reference/comparison against
    `run_multi_demo_fit_vmap`, which fixes this version's compile cost): fit
    alpha against the AVERAGE loss over all 3 CURATED_SCENES demos (matching
    what the pooled certificate actually measures), instead of a single
    demo."""
    if prob is None:
        prob = pp.PickPlaceProblem.load(str(URDF_PATH), str(SRDF_PATH), str(MESH_DIR))
    if forward_solver is None:
        forward_solver = pp.make_composed_forward_solver(n_iters=60)
    if cert is None:
        cert = run_certificate(seed=seed, whiten=True, scene_specs=CURATED_SCENES,
                                prob=prob, forward_solver=forward_solver)
    k, selected_idx = select_rank(cert["eigvals"], 0.95)

    per_scene = {}
    for name, spec in CURATED_SCENES.items():
        scene = _make_scene(spec)
        scenes, theta_trajopt_star, x0_star, inner_by_phase, _, _ = _setup(prob, forward_solver, scene, seed=seed)
        theta_transport_star = (
            theta_transport_star_override if theta_transport_star_override is not None
            else theta_trajopt_star[_TRANSPORT_IDX]
        )
        theta_trajopt_star = theta_trajopt_star.at[_TRANSPORT_IDX].set(theta_transport_star)
        xs_gt, ps_gt = _solve_all(prob, scenes, inner_by_phase, theta_trajopt_star, x0_star)
        demo_path = prob.full_ee_path(scenes, xs_gt, ps_gt, batch_index=0)
        per_scene[name] = dict(scenes=scenes, x0_star=x0_star, inner_by_phase=inner_by_phase,
                                theta_trajopt_star=theta_trajopt_star, demo_path=demo_path)

    def loss(alpha):
        total = 0.0
        for name, d in per_scene.items():
            theta_transport = theta_from_alpha_zero_prior(alpha, cert["eigvecs"], selected_idx)
            theta_trajopt = d["theta_trajopt_star"].at[_TRANSPORT_IDX].set(theta_transport)
            xs, ps = _solve_all(prob, d["scenes"], d["inner_by_phase"], theta_trajopt, d["x0_star"])
            path = prob.full_ee_path(d["scenes"], xs, ps, batch_index=0)
            total = total + jnp.mean(jnp.sum((path - d["demo_path"]) ** 2, axis=-1))
        return total / len(per_scene)

    gf = jax.jit(jax.value_and_grad(loss))
    ALPHA0_CANDIDATES = (0.001, 0.05, 0.2)
    theta_mags = {}
    for cand in ALPHA0_CANDIDATES:
        theta_mags[cand] = float(jnp.linalg.norm(theta_from_alpha_zero_prior(
            cand * jnp.ones(k, dtype=jnp.float32), cert["eigvecs"], selected_idx)))
    target_center = 0.2
    alpha0_scale = min(theta_mags, key=lambda c: abs(theta_mags[c] - target_center))
    alpha0 = alpha0_scale * jnp.ones(k, dtype=jnp.float32)
    print(f"  [multi-demo] selected alpha0_scale={alpha0_scale}")

    t0 = time.perf_counter()
    alpha_hat, _ = outer_opt.adam(gf, alpha0, lr=lr, n_steps=n_steps)
    fit_s = time.perf_counter() - t0

    theta_transport_hat = theta_from_alpha_zero_prior(alpha_hat, cert["eigvecs"], selected_idx)
    ref = per_scene["clear"]
    theta_trajopt_hat = ref["theta_trajopt_star"].at[_TRANSPORT_IDX].set(theta_transport_hat)
    theta_trajopt_nofit = ref["theta_trajopt_star"].at[_TRANSPORT_IDX].set(jnp.zeros(3, dtype=jnp.float32))

    param_err_hat = float(jnp.linalg.norm(theta_transport_hat - theta_transport_star))
    param_err_nofit = float(jnp.linalg.norm(theta_transport_star))
    fit_rmse = float(jnp.sqrt(loss(alpha_hat)))
    # THE question this diagnostic is actually meant to answer: does the fit
    # reconstruct the TRAINING demo(s) better than no-fit-at-all does, on
    # those SAME demos (not the held-out scene, which is a different check)?
    fit_rmse_nofit = float(jnp.sqrt(loss(jnp.zeros(k, dtype=jnp.float32))))

    gen_rmse_hat, gen_rmse_nofit = _gen_rmse(
        prob, ref["inner_by_phase"], HELD_OUT_SCENES, ref["theta_trajopt_star"], theta_trajopt_hat, theta_trajopt_nofit)

    print(f"theta_transport_star = {np.asarray(theta_transport_star)}")
    print(f"theta_transport_hat  = {np.asarray(theta_transport_hat)}  (alpha_hat={np.asarray(alpha_hat)})")
    print(f"param_err no-fit: {param_err_nofit:.4f}   param_err fitted: {param_err_hat:.4f}")
    print(f"fit RMSE (avg over 3 demos), fitted: {fit_rmse:.4f}   no-fit (alpha=0): {fit_rmse_nofit:.4f}")
    for gen_name in gen_rmse_hat:
        print(f"  [{gen_name}] gen_rmse fitted={gen_rmse_hat[gen_name]:.4f}  no-fit={gen_rmse_nofit[gen_name]:.4f}")
    return dict(param_err_hat=param_err_hat, param_err_nofit=param_err_nofit,
                fit_rmse=fit_rmse, fit_rmse_nofit=fit_rmse_nofit,
                gen_rmse_hat=gen_rmse_hat, gen_rmse_nofit=gen_rmse_nofit,
                theta_transport_hat=np.asarray(theta_transport_hat), fit_s=fit_s,
                prob=prob, forward_solver=forward_solver, cert=cert)


def run_single_demo_tuned(theta_transport_star_override=None, seed=0, n_steps=40, lr=0.01,
                           prob=None, forward_solver=None, cert=None):
    """H2 test: same single-demo ("clear") fit as the canonical procedure,
    but with a smaller lr and 3x the step budget, to check whether the
    negative result was purely an optimizer tuning artifact rather than a
    single-demo identifiability artifact."""
    if prob is None:
        prob = pp.PickPlaceProblem.load(str(URDF_PATH), str(SRDF_PATH), str(MESH_DIR))
    if forward_solver is None:
        forward_solver = pp.make_composed_forward_solver(n_iters=60)
    if cert is None:
        cert = run_certificate(seed=seed, whiten=True, scene_specs=CURATED_SCENES,
                                prob=prob, forward_solver=forward_solver)
    k, selected_idx = select_rank(cert["eigvals"], 0.95)

    fit_scene = _make_scene(CURATED_SCENES["clear"])
    scenes, theta_trajopt_star, x0_star, inner_by_phase, _, _ = _setup(prob, forward_solver, fit_scene, seed=seed)
    theta_transport_star = (
        theta_transport_star_override if theta_transport_star_override is not None
        else theta_trajopt_star[_TRANSPORT_IDX]
    )
    theta_trajopt_star = theta_trajopt_star.at[_TRANSPORT_IDX].set(theta_transport_star)
    xs_gt, ps_gt = _solve_all(prob, scenes, inner_by_phase, theta_trajopt_star, x0_star)
    demo_path = prob.full_ee_path(scenes, xs_gt, ps_gt, batch_index=0)

    loss = _fit_loss_single_demo(prob, forward_solver, inner_by_phase, scenes, x0_star,
                                  theta_trajopt_star, demo_path, cert["eigvecs"], selected_idx)
    gf = jax.jit(jax.value_and_grad(loss))

    ALPHA0_CANDIDATES = (0.001, 0.05, 0.2)
    theta_mags = {}
    for cand in ALPHA0_CANDIDATES:
        theta_mags[cand] = float(jnp.linalg.norm(theta_from_alpha_zero_prior(
            cand * jnp.ones(k, dtype=jnp.float32), cert["eigvecs"], selected_idx)))
    alpha0_scale = min(theta_mags, key=lambda c: abs(theta_mags[c] - 0.2))
    alpha0 = alpha0_scale * jnp.ones(k, dtype=jnp.float32)

    t0 = time.perf_counter()
    alpha_hat, history = outer_opt.adam(gf, alpha0, lr=lr, n_steps=n_steps)
    fit_s = time.perf_counter() - t0

    theta_transport_hat = theta_from_alpha_zero_prior(alpha_hat, cert["eigvecs"], selected_idx)
    theta_trajopt_hat = theta_trajopt_star.at[_TRANSPORT_IDX].set(theta_transport_hat)
    theta_trajopt_nofit = theta_trajopt_star.at[_TRANSPORT_IDX].set(jnp.zeros(3, dtype=jnp.float32))

    param_err_hat = float(jnp.linalg.norm(theta_transport_hat - theta_transport_star))
    param_err_nofit = float(jnp.linalg.norm(theta_transport_star))
    fit_rmse = float(jnp.sqrt(loss(alpha_hat)))
    fit_rmse_nofit = float(jnp.sqrt(loss(jnp.zeros(k, dtype=jnp.float32))))

    gen_rmse_hat, gen_rmse_nofit = _gen_rmse(
        prob, inner_by_phase, HELD_OUT_SCENES, theta_trajopt_star, theta_trajopt_hat, theta_trajopt_nofit)

    print(f"theta_transport_star = {np.asarray(theta_transport_star)}")
    print(f"theta_transport_hat  = {np.asarray(theta_transport_hat)}  (alpha_hat={np.asarray(alpha_hat)})")
    print(f"param_err no-fit: {param_err_nofit:.4f}   param_err fitted: {param_err_hat:.4f}")
    print(f"fit RMSE, fitted: {fit_rmse:.4f}   no-fit (alpha=0): {fit_rmse_nofit:.4f}")
    for gen_name in gen_rmse_hat:
        print(f"  [{gen_name}] gen_rmse fitted={gen_rmse_hat[gen_name]:.4f}  no-fit={gen_rmse_nofit[gen_name]:.4f}")
    return dict(param_err_hat=param_err_hat, param_err_nofit=param_err_nofit,
                fit_rmse=fit_rmse, fit_rmse_nofit=fit_rmse_nofit,
                gen_rmse_hat=gen_rmse_hat, gen_rmse_nofit=gen_rmse_nofit,
                theta_transport_hat=np.asarray(theta_transport_hat), fit_s=fit_s,
                prob=prob, forward_solver=forward_solver, cert=cert)


if __name__ == "__main__":
    prob = pp.PickPlaceProblem.load(str(URDF_PATH), str(SRDF_PATH), str(MESH_DIR))
    forward_solver = pp.make_composed_forward_solver(n_iters=60)
    cert = run_certificate(seed=0, whiten=True, scene_specs=CURATED_SCENES, prob=prob, forward_solver=forward_solver)

    print("=" * 60)
    print("H1 (vmap-batched): multi-demo (all 3 curated scenes) fit -- Study-0 theta_star")
    print("=" * 60)
    run_multi_demo_fit_vmap(prob=prob, forward_solver=forward_solver, cert=cert)

    print()
    print("=" * 60)
    print("H2: single-demo ('clear') fit, tuned (lr=0.01, n_steps=40) -- Study-0 theta_star")
    print("=" * 60)
    run_single_demo_tuned(prob=prob, forward_solver=forward_solver, cert=cert)
