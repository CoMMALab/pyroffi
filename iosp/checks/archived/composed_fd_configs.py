"""Does the single-segment finding about the soft flags hold on the COMPOSED chain?

`ioc.diagnostics`'s roughness probe measured, on one segment:

    early_stop=False only            rough/|g| = 72.2   FD cos = 0.906
    + soft_line_search + soft_gate   rough/|g| = 17.7   FD cos = 0.324

The second row is `iosp.pickplace.make_composed_forward_solver`'s default, which
stacked both soft flags while chasing the composed chain's cos = -0.71.  If the
composition behaves like the segment, turning the soft flags OFF should RAISE
cos, and the stacking was counterproductive -- but a 4-segment chain is not one
segment and that has to be measured, not assumed.

Sweeps {both soft flags on, both off} x {n_iters}, reusing
`study1_diagnostic_fd_check`'s exact K=2 loss and central-difference check so
the numbers are comparable to the cos = -0.71 already on record.  Runs in
float32, matching study1 (alpha0 is float32 there); the eps ladder is the guard
against reading a float-noise-dominated FD value.
"""
import jax, jax.numpy as jnp, numpy as np

from iosp import pickplace as pp
from iosp.study1_minimal_identifiable import (
    CURATED_SCENES, URDF_PATH, SRDF_PATH, MESH_DIR,
    _TRANSPORT_IDX, _make_scene, _setup, _solve_all,
    run_certificate, select_rank, theta_from_alpha_zero_prior,
)


def central_diff_grad(loss, alpha0, eps):
    k = alpha0.shape[0]
    g = np.zeros(k)
    for i in range(k):
        d = jnp.asarray(np.eye(k)[i] * eps, dtype=alpha0.dtype)
        g[i] = (float(loss(alpha0 + d)) - float(loss(alpha0 - d))) / (2 * eps)
    return g


prob = pp.PickPlaceProblem.load(str(URDF_PATH), str(SRDF_PATH), str(MESH_DIR))
alpha0 = jnp.array([0.2, 0.2], dtype=jnp.float32)

print(f"{'config':28}{'n_iters':>8}{'|g_adj|':>12}{'best cos':>10}{'at eps':>9}", flush=True)
for soft in (True, False):
    for n_iters in (60, 300, 1000):
        fs = pp.make_composed_forward_solver(n_iters=n_iters, soft_line_search=soft,
                                             soft_curvature_gate=soft)
        cert = run_certificate(seed=0, whiten=True, scene_specs=CURATED_SCENES,
                               prob=prob, forward_solver=fs)
        k, sel = select_rank(cert["eigvals"], 0.95)

        scenes, th_star, x0_star, inner_by_phase, _, _ = _setup(
            prob, fs, _make_scene(CURATED_SCENES["clear"]), seed=0)
        xs_gt, ps_gt = _solve_all(prob, scenes, inner_by_phase, th_star, x0_star)
        demo_path = prob.full_ee_path(scenes, xs_gt, ps_gt, batch_index=0)

        def loss(alpha):
            th_t = theta_from_alpha_zero_prior(alpha, cert["eigvecs"], sel)
            th = th_star.at[_TRANSPORT_IDX].set(th_t)
            xs, ps = _solve_all(prob, scenes, inner_by_phase, th, x0_star)
            return jnp.mean(jnp.sum((prob.full_ee_path(scenes, xs, ps, batch_index=0)
                                     - demo_path) ** 2, axis=-1))

        g = np.asarray(jax.grad(loss)(alpha0))
        best, best_eps = -2.0, None
        for eps in (1e-2, 1e-3, 3e-4):
            gfd = central_diff_grad(loss, alpha0, eps)
            c = float(np.dot(g, gfd) / (np.linalg.norm(g) * np.linalg.norm(gfd) + 1e-30))
            if c > best:
                best, best_eps = c, eps
        label = "soft flags ON (current)" if soft else "soft flags OFF"
        print(f"{label:28}{n_iters:>8}{np.linalg.norm(g):>12.4e}{best:>10.4f}"
              f"{best_eps:>9.0e}", flush=True)
