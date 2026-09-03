"""E1c — FD-vs-implicit-adjoint gradient check on E1's K=3 loss.

Verifies the implicit adjoint direction against central finite differences at
the canonical alpha0.
"""

import jax
import jax.numpy as jnp
import numpy as np

from iosp.model import pickplace as pp
from iosp.experiments.e1_minimal_identifiable import (
    CURATED_SCENES, URDF_PATH, SRDF_PATH, MESH_DIR,
    _TRANSPORT_IDX, _make_scene, _setup, _solve_all,
    run_certificate, select_rank, theta_from_alpha_zero_prior,
)


def central_diff_grad(loss, alpha0, eps=1e-3):
    k = alpha0.shape[0]
    g = np.zeros(k)
    for i in range(k):
        d = np.zeros(k); d[i] = eps
        d = jnp.asarray(d, dtype=jnp.float32)
        f_plus = float(loss(alpha0 + d))
        f_minus = float(loss(alpha0 - d))
        g[i] = (f_plus - f_minus) / (2 * eps)
    return g


if __name__ == "__main__":
    prob = pp.PickPlaceProblem.load(str(URDF_PATH), str(SRDF_PATH), str(MESH_DIR))
    forward_solver = pp.make_composed_forward_solver(n_iters=60)
    cert = run_certificate(seed=0, whiten=True, scene_specs=CURATED_SCENES,
                            prob=prob, forward_solver=forward_solver)
    k, selected_idx = select_rank(cert["eigvals"], 0.95)

    fit_scene = _make_scene(CURATED_SCENES["clear"])
    scenes, theta_trajopt_star, x0_star, inner_by_phase, _, _ = _setup(
        prob, forward_solver, fit_scene, seed=0)
    theta_transport_star = theta_trajopt_star[_TRANSPORT_IDX]
    theta_trajopt_star = theta_trajopt_star.at[_TRANSPORT_IDX].set(theta_transport_star)
    xs_gt, ps_gt = _solve_all(prob, scenes, inner_by_phase, theta_trajopt_star, x0_star)
    demo_path = prob.full_ee_path(scenes, xs_gt, ps_gt, batch_index=0)

    def loss(alpha):
        theta_transport = theta_from_alpha_zero_prior(alpha, cert["eigvecs"], selected_idx)
        theta_trajopt = theta_trajopt_star.at[_TRANSPORT_IDX].set(theta_transport)
        xs, ps = _solve_all(prob, scenes, inner_by_phase, theta_trajopt, x0_star)
        path = prob.full_ee_path(scenes, xs, ps, batch_index=0)
        return jnp.mean(jnp.sum((path - demo_path) ** 2, axis=-1))

    alpha0 = jnp.array([0.2, 0.2], dtype=jnp.float32)
    g_implicit = np.asarray(jax.grad(loss)(alpha0))
    print(f"implicit-adjoint grad @ alpha0=[0.2,0.2]: {g_implicit}")

    for eps in (1e-2, 1e-3, 3e-4):
        g_fd = central_diff_grad(loss, alpha0, eps=eps)
        cos = float(np.dot(g_implicit, g_fd) / (np.linalg.norm(g_implicit) * np.linalg.norm(g_fd) + 1e-30))
        print(f"  FD grad (eps={eps:.0e}): {g_fd}   cos(implicit, FD) = {cos:.4f}")
