"""Eigen-projected recovery error: separates "wrong along an identifiable
direction" from "wrong along a structurally near-null direction."

Motivation (see conversation this script was born from): `study0_segment_
ablation`'s `free_param_err` is a raw L2 norm over ALL free theta entries
lumped together. `identifiability_check.py` already found that this 9-dim
feature set has a near-exact collinearity (`transport.smooth` vs `transport.
upright`, cos=-0.9999) and several near-zero-gradient directions -- so a
flat/overfit direction and a genuinely-recovered direction contribute to that
raw norm identically, and the metric can't tell them apart. If the optimizer
is legitimately overfitting the demo along the null direction (the outer loss
genuinely cannot see it) while correctly recovering the identifiable part,
raw L2 error looks uniformly bad even though recovery is working exactly as
well as the data structurally allows.

This script:
  1. Rebuilds the feature-gradient Gram matrix `G` (same construction as
     `identifiability_check.py::run`) on the SAME anchored scene `study0_
     segment_ablation.anchor_obstacle_to_transport` builds -- the spectrum
     may have shifted now that `transport.clearance` is actually engaged.
  2. Selects the top-k "identifiable" eigendirections via the SAME 95%-
     cumulative-trace rule `study1_minimal_identifiable.py` uses.
  3. Runs the all-free (`free_ik=True`, all 4 phases) recovery fit on that
     same anchored scene (`study0_segment_ablation.run_ablation`, unmodified)
     and projects `theta_hat - theta_star` onto the top-k subspace and the
     null subspace SEPARATELY, instead of reporting one raw L2 norm.

Prediction being tested: if the null-space-overfitting story is right, the
null-subspace-projected error should be large (comparable to the raw error)
while the top-k-projected error should be small -- i.e. the part of theta
the data can actually see IS being recovered, and the raw metric was hiding
that under the part it structurally cannot.

Theory: `iosp/THEORY_IDENTIFIABLE_REFIT.md`.  This script covers stages 2, 3
and 5 of that document's procedure (build the Gram matrix, select r by the
95%-cumulative-trace rule, report error in the `U_r` projection rather than as
a raw ||theta_hat - theta_star||).  It does NOT do stage 4 -- re-parametrizing
onto `U_r` and refitting -- so the recovery fit it measures is still the
all-K fit, free to wander in the null component.  Adding that refit is the
proposed fix for the 7-DOF reconstruction residuals (§4 of the theory doc).

Usage:
    CUDA_VISIBLE_DEVICES=<idx> XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        python -m iosp.experiments.e0d_eigen_projection
"""

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np

from ioc.inner import make_inner_solver
from iosp.model import pickplace as pp
from iosp.config import THETA_IK_STAR, Z_TRAJOPT_STAR
from iosp.model.pickplace import split_trajopt as _split_trajopt
from iosp.config import MESH_DIR, PICK_POS, PLACE_POS, Q_START, SRDF_PATH, URDF_PATH
from iosp.experiments.e0_segment_ablation import anchor_obstacle_to_transport
from iosp.experiments.e0_segment_ablation import build_common
from iosp.experiments.e0_segment_ablation import run_ablation

EPS_IK = 0.02


def _embed(n_X, offsets, dims, phase, vec):
    full = jnp.zeros(n_X)
    return full.at[offsets[phase] : offsets[phase] + dims[phase]].set(vec)


def _concat_X(xs):
    return jnp.concatenate([xs[p][0] for p in pp.PHASES])


def compute_gram(prob, scenes, forward_solver, theta_ik_star, theta_trajopt_star,
                  seed=0, eps_ik=EPS_IK):
    """Same construction as `identifiability_check.run`, parameterized by
    `scenes` instead of the hardcoded module-level constants -- see this
    module's docstring for why that matters here."""
    x0_star, phase_scenes_star, _, _ = prob.seeds(scenes, theta_ik_star)
    inner_by_phase, residual_fn_by_phase = {}, {}
    for p in pp.PHASES:
        residual_fn, _ = prob.make_segment_inner(p, forward_solver)
        residual_fn_by_phase[p] = residual_fn
        scales = prob.calibrate_segment(p, residual_fn, phase_scenes_star[p], jax.random.PRNGKey(seed))
        inner_by_phase[p] = make_inner_solver(residual_fn, scales, forward_solver=forward_solver)

    def solve_all(theta_ik, theta_trajopt):
        theta_trajopt_by_phase = _split_trajopt(theta_trajopt)
        x0, phase_scenes, _, _ = prob.seeds(scenes, theta_ik)
        _, _, xs, phase_scenes2 = prob.solve(theta_ik, theta_trajopt_by_phase, scenes, inner_by_phase, x0)
        return xs, phase_scenes2

    print("  [gram] solving at theta_star (eager, uncompiled -- expect ~1-2 min, not 20)...", flush=True)
    xs_star, ps_star = solve_all(theta_ik_star, theta_trajopt_star)

    dims = {p: xs_star[p][0].shape[0] for p in pp.PHASES}
    offsets, o = {}, 0
    for p in pp.PHASES:
        offsets[p] = o
        o += dims[p]
    n_X = o

    columns, grad_norms = {}, {}
    for phase in pp.PHASES:
        sc = jax.tree.map(lambda a: a[0], ps_star[phase])
        residual_fn = residual_fn_by_phase[phase]
        x_star_phase = xs_star[phase][0]
        for idx, name in enumerate(pp.SEGMENT_FEATURES[phase]):
            def phi(x, idx=idx, residual_fn=residual_fn, sc=sc):
                return jnp.sum(residual_fn(x, sc)[idx] ** 2)
            g = jax.grad(phi)(x_star_phase)
            key = f"{phase}.{name}"
            columns[key] = _embed(n_X, offsets, dims, phase, g)
            grad_norms[key] = float(jnp.linalg.norm(g))

    print("  [gram] FD columns for theta_ik (2 extra eager solves)...", flush=True)
    for k, name in enumerate(pp.THETA_IK_NAMES):
        e_k = jnp.zeros(pp.K_IK).at[k].set(1.0)
        xs_plus, _ = solve_all(theta_ik_star + eps_ik * e_k, theta_trajopt_star)
        xs_minus, _ = solve_all(theta_ik_star - eps_ik * e_k, theta_trajopt_star)
        c = (_concat_X(xs_plus) - _concat_X(xs_minus)) / (2 * eps_ik)
        columns[name] = c
        grad_norms[name] = float(jnp.linalg.norm(c))

    order = list(pp.THETA_IK_NAMES) + list(pp.THETA_TRAJOPT_NAMES)
    B = jnp.stack([columns[n] for n in order], axis=-1)
    G = np.asarray(B.T @ B)
    G_normed = G / (np.trace(G) / len(order) + 1e-30)
    eigvals, eigvecs = np.linalg.eigh(G_normed)  # ascending; eigvecs[:, i] <-> eigvals[i]
    return dict(order=order, G=G, eigvals=eigvals, eigvecs=eigvecs, grad_norms=grad_norms)


def select_rank(eigvals, frac=0.95):
    """Same rule as `study1_minimal_identifiable.py`: keep the top
    eigendirections needed to reach `frac` of cumulative trace."""
    order_desc = np.argsort(eigvals)[::-1]
    ev_desc = eigvals[order_desc]
    cum = np.cumsum(ev_desc) / np.sum(ev_desc)
    k = int(np.searchsorted(cum, frac) + 1)
    return order_desc[:k], order_desc[k:], k


def project_err(delta, eigvecs, idx):
    if len(idx) == 0:
        return 0.0
    V = eigvecs[:, idx]
    return float(np.linalg.norm(V.T @ delta))


def main(out=None):
    prob = pp.PickPlaceProblem.load(str(URDF_PATH), str(SRDF_PATH), str(MESH_DIR))
    obs_center, obs_radius = anchor_obstacle_to_transport(
        prob, THETA_IK_STAR, Q_START, PICK_POS, PLACE_POS, seed=0)
    scene1 = pp.PickPlaceScene(
        q_start=Q_START, pick_pos=PICK_POS, place_pos=PLACE_POS,
        obs_center=obs_center, obs_radius=obs_radius)
    scenes = jax.tree.map(lambda a: a[None], scene1)
    forward_solver = pp.make_composed_forward_solver(n_iters=60)
    theta_trajopt_star = jax.nn.softmax(Z_TRAJOPT_STAR)

    print("Computing feature-gradient Gram matrix on the ANCHORED scene...")
    gram = compute_gram(prob, scenes, forward_solver, THETA_IK_STAR, theta_trajopt_star)
    top_idx, null_idx, k = select_rank(gram["eigvals"])
    ev_desc = gram["eigvals"][np.argsort(gram["eigvals"])[::-1]]
    print(f"eigenvalues (trace-normalized, descending): {np.round(ev_desc, 6)}")
    print(f"selected k={k} identifiable eigendirection(s) of {len(gram['order'])} (95% cumulative-trace rule)")
    print("per-feature gradient norms (X-space):")
    for n in gram["order"]:
        print(f"  {n:24s} {gram['grad_norms'][n]:.6f}")
    print()

    print("Fitting ALL-free (theta_ik + all 4 phases), single anchored demo...")
    common = build_common(seed=0)  # re-anchors with the SAME seed -> identical scene
    res = run_ablation(common, pp.PHASES, free_ik=True)

    theta_hat = np.concatenate([res["theta_ik_hat"], res["theta_trajopt_hat"]])
    theta_star_vec = np.concatenate([np.asarray(THETA_IK_STAR), np.asarray(theta_trajopt_star)])
    delta = theta_hat - theta_star_vec

    raw_err = float(np.linalg.norm(delta))
    top_err = project_err(delta, gram["eigvecs"], top_idx)
    null_err = project_err(delta, gram["eigvecs"], null_idx)

    print()
    print(f"raw free_param_err        = {raw_err:.4f}")
    print(f"top-{k} (identifiable) err = {top_err:.4f}")
    print(f"null ({len(null_idx)}-dim) err        = {null_err:.4f}")
    print(f"ee_rmse                    = {res['ee_rmse']:.4f}")
    print()
    print("(check: sqrt(top_err^2 + null_err^2) should ~= raw_err, since eigvecs are orthonormal:"
          f" {np.sqrt(top_err**2 + null_err**2):.4f})")

    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        np.savez(
            out,
            order=np.array(gram["order"], dtype=object),
            grad_norms=np.array([gram["grad_norms"][n] for n in gram["order"]]),
            G=gram["G"],
            eigvals=gram["eigvals"],
            eigvecs=gram["eigvecs"],
            top_idx=np.asarray(top_idx),
            null_idx=np.asarray(null_idx),
            k=k,
            theta_hat=theta_hat,
            theta_star=theta_star_vec,
            delta=delta,
            raw_err=raw_err,
            top_err=top_err,
            null_err=null_err,
            ee_rmse=float(res["ee_rmse"]),
            allow_pickle=True,
        )
        print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="npz path for the eigenspectrum figure")
    main(**vars(ap.parse_args()))
