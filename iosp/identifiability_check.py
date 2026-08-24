"""Identifiability certificate for the composed pick-and-place feature set.

Motivation
----------
`iosp/recovery_bench.py::sweep_demo_count` shows only weak improvement in
recovery error as the demo count grows (param_err 0.244 -> 0.216, N=1 -> 8).
`iosp/generalization_check.py` shows the generalization failure is
concentrated specifically in `transport.smooth` (~2.1-2.5x worse on a held-
out scene) while `transport.clearance` transfers fine.  A single feature
staying stubbornly unidentifiable regardless of demo count looks like a WEAK
IDENTIFIABILITY problem (near-collinearity between feature gradients) rather
than pure data scarcity -- this script checks that directly, the same way
`ioc/analytic.py::kkt_fit`'s Gram matrix serves as an identifiability
certificate for the single-segment study (see that module's docstring:
"G doubles as the study's identifiability certificate... lambda_min ~ 0 means
some combination of features leaves the demonstrations unchanged").

Method -- and where it departs from `kkt_fit`
-----------------------------------------------
`kkt_fit` builds `B(c) = [grad_x phi_1(x_demo,c), ..., grad_x phi_K(x_demo,c)]`,
the K feature gradients wrt the SAME inner decision vector x, evaluated at one
demo, then `G = B^T B`.  For the 7 trajopt features here that construction
applies exactly: each `phi_k` is a residual-sumsq of one phase's OWN free
decision vector, so `grad_x phi_k` is an ordinary `jax.grad` call, embedded
(zero-padded) into one shared 126-dim vector (`sum` of the 4 phases' free-var
counts: 42+14+56+14) so all 9 features' gradients live in the same space and
can be compared.

The 2 IK features (`grasp.standoff`, `place.standoff`) do NOT fit this
construction directly: `theta_ik` is not a KKT-stationarity weight multiplying
a residual, it's a geometric input to the IK subproblem, so there is no
`phi_grasp.standoff(x)` to differentiate wrt x.  The natural generalization is
`dX*/dtheta_ik_k` -- how much the demo's WHOLE decision state (all 4
segments' free vectors) shifts when `theta_ik_k` shifts -- treated as a
direction in the same X-space the trajopt gradients live in.

That Jacobian is NOT computed via `jax.grad`/`jax.jacrev` through the full
composed chain here: `solve_implicit` is a `jax.custom_vjp` (reverse-mode
only, per `ioc/inner.py`), and a full `dX*/dtheta_ik` Jacobian via reverse
mode would need one backward pass PER OUTPUT dimension of X* (126 of them) --
infeasible for a lightweight diagnostic.  Forward-mode (`jax.jacfwd`, which
would only need 2 passes since `theta_ik` is 2-dimensional) is unavailable
because `custom_vjp` has no attached JVP rule.  Since `theta_ik` is only
2-dimensional, this script instead uses CENTRAL FINITE DIFFERENCES directly
on the forward-solve-only pipeline (`prob.solve`, no autodiff) at
`eps=0.02`: `c_k ~= (X*(theta_ik_star + eps*e_k) - X*(theta_ik_star - eps*e_k))
/ (2*eps)`.

CAVEAT, carried over from this investigation's other FD findings on this
composed model: FD through the trajopt forward solver is measured elsewhere
(see `iosp/pickplace.py`'s module docstring) to be unstable at SMALL step
sizes -- FD estimates roughly double every time eps halves, never converging,
even after three separate hard-branch-to-soft fixes in the forward solver.
`eps=0.02` here is chosen to sit outside the worst of that regime (an order
of magnitude larger than the step sizes where the blowup was measured), which
trades some truncation error for a usable estimate, but the resulting
`grasp.standoff`/`place.standoff` columns of the Gram matrix should be read
as approximate directions, not exact gradients -- unlike the 7 trajopt
columns, which are exact.

MEASURED result (float32, `soft_line_search`+`soft_curvature_gate`+
`early_stop=False` forward solver, scene A = `recovery_bench`'s fitting
scene, seed 0; see `run()` below to reproduce):

    per-feature gradient norms (||grad|| in the shared 126-dim X-space):
        approach.smooth        0.136319
        approach.clearance     0.000147   <- ~0: demo never nears the margin
        grasp.smooth           0.000231   <- ~0: tiny in-place motion, little signal
        transport.smooth       0.637933
        transport.clearance    0.000000   <- ~0: demo never nears the margin
        transport.upright      1.432120   <- LARGE, not saturated (see below)
        place.smooth           0.000217   <- ~0: tiny in-place motion, little signal
        grasp.standoff (FD)    223.330688
        place.standoff (FD)    8.097362

    Gram eigenvalues (trace-normalized, ascending):
        5.3e-23, 6.8e-16, 8.5e-12, 9.6e-12, 1.1e-08, 3.3e-06, 4.4e-04, 1.16e-02, 8.99

    cos(transport.smooth, transport.upright) = -0.9999
    (every other pairwise cosine with transport.smooth has |cos| < 0.08)

Verdict
-------
**`transport.smooth`'s poor generalization is explained by (a) near-exact
collinearity with `transport.upright` specifically** -- cos=-0.9999 stands out
sharply from every other pair (next largest magnitude: `transport.clearance`
at -0.077).  This is NOT a near-zero-gradient/saturation problem for
`transport.upright`: its gradient (1.43) is actually LARGER than
`transport.smooth`'s own (0.64) -- superseding an earlier, wrong caveat in
`iosp/generalization_check.py` that speculated `transport.upright_max≈1.0`
meant the feature was saturated.  The mechanism is instead that `smooth` and
`upright` push the transport trajectory in nearly the same direction at this
demo, so the outer loop cannot tell how much of the observed trajectory shape
is due to one weight vs. the other.

Separately, the top of the spectrum shows a BROADER problem: only 1-2 of 9
directions carry real signal (top eigenvalue 8.99, dominated by
`grasp.standoff`; a distant second at 0.0116, dominated by `place.standoff`).
Six of nine directions are near-completely unidentifiable from this single
demo -- but for a DIFFERENT reason than the smooth/upright pair:
`approach.clearance`, `grasp.smooth`, `transport.clearance`, `place.smooth`
all have near-zero gradient outright (the demo simply never gets close enough
to the obstacle margin, or excites the tiny in-place grasp/place motions
enough, to identify those weights at all) -- genuine (b)-type near-zero-
gradient failures, not collinearity, and a separate problem from `transport.
smooth`'s (a)-type collinearity with `transport.upright`.

Practical implication: "just add more demos" of the same qualitative motion
won't resolve the `transport.smooth`/`transport.upright` collinearity -- more
demos of an upright transport near the same obstacle likely preserve the same
near-collinear direction.  What would help is a demo that specifically
decouples them (e.g. a clearance-forced detour that also tips the object, or
an explicit orientation perturbation) -- a scene-design fix, not an outer-loop
tuning knob.

Usage:
    CUDA_VISIBLE_DEVICES=<idx> XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        python -m iosp.identifiability_check
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from ioc.inner import make_inner_solver
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
)

EPS_IK = 0.02  # see module docstring's caveat on the FD approximation


def _embed(n_X, offsets, dims, phase, vec):
    full = jnp.zeros(n_X)
    return full.at[offsets[phase] : offsets[phase] + dims[phase]].set(vec)


def _concat_X(xs):
    return jnp.concatenate([xs[p][0] for p in pp.PHASES])


def run(eps_ik=EPS_IK, seed=0):
    prob = pp.PickPlaceProblem.load(str(URDF_PATH), str(SRDF_PATH), str(MESH_DIR))
    forward_solver = pp.make_composed_forward_solver(n_iters=60)

    scene_A = pp.PickPlaceScene(
        q_start=Q_START, pick_pos=PICK_POS, place_pos=PLACE_POS,
        obs_center=OBS_CENTER, obs_radius=OBS_RADIUS,
    )
    scenes_A = jax.tree.map(lambda a: a[None], scene_A)

    theta_trajopt_star = jax.nn.softmax(Z_TRAJOPT_STAR)
    x0_star, phase_scenes_star, _, _ = prob.seeds(scenes_A, THETA_IK_STAR)

    inner_by_phase, residual_fn_by_phase = {}, {}
    for p in pp.PHASES:
        residual_fn, _ = prob.make_segment_inner(p, forward_solver)
        residual_fn_by_phase[p] = residual_fn
        scales = prob.calibrate_segment(p, residual_fn, phase_scenes_star[p], jax.random.PRNGKey(seed))
        inner_by_phase[p] = make_inner_solver(residual_fn, scales, forward_solver=forward_solver)

    def solve_all(theta_ik, theta_trajopt):
        theta_trajopt_by_phase = _split_trajopt(theta_trajopt)
        x0, phase_scenes, _, _ = prob.seeds(scenes_A, theta_ik)
        _, _, xs, phase_scenes2 = prob.solve(theta_ik, theta_trajopt_by_phase, scenes_A, inner_by_phase, x0)
        return xs, phase_scenes2

    xs_star, ps_star = solve_all(THETA_IK_STAR, theta_trajopt_star)

    dims = {p: xs_star[p][0].shape[0] for p in pp.PHASES}
    offsets, o = {}, 0
    for p in pp.PHASES:
        offsets[p] = o
        o += dims[p]
    n_X = o

    columns = {}
    grad_norms = {}
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

    for k, name in enumerate(pp.THETA_IK_NAMES):
        e_k = jnp.zeros(pp.K_IK).at[k].set(1.0)
        xs_plus, _ = solve_all(THETA_IK_STAR + eps_ik * e_k, theta_trajopt_star)
        xs_minus, _ = solve_all(THETA_IK_STAR - eps_ik * e_k, theta_trajopt_star)
        c = (_concat_X(xs_plus) - _concat_X(xs_minus)) / (2 * eps_ik)
        columns[name] = c
        grad_norms[name] = float(jnp.linalg.norm(c))

    order = list(pp.THETA_IK_NAMES) + list(pp.THETA_TRAJOPT_NAMES)
    B = jnp.stack([columns[n] for n in order], axis=-1)
    G = np.asarray(B.T @ B)
    G_normed = G / (np.trace(G) / len(order) + 1e-30)
    eigvals, eigvecs = np.linalg.eigh(G_normed)

    def cos(a, b):
        a, b = np.asarray(a), np.asarray(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))

    cosines = {n: cos(columns["transport.smooth"], columns[n]) for n in order if n != "transport.smooth"}

    return dict(
        order=order, grad_norms=grad_norms, G=G, eigvals=eigvals, eigvecs=eigvecs,
        cos_transport_smooth=cosines,
    )


def _print_report(r):
    print("per-feature gradient norms:")
    for n in r["order"]:
        print(f"  {n:24s} {r['grad_norms'][n]:.6f}")
    print()
    print("Gram matrix (unnormalized):")
    print(np.array2string(r["G"], precision=4, suppress_small=True))
    print()
    print("eigenvalues (trace-normalized, ascending):", r["eigvals"])
    print()
    print("cos(transport.smooth, X):")
    for n, c in r["cos_transport_smooth"].items():
        print(f"  {n:24s} {c:.4f}")


if __name__ == "__main__":
    _print_report(run())
