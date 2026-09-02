"""Feeler for why Path B (RKHS/RFF) fits far worse than Path A (named library).

Asks ONE question, cheaply and off the bilevel path: can the hypothesis class
that `study3_identifiable_refit.make_rff_residual_fn` defines represent the
transport cost at all?  If it cannot, no amount of outer-loop tuning helps and
the 0.45 vs 0.024 RMSE gap is a modelling error, not an optimisation one.

The class is    C_w(x) = sum_j (w_j / s_j) * sum_t phi_j(u_t)^2,   w >= 0
with            phi_j(u) = sqrt(2/M) cos(omega_j . u + b_j),  u_t = [q_t, dq_t].

Target is the true transport cost   sum_k (theta*_k / s_k) ||r_k(x)||^2  over
{smooth, clearance, upright}, evaluated on jittered trajectories.

Reported per variant:
  R2_free  best-fit R^2 with UNCONSTRAINED coefficients + free offset
  R2_nneg  best-fit R^2 with w >= 0 (what the simplex parametrisation allows)
  gcos     mean cos(grad_x C_true, grad_x C_w) at the samples -- what actually
           determines where the inner solver puts x*.
"""
import sys, pathlib, time
import numpy as np
import jax, jax.numpy as jnp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from iosp import pickplace as pp
from iosp.recovery_bench import THETA_IK_STAR, Z_TRAJOPT_STAR, _split_trajopt
from iosp.study0_segment_ablation import MESH_DIR, SRDF_PATH, URDF_PATH
from iosp.study3_identifiable_refit import scene_a

N_SAMP = 768
JITTERS = (0.05, 0.15, 0.30)


_GEOM = {}


def descriptor(q, mode):
    dq = q[1:] - q[:-1]
    if mode == "geom":
        # + generic task-space geometry: per-link distance to the obstacle.
        # This is scene geometry any practitioner supplies, NOT the cost term
        # itself (no margin, no softplus, no soft-min reduction).
        prob, scene = _GEOM["prob"], _GEOM["scene"]
        p_l = prob.robot.forward_kinematics(q)[..., 4:7]
        d = jnp.linalg.norm(p_l - scene.obs_center, axis=-1)
        return jnp.concatenate([q[:-1], dq, d[:-1]], axis=-1)
    if mode == "base":                    # what the study uses today
        return jnp.concatenate([q[:-1], dq], axis=-1)
    if mode == "accel":                   # + second difference
        d2 = dq[1:] - dq[:-1]
        return jnp.concatenate([q[1:-1], dq[:-1], d2], axis=-1)
    raise ValueError(mode)


def make_feats(dim, M, key, ls, scale_vec=None, form="sq"):
    k1, k2 = jax.random.split(key)
    Omega = jax.random.normal(k1, (M, dim), dtype=jnp.float32) / ls
    if scale_vec is not None:             # per-dimension standardisation
        Omega = Omega / scale_vec
    b = jax.random.uniform(k2, (M,), dtype=jnp.float32) * 2.0 * jnp.pi
    amp = jnp.sqrt(2.0 / M)

    def phi_sq(x_flat, prob, scene, mode):
        q = prob.unpack(x_flat, scene)
        u = descriptor(q, mode)
        c = jnp.cos(u @ Omega.T + b)
        ph = amp * c if form == "sq" else (1.0 - c)
        return jnp.sum(ph ** 2 if form == "sq" else ph, axis=0)   # (M,)
    return phi_sq


def _score(y, yhat):
    return 1.0 - ((y - yhat) @ (y - yhat)) / ((y - y.mean()) @ (y - y.mean()))


def r2(y, X, tr, te):
    """Unconstrained ridge-stabilised LS with a free offset, scored HELD OUT.
    (In-sample R2 is meaningless once M approaches n.)"""
    A = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    At, Ae = A[tr], A[te]
    lam = 1e-8 * np.trace(At.T @ At) / At.shape[1]
    w = np.linalg.solve(At.T @ At + lam * np.eye(At.shape[1]), At.T @ y[tr])
    return _score(y[te], Ae @ w), w[:-1]


def r2_nneg(y, X, tr, te):
    """LS with w >= 0 and a free offset (absorbed by centring), scored HELD
    OUT -- the class the simplex/positive-mass parametrisation can express."""
    from scipy.optimize import nnls
    mu_x, mu_y = X[tr].mean(0), y[tr].mean()
    try:
        w, _ = nnls(X[tr] - mu_x, y[tr] - mu_y, maxiter=20000)
    except RuntimeError:
        return float("nan")
    return _score(y[te], (X[te] - mu_x) @ w + mu_y)


def main():
    prob = pp.PickPlaceProblem.load(str(URDF_PATH), str(SRDF_PATH), str(MESH_DIR))
    scenes = scene_a()
    seg = prob.seg["transport"]
    _, phase_scenes, _, _ = prob.seeds(scenes, THETA_IK_STAR)
    scene = jax.tree.map(lambda a: a[0], phase_scenes["transport"])
    x0 = seg.seed(scene)

    rfn = prob.segment_residual_fn("transport")
    scales = prob.calibrate_segment("transport", rfn, phase_scenes["transport"],
                                    jax.random.PRNGKey(0))
    theta_star = _split_trajopt(jax.nn.softmax(Z_TRAJOPT_STAR))["transport"]
    theta_star = theta_star / jnp.sum(theta_star)      # shape only; scale is free
    print("transport theta* (smooth, clearance, upright) =",
          np.asarray(theta_star), " scales =", np.asarray(scales))

    def c_true(x):
        rs = rfn(x, scene)
        return sum((theta_star[k] / scales[k]) * jnp.sum(r ** 2)
                   for k, r in enumerate(rs))
    c_true_j = jax.jit(c_true)
    g_true_j = jax.jit(jax.grad(c_true))

    key = jax.random.PRNGKey(1)
    xs = []
    for j in JITTERS:
        key, k = jax.random.split(key)
        xs.append(x0 + j * jax.random.normal(k, (N_SAMP // len(JITTERS),) + x0.shape))
    X = jnp.concatenate(xs, axis=0)
    y = np.asarray(jax.vmap(c_true_j)(X), dtype=np.float64)
    G_true = np.asarray(jax.vmap(g_true_j)(X))
    print(f"n={len(X)}  c_true range [{y.min():.4g}, {y.max():.4g}]")

    # descriptor scale, for the standardised variant
    q0 = seg.unpack(x0, scene)
    u_base = np.asarray(descriptor(q0, "base"))
    print("descriptor std: q-half %.4g  dq-half %.4g"
          % (u_base[:, :7].std(), u_base[:, 7:].std()))

    variants = [
        ("cos^2  base  M=16  ls=1.0  (current)", "base", 16, 1.0, False, "sq"),
        ("cos^2  base  M=64  ls=1.0",  "base", 64, 1.0, False, "sq"),
        ("cos^2  base  M=64  ls=3.0",  "base", 64, 3.0, False, "sq"),
        ("cos^2  base  M=64  ls=10",   "base", 64, 10.0, False, "sq"),
        ("cos^2  base  M=64  ls=30",   "base", 64, 30.0, False, "sq"),
        ("cos^2  base  M=256 ls=10",   "base", 256, 10.0, False, "sq"),
        ("1-cos  base  M=64  ls=3.0",  "base", 64, 3.0, False, "lin"),
        ("1-cos  base  M=64  ls=10",   "base", 64, 10.0, False, "lin"),
        ("1-cos  base  M=256 ls=10",   "base", 256, 10.0, False, "lin"),
        ("1-cos  accel M=256 ls=10",   "accel", 256, 10.0, False, "lin"),
        ("1-cos  accel M=256 ls=30",   "accel", 256, 30.0, False, "lin"),
        ("1-cos  GEOM  M=64  ls=3.0",  "geom", 64, 3.0, False, "lin"),
        ("1-cos  GEOM  M=256 ls=3.0",  "geom", 256, 3.0, False, "lin"),
        ("cos^2  GEOM  M=256 ls=3.0",  "geom", 256, 3.0, False, "sq"),
        ("1-cos  GEOM  M=256 ls=10",   "geom", 256, 10.0, False, "lin"),
    ]

    _GEOM["prob"], _GEOM["scene"] = seg, scene

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X)); ntr = int(0.7 * len(X))
    tr, te = perm[:ntr], perm[ntr:]

    # control: the NAMED library itself, scored the same way
    ctrl = np.stack([np.asarray(jax.vmap(jax.jit(
        lambda x, k=k: jnp.sum(rfn(x, scene)[k] ** 2)))(X)) for k in range(3)], 1)
    ctrl = ctrl / (np.abs(ctrl).mean(0, keepdims=True) + 1e-12)
    print("\ncontrol (named library, K=3): R2_free %.3f  R2_nneg %.3f"
          % (r2(y, ctrl, tr, te)[0], r2_nneg(y, ctrl, tr, te)))

    print(f"\n{'variant':38s} {'R2_free':>9s} {'R2_nneg':>9s} {'gcos':>8s}")
    for name, mode, M, ls, std, form in variants:
        q = seg.unpack(x0, scene)
        u = np.asarray(descriptor(q, mode))
        sv = jnp.asarray(u.std(axis=0) + 1e-6) if std else None
        phi_sq = make_feats(u.shape[1], M, jax.random.PRNGKey(7), ls, sv, form)
        f = jax.jit(lambda x: phi_sq(x, seg, scene, mode))
        Phi = np.asarray(jax.vmap(f)(X), dtype=np.float64)
        Phi = Phi / (np.abs(Phi).mean(axis=0, keepdims=True) + 1e-12)   # the /s_j whitening

        # gradient alignment at the (held-out-fit) unconstrained best fit
        rfree, wf = r2(y, Phi, tr, te)
        wj = jnp.asarray(wf / (np.abs(np.asarray(jax.vmap(f)(X))).mean(axis=0) + 1e-12))
        gfit = jax.jit(jax.grad(lambda x: jnp.sum(wj * phi_sq(x, seg, scene, mode))))
        Gf = np.asarray(jax.vmap(gfit)(X))
        num = (G_true * Gf).sum(1)
        den = np.linalg.norm(G_true, axis=1) * np.linalg.norm(Gf, axis=1) + 1e-12
        print(f"{name:38s} {rfree:9.3f} {r2_nneg(y, Phi, tr, te):9.3f} "
              f"{(num/den).mean():8.3f}", flush=True)

    # --- which NAMED term is the class unable to represent? ----------------
    print("\nper-term held-out R2 (best variant: 1-cos base M=64 ls=3.0)")
    q = seg.unpack(x0, scene)
    u = np.asarray(descriptor(q, "base"))
    for mode in ("base", "accel", "geom"):
        uu = np.asarray(descriptor(q, mode))
        phi_sq = make_feats(uu.shape[1], 64, jax.random.PRNGKey(7), 3.0, None, "lin")
        f = jax.jit(lambda x: phi_sq(x, seg, scene, mode))
        Phi = np.asarray(jax.vmap(f)(X), dtype=np.float64)
        Phi = Phi / (np.abs(Phi).mean(0, keepdims=True) + 1e-12)
        row = []
        for k, nm in enumerate(("smooth", "clearance", "upright")):
            yk = np.asarray(jax.vmap(jax.jit(
                lambda x, k=k: jnp.sum(rfn(x, scene)[k] ** 2)))(X), dtype=np.float64)
            row.append(f"{nm} free {r2(yk, Phi, tr, te)[0]:6.3f} / nneg "
                       f"{r2_nneg(yk, Phi, tr, te):6.3f}")
        print(f"  descriptor={mode:6s} " + "   ".join(row))


if __name__ == "__main__":
    main()
