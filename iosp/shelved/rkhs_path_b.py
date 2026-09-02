"""Path B (unknown cost, RKHS) -- SHELVED, kept for the record.

DECISION (recorded 2026-08-28): the unknown-cost path is out of scope for the
current paper and is its own future work.  It is kept here, out of
`experiments/`, because the negative result is worth preserving and the code is
the evidence for it: path B fails from MISSPECIFICATION (the clearance term),
not from optimization, and the geometry fix that was supposed to close the 35x
gap ran and did not close it.

Two live gotchas if this is ever picked back up, both measured:
  * the transport segment's inner solve does NOT reach stationarity here
    (worst ||grad_x C|| 1.3e-3 against a 1e-3 tolerance), so any sensitivity
    spectrum computed from it is untrustworthy -- screen first;
  * the random-Fourier residuals NaN under some lengthscale/mode combinations.

Nothing in `iosp` imports this module.
"""

import jax
import jax.numpy as jnp
import numpy as np

from ioc import identifiability as ident
from ioc.inner import make_inner_solver
from iosp import config
from iosp.config import THETA_IK_STAR, Z_TRAJOPT_STAR, URDF_PATH, SRDF_PATH, MESH_DIR
from iosp.fit.params import z_scale
from iosp.fit.parametric import _build_inner, screen_stationarity
from iosp.model import pickplace as pp
from iosp.model.pickplace import split_trajopt as _split_trajopt
from iosp.model.scenes import scene_a, scenes_ab

N_STEPS, LR, TRACE_FRAC = config.N_STEPS, config.LR, config.TRACE_FRAC

# ---------------------------------------------------------------------------
# Path B -- unknown cost, RKHS random features (theory doc §3)
# ---------------------------------------------------------------------------

def make_rff_residual_fn(problem, M, key, lengthscale=1.0):
    """`M` random-Fourier-feature residuals of a squared-exponential kernel on
    the per-waypoint descriptor `u_t = [q_t, q_{t+1} - q_t]`.

    Rahimi & Recht: for `omega ~ N(0, l^-2 I)`, `b ~ U[0, 2pi)`,
    `E[phi_j(u) phi_j(u')] = k(u, u')`, so `sum_j w_j sum_t phi_j(u_t)^2` is an
    unbiased M-term expansion of a generic RKHS running cost -- weights on the
    simplex keep it non-negative and scale-pinned, exactly as the named
    library's `theta_trajopt` is.

    Returned in `residual_fn(x_flat, scene) -> tuple` form, so it drops into
    `make_inner_solver` with nothing else changed.
    """
    k1, k2 = jax.random.split(key)
    dof = problem.dof
    Omega = jax.random.normal(k1, (M, 2 * dof), dtype=jnp.float32) / lengthscale
    b = jax.random.uniform(k2, (M,), dtype=jnp.float32) * 2.0 * jnp.pi
    scale = jnp.sqrt(2.0 / M)

    def residual_fn(x_flat, scene):
        q = problem.unpack(x_flat, scene)
        u = jnp.concatenate([q[:-1], q[1:] - q[:-1]], axis=-1)  # (T-1, 2*dof)
        phi = scale * jnp.cos(u @ Omega.T + b)                  # (T-1, M)
        return tuple(phi[:, j] for j in range(M))

    return residual_fn


def build_rkhs(M=16, seed=0, lengthscale=1.0):
    """`transport`'s named residuals replaced by `M` kernel features; the other
    three phases and `theta_ik` stay pinned at ground truth (the unknown-cost
    question is asked about the segment that carries the trajectory content --
    `identifiability_check` measured the other three phases' gradients at
    ~1e-4, so nothing is learnable there from this demo anyway)."""
    prob = pp.PickPlaceProblem.load(str(URDF_PATH), str(SRDF_PATH), str(MESH_DIR))
    forward_solver = pp.make_composed_forward_solver(n_iters=60)
    scenes = scenes_ab()
    theta_trajopt_star = jax.nn.softmax(Z_TRAJOPT_STAR)
    star_by_phase = _split_trajopt(theta_trajopt_star)
    transport_mass = float(jnp.sum(star_by_phase["transport"]))

    rff = make_rff_residual_fn(prob.seg["transport"], M, jax.random.PRNGKey(seed + 7),
                               lengthscale=lengthscale)

    # Two inner sets: the NAMED cost generates the demo (it is the unknown
    # ground truth), the KERNEL cost is what the fit gets to use.  Both
    # calibrated on scene A only -- see `build_parametric`.
    inner_star, _ = _build_inner(prob, scene_a(), THETA_IK_STAR, forward_solver, seed)
    inner_k, _ = _build_inner(prob, scene_a(), THETA_IK_STAR, forward_solver, seed,
                              {"transport": rff})

    def _roll(by_phase, inner):
        x0, _, _, _ = prob.seeds(scenes, THETA_IK_STAR)
        _, _, xs, ps = prob.solve(THETA_IK_STAR, by_phase, scenes, inner, x0)
        return jnp.stack([prob.full_ee_path(scenes, xs, ps, batch_index=i) for i in (0, 1)])

    demo = jax.block_until_ready(jax.jit(lambda: _roll(star_by_phase, inner_star))())

    # Screened on the KERNEL cost at its uniform-weight init: the 60-iteration
    # budget was calibrated on the named cost, and the RFF surface is a
    # different one -- exactly the case confound 4 exists to catch.
    k_by_phase = dict(star_by_phase)
    k_by_phase["transport"] = transport_mass * jnp.full((M,), 1.0 / M, dtype=jnp.float32)
    screen_stationarity(prob, scenes, inner_k, THETA_IK_STAR, k_by_phase, "path B (RKHS)")

    def paths(w):
        by_phase = dict(star_by_phase)
        # keep transport's TOTAL weight at its ground-truth mass so the kernel
        # cost competes with the other phases on the same scale the named
        # `transport.*` block did -- only its internal shape is learned.
        by_phase["transport"] = transport_mass * jax.nn.softmax(w)
        return _roll(by_phase, inner_k)

    paths_j = jax.jit(paths)

    def loss_a(w):
        return jnp.mean(jnp.sum((paths(w)[0] - demo[0]) ** 2, axis=-1))

    def rmse(w, i):
        P = paths_j(w)
        return float(jnp.sqrt(jnp.mean(jnp.sum((P[i] - demo[i]) ** 2, axis=-1))))

    return dict(
        gf=jax.jit(jax.value_and_grad(loss_a)),
        # exposed so a recorder can plot the SAME rollout the gradient is
        # computed from; without it a caller rebuilding `paths` itself can
        # silently pair a restart-enabled rollout with a no-restart gradient.
        paths_fn=paths_j, demo_paths=demo,
        jac_fn=ident.make_jac_fn(lambda w: paths(w)[0]),
        rmse_a=lambda w: rmse(w, 0),
        rmse_b=lambda w: rmse(w, 1),
        u_star=None, K=M, n_ik=0,  # all coordinates are logits: z_scale == 1
        theta_of=lambda w: np.asarray(jax.nn.softmax(w)),
        theta_star=None,
        names=[f"kernel[{j}]" for j in range(M)],
    )


# ---------------------------------------------------------------------------
# Path B-aligned -- Path A with ONLY the transport cost basis swapped
# ---------------------------------------------------------------------------
#
# `build_rkhs` above answers the MISSPECIFIED question: the demo comes from the
# named cost, the fit gets kernel features, and the two are different function
# classes.  That is the real IOC question, but it makes B's RMSE incomparable
# to A's -- A fits a demo its own parametrisation generated (a zero-loss
# optimum exists), B chases a target provably outside its class (measured
# held-out R^2 0.57 at the original config, `scratch/logs/rkhs_probe.log`).
#
# This builder removes every difference except the basis swap:
#
#   D1  demo is rolled through the SAME `inner` that the fit uses, at a
#       ground-truth `u_star` -- so a zero-loss optimum exists, as in A.
#   D2  theta_ik AND all four phases' weights are free, as in A.  (`build_rkhs`
#       pinned theta_ik and three phases at ground truth.)
#   D3  ONE softmax across all trajopt logits, so transport's total mass floats
#       exactly as in A.  (`build_rkhs` pinned it at `transport_mass`.)
#   D4  K = K_IK + (K_TRAJOPT - 3) + M.  Exactly matched to A's K=9 at M=3;
#       otherwise report r/K as a fraction.
#
# It also restores the parameter-space metrics: `build_rkhs` passed
# `theta_star=None`, which silently skipped ||dtheta||, the top-r/null split
# and `captured_frac`.  With a ground-truth `u_star` those are all defined.

_N_NONTRANSPORT = pp.K_TRAJOPT - len(pp.SEGMENT_FEATURES["transport"])


def _rff_residual_fn(problem, M, key, ls=1.0, mode="base", form="sq"):
    """`make_rff_residual_fn` generalised over descriptor and feature form.

    `mode="base", form="sq", M=16, ls=1.0` reproduces it exactly.

    `mode="geom"` appends per-link distance to the obstacle to the descriptor:
    generic scene geometry, NOT the clearance cost term (no margin, no
    softplus, no soft-min).  The probe measured the joint-only descriptor at
    held-out R^2 0.21 on the clearance term, which carries 63% of transport's
    weight -- that term is a near-hinge on Cartesian distances and is simply
    not reachable from `[q, dq]` through a smooth SE kernel.

    `form="lin"` uses `1 - cos` instead of `cos^2`, written as the residual
    `sqrt(2) sin(z/2)` via `1 - cos(z) = 2 sin^2(z/2)`.  Writing it that way
    keeps the residual SMOOTH: the naive `sqrt(max(1 - cos, 0))` has infinite
    derivative at each of its zeros and NaNs the Gauss-Newton solve.
    """
    k1, k2 = jax.random.split(key)
    dof = problem.dof
    dim = 2 * dof + (13 if mode == "geom" else 0)
    Omega = jax.random.normal(k1, (M, dim), dtype=jnp.float32) / ls
    b = jax.random.uniform(k2, (M,), dtype=jnp.float32) * 2.0 * jnp.pi
    amp = jnp.sqrt(2.0 / M)

    def residual_fn(x_flat, scene):
        q = problem.unpack(x_flat, scene)
        u = jnp.concatenate([q[:-1], q[1:] - q[:-1]], axis=-1)
        if mode == "geom":
            p_l = problem.robot.forward_kinematics(q)[..., 4:7]
            d = jnp.linalg.norm(p_l - scene.obs_center, axis=-1)
            u = jnp.concatenate([u, d[:-1]], axis=-1)
        z = u @ Omega.T + b
        ph = amp * jnp.cos(z) if form == "sq" else jnp.sqrt(2.0) * jnp.sin(0.5 * z)
        return tuple(ph[:, j] for j in range(M))

    return residual_fn


def _split_trajopt_m(theta_trajopt, M):
    """`_split_trajopt` with transport widened from 3 named features to M."""
    out, i = {}, 0
    for p in pp.PHASES:
        n = M if p == "transport" else len(pp.SEGMENT_FEATURES[p])
        out[p] = theta_trajopt[i : i + n]
        i += n
    return out


def _z_trajopt_star_m(M, seed):
    """A ground-truth logit vector for the widened basis.

    Non-transport logits are copied from `Z_TRAJOPT_STAR` unchanged, so the
    three untouched phases have EXACTLY Path A's ground truth.  The M transport
    logits are a fixed random draw shifted by a constant `c` chosen so that
    transport's softmax mass equals Path A's (0.4482) -- the swapped block
    carries the same total authority as the block it replaced, which is what
    makes the two paths' `theta_star` comparable.

    Closed form: with non-transport logits `a` and transport logits `c + eps`,
        mass = e^c E / (S + e^c E),  S = sum e^a,  E = sum e^eps
        =>  c = log( m S / ((1 - m) E) ).
    """
    z_a = np.asarray(Z_TRAJOPT_STAR, dtype=np.float64)
    star = np.asarray(jax.nn.softmax(Z_TRAJOPT_STAR), dtype=np.float64)
    i0 = sum(len(pp.SEGMENT_FEATURES[p]) for p in pp.PHASES[: pp.PHASES.index("transport")])
    n_t = len(pp.SEGMENT_FEATURES["transport"])
    m = float(star[i0 : i0 + n_t].sum())          # 0.4482, Path A's transport mass

    a = np.concatenate([z_a[:i0], z_a[i0 + n_t :]])
    eps = np.random.default_rng(seed).normal(size=M)
    eps -= eps.mean()                              # gauge: centred within the block
    c = np.log(m * np.exp(a).sum() / ((1.0 - m) * np.exp(eps).sum()))
    z_t = c + eps

    z = np.concatenate([z_a[:i0], z_t, z_a[i0:][n_t:]])
    return jnp.asarray(z, dtype=jnp.float32), m


def build_rkhs_aligned(M=64, ls=3.0, mode="base", form="sq", seed=0):
    prob = pp.PickPlaceProblem.load(str(URDF_PATH), str(SRDF_PATH), str(MESH_DIR))
    forward_solver = pp.make_composed_forward_solver(n_iters=60)
    scenes = scenes_ab()

    rff = _rff_residual_fn(prob.seg["transport"], M, jax.random.PRNGKey(seed + 7),
                           ls, mode, form)
    # identical to build_parametric's call, plus the transport override
    inner, _ = _build_inner(prob, scene_a(), THETA_IK_STAR, forward_solver, seed,
                            {"transport": rff})

    z_traj_star, mass = _z_trajopt_star_m(M, seed)
    theta_trajopt_star = jax.nn.softmax(z_traj_star)
    K = pp.K_IK + _N_NONTRANSPORT + M
    S = z_scale(K, pp.K_IK)

    def paths(u):
        z = S * u
        theta_ik, z_traj = z[: pp.K_IK], z[pp.K_IK :]
        x0, _, _, _ = prob.seeds(scenes, theta_ik)
        _, _, xs, ps = prob.solve(theta_ik, _split_trajopt_m(jax.nn.softmax(z_traj), M),
                                  scenes, inner, x0)
        return jnp.stack([prob.full_ee_path(scenes, xs, ps, batch_index=i) for i in (0, 1)])

    z_star = jnp.concatenate([THETA_IK_STAR, z_traj_star])
    u_star = z_star / S
    paths_j = jax.jit(paths)
    # D1: demo through the SAME `inner` the fit uses, exactly as build_parametric
    demo = jax.block_until_ready(paths_j(u_star))

    label = f"path B-aligned (M={M}, ls={ls}, {mode}/{form})"
    print(f"  [{label}] transport mass at ground truth = {mass:.4f} "
          f"(Path A: same); K = {K}", flush=True)
    screen_stationarity(prob, scenes, inner, THETA_IK_STAR,
                        _split_trajopt_m(theta_trajopt_star, M), label)

    def loss_a(u):
        return jnp.mean(jnp.sum((paths(u)[0] - demo[0]) ** 2, axis=-1))

    def rmse(u, i):
        P = paths_j(u)
        return float(jnp.sqrt(jnp.mean(jnp.sum((P[i] - demo[i]) ** 2, axis=-1))))

    def theta_of(u):
        z = np.asarray(S) * np.asarray(u)
        return np.concatenate([z[: pp.K_IK], np.asarray(jax.nn.softmax(z[pp.K_IK :]))])

    names = list(pp.THETA_IK_NAMES)
    for p in pp.PHASES:
        names += ([f"kernel[{j}]" for j in range(M)] if p == "transport"
                  else [f"{p}.{f}" for f in pp.SEGMENT_FEATURES[p]])

    return dict(
        gf=jax.jit(jax.value_and_grad(loss_a)),
        # exposed so a recorder can plot the SAME rollout the gradient is
        # computed from; without it a caller rebuilding `paths` itself can
        # silently pair a restart-enabled rollout with a no-restart gradient.
        paths_fn=paths_j, demo_paths=demo,
        jac_fn=ident.make_jac_fn(lambda u: paths(u)[0]),
        rmse_a=lambda u: rmse(u, 0),
        rmse_b=lambda u: rmse(u, 1),
        u_star=u_star, K=K, n_ik=pp.K_IK,
        theta_of=theta_of,
        theta_star=np.concatenate([np.asarray(THETA_IK_STAR),
                                   np.asarray(theta_trajopt_star)]),
        names=names,
        label=label,
    )


# ---------------------------------------------------------------------------
# Same-demo pair -- ONE demonstration, both hypothesis classes
# ---------------------------------------------------------------------------
#
# `build_rkhs_aligned` makes B structurally identical to A, but to do that it
# gives B its OWN demo (rolled from a kernel ground truth).  Two different
# targets means two different RMSE references, so the numbers are not directly
# comparable -- only each path's reduction against its own init is.
#
# This builder pins the demonstration instead.  ONE demo, generated once by the
# NAMED cost at Path A's ground truth, is handed to both paths.  Both fit the
# same trajectory and both report RMSE against it, so the fit/gen numbers are
# directly comparable on a common scale.
#
# The trade is forced and worth stating plainly: a demo is generated by exactly
# one cost, so whichever class generated it is realizable and the other is not.
# Here A is realizable (its own parametrisation made the demo) and B is
# misspecified.  That asymmetry is NOT a protocol confound to be engineered
# away -- it is the standard IOC benchmark shape (one dataset, several
# hypothesis classes) and the misspecification penalty is exactly the quantity
# being measured.  Every OTHER difference (D2 free parameters, D3 floating
# transport mass, D4 K, D5 gauge/metrics) stays aligned.
#
# B has no kernel ground truth here, so its `theta_star`/`u_star` are None and
# the parameter-space metrics are skipped for B -- there is no true kernel
# weight vector to measure against.  The RMSE comparison, which is the point,
# is unaffected.


def build_same_demo(M=256, ls=10.0, mode="geom", form="lin", seed=0,
                    n_fit=3, n_gen=2):
    """-> (built_A, built_B), both fitting ONE shared demo on the SAME scenes.

    Held constant between the two paths -- everything except the cost basis:
    problem, URDF/SRDF/meshes, forward solver and its 60 iterations, the scene
    set and its fit/held-out split, the demonstration itself, per-feature
    calibration (scene 0 only, same seed), the free-parameter structure
    (theta_ik + all four phases, one softmax so every phase's mass floats), the
    `z = S * u` scaling, the gauge convention, and the optimizer budget
    (N_STEPS, LR).  The ONLY difference is what spans transport's cost:
    3 named residuals (A) vs M kernel features (B).

    `K` necessarily differs (9 vs 6+M) -- that IS the known/unknown
    distinction, not a confound to remove.  It does mean equal N_STEPS is a
    different per-parameter budget; run `samedemo:3:...` for a K-matched
    control if that attribution matters.
    """
    prob = pp.PickPlaceProblem.load(str(URDF_PATH), str(SRDF_PATH), str(MESH_DIR))
    forward_solver = pp.make_composed_forward_solver(n_iters=60)
    scenes = _scenes_multi(n_fit, n_gen, seed)
    n_tot = n_fit + n_gen
    scene0 = jax.tree.map(lambda a: a[:1], scenes)   # calibration context, both paths

    theta_trajopt_star = jax.nn.softmax(Z_TRAJOPT_STAR)

    def _paths_fn(inner, split_fn, S):
        def paths(u):
            z = S * u
            theta_ik, z_traj = z[: pp.K_IK], z[pp.K_IK :]
            x0, _, _, _ = prob.seeds(scenes, theta_ik)
            _, _, xs, ps = prob.solve(theta_ik, split_fn(jax.nn.softmax(z_traj)),
                                      scenes, inner, x0)
            return jnp.stack([prob.full_ee_path(scenes, xs, ps, batch_index=i)
                              for i in range(n_tot)])
        return paths

    # --- Path A: named library ---------------------------------------------
    inner_a, _ = _build_inner(prob, scene0, THETA_IK_STAR, forward_solver, seed)
    K_a = pp.K_IK + pp.K_TRAJOPT
    S_a = z_scale(K_a, pp.K_IK)
    paths_a = _paths_fn(inner_a, _split_trajopt, S_a)
    paths_a_j = jax.jit(paths_a)
    u_star_a = jnp.concatenate([THETA_IK_STAR, Z_TRAJOPT_STAR]) / S_a

    # THE demonstration: one named-cost rollout on every scene, shared by both.
    demo = jax.block_until_ready(paths_a_j(u_star_a))

    # --- Path B: kernel basis on transport ---------------------------------
    rff = _rff_residual_fn(prob.seg["transport"], M, jax.random.PRNGKey(seed + 7),
                           ls, mode, form)
    inner_b, _ = _build_inner(prob, scene0, THETA_IK_STAR, forward_solver, seed,
                              {"transport": rff})
    K_b = pp.K_IK + _N_NONTRANSPORT + M
    S_b = z_scale(K_b, pp.K_IK)
    paths_b = _paths_fn(inner_b, lambda t: _split_trajopt_m(t, M), S_b)
    paths_b_j = jax.jit(paths_b)

    fit_idx, gen_idx = np.arange(n_fit), np.arange(n_fit, n_tot)

    def _mk(paths, paths_j, K, S, label, u_star, theta_star, names):
        def loss_a(u):
            d = paths(u)[:n_fit] - demo[:n_fit]
            return jnp.mean(jnp.sum(d ** 2, axis=-1))

        def rmse(u, idx):
            d = paths_j(u)[idx] - demo[idx]
            return float(jnp.sqrt(jnp.mean(jnp.sum(d ** 2, axis=-1))))

        def theta_of(u):
            z = np.asarray(S) * np.asarray(u)
            return np.concatenate([z[: pp.K_IK], np.asarray(jax.nn.softmax(z[pp.K_IK :]))])

        return dict(
            gf=jax.jit(jax.value_and_grad(loss_a)),
            jac_fn=ident.make_jac_fn(lambda u: paths(u)[:n_fit]),
            rmse_a=lambda u: rmse(u, fit_idx), rmse_b=lambda u: rmse(u, gen_idx),
            u_star=u_star, K=K, n_ik=pp.K_IK, theta_of=theta_of,
            theta_star=theta_star, names=names, label=label,
        )

    names_b = list(pp.THETA_IK_NAMES)
    for ph in pp.PHASES:
        names_b += ([f"kernel[{j}]" for j in range(M)] if ph == "transport"
                    else [f"{ph}.{f}" for f in pp.SEGMENT_FEATURES[ph]])

    built_a = _mk(paths_a, paths_a_j, K_a, S_a, "path A (named)", u_star_a,
                  np.concatenate([np.asarray(THETA_IK_STAR),
                                  np.asarray(theta_trajopt_star)]),
                  list(pp.THETA_IK_NAMES) + list(pp.THETA_TRAJOPT_NAMES))
    built_b = _mk(paths_b, paths_b_j, K_b, S_b,
                  f"path B (RKHS M={M}, ls={ls}, {mode}/{form})", None, None, names_b)

    print(f"  [same-demo] {n_fit} fit + {n_gen} held-out scenes; ONE named-cost "
          f"demo; A: K={K_a}, B: K={K_b}", flush=True)
    for built, inner, split in ((built_a, inner_a, _split_trajopt),
                                (built_b, inner_b, lambda t: _split_trajopt_m(t, M))):
        w = (theta_trajopt_star if built is built_a
             else jax.nn.softmax(jnp.zeros(built["K"] - pp.K_IK)))
        screen_stationarity(prob, scenes, inner, THETA_IK_STAR, split(w),
                            built["label"])
    return built_a, built_b


