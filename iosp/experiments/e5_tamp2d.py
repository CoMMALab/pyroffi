"""Study 5: the tied three-stage SPaSM inversion on a 2D TAMP benchmark.

Why this exists, and why it comes first
---------------------------------------
`study4_three_stage_pickplace.py` runs the same inversion on the 7-DOF Panda,
where the cost-sensitivity Gram is badly rank-deficient (measured elsewhere in
this suite: r = 6 of K = 9, and as low as r = 1 of 9 on a single scene).  On a
problem like that a failed recovery is uninterpretable -- it cannot be
attributed to the composed adjoint being wrong versus the parameters simply
not being identifiable from the data.

This benchmark is built so that r is as close to full as the parameterization
allows, so that the ONLY thing under test is whether the three-stage composed
implicit adjoint recovers behaviour.  If recovery fails here, the method is
wrong.  If it succeeds here and fails on the Panda, the problem is
identifiability, not the method.

The ceiling is r = K - 1, not r = K.  theta = softmax(z) has an exact null
direction (adding a constant to every z leaves theta unchanged), so one
eigenvalue of the Gram is structurally zero for any problem.  `--require-rank`
defaults to K-1 and the run ASSERTS on it: a benchmark that has quietly become
rank-deficient is not a valid control, and should fail loudly rather than
produce a number.

The forward model (2D TAMP pick-and-place)
------------------------------------------
Deliberately the same three-stage shape as `iosp.model.pickplace`, with a point mass
in the plane instead of an arm:

  1. The task skeleton supplies the pick and place waypoints.  These are FIXED
     and not estimated -- the 2D analogue of holding theta_ik fixed, and for
     the same reason: seeding is not a preference.
  2. One trajopt per segment: approach (start -> pick), transport (pick ->
     place), retreat (place -> home), chained through those waypoints as
     boundary conditions.
  3. One trajopt over the entire concatenated path, warm-started at stage 2.

There is no degenerate zero-length grasp/place segment here (the Panda model
has two, whose Scene has start == goal); in 2D they would carry no signal at
all, and including them would only add features that are identically zero.

The tied cost model
-------------------
ONE theta on the simplex over `{effort, smooth, clearance, skeleton}`, shared
by every segment and by the refine pass.  Segments carry the first three;
`skeleton` is refine-only, and needs no masking at segment level because a
segment's pick/place rows ARE its clamped boundary conditions, so its skeleton
residual is identically zero for any feasible x.

This tie is what makes the composition differentiable.  With per-stage
weights, a segment block reaches the loss only through the warm start, and a
converged argmin has exactly zero seed-sensitivity (x0 never enters
grad_x C = 0), so those weights would be dead parameters.  See
`iosp.model.pickplace`'s tied-model note.

Keeping the skeleton in genuine tension
---------------------------------------
The `skeleton` feature is only identifiable if the refine pass actually WANTS
to leave the waypoints.  If honouring the skeleton were free, its weight would
do nothing and r would drop by one.  `sample_contexts` therefore places pick
and place on OPPOSITE lateral sides of the start-home line, so the
skeleton-respecting path is a forced zig-zag that a smoothness-seeking refine
pass will try to cut -- putting `skeleton` directly in conflict with `smooth`
and `effort`, which is what makes the exchange rate between them measurable.

Runs on CPU: a 2D point mass needs no GPU.
"""

import argparse
import dataclasses
import os
import time

import jax

from iosp.config import enable_compilation_cache
enable_compilation_cache()

import jax.numpy as jnp
import numpy as np

from ioc import identifiability as ident, outer as outer_opt
from ioc.bench2d.problems import _soft_min, _softplus_hinge
from ioc.inner import make_inner_solver

PHASES = ("approach", "transport", "retreat")
SEGMENT_LEN = {"approach": 8, "transport": 10, "retreat": 6}

N_FULL = SEGMENT_LEN["approach"] + (SEGMENT_LEN["transport"] - 1) + (SEGMENT_LEN["retreat"] - 1)
PHASE_SPAN, _s = {}, 0
for _p in PHASES:
    PHASE_SPAN[_p] = (_s, _s + SEGMENT_LEN[_p])
    _s += SEGMENT_LEN[_p] - 1
del _s, _p
assert PHASE_SPAN["retreat"][1] == N_FULL

IDX_PICK = PHASE_SPAN["approach"][1] - 1      # row 7
IDX_PLACE = PHASE_SPAN["transport"][1] - 1    # row 16

SHARED_FEATURES = ("effort", "smooth", "clearance")
FEATURE_NAMES = SHARED_FEATURES + ("skeleton",)
K = len(FEATURE_NAMES)

CLEARANCE = 0.20
Z_STAR = jnp.array([0.5, 1.5, 2.0, 1.0], dtype=jnp.float32)


# --- contexts ---------------------------------------------------------------

@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Ctx:
    """One 2D TAMP context.  `q_start`/`q_goal` are named to match
    `ioc.robot.problem.Scene` so the same unpack/clamp convention applies."""

    q_start: jnp.ndarray   # (2,) start
    q_goal: jnp.ndarray    # (2,) home (end of retreat)
    pick: jnp.ndarray      # (2,)
    place: jnp.ndarray     # (2,)
    obstacles: jnp.ndarray  # (n_obs, 3) x, y, radius


def sample_contexts(rng, n):
    """Contexts where every feature is excited.

    pick and place sit on OPPOSITE sides of the start->home line (see module
    docstring): that puts `skeleton` in tension with `smooth`/`effort`.  The
    obstacle is placed on the transport leg's chord so `clearance` is active --
    an obstacle the demonstration never approaches leaves its weight
    unidentifiable no matter how many demonstrations are collected, the same
    argument as `ioc.bench2d.problems._sample_obstacles`.
    """
    st, ho, pk, pl, ob = [], [], [], [], []
    for _ in range(n):
        s = np.array([-2.2, rng.uniform(-0.4, 0.4)])
        h = np.array([2.2, rng.uniform(-0.4, 0.4)])
        p1 = np.array([-0.7, rng.uniform(0.8, 1.4)])       # above the line
        p2 = np.array([0.7, rng.uniform(-1.4, -0.8)])      # below it
        # ONE obstacle per leg.  A single obstacle on the transport chord makes
        # `clearance` identically inactive on approach and retreat -- and under
        # the TIED model that is fatal rather than merely wasteful: every stage
        # shares one clearance weight, so a stage that never engages the
        # feature contributes a structurally zero column to that stage's
        # whitening (measured: scale 3.4e-18, which the calibration assert
        # catches).  Blocking every leg keeps the shared weight excited
        # wherever it is applied.
        legs = [(s, p1), (p1, p2), (p2, h)]
        rows = []
        for a, b in legs:
            mid = 0.5 * (a + b)
            off = rng.uniform(-0.20, 0.20, size=2)
            rows.append([mid[0] + off[0], mid[1] + off[1], rng.uniform(0.22, 0.34)])
        ob.append(np.array(rows))
        st.append(s); ho.append(h); pk.append(p1); pl.append(p2)
    f = lambda a: jnp.asarray(np.stack(a), dtype=jnp.float32)
    return Ctx(f(st), f(ho), f(pk), f(pl), f(ob))


# --- residuals --------------------------------------------------------------

def unpack(x_flat, ctx, T):
    return jnp.concatenate(
        [ctx.q_start[None, :], x_flat.reshape(T - 2, 2), ctx.q_goal[None, :]], axis=0)


def seed(ctx, T):
    al = jnp.linspace(0.0, 1.0, T)[1:-1, None]
    return ((1 - al) * ctx.q_start + al * ctx.q_goal).reshape(-1)


def _shared(p, obstacles):
    """The shared vocabulary on any stage's path -- one implementation, so
    "the same preference at both levels" is literally the same code."""
    v = p[1:] - p[:-1]
    a = p[2:] - 2 * p[1:-1] + p[:-2]
    d = jnp.linalg.norm(p[:, None, :] - obstacles[None, :, :2], axis=-1) - obstacles[None, :, 2]
    clear = _softplus_hinge(CLEARANCE - _soft_min(d))
    return [v.reshape(-1), a.reshape(-1), clear]


def segment_residual_fn(phase):
    T = SEGMENT_LEN[phase]

    def residual_fn(x_flat, ctx):
        return tuple(_shared(unpack(x_flat, ctx, T), ctx.obstacles))

    return residual_fn


def full_residual_fn():
    def residual_fn(x_flat, ctx):
        p = unpack(x_flat, ctx, N_FULL)
        parts = _shared(p, ctx.obstacles)
        parts.append(jnp.concatenate([p[IDX_PICK] - ctx.pick, p[IDX_PLACE] - ctx.place]))
        return tuple(parts)

    return residual_fn


def phase_ctxs(ctx):
    """Per-segment contexts: the skeleton waypoints become the segments'
    boundary conditions, which is what chains the stages together."""
    return {
        "approach": dataclasses.replace(ctx, q_goal=ctx.pick),
        "transport": dataclasses.replace(ctx, q_start=ctx.pick, q_goal=ctx.place),
        "retreat": dataclasses.replace(ctx, q_start=ctx.place),
    }


def concat_segments(xs, pctx):
    rows = []
    for i, ph in enumerate(PHASES):
        q = jax.vmap(lambda x, c: unpack(x, c, SEGMENT_LEN[ph]))(xs[ph], pctx[ph])
        rows.append(q[:, 1:] if i > 0 else q)
    return jnp.concatenate(rows, axis=1)


def calibrate(residual_fn, ctxs, T, key, n_probe=12, jitter=0.25):
    """Whitening scales, probed on PERTURBED seeds: the straight-line seed has
    exactly zero acceleration, so calibrating `smooth` on it collapses to the
    numerical floor (see `ioc.robot.problem.RobotProblem.calibrate`)."""
    keys = jax.random.split(key, n_probe)

    def raw(c, k):
        x0 = seed(c, T)
        rs = residual_fn(x0 + jitter * jax.random.normal(k, x0.shape), c)
        return jnp.stack([jnp.sum(r ** 2) for r in rs])

    vals = jax.vmap(jax.vmap(raw, in_axes=(None, 0)), in_axes=(0, None))(ctxs, keys)
    sc = jnp.mean(jnp.abs(vals.reshape(vals.shape[0] * vals.shape[1], -1)), axis=0)
    assert bool(jnp.all(sc > 1e-8)), f"degenerate feature scale: {sc}"
    return sc


# --- model ------------------------------------------------------------------

def build(seed_=0, n_iters=120, n_ctx=8):
    from pyroffi.optimization_engines import DynamicsTrajOptConfig, dynamics_trajopt

    # Fixed-length scan (early_stop=False): a grad_tol-gated while_loop has a
    # data-dependent trip count, which makes x*(theta) discontinuous and breaks
    # the implicit adjoint's precondition.  Same reasoning as
    # `iosp.model.pickplace.make_composed_forward_solver`.
    cfg = DynamicsTrajOptConfig(n_iters=n_iters, early_stop=False, unroll_tail=0,
                                soft_line_search=False, soft_curvature_gate=False)
    fs = lambda x0, cost_fn: dynamics_trajopt(x0, cost_fn, cfg)

    rng = np.random.default_rng(seed_)
    allc = sample_contexts(rng, 2 * n_ctx)
    fit = jax.tree.map(lambda a: a[:n_ctx], allc)
    test = jax.tree.map(lambda a: a[n_ctx:], allc)

    key = jax.random.PRNGKey(seed_)
    pc = phase_ctxs(fit)
    inner = {}
    for ph in PHASES:
        rf = segment_residual_fn(ph)
        inner[ph] = make_inner_solver(rf, calibrate(rf, pc[ph], SEGMENT_LEN[ph], key),
                                      forward_solver=fs)
    frf = full_residual_fn()
    refine = make_inner_solver(frf, calibrate(frf, fit, N_FULL, key), forward_solver=fs)
    return dict(fit=fit, test=test, inner=inner, refine=refine)


def paths(built, ctxs, z, *, stage2=True):
    """(B, N_FULL, 2) refined trajectories under theta = softmax(z)."""
    theta = jax.nn.softmax(z)
    theta_seg = theta[:len(SHARED_FEATURES)]
    pc = phase_ctxs(ctxs)
    if stage2:
        xs = {ph: jax.vmap(built["inner"][ph].solve_implicit, in_axes=(0, None, 0))(
            jax.vmap(lambda c: seed(c, SEGMENT_LEN[ph]))(pc[ph]), theta_seg, pc[ph])
            for ph in PHASES}
        q = concat_segments(xs, pc)
        x0 = q[:, 1:-1, :].reshape(q.shape[0], -1)
    else:
        x0 = jax.vmap(lambda c: seed(c, N_FULL))(ctxs)
    xf = jax.vmap(built["refine"].solve_implicit, in_axes=(0, None, 0))(x0, theta, ctxs)
    return jax.vmap(lambda x, c: unpack(x, c, N_FULL))(xf, ctxs)


def make_loss(built, ctxs, demos, *, stage2=True):
    def loss(z):
        return jnp.mean(jnp.sum((paths(built, ctxs, z, stage2=stage2) - demos) ** 2, -1))
    return loss


def fit_z(gf, starts, *, lr=0.05, n_steps=60):
    best = None
    for z0 in starts:
        z, tr = outer_opt.adam(gf, z0, lr=lr, n_steps=n_steps)
        lN = min(v for _, v in tr)
        if best is None or lN < best["lN"]:
            best = dict(z=z, z0=z0, l0=tr[0][1], lN=lN)
    a0, ah = np.asarray(best["z0"], float), np.asarray(best["z"], float)
    mv = float(np.linalg.norm(ah - a0) / (np.linalg.norm(a0) + 1e-30))
    red = best["l0"] / max(best["lN"], 1e-30)
    best.update(move_rel=mv, loss_reduction=red,
                degenerate=(not np.isfinite(mv)) or mv < 1e-3 or red < 1.05)
    return best


def record_fit(built, fit_ctx, demos_fit, test_ctx, demos_test, z0, *,
               n_steps=60, lr=0.05, out="iosp/data/viz/tamp2d_fit.npz", n_show=3):
    """Re-run the fit from `z0`, saving the PREDICTED PATHS at every outer step.

    `ioc.outer.adam` returns only its best iterate, not the trajectory, so the
    loop is reproduced here (same optax.adamw, same ordering) purely to
    capture history.  What gets saved is behaviour, not parameters: the paths
    on held-out contexts plus held-out RMSE per step, which is what
    `iosp.viz.behavior` animates.  Recording from z0 = uniform rather than
    from a multistart winner is deliberate -- the animation is meant to show
    convergence from an uninformed prior, and starting it at an already-good
    iterate would show almost no motion.
    """
    import optax

    path_j = jax.jit(lambda z: paths(built, test_ctx, z))
    lf = jax.jit(jax.value_and_grad(make_loss(built, fit_ctx, demos_fit)))
    rmse_j = jax.jit(lambda z: jnp.sqrt(jnp.mean(
        jnp.sum((paths(built, test_ctx, z) - demos_test) ** 2, -1))))

    opt = optax.adamw(lr, weight_decay=0.0)
    z, st = z0, opt.init(z0)
    th_h, loss_h, rmse_h, path_h = [], [], [], []
    for t in range(n_steps + 1):
        val, g = lf(z)
        th_h.append(np.asarray(jax.nn.softmax(z)))
        loss_h.append(float(val))
        rmse_h.append(float(rmse_j(z)))
        path_h.append(np.asarray(path_j(z))[:n_show])
        if t % 10 == 0 or t == n_steps:
            print(f"[record] step {t:3d}/{n_steps}  train={val:.6f}  "
                  f"heldout_rmse={rmse_h[-1]:.6f}", flush=True)
        if t == n_steps:
            break
        upd, st = opt.update(g, st, z)
        z = optax.apply_updates(z, upd)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(
        out,
        path_hist=np.stack(path_h), demo=np.asarray(demos_test)[:n_show],
        rmse_hist=np.asarray(rmse_h), loss_hist=np.asarray(loss_h),
        theta_hist=np.stack(th_h),
        obstacles=np.asarray(test_ctx.obstacles)[:n_show],
        waypoints=np.stack([np.asarray(test_ctx.pick)[:n_show],
                            np.asarray(test_ctx.place)[:n_show]], axis=1),
        names=np.array(FEATURE_NAMES, dtype=object),
        label="2D TAMP pick-and-place, tied 3-stage SPaSM")
    print(f"[record] wrote {out}", flush=True)
    return out


def run(seed_=0, n_iters=120, n_ctx=8, n_steps=60, n_starts=8, require_rank=None,
        record=None):
    require_rank = K - 1 if require_rank is None else require_rank
    t0 = time.perf_counter()
    built = build(seed_=seed_, n_iters=n_iters, n_ctx=n_ctx)
    fit, test = built["fit"], built["test"]
    print(f"[build] {time.perf_counter()-t0:.1f}s  K={K} {FEATURE_NAMES}  "
          f"N_FULL={N_FULL}  pick@{IDX_PICK} place@{IDX_PLACE}  "
          f"fit/test={n_ctx}/{n_ctx}", flush=True)

    demos_fit = jax.jit(lambda: paths(built, fit, Z_STAR))()
    demos_test = jax.jit(lambda: paths(built, test, Z_STAR))()
    jax.block_until_ready((demos_fit, demos_test))

    lf = make_loss(built, fit, demos_fit)
    jt = jax.jit(make_loss(built, test, demos_test))
    sanity = float(jax.jit(lf)(Z_STAR))
    print(f"[sanity] loss(z_star) = {sanity:.3e}", flush=True)
    assert sanity < 1e-8, f"model does not reproduce its own demo: {sanity:.3e}"

    g0 = np.asarray(jax.grad(lf)(jnp.zeros(K, jnp.float32)))
    print(f"[grad] dL/dtheta at z=0: {g0}", flush=True)
    assert np.all(np.abs(g0) > 1e-12), f"dead parameter(s): {g0}"

    # -- the point of this benchmark: is theta actually identifiable here? --
    jac = ident.make_jac_fn(lambda z: paths(built, fit, z).reshape(-1))
    eig, _ = ident.sensitivity_spectrum(jac, Z_STAR)
    _, _, r = ident.select_rank(eig, rule="gap")
    print(f"[rank] eigenvalues {np.asarray(eig)}")
    print(f"[rank] r = {r} of K = {K}  (ceiling is K-1 = {K-1}: the softmax "
          f"gauge is an exact null direction)", flush=True)
    assert r >= require_rank, (
        f"benchmark is rank-deficient (r={r} < {require_rank}); it cannot serve "
        f"as an identifiability-controlled test -- fix the contexts, do not "
        f"lower the bar")

    rng = np.random.default_rng(seed_ + 1)
    z0 = jnp.zeros(K, jnp.float32)
    starts = [z0] + [jnp.asarray(rng.normal(0, 1, K), jnp.float32) for _ in range(n_starts - 1)]

    gf = jax.jit(jax.value_and_grad(lf))
    b = fit_z(gf, starts, n_steps=n_steps)
    b["test_rmse"] = float(jnp.sqrt(jt(b["z"])))

    gf2 = jax.jit(jax.value_and_grad(make_loss(built, fit, demos_fit, stage2=False)))
    jt2 = jax.jit(make_loss(built, test, demos_test, stage2=False))
    b2 = fit_z(gf2, starts, n_steps=n_steps)
    b2["test_rmse"] = float(jnp.sqrt(jt2(b2["z"])))

    if record:
        npz = record_fit(built, fit, demos_fit, test, demos_test, z0,
                         n_steps=n_steps, out=record)
        try:
            from iosp.viz import behavior as viz_behavior
            viz_behavior.render(npz, npz.replace(".npz", ".gif"))
        except Exception as e:  # rendering is optional; the .npz is the artifact
            print(f"[record] render skipped: {type(e).__name__}: {e}", flush=True)

    th, ths = np.asarray(jax.nn.softmax(b["z"])), np.asarray(jax.nn.softmax(Z_STAR))
    return dict(fit=b, no_stage2=b2, rank=int(r), eig=np.asarray(eig),
                baseline=float(jnp.sqrt(jt(z0))), oracle=float(jnp.sqrt(jt(Z_STAR))),
                theta_hat=th, theta_star=ths, param_err=float(np.linalg.norm(th - ths)))


def report(o):
    ref = o["baseline"]
    print("\n=== held-out behavioural recovery (RMSE) ===")
    for n, v in (("baseline uniform (theta = 1/K)", ref),
                 ("tied three-stage fit", o["fit"]["test_rmse"]),
                 ("  ablation: refine only, no stage 2", o["no_stage2"]["test_rmse"]),
                 ("oracle theta*", o["oracle"])):
        print(f"  {n:36s} {v:9.5f}   {100*(1-v/ref):+6.1f}% vs uniform")
    if o["fit"]["degenerate"]:
        print(f"\nNO VERDICT: fit degenerate (move_rel={o['fit']['move_rel']:.2e}, "
              f"reduction={o['fit']['loss_reduction']:.2f}x)")
        return
    a, b = o["fit"]["test_rmse"], o["no_stage2"]["test_rmse"]
    print(f"\nstage 2 seeding changes held-out RMSE by {100*(1-a/b):+.1f}% "
          f"({'load-bearing' if abs(a-b)/max(b,1e-30) > 0.05 else 'washes out'}).")
    print(f"\nrank r = {o['rank']} of K = {K}; param_err = {o['param_err']:.4f} "
          f"(reported, not the criterion)")
    for n, h, s in zip(FEATURE_NAMES, o["theta_hat"], o["theta_star"]):
        print(f"  {n:12s} {h:8.4f}  {s:8.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-iters", type=int, default=120)
    ap.add_argument("--n-ctx", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=60)
    ap.add_argument("--n-starts", type=int, default=8)
    ap.add_argument("--require-rank", type=int, default=None)
    ap.add_argument("--record", default=None,
                    help="path for the behavioural-convergence .npz/.gif")
    a = ap.parse_args()
    report(run(seed_=a.seed, n_iters=a.n_iters, n_ctx=a.n_ctx, n_steps=a.n_steps,
               n_starts=a.n_starts, require_rank=a.require_rank,
               record=a.record))
