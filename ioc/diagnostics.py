"""Single-segment IOC diagnostic suite: six failure-mode axes on the 7-DOF
Panda kinematic reach-with-obstacle problem (`ioc.robot.problem`).

Why one suite instead of decoupled studies
-------------------------------------------
`iosp/study0*` through `study3*` each isolated one hypothesis about why the
COMPOSED pick-and-place model recovers costs poorly, discovered in sequence
and left as separate scripts.  That made sense while the questions were being
found, but it means the same six axes -- demo quality, demo diversity, basis
size, identifiability, optimizer correctness, generalization -- are each
answered on a different scene family, with different confounds controlled in
each script and none controlled in all of them.

This module runs all six axes on ONE problem (the single-segment robot, not
the pick-and-place composition) with a shared scene distribution, a shared
fitting procedure, and a shared metric set, so the six results are directly
comparable to each other rather than each needing its own caveat section.
`iosp` is a composition of this same machinery (`ioc.inner`, `ioc.outer`,
`ioc.analytic`, `ioc.identifiability`) over four chained segments; the plan is
to establish that the method works cleanly here, on one segment, before
trusting any diagnosis of the composed model.

Common vocabulary
-----------------
Every test reports a subset of:

    e_demo    end-effector RMSE against the demo the fit was trained on
              (`ioc.robot.problem.evaluate`'s `ee_rmse`, "fit" scenes).
    e_test    same RMSE on a held-out scene set from the SAME distribution
              the fit never saw during optimization.
    e_global  behavioural regret (true-cost excess of the recovered policy)
              averaged over a LARGE independent scene pool -- the closest
              thing this problem has to "the whole cost space", as opposed to
              `e_test`'s fixed, small held-out set.
    param_err / param_cos   L1 distance and cosine similarity between
              theta_hat and theta_star on the simplex (`ioc.metrics`).
    Gram / eigvals   the feature-gradient Gram `G = mean_i B(c_i)^T B(c_i)`
              from `ioc.analytic.kkt_fit(..., return_gram=True)` -- zero
              forward solves, so it is available even where the fit itself is
              not being trusted yet.

The central methodological point (per the user's framing) is that `e_demo` is
NOT `e_global`: a fit can reconstruct its training demonstrations exactly
while being wrong everywhere else on the cost's domain, or refuse to
reconstruct the training demo while being closer to `theta_star` than a fit
that overfit it.  Every test below reports BOTH, never collapses them into
one number, and `test_generalization` makes the gap itself the measurement.

Usage
-----
    XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=<idx> \\
        python -m ioc.diagnostics                    # run everything, print + json
        python -m ioc.diagnostics --which demo_quality,generalization
"""

import dataclasses
import json
import time
import zlib

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro

from ioc import analytic, metrics, outer as outer_opt
from ioc.inner import make_inner_solver
from ioc.robot import bases, problem as prob


@dataclasses.dataclass(frozen=True)
class SuiteConfig:
    urdf_path: str = "resources/panda/panda_spherized.urdf"
    srdf_path: str = "resources/panda/panda.srdf"
    mesh_dir: str = "resources/panda/meshes"
    n_timesteps: int = 24
    n_newton: int = 600
    n_outer_steps: int = 60
    lr: float = 0.15
    conv_tol: float = 1e-4
    adjoint_ridge: float = 1e-9
    seed: int = 0
    n_starts: int = 4
    """Outer multistart: z0 draws per fit, best-of kept (`fit`).

    1 reproduces the old single-start behaviour.  Raised from 1 because a
    single bad draw sank every row of `generalization` seed 3; note this is
    outer-loop multistart, unrelated to `make_inner_solver`'s `n_restarts`.
    """

    max_batch_rollouts: int = 20
    """Cap on concurrent (start x scene) rollouts inside `fit` (memory).

    Measured on a 24 GB A5000: 4 starts x 5 scenes fits, 8 x 5 dies asking for
    14.7 GiB.  `fit` chunks the starts to this budget, so `n_starts` stays a
    statistical choice rather than one bounded by the largest `n_contexts` in a
    sweep.  Raise on a bigger card; lower if a wider basis pushes per-rollout
    memory up."""

    screen_steps: int = 0
    """MEMORY valve only (`fit`).  0 (default) keeps every start batched for
    the full run -- with the starts vmapped this is both faster and safer.  Set
    > 0 only if `n_starts * n_contexts` rollouts will not fit in memory: the
    starts are then screened for that many steps and the best continues alone,
    trading the batched speed back for a smaller live footprint."""

    unroll_tail: int = 1
    """Trailing solver steps that carry gradients (`DynamicsTrajOptConfig`).

    Only meaningful with `early_stop=False`, and the engine treats 0 as "unroll
    and differentiate ALL `n_iters` steps" -- which at `n_newton=600` builds a
    600-step L-BFGS graph and sent XLA into a >1 h single-threaded compile at
    0% GPU (26.5 GB RSS) before this was set.  That unrolled gradient is pure
    waste here: `ioc.inner` calls the forward solver under `stop_gradient`
    inside a `custom_vjp` (inner.py:139,245), so the outer gradient comes from
    the analytic adjoint and the unrolled one is discarded.

    1 keeps the fixed-length `lax.scan` (so the trip count stays a constant and
    x*(theta) does not jump, which is what `early_stop=False` was chosen for)
    while running the head under `stop_gradient`.  Forward values are unchanged
    -- same 600 iterations -- so this is a compile-time fix only, not a change
    to any measured quantity.
    """

    early_stop: bool = False
    """Fixed iteration count rather than convergence-triggered termination.

    With `early_stop=True` the stop index is an integer function of theta, so
    x*(theta) jumps whenever it flips.  Measured on k3/10 scenes, roughness of
    the outer loss (mean |divided difference| over a +/-1e-3 probe) against the
    adjoint's own |g|:

        all hard, early_stop=True     rough/|g| = 1458.8   FD cos = 0.873
        early_stop=False              rough/|g| =   72.2   FD cos = 0.906
        soft line search + gate       rough/|g| =   41.0   FD cos = 0.405
        open loop (fixed N, all soft) rough/|g| =   17.7   FD cos = 0.324

    The soft variants cut roughness further but DEGRADE agreement with finite
    differences, because softening the line search/curvature gate moves the
    fixed point the adjoint linearizes about.  Fixed-iteration with hard steps
    is the configuration that improves both at once, which is why it is the
    default here.  `n_newton` is correspondingly large: at 40 the adjoint's
    magnitude is ~47% off its converged value.
    """


# ---------------------------------------------------------------------------
# shared machinery: solver construction, scene sampling, fit-and-score
# ---------------------------------------------------------------------------

def make_forward_solver(n_iters, *, early_stop=False, **kw):
    """`Callable[[x0, cost_fn], x]` wrapping pyroffi's L-BFGS engine, the
    `forward_solver` `ioc.inner.make_inner_solver` requires.

    `early_stop` defaults to False here, unlike the engine's own default -- see
    `SuiteConfig.early_stop` for the measurements behind that choice.
    """
    from pyroffi.optimization_engines import DynamicsTrajOptConfig, dynamics_trajopt

    cfg = DynamicsTrajOptConfig(n_iters=n_iters, early_stop=early_stop, **kw)

    def fs(x0, cost_fn):
        return dynamics_trajopt(x0, cost_fn, cfg)

    return fs


def build_inner(problem, residual_fn, scenes, cfg, *, n_restarts=1):
    """Calibrate feature scales on `scenes` and build the `InnerSolver`."""
    scales = problem.calibrate(residual_fn, scenes, jax.random.key(cfg.seed))
    forward_solver = make_forward_solver(cfg.n_newton, early_stop=cfg.early_stop,
                                         unroll_tail=cfg.unroll_tail,
                                         grad_tol=min(cfg.conv_tol, 1e-6))
    inner = make_inner_solver(
        residual_fn, scales, forward_solver=forward_solver,
        adjoint_ridge=cfg.adjoint_ridge, n_restarts=n_restarts,
    )
    return inner, scales


def sample_diverse_scenes(problem, rng, n, diversity, q_start=None, q_goal=None):
    """`n` contexts interpolated between one repeated draw (`diversity=0`,
    every context informationally identical to the first) and `n` fully
    independent draws (`diversity=1`, `problem.sample_scenes`).

    This is `test_demo_diversity`'s knob on EXCITATION VARIETY, orthogonal to
    `test_demo_count`'s N: N can grow while diversity stays at 0 (more copies
    of the same context, no new information), which is exactly the
    distinction axis 2 of the suite is meant to expose.
    """
    diversity = float(np.clip(diversity, 0.0, 1.0))
    base = problem.sample_scenes(rng, 1, q_start=q_start, q_goal=q_goal)
    indep = problem.sample_scenes(rng, n, q_start=q_start, q_goal=q_goal)

    def mix(b, r):
        b = jnp.broadcast_to(b, r.shape)
        return b + diversity * (r - b)

    return jax.tree.map(mix, base, indep)


def scale_obstacle(scenes, excitation):
    """Scale `obs_radius` by `excitation` about its sampled value, holding
    everything else fixed.  `excitation -> 0` shrinks the obstacle toward
    nothing, so `clearance_residual` goes to zero everywhere and the
    collision weight becomes structurally unidentifiable; `excitation > 1`
    makes the obstacle block harder than sampled.  Used by
    `test_identifiability_conditioning` to control excitation independent of
    which contexts were sampled.
    """
    return dataclasses.replace(scenes, obs_radius=scenes.obs_radius * excitation)


def demo_pipeline(problem, inner, scenes, theta_star, rng, demo_noise=0.0,
                   demo_n_newton=None, cfg=None):
    """Build `(x0s, x_star, demos)` for `scenes`.

    `demo_n_newton`, when given, generates the demonstration with a SEPARATE,
    less-converged forward solver than the one the fit uses -- a demo that is
    optimal under a shorter budget rather than the true stationary point, i.e.
    demonstrator SUBOPTIMALITY rather than i.i.d. observation noise.  The two
    are structurally different failure sources (`test_demo_quality` reports
    both): noise is symmetric, mean-zero corruption of an otherwise-optimal
    trajectory; a truncated solve is a systematic bias toward the seed.
    """
    if demo_n_newton is not None:
        assert cfg is not None
        solver = make_forward_solver(demo_n_newton)
        x0s = problem.seeds(scenes)
        x_star = jax.vmap(lambda x0, s: solver(x0, lambda x: inner.cost(x, theta_star, s)))(x0s, scenes)
        demos = jax.vmap(problem.unpack)(x_star, scenes)
        if demo_noise > 0:
            noise = jnp.asarray(rng.normal(scale=demo_noise, size=demos.shape))
            demos = demos + noise.at[:, 0].set(0.0).at[:, -1].set(0.0)
        return x0s, x_star, demos
    return prob.make_demos(problem, inner.solve_implicit, scenes, theta_star, rng, demo_noise)


def row_seed(base_seed, *parts):
    """Deterministic z0 seed for ONE sweep point, mixing `base_seed` with the
    sweep coordinates.

    `fit` derives z0 from its `seed` alone, and every sweep here used to pass
    `seed=cfg.seed` from inside its loop -- so a whole sweep shared ONE
    initialization.  That made a sweep's rows non-independent and confounded
    the axis's own measurement: "does more data help" could not be separated
    from "was this one z0 draw lucky", which is how a single bad draw
    (generalization seed 3) became five bad rows that read as five failures.
    Keying on the sweep point makes each row an independent draw while staying
    reproducible from `cfg.seed`.

    `zlib.crc32` rather than `hash()`: Python salts string hashes per process,
    so `hash()` would not reproduce across invocations.
    """
    key = "|".join(str(x) for x in parts).encode()
    return int(base_seed) * 1_000_003 + (zlib.crc32(key) & 0xFFFFFFFF)


def fit(problem, inner, K, scenes, demos, x0s, *, seed, n_steps, lr, z0=None,
        n_starts=1, screen_steps=0, max_batch_rollouts=20):
    """Stage-1 wide fit: Adam(W) on all K weights, no subspace restriction.

    `n_starts > 1` runs OUTER multistart over z0 -- a different mechanism from
    `make_inner_solver`'s `n_restarts`, which reseeds the FORWARD solve to
    cover basins of x* and does nothing for a bad z0, because the failure is in
    the outer landscape rather than the inner one.  The motivating case is
    `generalization` seed 3, whose z0 put `collision` at 0.065 against a true
    0.3 and converged toward a simplex vertex (`param_err` ~1.5 of a max 2) at
    every `n_contexts`.

    The starts are run BATCHED, not sequentially: z carries a leading start
    axis of shape `(n_starts, K)`, the outer loss is `vmap`ped over it, and the
    whole `(n_starts, n_scenes)` block of rollouts is solved inside one JAX
    call.  That is the point of doing this in pyroffi at all -- the implicit
    adjoint makes each rollout's gradient independent of how the rollout was
    found, so covering starts costs batch width on the GPU rather than
    wall-clock.  Adam over the stacked array is exactly `n_starts` independent
    Adam runs, since row i's gradient depends only on row i.

    `screen_steps` is retained only as a MEMORY valve: with it > 0 the starts
    are screened for that many steps and the single best continues alone, which
    trades the batched speed back for a smaller live footprint if
    `n_starts * n_scenes` rollouts will not fit.  0 (the default) keeps every
    start batched for the full run and is both faster and safer.

    Starts are drawn from one `default_rng(seed)`, so start 0 is exactly the
    old single-start z0 and results stay reproducible.
    """
    loss = prob.make_outer(problem, inner.solve_implicit, scenes, demos, x0s)

    if z0 is not None:
        return outer_opt.adam(jax.jit(jax.value_and_grad(loss)), z0, lr=lr, n_steps=n_steps)

    rng = np.random.default_rng(seed)
    n_starts = max(1, int(n_starts))
    Z0 = jnp.asarray(rng.normal(scale=0.5, size=(n_starts, K)))
    if n_starts == 1:
        return outer_opt.adam(jax.jit(jax.value_and_grad(loss)), Z0[0], lr=lr, n_steps=n_steps)

    # starts are batched, but (n_starts x n_scenes) rollouts must still FIT:
    # measured on an A5000, 4 starts x 5 scenes runs while 8 x 5 dies asking
    # for 14.7 GiB, so a flat n_starts would crash generalization's n=20 row.
    # Chunk the starts to a rollout budget instead: full batching where it
    # fits, sequential only across chunks.
    batched = jax.jit(jax.vmap(jax.value_and_grad(loss)))
    n_scenes = int(np.shape(x0s)[0])
    per_chunk = max(1, int(max_batch_rollouts) // max(1, n_scenes))

    def run(Z, steps):
        """Batched Adam over a leading start axis.

        The running best is kept on device and pulled once at the end.  That
        was originally done to remove a suspected per-step host-sync stall, but
        the stall does not exist: `outer.adam` measures 1.00x with and without
        the per-step `float(...)`, because the loop is dominated by the solve
        the sync was waiting on anyway.  Keep it as tidiness, not as a
        speedup -- the reason 4 starts cost 2.3x one start is simply that the
        GPU is near saturation at ~10 concurrent rollouts, so extra batch width
        queues rather than overlapping (measured ceiling ~1.5x for widening).
        """
        opt = optax.adamw(lr)
        st = opt.init(Z)
        best_v = jnp.full((Z.shape[0],), jnp.inf)
        best_Z = Z
        hist = []
        for _ in range(steps):
            v, g = batched(Z)
            improved = v < best_v
            best_v = jnp.where(improved, v, best_v)
            best_Z = jnp.where(improved[:, None], Z, best_Z)
            hist.append(jnp.min(best_v))
            u, st = opt.update(g, st, Z)
            Z = optax.apply_updates(Z, u)
        return best_Z, best_v, np.asarray(jnp.stack(hist))   # one sync, at the end

    def run_chunked(Z, steps):
        outs = [run(Z[i:i + per_chunk], steps) for i in range(0, Z.shape[0], per_chunk)]
        bZ = jnp.concatenate([o[0] for o in outs], axis=0)
        bv = jnp.concatenate([o[1] for o in outs], axis=0)
        curve = np.minimum.reduce([o[2] for o in outs])
        trace = [((t + 1) * Z.shape[0] * n_scenes, float(c)) for t, c in enumerate(curve)]
        return bZ, bv, trace

    if screen_steps > 0:
        _, v_screen, _ = run_chunked(Z0, screen_steps)
        Z0 = Z0[int(jnp.argmin(v_screen))][None, :]

    best_Z, best_v, trace = run_chunked(Z0, n_steps)
    return best_Z[int(jnp.argmin(best_v))], trace


def score(problem, inner, z, scenes, demos, x0s, theta_star):
    return prob.evaluate(problem, z, jax.jit(inner.solve_implicit), inner.cost,
                         scenes, demos, x0s, theta_star)


def gram_certificate(inner, scenes, demos, K):
    """Feature-gradient Gram `G` (`ioc.analytic.kkt_fit`'s construction), its
    trace-normalized eigenvalues, effective rank, and condition number --
    zero forward solves, so it is available before trusting any fit."""
    _, G = analytic.kkt_fit(inner.grad_x, scenes, demos, K, n_steps=1, lr=0.0,
                            return_gram=True)
    G = np.asarray(G)
    eig = np.linalg.eigvalsh(G / (np.trace(G) / K + 1e-30))
    eig = np.clip(eig, 0.0, None)[::-1]  # descending, numerical negatives clipped
    eff_rank = float(np.sum(eig) ** 2 / (np.sum(eig ** 2) + 1e-30))  # participation ratio
    cond = float(eig[0] / max(eig[-1], 1e-30))
    return dict(eigvals=eig.tolist(), lambda_min=float(eig[-1]), lambda_max=float(eig[0]),
               effective_rank=eff_rank, cond=cond)


def gram_eigen(inner, scenes, demos, K):
    """`gram_certificate`'s decomposition, keeping the EIGENVECTORS so a weight
    error can be resolved per identifiability direction rather than as one
    scalar.  Returns `(eigvals, eigvecs)`, descending, trace-normalized; column
    `j` of `eigvecs` pairs with `eigvals[j]`."""
    _, G = analytic.kkt_fit(inner.grad_x, scenes, demos, K, n_steps=1, lr=0.0,
                            return_gram=True)
    G = np.asarray(G)
    Gn = G / (np.trace(G) / K + 1e-30)
    w, V = np.linalg.eigh(Gn)
    return np.clip(w, 0.0, None)[::-1], V[:, ::-1]


def eigen_projected_error(theta_hat, theta_star, eigvals, eigvecs):
    """Weight error resolved along the Gram's eigendirections.

    Raw simplex L1 is a poor headline whenever the Gram is ill-conditioned: an
    error along a low-`lambda` direction is one the demonstrations never
    constrained, and counting it equally with a well-excited direction reports a
    failure the data could not have prevented.  This returns both the
    per-direction components and the identifiability-weighted norm
    `sqrt(delta^T G_norm delta)`, which is the quantity a fit is actually
    penalized for getting wrong.
    """
    delta = np.asarray(theta_hat, dtype=np.float64) - np.asarray(theta_star, dtype=np.float64)
    comps = eigvecs.T @ delta
    return dict(
        components=[float(c) for c in comps],
        eigvals=[float(l) for l in eigvals],
        weighted_norm=float(np.sqrt(np.sum(eigvals * comps ** 2))),
    )


def global_regret(problem, inner, z, theta_star, rng, n_scenes=40):
    """`e_global`: mean true-cost regret of `softmax(z)` over a large,
    freshly-sampled scene pool -- independent of whatever scenes the fit was
    trained or held-out-tested on, so it stands in for "the whole cost
    space" rather than one fixed test set."""
    pool = problem.sample_scenes(rng, n_scenes)
    x0s = problem.seeds(pool)
    theta = jax.nn.softmax(z)

    def one(scene, x0):
        x_hat = inner.solve_implicit(x0, theta, scene)
        x_star = inner.solve_implicit(x0, theta_star, scene)
        return inner.cost(x_hat, theta_star, scene) - inner.cost(x_star, theta_star, scene)

    regret = jax.vmap(one)(pool, x0s)
    return float(jnp.mean(regret))


# ---------------------------------------------------------------------------
# axis 1 -- demonstration quality
# ---------------------------------------------------------------------------

def test_demo_quality(problem, theta_star, cfg=SuiteConfig(), *, basis="k3",
                      n_contexts=10, n_holdout=10,
                      noise_levels=(0.0, 0.01, 0.02, 0.05, 0.10),
                      suboptimal_iters=(40, 20, 10, 5)):
    """Axis 1: does degrading the DEMONSTRATION alone degrade recovery?

    Two independent corruptions are swept on the same fixed scene set, so
    scene identity cannot confound the comparison:

    - `noise_levels`: i.i.d. Gaussian joint-angle noise added to an otherwise
      exactly-optimal demonstration.
    - `suboptimal_iters`: the demonstration is generated by a solver STOPPED
      EARLY (fewer Newton iterations than the fit's own forward solver uses),
      i.e. a demonstrator that is systematically short of optimal rather than
      noisily observed.

    How to interpret
    -----------------
    Both sweeps hold the scene distribution and basis fixed, so if
    `param_err`/`e_demo`/`e_global` all grow monotonically as noise or
    suboptimality increases, degraded recovery in a downstream experiment
    that uses this basis and this scene family IS a demonstration-quality
    problem -- collect better/more-converged demonstrations rather than
    reworking the basis or the optimizer.

    If instead the curve is FLAT (recovery is already bad at `noise=0` /
    `suboptimal_iters=n_newton`, i.e. the clean-demo bookend), demonstration
    quality is NOT the bottleneck -- move to `test_demo_diversity` (is there
    enough information at all?) or `test_optimizer_correctness` (is the
    gradient trustworthy?) before touching the demonstration pipeline.

    Compare the noise curve's slope to the suboptimality curve's: if noise is
    far more damaging than a truncated solve of comparable behavioural
    displacement, the fit is more sensitive to unstructured corruption than
    to systematic bias, which argues for filtering/smoothing noisy
    demonstrations over re-solving suboptimal ones (or vice versa).
    """
    residual_fn, names = bases.kinematic(problem, basis)
    K = len(names)
    rng = np.random.default_rng(cfg.seed)

    pool = problem.sample_scenes(rng, (n_contexts + n_holdout) * 4)
    probe_inner, _ = build_inner(problem, residual_fn, pool, cfg)
    scenes_all, discard_rate, _ = prob.screen_scenes(
        problem, pool, probe_inner.stationarity, theta_star, cfg.conv_tol,
        n_contexts + n_holdout,
    )
    scenes_fit = jax.tree.map(lambda a: a[:n_contexts], scenes_all)
    scenes_test = jax.tree.map(lambda a: a[n_contexts:], scenes_all)

    inner, _ = build_inner(problem, residual_fn, scenes_fit, cfg)
    x0s_fit, x0s_test = problem.seeds(scenes_fit), problem.seeds(scenes_test)
    _, _, demos_test = prob.make_demos(problem, inner.solve_implicit, scenes_test, theta_star, rng, 0.0)

    def run_row(demo_noise=0.0, demo_n_newton=None):
        _, _, demos_fit = demo_pipeline(problem, inner, scenes_fit, theta_star, rng,
                                        demo_noise, demo_n_newton, cfg)
        z_hat, _ = fit(problem, inner, K, scenes_fit, demos_fit, x0s_fit,
                      seed=row_seed(cfg.seed, "demo_quality", demo_noise, demo_n_newton),
                      n_steps=cfg.n_outer_steps, lr=cfg.lr,
                      n_starts=cfg.n_starts, screen_steps=cfg.screen_steps,
                      max_batch_rollouts=cfg.max_batch_rollouts)
        m_fit = score(problem, inner, z_hat, scenes_fit, demos_fit, x0s_fit, theta_star)
        m_test = score(problem, inner, z_hat, scenes_test, demos_test, x0s_test, theta_star)
        return dict(param_err=m_fit["theta_l1"], param_cos=m_fit["theta_cos"],
                   e_demo=m_fit["ee_rmse"], e_test=m_test["ee_rmse"], regret=m_fit["regret"])

    noise_rows = [dict(noise=float(n), **run_row(demo_noise=n)) for n in noise_levels]
    subopt_rows = [dict(n_newton=int(it), **run_row(demo_n_newton=it)) for it in suboptimal_iters]

    return dict(axis="demo_quality", basis=basis, K=K, names=list(names),
               discard_rate=discard_rate, noise_sweep=noise_rows,
               suboptimality_sweep=subopt_rows)


# ---------------------------------------------------------------------------
# axis 2 -- number + diversity of demonstrations
# ---------------------------------------------------------------------------

def test_demo_diversity(problem, theta_star, cfg=SuiteConfig(), *, basis="k3",
                        n_values=(1, 3, 5, 10, 20), diversity_values=(0.0, 0.25, 0.5, 1.0),
                        n_holdout=10):
    """Axis 2: is poor recovery an information problem?

    Two sweeps, both on clean (noise-free, fully-converged) demonstrations so
    the ONLY thing varying is how much the demo set constrains theta:

    - `n_values` at fixed `diversity=1.0`: growing the number of independent
      contexts.
    - `diversity_values` at fixed `n=10`: interpolating between `n` copies of
      one context (`diversity=0`, informationally equivalent to N=1) and `n`
      independent contexts (`diversity=1`) via `sample_diverse_scenes`.

    How to interpret
    -----------------
    Read `gram.effective_rank` and `gram.lambda_min` first, `param_err`/
    `e_test` second -- the Gram is a property of the SCENES alone (zero
    forward solves), so it tells you the ceiling before any optimizer result
    can be trusted.

    - `effective_rank` saturating near `K` and `lambda_min` staying bounded
      away from 0 as N grows: the basis is identifiable from this scene
      family in the large-N limit, and if `param_err`/`e_test` are still bad
      at the largest `N` tested here, the fit itself is broken -- go to
      `test_optimizer_correctness`.
    - `effective_rank` plateauing BELOW `K` even as N grows large: some
      combination of features is *structurally* invisible to this scene
      family regardless of how many contexts you add -- this is
      `test_identifiability_conditioning`'s question, not a diversity
      problem, and no amount of additional demos (at fixed diversity) fixes
      it.
    - The diversity sweep isolates count from information directly: if
      `effective_rank` stays flat while `n` grows at `diversity=0` but rises
      with `diversity` at fixed `n`, then recovery quality tracks CONTEXT
      VARIETY, not demonstration COUNT -- matching `iosp/study2`'s finding on
      the composed model, now checked on the single segment first.
    """
    residual_fn, names = bases.kinematic(problem, basis)
    K = len(names)
    rng = np.random.default_rng(cfg.seed)

    def one_set(scenes_fit, label):
        probe_inner, _ = build_inner(problem, residual_fn, scenes_fit, cfg)
        x0s_fit, _, demos_fit = prob.make_demos(problem, probe_inner.solve_implicit,
                                                scenes_fit, theta_star, rng, 0.0)
        gram = gram_certificate(probe_inner, scenes_fit, demos_fit, K)

        holdout = problem.sample_scenes(rng, n_holdout)
        x0s_test = problem.seeds(holdout)
        _, _, demos_test = prob.make_demos(problem, probe_inner.solve_implicit, holdout,
                                           theta_star, rng, 0.0)

        z_hat, _ = fit(problem, probe_inner, K, scenes_fit, demos_fit, x0s_fit,
                      seed=row_seed(cfg.seed, "demo_diversity", label),
                      n_steps=cfg.n_outer_steps, lr=cfg.lr,
                      n_starts=cfg.n_starts, screen_steps=cfg.screen_steps,
                      max_batch_rollouts=cfg.max_batch_rollouts)
        m_fit = score(problem, probe_inner, z_hat, scenes_fit, demos_fit, x0s_fit, theta_star)
        m_test = score(problem, probe_inner, z_hat, holdout, demos_test, x0s_test, theta_star)
        return gram, m_fit, m_test

    n_rows = []
    for n in n_values:
        scenes = problem.sample_scenes(rng, n)
        gram, m_fit, m_test = one_set(scenes, f"n={n}")
        n_rows.append(dict(n=n, gram=gram, param_err=m_fit["theta_l1"],
                           e_demo=m_fit["ee_rmse"], e_test=m_test["ee_rmse"]))

    div_rows = []
    for d in diversity_values:
        scenes = sample_diverse_scenes(problem, rng, 10, d)
        gram, m_fit, m_test = one_set(scenes, f"div={d}")
        div_rows.append(dict(diversity=d, gram=gram, param_err=m_fit["theta_l1"],
                             e_demo=m_fit["ee_rmse"], e_test=m_test["ee_rmse"]))

    return dict(axis="demo_diversity", basis=basis, K=K, names=list(names),
               n_sweep=n_rows, diversity_sweep=div_rows)


# ---------------------------------------------------------------------------
# axis 3 -- basis size / representation
# ---------------------------------------------------------------------------

def test_basis_size(problem, theta_star_k3, cfg=SuiteConfig(), *,
                    n_contexts=10, n_holdout=10, rff_sizes=()):
    """Axis 3: is the cost representable, and is the dictionary over/under-
    complete?

    `rff_sizes` defaults to EMPTY: the unknown-cost (RKHS) arm is deliberately
    held back until recovery works with a known basis, since an RFF result is
    uninterpretable while the named-basis case is still failing -- there would
    be no way to tell a bad dictionary from the failure the other axes are
    chasing.  Pass e.g. `--rff-sizes 8,16,32` to re-enable it.

    `theta_star_k3` (length-3, over `bases.K3_NAMES`) generates every
    demonstration in this test -- the true cost is always the k3 one, and
    what varies is what basis the FIT is allowed to use:

    - `k3`: well-specified (fit basis == true basis) -- the ceiling.
    - `k9`, `k16`: OVER-complete kinematic bases (`ioc.robot.bases`'
      docstring) that can express the k3 cost via a subset of their weights
      plus zero elsewhere.
    - `rff[M]` for each `M` in `rff_sizes`: an UNKNOWN-cost basis
      (`bases.rff`) with no shared feature identity with k3 at all -- this is
      the representation question in its hardest form, whether a generic
      dictionary can match a hand-engineered one's `e_test` without ever
      being told the true feature names.

    Since only `k3` shares theta_star's coordinates, `param_err` is reported
    for `k3` only; every basis is compared on `e_demo`/`e_test`, which is
    well-defined regardless of what the fit basis's weights mean.

    How to interpret
    -----------------
    - `e_demo` low but `e_test` high on an OVER-complete basis (k9/k16) means
      the extra capacity is being used to overfit the specific training
      contexts rather than to express genuine cost structure -- the
      per-basis `gram.effective_rank` should also be checked: an
      over-complete basis with `effective_rank` well below its `K` is
      confirmed to be spending its extra dimensions on directions the demo
      set cannot pin down (this is `test_identifiability_conditioning`'s
      question, restated per-basis).
    - RFF matching or beating the named kinematic bases on `e_test` at
      moderate `M` says the recoverable cost-space structure does not
      require hand-engineered features on this problem; RFF requiring much
      larger `M` than any kinematic basis's `K` to match `e_test` quantifies
      the price of not knowing the feature identities in advance.
    - If NO basis, however large, drives `e_demo` toward zero, the true cost
      is not representable by ANY of these dictionaries and no amount of
      demonstration or optimizer tuning fixes it -- that is a modeling
      failure the other five axes cannot diagnose.
    """
    rng = np.random.default_rng(cfg.seed)
    residual_k3, _ = bases.kinematic(problem, "k3")
    pool = problem.sample_scenes(rng, (n_contexts + n_holdout) * 4)
    probe_inner, _ = build_inner(problem, residual_k3, pool, cfg)
    scenes_all, discard_rate, _ = prob.screen_scenes(
        problem, pool, probe_inner.stationarity, theta_star_k3, cfg.conv_tol,
        n_contexts + n_holdout,
    )
    scenes_fit = jax.tree.map(lambda a: a[:n_contexts], scenes_all)
    scenes_test = jax.tree.map(lambda a: a[n_contexts:], scenes_all)
    x0s_fit, x0s_test = problem.seeds(scenes_fit), problem.seeds(scenes_test)

    # Demos always come from the TRUE k3 cost, regardless of which basis fits it.
    demo_inner, _ = build_inner(problem, residual_k3, scenes_fit, cfg)
    _, _, demos_fit = prob.make_demos(problem, demo_inner.solve_implicit, scenes_fit,
                                      theta_star_k3, rng, 0.0)
    _, _, demos_test = prob.make_demos(problem, demo_inner.solve_implicit, scenes_test,
                                       theta_star_k3, rng, 0.0)

    def run_basis(residual_fn, names, label, theta_star=None):
        K = len(names)
        inner, _ = build_inner(problem, residual_fn, scenes_fit, cfg)
        gram = gram_certificate(inner, scenes_fit, demos_fit, K)
        z_hat, _ = fit(problem, inner, K, scenes_fit, demos_fit, x0s_fit,
                      seed=row_seed(cfg.seed, "basis_size", label),
                      n_steps=cfg.n_outer_steps, lr=cfg.lr,
                      n_starts=cfg.n_starts, screen_steps=cfg.screen_steps,
                      max_batch_rollouts=cfg.max_batch_rollouts)
        m_fit = score(problem, inner, z_hat, scenes_fit, demos_fit, x0s_fit,
                     theta_star if theta_star is not None else jnp.full((K,), 1.0 / K))
        m_test = score(problem, inner, z_hat, scenes_test, demos_test, x0s_test,
                      theta_star if theta_star is not None else jnp.full((K,), 1.0 / K))
        row = dict(basis=label, K=K, gram=gram, e_demo=m_fit["ee_rmse"], e_test=m_test["ee_rmse"])
        if theta_star is not None:
            row["param_err"] = m_fit["theta_l1"]
        return row

    rows = [run_basis(residual_k3, bases.K3_NAMES, "k3", theta_star_k3)]
    for name in ("k9", "k16"):
        residual_fn, names = bases.kinematic(problem, name)
        rows.append(run_basis(residual_fn, names, name))
    for M in rff_sizes:
        residual_fn, names = bases.rff(problem, M, jax.random.key(cfg.seed))
        rows.append(run_basis(residual_fn, names, f"rff[{M}]"))

    return dict(axis="basis_size", discard_rate=discard_rate, rows=rows)


# ---------------------------------------------------------------------------
# axis 4 -- identifiability / conditioning
# ---------------------------------------------------------------------------

def test_identifiability_conditioning(problem, theta_star, cfg=SuiteConfig(), *,
                                      basis="k3", n_contexts=10,
                                      excitation_values=(0.05, 0.25, 0.5, 1.0, 1.5),
                                      n_seeds=5):
    """Axis 4: is the cost fundamentally non-identifiable, or merely hard to
    optimize?

    Two controls, both on the SAME `n_contexts` clean demonstrations:

    - `collinear=True` (`bases.kinematic`'s deliberate control): the
      smoothness residual is replaced by a second copy of the effort
      residual, an EXACT redundancy no method can break. This is the
      ground-truth-known non-identifiable case every other reading is judged
      against.
    - `excitation_values` via `scale_obstacle`: shrinking `obs_radius` toward
      0 turns off the collision feature's activation everywhere, which
      should smoothly reproduce the same failure mode WITHOUT an engineered
      redundancy -- excitation, not basis design, is the cause.

    `param_stability` is the std-dev of `theta_hat` across `n_seeds`
    independent outer-optimization seeds at each excitation level: a fit
    that lands in a different place every seed despite similar loss values is
    the optimization-level symptom of a Gram null direction.

    How to interpret
    -----------------
    - The `collinear=True` row is the calibration point: `gram.lambda_min`
      should be ~0 (or several orders below the other eigenvalues) and
      `param_stability` large along the effort/smooth difference by
      construction. If it is NOT -- if collinear features still show a
      well-conditioned Gram -- something in `gram_certificate` or
      `bases.kinematic`'s `collinear` wiring is broken; trust nothing else in
      this test until that is fixed.
    - As `excitation` falls, `gram.lambda_min` -> 0 and `gram.cond` -> large
      should track `param_stability` -> large: agreement between a
      zero-forward-solve certificate and actual outer-optimization variance
      is the direct evidence that a bad fit at low excitation is a
      NON-IDENTIFIABILITY problem, not an optimizer problem, since the
      certificate never touches the optimizer at all.
    - If `param_stability` stays large even at `excitation=1.5` (well
      excited, well-conditioned Gram), the instability is NOT explained by
      identifiability -- move to `test_optimizer_correctness`.
    - `effective_rank` dropping smoothly with excitation (rather than a sharp
      cliff) says the basis has a continuum of near-null directions rather
      than one isolated redundancy, which argues for `test_basis_size`-style
      rank selection (retain the well-excited eigendirections, refit only
      those) over trying to fix the scene distribution.
    """
    residual_fn, names = bases.kinematic(problem, basis)
    residual_collinear, _ = bases.kinematic(problem, basis, collinear=True)
    K = len(names)
    rng = np.random.default_rng(cfg.seed)
    scenes = problem.sample_scenes(rng, n_contexts)

    def certificate_and_stability(residual, scenes_i, label):
        inner, _ = build_inner(problem, residual, scenes_i, cfg)
        x0s, _, demos = prob.make_demos(problem, inner.solve_implicit, scenes_i, theta_star, rng, 0.0)
        gram = gram_certificate(inner, scenes_i, demos, K)
        thetas = []
        for s in range(n_seeds):
            z0 = jnp.asarray(np.random.default_rng(1000 + s).normal(scale=0.5, size=K))
            z_hat, _ = fit(problem, inner, K, scenes_i, demos, x0s, seed=1000 + s,
                          n_steps=cfg.n_outer_steps, lr=cfg.lr, z0=z0)
            thetas.append(np.asarray(jax.nn.softmax(z_hat)))
        thetas = np.stack(thetas)
        return dict(label=label, gram=gram,
                   param_stability=float(np.mean(np.std(thetas, axis=0))),
                   theta_mean=thetas.mean(axis=0).tolist())

    collinear_row = certificate_and_stability(residual_collinear, scenes, "collinear")
    excitation_rows = []
    for e in excitation_values:
        scenes_e = scale_obstacle(scenes, e)
        excitation_rows.append(dict(excitation=e,
                                    **certificate_and_stability(residual_fn, scenes_e, f"exc={e}")))

    return dict(axis="identifiability_conditioning", basis=basis, K=K, names=list(names),
               collinear_control=collinear_row, excitation_sweep=excitation_rows)


# ---------------------------------------------------------------------------
# axis 5 -- optimizer / differentiation correctness
# ---------------------------------------------------------------------------

def test_optimizer_correctness(problem, theta_star, cfg=SuiteConfig(), *, basis="k3",
                               n_contexts=10, newton_iters=(5, 10, 20, 40, 80),
                               fd_eps_list=(1e-2, 1e-3, 1e-4, 1e-5)):
    """Axis 5: is the differentiable optimizer/gradient responsible for a bad
    fit, independent of demonstrations, diversity, basis or identifiability?

    Everything else in this suite is held at its best case: `n_contexts`
    well-excited scenes (`problem.sample_scenes`'s default anchoring), clean
    demonstrations, the well-specified `k3` basis.  Two things are swept:

    - `newton_iters`: the forward solver's iteration budget. `stationarity`
      (`||grad_x C||` at the returned point) is reported alongside
      `param_err`/`e_demo` at each budget -- the implicit adjoint and finite
      differences are BOTH invalid theory whenever `stationarity` is not
      small (`ioc.inner`'s module docstring), so this sweep finds the budget
      below which results elsewhere in this file cannot be trusted.
    - `fd_eps_list`, at the LARGEST `newton_iters` (best-converged forward
      solve): the implicit-adjoint gradient is compared against finite
      differences at several step sizes via `outer.fd_grad_fn`, following
      `ioc.robot.e1_identifiability.gradient_check`.

    How to interpret
    -----------------
    - `stationarity` should fall roughly monotonically as `newton_iters`
      grows and asymptote near the solver's numerical floor; `param_err`/
      `e_demo` should improve in step. If `param_err` is bad at LARGE
      `newton_iters` (stationarity already small), the forward solver's
      convergence is not the problem -- look at the gradient comparison
      instead, or move to `test_basis_size`/`test_identifiability_conditioning`.
    - The FD comparison: `cos` near 1.0 and `rel_err` bottoming out (not
      monotonically shrinking) as `eps` shrinks is the expected signature of
      an FD estimate whose error is dominated by truncation at large `eps`
      and by solver float-noise at small `eps` -- read the BEST `cos`/`rel_err`
      across the sweep, not the smallest-`eps` value alone, per `ioc.inner`'s
      documented float32 floor.
    - Low `cos` (well below the ~0.99+ range that held in `e1_identifiability`
      on this same problem) at every `eps`, WITH `stationarity` already
      small, is direct evidence of a bug in the adjoint or in `residual_fn`'s
      differentiability (e.g. a hard max/min reintroduced into a feature) --
      this is the one failure mode none of the other five axes can produce
      or explain, since they all assume the gradient is correct.
    """
    residual_fn, names = bases.kinematic(problem, basis)
    K = len(names)
    rng = np.random.default_rng(cfg.seed)
    scenes = problem.sample_scenes(rng, n_contexts)
    x0s = problem.seeds(scenes)

    rows = []
    best_inner, best_demos = None, None
    for n_iters in newton_iters:
        cfg_i = dataclasses.replace(cfg, n_newton=n_iters)
        inner, _ = build_inner(problem, residual_fn, scenes, cfg_i)
        _, _, demos = prob.make_demos(problem, inner.solve_implicit, scenes, theta_star, rng, 0.0)
        stat = jax.vmap(lambda x0, s: inner.stationarity(x0, theta_star, s))(x0s, scenes)
        z_hat, _ = fit(problem, inner, K, scenes, demos, x0s,
                      seed=row_seed(cfg.seed, "optimizer_correctness", n_iters),
                      n_steps=cfg.n_outer_steps, lr=cfg.lr,
                      n_starts=cfg.n_starts, screen_steps=cfg.screen_steps,
                      max_batch_rollouts=cfg.max_batch_rollouts)
        m = score(problem, inner, z_hat, scenes, demos, x0s, theta_star)
        rows.append(dict(n_newton=n_iters, stationarity_max=float(jnp.max(stat)),
                         stationarity_med=float(jnp.median(stat)),
                         param_err=m["theta_l1"], e_demo=m["ee_rmse"]))
        if n_iters == max(newton_iters):
            best_inner, best_demos = inner, demos

    loss_i = jax.jit(prob.make_outer(problem, best_inner.solve_implicit, scenes, best_demos, x0s))
    z0 = jnp.asarray(rng.normal(scale=0.5, size=K))
    gi = jax.grad(loss_i)(z0)
    fd_rows = []
    for eps in fd_eps_list:
        # batched=False: the vmapped probe path deviates by up to ~2e-10 at
        # eps=1e-6 through float reassociation, and this axis exists to judge a
        # gradient at exactly those eps -- take the bit-exact loop here.
        gfd = outer_opt.fd_grad_fn(loss_i, eps, batched=False)(z0)[1]
        fd_rows.append(dict(eps=eps, cos=metrics.cosine(gi, gfd),
                            rel_err=float(jnp.linalg.norm(gi - gfd) / (jnp.linalg.norm(gi) + 1e-30))))

    return dict(axis="optimizer_correctness", basis=basis, K=K, names=list(names),
               newton_sweep=rows, fd_check=fd_rows)


# ---------------------------------------------------------------------------
# axis 5b -- cross-method agreement in trajectory space
# ---------------------------------------------------------------------------

def test_method_agreement(problem, theta_star, cfg=SuiteConfig(), *, basis="k3",
                          n_contexts=10, n_holdout=10, budget_solves=3000,
                          fd_eps=1e-3, cma_sigma0=0.5):
    """Axis 5b: do implicit-diff, finite differences, and CMA-ES AGREE on the
    reconstructed trajectory error, given the same loss and the same budget?

    This exists because `test_optimizer_correctness`'s FD gradient check cannot
    settle the question it is asked to settle.  That check compares the adjoint
    against a forward-difference probe of the SAME solver, and the solver's own
    truncation (`stationarity` ~1e-4 even at the largest `n_newton` swept) sets
    a noise floor far above any usable `eps` -- so a low `cos` there is
    ambiguous between a wrong adjoint and an unprobeable loss, and neither
    float64 nor a smaller `eps` separates them (see `ioc.outer.fd_grad_fn`,
    which blames float32; the convergence floor dominates even in float64).

    The agreement test sidesteps the pointwise gradient entirely.  All three
    optimizers see the identical outer loss `prob.make_outer(...)` and start
    from the identical `z0`; what differs is only how each finds a direction:

        implicit   1 solve/context/step, direction from the adjoint
        fd         K+1 solves/context/step, direction from forward differences
        cmaes      lambda solves/context/generation, no direction at all

    CMA-ES is the arbiter: it needs nothing from the inner problem, not
    differentiability nor continuity, so it cannot be wrong in the way a
    gradient can.  Budgets are equalized in SOLVES (`ioc.outer`'s common
    currency), not steps, since per-step cost differs by ~K between methods.

    How to interpret
    -----------------
    Read `e_test` (held-out end-effector RMSE) first, `param_err_eig` second,
    and raw `param_err` last -- the Gram here is ill-conditioned, so methods can
    agree on behaviour while their raw weights differ along a low-`lambda`
    direction, and that disagreement is not evidence of a bug.

    - All three converging to the same `e_test`: the adjoint is FINE.  A low FD
      cosine in `test_optimizer_correctness` was then a measurement artifact of
      the probe, not a defect, and the recovery numbers elsewhere in this suite
      can be read as they stand.
    - `implicit` converging to a materially WORSE `e_test` than `cmaes` at equal
      budget: a real gradient defect -- the adjoint is steering somewhere the
      derivative-free search does not go, and every gradient-based result in
      this suite is suspect until it is fixed.
    - `implicit` converging BETTER than `cmaes`: expected, and the point of the
      method -- one solve per step versus a population buys many more steps at
      equal solves.  Not evidence of a bug in either.
    - `fd` tracking `cmaes` while `implicit` diverges from both isolates the
      defect to the adjoint rather than the loss; `fd` and `implicit` agreeing
      while `cmaes` beats both suggests the outer landscape is multi-modal and
      the gradient methods are in a different basin, which is a `z0`/multistart
      problem, not a correctness one.

    `budget_solves` is the honest knob: at too small a budget CMA-ES has not
    converged and losing to it means nothing, so `gens`/`steps` are reported
    per method and the traces are returned for a convergence check.
    """
    residual_fn, names = bases.kinematic(problem, basis)
    K = len(names)
    rng = np.random.default_rng(cfg.seed)

    scenes = problem.sample_scenes(rng, n_contexts)
    holdout = problem.sample_scenes(rng, n_holdout)
    inner, _ = build_inner(problem, residual_fn, scenes, cfg)

    x0s, _, demos = demo_pipeline(problem, inner, scenes, theta_star, rng, 0.0)
    x0s_h, _, demos_h = demo_pipeline(problem, inner, holdout, theta_star, rng, 0.0)

    stat = jax.vmap(lambda x0, s: inner.stationarity(x0, theta_star, s))(x0s, scenes)
    eigvals, eigvecs = gram_eigen(inner, scenes, demos, K)

    loss = jax.jit(prob.make_outer(problem, inner.solve_implicit, scenes, demos, x0s))
    loss_and_grad = jax.jit(jax.value_and_grad(
        prob.make_outer(problem, inner.solve_implicit, scenes, demos, x0s)))
    z0 = jnp.asarray(rng.normal(scale=0.5, size=K))

    def row(name, z_hat, trace, n_iters):
        m = score(problem, inner, z_hat, scenes, demos, x0s, theta_star)
        m_test = score(problem, inner, z_hat, holdout, demos_h, x0s_h, theta_star)
        return dict(
            method=name, n_iters=n_iters, solves_used=int(trace[-1][0]) if trace else 0,
            e_demo=m["ee_rmse"], e_test=m_test["ee_rmse"], regret=m["regret"],
            param_err=m["theta_l1"], param_cos=m["theta_cos"],
            theta_hat=m["theta_hat"],
            param_err_eig=eigen_projected_error(m["theta_hat"], theta_star,
                                                eigvals, eigvecs),
            final_loss=float(trace[-1][1]) if trace else None,
            trace=[(int(s), float(v)) for s, v in trace],
        )

    rows = []

    z_i, tr_i = outer_opt.adam(loss_and_grad, z0, lr=cfg.lr,
                               budget_solves=budget_solves,
                               solves_per_step=n_contexts, trace_best=True)
    rows.append(row("implicit", z_i, tr_i, len(tr_i)))

    fd_and_grad = outer_opt.fd_grad_fn(loss, fd_eps)
    z_f, tr_f = outer_opt.adam(fd_and_grad, z0, lr=cfg.lr,
                               budget_solves=budget_solves,
                               solves_per_step=(K + 1) * n_contexts, trace_best=True)
    rows.append(row("fd", z_f, tr_f, len(tr_f)))

    z_c, tr_c = outer_opt.cma_es(loss, z0, sigma0=cma_sigma0, seed=cfg.seed,
                                 budget_solves=budget_solves,
                                 solves_per_eval=n_contexts, trace_best=True)
    rows.append(row("cmaes", jnp.asarray(z_c), tr_c, len(tr_c)))

    e_tests = [r["e_test"] for r in rows]
    ref = next(r for r in rows if r["method"] == "cmaes")["e_test"]
    agreement = dict(
        e_test_spread=float(max(e_tests) - min(e_tests)),
        e_test_rel_spread=float((max(e_tests) - min(e_tests)) / (min(e_tests) + 1e-30)),
        implicit_vs_cmaes=float(
            next(r for r in rows if r["method"] == "implicit")["e_test"] / (ref + 1e-30)),
        fd_vs_cmaes=float(
            next(r for r in rows if r["method"] == "fd")["e_test"] / (ref + 1e-30)),
        theta_cos_implicit_cmaes=metrics.cosine(
            jnp.asarray(next(r for r in rows if r["method"] == "implicit")["theta_hat"]),
            jnp.asarray(next(r for r in rows if r["method"] == "cmaes")["theta_hat"])),
    )

    return dict(axis="method_agreement", basis=basis, K=K, names=list(names),
               budget_solves=budget_solves, z0=[float(v) for v in z0],
               stationarity_max=float(jnp.max(stat)),
               stationarity_med=float(jnp.median(stat)),
               gram=dict(eigvals=[float(v) for v in eigvals],
                         eigvecs=[[float(v) for v in col] for col in eigvecs.T],
                         cond=float(eigvals[0] / max(eigvals[-1], 1e-30))),
               methods=rows, agreement=agreement)


# ---------------------------------------------------------------------------
# axis 6 -- generalization / cost-space recovery
# ---------------------------------------------------------------------------

def test_generalization(problem, theta_star, cfg=SuiteConfig(), *, basis="k3",
                        n_contexts_values=(1, 3, 5, 10, 20), n_holdout=15,
                        n_global=40):
    """Axis 6: does reconstructing demonstrations mean the cost was recovered?

    For each `n_contexts` in the sweep, fits on `n_contexts` clean scenes and
    reports THREE numbers at that one fit, never collapsed into one:

        e_demo    RMSE on the training scenes themselves.
        e_test    RMSE on a FIXED held-out scene set (same size, same
                  distribution, never touched by the fit).
        e_global  mean regret over a large, freshly-sampled scene pool
                  (`global_regret`) -- the closest proxy this problem has to
                  "the recovered cost, evaluated everywhere".

    This is the test the user's framing identifies as central: behavioural
    fit (`e_demo`) is not the same claim as cost-space recovery (`e_global`),
    and the two are reported side by side by construction rather than as a
    single pass/fail reconstruction number.

    How to interpret
    -----------------
    - `e_demo` low, `e_test`/`e_global` high, PERSISTING as `n_contexts`
      grows: DEMONSTRATION OVERFITTING / incomplete cost-space recovery --
      the fit finds a theta that reproduces its training behaviour without
      generalizing, and this is not a small-sample artifact if it survives
      to the largest `n_contexts` tested. Cross-reference
      `test_demo_diversity`: this pattern with a low-`effective_rank` Gram at
      the same `n_contexts` says WHY -- the training scenes underdetermine
      theta even though they are perfectly reconstructed.
    - `e_demo` and `e_test`/`e_global` closing together as `n_contexts`
      grows: ordinary generalization improving with more data, the healthy
      case.
    - `e_test` and `e_global` diverging from EACH OTHER (not just from
      `e_demo`): `e_test` is a fixed small set and can be lucky/unlucky
      independent of true recovery; `e_global`'s larger pool is the more
      reliable number when the two disagree.
    - Report `param_err` alongside these: per `ioc.metrics.simplex_metrics`'s
      docstring, a large `param_err` with LOW `e_global` means the error
      lives along a weakly-identifiable direction that costs nothing
      behaviourally (harmless, matches `test_identifiability_conditioning`'s
      null directions); a large `param_err` WITH high `e_global` means the
      error is behaviourally consequential.
    """
    residual_fn, names = bases.kinematic(problem, basis)
    K = len(names)
    rng = np.random.default_rng(cfg.seed)

    pool = problem.sample_scenes(rng, (max(n_contexts_values) + n_holdout) * 4)
    probe_inner, _ = build_inner(problem, residual_fn, pool, cfg)
    scenes_all, discard_rate, _ = prob.screen_scenes(
        problem, pool, probe_inner.stationarity, theta_star, cfg.conv_tol,
        max(n_contexts_values) + n_holdout,
    )
    scenes_pool = jax.tree.map(lambda a: a[:max(n_contexts_values)], scenes_all)
    scenes_test = jax.tree.map(lambda a: a[max(n_contexts_values):], scenes_all)
    x0s_test = problem.seeds(scenes_test)

    rows = []
    for n in n_contexts_values:
        scenes_fit = jax.tree.map(lambda a: a[:n], scenes_pool)
        x0s_fit = problem.seeds(scenes_fit)
        inner, _ = build_inner(problem, residual_fn, scenes_fit, cfg)
        _, _, demos_fit = prob.make_demos(problem, inner.solve_implicit, scenes_fit, theta_star, rng, 0.0)
        _, _, demos_test = prob.make_demos(problem, inner.solve_implicit, scenes_test, theta_star, rng, 0.0)

        z_hat, _ = fit(problem, inner, K, scenes_fit, demos_fit, x0s_fit,
                      seed=row_seed(cfg.seed, "generalization", n),
                      n_steps=cfg.n_outer_steps, lr=cfg.lr,
                      n_starts=cfg.n_starts, screen_steps=cfg.screen_steps,
                      max_batch_rollouts=cfg.max_batch_rollouts)
        m_fit = score(problem, inner, z_hat, scenes_fit, demos_fit, x0s_fit, theta_star)
        m_test = score(problem, inner, z_hat, scenes_test, demos_test, x0s_test, theta_star)
        e_glob = global_regret(problem, inner, z_hat, theta_star, rng, n_global)
        rows.append(dict(n_contexts=n, param_err=m_fit["theta_l1"], param_cos=m_fit["theta_cos"],
                         e_demo=m_fit["ee_rmse"], e_test=m_test["ee_rmse"], e_global=e_glob))

    return dict(axis="generalization", basis=basis, K=K, names=list(names),
               discard_rate=discard_rate, rows=rows)


# ---------------------------------------------------------------------------
# axis 7 -- does the rank-restricted refit actually help?
# ---------------------------------------------------------------------------

def make_path_fn(problem, inner, scene, x0):
    """`z -> (T, 3)` end-effector path for ONE scene: the readout whose Jacobian
    `d(path)/dz` is the sensitivity Gram's square root (`ioc.identifiability`)."""
    def path(z):
        theta = jax.nn.softmax(z)
        x = inner.solve_implicit(x0, theta, scene)
        return problem.ee_positions(problem.unpack(x, scene))
    return path


def stacked_jacobian(problem, inner, scenes, x0s, z, n_scenes):
    """`J = d(all ee paths)/dz`, built ONE SCENE AT A TIME and stacked.

    `G = J^T J = sum_i J_i^T J_i`, so stacking the per-scene Jacobians row-wise
    is exactly equivalent to differentiating all scenes at once -- and the SVD
    in `ioc.identifiability.sensitivity_spectrum` consumes the stacked `J`
    directly, so nothing is approximated here.

    Done per-scene because `jacrev` over the batched readout allocates one
    cotangent per output element across every scene simultaneously: measured at
    k3/10 scenes it requested a single 4.82 GiB block and drove the allocator
    into a fragmentation warning, which would only worsen on the k16 basis.
    Chunking divides that peak by `n_scenes` at identical result.
    """
    rows = []
    for i in range(n_scenes):
        scene_i = jax.tree.map(lambda a, i=i: a[i], scenes)
        jac_fn = make_jac_fn_cached(problem, inner, scene_i, x0s[i])
        rows.append(np.asarray(jac_fn(z), dtype=np.float64).reshape(-1, z.shape[0]))
    return np.concatenate(rows, axis=0)


_JAC_CACHE = {}


def make_jac_fn_cached(problem, inner, scene, x0):
    """One jitted `jacrev` reused across scenes: the traced shapes are identical
    scene to scene, so recompiling per scene would pay the (dominant) compile
    cost `n_scenes` times over."""
    key = id(inner)
    if key not in _JAC_CACHE:
        def path(z, scene, x0):
            theta = jax.nn.softmax(z)
            x = inner.solve_implicit(x0, theta, scene)
            return problem.ee_positions(problem.unpack(x, scene))
        _JAC_CACHE[key] = jax.jit(jax.jacrev(path))
    f = _JAC_CACHE[key]
    return lambda z: f(z, scene, x0)


def test_identifiable_refit(problem, theta_star_k3, cfg=SuiteConfig(), *,
                            bases_to_test=("k3", "k9", "k16"), n_contexts=10,
                            n_holdout=10, n_global=40, trace_frac=0.95):
    """Axis 7: `iosp/THEORY_IDENTIFIABLE_REFIT.md`'s stages 1-4, measured.

    The other six axes diagnose WHY recovery fails; this one tests the proposed
    FIX.  For each basis: fit wide on all K (stage 1), eigendecompose the
    sensitivity Gram `J^T J` with `J = d(ee path)/dz` (stage 2), take the
    95%-cumulative-trace rank (stage 3), refit on that subspace (stage 4), and
    score init/wide/refit on the same three numbers `test_generalization` uses.

    Two confounds that dominate the pick-and-place caller are ABSENT here, which
    is the point of running it on one segment first:

    - No units mismatch.  Every coordinate of `z` is a softmax logit, where
      `iosp/study3` mixes standoff distances in metres with logits and needed an
      explicit rescale before its spectrum meant anything.
    - No composition.  One segment, so a rank deficiency cannot be blamed on
      four chained solves.

    The softmax gauge (`softmax(z + c*1) == softmax(z)`) survives, so a genuine
    null direction along `1` is EXPECTED; `gauge_index` reports where it lands
    in the descending spectrum.  It should be last.  If it is not, the Jacobian
    or the readout is wrong and the rank number means nothing.

    How to interpret
    -----------------
    - `refit` beating `wide` on `e_test`/`e_global` at equal-or-better `e_demo`
      is the theory working: pinning the unidentifiable component at the prior
      buys generalization at no reconstruction cost.
    - `refit` matching `wide` on `e_demo` but NOT improving `e_test`/`e_global`
      means the discarded directions were behaviourally inert anyway -- the
      refit is harmless but pointless at this `r`, and `captured_frac` (how much
      of `z_star - z_prior` lives in `span(U_r)`) says whether `r` was cut too
      aggressively.
    - `refit` WORSE than `wide` on `e_demo` says the retained subspace cannot
      express the fit the wide search already found: either `trace_frac` is too
      low, or the spectrum is degenerate at the cut (check `eigvals` for ties
      straddling `r` -- splitting a degenerate pair keeps one coordinate and
      discards its twin arbitrarily, which happened at r=2 in `iosp/study3`).
    - Compare across `bases_to_test`.  Note the ceiling is `K-1`, not `K`: the
      softmax gauge is an EXACT null direction of `d(path)/dz`, so one unit of
      rank is always spent on it and `r == K` should never occur.  `k3` is
      well-specified, so `r == K-1 == 2` is its healthy reading and the refit
      should be close to a no-op there; `k9`/`k16` are over-complete, so
      `r << K-1` is expected and is where the refit must earn its keep.  A
      refit that materially helps on k3 -- where the only discarded direction
      is the gauge, which by construction changes no behaviour -- indicates a
      problem in the refit or the spectrum, not a success.
    """
    from ioc import identifiability as ident

    rng = np.random.default_rng(cfg.seed)
    residual_k3, _ = bases.kinematic(problem, "k3")
    pool = problem.sample_scenes(rng, (n_contexts + n_holdout) * 4)
    probe_inner, _ = build_inner(problem, residual_k3, pool, cfg)
    scenes_all, discard_rate, _ = prob.screen_scenes(
        problem, pool, probe_inner.stationarity, theta_star_k3, cfg.conv_tol,
        n_contexts + n_holdout,
    )
    scenes_fit = jax.tree.map(lambda a: a[:n_contexts], scenes_all)
    scenes_test = jax.tree.map(lambda a: a[n_contexts:], scenes_all)
    x0s_fit, x0s_test = problem.seeds(scenes_fit), problem.seeds(scenes_test)

    demo_inner, _ = build_inner(problem, residual_k3, scenes_fit, cfg)
    _, _, demos_fit = prob.make_demos(problem, demo_inner.solve_implicit, scenes_fit,
                                      theta_star_k3, rng, 0.0)
    _, _, demos_test = prob.make_demos(problem, demo_inner.solve_implicit, scenes_test,
                                       theta_star_k3, rng, 0.0)

    rows = []
    for label in bases_to_test:
        residual_fn, names = bases.kinematic(problem, label)
        K = len(names)
        inner, _ = build_inner(problem, residual_fn, scenes_fit, cfg)
        loss = prob.make_outer(problem, inner.solve_implicit, scenes_fit, demos_fit, x0s_fit)
        gf = jax.jit(jax.value_and_grad(loss))
        z_prior = jnp.zeros(K, dtype=bases.x_dtype())

        z_wide, _ = ident.wide_fit(gf, z_prior, n_steps=cfg.n_outer_steps, lr=cfg.lr)

        t0 = time.perf_counter()
        J = stacked_jacobian(problem, inner, scenes_fit, x0s_fit, z_wide, n_contexts)
        eigvals, eigvecs = ident.sensitivity_spectrum(lambda _z: J, z_wide)
        gram_s = time.perf_counter() - t0
        # Report BOTH selectors: they disagree materially (gap r=2 vs trace r=1
        # on k3 alone), and which one is right is exactly what this axis tests.
        top, _, r = ident.select_rank(eigvals, trace_frac, rule="gap")
        _, _, r_trace = ident.select_rank(eigvals, trace_frac, rule="trace")
        ident.report_loadings(eigvals, eigvecs, names)

        # softmax gauge must be the LAST eigendirection; if not, J is wrong.
        gauge = np.ones(K) / np.sqrt(K)
        gauge_index = int(np.argmax(np.abs(eigvecs.T @ gauge)))

        U_r = eigvecs[:, top]
        z_refit, _ = ident.refit_on_subspace(gf, z_prior, U_r,
                                             n_steps=cfg.n_outer_steps, lr=cfg.lr)

        # theta* only exists in the k3 basis; a misspecified basis has no
        # ground truth, so the reference is the uniform weight.  This was
        # guarded at the two `score` calls but NOT at `global_regret`, which
        # passed the 3-vector into a K=9 solver and raised
        # "dot_general ... got (3,) and (9,)" -- killing the axis at its second
        # basis on every seed.
        theta_ref = theta_star_k3 if K == 3 else jnp.full((K,), 1.0 / K)

        def three(z):
            mf = score(problem, inner, z, scenes_fit, demos_fit, x0s_fit, theta_ref)
            mt = score(problem, inner, z, scenes_test, demos_test, x0s_test, theta_ref)
            row = dict(e_demo=mf["ee_rmse"], e_test=mt["ee_rmse"],
                       e_global=global_regret(problem, inner, z, theta_ref, rng, n_global))
            if K == 3:
                row["param_err"] = mf["theta_l1"]
            return row

        rows.append(dict(basis=label, K=K, r=r, r_trace=r_trace, names=list(names),
                         eigvals=eigvals.tolist(), gauge_index=gauge_index,
                         gram_secs=gram_s,
                         init=three(z_prior), wide=three(z_wide), refit=three(z_refit)))
        print(f"    {label}: r={r}/{K} (trace rule would say {r_trace})  "
              f"gauge_index={gauge_index} (want {K-1})  "
              f"e_demo wide={rows[-1]['wide']['e_demo']:.5f} "
              f"refit={rows[-1]['refit']['e_demo']:.5f}  "
              f"e_global wide={rows[-1]['wide']['e_global']:.3e} "
              f"refit={rows[-1]['refit']['e_global']:.3e}", flush=True)

    return dict(axis="identifiable_refit", discard_rate=discard_rate,
                trace_frac=trace_frac, rows=rows)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

TESTS = {
    "demo_quality": test_demo_quality,
    "demo_diversity": test_demo_diversity,
    "basis_size": test_basis_size,
    "identifiability_conditioning": test_identifiability_conditioning,
    "optimizer_correctness": test_optimizer_correctness,
    "method_agreement": test_method_agreement,
    "generalization": test_generalization,
    "identifiable_refit": test_identifiable_refit,
}


def run_suite(which=None, cfg=SuiteConfig(), theta_star_k3=jnp.asarray([0.5, 0.3, 0.2]),
              rff_sizes=(), out=None, keep_going=True):
    """Run the requested axes (default: all six) and return a results dict
    keyed by axis name.  Each test function's docstring is the interpretation
    guide; this driver does no interpretation of its own.

    `out`, when given, is rewritten after EVERY axis rather than once at the
    end.  A whole-suite run is hours long and a late crash previously discarded
    every completed axis with it -- `demo_quality` (636 s) and `demo_diversity`
    (699 s) were both computed and both lost that way.  `keep_going` likewise
    records a failing axis as an `error` entry and continues, so one OOM (as in
    `identifiable_refit`) does not cost the axes queued behind it.
    """
    if not jax.config.jax_enable_x64:
        print("WARNING: x64 is OFF. The implicit adjoint inverts the inner Hessian; "
              "run with JAX_ENABLE_X64=1 or treat gradients as float32-noisy.")
    problem = prob.RobotProblem.load(cfg.urdf_path, cfg.srdf_path, cfg.mesh_dir, cfg.n_timesteps)
    print(f"jax devices: {jax.devices()}  T={cfg.n_timesteps}  theta*={np.asarray(theta_star_k3)}")

    names = list(TESTS) if which is None else which
    results = {}

    def checkpoint():
        if out is None:
            return
        with open(out, "w") as f:
            json.dump(results, f, indent=2, default=float)

    for name in names:
        print(f"\n=== {name} ===", flush=True)
        t0 = time.perf_counter()
        kw = dict(rff_sizes=rff_sizes) if name == "basis_size" and rff_sizes else {}
        try:
            results[name] = TESTS[name](problem, theta_star_k3, cfg=cfg, **kw)
        except Exception as e:  # noqa: BLE001 -- an axis failing must not sink the run
            if not keep_going:
                checkpoint()
                raise
            results[name] = dict(axis=name, error=f"{type(e).__name__}: {e}")
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
        else:
            print(f"  done in {time.perf_counter() - t0:.1f}s")
        checkpoint()
    return results


def main(which: str = "", out: str = "diagnostics_results.json",
        n_timesteps: int = 24, n_newton: int = 600, n_outer_steps: int = 60,
        lr: float = 0.15, seed: int = 0, conv_tol: float = 1e-4,
        early_stop: bool = False, rff_sizes: str = "", keep_going: bool = True):
    cfg = SuiteConfig(n_timesteps=n_timesteps, n_newton=n_newton,
                     n_outer_steps=n_outer_steps, lr=lr, seed=seed,
                     conv_tol=conv_tol, early_stop=early_stop)
    names = [w.strip() for w in which.split(",") if w.strip()] or None
    rff = tuple(int(m) for m in rff_sizes.split(",") if m.strip())
    results = run_suite(names, cfg, rff_sizes=rff, out=out, keep_going=keep_going)
    failed = [k for k, v in results.items() if isinstance(v, dict) and "error" in v]
    print(f"\nwrote {out}"
          + (f"  ({len(failed)} axis/axes failed: {', '.join(failed)})" if failed else ""))


if __name__ == "__main__":
    tyro.cli(main)
