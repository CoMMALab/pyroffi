"""Outer-loop optimizers over the unconstrained weight parameterization z.

The outer variable is z, with theta = softmax(z) so the weights stay on the
simplex without a constrained optimizer.  All three methods here see only the
outer loss L(z); what separates them is how they get a search direction and,
crucially, how many *forward trajectory-optimization solves* that costs, since
the solve is the unit of work that dominates everything else:

    implicit / unrolled   1 solve per context per step   (gradient from `ioc.inner`)
    fd                    K+1 solves per context per step
    cmaes                 lambda ~ 4 + 3 ln K solves per context per generation

Solve counts are the common currency of every comparison in this study, so both
optimizers here report a `(solves_used, loss)` trace and both return their
*best-seen* iterate rather than their last: CMA-ES intrinsically reports its
best-ever candidate, so returning Adam's final iterate would compare a noisy
last step against a curated best and understate the gradient methods --
especially at high demonstration noise, where the outer landscape is rough.  The
loss at z_t is already computed by the gradient function, so this is free.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax


def _n_steps(n_steps, budget_solves, solves_per_step, max_steps):
    """Step count, resolved on the host before any device work.

    Both stopping rules are known up front -- `solves_per_step` is a constant,
    so a solve budget is just a step count in disguise -- which is what lets the
    loop below be a fixed-length `lax.scan` instead of a Python `while` that
    dispatches once per outer step.  `solves_per_step == 0` is the zero-solve
    fits (KKT, CIOC), where only `max_steps` bounds the run.
    """
    if n_steps is not None:
        return n_steps
    if solves_per_step <= 0:
        return max_steps
    return min(budget_solves // solves_per_step, max_steps)


def _adam_core(loss_and_grad, z0, opt, n_steps):
    """The AdamW iteration as one `lax.scan`; returns (z_final, best_z, vals, bests).

    Pure and fixed-length, so it is both vmappable and free of per-step host
    interaction.  Best-so-far is carried on device exactly as the old Python
    loop tracked it (compare at z, *then* step), so the returned iterate is
    unchanged.

    Rank-agnostic in the parameter: `z0` may be a single (K,) vector, in which
    case `loss_and_grad` returns a scalar, or a (C, K) stack of C INDEPENDENT
    fits, in which case it returns a (C,) vector of per-row losses and a (C, K)
    stack of their gradients (see `adam_multi`).  `best_val`'s shape is read off
    the loss with `jax.eval_shape` -- free, no computation -- because a `scan`
    carry may not change shape between the initial value and the first step.
    """
    val_shape = jax.eval_shape(lambda z: loss_and_grad(z)[0], z0).shape

    def step(carry, _):
        z, st, best_val, best_z = carry
        val, g = loss_and_grad(z)
        better = val < best_val
        best_val = jnp.where(better, val, best_val)
        # `better[..., None]` is what makes this work for both ranks: a scalar
        # loss broadcasts against (K,), a (C,) loss against (C, K).
        best_z = jnp.where(better[..., None], z, best_z)
        upd, st = opt.update(g, st, z)
        return (optax.apply_updates(z, upd), st, best_val, best_z), (val, best_val)

    init = (z0, opt.init(z0), jnp.full(val_shape, np.inf, dtype=z0.dtype), z0)
    (z_final, _, _, best_z), (vals, bests) = jax.lax.scan(
        step, init, None, length=n_steps)
    return z_final, best_z, vals, bests


def adam(
    loss_and_grad,
    z0,
    *,
    lr,
    n_steps=None,
    budget_solves=None,
    solves_per_step=0,
    max_steps=100_000,
    trace_best=False,
    weight_decay=0.0,
):
    """AdamW (`optax.adamw`) on z, returning (best_z, trace).

    Give either `n_steps` (fixed step count) or `budget_solves` (run until the
    next step would exceed a solve budget, which is how methods with different
    per-step costs are compared on equal terms).  `max_steps` bounds the loop
    for the zero-solve fits (KKT, CIOC), where `solves_per_step` is 0 and a
    solve budget alone would never be reached.

    `trace_best` records the running best rather than the current loss; the two
    trace conventions are kept because the robot experiments plot per-step loss
    curves and the 2D benchmarks read "solves to reach L < tol" off the trace.

    `weight_decay=0.0` (the default, matching every existing call site) makes
    this identical to plain Adam; pass a positive value to get AdamW's decoupled
    decay.

    The iteration itself is `_adam_core`'s `lax.scan`: both stopping rules
    reduce to a step count known before the first dispatch (see `_n_steps`), so
    nothing about the loop needs the host.  `loss_and_grad` is therefore traced
    once, into the scan body, rather than dispatched per step -- which also
    means it must be traceable (every call site already passes a jitted
    value-and-grad).  The trace is one device->host transfer at the end.
    """
    if (n_steps is None) == (budget_solves is None):
        raise ValueError("pass exactly one of n_steps / budget_solves")

    steps = _n_steps(n_steps, budget_solves, solves_per_step, max_steps)
    opt = optax.adamw(lr, weight_decay=weight_decay)
    _, best_z, vals, bests = _adam_core(loss_and_grad, z0, opt, steps)
    used = np.arange(1, steps + 1) * solves_per_step
    out = np.asarray(bests if trace_best else vals)
    return best_z, list(zip((int(u) for u in used), (float(v) for v in out)))


def adam_scan(loss_and_grad, z0, *, lr, n_steps, weight_decay=0.0,
              return_best=False):
    """`lax.scan` form of `adam` with no host interaction at all.

    Where `adam` still builds a host-side trace at the end, this returns
    everything on device, so `jax.vmap(adam_scan, ...)` runs many INDEPENDENT
    fits -- different cost seeds, different IK branches -- as one batched
    program on one device.

    Returns (z, losses) with `losses` of shape (n_steps,), the loss AT each
    iterate.  `z` is the final iterate by default: with many candidates fitted
    in parallel the selection has to happen once, at the end, over converged
    results, and returning a per-candidate running best would smuggle a hard
    argmin back inside the batched program -- exactly the discontinuity that
    multi-candidate batching exists to avoid.  Callers pick the winner
    themselves, on TRAINING loss (selecting on held-out loss is test-set
    leakage).

    `return_best=True` opts back in to `adam`'s best-seen-iterate convention.
    It exists for the single-fit analytic baselines (`ioc.analytic`), which are
    not multi-candidate and whose published numbers are best-of-trajectory;
    do not use it for a vmapped population.
    """
    opt = optax.adamw(lr, weight_decay=weight_decay)
    z_final, best_z, vals, _ = _adam_core(loss_and_grad, z0, opt, n_steps)
    return (best_z if return_best else z_final), vals


def adam_multi(
    loss_and_grad,
    U0,
    *,
    lr,
    n_steps=None,
    budget_solves=None,
    solves_per_step=0,
    max_steps=100_000,
    trace_best=False,
    weight_decay=0.0,
):
    """`adam` for C INDEPENDENT fits carried as ROWS of `U` -- the flattening.

    `loss_and_grad(U) -> (vals (C,), grads (C, K))`, where row c's loss depends
    only on row c of `U`.  That independence is the whole trick, and it is worth
    stating precisely because it is what makes this exact rather than an
    approximation: if `L_c` depends only on `U[c]`, then

        d/dU sum_c L_c(U[c])  ==  the stack of each row's own gradient,

    so ONE value-and-grad of the summed loss gives every fit its true gradient,
    with no interaction between rows and no per-fit Python loop.  Adam is
    elementwise, so it then updates the whole (C, K) array as C independent
    optimizers.  Borrowed from `iosp.fit.multistart.build`, which uses the same
    structure to run (IK branch x cost seed) candidates as one batched program.

    The payoff is occupancy, not arithmetic: the per-fit work is one batched
    trajectory solve, and on a benchmark whose batch is a handful of contexts
    the GPU is nowhere near saturated, so C fits cost about what one costs.
    MEASURED on `bench2d` field (M=8 contexts, K=8): a value-and-grad step is
    22.2 ms at C=1 and 22.8 ms at C=8 -- 8 fits for 3% more wall-clock.  It also
    pays compilation ONCE instead of once per fit, which the serial loop did not
    amortize (each fit built a fresh closure, so the jit cache missed every
    time: measured 1.7 s per fit per method).

    Returns `(best_U, traces)` with `best_U` the per-row best-seen iterate and
    `traces` a list of C traces in `adam`'s own `(solves_used, loss)` format,
    so a caller can record each fit exactly as if it had been run alone.
    """
    if (n_steps is None) == (budget_solves is None):
        raise ValueError("pass exactly one of n_steps / budget_solves")

    steps = _n_steps(n_steps, budget_solves, solves_per_step, max_steps)
    opt = optax.adamw(lr, weight_decay=weight_decay)
    _, best_U, vals, bests = _adam_core(loss_and_grad, U0, opt, steps)
    used = np.arange(1, steps + 1) * solves_per_step
    out = np.asarray(bests if trace_best else vals)          # (steps, C)
    traces = [list(zip((int(u) for u in used), (float(v) for v in out[:, c])))
              for c in range(U0.shape[0])]
    return best_U, traces


def summed_grad_fn(loss_rows):
    """Turn a row-wise loss into `adam_multi`'s `(vals, grads)` interface.

    `loss_rows(U) -> (C,)`.  The sum is taken only so reverse-mode AD has a
    scalar to differentiate; see `adam_multi` for why the result is each row's
    own gradient and not a mixture.
    """

    def loss_and_grad(U):
        vals, vjp = jax.vjp(loss_rows, U)
        (grads,) = vjp(jnp.ones_like(vals))
        return vals, grads

    return loss_and_grad


def fd_grad_multi_fn(loss_rows, eps, n_fits, K):
    """Finite differences for `n_fits` independent fits, as ONE batched call.

    Row-major layout: fit c owns rows `c*(K+1) .. c*(K+1)+K` of the probe stack,
    row 0 of that block being the base point and rows 1..K the coordinate
    perturbations.  `loss_rows` must accept the whole `(n_fits*(K+1), K)` stack
    and evaluate each row against ITS OWN fit's data -- `fit_index` below is the
    map from probe row to fit, which is what the caller uses to gather the right
    contexts and demonstrations per row.

    The method's cost is unchanged and still K+1 solves per fit per step, which
    is the currency every comparison in this study is stated in; only the
    dispatch differs.
    """
    block = K + 1

    def grad_fn(U):
        # (n_fits, K+1, K): the base point repeated, then one eps bump per axis.
        P = jnp.broadcast_to(U[:, None, :], (n_fits, block, K))
        P = P.at[:, 1:, :].add(eps * jnp.eye(K))
        vals = loss_rows(P.reshape(n_fits * block, K)).reshape(n_fits, block)
        return vals[:, 0], (vals[:, 1:] - vals[:, :1]) / eps

    grad_fn.fit_index = np.repeat(np.arange(n_fits), block)
    return grad_fn


def fd_grad_fn(loss, eps, *, batched=True):
    """Forward finite differences: a (value, gradient) function costing K+1 solves.

    This is the baseline the whole study is measured against.  It treats the
    solver as a black box -- no differentiable inner problem is required -- and
    pays for that with a per-step cost linear in the number of cost parameters.
    It is also the reference used to *validate* the adjoint, with one caveat
    that recurs throughout: FD cannot validate a gradient on a float32 solver,
    because the solver's noise floor (~1e-6) swamps an eps-sized probe.


    The K+1 probes are independent solves, so they are evaluated as ONE
    batched call rather than a Python loop.  That does not change the method's
    cost in solves -- still K+1 per step, which is the currency every
    comparison here is stated in -- but it stops the *wall-clock* from scaling
    with K for purely implementation reasons.  Pass `batched=False` for a loss
    that cannot be `vmap`ped.
    """

    def grad_fn(z):
        if not batched:
            val = loss(z)
            g = [(loss(z.at[k].add(eps)) - val) / eps for k in range(z.shape[0])]
            return val, jnp.stack(g)
        # row 0 is the base point, rows 1..K the coordinate perturbations
        Z = jnp.broadcast_to(z, (z.shape[0] + 1, z.shape[0])).at[
            jnp.arange(1, z.shape[0] + 1), jnp.arange(z.shape[0])].add(eps)
        vals = jax.vmap(loss)(Z)
        return vals[0], (vals[1:] - vals[0]) / eps

    return grad_fn


def _cma_constants(n):
    """(mu/mu_w, lambda)-CMA-ES's fixed strategy parameters for dimension `n`.

    Host-side floats: they depend only on `n`, so they are baked into the traced
    program as constants rather than carried.
    """
    lam = 4 + int(3 * np.log(n))
    mu = lam // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mueff = 1.0 / np.sum(w**2)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    return dict(
        n=n, lam=lam, mu=mu, w=w, mueff=mueff, c1=c1,
        cc=(4 + mueff / n) / (n + 4 + 2 * mueff / n),
        cs=(mueff + 2) / (n + mueff + 5),
        cmu=min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff)),
        damps=1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1)
              + (mueff + 2) / (n + mueff + 5),
        chiN=np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2)),
    )


def cma_population_size(n):
    """`lambda` for dimension `n` -- the number of solves one generation costs.

    Public because a flattened caller has to know how many rows per fit a
    generation will produce before it can build the row -> fit map.
    """
    return _cma_constants(n)["lam"]


def _cma_n_gens(n_gens, budget_solves, lam, solves_per_eval):
    """Generation count, resolved on the host -- `_n_steps`'s CMA-ES twin.

    The population size is fixed, so a solve budget is a generation count known
    before the first dispatch, which is what lets the run be one `lax.scan`.
    """
    if n_gens is not None:
        return n_gens
    per_gen = lam * solves_per_eval
    if per_gen <= 0:
        raise ValueError("solves_per_eval must be positive with budget_solves")
    return budget_solves // per_gen


def _cma_init(z0, sigma0, dtype):
    """The strategy state for ONE fit: (mean, C, pc, ps, sigma, best_z, best_val)."""
    n = z0.shape[0]
    return (
        jnp.asarray(z0, dtype=dtype),                 # mean
        jnp.eye(n, dtype=dtype),                      # covariance
        jnp.zeros(n, dtype=dtype),                    # pc
        jnp.zeros(n, dtype=dtype),                    # ps
        jnp.asarray(sigma0, dtype=dtype),             # sigma
        jnp.asarray(z0, dtype=dtype),                 # best_z
        jnp.asarray(np.inf, dtype=dtype),             # best_val
    )


def _cma_sample(state, normals):
    """(Y, X) for one fit: `normals` (lam, n) shaped by the current covariance."""
    mean, C, _, _, sigma, _, _ = state
    d, B = jnp.linalg.eigh(C)
    d = jnp.sqrt(jnp.maximum(d, 1e-14))
    Y = normals @ (B * d).T
    return Y, mean + sigma * Y


def _cma_step(cons, dtype):
    """The rank-mu/rank-one covariance and step-size recursion for ONE fit.

    Kept as a function of a single fit's state so `cma_es` can scan it directly
    and `cma_es_multi` can `vmap` it over independent fits -- the two drivers
    then differ only in how the population is evaluated, not in the algorithm.
    """
    n, mu, mueff = cons["n"], cons["mu"], cons["mueff"]
    cc, cs, c1, cmu = cons["cc"], cons["cs"], cons["c1"], cons["cmu"]
    damps, chiN = cons["damps"], cons["chiN"]
    wj = jnp.asarray(cons["w"], dtype=dtype)

    def step(state, Y, X, fs, gen):
        mean, C, pc, ps, sigma, best_z, best_val = state
        # The covariance is re-decomposed here rather than threaded from
        # `_cma_sample`: `ps`'s update needs C^{-1/2}, and recomputing it keeps
        # the state a plain tuple of arrays (so `vmap` over fits is trivial).
        d, B = jnp.linalg.eigh(C)
        d = jnp.sqrt(jnp.maximum(d, 1e-14))
        order = jnp.argsort(fs)
        top = order[:mu]

        improved = fs[order[0]] < best_val
        best_val = jnp.where(improved, fs[order[0]], best_val)
        best_z = jnp.where(improved, X[order[0]], best_z)

        Yw = wj @ Y[top]
        mean = mean + sigma * Yw
        ps = (1 - cs) * ps + jnp.sqrt(cs * (2 - cs) * mueff) * ((B / d) @ B.T @ Yw)
        hsig = (
            jnp.linalg.norm(ps) / jnp.sqrt(1 - (1 - cs) ** (2 * gen)) / chiN
            < 1.4 + 2 / (n + 1)
        ).astype(dtype)
        pc = (1 - cc) * pc + hsig * jnp.sqrt(cc * (2 - cc) * mueff) * Yw
        Ymu = Y[top]
        C = (
            (1 - c1 - cmu) * C
            + c1 * (jnp.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
            + cmu * (Ymu.T * wj) @ Ymu
        )
        sigma = sigma * jnp.exp((cs / damps) * (jnp.linalg.norm(ps) / chiN - 1))
        return (mean, C, pc, ps, sigma, best_z, best_val), (fs[order[0]], best_val)

    return step


def cma_es(
    loss,
    z0,
    *,
    sigma0=0.5,
    seed=0,
    n_gens=None,
    budget_solves=None,
    solves_per_eval=1,
    trace_best=False,
    batched_eval=True,
):
    """Compact (mu/mu_w, lambda)-CMA-ES; avoids a new dependency on `cma`.

    The derivative-free bilevel baseline, and the closest prior art: Mombaur,
    Truong & Laumond (2010) fit human locomotion costs exactly this way, by
    resampling the forward solve.  It needs nothing from the inner problem --
    not differentiability, not even continuity -- which is why it remains the
    honest comparison whenever x*(theta) genuinely jumps between basins.  It
    no longer *wins* that regime outright: `inner.make_inner_solver`'s
    `n_restarts`/`restart_seed_fn` gives the implicit path the same batched
    multistart (structured per-basin seeds, e.g. `bench2d.problems.
    make_topo_seed_fn`, cheap because the restarts are solved in parallel
    within one JAX call), so implicit can recover basin coverage without
    switching to a derivative-free search.  CMA-ES's cost is still a
    population of solves per generation, so the gap to a one-solve gradient
    step widens with the cost dimension whenever both use n_restarts=1.

    Give either `n_gens` or `budget_solves`; the budget form is what makes the
    comparison against the gradient methods equal-work rather than equal-step.

    The strategy update runs in `jnp` inside a `lax.scan` over generations, so
    the whole run -- sampling, the population's batched evaluation, the
    eigendecomposition and the covariance/step-size recursions -- is a single
    device program.  The previous NumPy implementation forced a host sync per
    generation, which is a hard barrier between two batched solves.  Two
    consequences worth stating:

    * The population is drawn from `jax.random`, not `np.random.default_rng`,
      so `seed` indexes a *different* stream.  Runs are still deterministic and
      reproducible; they are not bit-identical to results recorded before this
      change, and seeded comparisons against those must be re-run.
    * The recursions inherit JAX's default precision.  They were `float64` by
      construction before; enable `jax_enable_x64` to keep that.  At the cost
      dimensions used here (K <= ~16) float32 is adequate, but the covariance
      recursion is the sensitive part if K grows.

    `batched_eval=False` evaluates the population with `lax.map` instead of
    `vmap`, for a loss that cannot be vmapped; it is sequential but still on
    device, so it does not reintroduce the per-candidate host sync.
    """
    if (n_gens is None) == (budget_solves is None):
        raise ValueError("pass exactly one of n_gens / budget_solves")

    n = z0.shape[0]
    cons = _cma_constants(n)
    lam = cons["lam"]
    n_gens = _cma_n_gens(n_gens, budget_solves, lam, solves_per_eval)
    dtype = z0.dtype if jnp.issubdtype(z0.dtype, jnp.floating) else jnp.float32

    def evaluate(X):
        return jax.vmap(loss)(X) if batched_eval else jax.lax.map(loss, X)

    step = _cma_step(cons, dtype)

    def generation(carry, _):
        key, state, gen = carry
        gen = gen + 1
        key, sub = jax.random.split(key)
        Y, X = _cma_sample(state, jax.random.normal(sub, (lam, n), dtype=dtype))
        fs = evaluate(X)
        state, out = step(state, Y, X, fs, gen)
        return (key, state, gen), out

    init = (jax.random.key(seed), _cma_init(z0, sigma0, dtype), 
            jnp.asarray(0, dtype=dtype))
    (_, state, _), (gen_best, run_best) = jax.lax.scan(generation, init, None,
                                                       length=n_gens)
    best_z = state[5]
    used = np.arange(1, n_gens + 1) * (lam * solves_per_eval)
    out = np.asarray(run_best if trace_best else gen_best)
    return best_z, list(zip((int(u) for u in used), (float(v) for v in out)))


def cma_es_multi(
    loss_rows,
    Z0,
    *,
    sigma0=0.5,
    seed=0,
    n_gens=None,
    budget_solves=None,
    solves_per_eval=1,
    trace_best=False,
):
    """`cma_es` for C INDEPENDENT runs at once -- the flattening, for the
    derivative-free baseline.

    `loss_rows(X) -> (R,)` is evaluated on a flat `(C*lam, K)` stack, row-major
    by fit: rows `c*lam .. c*lam+lam-1` are fit c's population and must be
    scored against fit c's own data (`fit_index` gives that map).  So the whole
    generation -- every fit's whole population -- is ONE batched call, and the
    strategy recursion is `vmap`ped over fits on top of it.

    Unlike the gradient methods there is no summed-loss trick to justify: the
    runs are genuinely independent, each carrying its own mean, covariance,
    step size and best-seen candidate.  The only thing shared is the dispatch.

    Each fit gets its OWN RNG key (`jax.random.split` of `seed`), so run c here
    is not the same stream as `cma_es(..., seed=c)`.  Both are deterministic;
    neither reproduces the other, and neither reproduces the pre-JAX NumPy
    implementation.

    Returns `(best_Z, traces)` in `cma_es`'s per-run format.
    """
    if (n_gens is None) == (budget_solves is None):
        raise ValueError("pass exactly one of n_gens / budget_solves")

    C, n = Z0.shape
    cons = _cma_constants(n)
    lam = cons["lam"]
    n_gens = _cma_n_gens(n_gens, budget_solves, lam, solves_per_eval)
    dtype = Z0.dtype if jnp.issubdtype(Z0.dtype, jnp.floating) else jnp.float32
    step = _cma_step(cons, dtype)

    def generation(carry, _):
        key, state, gen = carry
        gen = gen + 1
        key, sub = jax.random.split(key)
        normals = jax.random.normal(sub, (C, lam, n), dtype=dtype)
        Y, X = jax.vmap(_cma_sample)(state, normals)          # (C, lam, n) each
        fs = loss_rows(X.reshape(C * lam, n)).reshape(C, lam)
        state, out = jax.vmap(step, in_axes=(0, 0, 0, 0, None))(state, Y, X, fs, gen)
        return (key, state, gen), out

    init_state = jax.vmap(lambda z: _cma_init(z, sigma0, dtype))(Z0)
    init = (jax.random.key(seed), init_state, jnp.asarray(0, dtype=dtype))
    (_, state, _), (gen_best, run_best) = jax.lax.scan(generation, init, None,
                                                       length=n_gens)
    best_Z = state[5]
    used = np.arange(1, n_gens + 1) * (lam * solves_per_eval)
    out = np.asarray(run_best if trace_best else gen_best)    # (n_gens, C)
    traces = [list(zip((int(u) for u in used), (float(v) for v in out[:, c])))
              for c in range(C)]
    return best_Z, traces



def fit_index_for(n_fits, per_fit):
    """Row -> fit map for a flattened `(n_fits*per_fit, K)` stack, row-major.

    The one piece of bookkeeping every flattened driver here needs: `per_fit`
    is 1 for `adam_multi`, `K+1` for `fd_grad_multi_fn`, and `lam` for
    `cma_es_multi`.
    """
    return np.repeat(np.arange(n_fits), per_fit)
