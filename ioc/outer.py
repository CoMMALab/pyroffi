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
    """
    if (n_steps is None) == (budget_solves is None):
        raise ValueError("pass exactly one of n_steps / budget_solves")

    opt = optax.adamw(lr, weight_decay=weight_decay)
    z, opt_state = z0, opt.init(z0)
    best_val, best_z = jnp.asarray(np.inf, dtype=z0.dtype), z0
    trace_vals, trace_used, used, t = [], [], 0, 0
    while True:
        if n_steps is not None:
            if t >= n_steps:
                break
        elif used + solves_per_step > budget_solves or t >= max_steps:
            break
        t += 1
        val, g = loss_and_grad(z)
        used += solves_per_step
        # best tracked on device.  Measured 1.00x against the old per-step
        # float(val) on the robot problem -- the sync waits on a solve that has
        # to happen regardless -- so this is hygiene, not a speedup.  It is
        # kept because it makes the trace one transfer instead of n_steps.
        better = val < best_val
        best_val = jnp.where(better, val, best_val)
        best_z = jnp.where(better, z, best_z)
        trace_vals.append(best_val if trace_best else val)
        trace_used.append(used)
        updates, opt_state = opt.update(g, opt_state, z)
        z = optax.apply_updates(z, updates)
    # single device->host transfer for the whole trace
    vals = np.asarray(jnp.stack(trace_vals)) if trace_vals else np.zeros(0)
    trace = list(zip(trace_used, (float(v) for v in vals)))
    return best_z, trace


def adam_scan(loss_and_grad, z0, *, lr, n_steps, weight_decay=0.0):
    """`lax.scan` form of `adam`, so a whole bilevel fit is one vmappable call.

    `adam` above cannot be vmapped: it is a Python loop that appends to a host
    list and calls `float()` on the trace.  This version carries (z, opt_state)
    through a fixed-length scan with no host interaction, so
    `jax.vmap(adam_scan, ...)` runs many INDEPENDENT fits -- different cost
    seeds, different IK branches -- as one batched program on one device.

    Returns (z_final, losses) with `losses` of shape (n_steps,), the loss AT
    each iterate.  Deliberately NOT best-of-trajectory: with many candidates
    fitted in parallel the selection has to happen once, at the end, over
    converged results, and returning a per-candidate running best here would
    smuggle a hard argmin back inside the batched program -- exactly the
    discontinuity that multi-candidate batching exists to avoid.  Callers pick
    the winner themselves, on TRAINING loss (selecting on held-out loss is
    test-set leakage).
    """
    import optax as _optax

    opt = _optax.adamw(lr, weight_decay=weight_decay)

    def step(carry, _):
        z, st = carry
        val, g = loss_and_grad(z)
        upd, st = opt.update(g, st, z)
        return (_optax.apply_updates(z, upd), st), val

    (z_final, _), losses = jax.lax.scan(step, (z0, opt.init(z0)), None,
                                        length=n_steps)
    return z_final, losses


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
    """
    if (n_gens is None) == (budget_solves is None):
        raise ValueError("pass exactly one of n_gens / budget_solves")

    n = z0.shape[0]
    lam = 4 + int(3 * np.log(n))
    mu = lam // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mueff = 1.0 / np.sum(w**2)
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

    rng = np.random.default_rng(seed)
    mean = np.asarray(z0, dtype=np.float64)
    C, pc, ps, sigma = np.eye(n), np.zeros(n), np.zeros(n), sigma0
    best_val, best_z, trace, used, gen = np.inf, mean.copy(), [], 0, 0

    while True:
        if n_gens is not None:
            if gen >= n_gens:
                break
        elif used + lam * solves_per_eval > budget_solves:
            break
        gen += 1
        d, B = np.linalg.eigh(C)
        d = np.sqrt(np.maximum(d, 1e-14))
        Y = rng.standard_normal((lam, n)) @ (B * d).T
        X = mean + sigma * Y
        # whole population in one batched call: the candidates are independent
        # solves, and evaluating them one at a time also forced a host sync per
        # candidate.  Values are unchanged; only the dispatch differs.
        fs = np.asarray(jax.vmap(loss)(jnp.asarray(X))) if batched_eval else \
            np.array([float(loss(jnp.asarray(x))) for x in X])
        used += lam * solves_per_eval
        order = np.argsort(fs)
        if fs[order[0]] < best_val:
            best_val, best_z = fs[order[0]], X[order[0]].copy()
        trace.append((used, best_val if trace_best else float(fs[order[0]])))

        Yw = w @ Y[order[:mu]]
        mean = mean + sigma * Yw
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * ((B / d) @ B.T @ Yw)
        hsig = (
            np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * gen)) / chiN
            < 1.4 + 2 / (n + 1)
        )
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * Yw
        Ymu = Y[order[:mu]]
        C = (
            (1 - c1 - cmu) * C
            + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
            + cmu * (Ymu.T * w) @ Ymu
        )
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))

    return jnp.asarray(best_z), trace
