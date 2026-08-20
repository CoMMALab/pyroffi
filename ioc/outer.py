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

import jax.numpy as jnp
import numpy as np


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
):
    """Adam on z, returning (best_z, trace).

    Give either `n_steps` (fixed step count) or `budget_solves` (run until the
    next step would exceed a solve budget, which is how methods with different
    per-step costs are compared on equal terms).  `max_steps` bounds the loop
    for the zero-solve fits (KKT, CIOC), where `solves_per_step` is 0 and a
    solve budget alone would never be reached.

    `trace_best` records the running best rather than the current loss; the two
    trace conventions are kept because the robot experiments plot per-step loss
    curves and the 2D benchmarks read "solves to reach L < tol" off the trace.
    """
    if (n_steps is None) == (budget_solves is None):
        raise ValueError("pass exactly one of n_steps / budget_solves")

    z, m, v = z0, jnp.zeros_like(z0), jnp.zeros_like(z0)
    best_val, best_z, trace, used, t = np.inf, z0, [], 0, 0
    while True:
        if n_steps is not None:
            if t >= n_steps:
                break
        elif used + solves_per_step > budget_solves or t >= max_steps:
            break
        t += 1
        val, g = loss_and_grad(z)
        used += solves_per_step
        fval = float(val)
        if fval < best_val:
            best_val, best_z = fval, z
        trace.append((used, best_val if trace_best else fval))
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g**2
        mh, vh = m / (1 - 0.9**t), v / (1 - 0.999**t)
        z = z - lr * mh / (jnp.sqrt(vh) + 1e-8)
    return best_z, trace


def fd_grad_fn(loss, eps):
    """Forward finite differences: a (value, gradient) function costing K+1 solves.

    This is the baseline the whole study is measured against.  It treats the
    solver as a black box -- no differentiable inner problem is required -- and
    pays for that with a per-step cost linear in the number of cost parameters.
    It is also the reference used to *validate* the adjoint, with one caveat
    that recurs throughout: FD cannot validate a gradient on a float32 solver,
    because the solver's noise floor (~1e-6) swamps an eps-sized probe.
    """

    def grad_fn(z):
        val = loss(z)
        g = [(loss(z.at[k].add(eps)) - val) / eps for k in range(z.shape[0])]
        return val, jnp.stack(g)

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
        fs = np.array([float(loss(jnp.asarray(x))) for x in X])
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
