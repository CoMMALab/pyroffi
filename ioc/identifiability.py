"""Rank-deficient bilevel IOC: fit wide, eigendecompose the sensitivity Gram,
refit on the identifiable subspace only.  Path-agnostic implementation of
`iosp/THEORY_IDENTIFIABLE_REFIT.md` stages 1-4.

    stage 1  wide_fit                  -- optimize all K free coordinates
    stage 2  sensitivity_spectrum       -- SVD of J = d(readout)/du, squared
    stage 3  select_rank                -- 95%-cumulative-trace rule
    stage 4  refit_on_subspace          -- reparametrize onto U_r, refit alpha

None of this is specific to the pick-and-place composition; anything that
plugs in a `(value, grad) -> loss_and_grad` function and a `path_fn` for the
Jacobian can use it.  `iosp/study3_identifiable_refit.py` is the pickplace
caller: it builds `gf`/`jac_fn` from `PickPlaceProblem`, then calls the four
functions below.
"""

import jax
import numpy as np

from ioc import outer as outer_opt


def wide_fit(loss_and_grad, u0, *, n_steps, lr, **adam_kwargs):
    """Stage 1: fit all K free coordinates with no subspace restriction."""
    return outer_opt.adam(loss_and_grad, u0, lr=lr, n_steps=n_steps, **adam_kwargs)


def make_jac_fn(path_fn):
    """Stage 2's Jacobian, JITTED.  `jax.jacrev` vmaps internally over the
    output cotangents, so jitting turns what would be many sequential eager
    backward passes into one batched backward.

    Forward mode would be cheaper when K << output dim, but is unavailable
    whenever `path_fn` runs through `ioc.inner.solve_implicit`'s `custom_vjp`,
    which has no JVP rule.
    """
    return jax.jit(jax.jacrev(path_fn))


def sensitivity_spectrum(jac_fn, u):
    """Stage 2.  The spectrum of `G = J^T J`, `J = d(path)/du`: the outer
    loss's own Gauss-Newton curvature, i.e. the sensitivity Gram of
    `THEORY_IDENTIFIABLE_REFIT.md` §2.

    `G` is never formed explicitly: squaring `J` squares its condition number,
    which in float32 pushes anything below ~1e-7 relative into pure noise.
    Taking `svd(J)` and squaring the singular values gives the identical
    spectrum at twice the usable precision.

    Returns eigenvalues DESCENDING (trace-normalized) with matching
    eigenvectors as columns.
    """
    J = np.asarray(jac_fn(u), dtype=np.float64).reshape(-1, u.shape[0])
    _, s, Vt = np.linalg.svd(J, full_matrices=False)
    eigvals = s ** 2
    eigvals = eigvals / (np.sum(eigvals) / len(eigvals) + 1e-300)
    return eigvals, Vt.T


def select_rank(eigvals, frac=0.95, *, rule="gap"):
    """Stage 3: retained indices, discarded indices, and r.

    `rule="gap"` (default) cuts at the LARGEST RATIO between consecutive
    descending eigenvalues -- the numerical-rank boundary, where the spectrum
    falls off a cliff rather than merely tapering.

    `rule="trace"` is the original 95%-cumulative-trace rule, kept so earlier
    results remain reproducible.  It should not be the default, because trace
    share answers "how much of the response does this direction carry", which is
    not the identifiability question.  A direction can be genuinely identifiable
    and still hold a small share of a trace dominated by one stiff direction.
    Two measurements:

      single segment, k3, 10 clean scenes, WELL-SPECIFIED basis (theta_star =
      [0.5, 0.3, 0.2], so all three features genuinely matter):
          lam = [2.8888, 0.11123, 3.4313e-32]
          trace rule -> r = 1   (lam[0] alone is 96.29% of the trace, just over
                                 the 95% line, so `smooth` is discarded)
          gap rule   -> r = 2   (the 3e31 cliff to the softmax gauge direction)
      Since the softmax gauge is an exact null direction, r = K-1 = 2 is the
      correct answer here and the trace rule is off by one on the easiest case
      the suite has.

      composed pickplace, K=9:
          trace rule -> r = 2,  cutting THROUGH a degenerate pair (lam[1] ==
                                lam[2] == 0.24379, ratio 1.0), which keeps one
                                standoff coordinate and discards its twin
                                arbitrarily.
          gap rule   -> r = 6,  at the 1.0e5 cliff between lam[5]=6.41e-05 and
                                lam[6]=6.26e-10; the degenerate pair is safely
                                inside the retained set.

    So the trace rule was pinning FOUR identifiable directions at the prior on
    the composed model, which is enough on its own to explain a refit that
    cannot express the fit the wide search already found.  (It does NOT explain
    the wide fit's own reconstruction floor -- that is a separate question, since
    the wide fit has every coordinate free.)

    `frac` is used only by `rule="trace"`.
    """
    desc = np.argsort(eigvals)[::-1]
    ev = eigvals[desc]
    if rule == "trace":
        cum = np.cumsum(ev) / np.sum(ev)
        r = int(np.searchsorted(cum, frac) + 1)
    elif rule == "gap":
        # +1e-300 guards the exact zeros that an exact null direction produces.
        r = int(np.argmax(ev[:-1] / (ev[1:] + 1e-300)) + 1)
    else:
        raise ValueError(f"unknown rule {rule!r}; expected 'gap' or 'trace'")
    return desc[:r], desc[r:], r


def refit_on_subspace(loss_and_grad, z_prior, U_r, *, n_steps, lr, **adam_kwargs):
    """Stage 4.  Optimize `alpha` under `z(alpha) = z_prior + U_r alpha` and
    return `(z_hat, alpha_hat)`.  The `(I - U_r U_r^T)` component of `z` never
    moves off `z_prior`: what the demo failed to determine is pinned, by
    construction, rather than left to wander.

    Deliberately NOT whitened by `Lambda_r^{-1/2}`: with `ioc.outer.adam`
    (AdamW) already normalizing per-coordinate by `sqrt(v_hat)`, an extra
    `1/sqrt(lam)` rescale only shrinks the step in the directions that carry
    the most trace, which measurably slowed convergence on Path B without
    improving Path A. `lam_r` is intentionally not a parameter here for that
    reason -- callers that select a rank should not also be tempted to
    whiten it.
    """
    import jax.numpy as jnp

    U_r = jnp.asarray(np.asarray(U_r, dtype=np.float32), dtype=z_prior.dtype)

    def z_of(alpha):
        return z_prior + U_r @ alpha

    def loss_and_grad_alpha(alpha):
        val, gz = loss_and_grad(z_of(alpha))
        return val, U_r.T @ gz  # exact chain rule, no extra solve

    alpha0 = jnp.zeros(U_r.shape[1], dtype=z_prior.dtype)
    alpha_hat, trace = outer_opt.adam(loss_and_grad_alpha, alpha0, lr=lr,
                                      n_steps=n_steps, **adam_kwargs)
    return z_of(alpha_hat), np.asarray(alpha_hat)


def report_loadings(eigvals, eigvecs, names, n_show=4, thresh=0.20):
    """Diagnostic: what the retained eigendirections are actually made of.
    Without this, a rank number `r` is uninterpretable -- `r=2` means very
    different things depending on which named coordinates load onto it."""
    print("  eigendirection loadings (|component| >= "
          f"{thresh}, descending eigenvalue):", flush=True)
    for j in range(min(n_show, len(eigvals))):
        v = eigvecs[:, j]
        big = np.argsort(np.abs(v))[::-1]
        terms = [f"{v[i]:+.3f}*{names[i]}" for i in big if abs(v[i]) >= thresh]
        print(f"    lam={eigvals[j]:11.4e}  " + ("  ".join(terms) or "(diffuse)"),
              flush=True)
