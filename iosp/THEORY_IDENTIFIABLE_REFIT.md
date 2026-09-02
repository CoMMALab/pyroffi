# Rank-deficient IOC: fit wide, analyse the Gram matrix, refit on the identifiable basis

Transcription of the handwritten derivation (2026-08-26). It covers the case
where the number of cost directions the demonstrations can actually resolve,
`r`, is **smaller** than the number of cost terms fit, `K`. Two paths are
derived — one where a cost library is given, one where it is not — and they
share the same three-stage shape:

> **fit many cost terms → eigendecompose the cost-gradient Gram matrix →
> refit on the identifiable subspace only.**

Nothing here changes the `r = K` case. It only bites when `r < K`, which is
why it was invisible in the 2-D studies (`ioc/bench2d`, where `r ≈ K`) and
shows up as large residuals in the 7-DOF composed studies (`iosp/`).

---

## 0. Common machinery: the bilevel problem and its implicit derivative

**Forward (inner) problem.** For weights `θ` and context `c`,

    x*(θ, c) = argmin_x  C(x; θ, c),      C(x; θ, c) = Σ_{k=1..K} θ_k C_k(x, c)

`x*` is optimal when the stationarity condition holds:

    ∇_x C(x*; θ, c) = 0.

**Differentiate that identity w.r.t. `θ`:**

    ∇²_xx C(x*; θ, c) · dx*/dθ  +  ∇²_xθ C(x*; θ, c) = 0
                                            └── mixed second derivative
    Hessian ──┘

    ⇒   dx*/dθ = -[∇²_xx C]^{-1} ∇²_xθ C = -H^{-1} B.

**Bilevel (outer) problem.**

    argmin_θ  L(x*(θ, c), x_demo)      subject to   x*(θ, c) = argmin_x C(x; θ, c)

    θ_{k+1} = θ_k - η dL/dθ

    Chain rule:  dL/dθ = dL/dx* · dx*/dθ
    Implicit:    dL/dθ = dL/dx* · (-H^{-1} B)

This is exactly `ioc/inner.py::solve_implicit` (the `custom_vjp`) driving
`ioc/outer.py::adam`; see `iosp/THEORY.md` §2 for the three ways to get
`dx*/dθ` and when they disagree.

---

## 1. The identifiability object: the cost-gradient Gram matrix

Let the trajectory cost be linear in the weights:

    C(x; θ, c) = Σ_{k=1..K} θ_k φ_k(x, c) = θᵀ φ(x, c)

where `φ = [φ_1, …, φ_K]ᵀ` is the set of cost terms and `θ` weights their
importance.

**Local gradient approximation for demonstration `i`:**

    g_{ik} = ∇_x φ_k(x_i*, c_i)

**Aggregate cost-gradient Gram matrix** over the `N` demonstrations:

    G = Σ_{i=1..N} G_i,      (G_i)_{kl} = ⟨g_{ik}, g_{il}⟩

**Eigendecomposition.** Let

    G = U Λ Uᵀ

The **demonstration-induced identifiable subspace** is

    S = span{ u_j : λ_j > 0 }

with orthonormal basis `U`, projector `P = U Uᵀ`; keeping only the `r`
retained directions gives `U_r` and `P_r = U_r U_rᵀ`. Any weight vector splits
as

    θ = θ_identifiable + θ_null = U_r U_rᵀ θ + (I - U_r U_rᵀ) θ

A cost direction with `G θ = 0` is **unidentifiable**: moving `θ` along it
leaves the demonstrations unchanged, so no amount of outer optimisation can
recover it, and any error it carries is invisible to the outer loss.

    r = rank(G) = number of identifiable cost directions.

In practice `λ_j > 0` is a thresholded test, not an exact rank — the existing
code uses a 95 %-cumulative-trace rule
(`iosp/study1_minimal_identifiable.py`, `iosp/study0d_eigen_projected_recovery.py`).

---

## 2. Path A — known cost library (parametric)

**Assumption.** Demonstrations are given, together with a known cost basis
(curvature, smoothness, clearance, …), and we recover the cost *weights*.

- If `r = K`, `θ` is trivially identifiable.
- If `r < K`, only an `r`-dimensional **behavioural** cost basis is
  identifiable — the remaining `K - r` directions are not recoverable from
  this demonstration set, in this scene set, at all.

**Setup.**

    C_θ(x, c) = Σ_{k=1..K} θ_k φ_k(x, c)

where each `φ_k` is drawn from a rich, pre-specified set of created cost
functions.

    D = { (x_i^demo, c_i) }_{i=1..N}

    x_i*(θ) = argmin_x C(x, c_i; θ)

    θ* = argmin_θ Σ_i ℓ( x_i*(θ), x_i^demo )

    dL/dθ = dL/dx* · dx*/dθ         (implicit derivative from §0)

**Determine the identifiable basis.** Using the *solution* sensitivities
rather than the raw feature gradients:

    S_i = dx_i*/dθ ,      G = Σ_i S_iᵀ S_i ,      G = U Λ Uᵀ

then **re-parametrise with only the relevant cost terms** — refit in the
`U_r` coordinates, i.e. optimise `α ∈ R^r` with `θ = U_r α` (plus whatever
fixed offset the null component is pinned to), instead of optimising all `K`
weights.

Note the two admissible constructions of `G` are the feature-gradient one
(§1, `g_{ik} = ∇_x φ_k`, what `ioc/analytic.py::kkt_fit` and
`iosp/identifiability_check.py` build) and the sensitivity one
(`S_i = dx_i*/dθ`). The second is the one that matters for bilevel recovery:
it measures what the *outer loss* can see, which is the quantity the refit
must be restricted to.

---

## 3. Path B — unknown cost function (nonparametric / RKHS)

**Assumption.** Demonstrations are optimal, but the cost basis is *unknown*
and must be recovered.

- If `r = K`, trivially recoverable.
- If `r < K`, only an `r`-dimensional behavioural cost basis is identifiable:

      C(x) = Σ_{k=1..K} θ_k φ_k(x)          → true cost function
      C_identifiable(x) = Σ_{j=1..r} d_j ψ_j(x)   → identifiable cost function

  i.e. the recoverable object is an `r`-dimensional *functional* basis
  `{ψ_j}`, not the original `K` terms.

**Assume** `C* ∈ H`, an RKHS with kernel `k(τ, τ')`, and assume the
environment is fully modelled (the demonstrator's cost is the only unknown —
no unmodelled obstacle or dynamics masquerading as a cost term).

    D = { (x_i^demo, c_i) }_{i=1..N}

    C_w(x, c) = ⟨ w, Φ(x, c) ⟩_H       → candidate cost function

**Bilevel optimisation** — structurally identical to Path A, `θ → w`:

    x*(w, c) = argmin_x C_w(x, c)

    w* = argmin_w Σ_i ℓ( x*(w, c_i), x_i^demo )

    dL/dw = dL/dx* · dx*/dw            → same bilevel structure, same
                                         implicit-derivative code path

**Component-wise evaluation of identifiable directions:**

    S_i = dx_i*/dw ,      G = Σ_i S_iᵀ S_i ,      G = U Λ Uᵀ

and retain only the identifiable kernel directions — the `r` eigendirections
of `G` with `λ_j > 0` — as the recovered basis `{ψ_j}`, refitting the
coefficients `d` in that basis.

---

## 4. Why this is the fix for the 7-DOF residuals

The failure mode is not optimiser tuning and not demo scarcity. When `r < K`,
the outer fit is free to wander in the `(I - P_r)` null component; the fitted
`θ̂` picks up arbitrary null-space content that:

1. does not reduce the outer loss (the loss cannot see it), and
2. inflates every raw `‖θ̂ - θ*‖` metric identically to a genuine recovery
   error — which is exactly the confound
   `iosp/study0d_eigen_projected_recovery.py` was written to separate, and
   which `iosp/THEORY.md` §9 warns about for fit-demo error.

In the 2-D benchmarks `r ≈ K`, so the null component is empty and the raw
metric is honest — which is why this went unnoticed. In the 7-DOF composed
pick-and-place the measured spectrum is badly rank-deficient (the
`cos(transport.smooth, transport.upright) = -0.9999` collinearity and four
near-zero-gradient features, i.e. ~6 of 9 directions unresolvable from one
demo; `iosp/identifiability_check.py`, `iosp/THEORY.md` §7), so the null
component dominates and reconstruction looks poor.

**The procedure that follows from the above, for either path:**

1. Fit wide — all `K` terms (Path A) or the full RKHS candidate `w` (Path B).
2. Build `G` from the solution sensitivities at the fitted point, eigendecompose.
3. Select `r` (95 %-cumulative-trace, or an explicit `λ` threshold).
4. Re-parametrise onto `U_r` and **refit**, holding the null component fixed.
5. Report error and generalisation in the `U_r` projection, never as a raw
   `‖θ̂ - θ*‖` over all `K`.

Step 4 is the piece not yet implemented: `study1_minimal_identifiable.py`
narrows the parameterisation *by hand* (`K = 3` chosen a priori), and
`study0d_eigen_projected_recovery.py` does steps 2–3 and 5 diagnostically but
does not feed the subspace back into a refit.
