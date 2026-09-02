# IOSP theory: inverse optimal control as loss geometry

This document lays out the mathematical structure of `iosp/` for someone reading
it as a loss-landscape / geometry problem rather than a robotics problem. It is
a synthesis of the theory scattered across module docstrings
(`ioc/inner.py`, `ioc/analytic.py`, `iosp/pickplace.py`,
`iosp/identifiability_check.py`), not new theory — cross-reference those files
for the exact code and measured numbers behind each claim.

## 1. The object being studied

There is an **inner problem**: a parameterized family of loss landscapes

    J_theta(x, c) = sum_k (theta_k / s_k) * ||r_k(x, c)||^2,     x in R^n

indexed by cost weights `theta` (a point on the simplex, in the trajopt case)
and context `c` (a scene: start/goal, obstacles). `s_k` are fixed calibration
scales, so `theta` reweights *normalized* feature magnitudes, not raw units.
For fixed `(theta, c)` this is an ordinary nonlinear least-squares landscape
in `x`, solved by damped Gauss-Newton to a stationary point

    x*(theta, c) = argmin_x J_theta(x, c),    grad_x J_theta(x*, c) = 0.

There is an **outer problem**: given one or more demonstrations `x~_i` at
contexts `c_i`, recover the `theta` under which those demonstrations look
optimal (or as close to optimal as the family allows). This is inverse optimal
control (IOC) / inverse reinforcement learning in continuous state.

`iosp/` composes four such inner problems (approach/grasp/transport/place)
end-to-end with a differentiable IK stage feeding their boundary conditions,
so `x*` is really a tuple of four stationary points chained by shared boundary
values, and `theta = (theta_ik, theta_trajopt)` is the concatenated parameter.
Sections 2–6 build up the single-segment theory; Section 7 covers what
composition changes.

## 2. Three ways to differentiate `x*(theta)`, as three geometric objects

The outer loop needs `dx*/dtheta` — how the *location of the minimizer* moves
as you tilt the landscape. There are exactly three routes, and they correspond
to three different pieces of the landscape's geometry.

**(a) Implicit function theorem (`ioc/inner.py::solve_implicit`).**
`x*` is defined implicitly by `F(x*, theta) = grad_x J_theta(x*, c) = 0`.
Differentiating that identity:

    H · dx*/dtheta + B = 0,      H = grad^2_xx J  (curvature at x*)
                                  B = grad^2_{x,theta} J  (how the gradient field tilts)
    =>  dx*/dtheta = -H^-1 B.

This is a **local, second-order** statement: it only asks about the curvature
`H` *at the single point* `x*`, and how the theta-gradient of the cost varies
locally around it (`B`). It is exact and cheap — one linear solve — but it is
only as good as the assumption that `x*` really is a stationary point that
moves *smoothly* with `theta`. If the solver's own stopping rule introduces a
discontinuity in where it lands (Section 7), this breaks even though the
formula itself is still "correct" for the point it was handed.

**(b) Truncated unrolling (`solve_unrolled`, Domke 2012).** Treat the *iterative
solver* itself as a computation graph and backprop through its last few steps.
This is a **path-dependent, first-order** statement: it doesn't need `H` at
all, but its cost scales with how many steps you keep in memory, and it only
*converges* to the implicit gradient as the unrolled tail grows — it is an
approximation to (a), not an independent ground truth.

**(c) Finite differences on the outer loss** (`ioc/outer.py`). Re-solve the
inner problem `K+1` times and difference. This needs no differentiable solver
at all, but it is the *only* one of the three that is sensitive to the
solver's actual finite-step behavior rather than to the idealized stationarity
condition — which is exactly why it's the arbiter when (a) and (b) disagree
(see Section 7's cos = -0.71 finding).

The geometric picture: (a) is "what does the *limiting* stationary manifold
look like," (b) is "what does the *optimization trajectory toward* that
manifold look like, truncated," (c) is "what does the *actual returned
output* of this concrete numerical procedure do." They agree only when the
solver's output is a faithful, smoothly-varying proxy for the true stationary
point — which is a real geometric assumption, not a formality (Section 7).

## 3. Curvature choices: which `H` you use is a choice of geometry

Two candidate Hessians:

- **True Hessian** `H = grad^2_xx J`: exact local curvature, including terms
  from `grad^2_xx r_k` (residual curvature). Needs the model to be twice
  differentiable — not available through some FFI kernels (GRiD's CUDA
  inverse dynamics only exposes one level of differentiation).
- **Gauss-Newton** `H_gn = 2 J^T J` (`J` = Jacobian of the whitened residual
  stack): drops the residual-curvature term, keeping only the
  first-derivative-squared part. **PSD by construction**, which is why it's
  also what the forward Gauss-Newton solver damps and steps with, and why
  CIOC's Laplace approximation (Section 5) needs it specifically — a
  log-det of an indefinite matrix isn't well-defined.

At a converged, small-residual solution the two nearly coincide (Gauss-Newton
is exact in the zero-residual limit); the gap is an empirical question, not
assumed away (measured ~14% magnitude bias in one setting — see `ioc/inner.py`
docstring). Geometrically: GN curvature is "the curvature you'd have if the
residuals were exactly zero," true curvature also accounts for how curved the
residual surfaces themselves are around a *not-quite-zero* residual.

`adjoint_cost_fn` decouples *which landscape produces `x*`* from *which
landscape's curvature the adjoint reads*: run the cheap/reduced-precision
forward solve, but build `H` from an expensive/exact model, evaluated once
per outer step rather than once per inner iteration. This is valid precisely
because (a) only needs `x*` to be *a* stationary point of *some* well-behaved
nearby landscape — the adjoint formula doesn't care which optimizer produced
the point being differentiated through, only that the point is genuinely
stationary.

## 4. Inverse KKT: theta as the null space of a feature-gradient Gram matrix

The cheapest possible fit (`ioc/analytic.py::kkt_fit`) skips solving the inner
problem altogether and asserts the demonstration is *already* the answer: if
`x~` is optimal under some `theta`, it must satisfy the KKT stationarity
condition

    grad_x J_theta(x~, c) = sum_k theta_k · grad_x phi_k(x~, c) = B(c) theta = 0,

where `B(c)` stacks the `K` feature gradients (each `grad_x phi_k`, a vector
in the same `x`-space) as columns. This turns IOC into a **linear-algebra
question about a matrix of gradients**, not an optimization-through-a-solver
question:

    theta_hat = argmin_theta  theta^T G theta,   G = (1/M) sum_i B(c_i)^T B(c_i),
    over the simplex.

`G` is a `K x K` Gram matrix of feature-gradient inner products, exactly
analogous to a Gram matrix in any other linear-algebra setting: `G_jk = <grad
phi_j, grad phi_k>` averaged over demos. The quadratic form `theta^T G theta`
is a sum of squared norms `||B(c) theta||^2`, so it is **PSD by
construction**. Minimizing it over the simplex asks: *which direction in
theta-space does the demonstration data leave the KKT residual smallest?*

**This is also the identifiability certificate**, and this is the load-bearing
geometric idea for the rest of the document: `theta` is only identifiable
along directions where `G` has appreciable curvature. Formally,

- `lambda_min(G) ~ 0` (relative to `trace(G)`) means there is a direction
  `v` in theta-space — a linear combination of features — along which
  `B(c) v ~ 0` for every demo in the set. Reweighting the cost along `v`
  leaves the KKT residual (and, to first order, the demonstrated behavior)
  unchanged. **No amount of data along the same demo distribution can resolve
  `v`** — it isn't a sample-size problem, it's a structural degeneracy of the
  feature Jacobian at the data you have. More demos help only if they change
  the *span* of `B(c)`, not just its count.
- This is the *same* Gram-matrix-of-feature-gradients construction used for
  `theta_ik` and `theta_trajopt` jointly in `iosp/identifiability_check.py`
  (Section 6) — `kkt_fit`'s docstring calls this out explicitly as the
  template the composed-model script reuses.

Geometrically: think of each demo as contributing one linear "no-signal"
constraint per near-zero row of `B(c)`, and `G`'s eigendecomposition as PCA
on the *directions the data can see*. The top eigenvectors are the
well-conditioned combinations of features the demos actually probe; the
bottom eigenvectors span the null cost directions — reweightings that are
invisible to this data, indistinguishable from noise.

Inverse KKT is exact and free at `sigma=0` (noiseless demos are exact
stationary points, so the residual truly vanishes along the true `theta`)
and degrades as demo noise grows, because a noisy demonstration is not the
stationary point of *any* cost — the method has no way to trade a small
stationarity violation for better behavioral match, unlike a rollout-based
loss. That failure mode is a direct consequence of using a *first-order
necessary condition* as the whole objective, rather than the actual value
function.

## 5. Continuous IOC: the Laplace approximation as a log-det regularizer

`cioc_fit` extends inverse KKT to a maximum-likelihood picture with an
explicit probabilistic model. Assume demonstrations are Boltzmann-distributed
around the optimum, `p(x) ∝ exp(-J_theta(x))`. The partition function is
intractable in general, so approximate `J` to second order around `x~` and
integrate the resulting Gaussian:

    log p(x~) ≈ -1/2 g^T H^-1 g + 1/2 log det H - (d/2) log 2π,
    g = grad_x J_theta(x~), H = grad^2_xx J_theta(x~) (Gauss-Newton, PSD).

Minimizing the negative log-likelihood over theta gives

    min_theta  1/2 g^T H^-1 g  -  1/2 log det H.

The first term is exactly inverse KKT's residual, but **Hessian-weighted**
(H^-1 rather than identity) — a stationarity residual measured in the metric
that curvature at `x~` actually induces, not in raw Euclidean feature-space.
The second term, `-1/2 log det H`, is what pure KKT is missing, and it plays
a specific geometric role: **it is the Gaussian normalizer**, and it rewards
`theta` that make the landscape around `x~` *sharply peaked* (large curvature,
large `det H`) — penalizing the flat, low-curvature landscapes that a pure
residual objective is indifferent to (a completely flat cost trivially makes
`g=0` everywhere, which minimizes the residual term for free but is not
informative about the demonstration). Structurally, this is the same
tension as a determinant/volume regularizer in any log-likelihood-under-a-
Gaussian setting — the residual term wants low curvature (to make error
"cheap"), the log-det term wants high curvature (to make the fit "sharp"),
and their sum has a genuine interior optimum.

Because `theta` here is *not* restricted to the simplex — the log-det term
is what pins its magnitude (unconstrained scale otherwise makes the residual
term trivially shrinkable by scaling `theta -> 0`, at least without the
compensating log-det growth) — the implementation carries an explicit
log-scale parameter alongside a softmax direction, rather than reusing the
simplex parameterization Inverse KKT can afford (empirically, forcing the
simplex here made the fit far worse: fixing scale by fiat while leaving
log-det free to distort direction).

Geometrically, CIOC sits strictly between KKT and the full rollout-based
inner-problem methods: like KKT it needs zero forward solves, but rather than
merely asking "is the gradient near zero," it's asking "how *peaked* is the
bowl around the demo" — a local second-order proxy for "how much of the
support of the true trajectory distribution does a rollout-based loss
actually integrate over."

## 6. Composing landscapes: the outer loss's geometry is a pullback

Everything above concerns one inner landscape. `iosp/` chains four such
landscapes (approach/grasp/transport/place) through a differentiable IK stage,
with the literal (non-`stop_gradient`) output of the IK solve feeding the next
segment's boundary condition. This makes the outer parameter-to-loss map a
**composition of four implicit functions and one algebraic (IK) map**:

    theta_ik --[canonical IK, custom_jvp]--> q_pick, q_place
             --[boundary condition]--> Scene for approach/transport
    theta_trajopt, Scene --[implicit adjoint, custom_vjp per segment]--> x*_phase
    x*_phase --[loss]--> L

Reverse-mode automatic differentiation composes these exactly as the chain
rule would: `dL/d(theta_ik)` is literally the product of the trajopt
segments' implicit-adjoint sensitivities to their own boundary condition and
the IK stage's own sensitivity to `theta_ik`. Geometrically, the outer loss
landscape over `(theta_ik, theta_trajopt)` is a **pullback** of the per-segment
landscapes through this chain — its curvature at any point is not simply "the
curvature of one bowl" but an accumulation of Jacobians across every stage the
gradient has to pass through. This is the thing a single hand-authored
monolithic trajopt cost (one flat decision vector, no separate IK subproblem)
structurally cannot represent: there is no analogue of "which redundant-IK
solution got selected" as a distinct differentiable stage in a system that
never solves IK as its own step.

Two geometric hazards specific to composition, both measured directly:

**(i) The redundant-IK null space.** The Panda is 7-DOF against a 6-DOF pose
task, so IK has a one-dimensional self-motion null space at each solution.
pyroffi's default implicit-diff rule assumes minimum-norm self-motion (a
pseudoinverse), but the actual solver's self-motion is arbitrary — measured
~80% gradient error vs. finite differences, a genuine null-space artifact, not
noise. The fix (canonical IK: reformulate as `q* = argmin ||q - q_ref||^2 s.t.
r(q,t)=0`) picks out the *specific* tangent direction in the self-motion null
space that the solver's own reference-tracking behavior selects, giving exact
KKT sensitivity instead of an assumed one. Geometrically: the naive adjoint
was differentiating through the wrong point in a positive-dimensional
solution manifold; canonicalization pins down which point (and hence which
tangent space) is actually meant.

**(ii) Discontinuity in the stopping rule breaks smoothness of `x*(theta)`.**
Section 2 flagged that implicit differentiation assumes `x*` moves smoothly
with `theta`. The composed chain's forward solver originally used an
early-stopping `while_loop` gated on a gradient-norm tolerance — a
**data-dependent trip count**. An infinitesimal shift in a smoothly-varying
boundary condition (`q_pick`/`q_place`, itself smooth in `theta_ik`) can flip
*which iteration* crosses the tolerance, landing the solver on a genuinely
different final iterate. That makes `x*(theta_ik)` **discontinuous**, not
merely "hard to differentiate" — violating the one precondition the implicit
adjoint needs. Measured consequence: `cos(implicit, FD) ≈ -0.71` on the full
composed chain, stable across three independent hard-branch-to-soft-branch
fixes (removing early stopping's `while_loop` trip-count dependence, replacing
the line search's hard argmax with a soft one, replacing the L-BFGS
curvature-pair admit/reject gate with a soft one), individually verified
non-degenerate and still producing the same disagreement stacked together.
This is presented as an open, honestly-reported problem, not a resolved one —
see `iosp/pickplace.py`'s module docstring for the elimination sequence and
`iosp/HANDOFF.md` for status. The chosen resolution is methodological, not a
fix to the disagreement itself: validate via ground-truth parameter recovery
and held-out generalization RMSE rather than FD agreement, since FD agreement
turns out to depend on a smoothness precondition the concrete numerical
solver may not satisfy even when the *idealized* stationary-point map would.

The takeaway for reading this as geometry: **a chain of implicit functions is
only as smooth as its least smooth link**, and "the adjoint formula is
locally correct" is a separate claim from "the map it's differentiating is
actually smooth at the scale you're evaluating it." The two can and did come
apart here.

## 7. Structural degeneracies found in the composed feature Gram matrix

`iosp/identifiability_check.py` extends Section 4's Gram-matrix construction
to the full composed model's 9-dimensional `theta = (theta_ik, theta_trajopt)`.
Two extensions to the construction, both forced by composition:

- For the 7 trajopt features, `grad_x phi_k` is an ordinary gradient of one
  phase's own residual sumsq w.r.t. that phase's own free decision vector,
  zero-padded into one shared 126-dimensional embedding space (concatenating
  all four phases' free-variable counts) so all columns of `B` are directly
  comparable, exactly as in Section 4.
- The 2 IK features (`theta_ik`, standoff offsets) have no `phi(x)` to
  differentiate — they are geometric inputs to the IK subproblem, not
  KKT-stationarity weights. The natural generalization is `dX*/d(theta_ik)_k`:
  how much the entire composed decision state shifts as the standoff moves,
  estimated by central finite differences directly on the forward solve
  (chosen deliberately over `jax.jacfwd`, unavailable here since
  `solve_implicit` is a `custom_vjp` with no attached JVP rule, and over
  reverse-mode, which would need 126 backward passes for a 2-dimensional
  input — FD needs only 2).

Measured findings (see docstring/header of `identifiability_check.py` for the
exact run and numbers):

- **Near-exact collinearity within one segment.** `cos(transport.smooth,
  transport.upright) = -0.9999`, sharply distinct from every other pairwise
  cosine (next largest magnitude ~-0.08). Geometrically, these two features'
  gradients point in nearly opposite directions along nearly the same line in
  the 126-dim embedding space — reweighting *between* them (moving along
  `transport.smooth - transport.upright`, say) leaves the demonstrated
  trajectory almost exactly unchanged. This is a **structural degeneracy of
  the residual formulation** — it held up regardless of scene geometry across
  the demos tested — not a data-curation artifact, and not a saturation
  artifact either: `transport.upright`'s own gradient norm (1.43) is *larger*
  than `transport.smooth`'s (0.64), ruling out the earlier hypothesis that
  `upright` was simply saturated and hence uninformative. The mechanism is
  that both features push the transport trajectory in nearly the same
  direction at this demo — the fit cannot tell how much of the shape is due
  to one weight versus the other, and no amount of more-of-the-same-motion
  data resolves it (Section 4's identifiability argument): the fix has to be
  a demo that geometrically decouples them (a forced detour that also tips
  the object, or an explicit orientation perturbation), not more data volume
  or an outer-loop tuning change.
- **Near-zero-gradient features, a distinct failure mode.** `approach.
  clearance`, `grasp.smooth`, `transport.clearance`, `place.smooth` all have
  near-zero gradient outright at this demo (the trajectory never approaches
  the obstacle margin, or the in-place grasp/place motions are too small to
  excite those weights). This is (b)-type — the demo simply never explores
  that direction — as opposed to (a)-type collinearity above. Both are
  "unidentifiable from this demo," but for different geometric reasons and
  with different fixes: (a) needs a decoupling demo, (b) needs *any* demo
  that actually excites that feature at all.
- **Spectrum is dominated by 1–2 directions.** Trace-normalized eigenvalues:
  `5.3e-23, 6.8e-16, 8.5e-12, 9.6e-12, 1.1e-8, 3.3e-6, 4.4e-4, 1.16e-2, 8.99` —
  only the top one or two eigenvectors (dominated by `grasp.standoff`, then
  `place.standoff`) carry real signal from a single demo; six of nine
  directions are near-completely flat. This matches, and gives a structural
  explanation for, the earlier observed weak effect of demo count on recovery
  error (`recovery_bench.py::sweep_demo_count`: 0.244 → 0.216 param error,
  N=1→8) — more demos of the same qualitative motion mostly add data along
  directions the Gram matrix already sees, not new directions.

## 8. The follow-on identifiability-clean procedure (Study 1) and its geometry

`iosp/study1_minimal_identifiable.py` (in-progress; see `HANDOFF.md` for
current status) takes the diagnosis in Section 7 seriously as a
recommendation for *how to fit*, not just how to diagnose. Given that only a
low-dimensional subspace of theta-space is identifiable at all, fitting the
full `theta` is fitting noise along the null directions. The procedure:

1. Build the same whitened, multi-demo-accumulated Gram matrix `G` as
   Sections 4/7, but restricted to a smaller, deliberately-decoupled K=3
   feature set within one phase (`clearance`, `smooth`, `line_dev` — the
   latter replacing `upright` after Section 7 found it structurally collinear
   with `smooth`).
2. **Rank selection**: keep only the eigendirections of `G` needed to reach
   95% of cumulative trace — a generic, threshold-based way of asking "how
   many directions does this data actually resolve," directly operationalizing
   Section 4's identifiability argument rather than eyeballing the spectrum.
   Confirmed reproducible: selects `k=2` with eigenvalues `[1e-4, 0.56-0.58,
   2.43-2.44]` — a clean gap between the bottom eigenvalue and the top two.
3. **Zero-prior convention**: components along the *unselected* (null)
   eigendirections are pinned at exactly zero rather than left free or
   leaked from ground truth — the honest statement that those directions
   carry no information from the data, matching Section 4's null-space
   argument precisely (there is no principled value to report there; zero is
   the only defensible one).
4. Fit only the top-`k` coefficients (in the whitened eigenbasis) via the
   implicit adjoint (Section 2a) with Adam.
5. Validate against held-out generalization RMSE, not fit-demo RMSE — Section
   9 explains why fit-demo error is a misleading proxy here.

**An open bug in this procedure, as of the last handoff, is itself a small
lesson in landscape geometry**: under the honest zero-prior convention, the
optimizer failed to move off its own initialization for both tested ground
truths — including one deliberately constructed to have real content in the
identifiable subspace. Diagnosis so far: `alpha=0` (the natural "null" choice
for the fitted coefficients too) maps to `theta_transport = [0,0,0]`, which
makes that segment's cost *identically* zero — the forward solve never leaves
its seed, and the implicit adjoint's own Hessian (Section 2a/3) is degenerate
exactly there. This is a **flat region of the outer loss surrounding the
origin**, wide enough that even a small (1e-3) nudge off zero didn't escape
it. The working theory is that this conflates two things that happen to share
a coordinate but shouldn't share a *value*: the null-eigendirection convention
(correctly zero, an identifiability statement) and the fitted-coefficient
optimizer's *starting point* (a purely numerical choice, which was also
initialized near zero and thereby landed in the same degenerate flat region
by coincidence, not by any principled connection between the two). The fix
under test is to initialize the optimizer away from that flat region — near
the trajopt weight magnitudes `calibrate_segment` was actually calibrated
against (~0.1–0.3) — while still holding the null directions exactly at zero,
plus a direct gradient-magnitude sanity check at a few candidate starting
points before trusting Adam's trajectory again.

## 9. Why fit-demo error is not the right loss geometry to read

`iosp/generalization_check.py`'s headline finding is a warning about which
loss surface you're actually looking at: RMSE evaluated on the *same* demo(s)
used for fitting was misleadingly small (~0.03) while RMSE on a *held-out*
scene was 17x worse (0.565) for the original 9-feature model. Geometrically,
fit-demo RMSE measures distance to a single point the fit was explicitly
optimized to be near — it says almost nothing about the shape of the loss
surface away from that point, which is exactly what Sections 4/7's
identifiability analysis is about: along near-null directions, many different
`theta` reproduce the fit demo almost equally well (that's what "near-null"
means), so fit-demo RMSE cannot distinguish them, but they generalize
differently to new contexts because the *behavior* implied by those
near-degenerate `theta` values genuinely differs once you move to a scene
where the near-null direction is no longer near-null. This is the practical
version of Section 4's abstract point: identifiability is about the
*curvature of the Gram matrix*, and a validation metric that only probes one
point near the fitted optimum cannot see that curvature — held-out
generalization is what actually samples a different region of the same
landscape.

## 9b. What to DO when r < K

Sections 4/7/9 diagnose rank deficiency; they don't prescribe a fix.
`iosp/THEORY_IDENTIFIABLE_REFIT.md` derives one, in two variants (known cost
library / unknown cost recovered in an RKHS), both reducing to: fit wide,
eigendecompose the cost-gradient Gram matrix `G = U Lambda U^T`, split
`theta = U_r U_r^T theta + (I - U_r U_r^T) theta`, and refit only the
identifiable component. Relevant exactly when `r = rank(G) < K` -- rare in the
2-D benchmarks (`r ~ K`), dominant in the 7-DOF composed model.

## 10. Summary map: concept -> geometric object

| Question | Geometric object | Where |
|---|---|---|
| How does `x*` move with `theta`? | Implicit-function-theorem sensitivity `-H^-1 B` at one stationary point | `ioc/inner.py` §2a |
| What curvature does the adjoint read? | True Hessian vs. PSD Gauss-Newton `J^T J` | `ioc/inner.py` §3 |
| Is `theta` recoverable at all from these demos? | Smallest eigenvalue(s) of the feature-gradient Gram matrix `G` | `ioc/analytic.py`, §4 |
| How sharply peaked is the landscape around the demo? | `log det H` term in the Laplace/CIOC objective | `ioc/analytic.py`, §5 |
| What does composing 4 solves + IK do to the outer landscape? | Pullback of per-segment sensitivities through the chain rule | `iosp/pickplace.py`, §6 |
| Which cost directions are entangled at this demo? | Off-diagonal structure / near-collinear eigenvectors of `G` | `iosp/identifiability_check.py`, §7 |
| Where does the optimizer stall? | A flat (near-degenerate-Hessian) region of the outer loss near the origin | `iosp/study1_minimal_identifiable.py`, §8 |
| Did the fit find the true minimum or just a nearby point along a flat direction? | Fit-demo RMSE vs. held-out RMSE — probing one point vs. sampling the surface | `iosp/generalization_check.py`, §9 |
| What do you do once you know `r < K`? | Projector `P_r = U_r U_r^T` onto the identifiable subspace; refit in those coordinates | `THEORY_IDENTIFIABLE_REFIT.md`, §9b |

Cross-cutting theme: almost every empirical finding in `iosp/` reduces to
asking "is this map actually smooth/well-conditioned at the point I'm
evaluating it," and the answer is "yes for the idealized stationary-point
map, not always for the concrete finite-step solver realizing it" (§6),
"yes along some directions of theta-space, not along others" (§4, §7), and
"only locally, not globally, so don't trust a metric that only samples
locally" (§9).
