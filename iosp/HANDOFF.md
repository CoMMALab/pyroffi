# IOSP handoff — 2026-08-24

## Update 2026-08-25: segment-freeze ablation + eigen-projected recovery

New investigation, parallel to Study 1 (below), asking a different question:
Study 1 diagnoses the minimal K=3 transport-only model; this asks WHERE in
the full 4-segment composed chain (`iosp/pickplace.py`) recovery breaks,
starting from the fact (verified) that `ioc.robot.e1_identifiability`'s
single-segment point-to-point IOC already works cleanly (theta_cos 0.97-1.00
across sigma=0-0.05). Two new scripts, both stock components only (no new
solver code):

- **`iosp/study0_segment_ablation.py`** — freezes all-but-a-growing-subset of
  `pickplace.py`'s 4 segments at their TRUE ground-truth weights and fits
  only the free subset, via a mask over a FIXED-SHAPE 9-dim `z` (not a
  variable-length vector) so the whole schedule shares ONE ~22min compile
  instead of one per rung (see module docstring for the trick). Also has
  `anchor_obstacle_to_transport`/`anchor_obstacles_batch` +
  `main_multidemo()` for the multi-demo coverage check.
- **`iosp/study0d_eigen_projected_recovery.py`** — rebuilds
  `identifiability_check.py`'s feature-gradient Gram matrix on a given scene,
  selects the top-k eigendirections by the same 95%-cumulative-trace rule
  Study 1 uses, and reports recovery error PROJECTED onto the identifiable
  vs. null subspace separately, instead of one raw L2 norm that conflates
  them.

**Finding 1 (ruled out a hypothesis): composition itself is not the problem.**
Freezing 3 of 4 segments at ground truth and fitting only `transport` gives
`free_param_err ~ 0.21-0.24`, essentially the SAME magnitude as fitting all 4
segments + `theta_ik` jointly (~0.21-0.24). Error does not grow as more
segments are unfrozen. Whatever is wrong is present at N=1 free segment,
inside `pickplace.py`'s existing per-segment machinery -- not an artifact of
chaining segments together.

**Finding 2: the demo scene, not the adjoint, is the problem — and it's the
SAME degeneracy `identifiability_check.py` already certified.**
`recovery_bench.py`'s fixed `OBS_CENTER=[0.3,0,0.4]` is not anchored to any
segment's actual path — measured closest approach of the straight
`transport` EE line to it is 0.091m, past `CLEARANCE_MARGIN=0.05m`, so
`transport.clearance`'s soft-hinge residual sits inert (near-zero gradient)
along the whole segment. This is bad ee-space-loss-vs-param-error signature
(ee_rmse ~0.001-0.007, tiny, while free_param_err ~0.2, large) is the exact
signature of a flat/null identifiability direction, not a broken gradient.

**Finding 3 (anchoring the obstacle to the SEED path did NOT fix it — a real,
useful negative result, not a papered-over one).** Anchored the obstacle to
sit ~0.02m inside the margin relative to the `transport` segment's
straight-line SEED path (mirroring `ioc.robot.problem.RobotProblem.
sample_scenes`'s construction). `free_param_err` did not improve (0.21→0.24,
if anything slightly worse). Root cause, found via
`study0d_eigen_projected_recovery.py`: the Gram-matrix gradient that
determines identifiability is evaluated at `x_star`, the CONVERGED solution
under `theta_star` (whose `transport.clearance` weight is 0.28, not small) —
not at the seed. A working avoidance term does its job: it pushes the solved
trajectory away from the obstacle until the constraint is satisfied with
margin, and once satisfied, the residual's local gradient at that stationary
point goes back to ~0 ("solved and inactive", not "never engaged").
Anchoring near the SEED guarantees the problem STARTS engaged; it does not
guarantee it STAYS engaded post-solve. Measured: `transport.clearance`
gradient norm is still ~0.0002 after anchoring (`identifiability_check.py`-
style Gram construction, `study0d`'s `compute_gram`). **Next step, not yet
tried**: anchor tightly enough that avoidance can't fully clear the margin
(smaller/negative offset), or anchor iteratively against the SOLVED
trajectory rather than the seed.

**Finding 4: eigen-projected error confirms the optimizer is behaving
correctly, not failing.** On the anchored scene, the Gram-matrix spectrum is
EVEN MORE degenerate than the original (pre-anchoring) certificate: trace-
normalized eigenvalues `[8.988, 0.0112, 4.3e-4, 1.3e-5, 0, 0, 0, 0, 0]` —
literally 5 of 9 directions exactly zero (below float precision), only k=1
clears the 95% threshold, and that one direction is utterly dominated by
`grasp.standoff` (gradient norm 225.8, next largest `place.standoff` at 8.0
— i.e. essentially ALL identifiable signal in this one demo comes from
`theta_ik`, NONE of the 7 trajopt weights are meaningfully identifiable from
this demo). Projecting `theta_hat - theta_star` onto that 1-dim identifiable
subspace vs. the 8-dim null subspace: `top-1 err = 0.0202` (small, recovered
correctly), `null (8-dim) err = 0.2422` (~= the raw 0.2431). **The optimizer
is doing exactly what an unbiased fit should: nailing the one direction the
data constrains and drifting freely, with zero gradient, along the eight it
structurally cannot see.** The raw `free_param_err` metric used throughout
`study0_segment_ablation.py`/`recovery_bench.py` cannot distinguish this from
genuine optimizer failure — always read eigen-projected error, not raw L2,
when judging a composed-chain recovery result.

**Finding 5: multi-demo coverage gives only a mild, noisy improvement (not a
fix) — consistent with Finding 4, not with the composition hypothesis.**
`study0_segment_ablation.main_multidemo()`, N in {1,3,5} independently-
sampled contexts, EACH with its own individually-anchored obstacle
(`anchor_obstacles_batch` — sharing one obstacle across jittered contexts
would silently reintroduce the Finding-2 bug for every context but one):
`free_param_err` transport-only 0.227→0.234→0.215 (N=1,3,5), all-free
0.215→0.222→0.186. Small and non-monotonic (N=3 worse than N=1 in both
rows), matching the pre-existing `sweep_demo_count` trend (0.244→0.216,
N=1→8) and Section 4/7 of THEORY.md's prediction: more demos help only if
they change the SPAN of the Gram matrix, and `transport.smooth`/`transport.
upright`'s collinearity was measured to hold "regardless of scene geometry"
-- so coverage of obstacle placement doesn't touch that degeneracy. The mild
gain is plausibly just occasional lucky scenes where an anchored obstacle
happens to still bind post-solve (see Finding 3), not a structural fix.

**Bottom line for whoever picks this up next**: stop trying to fix this by
tuning `theta_star`/recovery-quality numbers directly. The tools now exist
(`study0_segment_ablation.py`, `study0d_eigen_projected_recovery.py`) to (a)
isolate which segment/direction is broken and (b) separate "genuinely
unidentifiable, optimizer is fine" from "actually broken" via eigen-
projection -- USE THEM before spending time on a raw `free_param_err`/
`param_err` number again, on this or any future iosp/ recovery script. The
concrete open thread is Finding 3's iterative/tighter anchoring fix, and
whether it can turn `transport.clearance` into a genuinely second
identifiable direction (currently k=1 of 9).

---

Status snapshot for picking this back up. Written mid-investigation because
the coordinating session's context is being wound down, not because the work
is finished. A background subagent (task-id `abe82e171fc04674d` in the prior
session's tooling) has been doing the implementation; it currently has a
`SendMessage`-resumable in-flight task blocked on repeated session/rate
limits, not on a design question. If your tooling exposes that same agent id,
resuming it directly carries full context; otherwise use this doc plus the
files below.

## What IOSP is

`iosp/` = Inverse Sequential Motion Planning. Composed differentiable
pick-and-place: an IK stage (`sqp_ik_solve_cuda` with canonical IK,
`theta_ik`-parameterized standoff offsets) feeds, via real (non-stop-gradient)
data flow, four chained `ioc.inner.make_inner_solver`-based trajopt segments
(approach/grasp/transport/place). The research claim: invert a human
teleop demo through this whole composed differentiable planner to recover
cost parameters — the novelty argument (vs. e.g. curobo) is that curobo's
planner isn't differentiable end-to-end, only its inner optimizer is, so
"which redundant-IK solution got selected" can't backprop through curobo's
planning the way it can here.

Lives entirely in `iosp/`, separate from `ioc/` (the single-segment IOC
study) per explicit instruction. Reuses `ioc/inner.py`, `ioc/outer.py`,
`ioc/robot/problem.py` unmodified.

## File map

- `iosp/pickplace.py` — the composed model itself (IK stage + 4 trajopt
  segments). `make_composed_forward_solver` stacks three now-opt-in
  `DynamicsTrajOptConfig` flags added to
  `src/pyroffi/optimization_engines/_dynamics_trajopt.py` this investigation:
  `early_stop=False`, `soft_line_search=True`, `soft_curvature_gate=True`.
  All strictly opt-in — every other caller (`ioc/robot/e1_identifiability.py`,
  `ioc/bench2d/`) is unaffected by default.
- `iosp/recovery_bench.py` — ground-truth recovery benchmark: pin a
  `theta_star`, roll out a synthetic demo, recover `theta_hat` via implicit
  adjoint or CMA-ES, compare. Also has the demo-count sweep
  (`sweep_demo_count`, N∈{1,2,4,8}).
- `iosp/generalization_check.py` — held-out-scene validation. **Key finding**:
  fit-demo RMSE (~0.03) is a misleading proxy for true cost recovery; RMSE on
  a held-out scene was 17x worse (0.565) for the original 9-feature model.
  Docstring has been corrected once already (an earlier "saturation" caveat
  on `transport.upright` was wrong — see identifiability_check.py).
- `iosp/identifiability_check.py` — Gram-matrix-of-feature-gradients
  identifiability certificate (mirrors `ioc/analytic.py::kkt_fit`'s
  construction). **Key finding**: `transport.smooth` and `transport.upright`
  are near-exactly collinear (cos=-0.9999) at the demo, essentially
  regardless of scene geometry — a structural degeneracy in the residual
  formulation, not a data-curation problem. 6/9 of the original feature
  directions are near-unidentifiable from one demo (some collinear, some
  just never excited by the demo).
- `iosp/study1_minimal_identifiable.py` — **in progress, not finished**. Goal:
  a minimal (K=3, all within the `transport` phase: `clearance`, `smooth`,
  `line_dev` — `line_dev` replaces `upright`, which was dropped after being
  found structurally collinear with `smooth` across 3 differently-designed
  demos) identifiability-clean setup, with a formal canonical inference
  procedure:
  1. whitened (calibrated-scale), multi-demo-accumulated Gram matrix
  2. generic rank-selection rule (95%-cumulative-trace threshold — this part
     is believed done and stable, confirmed reproducible across reruns:
     `k=2` selected, eigenvalues ≈ `[1e-4, 0.56-0.58, 2.43-2.44]`)
  3. **zero-prior convention**: unselected (null) eigendirections pinned at
     exactly zero (not the earlier ground-truth-leakage version, which was
     an explicitly-flagged scaffold, not the honest version)
  4. fit only the top-k alpha via implicit-adjoint Adam
  5. validate via held-out generalization RMSE vs. a "no fitting at all"
     baseline, on scenes never used for fitting

  **Open bug, actively being debugged, NOT YET RESOLVED**: under the honest
  zero-prior convention, the optimizer fails to move off its own starting
  point (`alpha_hat` ends up pinned near `alpha0`) for BOTH tested
  `theta_star`s, including one deliberately constructed to have real content
  in the identifiable subspace (i.e. this is not just "this particular
  theta_star has no identifiable content," which was itself a separate,
  legitimate, already-documented null result for the ORIGINAL theta_star).
  Diagnosed as far as: `alpha=0` maps to `theta_transport=[0,0,0]`, which
  makes `make_inner_solver`'s transport-segment cost identically zero — the
  forward solve never leaves its seed and the implicit adjoint's Hessian is
  degenerate there. Nudging `alpha0` slightly off zero (1e-3) did NOT fix it
  — the degenerate/flat region is wider than the exact origin.

  **Working theory for the fix** (sent to the subagent, NOT yet confirmed
  executed/verified as of this doc): the bug conflates two things that
  should be separate —
  - null components should stay pinned at exactly zero (that's correct,
    keep it — it's the honest identifiability convention)
  - but the FITTED alpha's optimizer *starting point* was ALSO initialized
    near zero, which lands it in/near the degenerate flat region above. That
    initialization choice is a separate, purely numerical decision and
    should NOT be zero — it should be initialized so `theta_transport`'s
    magnitude lands near what `calibrate_segment` was actually calibrated
    against (~0.1-0.3), not ~0.001.
  - Also asked for a direct sanity check (`jax.grad` magnitude at a few
    candidate `alpha0` values, e.g. 0.001/0.05/0.2) BEFORE trusting Adam
    again, to confirm the gradient is genuinely nonzero and reasonably
    scaled at the new starting point.

  **Last known state**: subagent had reconfirmed the certificate (stable,
  matches prior runs) and was about to run the gradient-probe diagnostic +
  fit when it was cut off by a session limit (resets ~2026-08-24 00:00
  America/Indiana/Indianapolis, i.e. should already be clear by the time
  this is read). Resuming should pick straight back up on: run the
  gradient-probe check, then rerun canonical recovery on both theta_stars,
  then confirm Study 1 is genuinely working end-to-end before touching
  Study 2 at all.

- `iosp/study2_demo_quality_ablation.py` — **written, syntax-checked, NOT
  YET RUN**. Correctly gated on Study 1's canonical fit actually working
  first — do not run this until the bug above is confirmed fixed, or its
  output will be meaningless (all three demo regimes would show the same
  broken non-movement for a reason unrelated to demo quality). Design:
  compare (a) single demo, (b) N randomly-jittered demos, (c) N
  deliberately-curated decoupling demos, at matched N, all using Study 1's
  canonical procedure (imports it, does not duplicate). Claim being tested:
  does demo curation quality matter more than demo count, at matched N?

## Known, separately-documented, still-open (lower priority) items

- **FD-vs-implicit-adjoint gradient disagreement on the FULL composed
  chain** (not the minimal Study 1 model): `cos(implicit, FD) ≈ -0.71`,
  stable across three independent hard-branch-to-soft fixes in the trajopt
  solver (early-stop, line-search argmax, L-BFGS curvature gate) — all
  three verified individually non-degenerate (jaxpr-diffed, not just
  flag-set) and stacked together, with literally zero effect on the number.
  Float64 test of the "float32 FD noise" hypothesis is BLOCKED: canonical
  IK's `custom_jvp` is float32-only regardless of caller-side casting — a
  real, structural dtype-boundary constraint in
  `src/pyroffi/optimization_engines/_canonical_ik.py`, not something fixed
  from `iosp`. Current resolution: validate via ground-truth recovery +
  held-out generalization instead of FD agreement (this is now the
  established methodology throughout `iosp/`) — implicit reached comparable
  reconstruction quality to CMA-ES at 10x fewer forward solves on the full
  composed chain (steady-state, compile-excluded timing: implicit
  1.96s/solve vs CMA-ES 0.34s/solve — net wall-clock advantage only ~1.75x,
  NOT 10x; report "10x fewer solves," not "10x faster," if this goes in the
  paper).
- **Demo-count sweep on the ORIGINAL 9-feature model** (`recovery_bench.py`)
  showed only a weak trend (param_err 0.244→0.216, N=1→8) and a probably-
  artifactual RMSE spike at N=8 (likely a fixed-8-step-budget-didn't-converge
  issue on the harder joint problem, not a real regression with more data —
  flagged, unverified). Superseded in relevance by the identifiability
  finding — weak trend is now understood to be partly explained by
  structural collinearity, not just data volume.
- `iosp/examples/21_01_panda_pickplace_ioc_teleop_viser.py` — updated to the
  CURRENT composed API (`problem.seeds`, `problem.solve`, per-phase
  `inner_by_phase`, `theta=(theta_ik, z_trajopt)`), verified it launches and
  populates. "Run IOC" button not exercised end-to-end (compiles for ~20+
  min at this point in the composed model's complexity — expected, not a
  bug, but worth knowing before clicking it live).

## Recommended next steps, in order

1. Resume/rerun the gradient-probe diagnostic + fix for Study 1's canonical
   procedure (see "working theory for the fix" above). This is the actual
   blocker — everything else is gated on it.
2. Once Study 1's canonical fit is confirmed moving correctly and producing
   real (non-degenerate) numbers on both theta_stars, run Study 2 as
   already written.
3. Report Study 1 + Study 2 final numbers together — that combined report
   is the actual deliverable the user is waiting on.
4. Lower priority, not blocking: reconcile the "10x fewer solves, only
   1.75x faster wall-clock" nuance into whatever writeup cites the speed
   comparison, so the paper doesn't overclaim.

## Working conventions established this investigation (apply going forward)

- Every fix to shared solver code (`_dynamics_trajopt.py`,
  `_canonical_ik.py`) must be strictly opt-in — zero behavior change for
  existing callers (`e1_identifiability.py`, `bench2d`, etc.) unless
  explicitly told otherwise.
- Never present a numeric result without checking it's not vacuous/
  degenerate first (this investigation hit that failure mode twice —
  `alpha_hat` exactly equal to its own starting point both times — and both
  times the honest thing was to say so plainly rather than round up).
- FD-gradient-agreement is NOT the validation standard for this composed
  model — ground-truth recovery + held-out generalization RMSE vs. a
  no-fit baseline is. This was an explicit, deliberate methodology decision,
  not a fallback to avoid — keep using it.
- GPU selection: check `nvidia-smi` before launching, pick a free device via
  `CUDA_VISIBLE_DEVICES`, this repo's boxes are shared.
