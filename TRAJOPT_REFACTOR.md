# Unified 3-tier dynamics-aware trajopt

## Context

`src/pyroffi/optimization_engines/` currently has four independent trajopt
solvers that grew organically and now duplicate ~80% of their machinery:

- `_dynamics_trajopt.py` (`dynamics_trajopt`) — generic, caller-supplied
  `cost_fn`, unconstrained L-BFGS or box-projected GD. No dynamics, no
  collision, no AL. This is the **only** engine `ioc/` and `iosp/`
  (SPaSM / inverse-SPaSM) actually call — `sco_trajopt`,
  `flat_contact_trajopt`, `contact_rich_trajopt` have zero callers outside
  `tests/`.
- `_sco_optimization.py` (`sco_trajopt`) — SCO: linearized-collision penalty
  continuation (no duals), kinematic-only, no dynamics/GRiD.
- `_flat_contact_trajopt.py` (`flat_contact_trajopt`) — grasp/dynamics
  satisfied *by construction* (differential flatness + analytic force
  allocation), penalty-continuation on grasp tracking, no AL, hardcoded to
  one manipulator/object contact problem.
- `_contact_rich_trajopt.py` (`contact_rich_trajopt`) — true AL: dual
  variables `mu`/`nu` on grasp-closure and object Newton-Euler residuals,
  `rho` penalty continuation, GRiD-based torque cost — but per-manipulator
  **torque-limit feasibility is still a fixed-weight hinge penalty, not an
  AL term**, unlike the other two constraints in the same file.

All three L-BFGS inner solvers (`_sco_optimization._lbfgs_inner_solve`,
`_contact_rich_trajopt._inner_solve`, `_flat_contact_trajopt._inner_solve`)
are structurally identical copies (ring-buffer history, Nocedal two-loop,
5-point line search, endpoint mask, `while_loop`) differing only in the cost
closure. There is no generic, pluggable constraint interface anywhere in the
trajopt package — every constraint (collision, grasp closure, Newton-Euler,
torque limits) is hand-baked into its engine's private cost function. The
only generic constraint interface in the whole `optimization_engines`
package (`project_onto_constraints` in `_nullspace.py`) is a post-hoc IK
null-space projection, unrelated to trajopt.

The goal: collapse this into three additive tiers sharing one core (shared
L-BFGS driver + a generic AL outer loop that takes an arbitrary list of
constraint-residual functions), so that SPaSM/inverse-SPaSM (`ioc`/`iosp`)
get a much stronger tier-1 solver (SCO instead of plain L-BFGS) with
first-class dynamics feasibility and arbitrary-constraint support, while
`flat_contact_trajopt` and `contact_rich_trajopt` become tier 2 and tier 3
built on the same primitives instead of independent copies.

**Backward compatibility is the hard constraint on this refactor.** `ioc.inner`
(`solve`, `solve_implicit`, `solve_unrolled`) depends on `dynamics_trajopt`'s
exact `(x0, cost_fn, opt_cfg) -> x` signature and on `early_stop`/
`unroll_tail`/`soft_line_search`/`soft_curvature_gate` semantics documented
in `DynamicsTrajOptConfig` — these are load-bearing for the implicit-adjoint
IOC pipeline (`ioc/inner.py`) and for differentiating through upstream IK
boundary conditions (`iosp/model/pickplace.py:240-310`'s measured
`cos(implicit,FD)` regression). Every one of the ~13 call sites in
`ioc/robot/*.py`, `ioc/probes/*.py`, `ioc/diagnostics.py`,
`ioc/bench2d/run.py`, `iosp/experiments/*.py`, `iosp/model/{pickplace,tetris,
tower}.py` must keep working unmodified when the new SCO/AL features are
left at their default (off) settings.

## Tier design

### Shared core: `_trajopt_core.py` (new file)

Extract and generalize what's duplicated three times today:

1. **`_lbfgs_driver`** — the canonical L-BFGS step (ring-buffer history,
   `_lbfgs_two_loop`, 5-point line search from `_LS_ALPHAS`), generalizing
   `_dynamics_trajopt.dynamics_trajopt`'s existing feature set (it's the most
   complete of the four copies today): `early_stop` while_loop vs. fixed-length
   differentiable `lax.scan` with Domke-2012 `unroll_tail`, `soft_line_search`,
   `soft_curvature_gate`, `grad_tol`. `_sco_optimization._lbfgs_inner_solve`,
   `_contact_rich_trajopt._inner_solve`, `_flat_contact_trajopt._inner_solve`
   all become thin callers of this one driver with just their cost closure
   swapped in.
2. **`_projected_gd`** — moved as-is from `_dynamics_trajopt.py`
   (box-clip-after-step GD), reusable as the joint-limit-safe fallback for
   any tier, per the user's request to use it "if it helps satisfy
   correctness."
3. **Generic AL outer-loop driver**, `_al_outer_loop(z0, inner_solve_fn,
   constraints, opt_cfg)`: `constraints` is a tuple of
   `AugmentedLagrangianTerm(residual_fn, kind, rho0, rho_max, penalty_scale)`
   entries (`kind` = `"eq"` or `"ineq"`; inequality residuals get the
   `max(0, ...)` / projected dual-ascent treatment `contact_rich_trajopt`
   already applies implicitly via its hinge). Each term keeps its own dual
   variable and `rho`; the outer loop is a `while_loop` mirroring
   `_contact_rich_trajopt.py:370-432`'s existing `outer_body`/`outer_cond`
   exactly, generalized from the two hardcoded terms (`g`, `mu`, `rho_g`) and
   (`b`, `nu`, `rho_o`) to an arbitrary tuple. This is the "arbitrary
   constraints via AL" mechanism the user asked for — SCO's outer loop
   (currently pure penalty-continuation, no duals) becomes a call into this
   same driver with an empty-or-populated constraint tuple, i.e. SCO gains
   real AL for free instead of staying continuation-only.
4. **`dynamics_feasibility_residual`** (new, in
   `src/pyroffi/dynamics/_contact.py`, mirroring `object_dynamics_residual`'s
   shape/naming per the GRiD-integration research): wraps
   `grid.inverse_dynamics(q, qd, qdd, f_ext)` (already vmap-fused,
   differentiable via GRiD's analytic `custom_jvp`, no float64 twin needed)
   into `relu(|tau| - tau_max)`, exposed as a pluggable `AugmentedLagrangianTerm`
   so tier 3's currently-hardcoded torque hinge (`_contact_rich_trajopt.py:174-175`)
   and tier 1's optional dynamics-feasibility default both use one
   implementation. This directly fixes the asymmetry the GRiD research
   surfaced: grasp-closure/object-dynamics are already AL terms with duals,
   torque limits were not.

### Tier 1 — `dynamics_trajopt` (unconstrained → SCO+AL)

Signature and default behavior for existing callers stay **byte-identical**:
`dynamics_trajopt(x0, cost_fn, opt_cfg=DynamicsTrajOptConfig())` with today's
defaults (`method="lbfgs"`, no SCO, no AL) must produce the same trajectory
as before — verified by the parity tests below.

New, opt-in `DynamicsTrajOptConfig` fields, all defaulting to preserve exact
current behavior:
- `use_sco: bool = False` — when True, wraps the inner L-BFGS solve in an
  SCO-style outer loop (linearize any supplied inequality residuals, e.g.
  collision, at the current iterate via `jax.jacfwd`, à la
  `_sco_optimization.py:466-487`) instead of a single flat minimize. This is
  the "port SCO over" the user asked for, generalized beyond
  collision-only.
- `constraints: tuple = ()` — arbitrary `AugmentedLagrangianTerm`s (from
  `_trajopt_core`) folded into the outer AL loop. Empty tuple ⇒ current
  unconstrained behavior exactly.
- `robot: Robot | None = None`, `grid: GRiDDynamics | None = None` — when
  supplied, tier 1 auto-adds (a) the existing `_smoothness_cost`/
  `_limits_cost` kinematic terms and (b) a `dynamics_feasibility_residual`
  AL term, satisfying the user's "by default should satisfy kinematic and
  dynamic cost functions" requirement for standalone use — **without**
  changing behavior for every existing `ioc`/`iosp` call site, none of which
  pass `robot`/`grid` (they supply their own fully-custom `cost_fn` that
  already encodes the IOC objective).
- `method="projected_gd"` path is untouched (still available as the
  joint-limit-safe fallback), and can now also be composed as the descent
  step inside the AL inner solve when `constraints` is non-empty and box
  limits are supplied, per the user's "use projected GD's clipping to
  prevent leaving joint limits if it helps correctness."

### Tier 2 — `flat_contact_trajopt` (flatness + arbitrary AL constraints)

Structure unchanged (differential flatness for the object twist, analytic
force allocation via `allocate_forces_at` so grasp/dynamics stay satisfied
*by construction*, zero duals needed for those two). Add:
- `constraints: tuple = ()` on `FlatContactTrajOptConfig`, threaded through
  `_flat_contact_jax`'s stage loop the same way tier 1 does, using the same
  `_al_outer_loop` from the shared core in place of the current
  penalty-continuation-only `w_track` stage loop for any *extra* constraint
  the caller supplies (task-specific waypoint constraints, extra collision
  bodies, etc.) — while grasp tracking itself can stay a plain penalty
  (it's not measuring true infeasibility, `allocate_forces_at` already makes
  the hard constraints exact).
- Reuses tier 1's shared `_lbfgs_driver` instead of its own copy — no
  behavior change, verified against `tests/test_flat_contact_trajopt.py`.

### Tier 3 — `contact_rich_trajopt`: C3-inspired consensus reformulation

The repo has no existing complementarity code to preserve compatibility
with (confirmed: no Posa/C3 references anywhere), so this tier is a genuine
reformulation, not just a rewire onto the shared core — taking the
structure of C3 / C3+ (Aydinoglu et al.; Bui et al. 2510.19974, "Push
Anything") as inspiration for how contact forces are found, while keeping
the AL machinery for grasp-rigidity and dynamics that already works today.

**Today's limitation**: `_contact_rich_trajopt.py` pre-specifies which
bodies are in contact (the grasp geometry, `capture_grasp_offsets`) and
solves for forces consistent with that *fixed* contact set via AL. Its own
docstring (`:15`) calls discovering the contact set itself "deliberately
out of scope because it becomes intractable fast" — because the natural way
to do that is mode enumeration / MIQP, which doesn't scale or vmap.

**C3's actual mechanism** (this is what to borrow, not just gesture at):
per outer iteration, C3 does **not** enumerate contact modes. It alternates,
via consensus ADMM, between two subproblems tied by a shared consensus
variable and dual:
1. A "projection" step: given the current trajectory/force estimate, solve
   one small QP *per contact* enforcing the complementarity conditions
   (normal force `f_n >= 0`, gap `phi >= 0`, complementary slackness
   `f_n * phi = 0` relaxed to a QP via linearizing `phi` at the current
   contact point, plus a friction-cone QP or a Coulomb-linearized polytope
   for `f_t`) — this replaces "decide whether contact i is active" with "the
   QP's own inequality-constrained solution decides it," so no discrete mode
   variable ever appears.
2. A "dynamics" step: given the projected forces (treated as fixed
   `f_ext`), solve the smooth trajectory subproblem exactly as today's
   inner solve does (GRiD `inverse_dynamics` residual, smoothness, limits) —
   this is already the tier-3 inner solve, unchanged.
3. Consensus + dual update: the two force estimates (from the projection
   QP and from what the dynamics step's residual implies) are driven
   together by an ADMM dual, structurally identical to the
   `mu += dual_scale*rho*g` dual ascent already in
   `_contact_rich_trajopt.py:384-390` — C3's "consensus" step *is* an AL
   dual ascent, just on a force-consistency residual instead of a
   grasp-closure residual.

**Design for this codebase:**
- Add a new residual/term type, `_complementarity_projection(contact_pts,
  normals, f_raw) -> f_proj`, that per contact per timestep solves the small
  QP in (1) above. Because it's one independent small QP per
  `(batch, timestep, contact)` triple, it vmaps exactly like
  `grasp_closure_residual`/`object_dynamics_residual` already do
  (`jax.vmap(..., in_axes=(None, 0, ...))` over `T`, and the existing outer
  `vmap` over batch) — no Python loop, one fused launch pattern, consistent
  with how GRiD's own FFI calls are batched (`custom_batching.custom_vmap`
  folding the vmap axis into one launch). Use a fixed small number of
  projected-gradient/PGS iterations (not a generic QP solver) so the whole
  projection is itself an unrolled, differentiable `lax.scan` — matching
  the "reconciling with the parallelization scheme" concern you raised,
  since a black-box QP solver would not vmap or differentiate cleanly.
- Fold this into `_al_outer_loop` as one more consensus/dual pair
  (`f_consensus`, `dual_f`, `rho_f`) alongside the existing grasp (`mu`) and
  object-dynamics (`nu`) terms — same `while_loop` shape as
  `_contact_rich_trajopt.py:370-432`, generalized. Grasp-closure stays a
  *hard equality* AL term (rigid grasp, no complementarity needed there —
  C3's machinery is for *unilateral* environment/object contacts, not the
  gripper's own rigid grip); the new consensus term is what lets tier 3
  handle **environment contact** (the thing marked out-of-scope today)
  without mode enumeration.
- **Promote torque-limit feasibility from hinge penalty to AL term**: add
  `dynamics_feasibility_residual` (per-manipulator, from the shared core) as
  a further `constraints` entry with its own dual and `rho_tau`, replacing
  `_contact_rich_trajopt.py:174-175`'s fixed-weight `w_torque_limit` hinge —
  same reasoning as before, now folded into the same generalized AL/consensus
  loop as the complementarity term.
- `constraints: tuple = ()` still exposed for arbitrary caller-supplied
  terms on top of the built-in grasp/dynamics/torque/complementarity ones.
- **Open-loop vs. fixed-loop iteration**: generalize the existing
  `constraint_tol`-gated early exit (`_contact_rich_trajopt.py:393-399`)
  into `_al_outer_loop` itself (`n_outer_iters` cap = fixed/"open-loop"
  budget; `constraint_tol>0` = "closed-loop"/converge-to-tolerance mode),
  used uniformly by all three tiers and by the new consensus term's dual.

This is scoped as a genuine reformulation of the *environment-contact*
path, while leaving today's grasp-rigidity/object-dynamics AL terms and
their passing tests intact — the new complementarity machinery is additive
(a new constraint kind in the generic AL/consensus loop), so
`tests/test_contact_rich_trajopt.py`'s existing single/bimanual grasp cases
should still hit their current tolerances since the only new applicable
contacts are ones supplied to `constraints`, not the grasp itself.

## Files touched

- **New**: `src/pyroffi/optimization_engines/_trajopt_core.py` (shared
  L-BFGS driver, `_projected_gd`, `AugmentedLagrangianTerm`, `_al_outer_loop`).
- **New**: `dynamics_feasibility_residual` in `src/pyroffi/dynamics/_contact.py`.
- **Modified**: `_dynamics_trajopt.py`, `_sco_optimization.py`,
  `_flat_contact_trajopt.py`, `_contact_rich_trajopt.py` — rewired onto
  `_trajopt_core`, new opt-in config fields as above, no signature breaks.
- **Unmodified (legacy, kept for its own test)**: `_contact_trajopt.py` (the
  older superseded AL solver `contact_sco_trajopt`) — out of scope, only
  touched if `_fd_vel_acc` needs to move into `_trajopt_core` (it's imported
  from there by both `_flat_contact_trajopt.py` and
  `_contact_rich_trajopt.py` today; moving it is a pure relocation).
- **`__init__.py`**: export list unchanged in names; add nothing that
  shadows existing exports.

## Verification

1. **Regression parity (must not change)**: run and diff outputs of
   `ioc/robot/{e1_identifiability,e2_scaling,e3_dynamics}.py`,
   `ioc/bench2d/run.py`, `iosp/experiments/{e5_tamp2d,e10_method_comparison}.py`,
   `iosp/model/{pickplace,tetris,tower}.py` at their existing
   `DynamicsTrajOptConfig` defaults (`use_sco=False`, `constraints=()`) —
   trajectories/costs should match pre-refactor bit-for-bit (same RNG seeds,
   same `n_iters`), confirming the shared-core rewrite is behavior-preserving
   for every real caller.
2. **Existing trajopt tests must keep passing unmodified**:
   `tests/test_contact_rich_trajopt.py`, `tests/test_flat_contact_trajopt.py`,
   `tests/test_contact_trajopt.py` — except the two assertions in
   `test_contact_rich_trajopt.py` that concern torque-limit behavior, which
   need re-checking once torque limits become a true AL constraint (expect
   tighter, not looser, feasibility — should still pass, but re-verify
   `resid`/force values numerically since the mechanism changed).
3. **New tests** (none of tiers 1/1-with-SCO have any today):
   - `tests/test_dynamics_trajopt.py`: parity test (old vs. refactored
     `dynamics_trajopt` at default config, on `ioc/bench2d`'s 2D problem for
     speed); a new test with `use_sco=True` + a collision constraint,
     checking it out-performs (fewer iterations to a given cost, or lower
     final cost at equal iterations) plain L-BFGS, mirroring the framing in
     `test_flat_contact_trajopt.py::test_flat_beats_naive_and_al`.
   - A test exercising `constraints=()` → non-empty transition with a
     synthetic arbitrary constraint (not collision/grasp/dynamics) to prove
     the generic AL interface actually works for caller-supplied residuals.
   - A test that `dynamics_feasibility_residual`-as-AL-term genuinely drives
     `max|tau| - tau_max` to ~0 (not just discourages it) on a case where
     the old hinge penalty left measurable residual violation.
   - `tests/test_contact_rich_trajopt.py` addition: a new environment-contact
     scenario (single manipulator pushing against a fixed obstacle, no
     pre-declared grasp) that was previously impossible without mode
     enumeration — checks the complementarity projection converges to
     `f_n >= -tol`, `f_n * phi < tol` (approximate complementary slackness),
     and that it vmaps over a batch without a Python loop (timed comparison
     against a naive per-sample loop, or a shape/trace assertion).
4. **GPU dynamics correctness**: `tests/test_contact_rich_trajopt.py`'s
   GPU-gated fixtures (`ManipulatorSpec`/`ContactSystem` via
   `_build_system`) double as the integration check that
   `dynamics_feasibility_residual` composes correctly with GRiD's
   vmap-fused, `custom_jvp`-differentiable `inverse_dynamics` inside the new
   generic AL loop — run on a free GPU per `nvidia-smi` check before
   invoking (per project convention).
