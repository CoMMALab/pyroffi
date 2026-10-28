# Open tasks: PyRoFFI

Things that have been found but not fixed yet. Add an entry when you find one, and delete it
when it's fixed (the fix commit is the record). Format: **title**: symptom / evidence / where /
suggested fix. Items marked _(verify)_ were seeded from earlier session notes and haven't been
re-checked against current code.

## Bugs

- **Flat contact solver force residual** _(verify)_: `flat_contact_trajopt` force residual is
  ~10, not ~0, which contradicts the "satisfied by construction" flatness claim.
- **Redundant-arm IK gradient** _(verify)_: first-order gradient through IK is ~80% off vs FD
  on redundant arms. The pinv assumes a minimum-norm solution, but the solver moves in the null
  space.
- **Tier-1 trajopt feasibility regression** _(verify)_: 12/12 → 6/12 since 2026-09-27.
- **G1 collision-aware IK accuracy** _(verify)_: still open after 62cb549.
- **Collision sentinel** _(verify)_: inf vs 1e9 "no collision" sentinel is inconsistent across
  backends. `panda_spherized` needs an SRDF.

## Test debt

- **float32 borderline tests** _(verify)_: ~18 grid/autodiff/flat tests fail under float32 and
  pass with `JAX_ENABLE_X64=1`. Tolerances or formulation need a decision.

## Deferred features

- **Tier-3 C3 complementarity** in `_trajopt_core`: not done.
- **Hard torque-limit constraint on e3**: the AL adjoint is ill-conditioned (aug_stat stalls
  ~1e-1, 62% discard).
- **Usability refactor**: 5-phase install/layout/configs/docs/API plan, not started.
