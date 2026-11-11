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

- **Manual GLASS checkout**: GRiM's crocket IK needs GLASS `9a24f37` (block trsv/potrs fix for
  `__restrict__` callers), but upstream GRiD pins `9e57178` and we can't push to GRiD. Setup
  checks GLASS out by hand (`AGENTS.md`, Environment). When GRiD bumps GLASS, bump GRiM, then
  delete this entry and the notes in `AGENTS.md` and GRiM's `AGENTS.md` / `OPEN_TASKS.md`.

- **Panda sphere pair that always collides**: with `panda.srdf`'s disabled pairs, one
  `panda_link5`–`panda_link7` sphere pair is 0.0152 apart in every configuration (both centres
  sit on the joint axes), inside the default 0.02 margin. Every self-collision hinge on panda
  carries a constant 0.0048, so LS trajopt's cudals self-collision constraint can never reach 0.
  Fix: disable that pair in the SRDF, or move the sphere.

- **GRiDDynamics still on GRiD's generated kernels**: `dynamics/_grid_dynamics.py` (and
  everything built on it: `_contact.py`, dynamics-aware/contact trajopt, TOPP-RA's GRiD
  backend) uses what crocket cannot trace yet: floating base (g1), runtime inertia (payload
  ID, `_grid_runtime_inertia.py`), the second-order `idsva_so` tensors (`jax.hessian` /
  JVP-of-gradient rules) and a `gravity` argument (crocket fixes -9.81). Fix in crocket first
  (floating-base RNEA/CRBA/ABA, runtime inertia table, second-order RNEA derivatives), then
  swap `GRiDDynamics` onto `CrocketDynamics` and drop GRiM's `generated_dynamics`.
  `ContractDynamics` is already crocket-only.
- **pyroffi's own CUDA kernels are hand-written**: `src/pyroffi/cuda_kernels/` (ESDF,
  TOPP-RA, OptiX/roboGPU, world collision) stays outside crocket by decision; revisit if
  crocket gains the primitives.

## Test debt

- **float32 borderline tests** _(verify)_: ~18 grid/autodiff/flat tests fail under float32 and
  pass with `JAX_ENABLE_X64=1`. Tolerances or formulation need a decision.

## Deferred features

- **Tier-3 C3 complementarity** in `_trajopt_core`: not done.
- **Hard torque-limit constraint on e3**: the AL adjoint is ill-conditioned (aug_stat stalls
  ~1e-1, 62% discard).
- **Usability refactor**: 5-phase install/layout/configs/docs/API plan, not started.
- **LS trajopt on cudals lags the JAX solver**: on the `tests/test_ls_trajopt.py` scene (panda,
  T=32, one seed set) its mean cost is 3–4% above JAX's at B=64 and B=256, and the collision
  constraint ends ≤0.065 violated at the default `n_outer_iters=8` AL updates. It is 1.6× faster
  at B=256 and slower below B≈64. Tune the AL schedule (penalty growth, `al_iters` vs
  `max_iters`) and rerun with ≥6 seeds.
- **Native world collision uses finite differences**: `costs/_native.py` world mode takes each
  sphere centre's distance gradient by central differences (self mode is analytic). Fix:
  analytic per-primitive gradients from `collision.cuh`.
