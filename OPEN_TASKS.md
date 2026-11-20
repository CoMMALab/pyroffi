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

- **Manual GLASS checkout**: GRiM's mantis IK needs GLASS `9a24f37` (block trsv/potrs fix for
  `__restrict__` callers), but upstream GRiD pins `9e57178` and we can't push to GRiD. Setup
  checks GLASS out by hand (`AGENTS.md`, Environment). When GRiD bumps GLASS, bump GRiM, then
  delete this entry and the notes in `AGENTS.md` and GRiM's `AGENTS.md` / `OPEN_TASKS.md`.

- **cudals dense-problem size limit**: jaxls example 3 (50 2D waypoints, 100 vars, 198 rows)
  fails in `analyze()`'s default dense path with a bare `StopIteration` from `auto_tier`
  (`external/cudals/src/cudals/_compile.py:588`). `structure="auto"` only checks the 128-var
  cap, but per-problem state `4·(2n_q + 2n + n² + 5r + r·n)` is ~124 KB against the 48 KB
  shared budget; the stored dense Jacobian (`r·n`) dominates. Fix in cudals first, then bump
  the pointer: (1) accumulate JᵀJ / Jᵀr row by row instead of storing J (state ~n², plus
  opt-in >48 KB shared, columns looped past 128 threads); (2) `auto` checks `available()` and
  picks banded when the costs are chain-structured, else raises a clear error. Large sparse
  problems (jaxls g2o graphs) need a separate sparse/CG solver.

- **Panda sphere pair that always collides**: with `panda.srdf`'s disabled pairs, one
  `panda_link5`–`panda_link7` sphere pair is 0.0152 apart in every configuration (both centres
  sit on the joint axes), inside the default 0.02 margin. Every self-collision hinge on panda
  carries a constant 0.0048, so LS trajopt's cudals self-collision constraint can never reach 0.
  Fix: disable that pair in the SRDF, or move the sphere.

- **GRiDDynamics still on GRiD's generated kernels**: `dynamics/_grid_dynamics.py` (and
  everything built on it: `_contact.py`, dynamics-aware/contact trajopt, TOPP-RA's GRiD
  backend) uses what mantis cannot trace yet: floating base (g1), runtime inertia (payload
  ID, `_grid_runtime_inertia.py`), the second-order `idsva_so` tensors (`jax.hessian` /
  JVP-of-gradient rules) and a `gravity` argument (mantis fixes -9.81). Fix in mantis first
  (floating-base RNEA/CRBA/ABA, runtime inertia table, second-order RNEA derivatives), then
  swap `GRiDDynamics` onto `MantisDynamics` and drop GRiM's `generated_dynamics`.
  `ContractDynamics` is already mantis-only.
- **pyroffi's own CUDA kernels are hand-written**: `src/pyroffi/cuda_kernels/` (ESDF,
  TOPP-RA, OptiX/roboGPU, world collision) stays outside mantis by decision; revisit if
  mantis gains the primitives.

- **Undeclared `beautifulsoup4` for the CUDA paths**: in an environment holding only
  `pyproject.toml`'s dependencies, mantis-traced kernels (seen for IK, collision and dynamics
  tests) fail with `ModuleNotFoundError: No module named 'bs4'`, raised from GRiD's `URDFParser`
  (`external/GRiM/external/mantis/python/mantis/robot.py`, `_import_parser`). GRiD declares
  `beautifulsoup4>=4.9.3`, but pyroffi puts GRiM on `sys.path` instead of installing it, so
  nothing pulls it in. Add it to pyroffi's dependencies (or to an extra the CUDA paths document).
- **`external/cudals` still pins `numpy==2.3.4`**: harmless while cudals is imported from the
  submodule path, but `pip install external/cudals` would downgrade numpy now that pyroffi
  accepts `numpy>=2.3.4,<2.6`. Widen it in cudals, then bump the pointer.

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
