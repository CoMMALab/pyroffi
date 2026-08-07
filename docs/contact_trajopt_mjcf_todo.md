# Contact TrajOpt Generalization + MJCF-Sourced Dynamics — Status

This documents a completed refactor of pyroffi's contact-rich, dynamics-aware
SCO trajectory optimizer: generalizing it from a hardcoded two-arm/box design
to an arbitrary-manipulator-count design, and sourcing physical parameters
(mass/inertia/friction) from MuJoCo's compiler instead of hand-picked
constants. Kept as a reference for the design decisions/tradeoffs made.

## Summary of changes

### 1. `src/pyroffi/collision/_geometry.py`
- `CollGeom` gained three new **kw-only, defaulted** pytree fields (default
  zero, so all pre-existing `from_*` call sites across the codebase are
  unaffected): `mass: Float[Array, "*batch"]`, `inertia_diag: Float[Array,
  "*batch 3"]`, `friction: Float[Array, "*batch"]`.
- `CollGeom.with_physical_properties(mass, inertia_diag, friction)` — returns
  a batch-broadcast copy.
- `Sphere.from_center_and_radius`, `Box.from_center_and_dimensions` (+
  `from_center_and_half_lengths`), `Capsule.from_radius_height` all gained
  optional `mass=`/`inertia_diag=`/`friction=` kwargs, with an analytic
  solid-primitive inertia default when `mass` is given but `inertia_diag`
  isn't.
- `geom_from_mjcf_body(mj_model, body_name) -> Box | Sphere | Capsule` —
  parses a **compiled** `mujoco.MjModel`'s body (requires exactly one geom of
  a supported type) into a `CollGeom` with real `mass`/`inertia_diag`
  (`body_mass`/`body_inertia`, already diagonalized by MuJoCo's compiler) and
  `friction` (sliding component of `geom_friction`). Composes
  `body_pos/quat` with `geom_pos/quat` via `jaxlie.SE3` for the geom's world
  placement.
  - **Known simplification**: assumes the body's inertial frame coincides
    with its single geom's frame. True for MuJoCo's auto-computed inertias
    on single-geom bodies; not checked against explicit, non-coincident
    `body_ipos`/`body_iquat`.
- `box_with_mjcf_dynamics(center, half_lengths, mass, friction=0.0, wxyz=None) -> Box`
  — builds a `Box` whose inertia is computed by MuJoCo's compiler **without
  writing any file to disk**. Constructs a throwaway single-body,
  single-geom model in memory via `mujoco.MjSpec`, compiles it, and reads it
  back through `geom_from_mjcf_body`. Added after an initial version wrote a
  `resources/box_grasped.xml` asset file — the user pushed back on
  manifesting one file per grasped object, since that clutters the resources
  directory; `box_with_mjcf_dynamics` is the answer: same physics (literally
  the same MuJoCo compiler, same code path via `geom_from_mjcf_body`), zero
  files. **Prefer this over hand-authoring MJCF files for one-off objects.**
  Only write an actual `.xml` asset file when the object is a reusable,
  named resource worth version-controlling (e.g. a shared robot description).

### 2. `src/pyroffi/_robot.py`
- `Robot.from_urdf(urdf, default_joint_cfg=None, mjcf_path=None)` — new
  optional `mjcf_path` kwarg. When given, lazily imports `mujoco` and loads
  `mujoco.MjModel.from_xml_path(mjcf_path)`, stored as a new `jdc.Static[object
  | None]` field `_mjcf_model` (identity-hashed pytree aux-data, same pattern
  as `_backends`).
  - `robot.mjcf_model` property — raises `AttributeError` if no `mjcf_path`
    was supplied.
  - `robot.geom_from_mjcf(body_name)` — convenience wrapper around
    `geom_from_mjcf_body(self.mjcf_model, body_name)`.
- Note: SRDF is actually threaded through `RobotCollision.from_urdf`, not
  `Robot.from_urdf` (`Robot` had no existing SRDF hook to literally mirror).
  This `mjcf_path` kwarg establishes the same "optional file path, ignored if
  omitted" convention directly on `Robot`, for callers who want MJCF-derived
  per-link geometry via `robot.geom_from_mjcf(...)` without needing a
  separate `RobotCollision`. This is orthogonal to `GRiDDynamics`, which
  still gets its (correct) arm-link mass/inertia straight from the URDF —
  MJCF is for supplementary bodies (e.g. a grasped object) or friction data
  URDF doesn't carry, not a replacement for GRiD's dynamics pipeline.
- `resources/panda/*.xml` (MJCF conversions of the spherized Panda URDFs) and
  `scripts/urdf_to_mjcf.py` (the URDF→MJCF converter) were added by the user
  between sessions; `urdf_to_mjcf.py` copies URDF `<inertial>` data 1:1 (the
  panda URDFs have **placeholder** identical inertia `ixx=iyy=izz=0.1` on
  every link — not system-identified — so converting to MJCF doesn't by
  itself improve arm-link inertia accuracy, only adds real `friction`, which
  URDF has no concept of).

### 3. `src/pyroffi/dynamics/_contact.py` — fully rewritten for generality
Renamed/generalized (old → new):
- `ArmSpec` → `ManipulatorSpec` (same fields).
- `BoxSpec` → `GraspedObject(geom: CollGeom)` — `.mass`/`.friction` proxy
  `geom.mass`/`geom.friction`.
- `BimanualContactSystem` → `ContactSystem(manipulators: tuple[ManipulatorSpec, ...],
  body: GraspedObject, grasp_offsets: tuple[SE3, ...], gravity)`. Supports
  **any** manipulator count `k >= 1`. `manipulators[0]` is the reference
  gripper; `grasp_offsets[i]` is the reference→`manipulators[i+1]` transform
  captured at grasp time (`len == k - 1`, validated in `__post_init__`).
- Residual functions now loop over `system.manipulators` in Python (unrolled
  at JIT trace time — tuple length is static, not `vmap`'d, since
  manipulators may have heterogeneous DOF counts):
  - `grasp_closure_residual` — one `se(3)` residual per non-reference
    manipulator, shape `(6*(k-1),)` (empty for `k=1`).
  - `box_dynamics_residual` → `object_dynamics_residual` — force/torque
    balance summed over `forces: [k, 3]`.
  - `grip_validity_penalty` — loops over all `k` contacts; `mu_friction=None`
    now falls back to `system.body.friction` (the grasped geometry's own
    friction) instead of requiring a config value.
  - `arm_contact_fext` → `manipulator_contact_fext` (logic unchanged).
  - `capture_grasp_offset` → `capture_grasp_offsets(manipulators, qs)`.
- **No backward-compat aliases kept** — old names are gone, not deprecated.

### 4. `src/pyroffi/optimization_engines/_contact_trajopt.py` — rewritten
- Decision variable `lambda_t` generalized from a hardcoded `[T, 6]`
  `[f_L | f_R]` to `[T, k, 3]` (one 3-vector per manipulator).
- `_contact_cost` loops over `system.manipulators` for GRiD inverse-dynamics
  torques/effort/torque-limit costs.
- `ContactTrajOptConfig.mu_friction: float | None` (was `float`) — `None`
  uses the grasped object's own friction.
- `rho_box`/`rho_box_max` → `rho_obj`/`rho_obj_max`.
- `contact_sco_trajopt`'s default `init_forces` splits the object's weight
  evenly across however many manipulators there are.

### 5. `src/pyroffi/dynamics/__init__.py`
- Lazy `__getattr__` allowlist updated:
  `ArmSpec, BoxSpec, BimanualContactSystem, capture_grasp_offset` →
  `ManipulatorSpec, GraspedObject, ContactSystem, capture_grasp_offsets`.

### 6. `examples/16_00_bimanual_box_lift_contact.py` — rewritten
- Uses the new `ManipulatorSpec`/`GraspedObject`/`ContactSystem` API.
- Box built via `pk.collision.box_with_mjcf_dynamics(center=BOX_CENTER,
  half_lengths=BOX_HALF_LENGTHS, mass=0.5, friction=0.6)` — MJCF-derived
  inertia, zero files written.
- Visualization updated for `forces[k, i]` (`i` = manipulator index) instead
  of the old `forces[k, :3]`/`forces[k, 3:]`.
- The bimanual-specific assembly (two Panda arms, base offsets, pinch
  geometry) lives entirely in this example file — the library code
  (`_contact.py`, `_contact_trajopt.py`) has no arm-count assumptions.

### 7. `tests/test_contact_trajopt.py` — updated
- New API names; box built via `Box.from_center_and_dimensions(..., mass=0.5)`
  (analytic default inertia — this test doesn't need MJCF specifically, just
  exercises the solver); `forces.shape == (T, 2, 3)`.

## Verification

`pytest tests/test_contact_trajopt.py -q` → **2 passed** (~83s), run via
`/home/sadmin/miniconda3/envs/pyroffi/bin/python3` on the GPU machine (GRiD
requires CUDA; tests skip otherwise). `geom_from_mjcf_body` and
`box_with_mjcf_dynamics` were also spot-checked directly against both a
real MJCF file (`resources/panda/panda_spherized.xml`) and in-memory
`MjSpec`-built models, confirming mass/inertia/friction/pose all round-trip
correctly and that the file-based and in-memory paths produce identical
results for the same box.

## Possible future work (not requested, not started)

- Per-link friction/inertia sourcing for manipulator links themselves (as
  opposed to just the grasped object) via MJCF — `geom_from_mjcf_body`
  currently requires exactly one geom per body, which is false for the
  spherized Panda links (multiple collision spheres per link). Would need
  either picking a representative geom or extending the function to return
  a composite. Not needed for the grasped-object use case, so left
  unimplemented.
- `contact_sco_trajopt`'s `use_cuda=True` path is still `NotImplementedError`
  (pre-existing, unrelated to this refactor).
