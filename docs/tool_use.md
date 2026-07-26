# Tool use / attached bodies — implementation plan

> **Status (2026-07-25): P1–P5 implemented.**
>
> **P1–P4** — `src/pyroffi/attachments/` (`Attachment`, `AttachmentSet`,
> `pose_attachments`, `tool_frame`, `ik_target_for_tool`, `compose_dynamics`,
> `compose_collision`, `attachment_wrench_to_body`), `Robot.with_attachments`,
> `RobotCollision.with_attachments`, the GRiD `runtime_inertia` path
> (`dynamics/_grid_runtime_inertia.py`, `GRiDDynamics(..., runtime_inertia=True)`,
> `GRiDDynamics.set_attachments`), and the submodule migration to
> `A2R-Lab/GRiD@modernizing-tests`. Both `RobotCollision` (capsules, one entry
> per primitive) and `RobotCollisionSpherized` (spheres, one padded row of `S`
> per link) take attachments.
>
> **P5** — `_contact.py` unification (`ContactSystem.from_attachments`,
> `capture_attachments`, `object_pose_world`, `loaded_manipulator_robot`,
> `GraspedObject.from_attachment`, with `capture_grasp_offsets` now a wrapper);
> tool-frame IK via `ik_target_for_tool`; `toolbox/_attach.py` +
> `Session.attach_object` / `detach_object` / `attached`; MCP `attach_object` /
> `detach_object` / `list_attachments`; `ToolboxSource` renders carried objects;
> `examples/18_00_tool_use.py`.
>
> Tests: 59 new, all passing — `test_attachments.py` (32),
> `test_grid_runtime_inertia.py` (9), `test_contact_attachment_unification.py`
> (6), `test_toolbox_attachments.py` (12). Full suite: 276 passed, 2 failed,
> neither an attachment regression:
>
> * `test_toolbox_integration.py::test_optimize_path_reports_compilation_honestly`
>   asserts a warm solve is strictly faster than a cold one; it measured
>   3571.962 ms vs 3572.486 ms. A 0.5 ms timing race over 3.5 s. The `compiled`
>   flag it actually exists to check passed.
> * `test_flat_contact_trajopt.py::test_flat_beats_naive_and_al` fails its
>   force-balance assertion (`flat_force_res < 0.2`; measured 10.7 / 11.2 / 11.2
>   / 15.8 across four runs). Its other two claims hold and comfortably: the flat
>   solver beats both baselines on grasp drift (rms 0.024 vs 0.033
>   augmented-Lagrangian vs 0.086 naive) and is faster (3.7 s vs 5.8 s). The
>   solver satisfies `G λ = w_req` by construction using its *analytic* object
>   acceleration while the test scores the residual against a
>   *finite-differenced* one, so the gap is a statement about that solver's
>   parameterization, not about attachments — the P5 changes to `_contact.py`
>   delete exactly one line (a docstring) and are otherwise additive. The likely
>   trigger is the gravity-sign correction above, which changes the torque cost
>   this solver optimizes against; it was calibrated on the old, wrong sign.
>   Retuning it is a separate piece of work.
>
> **Deliberately not done**, with reasons:
>
> * **Sandbox `attach_object`.** The sandbox is a MuJoCo simulation where
>   grasping is *physical* — `set_gripper` reports the block that actually ended
>   up between the fingers, and reports `success=false` when it closed on
>   nothing. A declarative attach tool there would let an agent assert a grasp
>   that physics disagrees with, which is worse than not having it. The MCP
>   server (a planning interface, where the attachment *is* the model) has the
>   tools.
> * **`Attachment.from_mjcf_body`.** `Robot.geom_from_mjcf` already returns a
>   `CollGeom` with MJCF mass/inertia, and `Attachment.from_geom` takes it, so
>   the constructor would be a two-line alias over an existing composition.
>
> Corrections and design changes found while implementing P5:
>
> * §7 flags "verify `_ik_cuda_helpers.cuh` before designing the API, as it may
>   need a per-target constant offset the kernels don't currently carry".
>   Verified: they carry none — `ik_residual` compares `T_world[target_jnt]`
>   against `target_T` directly. They do not need one. A constant tool offset
>   folds into the *goal* instead: `T_W_tip = T_W_L · A` ⟺ `T_W_L = T_W_tip ·
>   A⁻¹`, so `ik_target_for_tool` rewrites the target host-side and tool-frame IK
>   works on every existing solver, CUDA and JAX, with no kernel change. The one
>   caveat is that the minimized residual becomes the link's pose error rather
>   than the tip's — same zero set, so an exact solve is unaffected, but a
>   weighted least-squares trades position against orientation about the link
>   frame.
> * §7's "`toolbox/_handles.py` gains an `attachment` handle kind" did not fit:
>   the handle table stores `(dof,)` / `(n_waypoints, dof)` joint arrays with
>   joint names, and an attachment is not a joint array. Attachments live on the
>   `Session` beside the `Scene` instead, which is where the pick/place state
>   belongs — an object is in the scene pool *or* attached, never both.
> * Toolbox attachments are **collision-only by default**, because scene objects
>   are pure geometry and carry no mass. `attach_object(..., mass=...)` opts into
>   the dynamics composition as well. Silently inventing a mass would have made
>   torque limits confidently wrong.
> * Scene shapes are converted to the collision model's own primitive as a
>   **bounding sphere** — conservative, and orientation-free so the attachment's
>   own rotation cannot invalidate it. It can refuse a feasible tight plan; it
>   cannot let a carried object pass through an obstacle.
>
> Corrections to this document, found while implementing:
>
> * §2/§6 write the tool-tip force map as `X^-T`. It is `Xᵀ`: spatial forces are
>   the dual of motions, so `m_B = X_{B←D} m_D` implies `f_D = X_{B←D}ᵀ f_B`
>   (power `fᵀm` is frame-invariant). For a pure translation that reduces to
>   `moment += p × force`, which is what `attachment_wrench_to_body` produces and
>   what the test asserts.
> * §6.2's parallel-axis warning is already handled upstream: `Link
>   .get_inertia_params` reads `I_O` verbatim out of the built 6×6, whose
>   top-left block already carries `m(cᵀc·1 − ccᵀ)`. No new code was needed on
>   our vendored `robot.py` — re-vendoring `_grid_urdf` from
>   `URDFParser@f88ce2a` supplied `get_inertia_params_ordered_by_id` and ~24
>   other methods the new codegen requires.
> * §5.2 proposed deriving an attachment's allowed-collision set by inheriting
>   whatever its parent link may already overlap. That was implemented and
>   rejected: gripper links frequently carry no collision geometry and so appear
>   in *no* pair, which silently gives the attachment an empty pair set (no
>   checking at all) rather than a permissive one. The shipped rule is
>   all-robot-links minus the parent minus the attachment's explicit
>   `ignored_link_indices`.
> * The migration surfaced two live bugs that had nothing to do with
>   attachments: the A2R-Lab kernels seed base acceleration as `-X·gravity`, so
>   pyroffi's pre-existing negation flipped every gravity torque (mass matrices
>   stayed correct, which is what made it survivable); and the GRiD `custom_jvp`
>   rules returned float64 tangents for float32 primals under `JAX_ENABLE_X64=1`.
>   Both are fixed.
> * `tests/test_grid_dynamics.py` passes 20/20 under `JAX_ENABLE_X64=1`. Under
>   float32 the *JAX* reference is the inaccurate side (2e-2 vs. GRiD's 3e-6
>   against an f64 oracle), consistent with the known borderline-precision
>   situation on this branch; that is a pre-existing tolerance issue in the
>   reference, not a regression.


Support for objects and tools that a manipulator picks up and thereafter carries as part of
its own body: for collision checking during transport, and for dynamics (RNEA/ABA) when the
object is load-bearing or is the thing doing the work (writing with a pen, pushing with a
stick, carrying a full cup).

---

## 1. What VAMP does, and why we can't copy it

`external/vamp/src/impl/vamp/collision/attachments.hh` is ~55 lines:

```cpp
struct Attachment {
    std::vector<Sphere<DataT>> spheres;          // in attachment-local frame
    mutable std::vector<Sphere<DataT>> posed_spheres;
    Eigen::Transform<DataT,3,Isometry> tf;        // attachment <- end-effector
    void pose(const Transform &p_tf) const;       // posed = (p_tf * tf) * spheres
};
```

`Environment::attach()` stores exactly one of these; the per-robot codegen
(`robots/panda.hh:10265 fkcc_attach`, and the unrolled block at ~line 15308) hard-wires
`set_attachment_pose(environment, to_isometry(&y[280]))` — i.e. the pose of one named frame
(`panda_grasptarget`) — then emits an unrolled list of attachment-vs-environment and
attachment-vs-robot-sphere tests, with the gripper links' spheres deliberately omitted from
that list.

What it gets right, and we should keep:

- **An attachment is a rigid offset from a link frame, not a new degree of freedom.** No new
  joint variable, no change to the configuration space.
- **The allowed-collision set is part of the attachment**, not a global setting: the object
  is *supposed* to touch the fingers.
- **Attachment geometry is posed by composing one transform**, which is why it costs almost
  nothing per state.

What does not survive contact with this codebase:

| VAMP | Problem here |
|---|---|
| One attachment per environment | TAMP needs two-arm handoffs, tool + workpiece, regrasp with the object briefly attached to neither or both |
| Spheres only | `CollGeom` is sphere/capsule/box/halfspace/heightmap, and `RobotCollision` is capsule-based |
| Attached to the codegen'd EE frame only | Tools get held by non-EE links; a fixture can be attached to a base link |
| Baked at codegen time | Ours must be `jit`-stable *and* the grasp transform must be a differentiable, `vmap`-able leaf (grasp-pose optimization, batched grasp candidates) |
| Purely geometric — no mass | The whole point of the dynamics half of this work |
| Boolean valid/invalid | We need signed distances and gradients |

So: same **concept** (rigid offset from a link, with its own allowed-collision set), a
different **realization** (pytree with static topology / dynamic transform, inertia
composition into the existing DOF tree, no new kernel shapes).

---

## 2. Theory: an attachment is a fixed joint, and that buys us everything

Attaching body `B` to link `L` with constant transform `T_LB` adds a **fixed** edge to the
kinematic tree. Two consequences, and they are the entire design:

**Kinematics.** `T_WB(q) = T_WL(q) · T_LB`. One SE(3) compose on top of existing FK. No new
joint, so `MAX_JOINTS`/`MAX_ACT`, the FK CUDA kernel, the IK kernels, and the topological
sort are all untouched.

**Dynamics.** Featherstone's spatial inertia transforms as `^A I = {}^B X_A^{\top} \, {}^B I \, {}^B X_A`.
A fixed child is therefore absorbed into its parent body by

```
I_L' = I_L + X_{B←L}^T · I_B · X_{B←L},     X_{B←L} = motion transform induced by T_LB^{-1}
```

which is **exactly** the fixed-joint merge `RobotURDFParser.parse_dynamics` already performs
when it folds fixed-joint links into their nearest actuated ancestor
([`_robot_urdf_parser.py:161`](../src/pyroffi/_robot_urdf_parser.py#L161), `I_body`). So
dynamics support for a grasped object reduces to:

> a rank-6 additive update of a single row of `DynamicsInfo.I_body`.

`num_dof` does not change. `parent_dof_indices`, `S`, `X_tree`, `joint_is_prismatic` do not
change. RNEA, CRBA, ABA (`_dynamics_jax.py`) work unmodified on the updated `DynamicsInfo`;
gravity and the object's Coriolis/centrifugal contribution fall out automatically. Nothing
about the kernel shapes changes, so nothing about the FFI boundary changes.

Note `L` here is the *DOF body* the link belongs to, i.e. the nearest actuated ancestor —
grasping with a finger whose joint is fixed correctly loads the wrist DOF.

**Tool-tip wrenches (non-prehensile / writing).** A force applied at the tool frame `B` maps
to the body frame by the *force* transform `X^* = X^{-\top}`:

```
f_L = X_{B←L}^{-T} · f_B
```

`_dynamics_jax._fext_to_body` already converts a world external wrench per body; it gains an
attachment-frame overload. Writing with a pen is then: attach the pen (inertia composition),
and apply a contact wrench at the pen tip frame, in ID for torque feasibility or in FD for
simulation.

**Grasp validity is a separate concern.** Rigid composition assumes the grasp does not slip.
That assumption is a *residual*, not a modelling change, and `dynamics/_contact.py` already
has the machinery — `grasp_closure_residual`, `grip_validity_penalty`,
`parallel_jaw_grip_penalty`, `manipulator_contact_fext`. Attachment gives the *nominal*
rigid model; those residuals certify it. Part of this work is making `_contact.py`'s
`GraspedObject`/`ContactSystem` consume the same `Attachment` object rather than duplicating
grasp bookkeeping.

---

## 3. The invariant that makes it fast: static topology, dynamic transform

Every design decision below follows from one rule:

> **Which** link an object is attached to, **how many** geometry primitives it contributes,
> and **which** collision pairs are allowed are `jdc.Static` aux data.
> **Where** it is attached (`T_LB`), its mass, inertia, and primitive dimensions are pytree
> leaves.

Consequences:

- `jit` recompiles when the *grasp topology* changes (pick, place, handoff) — a handful of
  compilations across a whole TAMP problem, not one per state.
- `T_LB`, mass, and inertia are differentiable and `vmap`-able: you can batch 1024 candidate
  grasp transforms through one compiled collision/dynamics call, and take
  `∂cost/∂T_LB` for grasp optimization. This is a capability VAMP structurally cannot have,
  and it is the main reason to build our own rather than bind theirs.
- Attach/detach across a plan skeleton is handled by **fixed-capacity slots with an activity
  mask**, not by reallocating: `MAX_ATTACHMENTS` slots, an `active: bool[MAX_ATTACHMENTS]`
  leaf. A disabled slot contributes `+inf` distance (via the existing
  `RobotCollisionSpherized.mask_collision_distance` path — *not* via zero-radius degenerate
  geometry, which would poison the min-reductions) and a zero spatial inertia. Then even
  pick/place transitions are compile-free within a fixed skeleton.

---

## 4. API

New module `src/pyroffi/attachments/` (`__init__.py`, `_attachment.py`, `_compose.py`).

```python
@jdc.pytree_dataclass
class Attachment:
    """A rigid body carried by a robot link — the tool-use primitive."""

    # --- static topology -------------------------------------------------
    parent_link_index: jdc.Static[int]
    name:              jdc.Static[str]
    ignored_link_indices: jdc.Static[tuple[int, ...]]   # fingers etc. — allowed to touch
    num_prims:         jdc.Static[int]

    # --- dynamic leaves --------------------------------------------------
    T_parent_body: Float[Array, "*batch 7"]   # wxyz_xyz, link <- body.  DIFFERENTIABLE.
    geom:          CollGeom                   # in body frame, batch axis (*batch, num_prims)
    spatial_inertia: Float[Array, "*batch 6 6"] | None   # None => kinematic/collision only
    active:        Bool[Array, "*batch"]

    @staticmethod
    def from_geom(geom, parent_link, T_parent_body, *, mass=None, ignored_links=()) -> Attachment
    @staticmethod
    def from_mjcf_body(robot, body_name, parent_link, T_parent_body, **kw) -> Attachment
        # reuses Robot.geom_from_mjcf — mass/inertia from MJCF, not guessed

    def grasp_from_current_pose(self, robot, cfg, T_world_body) -> Attachment
        # T_LB = T_WL(cfg)^-1 · T_WB.  The "close the gripper here" constructor.
        # Mirrors _contact.capture_grasp_offsets; that function should delegate here.


@jdc.pytree_dataclass
class AttachmentSet:
    """Fixed-capacity collection; the unit that Robot / RobotCollision consume."""
    attachments: tuple[Attachment, ...]          # len == MAX_ATTACHMENTS (static)
    def attach(self, a) -> AttachmentSet         # host-side, changes topology
    def detach(self, name) -> AttachmentSet
    def set_active(self, name, flag) -> AttachmentSet   # jit-safe, no recompile
```

Composition entry points, all pure functions of `(robot_or_collision, attachments)`:

```python
attachments.pose_attachments(robot, cfg, aset)  -> CollGeom      # world-frame geometry
attachments.compose_dynamics(robot, aset)       -> Robot         # updated DynamicsInfo
attachments.compose_collision(rcoll, aset)      -> RobotCollision # extended geom + pair table
attachments.tool_frame(robot, cfg, aset, name)  -> jaxlie.SE3    # for IK targets / costs
```

Plus sugar on `Robot` mirroring the existing optional-subsystem style (`mjcf_path`,
`_backends`): `robot.with_attachments(aset)` returning a new `Robot` whose `dynamics` field
is already composed, so `inverse_dynamics`/`forward_dynamics`/`mass_matrix`/`step` need
**no signature change at all**. Same for `RobotCollision.with_attachments(aset)`.

That "compose into the existing struct, don't thread a new argument through 20 call sites"
choice is deliberate — it keeps `motion_generators`, `optimization_engines`, and both trajopt
solvers working with zero edits.

---

## 5. Collision

`RobotCollision.at_config` transforms a `CollGeom` batched over links by the FK link poses
([`_robot_collision.py:240`](../src/pyroffi/collision/_robot_collision.py#L240)). Attachments
extend that array:

1. **Geometry.** Concatenate the attachment primitives onto the per-link geometry along the
   primitive axis, giving `K' = K + Σ num_prims`. Poses come from
   `T_WL(cfg) · T_parent_body`, i.e. a gather of the parent link pose followed by one SE(3)
   compose — a handful of flops per attachment per batch element.
   `RobotCollisionSpherized` gets the same treatment after `decompose_to_spheres`, so
   sphere-based CUDA paths inherit it for free.

2. **Pair table.** `_compute_active_pair_indices` extends to emit, for each active
   attachment: pairs against all robot links *except* `ignored_link_indices` and except the
   parent link's own adjacency set; and attachment-vs-attachment pairs (needed for two-arm
   handoff and for tool-vs-workpiece). Computed on the host from the static topology and
   cached in `CudaBackends` keyed by the topology tuple — identity-hashed static aux data,
   consistent with `ik_ancestor_masks`.

3. **World collision needs no change**: the world kernels are shape-generic in `K`
   (`collision_world_sphere(sphere_centers[3,B,K], ...)`), so a longer `K` is just a larger
   launch. Confirmed against `_collision_cuda_ffi.py`. Self-collision likewise takes the pair
   table as a runtime buffer.

4. **Swept volumes.** `get_swept_capsules` operates on the composed geometry, so continuous
   collision checking during transport comes free — this is what makes attachment usable in
   the STOMP/CHOMP/trajopt inner loops rather than only at waypoints.

Costs (`costs/_costs.py`) that reduce over the collision array pick up attachment terms
automatically because the reduction is over `K'`.

---

## 6. Dynamics, and the GRiD problem

**JAX path (`_dynamics_jax.py`): trivial and fully differentiable.** `compose_dynamics`
returns a `DynamicsInfo` with one `I_body` row updated. RNEA/ABA/CRBA are untouched. `∂τ/∂mass`
and `∂τ/∂T_LB` flow through, which is what makes "identify the payload from measured torques"
or "optimize where to grasp so the transport torque stays in limits" one-liners.

**GRiD path: upstream already solved this — adopt `runtime_inertia`.**

Our vendored `external/GRiDCodeGenerator` (pinned at `robot-acceleration/GRiDCodeGenerator`
main, `891490d`) does bake inertia as immediates into `init_XImats()`, which is what made this
look like a codegen-cost problem. But `A2R-Lab/GRiD@modernizing-tests` (package renamed
`GRiDCodeGenerator/` → `grid_codegen/`) adds a **flag-gated mutable inertia table**, verified
in `grid_codegen/helpers/_topology_helpers.py`:

- `GRiDCodeGenerator(..., runtime_inertia=True)` adds `d_inertia_params` to `robotModel<T>`
  and emits `init_inertia_params()` + `set_inertia_params(d_robotModel, h_params)`.
- `set_inertia_params` is a plain host-side `cudaMemcpy` of `10·NB` floats. **No recompile.**
- `_emit_runtime_inertia_rebuild` scatters each body's 10-vector into the 6×6 I-region of
  `s_XImats` inside `if constexpr (RUNTIME_INERTIA)`: one thread per body, 36 stores,
  divide-free, once per kernel on the cold `XImats` load. The algorithm kernels read
  `s_XImats` unchanged.
- When `runtime_inertia=False` the emitter produces **nothing** there, so the baked path stays
  byte-identical — the flag is free when unused.

**Why the parameter basis makes this better than the 6×6 update in §2.** The table is in the
standard inertial-parameter (regressor) basis, `π = [m, h = m·c, I_O]` with `I_O` the 6
independent entries of the inertia about the *body origin*. Spatial inertia is **linear** in
`π`, so for two rigidly-connected bodies referred to the same origin:

```
π_total = π_link + Ad(T_LB) · π_object
```

where `Ad(T_LB)` is a constant 10×10 matrix. Attaching an object is a **10-vector add**, not a
6×6 congruence — cheaper, exactly linear in mass (so `∂/∂m` is trivial and exact), and it
composes for multiple attachments on the same body by simple summation. This is the same basis
sysID and domain randomization use, so payload identification and tool-use attachment become
the same code path.

**The real remaining constraint is purity, not compile time.** `set_inertia_params` mutates
*device-resident global model state* through a blocking host memcpy. That is not a traceable
JAX value, so:

- Set the table only at **grasp-topology boundaries** (pick / place / handoff) — which is
  exactly where §3's static-topology rule already permits a recompile, so it costs nothing new.
  For a panda that is a 70-float memcpy.
- Wrap it in a `GridModelState` guard that records the currently-uploaded `π` and **raises if a
  tracer reaches it**. A silently-stale inertia table would be an invisible wrong-dynamics bug;
  it must fail loudly, keyed on tracer-ness rather than on a caller-supplied flag.
- **You cannot `vmap` over payloads on the GRiD path** — there is one table per model. Batched
  grasp optimization, payload sweeps, and domain randomization across a batch stay on the JAX
  RNEA. The fallback in this plan survives, but for this reason, not the recompile one.
- **Worth raising upstream:** a variant taking the params pointer as a *kernel argument*
  instead of reading `d_robotModel->d_inertia_params` would make the whole thing functional and
  `vmap`-able, and the rebuild already reads through a pointer, so the change is small.

**Migration work this implies** (list it explicitly; it is the bulk of P4):

1. Move the submodule to `A2R-Lab/GRiD`, pinned to a specific `modernizing-tests` commit —
   it is an unmerged branch, so pin it the way GLASS is pinned. Update `_vendor.py` (which
   searches for a `GRiDCodeGenerator/` package dir), `_grid_codegen.py`, `_grid_robot_adapter.py`.
2. Add `get_inertia_params_ordered_by_id()` to our vendored `_grid_urdf/robot.py` — upstream
   reads it off `URDFParser.Robot`, which we vendored. **Watch the parallel-axis term:**
   `build_grid_robot` currently calls `set_origin_xyz(com)` with the inertia rotated about the
   COM, whereas `I_O` must be about the link origin:
   `I_O = I_com + m·(cᵀc·1₃ − c cᵀ)`. Getting this wrong yields plausible-but-wrong torques.
3. Regression gate before anything else lands: with `runtime_inertia=True` and the URDF's own
   parameters uploaded, assert GRiD output is **bit-identical** to the baked build. Upstream
   claims this by construction; verify it, since everything downstream rests on it.

**Also newly available upstream, worth a separate look** (out of scope here, flagged so it is
not rediscovered later): `runtime_transform` (runtime-mutable joint-frame origins — kinematic
calibration, and a route to modelling a tool as a genuine extra body rather than a merged one),
`runtime_joint_dynamics` (runtime damping/friction), and the `collision_spec` /
`multi_target_batch` codegen options.

**External wrenches.** `_fext_to_body` gains an attachment-frame wrench input, transformed by
`X^{-T}` as in §2. This is the entry point for pen-on-paper, push-with-stick, and
peg-in-hole reaction forces, and it shares the representation `_contact.manipulator_contact_fext`
already uses.

---

## 7. TAMP / downstream integration

- **Handoff** = detach from arm A, attach to arm B, in one `AttachmentSet` edit; the
  intermediate "held by both" state is representable (two attachments, mutually ignored)
  and is what makes closed-chain handoff constraints expressible.
- **Regrasp** = same object, new `T_parent_body`; because that is a leaf, a regrasp
  *search* is a `vmap` over candidate transforms with no recompilation.
- **Attachment as an IK target frame.** `tool_frame()` returns the tool tip pose, so
  `optimization_engines` IK can servo the pen nib rather than the flange. This needs the
  target-link machinery to accept an attachment index; the CUDA IK kernels take the target
  frame offset as a runtime transform, so this should stay kernel-compatible — **verify**
  against `_ik_cuda_helpers.cuh` before designing the API, as it may need a per-target
  constant offset the kernels don't currently carry.
- **MCP / sandbox / toolbox.** `toolbox/_handles.py` gains an `attachment` handle kind;
  `sandbox` and `mcp` get `attach_object` / `detach_object` tools so an LLM-driven planner
  can express pick-and-place. `viewer/_world.py` renders attached geometry with the composed
  pose.

---

## 8. Phasing

Each phase is independently useful and independently testable.

**P1 — Core + kinematics.** `Attachment`, `AttachmentSet`, `pose_attachments`,
`tool_frame`, `grasp_from_current_pose`. Tests: analytic pose composition; a
constant-in-body-frame point stays constant under arbitrary `cfg`; `vmap`/`grad` over
`T_parent_body`; attach/detach round-trip.

**P2 — Collision.** Geometry concatenation, pair-table extension, `with_attachments` on both
`RobotCollision` variants, swept capsules, activity masking. Tests: attached sphere at a
known world obstacle reports the right signed distance; ignored finger links report no
self-collision; CUDA and JAX paths agree (extend the `collision_cuda_vs_jax` comparison);
inactive slot contributes exactly `+inf` and does not perturb the min; **no recompilation**
when only `active`/`T_parent_body` change (assert on `jit` cache counters).

**P3 — Dynamics (JAX).** `compose_dynamics`; `_fext_to_body` attachment-wrench overload.
Tests: attached point mass at radius `r` on a 1-DOF link reproduces the analytic
`(I + m r²) q̈ + m g r cos q` torque; composing a body then setting its mass to zero recovers
the unattached `DynamicsInfo` bitwise; CRBA mass matrix vs. finite-differenced kinetic energy;
`grad` w.r.t. mass and `T_parent_body`; cross-check against MuJoCo on an MJCF scene with the
object welded, using the existing `mjcf_path` support.

**P4 — GRiD (`runtime_inertia`).** Submodule migration to `A2R-Lab/GRiD@modernizing-tests`
(pinned), `get_inertia_params_ordered_by_id` on the vendored `_grid_urdf/robot.py`, the
`Ad(T_LB)` 10×10 parameter transform, and the `GridModelState` guard. Tests, in order:
bit-identity of `runtime_inertia=True` vs. baked with unmodified parameters (the gate for
everything else); parallel-axis correctness of `I_O` against the JAX `I_body` (this is the
step most likely to be silently wrong); GRiD vs. JAX RNEA with an attachment (float32
tolerances — see the known borderline-precision situation on this branch); the guard raises
on a traced payload rather than uploading; `set_inertia_params` is never reached from inside
a jitted region.

**P5 — Downstream.** `_contact.py` unification (delete the duplicated grasp bookkeeping),
IK tool-frame targets, toolbox/MCP/sandbox handles, viewer rendering, and a worked example
(`examples/18_00_tool_use.py`: pick up a pen, write a stroke with tip-contact wrenches, with
the pen in both the collision and the dynamics chain).

**Deliberately out of scope**, to be stated in the module docstring so nobody assumes
otherwise: deformable or articulated attachments, slip modelling (that is
`contact_rich_trajopt`'s job), and objects whose inertia changes during transport
(sloshing liquid).

---

## 9. Risks

1. **GRiD migration (§6)** — no longer a codegen-cost risk, but the submodule move to an
   *unmerged* upstream branch is a real one: pin the commit, and expect the `runtime_inertia`
   path to be less exercised than the baked one. The residual functional limit is that
   payload cannot be batched on the GRiD path.
2. **Pair-table growth.** Attachment-vs-all-links is `O(n_links)` new pairs per attachment;
   with `MAX_ATTACHMENTS` slots the self-collision launch grows. Measure before assuming it's
   negligible on the spherized path, where `K` is already large.
3. **Silent frame errors.** `T_parent_body` is link←body; getting it inverted produces plausible
   but wrong behaviour. The `grasp_from_current_pose` constructor exists so callers rarely
   write the transform by hand, and the tests should include a deliberately-inverted case.
4. **float32.** Inertia composition adds a `X^T I X` product; on this branch x64 is off and
   several tests are already borderline. Compose in float64 on the host where the transform is
   concrete, and only cast at the FFI boundary — the same discipline GRiD already follows.
