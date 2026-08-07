# Scoping: an MCP server for pyroffi

Status: **implemented** — see [`mcp_server.md`](mcp_server.md) for what was
built, measured costs, and the four places where this design met reality and
had to change (collision batching, collision-model fidelity, static links, and
the simulation control rate). This document is kept as the original scope.

**Goal.** Expose pyroffi as a **toolbox of motion primitive operations** over MCP, so a VLM
orchestrating TAMP can use it for the geometric and dynamic sub-questions it cannot answer
itself: where the arm has to be, whether a configuration or motion is valid, and how to
turn a rough path into an optimized one.

**Explicitly not a planner.** pyroffi does not own the plan. The agent decides the task
decomposition and the sequencing; other tools may own the search. A realistic call sequence
in a TAMP stack looks like:

```
VLM: "move the block from A to B"
  → pyroffi.create_scene / add_object            (persistent scene)
  → pyroffi.solve_ik(pose_A), solve_ik(pose_B)   (endpoint configurations)
  → [another MCP server: RRT-Connect between those endpoints]
  → pyroffi.validate_path(waypoints)             (check what came back)
  → pyroffi.optimize_path(waypoints)             (smooth / refine via trajopt)
  → pyroffi.retime(path) → executable trajectory
```

pyroffi's trajopt is only loosely a planner — it's a local optimizer that needs a seed.
Scoping the server around **primitives that compose with foreign planners** matches what
the library actually is, and matches how it will be used.

**Shape of the work.** The MCP server is a thin adapter. The substance is a new
**session/toolbox layer** underneath it, which is the actual deliverable.

---

## 0. The session layer

Two facts drive the architecture.

**Warm state.** pyroffi's cost model is inverted relative to a classical library. A
trajopt call is dominated by XLA compilation on first invocation — tens of seconds (see
`docs/sco_trajopt.md`, and the warmup-recompile note in project memory). Steady-state
solves are milliseconds. A process-per-request design pays the compile every time and is
useless. The server must be one long-lived process holding a warm, jitted robot and scene.

**pyroffi's entry points are bare library calls.** `Robot.inverse_kinematics` and
`sco_trajopt` do what you ask and return. Something has to own the persistent scene, the
padded array layout, the handle table, and the honest reporting of what happened. That
layer is transport-agnostic and independently testable.

```
   MCP server  (pyroffi.mcp)      ← thin: tool schemas, summaries, handle plumbing
        │
   pyroffi.toolbox                ← sessions, scene, warm caches, primitives, exchange
        │
   pyroffi core                   ← unchanged
```

`mcp` must never be imported below `pyroffi.mcp`.

### What the session layer owns — and what it deliberately doesn't

**Owns:** the `Robot` / `RobotCollision` / scene triple, GPU selection, warm jit caches,
the named-object scene graph over fixed-capacity arrays, the handle table, shape bucketing,
time parameterization, and structured reporting of failures.

**Does not own:** planning policy. No automatic retry escalation, no "try a different
solver and don't tell anyone", no cost-acceptance heuristic. Those are the orchestrator's
decisions. Primitives expose their knobs (`num_seeds`, `solver`, tolerances) and report
what actually happened — including partial success, which for batched IK is real
information ("47/64 seeds converged, best residual 1.2e-4"). An agent that wants retry
logic writes it; a primitive that hides the retry makes the agent's model of the world
wrong.

### Shape-static requests

XLA specializes on shape. `n_timesteps`, `num_seeds`, batch size, and **object count** are
all static. A request with 41 objects after one with 40 triggers a full recompile. So:

- **Bucket** path lengths (32/64/128) and pad, or fix `n_timesteps` per scene.
- Hold scene geometry in **fixed-capacity arrays with an active mask**, sized at scene
  creation (`max_objects`), not per request. `pyroffi.collision` geometry is already
  batched-array shaped, so this is a padding convention, not a rewrite.
- Expose compilation explicitly: a `warmup()` entry point, and `compiled: bool` on every
  response so the agent can tell a 40 s answer from a 4 ms one.

This is the most error-prone part of the project. Without it, production looks like random
40-second stalls.

---

## 1. Two response registers: agent-facing and machine-facing

A 64×7 float64 path is ~4 k tokens of noise to a VLM. But the RRT server in the middle of
the pipeline needs the actual numbers. Both are true, so the server needs both registers:

- **Agent-facing (default).** Handle + decision summary. Everything the model needs to
  decide what to call next, and nothing else.

  ```jsonc
  optimize_path(...) -> {
    "path_id": "path_7f3a", "success": true, "n_waypoints": 64,
    "path_length_rad": 3.41, "min_clearance_m": 0.043,
    "cost_before": 8.9, "cost_after": 2.1, "solve_ms": 12.4, "compiled": false
  }
  ```

- **Machine-facing (explicit).** `export_path(id)` / `import_path(waypoints)` move real
  joint arrays across the server boundary, for handoff to a foreign planner and back.

**This is the piece that makes cross-server TAMP work, and it has to be designed, not
assumed.** Handles are server-local and opaque; a different MCP server cannot dereference
`path_7f3a`. The interop contract needs to be explicit about:

- **Joint ordering** — always name-keyed (`robot.joints.actuated_names`), never positional.
  Passive and mimic joints filtered (cf. the finger-joint masking in
  `examples/01_00_basic_ik.py`).
- **Quaternion convention** — pyroffi/jaxlie is `wxyz`. Anything crossing the boundary
  should be labeled, or you will chase a silent 90° error between servers.
- **Units and frames** — radians, meters, and the scene's world frame, stated in the schema.
- **Scene identity** — a path validated against pyroffi's scene is only meaningful if the
  foreign planner saw the same obstacles. `export_scene(format)` (URDF+SRDF, or a plain
  primitive list) lets the other server load it; responses carry a `scene_version` so a
  stale validation is detectable.

---

## 2. Primitives

### Scene

| Tool | Purpose |
|---|---|
| `create_scene(robot, max_objects, n_timesteps, gpu?)` | Session id + capabilities: DOF, joint names/limits, EE links, available backends (CUDA / GRiD / VAMP), `x64` on/off |
| `add_object(name, shape, pose, params)` | box / sphere / capsule / mesh |
| `remove_object(name)` / `list_objects()` | scene state + `scene_version` |
| `export_scene(format)` | hand the same world to another server |
| `set_robot_state(config_id \| named)` | current configuration |

Scenes are persistent and mutable across calls. `list_objects` must be cheap and the tool
description should encourage it — stale scene state is the most likely practical failure.

### Kinematic primitives

- `solve_ik(link, pose, num_seeds?, solver?, seed_config?)` → config handle + residual +
  seed-convergence count. Batched by construction, so return the distribution, not just
  the winner. `solver` is the caller's choice (`hjcd` / `ls` / `quik`), not the server's.
- `solve_ik_batch(targets)` → N config handles in one dispatch. **The tool that justifies
  pyroffi in this role**: an agent enumerating candidate grasps or placements gets all of
  them evaluated in a single GPU call. Tool descriptions should push toward this over
  serial `solve_ik`.
- `forward_kinematics(config_id, links?)` → poses. Useful for the agent to confirm where a
  config actually puts the hand.

### Validation primitives

- `check_collision(config_id | joint_values)` → bool + colliding pairs, **named**.
- `check_edge(config_a, config_b)` → is the straight-line motion valid, and where it first
  fails. Cheap; the natural pre-filter.
- `validate_path(path_id | waypoints)` → per-waypoint and per-edge validity, min clearance,
  joint-limit violations. **This is the primary consumer of `import_path`** — the tool an
  orchestrator calls on whatever an external RRT produced.
- `check_reachable(link, pose)` → thin convenience over batched IK; answers the pruning
  question without materializing a config the agent won't use.

### Optimization and refinement

- `optimize_path(path_id | waypoints, ...)` → smoothed/refined path via trajopt.
  **Seeded from a caller-supplied path** — this is the primary trajopt entry point, not
  start→goal, because in the intended pipeline the seed comes from an external planner.
- `optimize_between(config_a, config_b)` → the convenience wrapper that seeds trajopt
  itself (Cartesian spline / linear interpolation, per `TrajoptMotionGenerator`). Honest
  framing in the description: a local optimizer that will fail in a maze, not a planner.
  The agent should know when to reach for the RRT server instead.
- `optimize_transport(object, from, to)` → contact-aware refinement via
  `flat_contact_trajopt`.
- `retime(path_id)` → velocity/acceleration-limited time parameterization. **Missing from
  pyroffi today and real work** (trapezoidal, or TOPP-RA-style). Needed for any honest
  duration, and for `simulate` to mean anything.
- `concat_paths([ids])` → one handle, checking that segment *k* ends where *k+1* begins.
  Without this the agent hand-tracks continuity and gets it wrong.

### Dynamics and inspection

- `simulate(path_id)` → rollout via `Robot.step`; divergence, peak torque, final object
  pose. Verify before committing.
- `render_scene(...)` / `render_path(id)` → an image the model can look at. pyroffi already
  has a viser-based viewer; an offscreen still-frame path is a modest add and
  disproportionately improves spatial reasoning.
- `explain_failure(request_id)` → structured cause: joint limit on `panda_joint4`,
  collision at waypoint 23 with `shelf`, cost plateau. Turns a one-shot tool into something
  the agent can iterate against.

### Cheap-before-expensive

Tool descriptions should state relative cost (`check_edge` ~ms, `solve_ik_batch` ~ms,
`optimize_path` ~10 ms warm / ~40 s cold, `simulate` ~100 ms) so the model self-orders:
prune with validation, batch the kinematics, optimize once.

---

## 3. Transport and safety

- Ship as a stdio MCP server (`pyroffi-mcp`) for local agents; HTTP optional later.
- **Compute and simulate only.** No tool commands hardware. If execution is ever added it
  goes behind a separately-gated tool in a different adapter, and that boundary lives in
  the code.
- Sessions are stateful and the agent shares state with itself across calls.

---

## 4. Packaging

```
src/pyroffi/toolbox/
  _session.py      # Session: robot + scene + warm caches + device
  _scene.py        # named objects over fixed-capacity padded arrays, scene_version
  _handles.py      # path/config handle table
  _exchange.py     # import/export: name-keyed joints, frames, scene export
  _retiming.py     # velocity/accel-limited time parameterization   (NEW work)
  _primitives.py   # the operations above, transport-agnostic
src/pyroffi/mcp/
  _server.py       # stdio server, tool registration
  _tools.py        # schema <-> toolbox calls, summary construction
```

Extras: `pyroffi[mcp]` → `mcp`, plus `pillow` if renders are in scope. No new core deps.

---

## 5. Phasing

| Phase | Content |
|---|---|
| 0 | `pyroffi.toolbox`: session, warmup, handle table, padded scene, shape bucketing. Prerequisite — do not start at the server. |
| 1 | MCP server + `create_scene`, scene tools, `solve_ik`, `check_collision`. First demoable slice. |
| 2 | `import_path` / `export_path` / `export_scene`, `validate_path`, `check_edge`. **The interop slice — this is what makes the cross-server TAMP story real, and it should come early, not last.** |
| 3 | `optimize_path` (seeded), `solve_ik_batch`, `concat_paths`. |
| 4 | Retiming, then `simulate` and honest durations. |
| 5 | `explain_failure` plumbed through the primitives. |
| 6 | Renders; `optimize_transport` / contact tools. |

---

## 6. Ranked risks

1. **Interop contract underspecified.** Joint ordering, quaternion convention, and scene
   identity across server boundaries are exactly where a multi-server TAMP stack breaks,
   and the failures are silent. Pin them in the schema in phase 2.
2. **Recompilation on shape change.** Random 40-second stalls unless padding/bucketing is
   enforced at the toolbox boundary from day one.
3. **Context blowout.** If agent-facing responses leak raw arrays, the design fails in a
   way that looks like "the VLM is bad at planning." Enforce the two registers in the
   response schema, not by convention.
4. **Missing time parameterization.** Small, self-contained, easy to overlook; `simulate`
   and any duration estimate depend on it.
5. **GPU contention.** Shared 4×A5000 boxes. A long-lived server pins memory for its
   lifetime: explicit device selection, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, and an
   idle-release story.
6. **float32 / x64 mismatch.** Pin `jax_enable_x64` at process start and report it in
   `create_scene` capabilities. Several solver paths are precision-sensitive (see the
   float32 borderline-test note); silently running float32 gives worse IK than the
   examples do.

**Rough size.** Phases 0–3 are the bulk of the design work but modest code (~1.5–2 k lines).
Phase 4 is small and self-contained. Renders and contact tools are additive.
