# The pyroffi MCP server

Implementation notes for the design in [`mcp_scope.md`](mcp_scope.md). That
document is the scope; this one records what was built, what the numbers
actually are, and the four places where reality pushed back on the design.

```
   pyroffi.mcp        _server.py   stdio server, session lifecycle, CLI
                      _tools.py    tool schemas + the descriptions the model reads
        │
   pyroffi.toolbox    _session.py  robot + collision model + device + warm caches
                      _scene.py    named objects over fixed-capacity padded arrays
                      _handles.py  config / path / trajectory handle table
                      _exchange.py the interop contract
                      _retiming.py velocity/acceleration-limited timing
                      _primitives.py the operations, transport-agnostic
        │
   pyroffi core       unchanged
```

`mcp` is imported only under `pyroffi.mcp`. The toolbox has no new dependencies.

## Running it

```bash
pip install -e '.[mcp]'

# Pick a free GPU: a long-lived server pins its device memory for its lifetime.
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
pyroffi-mcp --gpu 1 --robot panda_spherized --max-objects 16 --warmup
```

`--warmup` pays the XLA compile up front (~40 s) so no client call does. Device
and `jax_enable_x64` are pinned before JAX initialises its backend, which is why
they are launch flags rather than tool arguments.

As an MCP client entry:

```json
{"mcpServers": {"pyroffi": {
  "command": "pyroffi-mcp",
  "args": ["--gpu", "1", "--robot", "panda_spherized", "--warmup"]
}}}
```

## Measured costs (Panda, 7 DOF, RTX A5000, float64)

| Operation | Warm | First call |
|---|---|---|
| `check_collision` | 1.9 ms | ~1.7 s |
| `check_edge` (32 samples) | 2.8 ms | ~1.1 s |
| `validate_path` (32 waypoints, 4 substeps) | 8.3 ms | ~0.8 s |
| `solve_ik` (64 seeds) | ~170 ms | ~6 s |
| `solve_ik_batch`, 1 / 8 / 32 targets | 160 ms / 410 ms / 1.2 s | ~3 s |
| `optimize_path` (32 wp, batch 16, defaults) | ~3.6 s | ~8 s |
| `retime` | <1 ms | — (CPU, closed form) |
| `simulate` (≈500 control ticks) | ~300 ms | ~2 s |
| `warmup` (all of the above) | — | ~41 s |

Every response carries `compiled`, so an agent can tell an 8-second answer from
a 3.6-second one instead of concluding the server is slow.

Two of these corrected assumptions in the scope:

- **`optimize_path` is seconds, not milliseconds.** The scope estimated ~10 ms
  warm. SCO trajopt cost is `n_outer_iters × n_inner_iters`, and the defaults
  copied from `examples/07_00` (50×100) ran 11.7 s. On every obstacle case
  tested, 50×100 reached *exactly* the same final clearance (+0.0087 m) as 5×15
  at 0.7 s, so the defaults are now 20×50 (~3.6 s) with the knobs exposed.
- **Batched IK is cheaper per target, not free.** 8 targets cost ~2.6× one
  target rather than 8×, i.e. ~160 ms/target down to ~51 ms/target. Worth
  preferring, but the tool description no longer claims it is nearly free.

## Four things the design did not anticipate

These are the places where the scope's assumptions met the library, and all four
are load-bearing enough to state explicitly.

### 1. Collision batching goes through `vmap`, not leading dimensions

`RobotCollision.at_config` transforms a fixed `(n_links,)` geometry pytree in
place, and `jax_dataclasses` asserts the shape is preserved — so a leading batch
dimension raises rather than batching. Every batched query in `_primitives.py`
is `jax.jit(jax.vmap(single, in_axes=(0, None)))`, with world geometry passed as
a runtime argument so scene edits never retrace.

### 2. The default collision model reports collisions that are not real

`RobotCollision.from_urdf` fits one capsule per link. On the Panda that geometry
is coarse enough that **100 %** of uniformly sampled configurations report a
self-collision, and four link pairs are in collision in *every* configuration —
artifacts of the model, not of any configuration. Left alone, `collision_free`
is false forever and the entire validation half of the toolbox is worthless.

Two mitigations, both reported rather than silent:

- **Empirical pair pruning** at session creation (`_calibrate_self_collision`):
  sample 512 configurations, disable pairs colliding in >99 % of them. This is
  the empirical version of an SRDF's `disable_collisions`, and it is what
  MoveIt's setup assistant does. One batched GPU call.
- **Model selection.** `RobotCollisionSpherized` is far more faithful but needs
  primitive (non-mesh) collision geometry, so `collision_model="auto"` tries it
  and falls back to capsules, saying which it got.

The residual false-positive rate is then *measured* and surfaced in
`get_capabilities` as `self_collision_calibration.reliable`:

| Robot | Model | Pairs pruned | Random configs still self-colliding | Reliable |
|---|---|---|---|---|
| `panda_spherized` | spherized | 1 of 66 | 10 % | yes |
| `panda` (mesh URDF) | capsule | 4 of 66 | 99.8 % | **no** |

Hence the server default of `panda_spherized`. On a mesh URDF the tool still
runs and still answers, but says its self-collision answers are advisory.

### 3. Statically-posed links are excluded from world collision

The Panda's base link intersects the ground plane by construction — the mounting
plate sits at z=0. Reporting that makes every configuration collide with the
floor, and no motion the agent chooses could ever clear it. Links whose world
pose does not depend on the configuration (detected by FK invariance) are
excluded from world-collision reporting, and listed in `get_capabilities`. An
object that *does* intersect one is flagged at `add_object` time, since that is
a scene-authoring mistake rather than a motion decision.

### 4. `simulate` needed a control rate derived from the mass matrix

Torque passed to `Robot.step` is held constant across its `substeps`, so closing
a PD loop once per waypoint samples the velocity feedback at the waypoint
period. With the Panda's smallest mass-matrix eigenvalue (λ_min ≈ 0.10 kg m²),
`kd·dt/λ_min` exceeds 2 at any realistic waypoint spacing and the rollout
diverges — a sampled-data instability in the *controller*, not in the trajectory
or the integrator. (The dynamics were verified independently: forward/inverse
round-trip agrees to 1e-14, and semi-implicit Euler and RK4 agree on energy.)

`simulate` therefore runs the controller at `dt/substeps` with `substeps`
derived from λ_min, and adds inverse-dynamics feedforward so the PD term only
corrects residuals. Measured on a 24-waypoint path: dt=0.078 s and 0.019 s both
diverge, 0.005 s is stable with 2.8 mrad mean tracking error and 35 N·m peak
torque. Feedforward improves max tracking error roughly tenfold.

## Retiming

Deliberately a **uniform timestep** at the binding constraint rather than
TOPP-RA, in closed form with no iteration. A per-segment schedule is faster, but
the natural discretisation (constant-velocity segments, averaged spans) admits
*zigzag* solutions — alternating fast and slow segments that score as feasible
because the metric averages across them. Iterative stretching converges to
exactly those, and duration then stops being monotone in the limits: during
development, tightening the velocity ceiling from 1.0× to 0.75× returned a
trajectory that was *shorter* (3.9 s → 1.7 s).

A uniform step cannot express a zigzag and is monotone in both limits by
construction. The cost is roughly 2–3× a time-optimal schedule, which is stated
in the tool description. Replacing it with TOPP-RA would not change the
interface.

## What is not implemented

- **`optimize_transport`** is wired to `flat_contact_trajopt` but requires the
  GRiD CUDA dynamics backend, which is built out-of-tree; it raises a clear
  error when unavailable and is not covered by the test suite here.
- **`render_scene`** uses trimesh's offscreen rasteriser and returns
  `renderer_unavailable` on a headless host without a GL context, rather than
  crashing.
- **HTTP transport.** stdio only, per the scope.
- **No tool commands hardware.** Compute and simulate only.

## Tests

```bash
pytest tests/test_toolbox_units.py -q                       # CPU, ~3 s
CUDA_VISIBLE_DEVICES=1 pytest tests/test_toolbox_integration.py -q   # GPU, ~2 min
CUDA_VISIBLE_DEVICES=1 pytest tests/test_mcp_server.py -q            # GPU, real stdio client
```

The integration suite asserts the headline pipeline end to end: a straight-line
seed that drives through a wall is rejected by `validate_path`, repaired by
`optimize_path` (min clearance −0.029 m → +0.009 m), and confirmed valid on
re-validation.
