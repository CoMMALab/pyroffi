# TAMP experiments for the pyroffi paper

This directory holds the task-and-motion-planning experiments: PDDLStream doing
the symbolic search with **pyroffi as the entire geometric oracle**, plus the
baselines those results are measured against.

The runnable entry points are in `../examples/20_*`. This directory is the
library they import.

## What this is (and is not)

The code here is the *stock* SPaSM solver reimplemented on pyroffi, wired to
PDDLStream. It is deliberately **not** the differentiable-TAMP work: the
bilevel event-graph solver, the diffopt factor graphs, the EBM formulations and
the SMC skeleton discovery from `spasm-pyroffi`'s `diffTAMP` branch are all
excluded, since those belong to a separate paper. What was pulled across:

| pulled | purpose |
|---|---|
| `spasm/backend.py`, `conversions.py`, `util.py` | pyroffi interop: FK, analytic IK, collision spheres |
| `spasm/tetris/`, `spasm/tower/` | the stock SPaSM packing and tower solvers |
| `spasm/extensions/dynamics*.py` | pyroffi inverse/forward dynamics, torque cost, PD rollout |
| `spasm/extensions/dynamic_tower.py` | the dynamics-aware vs kinematic-only trajopt comparison |
| `spasm/tamp/` | the PDDLStream bridge: domains, streams, geometry, motion backends |

One caveat worth knowing: `spasm/tetris/solve.py` also carries a `solve_smc`
function (a resample-move SMC placement solver) that *is* from the newer work.
It is not called by anything here — the experiments use the stock `solve` — but
it was left in place rather than surgically removed, so the file stays
byte-comparable with its origin.

## Setup

```bash
conda activate pyroffi-tamp        # cloned from `pyroffi`, plus robosuite/meshcat/xmltodict
./setup_externals.sh               # clones + builds pddlstream, FastDownward, stock spasm
```

`external/` is gitignored — it holds third-party checkouts (~700 MB), pinned by
`setup_externals.sh`:

- `pddlstream` @ `2c7d6f5` (caelan/pddlstream) with its FastDownward submodule,
  which is compiled during setup
- `spasm_stock` (commalab/spasm) — the original kinematic-only solver, used as
  the baseline in `20_02`

Paths are resolved relative to this directory by `spasm/paths.py`, with
environment-variable overrides (`PYROFFI_ROOT`, `SPASM_STOCK_ROOT`,
`PDDLSTREAM_ROOT`) if you would rather point at existing sibling checkouts.

## The experiments

### `20_00` — PDDLStream with a pyroffi oracle

Tabletop rearrangement on a Panda: N cubes scattered in a start region, packed
collision-free into a goal box. Every geometric query PDDLStream issues — IK,
motion validity, placement collision — is answered by pyroffi. No pybullet, no
simulator in the loop.

Measured on an RTX A5000, `adaptive` algorithm, 2 seeds:

| blocks | solved | median wall | median plan length |
|--:|--:|--:|--:|
| 2 | 2/2 | 1.52 s | 8 |
| 3 | 2/2 | 0.49 s | 13 |
| 5 | 2/2 | 2.44 s | 21 |

Oracle primitive cost: analytic IK **1.14 ms/call**, 20-waypoint motion
validity check **2.35 ms/call**.

### `20_01` — dynamics-aware trajopt (the main claim)

One optimiser, run twice on the 10-block tower task with identical cost terms,
schedule and initialisation. The only difference is pyroffi's differentiable
torque penalty, which back-propagates through inverse dynamics.

| metric | kinematic only | + torque penalty |
|---|--:|--:|
| peak actuator torque | **2665 Nm** | **107 Nm** |
| task cost | 1.249 | 0.083 |
| path length | 77.6 | 67.1 |
| waypoints over torque limit | 10.5 % | 6.0 % |
| mean PD tracking RMS | 0.202 | 0.138 |

Per-joint, the kinematic-only trajectory exceeds the Franka's limits (87 Nm on
joints 1–4, 12 Nm on 5–7) on **all seven joints**, peaking at 1344/1052/486/
2666/657/772/1030 Nm. It is not a rough plan; it is an unexecutable one.

The dynamics-aware run reduces peak torque **24.9×** while *improving* task cost
and shortening the path — so this is not a quality-for-feasibility trade. The
kinematic optimiser was simply unconstrained in a direction that never helped
it. PD tracking error through pyroffi's forward dynamics roughly halves, which
is the independent confirmation.

### `20_02` — stock SPaSM vs the pyroffi backend

The control for `20_01`: the same tetris-packing solver on stock SPaSM's
hand-written kinematics vs. on pyroffi loaded from the same URDF (same 59
collision spheres), same params, same seed. Each side runs in its own
subprocess, since both define a top-level `spasm` package and neither may warm
the other's JIT cache.

Measured (RTX A5000, median of 3 timed calls after a warm-up):

| blocks | stock | pyroffi | ratio | solution cost |
|--:|--:|--:|--:|--:|
| 3 | 3.5 ms | 3.5 ms | 1.01× | 0.2718 both |
| 5 | 13.1 ms | 12.9 ms | 0.98× | 0.4159 both |

Parity in wall-clock, and the solution costs are *identical* — the pyroffi
backend reproduces stock SPaSM's solutions exactly, at the same speed, while
additionally providing the differentiable dynamics `20_01` depends on.

## A negative result worth recording

The `dynamics` motion backend in `spasm/tamp/motion.py` applies the torque
penalty per PDDLStream motion segment. On the **rearrangement** domain it makes
essentially no difference: all backends land at ~37 Nm peak (0.42× of limit),
because those segments are short, obstacle-free connections whose
straight-line initialisation is already smooth, and a minimum-acceleration path
between pinned endpoints *is* that straight line.

The torque blow-up in `20_01` is real but comes from a harder regime — the
tower task's collision-driven optimiser genuinely contorts the path, and
curvature is what costs torque. Do not quote a per-segment rearrangement number
as evidence for the dynamics claim; quote `20_01`.

`motion.py` also exposes `retime()`, which wraps pyroffi's TOPP-RA to compute
the fastest torque-feasible timing an existing path admits. That is a bound a
kinematic pipeline cannot produce, and it is available for reporting, but it is
not what the headline number rests on.
