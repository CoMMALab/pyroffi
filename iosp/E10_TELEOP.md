# E10 — IOSP on human teleop demonstrations

Status as of 2026-09-02. Read this before touching `fit/teleop.py`,
`model/fr3.py`, `experiments/e10_teleop.py`, or the anchoring paths in
`model/pickplace.py`.

Every other iosp experiment fits a demonstration this model generated, at a
known `theta_star`. E10 removes that: the demonstrations are teleoperated
pick-and-place episodes recorded through a GELLO leader into the MuJoCo scene in
`../sim_teleop`, and no vector of weights generated them. So the claim under
test is behavioural — fitted on some episodes, does the composed planner
reproduce others — and every parameter-recovery metric is undefined.

## Data and provenance

Ten episodes in `../sim_teleop/data/demos/ep_*/`, each carrying `state.jsonl`
(the schema `pickplace.iosp_export` parses), `episode.npz` (velocities, torques
— recorded but NOT read by the fit) and `factors.json` (the randomisation
record, without which an episode can be replayed but not fitted).

`sim_teleop/pickplace/iosp_export.py` collapses each episode to an
`(N_FULL, 7)` waypoint path. It imports `N_FULL`/`PHASE_SPAN` from THIS package
rather than restating them, so the phase layout cannot drift. Do not vendor a
copy of it here.

**Robot: FR3, not Panda.** The episodes were recorded on an FR3 and every
synthetic iosp result used `resources/panda/panda_spherized.urdf`. Fitting one
through the other charges the kinematic mismatch to the cost weights.
`model/fr3.py` derives a 7-DOF FR3 (finger joints welded open) from
`resources/fr3/fr3_spherized.urdf`; pyroffi's FK on it was checked against the
recorded MuJoCo `ee_pos` and agrees to **1.3e-4 m**.

## What was wrong, and what fixed it

The first full run reconstructed poorly (0.162 m EE RMSE). The cause was not the
cost model but the **IK seed**, and specifically its TARGET ORIENTATION.

`||q_seed - q_demo||` at the pinned skeleton rows, against a 0.938 rad approach
motion — i.e. how much of the whole reach the seed is already wrong by before
trajopt starts:

| IK target | q_pick | q_place |
| --- | --- | --- |
| fixed `DOWN_WXYZ` (every run before this) | 0.618 | — |
| `DOWN * cube_yaw` | 0.530 | — |
| `DOWN`, + fitted in-plane release offsets | 0.490 | 0.714 |
| fixed `DOWN`, unfixed release | — | 1.290 |
| **anchored orientation + seed** | **0.091** | **0.001** |

The solver was never at fault: it hits whatever target it is given to
**0.35 mm / 0.02 deg**. It was solving for a grasp the demonstrator never used.

Three things that cost time and are worth not rediscovering:

1. **Seeding alone is nearly useless** (0.530 -> 0.504). `previous_cfg` selects
   among solutions to the target POSE; it slides `q_pick` along the self-motion
   manifold and cannot repair a wrong orientation. Anchor the orientation; the
   seed is a second-order refinement on top (0.146 -> 0.084).
2. **Null-space is not the problem.** Given an exactly correct 6-DOF pose the
   solver recovers the human's configuration to 0.116 rad from `q_start`, and to
   0.001 rad seeded from the demo itself.
3. **A global tilt offset does not work** (23.5 -> 20.9 deg). The systematic
   part of the residual orientation is small; the scatter dominates (per-axis
   std up to 24 deg). Grasp orientation is per-episode operator choice, not
   something the scene predicts.

## The two model changes

**Anchoring** (`PickPlaceScene.pick_wxyz` / `grasp_ref` / `place_wxyz` /
`place_ref`, all optional, all `None` on synthetic scenes). `grasp_ik` resolves
orientation in three tiers — anchored, else `DOWN * pick_yaw` from the recorded
`cube_yaw`, else fixed `DOWN`. `place_ik` has only two tiers: the bucket is an
n-gon approximating a cylinder and has no yaw to predict from.

*This changes the claim, not just the number.* With `pick_wxyz` set the pipeline
no longer predicts HOW to grasp — the grasp pose is an input. On held-out
episodes that is demonstration information at test time, so a generalization
result reads "given this episode's grasp, the fitted cost reproduces the
motion". That is a normal skeleton-given formulation and it isolates the
trajectory cost from grasp selection, but it is NOT the claim the synthetic
experiments make. Say so explicitly in any writeup.

**`theta_ik` widened K=2 -> K=4**: `place.radial`, `place.tangential`, in the
scene's base->bucket frame (`pp._place_frame`), so a fitted offset transfers to
a bucket elsewhere. Motivation: all ten operators released **6.3 cm short of the
bucket centre toward the arm base** (std 1.8 cm, bucket inner radius 6.5 cm) —
they let go over the NEAR RIM. A `+z` standoff is perpendicular to that and
cannot express it. Fitting one global pair cuts release position error
0.068 -> 0.022 m. `THETA_IK_STAR` gained two zeros, so synthetic demonstrators
still release on the bucket axis and recorded synthetic results are unchanged.

## Measured effect

One episode, 12 candidates (4 branches x 3 starts), 40 steps, two seeds, fit set,
`space="joint"`:

| | joint RMSE (u=0 -> winner) | EE RMSE (u=0 -> winner) |
| --- | --- | --- |
| plain | 1.034 -> **0.359** rad | 0.614 -> 0.088 m |
| anchored | 0.944 -> **0.230** rad | 0.567 -> 0.103 m |

Anchoring is worth **36% of the fitted objective**, with a seed-to-seed spread
of 0.001-0.003 rad against a 0.130 rad effect — decisively resolved by two seeds.

**EE position is flat-to-slightly-worse, and that is expected**: EE position
cannot see grasp orientation or elbow, so anchoring had little to gain there,
and constraining the endpoints spends freedom that previously went into that
unoptimized metric. **Report the joint-space number.** An EE-only headline makes
this change look like nothing.

## Gotchas

**The fit does not call `grasp_ik`/`place_ik`.**
`multistart.build_from_demos.batched_paths` calls `pp._ik_batch` directly,
because it folds the candidate axis into the kernel's problem batch. Anything
added to the IK stage must be mirrored there or it is **silently inert inside
the fit**. Anchoring was inert for two full comparison runs; the tell was an
anchored and an unanchored run agreeing to 0.9%. Verify with a rollout diff
through `batched_paths` itself (`scratch/e10_check_live.py`), never by testing
`grasp_ik` — that tests a path the fit does not use.

**Memory.** The batch axis is candidates x episodes and the refine stage's
collision Jacobian is dense in it. Budget **~0.35 GiB per row**. Measured on a
24 GiB A5000 at 12 candidates x 10 episodes: 120 rows died in AUTOTUNING (not
execution), 40 rows died in execution at one 14.19 GiB allocation, 10 rows ran.
Use `multistart.run(chunk=...)`; it is exact, since each candidate's loss
depends only on its own row of `U`.

**Do not read instability off the loss history.** The summed loss is over all
candidates and is dominated by stuck branches — it oscillated 28-31 while the
selected winner was reproducible to three decimals. Seeds differ 3-6x in summed
loss and <1% in the winner.

**Branch choice still dominates candidate quality**, anchored or not: per-branch
held-out RMSE spread 4.3x on the full run, and branch 1 sits at ~2.8 rad while
branch 0 sits at ~0.23. `grasp_ik_branched`/`place_ik_branched` and the
multistart map therefore keep `refs`/`q_pick` as SEEDS even when the scene
carries `grasp_ref`/`place_ref` — a shared anchor would collapse the branch axis
that multistart exists to cover. Orientation is anchored there; the seed is not.

## Running it

```bash
CUDA_VISIBLE_DEVICES=<idx> XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python -m iosp.experiments.e10_teleop --mode multistart --chunk 1 \
        --out iosp/data/e10_teleop.json
```

`--mode multistart` (default) is the three-stage forward model with B branches x
S starts and one selection at the end on TRAINING loss. `--mode procedure` is
E3's five-stage identifiable refit (single-start, two-stage), which answers the
different question of how many cost directions ten demonstrations resolve.
`--mode both` runs both. `--n-fit` splits chronologically, default 8/2.

`anchor_grasp` is **off by default** and is currently only reachable through
`multistart.build_from_demos(anchor_grasp=True)` — it is not yet an
`e10_teleop.py` flag.

## State and next steps

`iosp/data/e10_teleop.json` is **stale**: it is the original unanchored K=13
8-episode run (0.162 m EE). Everything since has been single-episode scratch
runs under `scratch/e10_*.py`, logs in `iosp/data/logs/`.

1. Re-run full E10 with anchoring on (~50 min). Per-episode averaging damps the
   basin lottery that made single-episode runs noisy. Headline the joint number.
2. Then the cost model. With seeds contributing ~0.001-0.09 rad, reconstruction
   is still 0.230 rad / 0.10 m on ONE episode, on the FIT set, with K=15. The
   seed can no longer be blamed; the residual is the trajectory cost family.

Two things the recorded data cannot answer, by construction: `clearance` is
unidentifiable (the recording scene has no obstacle and the exporter emits a
constant placeholder — it should land in the Gram's null space, and did), and
`upright` is weakly excited (a human keeps the gripper roughly down throughout).
