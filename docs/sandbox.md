# The execution sandbox

A simulated world an agent acts on, exposed over MCP. Two servers, and the
split is load-bearing:

| Server | Role | Boundary |
|---|---|---|
| `pyroffi-mcp` | plans — IK, collision, trajopt, retiming | computes, never commands |
| `pyroffi-sandbox` | executes — stepped MuJoCo + the viser render layer | commands a simulation, nothing else |

The planning server's scope states that if execution is ever added it goes
behind a separate adapter and *that boundary lives in the code*. It does:
`pyroffi.sandbox` is a different package with a different entry point, and
`pyroffi.mcp` does not import it.

Neither server owns the plan. The agent decides the order, the grasps and the
recovery; pyroffi answers geometric questions and the sandbox answers "what
actually happened".

## Running it

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
pyroffi-mcp --gpu 1 --robot panda_spherized --max-objects 8 --warmup
pyroffi-sandbox --task examples/tasks/block_stacking_panda.json --variant wall
```

`python examples/17_00_block_stacking_sandbox.py --mcp-config` prints the client
entry wiring both at once. The sandbox prints a viser URL — **open it**, because
`render()` captures through a connected client and fails without one.

## Tools

| Tool | Cost | Notes |
|---|---|---|
| `get_task` | free | blocks, goal, grasp convention, tolerances |
| `observe` | free | the simulator's actual state — ground truth, not your plan |
| `render` | ~1 s | PNG through viser; needs a browser client |
| `execute_path` | the trajectory's duration | tracks a joint path, reports what the arm did |
| `set_gripper` | ~1 s | open/close; reports whether anything was actually caught |
| `reset` | ~1 s | back to the start, clears the history |
| `report` | free | success + cost metrics, read from the simulator |

## Two robot models, and why that is sound

Planning runs on pyroffi's `panda_spherized` (7 DOF, primitive collision
geometry, the model whose self-collision calibration is reliable). Execution
runs on the MuJoCo Menagerie Franka, which has real finger meshes and a
tendon-driven gripper.

That only works because the kinematics agree: measured over random
configurations, `panda_hand` and the Menagerie `hand` body differ by **1e-7 m**
in position and 1e-7 in quaternion, so a joint-space plan transfers with no
translation layer.

The spherized URDF is not the execution model because it *cannot* be: its
fingers are fixed joints, and its finger collision spheres have 3.8–4.3 cm radii
that interpenetrate each other at every opening. It cannot express a pinch. The
`panda_fin_spherized` variant does have prismatic fingers, but its mimic joint
makes pyroffi drop dynamics support entirely and its finger geometry is the same
coarse blob.

## What the sandbox is honest about

**Nothing is welded or teleported.** Blocks are free bodies with mass and real
collision geometry, held by a gripper closing on them. A tower that gets knocked
over falls over. `report()` reads the simulator, so it cannot be satisfied by a
plan that was never executed.

**Motions start from where the arm actually is.** `execute_path` rejects a path
whose first waypoint is more than 0.05 rad from the current configuration.

**Timing matters, measurably.** Pass `times_s` from `retime()`. Without it the
path runs at a fixed joint speed, and a reference the arm cannot follow pulls a
held block out of the gripper — that is measured behaviour, not a warning
written defensively.

**Closing the gripper is a command, not an outcome.** `set_gripper("close")`
reports `success: false` when it closed on nothing.

## Two traps the task is built around

Both are real gaps in the toolbox, stated rather than designed around.

**1. The block you are picking up is not an obstacle to picking it up.** The
planning server has no `attach_object`/`detach_object`, so scene bookkeeping is
the orchestrator's job: `remove_object` before descending onto a grasp, and
`add_object` at the new pose after releasing. Skipping this makes trajopt swerve
away from the very thing you are reaching for — in testing it knocked the block
aside and the gripper closed on air. A grasp pose is *supposed* to end in
contact, which is also why the final descent should not be run through
`optimize_path` at all.

**2. `validate_path` checks the robot, not the payload.** The server does not
know a block is in the gripper, so a transfer that sweeps the held cube through
the divider validates clean. The agent has to check the carried object itself.

## The task

Three 5 cm cubes, stack them red → green → blue on the red block's footprint.
The `wall` variant puts a 25 cm divider between `block_blue` and the rest of the
workspace: a straight-line joint-space transfer drives through it, so
`optimize_between` returns something invalid and says so — the seed has to lift
over.

The divider sits 0.15 m out because the spherized hand is a coarse collision
blob and fouls it from further away than the visual mesh suggests.

## Smoke test

```bash
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl \
  python examples/17_00_block_stacking_sandbox.py --variant wall --demo
```

Picks up one block through the full stack — IK → optimize → validate → retime →
execute → observe. It reports `success: false` on purpose: it exercises every
seam, it does not build the tower.
