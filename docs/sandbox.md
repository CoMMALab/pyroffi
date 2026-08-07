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
python pyroffi_endpoint.py serve --gpu 1        # persistent, problem-agnostic
pyroffi-sandbox --task examples/tasks/block_stacking_panda.json --variant wall
```

`python pyroffi_endpoint.py config --gpu 1 [--sandbox-task <task.json>]` prints
the client entry. The sandbox prints a viser URL — **open it**, because
`render()` captures through a connected client and fails without one.

The two servers do not share a lifetime. `pyroffi-mcp` is started once and kept
across problems: its cost is all in the first call (tens of seconds of XLA
compilation, milliseconds thereafter), so restarting it per problem throws away
the only thing that makes it fast. The sandbox is one world for one problem and
dies with it.

That only works if the planning scene is emptied between problems, since a scene
left behind by a dead problem is invisible to the next one and silently makes
its paths invalid. `reset_scene` is the operation: it drops every obstacle,
detaches everything, invalidates all handles and returns the robot to its
default configuration, while keeping the compiled functions (`create_scene`
rebuilds the session and does not). Call it at *both* ends of a problem — on
connect, in case the last one never got to clean up, and on finish. Example 17
does this in a `finally` around the whole problem, including on Ctrl-C and
SIGTERM; an agent driving the endpoint over MCP calls the tool itself.

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

## Picking things up

`attach_object` / `detach_object` move an object between the world and the
robot's body, so a carried block is checked by every collision primitive —
`validate_path` on a transfer reports the payload by name, alongside the robot's
own links. The round trip is exact (1e-8 m), and an object is never both an
obstacle and part of the robot.

The pick sequence that works:

1. **Approach** with `optimize_path` — normal obstacle avoidance.
2. **Descend unoptimized.** A grasp pose ends in contact by definition, so
   trajopt would swerve away from the block you are reaching for. In testing it
   knocked the block aside and the gripper closed on air. Expect
   `validate_path` to report hand-vs-block contact here; that *is* the grasp.
3. `set_gripper("close")`, check it actually caught something.
4. `set_robot_state` to the grasp config, then `attach_object(name,
   ignore_objects=["ground"])`. The target must still be in the scene — attach
   moves it *from* the world, so removing it first makes it unpickable.
5. **Transfer and place**, then `detach_object` to put it back in the world
   where the robot is holding it.

Two things to know:

- **Attachment geometry is a conservative bounding sphere**, and it is wider
  than the object is tall. A 5 cm cube becomes radius 0.0433 (its half-diagonal)
  against a half-height of 0.025, so a block still resting on the table reports
  ~18 mm of ground penetration the moment it is attached, and every legitimate
  lift-off would validate as invalid at its first waypoint. The
  over-approximation itself is deliberate — it never lets a carried object pass
  through an obstacle — but an agent cannot tell that noise apart from a real
  fault, so name the supporting surface in `ignore_objects` and the pair is
  muted. It is muted per *(carried object, obstacle)*: the block's other
  collisions still report, and so do the robot's own collisions with the
  ground.
- **Attaching is a topology change**, so it invalidates the jit cache for
  anything reducing over the collision array: a handful of recompiles per plan,
  not per state.

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
