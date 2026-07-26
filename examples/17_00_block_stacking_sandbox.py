"""Block-stacking sandbox: a simulated world an agent solves through MCP.

Two servers, on purpose, with two different lifetimes.

    pyroffi-mcp       plans      IK, collision checks, trajopt, retiming
    pyroffi-sandbox   executes   a stepped MuJoCo world + the viser render layer

The agent orchestrates: it asks the planning server where the arm has to be and
how to get there, then commands the sandbox to actually do it, then looks at
what happened and decides what to do next. Neither server owns the plan, which
is the whole point -- pyroffi is a toolbox of motion primitives, not a planner,
and the sandbox is a world, not a policy.

This file is the *problem* half and owns nothing else. The endpoint lives in
``pyroffi_endpoint.py`` at the repo root, comes up empty -- a robot, a capacity,
a ground plane -- and knows nothing about block stacking; everything here is
poured into it on the way in and taken back out on the way out
(``Toolbox.reset_scene``, which keeps the warm session and its compiled
functions; ``create_scene`` would throw them away and pay the cold-start compile
again). That asymmetry is what makes the endpoint persistent: the scene is the
only per-problem state, so the scene is the only thing that has to be wiped.
Anything left in it is invisible to the next problem and silently makes its
paths invalid, so this script resets at *both* ends of its life -- on the way
in, in case a previous problem died badly, and in a ``finally`` on the way out.

Nothing is welded or teleported. Blocks are free bodies with mass and real
collision geometry, held by a tendon-driven gripper closing on them. A tower
that gets knocked over falls over, and `report()` reads the simulator, so it
cannot be satisfied by a plan that was never run.

Run it::

    # 1. the planning endpoint, problem-agnostic and long-lived -- start it once
    #    and leave it up across problems (see pyroffi_endpoint.py)
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
    python pyroffi_endpoint.py serve --gpu 1

    # 2. the sandbox -- this one IS per-problem and dies with it
    #    (prints a viser URL -- OPEN IT, render() needs a client)
    python examples/17_00_block_stacking_sandbox.py --variant wall

    # or drive both layers from Python without MCP:
    python examples/17_00_block_stacking_sandbox.py --variant wall --demo
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import sys
import time

import numpy as np

TASK = os.path.join(os.path.dirname(__file__), "tasks", "block_stacking_panda.json")

# The endpoint is a root-level script, not part of this example, and is imported
# rather than reimplemented so there is exactly one definition of what the
# planning API is configured with.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyroffi_endpoint import open_endpoint  # noqa: E402


@contextlib.contextmanager
def problem_scene(tb, task: dict, variant: str, realtime: bool = False):
    """Own one problem's lifetime on an endpoint that outlives it.

    Loads the task into the planning scene, brings up the sandbox, and -- however
    this block is left, including Ctrl-C, an exception, or a ``kill`` -- tears
    the sandbox down and resets the endpoint to empty. The reset also happens on
    the way *in*, because the failure this guards against is precisely the one
    where the previous problem never got to run its cleanup.

    SIGTERM is converted to an exception rather than left alone: its default
    disposition kills the process outright, ``finally`` blocks do not run, and
    the endpoint is left holding a dead problem's obstacles.
    """
    from pyroffi.sandbox import Sandbox

    def _terminated(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    previous = signal.signal(signal.SIGTERM, _terminated)

    wiped = tb.reset_scene()
    if wiped["removed_objects"] or wiped["detached"]:
        print(f"endpoint was not clean; wiped {wiped['removed_objects']} "
              f"and detached {wiped['detached']}")

    size = task["block_size_m"]
    for block in task["blocks"]:
        tb.add_object(block["name"], "box", position=block["position"],
                      params={"length": size[0], "width": size[1], "height": size[2]})
    for obs in task["variants"][variant]["obstacles"]:
        tb.add_object(obs["name"], "box", position=obs["position"],
                      wxyz=obs.get("wxyz", (1, 0, 0, 0)), params=obs["params"])

    sandbox = Sandbox(task, variant=variant, realtime=realtime)
    print(f"viewer: {sandbox.viewer.url}  (open it if you want render() to work)")
    try:
        yield sandbox
    finally:
        signal.signal(signal.SIGTERM, previous)
        sandbox.close()
        left = tb.reset_scene()
        print(f"endpoint reset: removed {left['removed_objects']}, "
              f"detached {left['detached']} -- ready for a new problem")


# ── the problem: block stacking, which the endpoint learns only by being told ─


def demo(task: dict, variant: str, gpu: int | None) -> int:
    """Pick and place one block, driving both layers directly from Python.

    Not a solution to the task -- it is the shortest path that exercises every
    seam (plan -> validate -> retime -> execute -> observe) so a broken one
    shows up here instead of halfway through an agent's run.
    """
    tb = open_endpoint(gpu, robot=task["robot"])
    with problem_scene(tb, task, variant) as sandbox:
        return _pick_and_lift(tb, sandbox, task)


def _pick_and_lift(tb, sandbox, task: dict) -> int:
    from pyroffi.toolbox import joint_dict

    session = tb.session

    standoff = task["robot_setup"]["grasp_standoff_m"]
    target = np.asarray(task["blocks"][1]["position"], dtype=float)   # block_green
    obs = sandbox.observe()
    q_now = np.array([obs["joint_values"][n] for n in session.joint_names])

    def ik(position, seed):
        res = tb.solve_ik(
            pose={"position": [float(v) for v in position], "wxyz": [0, 0, 1, 0]},
            num_seeds=64, num_restarts=3, seed_config=joint_dict(seed, session.joint_names),
        )
        print(f"  ik {np.round(position, 3).tolist()}: ok={res['success']} "
              f"err={res['pos_error_m']:.5f}")
        if not res["success"]:
            raise SystemExit(f"IK failed: {tb.explain_failure(res['request_id'])}")
        return session.handles.get(res["config_id"]).values

    q_above = ik(target + [0, 0, standoff + 0.15], q_now)
    q_grasp = ik(target + [0, 0, standoff], q_above)

    def go(q_from, q_to, label, optimize=True):
        """Plan one leg the way an agent would: optimize, validate, retime, run.

        ``optimize=False`` is for the final descent onto a grasp. A grasp pose
        is *supposed* to end in contact, so running trajopt on it asks the
        optimizer to push the arm away from the very thing it is reaching for —
        which it does, by swerving sideways and knocking the block over.
        """
        seed = np.linspace(q_from, q_to, 32)
        if optimize:
            opt = tb.optimize_path([joint_dict(r, session.joint_names) for r in seed])
            path = session.handles.get(opt["path_id"]).values
            handle = opt["path_id"]
        else:
            imported = tb.import_path(
                [joint_dict(r, session.joint_names) for r in seed], source="approach"
            )
            path, handle = seed, imported["path_id"]
        val = tb.validate_path(handle)
        timing = tb.retime(handle, velocity_scale=0.3)
        times = session.handles.get(timing["trajectory_id"]).times
        print(f"  {label}: valid={val['valid']} clearance={val['min_clearance_m']:.4f} "
              f"duration={timing['duration_s']:.2f}s")
        run = sandbox.execute_path(
            [joint_dict(r, session.joint_names) for r in path], times_s=times.tolist()
        )
        print(f"    executed: tracking err {run['max_tracking_error_rad']:.4f} rad")
        return path[-1]

    print("\nreach over the block:")
    q = go(q_now, q_above, "approach")

    # The descent is NOT optimized. A grasp pose ends in contact by definition,
    # so trajopt would swerve away from the very block being reached for. The
    # target also stays in the scene: attach_object moves it from the world onto
    # the robot, so it has to still be in the world to be picked up. Expect
    # validate_path to report hand-vs-block contact here -- that is the grasp.
    print("descend to the grasp (unoptimized; contact is the point):")
    q = go(q, q_grasp, "descend", optimize=False)

    print("\nclose the gripper:")
    grip = sandbox.set_gripper("close")
    print(f"  held: {grip['held_block']}  opening {grip['gripper_opening_m']:.4f} m")

    # Now that it is genuinely held, hand it to the planner as part of the robot.
    tb.set_robot_state(joint_dict(q, session.joint_names))
    # ignore_objects=["ground"]: the bounding sphere of a 5 cm cube has a
    # 0.0433 m radius against a 0.025 m half-height, so a block still sitting on
    # the table overlaps it the instant it is attached. Without this the lift
    # that resolves it validates as invalid, and a real fault would be
    # indistinguishable from that noise.
    att = tb.attach_object("block_green", ignore_objects=["ground"])
    print(f"  attached: bounding radius {att['bounding_radius']:.4f} m, "
          f"in_dynamics={att['in_dynamics']}")

    print("lift (the carried block is now validated too):")
    go(q, q_above, "lift")
    print(f"  still attached: {[a['name'] for a in tb.list_attachments()['attachments']]}")
    after = sandbox.observe()
    print(f"\nblock_green is now at "
          f"{after['blocks']['block_green']['position']} (started at {target.tolist()})")
    print(json.dumps(sandbox.report(), indent=2))
    # No teardown here: problem_scene closes the sandbox and wipes the endpoint,
    # including on the paths where this function raises instead of returning.
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--variant", default="wall")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--demo", action="store_true",
                        help="Run a one-block pick-and-lift through both layers.")
    # No --mcp-config here: the client config is the endpoint's business and
    # lives in `python pyroffi_endpoint.py config`.
    parser.add_argument("--plan-here", action="store_true",
                        help="Also open a planning endpoint in this process and load "
                             "the problem into it, wiping it again on exit.")
    args = parser.parse_args()

    with open(args.task) as fh:
        task = json.load(fh)

    if args.demo:
        return demo(task, args.variant, args.gpu)

    # Host the problem for someone else to solve: an agent on the persistent
    # planning endpoint, or a notebook. The endpoint is only loaded here if it
    # lives in this process (--plan-here). When the agent is talking to a
    # separate pyroffi-mcp, that server's scene is not reachable from here, and
    # pretending otherwise would be worse than saying so -- the agent bookends
    # the problem with reset_scene itself.
    if args.plan_here:
        tb = open_endpoint(args.gpu, robot=task["robot"])
        with problem_scene(tb, task, args.variant, realtime=True) as sandbox:
            print(f"sandbox up, endpoint loaded. viewer: {sandbox.viewer.url}")
            print(json.dumps(sandbox.observe(), indent=2))
            _wait_for_interrupt()
        return 0

    from pyroffi.sandbox import Sandbox

    sandbox = Sandbox(task, variant=args.variant)
    print(f"sandbox up. viewer: {sandbox.viewer.url}")
    print("the planning endpoint is a separate process and is NOT loaded from here:\n"
          "  on connect, call reset_scene then add_object for each block/obstacle\n"
          "  on finish, call reset_scene again to hand it back empty")
    print(json.dumps(sandbox.observe(), indent=2))
    try:
        _wait_for_interrupt()
    finally:
        sandbox.close()
    return 0


def _wait_for_interrupt() -> None:
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
