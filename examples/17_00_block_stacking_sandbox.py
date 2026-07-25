"""Block-stacking sandbox: a simulated world an agent solves through MCP.

Two servers, on purpose.

    pyroffi-mcp       plans      IK, collision checks, trajopt, retiming
    pyroffi-sandbox   executes   a stepped MuJoCo world + the viser render layer

The agent orchestrates: it asks the planning server where the arm has to be and
how to get there, then commands the sandbox to actually do it, then looks at
what happened and decides what to do next. Neither server owns the plan, which
is the whole point -- pyroffi is a toolbox of motion primitives, not a planner,
and the sandbox is a world, not a policy.

Nothing is welded or teleported. Blocks are free bodies with mass and real
collision geometry, held by a tendon-driven gripper closing on them. A tower
that gets knocked over falls over, and `report()` reads the simulator, so it
cannot be satisfied by a plan that was never run.

Run it::

    # 1. check the GPU, then start the planning server
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
    pyroffi-mcp --gpu 1 --robot panda_spherized --max-objects 8 --warmup

    # 2. start the sandbox (prints a viser URL -- OPEN IT, render() needs a client)
    pyroffi-sandbox --task examples/tasks/block_stacking_panda.json --variant wall

    # or drive it from Python without MCP, which is what this file does:
    python examples/17_00_block_stacking_sandbox.py --variant wall --demo
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

TASK = os.path.join(os.path.dirname(__file__), "tasks", "block_stacking_panda.json")


def mcp_config(task_path: str, variant: str, gpu: int) -> dict:
    """The MCP client entry that wires an agent to both servers at once."""
    return {
        "mcpServers": {
            "pyroffi": {
                "command": "pyroffi-mcp",
                "args": ["--gpu", str(gpu), "--robot", "panda_spherized",
                         "--max-objects", "8", "--n-timesteps", "32", "--warmup"],
            },
            "pyroffi-sandbox": {
                "command": "pyroffi-sandbox",
                "args": ["--task", os.path.abspath(task_path), "--variant", variant],
            },
        }
    }


def demo(task: dict, variant: str, gpu: int | None) -> int:
    """Pick and place one block, driving both layers directly from Python.

    Not a solution to the task -- it is the shortest path that exercises every
    seam (plan -> validate -> retime -> execute -> observe) so a broken one
    shows up here instead of halfway through an agent's run.
    """
    from pyroffi.sandbox import Sandbox
    from pyroffi.toolbox import Session, Toolbox, configure_process, joint_dict

    configure_process(gpu=gpu, x64=True)
    session = Session(robot=task["robot"], max_objects=8, n_timesteps=32)
    tb = Toolbox(session)

    size = task["block_size_m"]
    for block in task["blocks"]:
        tb.add_object(block["name"], "box", position=block["position"],
                      params={"length": size[0], "width": size[1], "height": size[2]})
    for obs in task["variants"][variant]["obstacles"]:
        tb.add_object(obs["name"], "box", position=obs["position"],
                      wxyz=obs.get("wxyz", (1, 0, 0, 0)), params=obs["params"])

    sandbox = Sandbox(task, variant=variant, realtime=False)
    print(f"viewer: {sandbox.viewer.url}  (open it if you want render() to work)")

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

    # The block you are about to pick up is not an obstacle to picking it up.
    # The planning server has no attach/detach, so the scene bookkeeping is the
    # orchestrator's job: drop it while grasping, put it back where it lands.
    print("descend to the grasp (target block removed from the planning scene):")
    tb.remove_object("block_green")
    q = go(q, q_grasp, "descend", optimize=False)

    print("\nclose the gripper:")
    grip = sandbox.set_gripper("close")
    print(f"  held: {grip['held_block']}  opening {grip['gripper_opening_m']:.4f} m")

    print("lift:")
    go(q, q_above, "lift")
    after = sandbox.observe()
    print(f"\nblock_green is now at "
          f"{after['blocks']['block_green']['position']} (started at {target.tolist()})")
    print(json.dumps(sandbox.report(), indent=2))
    sandbox.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--variant", default="wall")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--demo", action="store_true",
                        help="Run a one-block pick-and-lift through both layers.")
    parser.add_argument("--mcp-config", action="store_true",
                        help="Print the MCP client entry for both servers.")
    args = parser.parse_args()

    with open(args.task) as fh:
        task = json.load(fh)

    if args.mcp_config:
        print(json.dumps(mcp_config(args.task, args.variant, args.gpu or 0), indent=2))
        return 0
    if args.demo:
        return demo(task, args.variant, args.gpu)

    from pyroffi.sandbox import Sandbox

    sandbox = Sandbox(task, variant=args.variant)
    print(f"sandbox up. viewer: {sandbox.viewer.url}")
    print(json.dumps(sandbox.observe(), indent=2))
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        sandbox.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
