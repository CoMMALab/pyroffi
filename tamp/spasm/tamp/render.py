"""Offscreen MP4 rendering of a kinematic plan replay.

Follows the standards of ``pyroffi/examples/16_01_single_arm_grasp_transport.py``
(``_record_video``): an actively-stepped MuJoCo sim written offscreen via
``mujoco.Renderer`` + ``imageio`` with a fixed ``MjvCamera``. Headless by default
(``MUJOCO_GL=egl``, set in ``_setup``); no viewer is launched unless asked.
"""
from __future__ import annotations

import numpy as np

from . import _setup  # noqa: F401  (sets MUJOCO_GL=egl)
from .robosuite_bridge import compose_scene, pose4_to_qpos7


def _set_state(model, data, info, arm_q, obj_poses):
    import mujoco
    data.qpos[info["arm_adr"]] = np.asarray(arm_q, float)[:7]
    for name, adr in info["cube_adr"].items():
        data.qpos[adr:adr + 7] = pose4_to_qpos7(np.asarray(obj_poses[name], float))
    mujoco.mj_forward(model, data)


def render_plan(world, frames, path, fps=30, width=1280, height=720,
                seconds_per_frame=0.06):
    """Render a timeline (from ``execute_plan``) to an MP4 at ``path``."""
    import mujoco
    import imageio.v2 as imageio

    model, info = compose_scene(world)
    data = mujoco.MjData(model)

    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)

    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.45, -0.02, 0.05]
    cam.distance = 1.7
    cam.azimuth = 140.0
    cam.elevation = -28.0

    renderer = mujoco.Renderer(model, height=height, width=width)
    writer = imageio.get_writer(path, fps=fps)
    reps = max(1, int(round(seconds_per_frame * fps)))
    try:
        for arm_q, obj_poses in frames:
            _set_state(model, data, info, arm_q, obj_poses)
            renderer.update_scene(data, camera=cam)
            img = renderer.render()
            for _ in range(reps):
                writer.append_data(img)
    finally:
        writer.close()
        renderer.close()
    print(f"Saved video to {path} ({len(frames)} plan frames)")
    return path


def main():
    import argparse
    from .problems import make_rearrange_world, pddlstream_from_world
    from .robosuite_bridge import execute_plan
    from pddlstream.algorithms.meta import solve

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", "--num-blocks", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-time", type=float, default=120.0)
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    world = make_rearrange_world(args.num_blocks, seed=args.seed)
    problem = pddlstream_from_world(world)
    plan, cost, _ = solve(problem, algorithm="adaptive", unit_costs=False,
                          max_time=args.max_time, verbose=False)
    if plan is None:
        raise SystemExit("No plan found — nothing to render.")
    frames, success, _ = execute_plan(world, plan)
    print(f"Plan length {len(plan)}, cost {cost:.2f}, success={success}")
    out = args.out or f"benchmarks/tamp/results/rearrange_n{args.num_blocks}_seed{args.seed}.mp4"
    render_plan(world, frames, out)


if __name__ == "__main__":
    main()
