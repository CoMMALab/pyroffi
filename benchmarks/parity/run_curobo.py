"""Solve the shared problem set with cuRobo (run under the `curobo` env)."""
from __future__ import annotations

import time

import numpy as np
import torch

from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

from _problems import OBSTACLE_CENTER, OBSTACLE_DIMS, load, save_result

REPS = 5


def main() -> None:
    q_ref, target_wxyz_xyz, _ee = load()
    n = q_ref.shape[0]

    tensor_args = TensorDeviceType()
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]

    # Same obstacle as pyroffi, built from the shared constants rather than
    # restated, so the two stacks cannot silently diverge on the world.
    world = WorldConfig(cuboid=[Cuboid(
        name="obstacle",
        pose=[float(OBSTACLE_CENTER[0]), float(OBSTACLE_CENTER[1]),
              float(OBSTACLE_CENTER[2]), 1.0, 0.0, 0.0, 0.0],
        dims=[float(d) for d in OBSTACLE_DIMS])])

    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg, world,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=32,
        self_collision_check=True,
        self_collision_opt=True,
        use_cuda_graph=True,
        tensor_args=tensor_args,
    )
    solver = IKSolver(ik_config)

    quat = torch.as_tensor(target_wxyz_xyz[:, :4], dtype=torch.float32, device=tensor_args.device)
    pos = torch.as_tensor(target_wxyz_xyz[:, 4:7], dtype=torch.float32, device=tensor_args.device)
    goal = Pose(pos, quat)

    result = solver.solve_batch(goal)          # warm up (CUDA graph capture)
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(REPS):
        t0 = time.perf_counter()
        result = solver.solve_batch(goal)
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)

    cfg = result.solution.squeeze(1).detach().cpu().numpy().astype(np.float64)
    # cuRobo's own verdict is recorded but NOT used for scoring: the comparison
    # recomputes success from the returned configuration with one shared metric,
    # so neither library grades its own homework.
    success = result.success.squeeze(-1).detach().cpu().numpy()

    save_result("curobo", cfg=cfg, seconds=np.array(best),
                n_problems=np.array(n), self_reported_success=success)
    print(f"{'curobo':14} {best*1e3:8.2f} ms   {n/best/1e3:7.1f} kIK/s   "
          f"self-reported success {success.mean()*100:.1f}%")


if __name__ == "__main__":
    main()
