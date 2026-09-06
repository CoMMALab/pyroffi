"""Solve the shared trajopt problems with cuRobo (run under `curobo`).

cuRobo's TrajOptSolver (default optimizer = lbfgs_bspline_trajopt) via
solve_cspace -- config->config local trajopt, the apples-to-apples analogue of
pyroffi's L-BFGS dynamics_trajopt. Same cuboid world, CUDA graphs ON.

Reports BOTH axes, matching pyroffi:
- batched throughput: one solve_cspace over all N (max_batch_size=N),
- single-problem latency: a max_batch_size=1 solver, best-of-REPS warm.
Trajectories for scoring come from the batched run's interpolated plan.
"""
from __future__ import annotations
import time
import numpy as np
import torch
from curobo._src.solver.solver_trajopt import TrajOptSolver
from curobo._src.solver.solver_trajopt_cfg import TrajOptSolverCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from _problems import (OBSTACLE_CENTER, OBSTACLE_DIMS, T_WAYPOINTS, load, save_result)

REPS = 3


def main():
    q_start, q_goal, lo, hi = load()
    N, dof = q_start.shape
    dev = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    world = {"cuboid": {"obstacle": {
        "dims": [float(d) for d in OBSTACLE_DIMS],
        "pose": [float(OBSTACLE_CENTER[0]), float(OBSTACLE_CENTER[1]),
                 float(OBSTACLE_CENTER[2]), 1.0, 0.0, 0.0, 0.0]}}}

    def make(max_bs):
        cfg = TrajOptSolverCfg.create(
            robot="franka.yml", device_cfg=dev, num_seeds=4, use_cuda_graph=True,
            scene_model=world, self_collision_check=True,
            interpolation_buffer_size=1024, max_batch_size=max_bs)
        return TrajOptSolver(cfg)

    def js(pos, jn):
        return JointState.from_position(
            torch.as_tensor(pos, device=dev.device, dtype=torch.float32), joint_names=jn)

    # ---- batched throughput (timing only; batched interp plan unsupported) ----
    bsolver = make(N); jn = bsolver.joint_names
    print("cuRobo joint_names:", jn)
    cur = js(q_start, jn); goal = js(q_goal, jn)
    bsolver.solve_cspace(goal_state=goal, current_state=cur); torch.cuda.synchronize()
    best_batch = 1e9
    for _ in range(REPS):
        t0 = time.perf_counter()
        bsolver.solve_cspace(goal_state=goal, current_state=cur); torch.cuda.synchronize()
        best_batch = min(best_batch, time.perf_counter() - t0)

    # ---- per-problem: interpolated trajectories (for scoring) + latency ----
    ssolver = make(1); jn1 = ssolver.joint_names
    ssolver.solve_cspace(goal_state=js(q_goal[:1], jn1), current_state=js(q_start[:1], jn1))
    torch.cuda.synchronize()
    trajs, lat = [], []
    for i in range(N):
        c = js(q_start[i:i+1], jn1); g = js(q_goal[i:i+1], jn1)
        t0 = time.perf_counter()
        r = ssolver.solve_cspace(goal_state=g, current_state=c); torch.cuda.synchronize()
        lat.append(time.perf_counter() - t0)
        tr = r.get_interpolated_plan()
        pos = np.asarray(tr.position.detach().cpu()).reshape(-1, len(tr.joint_names))
        arm = [list(tr.joint_names).index(n) for n in jn1]
        pos = pos[:, arm]
        idx = np.linspace(0, len(pos) - 1, int(T_WAYPOINTS)).round().astype(int)
        trajs.append(pos[idx])
    trajs = np.stack(trajs)
    single_ms = float(np.median(lat) * 1e3)

    save_result("curobo_trajopt", trajectories=trajs.astype(np.float64),
                per_problem_ms=np.array(best_batch / N * 1e3),
                single_problem_ms=np.array(single_ms),
                batch_time_s=np.array(best_batch), joint_names=np.array(jn))
    print(f"cuRobo TrajOptSolver: batch({N})={best_batch*1e3:.0f}ms "
          f"=> {best_batch/N*1e3:.2f} ms/prob amortized; "
          f"single-problem latency={single_ms:.1f} ms (CUDA graph, warm)")


if __name__ == "__main__":
    main()
