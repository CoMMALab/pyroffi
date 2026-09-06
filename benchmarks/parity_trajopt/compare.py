"""Score both stacks' trajectories with ONE shared metric (run under `pyroffi`).

Neither stack grades its own homework: collision, endpoint accuracy and
smoothness are recomputed here with pyroffi's FK + collision model on the
returned trajectories. Reports per-problem time (the comparison the user asked
for) alongside quality, so "slower but competitive" is quantified on both axes.
"""
from __future__ import annotations
import pathlib
import numpy as np
import jax, jax.numpy as jnp
import yourdfpy
import pyroffi as pk
from pyroffi.collision import Box, RobotCollisionSpherized
from _problems import (OBSTACLE_CENTER, OBSTACLE_DIMS, CLEARANCE_MARGIN,
                       RESULT_DIR, load)

RESOURCE_ROOT = pathlib.Path(__file__).resolve().parents[2] / "resources"


def main():
    q_start, q_goal, lo, hi = load()
    N, dof = q_start.shape
    urdf = yourdfpy.URDF.load(str(RESOURCE_ROOT / "panda" / "panda_spherized.urdf"))
    robot = pk.Robot.from_urdf(urdf)
    coll = RobotCollisionSpherized.from_urdf(
        urdf, srdf_path=str(RESOURCE_ROOT / "panda" / "panda.srdf"))
    obstacle = Box.from_center_and_dimensions(
        jnp.asarray(OBSTACLE_CENTER, jnp.float32)[None],
        jnp.float32(OBSTACLE_DIMS[0])[None], jnp.float32(OBSTACLE_DIMS[1])[None],
        jnp.float32(OBSTACLE_DIMS[2])[None])
    pyro_names = list(robot.joints.names)[:dof]

    @jax.jit
    def min_clearance_path(traj):  # (T, dof) -> scalar min over path
        dw = coll.compute_world_collision_distance(robot, traj, obstacle).reshape(traj.shape[0], -1)
        ds = coll.compute_self_collision_distance(robot, traj).reshape(traj.shape[0], -1)
        return jnp.minimum(dw.min(), ds.min())

    def reorder(trajs, jn):
        jn = [j if isinstance(j, str) else j.decode() for j in jn]
        if jn[:dof] == pyro_names:
            return trajs
        perm = [jn.index(n) for n in pyro_names]
        return trajs[..., perm]

    def score(name, jn=None):
        f = RESULT_DIR / f"{name}.npz"
        if not f.exists():
            print(f"  [{name}] missing ({f}) -- run its runner"); return
        d = np.load(f, allow_pickle=True)
        trajs = d["trajectories"].astype(np.float32)
        if jn is not None and "joint_names" in d:
            trajs = np.asarray(reorder(trajs, d["joint_names"]))
        tj = jnp.asarray(trajs)
        mins = np.asarray(jax.vmap(min_clearance_path)(tj))
        collfree = float(np.mean(mins >= 0.0)) * 100
        safe = float(np.mean(mins >= CLEARANCE_MARGIN)) * 100
        start_err = np.linalg.norm(trajs[:, 0] - q_start, axis=-1)
        goal_err = np.linalg.norm(trajs[:, -1] - q_goal, axis=-1)
        # path length in joint space (smoothness proxy)
        plen = np.mean(np.sum(np.linalg.norm(np.diff(trajs, axis=1), axis=-1), axis=1))
        ppm = float(d["per_problem_ms"]) if "per_problem_ms" in d else float("nan")
        print(f"  {name:18s}  time={ppm:8.3f} ms/prob   coll-free={collfree:5.1f}%  "
              f"safe(>={CLEARANCE_MARGIN}m)={safe:5.1f}%  goal_err={np.median(goal_err):.4f}rad  "
              f"start_err={np.median(start_err):.4f}  path_len={plen:.2f}")

    print(f"Shared metric on {N} config->config problems (Panda, one cuboid). "
          f"Endpoints collision-free by construction; straight-line grazes obstacle.\n")
    print(f"  {'solver':18s}  {'time':>13s}   quality")
    score("pyroffi_trajopt")
    score("curobo_trajopt", jn=True)
    print("\nNotes: pyroffi time is amortized batch throughput (all problems in one "
          "vmap); cuRobo time is per-problem solve_cspace (best-of-N warm). Both "
          "exclude compile. coll-free uses pyroffi's shared collision model for both.")


if __name__ == "__main__":
    main()
