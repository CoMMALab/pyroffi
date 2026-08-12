"""Generate the shared problem set (run under the `pyroffi` env)."""
from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import yourdfpy

import pyroffi as pk
from pyroffi.collision import Box, RobotCollisionSpherized
from _problems import OBSTACLE_CENTER, OBSTACLE_DIMS, PROBLEM_FILE

N_PROBLEMS = 256
SEED = 0
RESOURCE_ROOT = pathlib.Path(__file__).resolve().parents[2] / "resources"
EE_LINK_NAME = "panda_hand"


def main() -> None:
    urdf = yourdfpy.URDF.load(str(RESOURCE_ROOT / "panda" / "panda_spherized.urdf"))
    robot = pk.Robot.from_urdf(urdf)

    # MUST match cuRobo's franka.yml `ee_link: "panda_hand"`. The URDF's last
    # link is panda_grasptarget, which sits at a fixed offset from panda_hand --
    # using it scored cuRobo at 0.0% pose success while cuRobo was solving
    # correctly for a different frame. A frame mismatch does not look like a
    # frame mismatch in the results; it looks like the other library is broken.
    ee = robot.links.names.index(EE_LINK_NAME)

    # Targets from FK on sampled in-limit configurations: every pose is
    # reachable by construction, so neither solver can score well by failing
    # fast on impossible problems. q_ref is kept only for reference/debugging --
    # it is NOT a ground-truth answer, since IK is many-to-one.
    rng = np.random.default_rng(SEED)
    lo = np.asarray(robot.joints.lower_limits, dtype=np.float64)
    hi = np.asarray(robot.joints.upper_limits, dtype=np.float64)

    # Every target must admit a COLLISION-FREE solution, so q_ref is rejected
    # unless it is itself collision-free. Without this, ~15% of targets could
    # only be reached by passing through the obstacle, and the two stacks were
    # being scored on impossible problems -- which flattered cuRobo precisely
    # BECAUSE it ignores the obstacle: it "succeeded" on the infeasible ones by
    # driving straight through. A benchmark that rewards constraint violation is
    # measuring the wrong thing.
    coll = RobotCollisionSpherized.from_urdf(
        urdf, srdf_path=str(RESOURCE_ROOT / "panda" / "panda.srdf"))
    obstacle = Box.from_center_and_dimensions(
        jnp.asarray(OBSTACLE_CENTER, jnp.float32)[None],
        jnp.float32(OBSTACLE_DIMS[0])[None],
        jnp.float32(OBSTACLE_DIMS[1])[None],
        jnp.float32(OBSTACLE_DIMS[2])[None])

    kept = []
    while len(kept) < N_PROBLEMS:
        batch = lo + (hi - lo) * rng.random((4 * N_PROBLEMS, lo.shape[0]))
        qb = jnp.asarray(batch, jnp.float32)
        w = np.asarray(coll.compute_world_collision_distance(robot, qb, obstacle)
                       ).reshape(len(batch), -1).min(axis=-1)
        s_ = np.asarray(coll.compute_self_collision_distance(robot, qb)).min(axis=-1)
        ok = (w >= 0) & (s_ >= 0)
        kept.extend(batch[ok])
    q_ref = np.asarray(kept[:N_PROBLEMS])

    T = jax.vmap(lambda c: robot.forward_kinematics(c)[ee])(jnp.asarray(q_ref))
    np.savez(
        PROBLEM_FILE,
        q_ref=q_ref,
        target_wxyz_xyz=np.asarray(T, dtype=np.float64),
        ee_link_index=np.array(ee),
        joint_lower=lo,
        joint_upper=hi,
    )
    print(f"wrote {PROBLEM_FILE}: {N_PROBLEMS} reachable targets, "
          f"{lo.shape[0]} DOF, ee = {EE_LINK_NAME} (index {ee}); "
          f"every target admits a collision-free solution")


if __name__ == "__main__":
    main()
