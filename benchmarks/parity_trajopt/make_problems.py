"""Generate the shared trajopt problem set (run under `pyroffi`).

Each problem = (q_start, q_goal): both collision-free, and the straight-line
interpolation between them PASSES NEAR the obstacle (min clearance along the
lerp below a band), so local collision-avoidance actually matters -- but a
collision-free detour is known to exist because we require the lerp to be only
mildly blocked, not deeply trapped. That keeps the task in the regime a LOCAL
trajopt (both stacks' inner solver) can solve, so the comparison is about speed
and quality, not global planning.
"""
from __future__ import annotations
import pathlib
import jax, jax.numpy as jnp, numpy as np
import yourdfpy
import pyroffi as pk
from pyroffi.collision import Box, RobotCollisionSpherized
from _problems import OBSTACLE_CENTER, OBSTACLE_DIMS, PROBLEM_FILE, T_WAYPOINTS

N_PROBLEMS = 128
SEED = 0
RESOURCE_ROOT = pathlib.Path(__file__).resolve().parents[2] / "resources"


def main():
    urdf = yourdfpy.URDF.load(str(RESOURCE_ROOT / "panda" / "panda_spherized.urdf"))
    robot = pk.Robot.from_urdf(urdf)
    coll = RobotCollisionSpherized.from_urdf(
        urdf, srdf_path=str(RESOURCE_ROOT / "panda" / "panda.srdf"))
    obstacle = Box.from_center_and_dimensions(
        jnp.asarray(OBSTACLE_CENTER, jnp.float32)[None],
        jnp.float32(OBSTACLE_DIMS[0])[None], jnp.float32(OBSTACLE_DIMS[1])[None],
        jnp.float32(OBSTACLE_DIMS[2])[None])
    lo = np.asarray(robot.joints.lower_limits, np.float64)
    hi = np.asarray(robot.joints.upper_limits, np.float64)

    def min_clear(qb):
        w = np.asarray(coll.compute_world_collision_distance(robot, jnp.asarray(qb, jnp.float32), obstacle)
                       ).reshape(len(qb), -1).min(axis=-1)
        s = np.asarray(coll.compute_self_collision_distance(robot, jnp.asarray(qb, jnp.float32))
                       ).reshape(len(qb), -1).min(axis=-1)
        return np.minimum(w, s)

    rng = np.random.default_rng(SEED)
    starts, goals = [], []
    while len(starts) < N_PROBLEMS:
        a = lo + (hi - lo) * rng.random((8 * N_PROBLEMS, lo.shape[0]))
        b = lo + (hi - lo) * rng.random((8 * N_PROBLEMS, lo.shape[0]))
        ca, cb = min_clear(a), min_clear(b)
        endpoints_ok = (ca >= 0.0) & (cb >= 0.0)
        # straight-line midpoints: require the lerp to graze but not be deeply
        # trapped -> local avoidance matters, detour exists.
        mids = 0.5 * (a + b)
        cm = min_clear(mids)
        lerp_grazes = (cm < 0.0) & (cm > -0.15)
        keep = endpoints_ok & lerp_grazes
        for i in np.nonzero(keep)[0]:
            if len(starts) >= N_PROBLEMS:
                break
            starts.append(a[i]); goals.append(b[i])
    q_start = np.asarray(starts[:N_PROBLEMS]); q_goal = np.asarray(goals[:N_PROBLEMS])
    np.savez(PROBLEM_FILE, q_start=q_start, q_goal=q_goal,
             joint_lower=lo, joint_upper=hi, T=np.array(T_WAYPOINTS))
    print(f"wrote {PROBLEM_FILE}: {N_PROBLEMS} config->config problems, "
          f"{lo.shape[0]} DOF, straight-line grazes obstacle (local-avoidance regime)")


if __name__ == "__main__":
    main()
