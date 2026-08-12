"""Generate the shared problem set (run under the `pyroffi` env)."""
from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import yourdfpy

import pyroffi as pk
from _problems import PROBLEM_FILE

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
    q_ref = lo + (hi - lo) * rng.random((N_PROBLEMS, lo.shape[0]))

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
          f"{lo.shape[0]} DOF, ee = {EE_LINK_NAME} (index {ee})")


if __name__ == "__main__":
    main()
