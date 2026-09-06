"""Feasibility verdict for a domain's forward-pass trajectory.

Robot-exact, no cross-model confound:
  * reach     -- pyroffi's OWN forward kinematics on the extracted `q`, at the
    grasp/place skeleton rows, vs the demonstration's pick/place targets;
  * collision -- the SAME `panda_spherized` URDF in MuJoCo: worst self-collision
    penetration along the path (minus the constant adjacent-link sphere overlap
    the raw URDF import can't exclude), and joint-limit compliance.

    python -m iosp.checks.feasibility_report scratch/feas/tower.npz
"""
import argparse

import numpy as np


def _reach(q, idx_pick, idx_place, pick, place):
    import jax
    from iosp.config import URDF_PATH, SRDF_PATH, MESH_DIR
    from ioc.robot.problem import RobotProblem
    prob = RobotProblem.load(str(URDF_PATH), str(SRDF_PATH), str(MESH_DIR),
                             n_timesteps=q.shape[1])
    ee = np.asarray(jax.vmap(lambda qq: jax.vmap(prob.ee_positions)(qq))(q))
    pe = np.linalg.norm(ee[:, idx_pick] - pick, axis=-1) * 1000
    pl = np.linalg.norm(ee[:, idx_place] - place, axis=-1) * 1000
    return pe, pl


def _collision(q):
    """Worst REAL self-collision (non-adjacent links only, |bodyid diff|>1, so
    the always-touching neighbour-link spheres don't count) and the joint-limit
    overshoot in degrees."""
    import mujoco
    from iosp.config import URDF_PATH
    m = mujoco.MjModel.from_xml_path(str(URDF_PATH))
    data = mujoco.MjData(m)
    adr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "panda_joint1")]
    worst = np.zeros(q.shape[0])
    for s in range(q.shape[0]):
        mp = 0.0
        for t in range(q.shape[1]):
            data.qpos[adr:adr + 7] = q[s, t, :7]
            mujoco.mj_forward(m, data)
            for i in range(data.ncon):
                c = data.contact[i]
                if abs(m.geom_bodyid[c.geom1] - m.geom_bodyid[c.geom2]) > 1:
                    mp = max(mp, -c.dist)
        worst[s] = mp
    over = np.maximum(0.0, np.maximum(m.jnt_range[:7, 0] - q[..., :7],
                                      q[..., :7] - m.jnt_range[:7, 1]))
    return worst * 1000, float(over.max()) * 57.2958, bool(over.max() < 1e-3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    args = ap.parse_args()
    d = np.load(args.npz, allow_pickle=True)
    q = d["q"]
    ip, ipl = int(d["idx_pick"]), int(d["idx_place"])
    pick, place = d["pick_pos"], d["place_pos"]

    pe, pl = _reach(q, ip, ipl, pick, place)
    worst, over_deg, lim_ok = _collision(q)

    print(f"\n===== {str(d['domain'])}  ({q.shape[0]} scenes, {q.shape[1]} frames) =====")
    print(f"  reach  pick : {pe.mean():5.0f} mm  (min {pe.min():.0f}, max {pe.max():.0f})")
    print(f"  reach  place: {pl.mean():5.0f} mm  (min {pl.min():.0f}, max {pl.max():.0f})")
    print(f"  real self-collision (non-adjacent links): max {worst.max():.0f} mm")
    print(f"  joint-limit overshoot: {over_deg:.1f} deg  ({'OK' if lim_ok else 'VIOLATED'})")
    grasps = pe.mean() < 50
    print(f"  VERDICT: place {'reached' if pl.mean()<50 else 'MISSED'}; "
          f"pick {'reached' if grasps else 'MISSED (~%.0fmm short)'%pe.mean()}; "
          f"{'collision-free' if worst.max()<5 else 'COLLISIONS'}; "
          f"{'in-limits' if lim_ok else 'LIMIT-VIOLATION'}")


if __name__ == "__main__":
    main()
