"""Isolate WHERE a domain's reach error comes from: the IK seed or the trajopt.

Skeleton cost is joint-space (`q[IDX_PICK] - q_pick`), with `q_pick` an IK solve
for `pick_pos`.  So a big reach error is one of:
  (A) IK quality   -- EE(q_pick) is already far from pick_pos (IK failed / wrong
                      target frame);
  (B) trajopt      -- q[IDX_PICK] never reached q_pick (skeleton under-converged).
Reports both, at panda_hand AND panda_grasptarget, for tower/tetris/pickplace.

    python -m iosp.checks.ik_diag tower
"""
import argparse
import numpy as np

from iosp import config
config.setup()
import jax


def _load(domain):
    from iosp.config import URDF_PATH, SRDF_PATH, MESH_DIR
    if domain == "tower":
        from iosp.experiments import e9_tower as E
        built = E.build(seed=0, n_iters=60, n_scenes=6, stack_level=0)
        sc = built["fit"]
        q_pick, q_place = built["prob"].pick_ik(sc.pick_pos, sc.q_start), None
        seeds = built["prob"].seeds(sc)
        return built, sc, seeds
    if domain == "tetris":
        from iosp.experiments import e8_tetris as E
        built = E.build(seed=0, n_iters=60, n_scenes=6, num_blocks=1)
        sc = built["fit"]
        return built, sc, built["prob"].seeds(sc)
    from iosp.experiments import e4_three_stage as E
    built = E.build(seed=0, n_iters=60, n_scenes=6)
    sc = built["fit"]
    return built, sc, built["prob"].seeds(sc, E.THETA_IK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("domain", choices=["tower", "tetris", "pickplace"])
    args = ap.parse_args()
    from iosp.config import URDF_PATH, SRDF_PATH, MESH_DIR
    from ioc.robot.problem import RobotProblem

    built, sc, seeds = _load(args.domain)
    # seeds() returns q_pick/q_place among its outputs; grab them generically
    print("seeds returned", type(seeds), len(seeds) if isinstance(seeds, tuple) else "")
    # locate q_pick / q_place by shape (B, dof)
    prob = built["prob"]
    pick = np.asarray(sc.pick_pos); place = np.asarray(sc.place_pos)

    def ee_at(qs, link):
        p = RobotProblem.load(str(URDF_PATH), str(SRDF_PATH), str(MESH_DIR),
                              n_timesteps=1, ee_link=link)
        return np.asarray(jax.vmap(p.ee_positions)(qs))

    # find q_pick/q_place candidates in the seeds tuple
    cands = [np.asarray(x) for x in (seeds if isinstance(seeds, tuple) else [seeds])
             if hasattr(x, "shape") and np.asarray(x).ndim == 2
             and np.asarray(x).shape[-1] == prob.base.dof]
    print(f"{args.domain}: {len(cands)} (B,dof) seed arrays")
    for i, qs in enumerate(cands):
        for link in ("panda_hand", "panda_grasptarget"):
            ee = ee_at(qs, link)
            dp = np.linalg.norm(ee - pick, axis=-1).mean() * 1000
            dpl = np.linalg.norm(ee - place, axis=-1).mean() * 1000
            print(f"  seed[{i}] @ {link:18s}: dist->pick {dp:5.0f}mm  dist->place {dpl:5.0f}mm")


if __name__ == "__main__":
    main()
