"""Score every solver's results with ONE shared metric (run under `pyroffi`).

Neither library grades its own homework: success is recomputed here from the
returned configuration using the same FK and the same collision model, so a
difference in reported success cannot come from a difference in what the two
stacks consider converged.
"""
from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import yourdfpy

import pyroffi as pk
from pyroffi.collision import Box, RobotCollisionSpherized
from _problems import OBSTACLE_CENTER, OBSTACLE_DIMS, RESULT_DIR, load

RESOURCE_ROOT = pathlib.Path(__file__).resolve().parents[2] / "resources"

# cuRobo's own defaults, adopted so the bar is one it was tuned to clear.
POS_TOL = 0.005    # m
ROT_TOL = 0.05     # rad


def main() -> None:
    _q_ref, target_wxyz_xyz, ee = load()

    urdf = yourdfpy.URDF.load(str(RESOURCE_ROOT / "panda" / "panda_spherized.urdf"))
    robot = pk.Robot.from_urdf(urdf)
    coll = RobotCollisionSpherized.from_urdf(
        urdf, srdf_path=str(RESOURCE_ROOT / "panda" / "panda.srdf"))
    obstacle = Box.from_center_and_dimensions(
        jnp.asarray(OBSTACLE_CENTER, jnp.float32)[None],
        jnp.float32(OBSTACLE_DIMS[0])[None],
        jnp.float32(OBSTACLE_DIMS[1])[None],
        jnp.float32(OBSTACLE_DIMS[2])[None])

    target = jaxlie.SE3(jnp.asarray(target_wxyz_xyz, jnp.float32))

    def pose_err(cfgs):
        T = jax.vmap(lambda c: robot.forward_kinematics(c)[ee])(cfgs)
        got = jaxlie.SE3(T)
        d = jaxlie.SE3(target.wxyz_xyz).inverse() @ got
        log = d.log()
        return (np.asarray(jnp.linalg.norm(log[:, :3], axis=-1)),
                np.asarray(jnp.linalg.norm(log[:, 3:], axis=-1)))

    rows = []
    for f in sorted(RESULT_DIR.glob("*.npz")):
        d = np.load(f)
        cfg = jnp.asarray(d["cfg"], jnp.float32)
        secs = float(d["seconds"])
        n = int(d["n_problems"])

        p_err, r_err = pose_err(cfg)
        pose_ok = (p_err <= POS_TOL) & (r_err <= ROT_TOL)

        self_d = np.asarray(coll.compute_self_collision_distance(robot, cfg)).min(axis=-1)
        world_d = np.asarray(
            coll.compute_world_collision_distance(robot, cfg, obstacle)
        ).reshape(len(cfg), -1).min(axis=-1)
        free = (self_d >= 0) & (world_d >= 0)

        rows.append(dict(
            name=f.stem, ms=secs * 1e3, kips=n / secs / 1e3,
            pose=100 * pose_ok.mean(),
            free=100 * free.mean(),
            both=100 * (pose_ok & free).mean(),
            selfhit=100 * (self_d < 0).mean(),
            worldhit=100 * (world_d < 0).mean(),
        ))

    print(f"\n{len(rows)} solvers, {int(np.load(sorted(RESULT_DIR.glob('*.npz'))[0])['n_problems'])} "
          f"reachable targets, one cuboid obstacle\n")
    hdr = (f"{'solver':14}{'ms':>9}{'kIK/s':>8}{'pose ok':>9}"
           f"{'coll-free':>11}{'BOTH':>8}{'self hit':>10}{'world hit':>11}")
    print(hdr); print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: -x["both"]):
        print(f"{r['name']:14}{r['ms']:9.1f}{r['kips']:8.2f}{r['pose']:8.1f}%"
              f"{r['free']:10.1f}%{r['both']:7.1f}%{r['selfhit']:9.1f}%{r['worldhit']:10.1f}%")
    print("\nBOTH = pose within tolerance AND collision-free. That column is the"
          "\ncomparison; the split shows WHERE each stack fails.")


if __name__ == "__main__":
    main()
