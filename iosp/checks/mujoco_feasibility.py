"""Check a domain's forward-pass trajectory for feasibility in MuJoCo.

Loads the joint path `q` extracted by `forward_extract` and, on the Menagerie
Panda, reports:

  * reachability  -- Panda EE (hand TCP) at the pick/place skeleton rows vs the
    demonstration's pick/place targets;
  * collision     -- worst penetration depth along the trajectory (self + the
    table), i.e. does the arm pass through the world to hit those targets;
  * grasp/place   -- the cube, carried on the TCP between the grasp and release
    rows, ends near the place target.

A feasible forward pass (small reach error, no deep penetration, cube lands on
target) means the inverse pass is chasing an achievable demonstration.

    python -m iosp.checks.mujoco_feasibility scratch/feas/tower.npz [--view]
"""
import argparse
import pathlib

import numpy as np
import mujoco
from robot_descriptions import panda_mj_description

TCP_OFFSET = 0.1034  # hand frame -> fingertip TCP, along hand +z


def build_scene(cube_half, cube_xy):
    spec = mujoco.MjSpec.from_file(panda_mj_description.MJCF_PATH)
    wb = spec.worldbody
    light = wb.add_light(); light.pos = [0, 0, 2]; light.dir = [0, 0, -1]
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    floor = wb.add_geom(); floor.name = "floor"; floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0, 0, 0.05]; floor.rgba = [0.3, 0.3, 0.35, 1]
    cube = wb.add_body(); cube.name = "cube"; cube.pos = [cube_xy[0], cube_xy[1], cube_half]
    fj = cube.add_freejoint(); fj.name = "cube_free"
    g = cube.add_geom(); g.name = "cube_geom"; g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [cube_half] * 3; g.rgba = [0.8, 0.2, 0.2, 1]; g.mass = 0.05
    return spec.compile()


def _tcp(model, data):
    hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    R = data.xmat[hid].reshape(3, 3)
    return data.xpos[hid] + R @ np.array([0, 0, TCP_OFFSET])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--scene", type=int, default=0, help="which scene row")
    ap.add_argument("--view", action="store_true")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    q = d["q"][args.scene]                     # (N_FULL, dof)
    ipick, iplace = int(d["idx_pick"]), int(d["idx_place"])
    pick = d["pick_pos"][args.scene]
    place = d["place_pos"][args.scene]
    cube_half = float(d["block_half"]) if "block_half" in d.files else 0.03
    dof = q.shape[-1]

    model = build_scene(cube_half, pick[:2])
    data = mujoco.MjData(model)
    arm = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint1")]
    cube_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")]

    max_pen = 0.0
    tcp_pick = tcp_place = None
    for t in range(q.shape[0]):
        data.qpos[arm:arm + 7] = q[t, :7]
        # cube: on the table at pick until grasp, on the TCP through the carry,
        # at the place target after release.
        data.qpos[arm + 7] = data.qpos[arm + 8] = cube_half  # fingers open ~cube
        mujoco.mj_forward(model, data)
        tcp = _tcp(model, data)
        if t <= ipick:
            cpos = np.array([pick[0], pick[1], cube_half])
        elif t <= iplace:
            cpos = tcp
        else:
            cpos = place
        data.qpos[cube_adr:cube_adr + 3] = cpos
        data.qpos[cube_adr + 3:cube_adr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)
        # worst penetration (contacts report negative dist when interpenetrating)
        for i in range(data.ncon):
            max_pen = max(max_pen, -data.contact[i].dist)
        if t == ipick:
            tcp_pick = tcp.copy()
        if t == iplace:
            tcp_place = tcp.copy()

    print(f"domain={str(d['domain'])}  scene={args.scene}  dof={dof}  "
          f"N_FULL={q.shape[0]}  cube={2*cube_half*100:.0f}cm")
    print(f"  reach @pick : TCP {np.round(tcp_pick,3)}  target {np.round(pick,3)}  "
          f"err {np.linalg.norm(tcp_pick-pick)*1000:.0f} mm")
    print(f"  reach @place: TCP {np.round(tcp_place,3)}  target {np.round(place,3)}  "
          f"err {np.linalg.norm(tcp_place-place)*1000:.0f} mm")
    print(f"  worst penetration along path: {max_pen*1000:.1f} mm")
    print(f"  q joint range: [{q[:, :7].min():.2f}, {q[:, :7].max():.2f}] rad")

    if args.view:
        import viser
        from mjviser import Viewer
        server = viser.ViserServer(port=8080)
        Viewer(model, data, server=server).run()


if __name__ == "__main__":
    main()
