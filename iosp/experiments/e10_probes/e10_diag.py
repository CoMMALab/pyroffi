"""Where does E10's 0.16 m reconstruction error actually live?

Four questions, one GPU pass (executables are in the persistent cache):
  1. what does the UNFITTED planner (u=0) score -- the missing baseline
  2. is the error concentrated at the grasp, or spread over the path
  3. does the IK stage's q_pick match the configuration the human grasped from
  4. is any q_pick mismatch a POSE error (IK missed) or a BRANCH/orientation
     error (IK hit a pose the demonstrator never used)
"""
import json
import numpy as np, jax, jax.numpy as jnp
from iosp import config
config.enable_compilation_cache()
from iosp.fit import multistart as ms
from iosp.model import pickplace as pp

built = ms.build_from_demos(n_fit=8, seed=0, n_iters=60, n_branches=4)
refs, K = built["refs"], built["K"]
fit_idx, gen_idx = built["fit_idx"], built["gen_idx"]
demo_q, demo_ee = built["demo"], built["demo_ee"]
res = json.load(open("iosp/data/e10_teleop.json"))
u_win = jnp.asarray(res["u_winner"], jnp.float32)[None]
b_win = res["winner"] // res["n_starts"]

rmse = lambda P, D, idx: float(jnp.sqrt(jnp.mean(jnp.sum((P[idx] - D[idx])**2, -1))))

# -- 1. baseline vs winner ---------------------------------------------------
u0 = jnp.zeros((1, K), jnp.float32)
print(f"\n{'':26s} {'fit joint':>10} {'held joint':>11} {'fit EE':>9} {'held EE':>9}")
rows = {}
for lab, u, b in (("u=0 prior (branch 0)", u0, 0),
                  (f"u=0 prior (branch {b_win})", u0, b_win),
                  (f"WINNER (branch {b_win})", u_win, b_win)):
    r = refs[b][None]
    qj = built["batched_paths"](u, r, "joint")[0]
    qe = built["batched_paths"](u, r, "ee")[0]
    rows[lab] = (qj, qe)
    print(f"{lab:26s} {rmse(qj,demo_q,fit_idx):10.4f} {rmse(qj,demo_q,gen_idx):11.4f} "
          f"{rmse(qe,demo_ee,fit_idx):9.4f} {rmse(qe,demo_ee,gen_idx):9.4f}", flush=True)

d = np.asarray(demo_ee)
print(f"{'predict the mean demo':26s} {'-':>10} {'-':>11} "
      f"{float(np.sqrt(np.mean(np.sum((d-d.mean(0))**2,-1)))):9.4f} {'-':>9}")

# -- 2. per-waypoint error profile ------------------------------------------
qj_w, qe_w = rows[f"WINNER (branch {b_win})"]
per_wp = np.sqrt(np.mean(np.sum((np.asarray(qe_w)-d)**2, -1), axis=0))  # (T,)
print("\nper-waypoint EE error of the winner [m], row 0..22:")
print("  " + " ".join(f"{v:.3f}" for v in per_wp))
print(f"  phase spans: {pp.PHASE_SPAN}   skeleton pick {pp.SKELETON_PICK} place {pp.SKELETON_PLACE}")

# -- 3/4. the IK stage in isolation -----------------------------------------
scenes = built["scenes"]
theta_ik = built["P"][:pp.K_IK]
prob = built["prob"]
q_pick = prob.grasp_ik(theta_ik, scenes)
q_place = prob.place_ik(theta_ik, scenes, q_pick)
dq = np.asarray(demo_q)
for lab, q_ik, row, tgt in (("q_pick", q_pick, pp.SKELETON_PICK[0], scenes.pick_pos),
                            ("q_place", q_place, pp.SKELETON_PLACE[0], scenes.place_pos)):
    q_ik = np.asarray(q_ik)
    dj = np.linalg.norm(q_ik - dq[:, row], axis=-1)
    ee_ik = np.asarray(prob.ee_positions(jnp.asarray(q_ik)))
    ee_demo_row = np.asarray(demo_ee)[:, row]
    off = float(theta_ik[0] if lab == "q_pick" else theta_ik[1])
    want = np.asarray(tgt) + np.array([0, 0, off])
    print(f"\n{lab}:  ||q_ik - q_demo|| per episode [rad]")
    print("   " + " ".join(f"{v:.2f}" for v in dj))
    print(f"   IK pose residual (achieved vs TARGET) [m]: "
          f"{np.round(np.linalg.norm(ee_ik-want,axis=-1),4)}")
    print(f"   achieved vs DEMO hand position     [m]: "
          f"{np.round(np.linalg.norm(ee_ik-ee_demo_row,axis=-1),4)}")
