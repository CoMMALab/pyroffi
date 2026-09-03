"""One episode, multistart, everything anchored + the fitted release offset.

Reports RECONSTRUCTION loss: this is the fit set, so there is no held-out claim
here -- the question is only whether the composed planner can now reproduce a
single human demonstration, with the grasp and release seeds no longer wrong by
more than half the reach.
"""
import argparse, json
import numpy as np, jax, jax.numpy as jnp
from iosp import config
config.enable_compilation_cache()
from iosp.fit import multistart as ms
from iosp.model import pickplace as pp

ap = argparse.ArgumentParser()
ap.add_argument("--anchor", action="store_true")
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--n-branches", type=int, default=4)
ap.add_argument("--n-starts", type=int, default=3)
ap.add_argument("--lr", type=float, default=config.LR)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

built = ms.build_from_demos(n_fit=1, max_episodes=1, n_branches=a.n_branches,
                            anchor_grasp=a.anchor, seed=a.seed)
print(f"episode {built['episodes'][0]}   anchor_grasp={a.anchor}   "
      f"K={built['K']}  theta_ik prior "
      f"{np.round(np.asarray(built['standoff_prior']), 4)}", flush=True)

# The seed error this whole exercise is about, on THIS episode.
th = built["P"][:pp.K_IK]
prob, sc, dq_ = built["prob"], built["scenes"], np.asarray(built["demo"])
qp = prob.grasp_ik(th, sc); ql = prob.place_ik(th, sc, qp)
print(f"  seed error: ||q_pick - demo|| {np.linalg.norm(np.asarray(qp)-dq_[:,pp.SKELETON_PICK[0]],axis=-1)[0]:.3f} rad"
      f"   ||q_place - demo|| {np.linalg.norm(np.asarray(ql)-dq_[:,pp.SKELETON_PLACE[0]],axis=-1)[0]:.3f} rad",
      flush=True)

res = ms.run(seed=a.seed, n_branches=a.n_branches, n_starts=a.n_starts,
             n_steps=a.steps, built=built, chunk=2, lr=a.lr)
w = int(np.argmin(res["train"]))
u_w, refs_w = jnp.asarray(res["u"][w])[None], built["refs"][w // res["S"]][None]
ee = built["batched_paths"](u_w, refs_w, "ee")[0]
ee0 = built["batched_paths"](jnp.zeros((1, built["K"]), jnp.float32), refs_w, "ee")[0]
d_ee = built["demo_ee"]
r = lambda P: float(jnp.sqrt(jnp.mean(jnp.sum((P - d_ee)**2, -1))))
# The TRUE u=0 rollout, in both spaces.  The previous version of this script
# printed `sqrt(train).max()` under a "prior(u=0)" label, which is the worst
# CANDIDATE's converged loss and not a baseline at all.
qj0 = built["batched_paths"](jnp.zeros((1, built["K"]), jnp.float32), refs_w, "joint")[0]
dq_demo = built["demo"]
rj = lambda P: float(jnp.sqrt(jnp.mean(jnp.sum((P - dq_demo) ** 2, -1))))
print(f"\nRECONSTRUCTION (1 episode, {res['u'].shape[0]} candidates, "
      f"{a.steps} steps, lr={a.lr}, anchor={a.anchor}, seed={a.seed})")
print(f"  joint RMSE : u=0 {rj(qj0):.4f}   winner {np.sqrt(res['train'][w]):.4f} rad")
print(f"  EE RMSE    : u=0 {r(ee0):.4f} m  winner {r(ee):.4f} m")
print(f"  per-candidate joint RMSE: {np.round(np.sqrt(res['train']),3)}")
print(f"  loss history (every 5): {np.round(res['losses'][::5],2)}")
