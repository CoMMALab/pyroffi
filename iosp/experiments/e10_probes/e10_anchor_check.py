"""`anchor_grasp` end-to-end, at BOTH events, through the real code path."""
import numpy as np, jax.numpy as jnp
from iosp import config
config.enable_compilation_cache()
from iosp.fit import teleop as tl
from iosp.model import fr3, pickplace as pp

prob = pp.PickPlaceProblem.load(*fr3.paths()[:3], ee_link=fr3.paths()[3])
r_pick, r_place = pp.SKELETON_PICK[0], pp.SKELETON_PLACE[0]
print(f"\n{'':20s} {'q_pick vs demo':>20s} {'q_place vs demo':>20s}   [rad]")
for anchor in (False, True):
    names, demo_q, scenes = tl.load_demos(prob=prob, anchor_grasp=anchor)
    th = tl.z_prior(pp.K_IK + pp.K_TRAJOPT, pp.K_IK,
                    tl.measure_standoffs(prob, demo_q, scenes, np.arange(len(names))))[:pp.K_IK]
    qp = prob.grasp_ik(th, scenes)
    ql = prob.place_ik(th, scenes, qp)
    d = np.asarray(demo_q)
    dp = np.linalg.norm(np.asarray(qp) - d[:, r_pick], axis=-1)
    dl = np.linalg.norm(np.asarray(ql) - d[:, r_place], axis=-1)
    print(f"anchor={str(anchor):5s}        mean {dp.mean():.3f} max {dp.max():.3f}"
          f"     mean {dl.mean():.3f} max {dl.max():.3f}", flush=True)
    if anchor:
        print(f"{'':20s} per-ep place: " + " ".join(f"{v:.2f}" for v in dl))
