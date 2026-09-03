"""Is the residual 0.382 rad at the release POSITION error, as predicted?

Same isolation as the grasp case: feed the demo's exact release pose and see
what survives.  If the anchored-orientation number collapses toward the grasp's
0.084, the remaining term is the lateral offset of the release point, which no
+z standoff can express.
"""
import numpy as np, jax.numpy as jnp, jaxlie
from iosp import config
config.enable_compilation_cache()
from iosp.fit import teleop as tl
from iosp.model import fr3, pickplace as pp
from pyroffi.optimization_engines._sqp_ik import sqp_ik_solve_cuda_batch

prob = pp.PickPlaceProblem.load(*fr3.paths()[:3], ee_link=fr3.paths()[3])
names, demo_q, scenes = tl.load_demos(prob=prob, anchor_grasp=True)
th = tl.z_prior(pp.K_IK + pp.K_TRAJOPT, pp.K_IK,
                tl.measure_standoffs(prob, demo_q, scenes, np.arange(len(names))))[:pp.K_IK]
row = pp.SKELETON_PLACE[0]
q_ref = jnp.asarray(np.asarray(demo_q)[:, row])
qr = np.asarray(q_ref)
demo_p = prob.ee_positions(q_ref)
model_p = scenes.place_pos + th[1] * pp.UP_AXIS

d = np.asarray(demo_p) - np.asarray(model_p)
print(f"\nmodel release target vs demo hand:  lateral "
      f"{np.linalg.norm(d[:, :2], axis=-1).mean():.4f} m   dz {d[:, 2].mean():+.4f} m")

def solve(pos, prev):
    tgt = jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3(wxyz=jnp.asarray(scenes.place_wxyz, jnp.float32)),
        translation=jnp.asarray(pos, jnp.float32))
    return np.asarray(sqp_ik_solve_cuda_batch(
        prob.base.robot, prob.ee_index, tgt, pp.IK_RNG_KEY,
        jnp.asarray(prev, jnp.float32), continuity_weight=pp.IK_CONTINUITY_WEIGHT))

print(f"\n{'release target position':44s} {'||dq||':>8} {'max':>7}")
for lab, pos in (("model: place_pos + standoff (current)", model_p),
                 ("demo's own release position", demo_p)):
    q = solve(pos, np.asarray(scenes.place_ref))
    dq = np.linalg.norm(q - qr, axis=-1)
    print(f"{lab:44s} {dq.mean():8.3f} {dq.max():7.3f}")
