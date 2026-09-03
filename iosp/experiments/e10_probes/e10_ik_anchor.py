"""Anchor q_pick to the demonstrated grasp configuration -- seed vs target.

`previous_cfg` picks among solutions to the target pose; it cannot change which
pose is solved for.  So there are two distinct proposals here and they are not
the same experiment:

  seed only    target = the model's own (pick_pos + standoff, DOWN*yaw),
               previous_cfg = the demo's grasp configuration
  seed+orient  target orientation taken from that same configuration's FK,
               position still the model's -- i.e. the grasp POSE becomes an
               input rather than a prediction
"""
import numpy as np, jax.numpy as jnp, jaxlie
from iosp import config
config.enable_compilation_cache()
from iosp.fit import teleop as tl
from iosp.model import fr3, pickplace as pp
from pyroffi.optimization_engines._sqp_ik import sqp_ik_solve_cuda_batch

names, demo_q, scenes = tl.load_demos()
prob = pp.PickPlaceProblem.load(*fr3.paths()[:3], ee_link=fr3.paths()[3])
theta_ik = tl.z_prior(pp.K_IK + pp.K_TRAJOPT, pp.K_IK,
                      tl.measure_standoffs(prob, demo_q, scenes, np.arange(len(names))))[:pp.K_IK]
row = pp.SKELETON_PICK[0]
q_ref = jnp.asarray(np.asarray(demo_q)[:, row])
qr = np.asarray(q_ref)
demo_wxyz = prob.base.robot.forward_kinematics(q_ref)[:, prob.ee_index, :4]

pos = scenes.pick_pos + theta_ik[0] * pp.UP_AXIS
yaw_wxyz = pp._down_yaw_wxyz(scenes.pick_yaw)
q_start = np.asarray(scenes.q_start)

def solve(wxyz, prev):
    tgt = jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3(wxyz=jnp.asarray(wxyz, jnp.float32)),
        translation=pos.astype(jnp.float32))
    return np.asarray(sqp_ik_solve_cuda_batch(
        prob.base.robot, prob.ee_index, tgt, pp.IK_RNG_KEY,
        jnp.asarray(prev, jnp.float32), continuity_weight=pp.IK_CONTINUITY_WEIGHT))

print(f"\n{'configuration':44s} {'||dq||':>8} {'max':>7}")
for lab, wxyz, prev in (
        ("baseline: DOWN*yaw target, seed q_start",      yaw_wxyz,   q_start),
        ("SEED ONLY: DOWN*yaw target, seed demo cfg",    yaw_wxyz,   qr),
        ("SEED+ORIENT: demo orientation, seed q_start",  demo_wxyz,  q_start),
        ("SEED+ORIENT: demo orientation, seed demo cfg", demo_wxyz,  qr)):
    q = solve(wxyz, prev)
    dq = np.linalg.norm(q - qr, axis=-1)
    print(f"{lab:44s} {dq.mean():8.3f} {dq.max():7.3f}")
    print(f"{'':44s} per-ep: " + " ".join(f"{v:.2f}" for v in dq))
print(f"\nfor scale: ||q_demo_grasp - q_start|| mean "
      f"{np.linalg.norm(qr - q_start, axis=-1).mean():.3f} rad")
