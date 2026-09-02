"""Given a PERFECT pose target, does IK recover the human's configuration?

The yaw fix left 0.53 rad of joint error with a solver that hits its target to
0.35 mm.  Two things can still explain that: the target orientation is only
approximately right (we model yaw, not the ~16 deg tilt), or the arm is
redundant and the human's elbow is simply not where IK puts it.

Feeding the demo's OWN EE pose back in as the target separates them.  Any joint
error that survives an exactly-correct 6-DOF target is null-space, by
definition -- it is the one thing a pose target cannot determine.
"""
import numpy as np, jax.numpy as jnp, jaxlie
from iosp import config
config.enable_compilation_cache()
from iosp.fit import teleop as tl
from iosp.model import fr3, pickplace as pp
from pyroffi.optimization_engines._sqp_ik import sqp_ik_solve_cuda_batch

names, demo_q, scenes = tl.load_demos()
prob = pp.PickPlaceProblem.load(*fr3.paths()[:3], ee_link=fr3.paths()[3])
row = pp.SKELETON_PICK[0]
q_ref = jnp.asarray(np.asarray(demo_q)[:, row])

fk = prob.base.robot.forward_kinematics(q_ref)[:, prob.ee_index]
want_q, want_p = fk[:, :4], fk[:, 4:7]           # the human's exact grasp pose

def solve(prev):
    tgt = jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3(wxyz=want_q.astype(jnp.float32)),
        translation=want_p.astype(jnp.float32))
    return np.asarray(sqp_ik_solve_cuda_batch(
        prob.base.robot, prob.ee_index, tgt, pp.IK_RNG_KEY,
        jnp.asarray(prev, jnp.float32),
        continuity_weight=pp.IK_CONTINUITY_WEIGHT))

qr = np.asarray(q_ref)
print(f"\ntarget = the demo's OWN grasp pose (exact), row {row}")
for lab, prev in (("prev = q_start (what the pipeline uses)", np.asarray(scenes.q_start)),
                  ("prev = the demo cfg itself (upper bound)", qr)):
    q = solve(prev)
    dq = np.linalg.norm(q - qr, axis=-1)
    p = np.asarray(prob.ee_positions(jnp.asarray(q)))
    print(f"  {lab:42s} ||dq|| mean {dq.mean():.3f} rad, max {dq.max():.3f}")
    print(f"  {'':42s} pose residual {np.linalg.norm(p - np.asarray(want_p), axis=-1).max():.5f} m")
    print(f"  {'':42s} per-ep: " + " ".join(f"{v:.2f}" for v in dq))

# How much of q_start->grasp motion is null-space to begin with?
print(f"\nreference scales:  ||q_demo_grasp - q_start|| mean "
      f"{np.linalg.norm(qr - np.asarray(scenes.q_start), axis=-1).mean():.3f} rad")
