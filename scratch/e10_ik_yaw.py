"""Does object-yaw-aware IK land on the grasp the human actually used?

IK stage ONLY -- no trajopt, no fit.  Compares q_pick against the demo's own
configuration at the skeleton grasp row, under three target orientations:
fixed DOWN (what every run so far used), +cube_yaw, and -cube_yaw (to settle
the sign empirically rather than by reasoning about frame conventions).
"""
import numpy as np, jax, jax.numpy as jnp, jaxlie
from iosp import config
config.enable_compilation_cache()
from iosp.fit import teleop as tl
from iosp.model import fr3, pickplace as pp

names, demo_q, scenes = tl.load_demos()
prob = pp.PickPlaceProblem.load(*fr3.paths()[:3], ee_link=fr3.paths()[3])
theta_ik = tl.z_prior(pp.K_IK + pp.K_TRAJOPT, pp.K_IK,
                      tl.measure_standoffs(prob, demo_q, scenes,
                                           np.arange(len(names))))[:pp.K_IK]
row = pp.SKELETON_PICK[0]
q_ref = np.asarray(demo_q)[:, row]                      # the human's grasp cfg
yaw = np.asarray(scenes.pick_yaw).ravel()

def ee_quat(q):
    fk = prob.base.robot.forward_kinematics(jnp.asarray(q))
    return np.asarray(fk[..., prob.ee_index, :4])

def ang_deg(qa, qb):
    """Geodesic angle between two wxyz rotations, per row."""
    Ra, Rb = jaxlie.SO3(jnp.asarray(qa)), jaxlie.SO3(jnp.asarray(qb))
    return np.degrees(np.asarray(jnp.linalg.norm((Ra.inverse() @ Rb).log(), axis=-1)))

print(f"\ndemo grasp row {row}; standoff prior {float(theta_ik[0]):.4f} m")
print(f"{'target orientation':>22} {'||dq|| rad':>11} {'EE pos err':>11} {'EE ori err':>11}")
for lab, yv in (("fixed DOWN (current)", None), ("+cube_yaw", yaw), ("-cube_yaw", -yaw)):
    sc = pp.PickPlaceScene(
        q_start=scenes.q_start, pick_pos=scenes.pick_pos, place_pos=scenes.place_pos,
        obs_center=scenes.obs_center, obs_radius=scenes.obs_radius,
        pick_yaw=None if yv is None else jnp.asarray(yv[:, None], jnp.float32))
    q_ik = np.asarray(prob.grasp_ik(theta_ik, sc))
    dq = np.linalg.norm(q_ik - q_ref, axis=-1)
    dp = np.linalg.norm(np.asarray(prob.ee_positions(jnp.asarray(q_ik)))
                        - np.asarray(prob.ee_positions(jnp.asarray(q_ref))), axis=-1)
    do = ang_deg(ee_quat(q_ik), ee_quat(q_ref))
    print(f"{lab:>22} {dq.mean():11.3f} {dp.mean():11.4f} {do.mean():11.1f}")
    print(f"{'':22} per-ep dq: " + " ".join(f"{v:.2f}" for v in dq))

# Is the IK even hitting its own target?  A large dq with a tiny pose residual
# is a branch/orientation problem; a large residual is a solver problem.
sc = pp.PickPlaceScene(q_start=scenes.q_start, pick_pos=scenes.pick_pos,
                       place_pos=scenes.place_pos, obs_center=scenes.obs_center,
                       obs_radius=scenes.obs_radius,
                       pick_yaw=jnp.asarray(yaw[:, None], jnp.float32))
q_ik = prob.grasp_ik(theta_ik, sc)
want_p = np.asarray(scenes.pick_pos) + float(theta_ik[0]) * np.array([0, 0, 1.0])
want_q = np.asarray(pp._down_yaw_wxyz(sc.pick_yaw))
print(f"\nIK solver accuracy at +cube_yaw:  pos residual "
      f"{np.linalg.norm(np.asarray(prob.ee_positions(q_ik)) - want_p, axis=-1).max():.5f} m max, "
      f"ori residual {ang_deg(ee_quat(np.asarray(q_ik)), want_q).max():.2f} deg max")
