"""Confirming test: are the behavioural-loss spikes IK branch flips?

Costs no fit.  `pathA_behavior_r3.npz` already stores theta at every outer
step, and theta_ik is its first two coordinates, so the IK stage can simply be
re-evaluated along the SAME trajectory the fit took -- 41 IK calls, no trajopt.

Prediction, if the spikes are self-motion branch flips on the 7-DOF-vs-6-DOF
redundant IK: at the spike steps the joint-space jump ||q_pick(t) -
q_pick(t-1)|| is large and OUT OF LINE with the smooth theta step, while the
achieved EE position is unchanged (the flip moves along the self-motion
manifold, which by definition preserves the pose).
"""
import numpy as np, jax, jax.numpy as jnp
from iosp import pickplace as pp
from iosp import study3_identifiable_refit as s3

d = np.load('scratch/viz/pathA_behavior_r3.npz', allow_pickle=True)
th = d['theta_hist']                       # (41, 9): [theta_ik(2) | theta_traj(7)]
prob = pp.PickPlaceProblem.load(str(s3.URDF_PATH), str(s3.SRDF_PATH), str(s3.MESH_DIR))
scenes = s3._scenes_ab()

@jax.jit
def ik(theta_ik):
    qp = prob.grasp_ik(theta_ik, scenes)
    qpl = prob.place_ik(theta_ik, scenes, qp)
    return qp, qpl, prob.ee_positions(qp), prob.ee_positions(qpl)

QP, QPL, EP, EPL = [], [], [], []
for t in range(th.shape[0]):
    a, b, c, e = ik(jnp.asarray(th[t, :2], jnp.float32))
    QP.append(np.asarray(a)); QPL.append(np.asarray(b))
    EP.append(np.asarray(c)); EPL.append(np.asarray(e))
QP, QPL, EP, EPL = map(np.stack, (QP, QPL, EP, EPL))

SPIKES = {14: 'A', 17: 'B', 24: 'A', 36: 'B', 37: 'A', 39: 'A'}
dq = np.linalg.norm(np.diff(QP, axis=0), axis=-1)      # (40, 2) joint jump
de = np.linalg.norm(np.diff(EP, axis=0), axis=-1)      # (40, 2) EE jump
dth = np.abs(np.diff(th[:, :2], axis=0)).sum(1)

print(f'{"step":>5} {"|dtheta_ik|":>12} {"|dq_pick| A":>12} {"|dq_pick| B":>12} '
      f'{"|dEE| A":>10} {"|dEE| B":>10}  spike')
for t in range(1, th.shape[0]):
    mark = SPIKES.get(t, '')
    flag = '  <<<' if mark else ''
    print(f'{t:5d} {dth[t-1]:12.6f} {dq[t-1,0]:12.5f} {dq[t-1,1]:12.5f} '
          f'{de[t-1,0]:10.5f} {de[t-1,1]:10.5f}  {mark}{flag}')

print('\n--- summary ---')
nz = [t for t in range(1, th.shape[0]) if t not in SPIKES]
for i, nm in enumerate('AB'):
    print(f'scene {nm}: median |dq_pick| off-spike {np.median(dq[[t-1 for t in nz], i]):.5f}, '
          f'at ITS spikes {[round(float(dq[t-1,i]),4) for t,s in SPIKES.items() if s==nm]}')
    print(f'          median |dEE|     off-spike {np.median(de[[t-1 for t in nz], i]):.5f}, '
          f'at ITS spikes {[round(float(de[t-1,i]),5) for t,s in SPIKES.items() if s==nm]}')
