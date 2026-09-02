"""How many DISTINCT IK branches does the redundant Panda actually have here?

Cheap: IK only, no trajopt.  `IK_CONTINUITY_WEIGHT` makes the solver return the
branch nearest `previous_cfgs`, so sweeping many well-spread references and
clustering the returned q in JOINT space counts the distinct self-motion
solutions reachable at this target.  Two references that collapse to the same
configuration are duplicates of one branch; the number of surviving clusters is
the number of classes worth spending a bilevel candidate on.

Also reports EE error per cluster: every branch must hit the SAME pose (that is
what makes them branches rather than failures), so a cluster with large EE error
is a failed solve, not a homotopy class.
"""
import numpy as np, jax, jax.numpy as jnp
from iosp import pickplace as pp
from iosp import study3_identifiable_refit as s3

prob = pp.PickPlaceProblem.load(str(s3.URDF_PATH), str(s3.SRDF_PATH), str(s3.MESH_DIR))
sc = s3._scenes_ab()
M = sc.q_start.shape[0]
N = 64
rng = np.random.default_rng(0)
lo = np.asarray(prob.base.robot.joints.lower_limits)
hi = np.asarray(prob.base.robot.joints.upper_limits)
refs = jnp.asarray(rng.uniform(lo, hi, size=(N, lo.shape[0])), jnp.float32)

theta_ik = jnp.array([0.06, 0.04], jnp.float32)
tgt = sc.pick_pos[0] + theta_ik[0] * pp.UP_AXIS
tgt_b = jnp.broadcast_to(tgt, (N, 3))
q = np.asarray(pp._ik_batch(prob, tgt_b, refs))          # (N, dof)
ee = np.asarray(prob.ee_positions(jnp.asarray(q)))       # (N, 3)
err = np.linalg.norm(ee - np.asarray(tgt), axis=-1)

# cluster in joint space (single-linkage at a generous threshold: distinct
# branches are O(1) rad apart, numerical spread within a branch is ~1e-3)
TOL = 0.25
order = np.argsort(err)
labels = -np.ones(N, int); k = 0
for i in order:
    if labels[i] >= 0: continue
    labels[i] = k
    for j in order:
        if labels[j] < 0 and np.linalg.norm(q[j] - q[i]) < TOL:
            labels[j] = k
    k += 1
print(f"{N} random references -> {k} distinct clusters (joint-space tol {TOL} rad)")
print(f"{'cluster':>8} {'members':>8} {'EE err (m)':>12} {'reachable?':>11}")
for c in range(k):
    m = labels == c
    print(f"{c:8d} {m.sum():8d} {err[m].mean():12.5f} "
          f"{'yes' if err[m].mean() < 5e-3 else 'FAILED SOLVE':>11}")
good = [c for c in range(k) if (labels == c).sum() > 0 and err[labels == c].mean() < 5e-3]
print(f"\n{len(good)} clusters hit the target pose -> that many genuine homotopy classes")
if len(good) > 1:
    C = np.stack([q[labels == c].mean(0) for c in good])
    D = np.linalg.norm(C[:, None] - C[None], axis=-1)
    print("pairwise joint-space separation between classes (rad):")
    print(np.round(D, 2))
