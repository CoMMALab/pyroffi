"""Does a loss still differentiate through IK -> 4 chained trajopt solves?

The `grasp_ik` reproducer covers the IK stage alone.  Every iosp study
differentiates a loss through the WHOLE composition, which is a different code
path (canonical IK's custom_jvp feeding four `solve_implicit` custom_vjps), so
it is checked separately rather than assumed to follow.
"""
import jax, jax.numpy as jnp, numpy as np
from iosp import pickplace as pp
from iosp import study3_identifiable_refit as s3

prob = pp.PickPlaceProblem.load(str(s3.URDF_PATH), str(s3.SRDF_PATH), str(s3.MESH_DIR))
sc = s3._scenes_ab()
fs = pp.make_composed_forward_solver(n_iters=20)
inner, _ = s3._build_inner(prob, s3.scene_a(), s3.THETA_IK_STAR, fs, 0)
S = s3.z_scale(9, 2)

def loss(u):
    z = u * S
    x0, _, _, _ = prob.seeds(sc, z[:2])
    _, _, xs, ps = prob.solve(z[:2], s3._split_trajopt(jax.nn.softmax(z[2:])),
                              sc, inner, x0)
    return jnp.mean(jnp.sum(prob.full_ee_path(sc, xs, ps, batch_index=0) ** 2, -1))

v, g = jax.jit(jax.value_and_grad(loss))(jnp.zeros(9, jnp.float32))
g = np.asarray(g)
print("full-chain loss      :", float(v))
print("full-chain grad norm :", float(np.linalg.norm(g)))
print("all finite           :", bool(np.all(np.isfinite(g))))
print("theta_ik block nonzero:", bool(np.all(np.abs(g[:2]) > 0)), np.round(g[:2], 6))
print("RESULT:", "PASS" if np.all(np.isfinite(g)) and np.linalg.norm(g) > 0 else "FAIL")
