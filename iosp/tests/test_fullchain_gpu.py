import numpy as np
import pytest

try:
    import jax
    import jax.numpy as jnp
    from iosp import config
    from iosp.fit import parametric as s3
    from iosp.fit.params import z_scale
    from iosp.model import pickplace as pp
    from iosp.model.scenes import scene_a, scenes_ab
    _HAS_PYROFFI = True
except ImportError:
    _HAS_PYROFFI = False

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not _HAS_PYROFFI, reason="pyroffi not installed"),
]


def test_fullchain_grad_finite():
    prob = pp.PickPlaceProblem.load(
        str(config.URDF_PATH), str(config.SRDF_PATH), str(config.MESH_DIR)
    )
    sc = scenes_ab()
    fs = pp.make_composed_forward_solver(n_iters=20)
    inner, _ = s3._build_inner(prob, scene_a(), config.THETA_IK_STAR, fs, 0)
    S = z_scale(9, 2)

    def loss(u):
        z = u * S
        x0, _, _, _ = prob.seeds(sc, z[:2])
        _, _, xs, ps = prob.solve(
            z[:2], s3._split_trajopt(jax.nn.softmax(z[2:])), sc, inner, x0
        )
        return jnp.mean(jnp.sum(prob.full_ee_path(sc, xs, ps, batch_index=0) ** 2, -1))

    v, g = jax.jit(jax.value_and_grad(loss))(jnp.zeros(9, jnp.float32))
    g_np = np.asarray(g)
    assert np.all(np.isfinite(g_np)), "gradient contains non-finite values"
    assert np.linalg.norm(g_np) > 0, "gradient is zero"
    assert np.all(np.abs(g_np[:2]) > 0), "theta_ik gradient block is zero"
