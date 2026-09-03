"""The batched readouts and the per-row solve must match what they replaced.

These are equivalence tests, not behaviour tests: `full_ee_paths` /
`full_joint_paths` / `solve_batched_theta` exist purely to stop a batch axis
from being unrolled into the graph as a Python loop, so the only thing worth
asserting is that they return exactly the loop's values.
"""

import numpy as np
import pytest

try:
    import jax
    import jax.numpy as jnp
    from iosp import config
    from iosp.fit import parametric as s3
    from iosp.model import pickplace as pp
    from iosp.model.scenes import scene_a, scenes_ab
    _HAS_PYROFFI = True
except ImportError:
    _HAS_PYROFFI = False

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not _HAS_PYROFFI, reason="pyroffi not installed"),
]


@pytest.fixture(scope="module")
def solved():
    prob = pp.PickPlaceProblem.load(
        str(config.URDF_PATH), str(config.SRDF_PATH), str(config.MESH_DIR)
    )
    scenes = scenes_ab()
    fs = pp.make_composed_forward_solver(n_iters=20)
    inner, _ = s3._build_inner(prob, scene_a(), config.THETA_IK_STAR, fs, 0)
    theta_traj = jax.nn.softmax(config.Z_TRAJOPT_STAR)
    x0, _, _, _ = prob.seeds(scenes, config.THETA_IK_STAR)
    _, _, xs, ps = prob.solve(
        config.THETA_IK_STAR, s3._split_trajopt(theta_traj), scenes, inner, x0
    )
    return prob, scenes, inner, x0, theta_traj, xs, ps


def test_full_ee_paths_matches_loop(solved):
    """Compared INSIDE ONE jit, and that is not pedantry.

    The EE readout is float32 forward kinematics, and it moves by ~3e-4 m
    between compilation/precision contexts -- the looped form moves by exactly
    as much as the batched one.  So comparing a jitted call against an eager one
    measures the FK's precision sensitivity, not whether the batching is right;
    only a same-context comparison isolates the thing under test.  Bit-equality
    is the correct bar once the context is matched.
    """
    prob, scenes, _, _, _, xs, ps = solved
    B = scenes.q_start.shape[0]

    @jax.jit
    def both():
        looped = jnp.stack(
            [prob.full_ee_path(scenes, xs, ps, batch_index=i) for i in range(B)])
        return prob.full_ee_paths(scenes, xs, ps), looped

    batched, looped = both()
    assert batched.shape == looped.shape
    np.testing.assert_array_equal(np.asarray(looped), np.asarray(batched))


def test_full_joint_paths_matches_loop(solved):
    prob, scenes, _, _, _, xs, ps = solved
    B = scenes.q_start.shape[0]

    @jax.jit
    def both():
        looped = jnp.stack(
            [prob.full_joint_path(scenes, xs, ps, batch_index=i) for i in range(B)])
        return prob.full_joint_paths(scenes, xs, ps), looped

    batched, looped = both()
    assert batched.shape == looped.shape
    np.testing.assert_array_equal(np.asarray(looped), np.asarray(batched))


def test_solve_batched_theta_matches_solve(solved):
    """With the SAME theta on every row, the per-row solve is plain `solve`."""
    prob, scenes, inner, x0, theta_traj, xs, ps = solved
    B = scenes.q_start.shape[0]
    by_phase = s3._split_trajopt(theta_traj)
    rep = {p: jnp.repeat(by_phase[p][None], B, axis=0) for p in pp.PHASES}
    _, _, xs_b, ps_b = prob.solve_batched_theta(
        config.THETA_IK_STAR, rep, scenes, inner, x0)
    for p in pp.PHASES:
        np.testing.assert_allclose(np.asarray(xs[p]), np.asarray(xs_b[p]),
                                   rtol=1e-5, atol=1e-6)


def test_solve_batched_theta_is_row_independent(solved):
    """Row c's solution must not depend on what the other rows' costs are.

    This is the property the flattened multistart is built on: if it failed,
    one gradient of a summed loss would not be each row's own gradient.
    """
    prob, scenes, inner, x0, theta_traj, _, _ = solved
    B = scenes.q_start.shape[0]
    by_phase = s3._split_trajopt(theta_traj)
    other = s3._split_trajopt(jax.nn.softmax(config.Z_TRAJOPT_STAR * 0.5 + 0.3))

    def run(row1):
        rep = {p: jnp.stack([by_phase[p], row1[p]]) for p in pp.PHASES}
        sc = jax.tree.map(lambda a: a[:2], scenes)
        x0b = {p: x0[p][:2] for p in pp.PHASES}
        _, _, xs_b, _ = prob.solve_batched_theta(
            config.THETA_IK_STAR, rep, sc, inner, x0b)
        return xs_b

    a, b = run(by_phase), run(other)
    for p in pp.PHASES:
        np.testing.assert_allclose(np.asarray(a[p][0]), np.asarray(b[p][0]),
                                   rtol=1e-5, atol=1e-6,
                                   err_msg=f"row 0 of {p} moved when row 1's cost changed")
