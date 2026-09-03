import pytest
import jax.numpy as jnp

try:
    from iosp.model.scenes import scene_a, scene_b, scenes_ab
    _HAS_PYROFFI = True
except ImportError:
    _HAS_PYROFFI = False

pytestmark = pytest.mark.skipif(not _HAS_PYROFFI, reason="pyroffi not installed")


def _check_scene_structure(sc, batch_dim):
    assert sc.q_start.shape[0] == batch_dim
    assert sc.pick_pos.shape[0] == batch_dim
    assert sc.place_pos.shape[0] == batch_dim
    assert sc.obs_center.shape[0] == batch_dim
    assert sc.obs_radius.shape[0] == batch_dim
    for field in (sc.q_start, sc.pick_pos, sc.place_pos, sc.obs_center, sc.obs_radius):
        assert field.dtype == jnp.float32


def test_scene_a():
    sc = scene_a()
    _check_scene_structure(sc, 1)


def test_scene_b():
    sc = scene_b()
    _check_scene_structure(sc, 1)


def test_scene_b_differs_from_a():
    a = scene_a()
    b = scene_b()
    assert not jnp.allclose(a.q_start, b.q_start)


def test_scenes_ab():
    sc = scenes_ab()
    _check_scene_structure(sc, 2)
