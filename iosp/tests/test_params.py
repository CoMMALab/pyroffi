import numpy as np
import numpy.testing as npt

from iosp.config import STANDOFF_SCALE
from iosp.fit.params import _proj_norm, gauge_fix, gauge_vector, z_scale


def test_z_scale_shape_and_values():
    s = z_scale(9, 2)
    assert s.shape == (9,)
    npt.assert_allclose(float(s[0]), STANDOFF_SCALE)
    npt.assert_allclose(float(s[1]), STANDOFF_SCALE)
    for i in range(2, 9):
        npt.assert_allclose(float(s[i]), 1.0)


def test_gauge_fix_centres_logit_block():
    z = np.array([0.1, 0.2, 1.0, 2.0, 3.0])
    fixed = gauge_fix(z, n_ik=2)
    npt.assert_allclose(fixed[:2], [0.1, 0.2])
    npt.assert_allclose(fixed[2:].sum(), 0.0, atol=1e-12)


def test_gauge_vector_shape_and_structure():
    g = gauge_vector(9, 2)
    assert g.shape == (9,)
    npt.assert_allclose(g[:2], 0.0)
    npt.assert_allclose(np.linalg.norm(g), 1.0, atol=1e-12)


def test_proj_norm():
    delta = np.ones(3)
    V = np.eye(3, 2)
    result = _proj_norm(delta, V)
    assert np.isfinite(result)
    npt.assert_allclose(result, np.sqrt(2.0), atol=1e-12)


def test_proj_norm_empty_subspace():
    delta = np.ones(3)
    V = np.zeros((3, 0))
    assert _proj_norm(delta, V) == 0.0
