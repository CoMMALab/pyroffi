import pytest

try:
    from iosp import config
    from iosp.model import pickplace as pp
    from iosp.model.scenes import scene_a
    _HAS_PYROFFI = True
except ImportError:
    _HAS_PYROFFI = False

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not _HAS_PYROFFI, reason="pyroffi not installed"),
]


def test_load_and_seeds():
    prob = pp.PickPlaceProblem.load(
        str(config.URDF_PATH), str(config.SRDF_PATH), str(config.MESH_DIR)
    )
    sc = scene_a()
    x0, phase_scenes, _, _ = prob.seeds(sc, config.THETA_IK_STAR)
    assert isinstance(x0, dict)
    for p in pp.PHASES:
        assert p in x0, f"missing phase {p} in seeds output"
