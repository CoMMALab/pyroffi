import pytest


def _gpu_available():
    try:
        import jax
        return jax.devices()[0].platform == "gpu"
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    if _gpu_available():
        return
    skip_gpu = pytest.mark.skip(reason="no GPU available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
