import pytest


def _gpu_available():
    try:
        import jax
        return jax.devices()[0].platform == "gpu"
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(
                pytest.mark.skipif(not _gpu_available(), reason="no GPU available")
            )
