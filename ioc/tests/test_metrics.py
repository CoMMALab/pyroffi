import jax.numpy as jnp
import pytest

from ioc.metrics import cosine, simplex_metrics


class TestCosine:
    def test_identical(self):
        a = jnp.array([1.0, 2.0, 3.0])
        assert cosine(a, a) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal(self):
        a = jnp.array([1.0, 0.0])
        b = jnp.array([0.0, 1.0])
        assert cosine(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite(self):
        a = jnp.array([1.0, 0.0])
        assert cosine(a, -a) == pytest.approx(-1.0, abs=1e-6)

    def test_scale_invariant(self):
        a = jnp.array([1.0, 2.0])
        b = jnp.array([2.0, 4.0])
        assert cosine(a, b) == pytest.approx(1.0, abs=1e-6)


class TestSimplexMetrics:
    def test_identical(self):
        theta = jnp.array([0.5, 0.3, 0.2])
        l1, cos = simplex_metrics(theta, theta)
        assert l1 == pytest.approx(0.0, abs=1e-6)
        assert cos == pytest.approx(1.0, abs=1e-6)

    def test_opposite_vertices(self):
        a = jnp.array([1.0, 0.0, 0.0])
        b = jnp.array([0.0, 1.0, 0.0])
        l1, cos = simplex_metrics(a, b)
        assert l1 == pytest.approx(2.0, abs=1e-6)
        assert cos == pytest.approx(0.0, abs=1e-6)

    def test_uniform(self):
        a = jnp.array([1 / 3, 1 / 3, 1 / 3])
        b = jnp.array([1 / 3, 1 / 3, 1 / 3])
        l1, _ = simplex_metrics(a, b)
        assert l1 == pytest.approx(0.0, abs=1e-6)
