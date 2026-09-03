import jax.numpy as jnp
import numpy as np
import pytest

from ioc.identifiability import (
    refit_on_subspace,
    select_rank,
    sensitivity_spectrum,
    wide_fit,
)


class TestSelectRank:
    def test_gap_rule(self):
        eigvals = np.array([10.0, 5.0, 0.001])
        retained, discarded, r = select_rank(eigvals, rule="gap")
        assert r == 2
        assert len(retained) == 2
        assert len(discarded) == 1

    def test_gap_rule_single_cliff(self):
        eigvals = np.array([100.0, 0.0001, 0.00005])
        _, _, r = select_rank(eigvals, rule="gap")
        assert r == 1

    def test_trace_rule(self):
        eigvals = np.array([9.0, 0.8, 0.2])
        _, _, r = select_rank(eigvals, frac=0.95, rule="trace")
        assert r == 2

    def test_trace_rule_one_dominant(self):
        eigvals = np.array([9.6, 0.2, 0.2])
        _, _, r = select_rank(eigvals, frac=0.95, rule="trace")
        assert r == 1

    def test_unknown_rule(self):
        with pytest.raises(ValueError, match="unknown rule"):
            select_rank(np.array([1.0, 0.5]), rule="bogus")


class TestWideFit:
    def test_returns_z_and_trace(self):
        def loss_and_grad(u):
            val = jnp.sum(u ** 2)
            return val, 2.0 * u

        u0 = jnp.ones(4, dtype=jnp.float32)
        z, trace = wide_fit(loss_and_grad, u0, n_steps=10, lr=0.01)
        assert z.shape == (4,)
        assert len(trace) == 10


class TestRefitOnSubspace:
    def test_output_shape(self):
        def loss_and_grad(z):
            val = jnp.sum(z ** 2)
            return val, 2.0 * z

        K = 5
        r = 3
        z_prior = jnp.zeros(K, dtype=jnp.float32)
        U_r = np.linalg.qr(np.random.default_rng(0).normal(size=(K, r)))[0]
        z_hat, alpha_hat = refit_on_subspace(
            loss_and_grad, z_prior, U_r, n_steps=5, lr=0.01,
        )
        assert z_hat.shape == (K,)
        assert alpha_hat.shape == (r,)
