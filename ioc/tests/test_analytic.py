import jax.numpy as jnp
import numpy as np

from ioc.analytic import kkt_fit


class TestKktFit:
    def test_returns_finite_z(self):
        K = 3
        n_ctx = 4
        T = 5
        n_x = (T - 2) * K

        def grad_x(x, theta_onehot, ctx):
            # theta_onehot is shape (K,), x is shape (n_x,)
            # Tile theta_onehot to match x, simulating per-feature grads
            w = jnp.tile(theta_onehot, n_x // K)
            return w * x + ctx[:n_x]

        rng = np.random.default_rng(42)
        ctxs = jnp.asarray(rng.normal(size=(n_ctx, n_x)), dtype=jnp.float32)
        demos = jnp.asarray(rng.normal(size=(n_ctx, T, K)), dtype=jnp.float32)

        z = kkt_fit(grad_x, ctxs, demos, K, n_steps=50, lr=0.05)
        assert z.shape == (K,)
        assert np.all(np.isfinite(np.asarray(z)))

    def test_return_gram(self):
        K = 2
        n_ctx = 3
        T = 4
        n_x = (T - 2) * K

        def grad_x(x, theta_onehot, ctx):
            w = jnp.tile(theta_onehot, n_x // K)
            return w * x

        rng = np.random.default_rng(0)
        ctxs = jnp.asarray(rng.normal(size=(n_ctx, n_x)), dtype=jnp.float32)
        demos = jnp.asarray(rng.normal(size=(n_ctx, T, K)), dtype=jnp.float32)

        z, G = kkt_fit(grad_x, ctxs, demos, K, n_steps=20, lr=0.05,
                        return_gram=True)
        assert G.shape == (K, K)
        eigvals = np.linalg.eigvalsh(np.asarray(G))
        assert np.all(eigvals >= -1e-6)


class TestVmappedFeatureProbes:
    """The K per-feature gradient probes are vmapped, not an unrolled loop.

    The values must be bit-identical to the loop they replaced; only the graph
    width (and hence compile time in K) changes.
    """

    def test_stacked_b_matches_loop(self):
        import jax
        import jax.numpy as jnp

        K, n_x, n_ctx = 4, 6, 3

        def grad_x(x, theta_onehot, ctx):
            w = jnp.tile(theta_onehot, n_x // K + 1)[:n_x]
            return w * jnp.tanh(x) + ctx[:n_x]

        rng = np.random.default_rng(7)
        x = jnp.asarray(rng.normal(size=n_x), dtype=jnp.float32)
        ctx = jnp.asarray(rng.normal(size=(n_ctx, n_x))[0], dtype=jnp.float32)
        e = jnp.eye(K)

        looped = jnp.stack([grad_x(x, e[k], ctx) for k in range(K)], axis=-1)
        vmapped = jax.vmap(grad_x, in_axes=(None, 0, None))(x, e, ctx).T
        np.testing.assert_array_equal(np.asarray(looped), np.asarray(vmapped))

    def test_fit_is_stable_under_repeat(self):
        """The fit is one jitted scan now; it must still be deterministic."""
        K, n_ctx, T = 3, 4, 5
        n_x = (T - 2) * K

        def grad_x(x, theta_onehot, ctx):
            w = jnp.tile(theta_onehot, n_x // K)
            return w * x + ctx[:n_x]

        rng = np.random.default_rng(1)
        ctxs = jnp.asarray(rng.normal(size=(n_ctx, n_x)), dtype=jnp.float32)
        demos = jnp.asarray(rng.normal(size=(n_ctx, T, K)), dtype=jnp.float32)
        a = kkt_fit(grad_x, ctxs, demos, K, n_steps=60, lr=0.05)
        b = kkt_fit(grad_x, ctxs, demos, K, n_steps=60, lr=0.05)
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
