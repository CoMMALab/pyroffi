import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ioc.outer import (adam, adam_multi, adam_scan, cma_es, cma_es_multi,
                       fd_grad_fn, fd_grad_multi_fn, fit_index_for,
                       summed_grad_fn)


def _quadratic_loss_and_grad(z):
    val = jnp.sum(z ** 2)
    return val, 2.0 * z


class TestAdam:
    def test_loss_decreases(self):
        z0 = jnp.ones(3)
        best_z, trace = adam(_quadratic_loss_and_grad, z0, lr=0.01, n_steps=50)
        assert len(trace) == 50
        assert trace[-1][1] < trace[0][1]
        assert float(jnp.sum(best_z ** 2)) < float(jnp.sum(z0 ** 2))

    def test_budget_solves(self):
        z0 = jnp.ones(3)
        _, trace = adam(
            _quadratic_loss_and_grad, z0, lr=0.01,
            budget_solves=10, solves_per_step=3,
        )
        total_solves = trace[-1][0]
        assert total_solves <= 10

    def test_trace_best_monotonic(self):
        z0 = jnp.ones(3)
        _, trace = adam(
            _quadratic_loss_and_grad, z0, lr=0.01,
            n_steps=30, trace_best=True,
        )
        vals = [v for _, v in trace]
        for i in range(1, len(vals)):
            assert vals[i] <= vals[i - 1] + 1e-7

    def test_mutual_exclusion(self):
        z0 = jnp.ones(3)
        with pytest.raises(ValueError, match="exactly one"):
            adam(_quadratic_loss_and_grad, z0, lr=0.01,
                 n_steps=10, budget_solves=100)
        with pytest.raises(ValueError, match="exactly one"):
            adam(_quadratic_loss_and_grad, z0, lr=0.01)


class TestAdamScan:
    def test_shape_and_decrease(self):
        z0 = jnp.ones(3)
        loss_and_grad = jax.jit(_quadratic_loss_and_grad)
        z_final, losses = adam_scan(loss_and_grad, z0, lr=0.01, n_steps=20)
        assert z_final.shape == (3,)
        assert losses.shape == (20,)
        assert float(losses[-1]) < float(losses[0])


class TestFdGradFn:
    def test_gradient_accuracy(self):
        loss = lambda z: jnp.sum(z ** 2)
        grad_fn = fd_grad_fn(loss, eps=1e-4)
        z = jnp.array([1.0, 2.0, 3.0])
        val, g = grad_fn(z)
        assert val == pytest.approx(14.0, abs=1e-3)
        np.testing.assert_allclose(np.asarray(g), [2.0, 4.0, 6.0], atol=0.01)

    def test_unbatched(self):
        loss = lambda z: jnp.sum(z ** 2)
        grad_fn = fd_grad_fn(loss, eps=1e-4, batched=False)
        z = jnp.array([1.0, 1.0])
        val, g = grad_fn(z)
        assert val == pytest.approx(2.0, abs=1e-3)
        np.testing.assert_allclose(np.asarray(g), [2.0, 2.0], atol=1e-3)


class TestCmaEs:
    def test_finds_minimum(self):
        loss = lambda z: jnp.sum(z ** 2)
        z0 = jnp.ones(3)
        best_z, trace = cma_es(loss, z0, sigma0=0.5, n_gens=30)
        assert float(jnp.sum(best_z ** 2)) < float(jnp.sum(z0 ** 2))

    def test_budget_solves(self):
        loss = lambda z: jnp.sum(z ** 2)
        z0 = jnp.ones(3)
        _, trace = cma_es(loss, z0, sigma0=0.5, budget_solves=100)
        total_solves = trace[-1][0]
        assert total_solves <= 100

    def test_mutual_exclusion(self):
        loss = lambda z: jnp.sum(z ** 2)
        z0 = jnp.ones(3)
        with pytest.raises(ValueError, match="exactly one"):
            cma_es(loss, z0, sigma0=0.5, n_gens=10, budget_solves=100)
        with pytest.raises(ValueError, match="exactly one"):
            cma_es(loss, z0, sigma0=0.5)


def _rosen_like(z):
    """A non-quadratic loss, so a wrong step order actually shows up."""
    return jnp.sum(jnp.sin(z) ** 2 + 0.3 * z ** 2)


def _rosen_loss_and_grad(z):
    return _rosen_like(z), jax.grad(_rosen_like)(z)


def _reference_adam(loss_and_grad, z0, *, lr, n_steps=None, budget_solves=None,
                    solves_per_step=0, max_steps=100_000, trace_best=False):
    """The pre-scan Python loop, kept as the oracle for `adam`'s semantics.

    `adam` resolves both stopping rules to a step count and runs a fixed-length
    `lax.scan`; this asserts that rewrite did not move the returned iterate, the
    solve column of the trace, or the best-so-far convention.
    """
    import optax

    opt = optax.adamw(lr, weight_decay=0.0)
    z, st = z0, opt.init(z0)
    best_val, best_z = jnp.asarray(np.inf, dtype=z0.dtype), z0
    vals, used_col, used, t = [], [], 0, 0
    while True:
        if n_steps is not None:
            if t >= n_steps:
                break
        elif used + solves_per_step > budget_solves or t >= max_steps:
            break
        t += 1
        val, g = loss_and_grad(z)
        used += solves_per_step
        better = val < best_val
        best_val = jnp.where(better, val, best_val)
        best_z = jnp.where(better, z, best_z)
        vals.append(float(best_val if trace_best else val))
        used_col.append(used)
        upd, st = opt.update(g, st, z)
        z = optax.apply_updates(z, upd)
    return best_z, list(zip(used_col, vals))


class TestAdamMatchesReference:
    @pytest.mark.parametrize("kw", [
        dict(n_steps=37),
        dict(n_steps=37, trace_best=True),
        dict(n_steps=0),
        dict(budget_solves=100, solves_per_step=7),
        dict(budget_solves=100, solves_per_step=7, trace_best=True),
        dict(budget_solves=50, solves_per_step=0, max_steps=13),
    ])
    def test_equivalent_to_python_loop(self, kw):
        z0 = jnp.array([0.7, -1.3, 2.1, 0.05])
        z_ref, tr_ref = _reference_adam(_rosen_loss_and_grad, z0, lr=0.05, **kw)
        z_new, tr_new = adam(_rosen_loss_and_grad, z0, lr=0.05, **kw)
        assert [u for u, _ in tr_ref] == [u for u, _ in tr_new]
        np.testing.assert_allclose([v for _, v in tr_ref], [v for _, v in tr_new],
                                   rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np.asarray(z_ref), np.asarray(z_new),
                                   rtol=1e-5, atol=1e-6)

    def test_zero_steps_returns_start(self):
        z0 = jnp.ones(3)
        best_z, trace = adam(_quadratic_loss_and_grad, z0, lr=0.01, n_steps=0)
        assert trace == []
        np.testing.assert_array_equal(np.asarray(best_z), np.asarray(z0))

    def test_trace_values_are_python_floats(self):
        _, trace = adam(_quadratic_loss_and_grad, jnp.ones(3), lr=0.01, n_steps=5)
        assert all(type(u) is int and type(v) is float for u, v in trace)


class TestAdamScanBest:
    def test_return_best_matches_adam(self):
        """`return_best=True` is the convention the analytic baselines publish."""
        z0 = jnp.array([0.7, -1.3, 2.1, 0.05])
        z_adam, _ = adam(_rosen_loss_and_grad, z0, lr=0.05, n_steps=40)
        z_scan, losses = adam_scan(_rosen_loss_and_grad, z0, lr=0.05, n_steps=40,
                                   return_best=True)
        assert losses.shape == (40,)
        np.testing.assert_allclose(np.asarray(z_adam), np.asarray(z_scan),
                                   rtol=1e-6, atol=1e-7)

    def test_final_is_the_default(self):
        z0 = jnp.array([0.7, -1.3, 2.1, 0.05])
        z_final, _ = adam_scan(_rosen_loss_and_grad, z0, lr=0.05, n_steps=40)
        z_best, _ = adam_scan(_rosen_loss_and_grad, z0, lr=0.05, n_steps=40,
                              return_best=True)
        assert z_final.shape == z_best.shape

    def test_vmappable(self):
        """The reason `adam_scan` exists: many independent fits, one program."""
        Z0 = jnp.array([[0.7, -1.3], [2.1, 0.05], [-0.4, 0.9]])
        zs, losses = jax.vmap(
            lambda z: adam_scan(_rosen_loss_and_grad, z, lr=0.05, n_steps=20))(Z0)
        assert zs.shape == (3, 2) and losses.shape == (3, 20)
        assert np.all(np.asarray(losses)[:, -1] < np.asarray(losses)[:, 0])


class TestCmaEsJax:
    def test_deterministic_in_seed(self):
        loss = lambda z: jnp.sum(z ** 2)
        z0 = jnp.ones(3)
        a, ta = cma_es(loss, z0, sigma0=0.5, n_gens=10, seed=3)
        b, tb = cma_es(loss, z0, sigma0=0.5, n_gens=10, seed=3)
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        assert ta == tb

    def test_seeds_differ(self):
        loss = lambda z: jnp.sum(z ** 2)
        a, _ = cma_es(loss, jnp.ones(3), sigma0=0.5, n_gens=5, seed=0)
        b, _ = cma_es(loss, jnp.ones(3), sigma0=0.5, n_gens=5, seed=1)
        assert not np.allclose(np.asarray(a), np.asarray(b))

    def test_trace_best_monotonic(self):
        loss = lambda z: jnp.sum(z ** 2)
        _, trace = cma_es(loss, jnp.ones(3), sigma0=0.5, n_gens=20, trace_best=True)
        vals = [v for _, v in trace]
        assert all(vals[i] <= vals[i - 1] + 1e-7 for i in range(1, len(vals)))

    def test_unbatched_eval_matches_batched(self):
        """`batched_eval=False` uses `lax.map`, not a host loop; same values."""
        loss = lambda z: jnp.sum(z ** 2)
        a, ta = cma_es(loss, jnp.ones(3), sigma0=0.5, n_gens=8, seed=0)
        b, tb = cma_es(loss, jnp.ones(3), sigma0=0.5, n_gens=8, seed=0,
                       batched_eval=False)
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5,
                                   atol=1e-6)
        np.testing.assert_allclose([v for _, v in ta], [v for _, v in tb],
                                   rtol=1e-5, atol=1e-6)

    def test_trace_values_are_python_floats(self):
        loss = lambda z: jnp.sum(z ** 2)
        _, trace = cma_es(loss, jnp.ones(3), sigma0=0.5, n_gens=4)
        assert all(type(u) is int and type(v) is float for u, v in trace)


class TestAdamMulti:
    """The flattening: C independent fits as rows of one array."""

    @staticmethod
    def _rows(centres):
        """Row c is a quadratic centred at `centres[c]` -- independent by construction."""
        def rows(U):
            return jnp.sum((U - centres) ** 2, axis=-1)
        return rows

    def test_matches_running_each_fit_alone(self):
        centres = jnp.array([[1.0, 0.0], [0.0, 2.0], [-1.0, -1.0]])
        U0 = jnp.zeros((3, 2))
        rows = self._rows(centres)
        Z, traces = adam_multi(summed_grad_fn(rows), U0, lr=0.1, n_steps=60)
        for c in range(3):
            lg = lambda z, c=c: (jnp.sum((z - centres[c]) ** 2),
                                 2.0 * (z - centres[c]))
            z_alone, tr_alone = adam(lg, U0[c], lr=0.1, n_steps=60)
            np.testing.assert_allclose(np.asarray(Z[c]), np.asarray(z_alone),
                                       rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose([v for _, v in traces[c]],
                                       [v for _, v in tr_alone],
                                       rtol=1e-5, atol=1e-6)

    def test_summed_grad_is_per_row_gradient(self):
        """One gradient of the SUM must equal each row's own gradient."""
        centres = jnp.array([[1.0, 0.0], [0.0, 2.0], [-1.0, -1.0]])
        U = jnp.array([[0.3, -0.2], [1.1, 0.7], [-0.5, 0.4]])
        vals, grads = summed_grad_fn(self._rows(centres))(U)
        np.testing.assert_allclose(np.asarray(grads),
                                   np.asarray(2.0 * (U - centres)),
                                   rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(np.asarray(vals),
                                   np.asarray(jnp.sum((U - centres) ** 2, -1)),
                                   rtol=1e-6, atol=1e-7)

    def test_budget_is_per_fit(self):
        """Batching must not change how many solves each fit is charged."""
        centres = jnp.zeros((4, 2))
        _, traces = adam_multi(summed_grad_fn(self._rows(centres)),
                              jnp.ones((4, 2)), lr=0.1, budget_solves=100,
                              solves_per_step=7)
        for tr in traces:
            assert tr[-1][0] <= 100
            assert len(tr) == 100 // 7


class TestFdGradMulti:
    def test_matches_single_fit_fd(self):
        centres = jnp.array([[1.0, 0.0, 0.5], [0.0, 2.0, -1.0]])
        K, n_fits = 3, 2

        def rows(P):
            idx = jnp.asarray(fit_index_for(n_fits, K + 1))
            return jnp.sum((P - centres[idx]) ** 2, axis=-1)

        U = jnp.array([[0.3, -0.2, 0.1], [1.1, 0.7, 0.0]])
        vals, grads = fd_grad_multi_fn(rows, 1e-4, n_fits, K)(U)
        np.testing.assert_allclose(np.asarray(vals),
                                   np.asarray(jnp.sum((U - centres) ** 2, -1)),
                                   rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(np.asarray(grads),
                                   np.asarray(2.0 * (U - centres)),
                                   rtol=1e-2, atol=1e-3)

    def test_fit_index_layout(self):
        g = fd_grad_multi_fn(lambda P: jnp.zeros(P.shape[0]), 1e-4, 3, 4)
        np.testing.assert_array_equal(g.fit_index, np.repeat(np.arange(3), 5))


class TestCmaEsMulti:
    def test_finds_each_minimum(self):
        centres = jnp.array([[1.0, 0.0], [0.0, 2.0], [-1.0, -1.0]])

        def rows(X):
            lam = X.shape[0] // 3
            return jnp.sum((X - centres[jnp.repeat(jnp.arange(3), lam)]) ** 2, -1)

        Z, traces = cma_es_multi(rows, jnp.zeros((3, 2)), n_gens=60, sigma0=0.5)
        np.testing.assert_allclose(np.asarray(Z), np.asarray(centres),
                                   rtol=0, atol=1e-3)
        assert len(traces) == 3 and len(traces[0]) == 60

    def test_deterministic(self):
        rows = lambda X: jnp.sum(X ** 2, -1)
        a, ta = cma_es_multi(rows, jnp.ones((2, 3)), n_gens=8, seed=1)
        b, tb = cma_es_multi(rows, jnp.ones((2, 3)), n_gens=8, seed=1)
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        assert ta == tb

    def test_budget_is_per_fit(self):
        rows = lambda X: jnp.sum(X ** 2, -1)
        _, traces = cma_es_multi(rows, jnp.ones((2, 3)), budget_solves=100,
                                 solves_per_eval=1)
        for tr in traces:
            assert tr[-1][0] <= 100
