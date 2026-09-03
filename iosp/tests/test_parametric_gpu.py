import pytest

import jax.numpy as jnp


@pytest.mark.gpu
def test_build_parametric_gf():
    from iosp.fit.parametric import build_parametric

    built = build_parametric(seed=0, n_iters=10, n_restarts=1)
    u0 = jnp.zeros(built["K"], dtype=jnp.float32)
    val, grad = built["gf"](u0)
    assert jnp.isfinite(val), f"loss is not finite: {val}"
    assert jnp.all(jnp.isfinite(grad)), "gradient contains non-finite values"
