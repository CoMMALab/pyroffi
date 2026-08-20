"""Tests for extensions/dynamics_grid.py (GRiD CUDA dynamics backend).
Run from repo root:  PYTHONPATH=. python tests/test_dynamics_grid.py

These build per-robot CUDA kernels on first use (needs a GPU + sympy). They
assert the property the port relies on: GRiD is vmap-safe and batch-correct
(single/loop/scan/vmap agree to the bit, and match pure-JAX autodiff), so it
can be used directly in batched differentiable code. The reference for
correctness is a JAX *single-point loop*, never pure-JAX vmap (whose f32
reduction order differs and would give spurious large absolute diffs on
sum-of-squares gradients).
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import jax
import jax.numpy as jnp

import spasm.extensions.dynamics as D
import spasm.extensions.dynamics_grid as G


def test_single_point_forward_and_grad_match_jax():
    """GRiD single-point inverse dynamics + its analytic gradient must match
    the pure-JAX path (forward ~1e-5, gradient ~1e-3 absolute)."""
    q = jnp.array([0.1, -0.3, 0.2, -1.5, 0.1, 1.0, 0.5])
    qd = jnp.linspace(-0.2, 0.2, 7)
    qdd = jnp.ones(7) * 0.5

    fwd_md = float(jnp.max(jnp.abs(D.inverse_dynamics(q, qd, qdd)
                                   - G.inverse_dynamics_grid(q, qd, qdd))))
    gj = jax.grad(lambda q: jnp.sum(D.DYN_ROBOT.inverse_dynamics(q, qd, qdd) ** 2))(q)
    gg = jax.grad(lambda q: jnp.sum(G.get_grid().inverse_dynamics(q, qd, qdd) ** 2))(q)
    grad_md = float(jnp.max(jnp.abs(gj - gg)))

    assert fwd_md < 1e-4, f"GRiD single-point forward off: {fwd_md}"
    assert grad_md < 1e-2, f"GRiD single-point gradient off: {grad_md}"
    print(f"  single-point fwd maxdiff {fwd_md:.2e}  grad maxdiff {grad_md:.2e}")
    print("PASS test_single_point_forward_and_grad_match_jax")


def test_batch_is_bit_identical_to_loop():
    """The property that makes GRiD safe in batched code: vmap / native-batch /
    scan all agree with an unrolled single-point loop *to the bit* (the
    custom_vmap rule is a single fused launch, not a batch-of-batches). Checked
    for both the forward pass and the gradient."""
    gd = G.get_grid()
    B = 128
    k = jax.random.split(jax.random.key(0), 3)
    Q = jax.random.normal(k[0], (B, 7)) * 0.3
    QD = jax.random.normal(k[1], (B, 7)) * 0.2
    QDD = jax.random.normal(k[2], (B, 7)) * 0.5

    loop = jnp.stack([gd.inverse_dynamics(Q[i], QD[i], QDD[i]) for i in range(B)])
    fwd_vmap = float(jnp.max(jnp.abs(jax.vmap(gd.inverse_dynamics)(Q, QD, QDD) - loop)))
    fwd_nat = float(jnp.max(jnp.abs(gd.inverse_dynamics(Q, QD, QDD) - loop)))

    grad_loop = jnp.stack([
        jax.grad(lambda q, i=i: jnp.sum(gd.inverse_dynamics(q, QD[i], QDD[i]) ** 2))(Q[i])
        for i in range(B)])
    grad_vmap = jax.grad(lambda Q: jnp.sum(jax.vmap(gd.inverse_dynamics)(Q, QD, QDD) ** 2))(Q)
    grad_md = float(jnp.max(jnp.abs(grad_vmap - grad_loop)))

    assert fwd_vmap == 0.0 and fwd_nat == 0.0, f"GRiD forward batch != loop: vmap {fwd_vmap}, native {fwd_nat}"
    assert grad_md == 0.0, f"GRiD gradient batch != loop: {grad_md}"
    print(f"  batch-vs-loop maxdiff: fwd(vmap) {fwd_vmap}  fwd(native) {fwd_nat}  grad(vmap) {grad_md}")
    print("PASS test_batch_is_bit_identical_to_loop")


def test_batched_grad_matches_jax_autodiff():
    """GRiD's batched analytic gradient matches pure-JAX autodiff, measured
    against a JAX single-point loop (reduction-order-stable) in *relative*
    terms -- the gradients are ~1e6-scale so only relative error is meaningful."""
    gd = G.get_grid()
    T = 70
    k = jax.random.split(jax.random.key(1), 3)
    Q = jax.random.normal(k[0], (T, 7)) * 0.3
    QD = jax.random.normal(k[1], (T, 7)) * 0.2
    QDD = jax.random.normal(k[2], (T, 7)) * 0.5

    grid = jax.grad(lambda Q: jnp.sum(jax.vmap(gd.inverse_dynamics)(Q, QD, QDD) ** 2))(Q)
    jax_loop = jnp.stack([
        jax.grad(lambda q, i=i: jnp.sum(D.DYN_ROBOT.inverse_dynamics(q, QD[i], QDD[i]) ** 2))(Q[i])
        for i in range(T)])
    rel = float(jnp.max(jnp.abs(grid - jax_loop)) / jnp.max(jnp.abs(jax_loop)))
    assert rel < 1e-4, f"GRiD batched gradient off from JAX autodiff (rel): {rel}"
    print(f"  batched grad rel-err vs JAX-loop autodiff {rel:.2e}")
    print("PASS test_batched_grad_matches_jax_autodiff")


if __name__ == '__main__':
    test_single_point_forward_and_grad_match_jax()
    test_batch_is_bit_identical_to_loop()
    test_batched_grad_matches_jax_autodiff()
    print("\nAll tests passed.")
