"""Canonical subproblem correctness.

Each subproblem is checked two ways:

* **Round-trip** — build an instance from a known angle, solve it, and require
  the residual to vanish (not that the recovered angle matches, since several
  subproblems legitimately have two roots and either is correct).
* **Least-squares** — build a deliberately unsatisfiable instance and require a
  finite minimising answer with ``is_ls`` raised, rather than NaN.

Randomised over many seeds because the failure modes are geometric edge cases
(axes nearly parallel, targets nearly out of reach) that a single hand-picked
instance will not surface.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyroffi.kinematics._subproblems import (
    residual1,
    residual2,
    residual3,
    residual4,
    rot,
    subproblem1,
    subproblem2,
    subproblem3,
    subproblem4,
)

jax.config.update("jax_enable_x64", True)

SEEDS = range(64)
ATOL = 1e-8


def _unit(rng, n=3):
    v = rng.normal(size=n)
    return v / np.linalg.norm(v)


# --------------------------------------------------------------------------- #
# Subproblem 1
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("seed", SEEDS)
def test_subproblem1_roundtrip(seed):
    rng = np.random.default_rng(seed)
    k = jnp.asarray(_unit(rng))
    p1 = jnp.asarray(rng.normal(size=3))
    theta_true = float(rng.uniform(-np.pi, np.pi))
    p2 = rot(k, theta_true) @ p1

    theta, is_ls = subproblem1(p1, p2, k)
    assert not bool(is_ls)
    assert float(residual1(p1, p2, k, theta)) < ATOL


@pytest.mark.parametrize("seed", range(16))
def test_subproblem1_least_squares_flagged(seed):
    """A p2 with the wrong norm cannot be reached; expect a flagged minimiser."""
    rng = np.random.default_rng(seed)
    k = jnp.asarray(_unit(rng))
    p1 = jnp.asarray(rng.normal(size=3))
    p2 = 2.0 * rot(k, 0.3) @ p1          # norm doubled -> unreachable

    theta, is_ls = subproblem1(p1, p2, k)
    assert bool(is_ls)
    assert np.isfinite(float(theta))


# --------------------------------------------------------------------------- #
# Subproblem 4
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("seed", SEEDS)
def test_subproblem4_roundtrip(seed):
    rng = np.random.default_rng(seed)
    k = jnp.asarray(_unit(rng))
    h = jnp.asarray(_unit(rng))
    p = jnp.asarray(rng.normal(size=3))
    theta_true = float(rng.uniform(-np.pi, np.pi))
    d = jnp.dot(h, rot(k, theta_true) @ p)

    theta, valid, is_ls = subproblem4(h, p, k, d)
    assert not bool(is_ls)
    # At least one root must reproduce d; both are legitimate roots of the
    # sinusoid, and the true angle is one of them.
    res = [float(residual4(h, p, k, d, t)) for t in theta]
    assert min(res) < ATOL
    assert any(abs(float(t) - theta_true) < 1e-7 for t in theta)


@pytest.mark.parametrize("seed", range(16))
def test_subproblem4_least_squares_flagged(seed):
    rng = np.random.default_rng(seed)
    k = jnp.asarray(_unit(rng))
    h = jnp.asarray(_unit(rng))
    p = jnp.asarray(rng.normal(size=3))
    # Demand a projection larger than |p|, which no rotation can deliver.
    d = jnp.asarray(10.0 * float(np.linalg.norm(np.asarray(p))))

    theta, valid, is_ls = subproblem4(h, p, k, d)
    assert bool(is_ls)
    assert np.all(np.isfinite(np.asarray(theta)))

    # The returned angle must actually minimise the residual: perturbing it
    # in either direction cannot do better.
    t = theta[0]
    best = float(residual4(h, p, k, d, t))
    for dt in (-1e-3, 1e-3, 0.1, -0.1):
        assert float(residual4(h, p, k, d, t + dt)) >= best - 1e-9


# --------------------------------------------------------------------------- #
# Subproblem 3
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("seed", SEEDS)
def test_subproblem3_roundtrip(seed):
    rng = np.random.default_rng(seed)
    k = jnp.asarray(_unit(rng))
    p1 = jnp.asarray(rng.normal(size=3))
    p2 = jnp.asarray(rng.normal(size=3))
    theta_true = float(rng.uniform(-np.pi, np.pi))
    d = jnp.linalg.norm(rot(k, theta_true) @ p1 - p2)

    theta, valid, is_ls = subproblem3(p1, p2, k, d)
    assert not bool(is_ls)
    res = [float(residual3(p1, p2, k, d, t)) for t in theta]
    assert min(res) < ATOL


@pytest.mark.parametrize("seed", range(16))
def test_subproblem3_out_of_reach_flagged(seed):
    """The 'elbow cannot stretch that far' case must not produce NaN."""
    rng = np.random.default_rng(seed)
    k = jnp.asarray(_unit(rng))
    p1 = jnp.asarray(rng.normal(size=3))
    p2 = jnp.asarray(rng.normal(size=3))
    d = jnp.asarray(1e3)                  # absurdly far

    theta, valid, is_ls = subproblem3(p1, p2, k, d)
    assert bool(is_ls)
    assert np.all(np.isfinite(np.asarray(theta)))


# --------------------------------------------------------------------------- #
# Subproblem 2
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("seed", SEEDS)
def test_subproblem2_roundtrip(seed):
    rng = np.random.default_rng(seed)
    k1 = jnp.asarray(_unit(rng))
    k2 = jnp.asarray(_unit(rng))
    p2 = jnp.asarray(rng.normal(size=3))
    t1_true = float(rng.uniform(-np.pi, np.pi))
    t2_true = float(rng.uniform(-np.pi, np.pi))
    # Construct p1 so the equation holds exactly at (t1_true, t2_true).
    p1 = rot(k1, -t1_true) @ (rot(k2, t2_true) @ p2)

    theta1, theta2, valid, is_ls = subproblem2(p1, p2, k1, k2)
    assert not bool(is_ls)
    res = [float(residual2(p1, p2, k1, k2, a, b)) for a, b in zip(theta1, theta2)]
    assert min(res) < 1e-7


@pytest.mark.parametrize("seed", range(16))
def test_subproblem2_norm_mismatch_flagged(seed):
    rng = np.random.default_rng(seed)
    k1 = jnp.asarray(_unit(rng))
    k2 = jnp.asarray(_unit(rng))
    p1 = jnp.asarray(rng.normal(size=3))
    p2 = jnp.asarray(rng.normal(size=3)) * 5.0   # norms cannot match

    theta1, theta2, valid, is_ls = subproblem2(p1, p2, k1, k2)
    assert bool(is_ls)
    assert np.all(np.isfinite(np.asarray(theta1)))
    assert np.all(np.isfinite(np.asarray(theta2)))


# --------------------------------------------------------------------------- #
# Transform properties the solver relies on
# --------------------------------------------------------------------------- #

def test_rot_is_orthonormal():
    rng = np.random.default_rng(0)
    for _ in range(32):
        k = jnp.asarray(_unit(rng))
        th = float(rng.uniform(-np.pi, np.pi))
        R = np.asarray(rot(k, th))
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert abs(np.linalg.det(R) - 1.0) < 1e-12


def test_subproblems_are_jit_and_vmap_safe():
    """The solver wraps these in jit/vmap over batches of targets."""
    rng = np.random.default_rng(0)
    n = 8
    k = jnp.asarray(np.stack([_unit(rng) for _ in range(n)]))
    p1 = jnp.asarray(rng.normal(size=(n, 3)))
    p2 = jnp.asarray(rng.normal(size=(n, 3)))
    d = jnp.asarray(rng.uniform(0.5, 1.5, size=n))

    f = jax.jit(jax.vmap(subproblem3))
    theta, valid, is_ls = f(p1, p2, k, d)
    assert theta.shape == (n, 2)
    assert np.all(np.isfinite(np.asarray(theta)))

    g = jax.jit(jax.vmap(subproblem2))
    t1, t2, valid, is_ls = g(p1, p2, k, k)
    assert t1.shape == (n, 2) and t2.shape == (n, 2)
