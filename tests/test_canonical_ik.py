"""Canonical IK: a well-posed solve with derivatives verified against FD.

Every assertion here is against FINITE DIFFERENCES or an unrolled reference,
never against another implementation of the same formula. That distinction is
the reason this module exists: the previous implicit rule was validated by
comparing two J_q sources to each other, they agreed to 1e-3, and both were ~80%
wrong against ground truth.
"""

from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pytest
import yourdfpy

import pyroffi as pk
from pyroffi.optimization_engines._canonical_ik import (
    _residual,
    canonical_ik,
    canonicalize,
)

RESOURCES = pathlib.Path(__file__).resolve().parents[1] / "resources"

pytestmark = pytest.mark.skipif(
    not jax.config.read("jax_enable_x64"),
    reason="canonical-IK derivative checks need float64; run with JAX_ENABLE_X64=1")


@pytest.fixture(scope="module")
def setup():
    urdf = yourdfpy.URDF.load(str(RESOURCES / "panda" / "panda_spherized.urdf"))
    robot = pk.Robot.from_urdf(urdf)
    ee = robot.links.names.index("panda_hand")
    n = robot.joints.num_actuated_joints
    rng = np.random.default_rng(0)
    lo = np.asarray(robot.joints.lower_limits)
    hi = np.asarray(robot.joints.upper_limits)
    targets = jnp.asarray(
        jax.vmap(lambda c: robot.forward_kinematics(c)[ee])(
            jnp.asarray(lo + (hi - lo) * rng.random((3, n)))), jnp.float64)
    qref = jnp.zeros((3, n), jnp.float64)
    # q0 must be an ACTUAL solve, not a random configuration. Canonicalisation
    # is well posed within a basin; started far from the manifold it can walk
    # onto a different self-motion branch, which is honest behaviour but makes
    # a finite-difference reference meaningless.
    from pyroffi.optimization_engines._ls_ik import ls_ik_solve_cuda_batch
    out = ls_ik_solve_cuda_batch(
        robot, ee, jaxlie.SE3(jnp.asarray(targets, jnp.float32)),
        rng_key=jax.random.PRNGKey(0),
        previous_cfgs=jnp.zeros((3, n), jnp.float32))
    q0 = jnp.asarray(jnp.asarray(getattr(out, "cfg", out)), jnp.float64)
    return robot, ee, n, targets, q0, qref


def test_canonicalize_lands_exactly_on_the_manifold(setup):
    robot, ee, n, targets, q0, qref = setup
    qc = jax.vmap(lambda q, qr, t: canonicalize(q, qr, robot, (ee,), t[None]))(
        q0, qref, targets)
    r = jax.vmap(lambda q, t: _residual(q, robot, (ee,), t[None]))(qc, targets)
    # The raw solve is only float32-accurate; canonicalisation is exact.
    assert float(jnp.max(jnp.linalg.norm(r, axis=-1))) < 1e-10


def test_canonicalize_is_well_posed(setup):
    """Independent of where it starts -- the property the KKT rule relies on."""
    robot, ee, n, targets, q0, qref = setup
    noise = jnp.asarray(np.random.default_rng(3).normal(size=q0.shape)) * 0.05
    a = jax.vmap(lambda q, qr, t: canonicalize(q, qr, robot, (ee,), t[None]))(
        q0, qref, targets)
    b = jax.vmap(lambda q, qr, t: canonicalize(q, qr, robot, (ee,), t[None]))(
        q0 + noise, qref, targets)
    assert float(jnp.max(jnp.linalg.norm(a - b, axis=-1))) < 1e-6


def _dir(setup):
    _, _, _, targets, _, _ = setup
    v = jnp.asarray(np.random.default_rng(7).normal(size=targets.shape))
    return v / jnp.linalg.norm(v)


def test_first_order_is_exact_against_finite_differences(setup):
    """The claim the old rule failed: dq*/dt must match FD, not another formula."""
    robot, ee, n, targets, q0, qref = setup
    v = _dir(setup)

    def loss(t, unrolled):
        return jnp.sum(canonical_ik(q0, qref, robot, ee, jaxlie.SE3(t),
                                    unrolled=unrolled) ** 2)

    g_imp = float(jnp.sum(jax.grad(lambda t: loss(t, False))(targets) * v))
    g_unr = float(jnp.sum(jax.grad(lambda t: loss(t, True))(targets) * v))

    h = 1e-4
    fd = float((loss(targets + h * v, True) - loss(targets - h * v, True)) / (2 * h))

    assert abs(g_imp - fd) / abs(fd) < 1e-5, f"implicit {g_imp} vs FD {fd}"
    assert abs(g_unr - fd) / abs(fd) < 1e-5, f"unrolled {g_unr} vs FD {fd}"


def test_unrolled_hessian_is_exact_against_finite_differences(setup):
    robot, ee, n, targets, q0, qref = setup
    v = _dir(setup)

    def loss(t):
        return jnp.sum(canonical_ik(q0, qref, robot, ee, jaxlie.SE3(t),
                                    unrolled=True) ** 2)

    H = jax.hessian(loss)(targets)
    vHv = float(jnp.tensordot(H, v, axes=([2, 3], [0, 1])).ravel() @ v.ravel())

    # h = 1e-4: a second difference at 1e-3 is destroyed by cancellation here
    # (it read 9.2e8 against a true 153 in the same setup during development).
    h = 1e-4
    f0, fp, fm = float(loss(targets)), float(loss(targets + h * v)), float(loss(targets - h * v))
    fd = (fp - 2 * f0 + fm) / h ** 2

    assert abs(vHv - fd) / abs(fd) < 1e-3, f"unrolled Hessian {vHv} vs FD {fd}"


def test_implicit_hessian_is_documented_as_wrong(setup):
    """Pins the limitation so it cannot be forgotten or silently relied on.

    The implicit rule's primal is a CONSTANT q* handed in from the kernel rather
    than a function of t, so higher-order terms are unrecoverable no matter how
    the tangent rule recurses. Use `unrolled=True` for curvature.

    If this ever FAILS because the two agree, someone made the implicit rule
    second-order correct -- delete this test and the caveats in _canonical_ik.
    """
    robot, ee, n, targets, q0, qref = setup
    v = _dir(setup)

    def loss(t, unrolled):
        return jnp.sum(canonical_ik(q0, qref, robot, ee, jaxlie.SE3(t),
                                    unrolled=unrolled) ** 2)

    def vHv(unrolled):
        H = jax.hessian(lambda t: loss(t, unrolled))(targets)
        return float(jnp.tensordot(H, v, axes=([2, 3], [0, 1])).ravel() @ v.ravel())

    assert abs(vHv(False) - vHv(True)) / abs(vHv(True)) > 0.1
