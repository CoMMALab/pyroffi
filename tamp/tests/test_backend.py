"""Numeric checks for backend.py against the original SPaSM kinematics."""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import jax
import jax.numpy as jnp

from spasm import backend

from spasm.paths import SPASM_STOCK_ROOT as _SPASM


def _random_qs(n=10, seed=0):
    lo, hi = backend.get_joint_limits()
    rng = np.random.default_rng(seed)
    # Sample away from the extremes to stay in the analytic-IK friendly range.
    return jnp.array(rng.uniform(np.asarray(lo) * 0.9, np.asarray(hi) * 0.9, size=(n, 7)))


def test_fk_spheres_sane():
    q0 = jnp.zeros(7)
    pos, radii = backend.fk(q0)
    assert pos.shape == (backend.NUM_SPHERES, 3)
    assert radii.shape == (backend.NUM_SPHERES,)
    # SPaSM's URDF defines 59 collision spheres.
    assert backend.NUM_SPHERES == 59, backend.NUM_SPHERES
    r = np.asarray(radii)
    assert (r > 0.0).all() and (r < 0.2).all(), r

    # Positions must move with q.
    q1 = jnp.array([0.5, -0.4, 0.3, -1.5, 0.2, 1.2, 0.1])
    pos1, radii1 = backend.fk(q1)
    assert np.allclose(np.asarray(radii), np.asarray(radii1))
    assert np.abs(np.asarray(pos1) - np.asarray(pos)).max() > 0.05

    # Batched path agrees with single path.
    posb, radiib = backend.fk_batched(jnp.stack([q0, q1]))
    assert posb.shape == (2, backend.NUM_SPHERES, 3)
    assert np.allclose(np.asarray(posb[1]), np.asarray(pos1), atol=1e-5)
    print("test_fk_spheres_sane passed")


def test_ee_pose_matches_original():
    # Run the original pure-JAX kinematics in-process.
    cwd = os.getcwd()
    sys.path.insert(0, _SPASM)
    os.chdir(_SPASM)  # original uses relative urdf path
    try:
        from kinematics.kinematics import get_ee_pose as orig_get_ee_pose
        qs = _random_qs(10)
        for q in qs:
            ours = np.asarray(backend.get_ee_pose(q))
            theirs = np.asarray(orig_get_ee_pose(jnp.pad(q, (0, 2))))
            err = np.abs(ours - theirs).max()
            assert err < 1e-4, (err, ours, theirs)
    finally:
        os.chdir(cwd)
        sys.path.remove(_SPASM)
    print("test_ee_pose_matches_original passed")


def test_analytic_ik_roundtrip():
    qs = _random_qs(10, seed=1)
    neutral = jnp.array([0., -jnp.pi/4, 0., -2*jnp.pi/4, 0., jnp.pi/2, jnp.pi/4])
    n_checked = 0
    for q in qs:
        pose = backend.get_ee_pose(q)
        q_sol = backend.ik(pose, q)
        if not np.isfinite(np.asarray(q_sol)).all():
            continue  # no valid analytic solution for this target
        pose_rt = np.asarray(backend.get_ee_pose(q_sol))
        err = np.abs(pose_rt - np.asarray(pose)).max()
        assert err < 5e-3, (err, q, q_sol)
        n_checked += 1
    assert n_checked >= 5, f"too few valid IK round-trips ({n_checked}/10)"
    print(f"test_analytic_ik_roundtrip passed ({n_checked}/10 targets valid)")


def test_ik_numeric_crosscheck():
    # pyroffi numeric IK should reach the same EE pose as the analytic solver.
    q = jnp.array([0., -jnp.pi/4, 0., -2*jnp.pi/4, 0., jnp.pi/2, jnp.pi/4])
    pose = backend.get_ee_pose(q)
    q_num = backend.ik_numeric(pose, q_ref=q)
    pose_num = np.asarray(backend.get_ee_pose(q_num))
    err = np.abs(pose_num - np.asarray(pose)).max()
    assert err < 1e-2, err
    print("test_ik_numeric_crosscheck passed")


if __name__ == '__main__':
    test_fk_spheres_sane()
    test_ee_pose_matches_original()
    test_analytic_ik_roundtrip()
    test_ik_numeric_crosscheck()
    print("ALL TESTS PASSED")
