"""Native collision residuals (``costs/_native.py``) against a JAX reference.

The native rows are ``sum max(0, margin - d)`` over sphere pairs of the active link pairs
(self) or over spheres x obstacles (world). Their value is read back as the constraint
violation of a zero-iteration solve; their Jacobian from one LM step on the same row as a
least-squares residual, which for a single row points along ``-J``.
"""

import shutil

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yourdfpy

if shutil.which("nvcc") is None or jax.default_backend() != "gpu":
    pytest.skip("native residuals need nvcc and a GPU", allow_module_level=True)

import pyroffi as pk
from cudals import Cost
from pyroffi._robot_srdf_parser import read_disabled_collisions_from_srdf
from pyroffi.collision import Sphere, collide
from pyroffi.costs._native import _collision_source

MARGIN = 0.02


@pytest.fixture(scope="module")
def scene():
    urdf = yourdfpy.URDF.load("resources/panda/panda_spherized.urdf", mesh_dir="resources/panda/meshes")
    robot = pk.Robot.from_urdf(urdf)
    ignore = tuple((p["link1"], p["link2"]) for p in read_disabled_collisions_from_srdf("resources/panda/panda.srdf"))
    coll = pk.collision.RobotCollisionSpherized.from_urdf(urdf, user_ignore_pairs=ignore)
    lo, hi = np.asarray(robot.joints.lower_limits), np.asarray(robot.joints.upper_limits)
    q = jnp.asarray(np.random.default_rng(0).uniform(lo, hi, (16, lo.shape[0])), jnp.float32)
    mid = robot.forward_kinematics(q[0])[robot.links.names.index("panda_link4")][4:]
    world = (Sphere.from_center_and_radius(mid[None], jnp.array([0.15])),)
    return robot, coll, q, world


def _reference(mode, robot, coll, world):
    """JAX ``sum max(0, margin - d)`` with the native's pairing."""
    def h(q):
        g = coll.at_config(robot, q)                       # (S, N)
        c, r = g.pose.translation().reshape(-1, 3), g.radius.reshape(-1)
        N = g.radius.shape[-1]
        link = jnp.arange(c.shape[0]) % N
        live = r > 0
        if mode == "self":
            on = np.zeros((N, N), bool)
            on[np.asarray(coll.active_idx_i), np.asarray(coll.active_idx_j)] = True
            pair = jnp.asarray(on)[link[:, None], link[None, :]] & live[:, None] & live[None, :]
            d = jnp.linalg.norm(c[:, None] - c[None, :] + 1e-12, axis=-1) - r[:, None] - r[None, :]
            return jnp.sum(jnp.where(pair, jnp.maximum(MARGIN - d, 0.0), 0.0))
        spheres = Sphere.from_center_and_radius(c, r)
        d = jax.vmap(lambda s: jax.vmap(lambda o: collide(s, o))(world[0]))(spheres)
        return jnp.sum(jnp.where(live[:, None], jnp.maximum(MARGIN - d, 0.0), 0.0))
    return h


def _native(mode, robot, coll, world):
    return _collision_source(robot, coll, world if mode == "world" else (), MARGIN, tag=f"t{mode}", mode=mode)


def _solve(costs, var, q, **settings):
    a = pk.LeastSquaresProblem(costs, [var]).analyze(structure="dense")
    return a.solve_batch(a.params(), q, tier="thread", **settings)


@pytest.mark.parametrize("mode", ["self", "world"])
def test_value_matches_jax(scene, mode):
    robot, coll, q, world = scene
    ctor = (lambda v: pk.costs.self_collision_native(robot, coll, v, MARGIN)) if mode == "self" \
        else (lambda v: pk.costs.world_collision_native(robot, coll, v, world, MARGIN))
    var = robot.joint_var_cls(0)
    _, _, viol = _solve([ctor(var)], var, q, max_iters=0, al_iters=0)
    want = jax.vmap(_reference(mode, robot, coll, world))(q)
    assert float(jnp.max(want)) > 1e-3, "scene must have active collision terms"
    np.testing.assert_allclose(np.asarray(viol), np.asarray(want), rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("mode", ["self", "world"])
def test_jacobian_matches_jax(scene, mode):
    robot, coll, q, world = scene
    nat = _native(mode, robot, coll, world)
    var = robot.joint_var_cls(0)
    row = Cost(compute_residual=lambda vals, a, kw: jnp.zeros(1) + 0.0 * jnp.sum(vals[a[0]]),
               args=((var,), {}), name=f"native_row_{mode}", native=nat)
    h = _reference(mode, robot, coll, world)
    grad = np.asarray(jax.vmap(jax.grad(h))(q))
    x, _, _ = _solve([row], var, q, max_iters=1, lambda_initial=1.0)
    step = np.asarray(x - q)
    # A pair that overlaps rigidly (no joint between its spheres) is active with zero gradient:
    # there the native's step is zero too.
    flat = np.linalg.norm(grad, axis=1) < 1e-5
    np.testing.assert_array_equal(step[flat], 0.0)
    moving = ~flat & (np.asarray(jax.vmap(h)(q)) > 1e-3)
    assert moving.sum() >= 4, "need colliding configurations"
    s, g = step[moving], grad[moving]
    cos = np.sum(s * -g, 1) / (np.linalg.norm(s, axis=1) * np.linalg.norm(g, axis=1))
    assert cos.min() > 0.999, cos
