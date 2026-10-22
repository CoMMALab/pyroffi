"""Floating-base robots: cfg = [base wxyz_xyz | joints], tangents [base twist | joints].

Kinematics, dynamics and integration against pinocchio's free-flyer model (which orders its
quaternion xyzw and its base twist/wrench linear-first), plus URDF detection, IK and collision.
"""

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pytest
import yourdfpy

from pyroffi import Robot, dynamics
from pyroffi import _floating_base as FB
from pyroffi.collision import RobotCollisionSpherized, Sphere

pin = pytest.importorskip("pinocchio")
G1 = "resources/g1_description/g1_29dof__dynbench.urdf"


@pytest.fixture(scope="module")
def g1():
    robot = Robot.from_urdf(yourdfpy.URDF.load(G1, load_meshes=False), floating_base=True)
    model = pin.buildModelFromUrdf(G1, pin.JointModelFreeFlyer())
    act = list(robot.joints.actuated_names)
    perm = np.array([act.index(model.names[j]) for j in range(2, model.njoints)])  # joint 1: base
    return robot, model, perm


def _states(robot, rng, k):
    for _ in range(k):
        base = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.exp(jnp.asarray(rng.normal(size=3), jnp.float32)), jnp.asarray(rng.normal(size=3), jnp.float32))
        cfg = np.concatenate([np.asarray(base.wxyz_xyz), rng.uniform(-0.5, 0.5, robot.joints.num_actuated_joints)])
        yield cfg.astype(np.float32), *(rng.normal(size=robot.nv).astype(np.float32) for _ in range(3))


def _pin_q(cfg, perm):
    w, x, y, z = cfg[:4]
    return np.concatenate([cfg[4:7], [x, y, z, w], cfg[7:][perm]]).astype(np.float64)


def _pin_v(v, perm):
    return np.concatenate([v[3:6], v[:3], v[6:][perm]]).astype(np.float64)


def _from_pin_v(v, perm):
    out = np.empty_like(v)
    out[:3], out[3:6] = v[3:6], v[:3]
    out[6:][perm] = v[6:]
    return out


def _rel(a, b):
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1.0)


def test_dimensions(g1):
    robot, _, _ = g1
    assert (robot.nq, robot.nv) == (7 + 29, 6 + 29)
    assert robot.default_cfg.shape == (robot.nq,)


def test_forward_kinematics_matches_pinocchio(g1):
    robot, model, perm = g1
    data = model.createData()
    for cfg, *_ in _states(robot, np.random.default_rng(0), 4):
        T = np.asarray(robot.forward_kinematics(jnp.asarray(cfg)))
        pin.framesForwardKinematics(model, data, _pin_q(cfg, perm))
        for li, name in enumerate(robot.links.names):
            if model.existFrame(name):
                np.testing.assert_allclose(T[li, 4:], data.oMf[model.getFrameId(name)].translation, atol=1e-5)


def test_dynamics_match_pinocchio(g1):
    robot, model, perm = g1
    data = model.createData()
    for cfg, v, a, tau in _states(robot, np.random.default_rng(1), 4):
        tau[:6] = 0.0  # unactuated base
        q_p, v_p = _pin_q(cfg, perm), _pin_v(v, perm)
        assert _rel(np.asarray(dynamics.inverse_dynamics(robot, cfg, v, a)),
                    _from_pin_v(pin.rnea(model, data, q_p, v_p, _pin_v(a, perm)), perm)) < 1e-5
        assert _rel(np.asarray(dynamics.forward_dynamics(robot, cfg, v, tau)),
                    _from_pin_v(pin.aba(model, data, q_p, v_p, _pin_v(tau, perm)), perm)) < 1e-5
        M = pin.crba(model, data, q_p)
        M = np.triu(M) + np.triu(M, 1).T
        order = np.concatenate([[3, 4, 5, 0, 1, 2], 6 + np.argsort(perm)])
        assert _rel(np.asarray(dynamics.mass_matrix(robot, cfg)), M[np.ix_(order, order)]) < 1e-5


def test_integrate_matches_pinocchio_and_difference_inverts(g1):
    robot, model, perm = g1
    for cfg, v, *_ in _states(robot, np.random.default_rng(2), 4):
        q1 = np.asarray(FB.integrate(robot, jnp.asarray(cfg), jnp.asarray(v), 0.1))
        ref = pin.integrate(model, _pin_q(cfg, perm), _pin_v(v, perm) * 0.1)
        got = _pin_q(q1, perm)
        np.testing.assert_allclose(got[:3], ref[:3], atol=1e-5)
        assert min(np.abs(got[3:7] - ref[3:7]).max(), np.abs(got[3:7] + ref[3:7]).max()) < 1e-5
        np.testing.assert_allclose(np.asarray(FB.difference(robot, jnp.asarray(cfg), jnp.asarray(q1))),
                                   v * 0.1, atol=1e-4)


@pytest.mark.parametrize("method", ["semi_implicit", "rk4", "linearly_implicit"])
def test_step_stays_on_the_manifold(g1, method):
    robot, _, _ = g1
    cfg, v, *_ = next(_states(robot, np.random.default_rng(3), 1))
    q, qd = dynamics.step(robot, jnp.asarray(cfg), jnp.asarray(v), jnp.zeros(robot.nv), 0.005, method=method)
    assert q.shape == (robot.nq,) and qd.shape == (robot.nv,)
    assert np.isfinite(np.asarray(q)).all() and abs(float(jnp.linalg.norm(q[:4])) - 1.0) < 1e-5


def test_urdf_floating_joint_is_detected():
    """world --floating--> panda_link0: detected, stripped, its origin the default base."""
    text = open("resources/panda/panda_spherized.urdf").read()
    root = yourdfpy.URDF.load("resources/panda/panda_spherized.urdf", load_meshes=False).base_link
    wrapped = text.replace("</robot>", f'''  <link name="world"/>
  <joint name="float" type="floating"><parent link="world"/><child link="{root}"/>
    <origin xyz="0.1 0.2 0.3" rpy="0 0 0.5"/></joint>
</robot>''')
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".urdf", dir="resources/panda", delete=True) as f:
        f.write(wrapped)
        f.flush()
        floating = Robot.from_urdf(yourdfpy.URDF.load(f.name, load_meshes=False))
    fixed = Robot.from_urdf(yourdfpy.URDF.load("resources/panda/panda_spherized.urdf", load_meshes=False))
    assert floating.floating_base and floating.nq == fixed.nq + 7
    base = jaxlie.SE3(floating.default_cfg[:7])
    np.testing.assert_allclose(np.asarray(base.translation()), [0.1, 0.2, 0.3], atol=1e-6)
    T_float = floating.forward_kinematics(floating.default_cfg)
    T_fixed = (base @ jaxlie.SE3(fixed.forward_kinematics(floating.default_cfg[7:]))).wxyz_xyz
    np.testing.assert_allclose(np.asarray(T_float), np.asarray(T_fixed), atol=1e-5)


def test_ik_converges_in_the_tangent_space(g1):
    from pyroffi.optimization_engines._ls_ik import ls_ik_solve

    robot, _, _ = g1
    names = robot.links.names
    links = tuple(names.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link",
                                           "left_wrist_yaw_link", "right_wrist_yaw_link"))
    rng = np.random.default_rng(4)
    lo, hi = np.asarray(robot.joints.lower_limits), np.asarray(robot.joints.upper_limits)
    base = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3.exp(jnp.array([0.2, -0.3, 0.4])), jnp.array([0.3, -0.2, 0.1]))
    q_true = jnp.concatenate([base.wxyz_xyz, jnp.asarray(rng.uniform(lo * 0.5, hi * 0.5), jnp.float32)])
    T = robot.forward_kinematics(q_true)
    targets = tuple(jaxlie.SE3(T[i]) for i in links)
    prev = FB.integrate(robot, q_true, jnp.asarray(rng.normal(size=robot.nv) * 0.2, jnp.float32))
    q = ls_ik_solve(robot, links, targets, jax.random.PRNGKey(0), prev, num_seeds=16, max_iter=120)
    Tq = robot.forward_kinematics(q)
    for k, i in enumerate(links):
        assert float(jnp.linalg.norm((jaxlie.SE3(Tq[i]).inverse() @ targets[k]).translation())) < 1e-3
    assert abs(float(jnp.linalg.norm(q[:4])) - 1.0) < 1e-5


def test_collision_moves_with_the_base():
    """Self-collision ignores the base; world distance equals the fixed-base distance to the
    world moved by the inverse base pose."""
    urdf = yourdfpy.URDF.load("resources/panda/panda_spherized.urdf", load_meshes=False)
    fixed, floating = Robot.from_urdf(urdf), Robot.from_urdf(urdf, floating_base=True)
    model = RobotCollisionSpherized.from_urdf(urdf)
    base = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3.exp(jnp.array([0.1, 0.4, -0.2])), jnp.array([0.5, -0.3, 0.2]))
    rng = np.random.default_rng(5)
    lo, hi = np.asarray(fixed.joints.lower_limits), np.asarray(fixed.joints.upper_limits)
    q = jnp.asarray(rng.uniform(lo, hi, (32, lo.size)), jnp.float32)
    cfg = jnp.concatenate([jnp.broadcast_to(base.wxyz_xyz, (32, 7)), q], axis=-1)

    np.testing.assert_allclose(np.asarray(model.compute_self_collision_distance(floating, cfg)),
                               np.asarray(model.compute_self_collision_distance(fixed, q)), atol=1e-5)
    centers = jnp.array([[0.8, -0.2, 0.5]])
    world = Sphere.from_center_and_radius(centers, jnp.array([0.1]))
    world_in_base = Sphere.from_center_and_radius(base.inverse().apply(centers), jnp.array([0.1]))
    d_float = jax.vmap(lambda c: model.compute_world_collision_distance(floating, c, world))(cfg)
    d_fixed = jax.vmap(lambda c: model.compute_world_collision_distance(fixed, c, world_in_base))(q)
    finite = np.isfinite(np.asarray(d_fixed))
    np.testing.assert_allclose(np.asarray(d_float)[finite], np.asarray(d_fixed)[finite], atol=1e-4)
