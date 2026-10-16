"""Floating base across the public API.

Every public entry point that takes a robot (or a configuration) runs on a floating-base robot
and agrees with a reference: the JAX model, pinocchio's free-flyer, or finite differences.
`test_every_public_entry_point_is_covered` enumerates the public API (the modules exported by
``pyroffi/__init__.py``) and fails while any robot-taking name lacks a test here.
"""

import inspect
import shutil

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pytest
import yourdfpy

import pyroffi
from pyroffi import Robot
from pyroffi import _floating_base as FB
from pyroffi._robot_srdf_parser import read_disabled_collisions_from_srdf
from pyroffi.collision import RobotCollision, RobotCollisionSpherized, Sphere

GPU = shutil.which("nvcc") is not None and jax.default_backend() == "gpu"
needs_gpu = pytest.mark.skipif(not GPU, reason="CUDA kernels need nvcc and a GPU")
PANDA = "resources/panda/panda_spherized.urdf"
EE = "panda_hand"
COVERED: set[str] = set()
# Deliberately out of scope: link_poses_from_joint_poses takes joint poses, not a configuration.
# pose_with_base_residual: PyRoKi's mobile base (fixed-base robot + separate base variable); a
# floating-base robot carries its base in the configuration instead.
EXEMPT = {"kinematics.link_poses_from_joint_poses", "costs.pose_with_base_residual"}


def covers(*names):
    def deco(fn):
        COVERED.update(names)
        return fn
    return deco


@pytest.fixture(scope="module")
def scene():
    urdf = yourdfpy.URDF.load(PANDA)
    robot = Robot.from_urdf(urdf, floating_base=True)
    ignore = tuple((p["link1"], p["link2"])
                   for p in read_disabled_collisions_from_srdf("resources/panda/panda.srdf"))
    model = RobotCollisionSpherized.from_urdf(urdf, user_ignore_pairs=ignore)
    rng = np.random.default_rng(0)
    lo, hi = np.asarray(robot.joints.lower_limits), np.asarray(robot.joints.upper_limits)
    B = 64
    rots = [jaxlie.SO3.exp(jnp.asarray(rng.normal(size=3) * 0.5, jnp.float32)) for _ in range(B - 2)]
    rots += [jaxlie.SO3.from_y_radians(jnp.pi / 2), jaxlie.SO3.from_rpy_radians(0.3, -jnp.pi / 2, 1.0)]
    bases = jnp.stack([jaxlie.SE3.from_rotation_and_translation(
        r, jnp.asarray(rng.normal(size=3) * 0.1, jnp.float32)).wxyz_xyz for r in rots])
    cfg = jnp.concatenate([bases, jnp.asarray(rng.uniform(lo, hi, (B, lo.size)), jnp.float32)], -1)
    world = Sphere.from_center_and_radius(jnp.array([[0.4, 0.1, 0.4], [0.1, -0.3, 0.6]]),
                                          jnp.array([0.15, 0.12]))
    # Tangent directions (not ambient: a quaternion's radial direction is not a motion).
    V = jnp.asarray(rng.normal(size=(B, robot.nv)), jnp.float32)
    dcfg = jax.vmap(lambda c, v: jax.jvp(lambda e: FB.integrate(robot, c, v * e), (0.0,), (1.0,))[1])(cfg, V)
    return dict(urdf=urdf, robot=robot, model=model, cfg=cfg, dcfg=dcfg, world=world, lo=lo, hi=hi)


def _ik_problem(scene, seed=1):
    robot, rng = scene["robot"], np.random.default_rng(seed)
    base = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.exp(jnp.array([0.1, 0.2, -0.3])), jnp.array([0.5, 0.1, 0.2]))
    q_true = jnp.concatenate([base.wxyz_xyz, jnp.asarray(
        rng.uniform(scene["lo"] * 0.6, scene["hi"] * 0.6), jnp.float32)])
    target = jaxlie.SE3(robot.forward_kinematics(q_true)[robot.links.names.index(EE)])
    prev = q_true.at[-7:].add(jnp.asarray(rng.normal(size=7) * 0.1, jnp.float32))
    return target, prev


def _ee_error(robot, q, target):
    e = (jaxlie.SE3(robot.forward_kinematics(q)[robot.links.names.index(EE)]).inverse() @ target).log()
    return float(jnp.linalg.norm(e[:3])), float(jnp.linalg.norm(e[3:]))


def _agree(f_test, f_ref, cfg, dcfg, atol=1e-4, rtol_jvp=1e-3):
    """Values agree where the reference is live, and so do tangent-direction derivatives."""
    a, b = np.asarray(f_test(cfg)), np.asarray(f_ref(cfg))
    live = np.isfinite(b) & (np.abs(b) < 1e5)
    np.testing.assert_allclose(a[live], b[live], atol=atol)
    ja = jax.jvp(lambda c: jnp.sum(jnp.where(live, f_test(c), 0.0)), (cfg,), (dcfg,))[1]
    jb = jax.jvp(lambda c: jnp.sum(jnp.where(live, f_ref(c), 0.0)), (cfg,), (dcfg,))[1]
    assert abs(float(ja - jb)) <= rtol_jvp * (abs(float(jb)) + 1e-3)


# ---------------------------------------------------------------------------
# The chart the joint-space solvers and kernels run on
# ---------------------------------------------------------------------------

def test_chart_is_exact_even_at_gimbal_lock(scene):
    robot, cfg = scene["robot"], scene["cfg"]
    for ref in (FB.neutral_base(), cfg[0, :7]):
        chart = FB.chart_robot(robot, ref)
        q = FB.to_chart(robot, cfg, ref)
        np.testing.assert_allclose(
            np.asarray(jaxlie.SE3(chart.forward_kinematics(q)).as_matrix()),
            np.asarray(jaxlie.SE3(robot.forward_kinematics(cfg)).as_matrix()), atol=1e-5)
        back = FB.from_chart(robot, q, ref)
        assert float(jnp.abs((jaxlie.SE3(back[:, :7]).inverse() @ jaxlie.SE3(cfg[:, :7])).log()).max()) < 1e-5


def test_evaluation_chart_gradients_are_base_twists(scene):
    """to_eval_chart: exact values, and tangent derivatives equal to the floating FK's."""
    robot, cfg, dcfg = scene["robot"], scene["cfg"], scene["dcfg"]
    ev = FB.eval_chart(robot)
    rot = lambda T: jaxlie.SE3(T).as_matrix()
    _agree(lambda c: rot(ev.forward_kinematics(FB.to_eval_chart(robot, c))),
           lambda c: rot(robot.forward_kinematics(c)), cfg, dcfg, atol=1e-5)


def test_unbounded_limits_sample_the_warm_start():
    lo = jnp.array([-1.0, -jnp.inf]); hi = jnp.array([1.0, jnp.inf])
    s = FB.sample_in_limits(jax.random.PRNGKey(0), (100, 2), lo, hi, jnp.array([0.0, 7.0]))
    assert bool(jnp.all(jnp.abs(s[:, 0]) <= 1.0)) and bool(jnp.all(s[:, 1] == 7.0))
    np.testing.assert_allclose(np.asarray(FB.finite_range(lo, hi)), [2.0, 1.0])


def _screw_curve(robot, cfg, v, k, t0=0.3):
    """``T0 exp(s(t) v)`` with ``s = t0 + t + k t^2 / 2``: body twist ``v`` and rate ``k v`` at
    ``t = 0``, and ambient derivatives taken away from ``exp``'s small-angle branch (whose
    second derivative jaxlie truncates)."""
    curve = lambda t: FB.integrate(robot, cfg, v * (t0 + t + 0.5 * k * t * t))
    qd = jax.jvp(curve, (0.0,), (1.0,))[1]
    qdd = jax.jvp(lambda t: jax.jvp(curve, (t,), (1.0,))[1], (0.0,), (1.0,))[1]
    return curve(0.0), qd, qdd


def test_curve_tangents_are_the_body_twist_and_its_rate(scene):
    robot, cfg = scene["robot"], scene["cfg"][0]
    v = jnp.asarray(np.random.default_rng(3).normal(size=robot.nv), jnp.float32)
    q, qd, qdd = _screw_curve(robot, cfg, v, 0.7)
    vel, acc = FB.curve_tangents(q, qd, qdd)
    np.testing.assert_allclose(np.asarray(vel), np.asarray(v), atol=1e-4)
    np.testing.assert_allclose(np.asarray(acc), np.asarray(0.7 * v), atol=1e-3)


# ---------------------------------------------------------------------------
# Kinematics and IK
# ---------------------------------------------------------------------------

@covers("kinematics.forward_kinematics", "kinematics.forward_kinematics_joints_jax")
def test_forward_kinematics(scene):
    robot, cfg = scene["robot"], scene["cfg"][:8]
    T = pyroffi.kinematics.forward_kinematics(robot, cfg)
    Tj = pyroffi.kinematics.forward_kinematics_joints_jax(robot, cfg)
    li = robot.links.names.index(EE)
    j = int(robot.links.parent_joint_indices[li])
    np.testing.assert_allclose(np.asarray(jaxlie.SE3(Tj[:, j]).as_matrix()),
                               np.asarray(jaxlie.SE3(T[:, li]).as_matrix()), atol=1e-5)
    if GPU:
        np.testing.assert_allclose(np.asarray(jaxlie.SE3(robot.forward_kinematics(cfg, use_cuda=True)).as_matrix()),
                                   np.asarray(jaxlie.SE3(T).as_matrix()), atol=1e-4)


@covers("kinematics.inverse_kinematics")
@pytest.mark.parametrize("solver,cuda", [
    ("ls", False), ("hjcd", False), ("analytic", False),
    pytest.param("ls", True, marks=needs_gpu), pytest.param("hjcd", True, marks=needs_gpu)])
def test_inverse_kinematics(scene, solver, cuda):
    robot = scene["robot"]
    target, prev = _ik_problem(scene)
    q = pyroffi.kinematics.inverse_kinematics(robot, EE, target, previous_cfg=prev, solver=solver,
                                              use_cuda=cuda)
    pos, rot = _ee_error(robot, q, target)
    assert pos < 1e-2 and rot < 5e-2, (pos, rot)
    assert abs(float(jnp.linalg.norm(q[:4])) - 1.0) < 1e-5


def test_inverse_kinematics_base_region(scene):
    robot = scene["robot"]
    target, prev = _ik_problem(scene)
    lo, hi = prev[4:7] - 0.05, prev[4:7] + 0.05
    q = robot.inverse_kinematics(EE, target, previous_cfg=prev, solver="hjcd", base_region=(lo, hi))
    assert bool(jnp.all((q[4:7] >= lo - 1e-6) & (q[4:7] <= hi + 1e-6)))
    assert _ee_error(robot, q, target)[0] < 1e-2


@covers("kinematics.extract_dh")
def test_chain_models_live_in_the_base_frame(scene):
    robot = scene["robot"]
    dh = pyroffi.kinematics.extract_dh(robot, EE, validate=True)  # checked against base-frame FK
    assert dh.dof == 7


# ---------------------------------------------------------------------------
# Dynamics (JAX and GRiD)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def g1():
    pin = pytest.importorskip("pinocchio")
    path = "resources/g1_description/g1_29dof__dynbench.urdf"
    robot = Robot.from_urdf(yourdfpy.URDF.load(path, load_meshes=False), floating_base=True)
    model = pin.buildModelFromUrdf(path, pin.JointModelFreeFlyer())
    act = list(robot.joints.actuated_names)
    perm = np.array([act.index(model.names[j]) for j in range(2, model.njoints)])
    return robot, model, perm, pin


def _g1_states(robot, k=8):
    rng = np.random.default_rng(1)
    base = [jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3.exp(jnp.asarray(rng.normal(size=3), jnp.float32)),
                                                     jnp.asarray(rng.normal(size=3), jnp.float32)).wxyz_xyz
            for _ in range(k)]
    q = np.concatenate([np.stack(base), rng.uniform(-0.5, 0.5, (k, robot.joints.num_actuated_joints))], -1)
    v, a, tau = (rng.normal(size=(k, robot.nv)).astype(np.float32) for _ in range(3))
    tau[:, :6] = 0.0
    return q.astype(np.float32), v, a, tau


def _pin(model, pin, perm, q, v, a, tau):
    data = model.createData()
    pq = lambda c: np.concatenate([c[4:7], c[1:4], c[:1], c[7:][perm]]).astype(np.float64)
    pv = lambda x: np.concatenate([x[3:6], x[:3], x[6:][perm]]).astype(np.float64)
    back = lambda x: np.concatenate([x[3:6], x[:3], x[6:][np.argsort(perm)]])
    order = np.concatenate([[3, 4, 5, 0, 1, 2], 6 + np.argsort(perm)])
    ids, fds, Ms = [], [], []
    for k in range(len(q)):
        ids.append(back(pin.rnea(model, data, pq(q[k]), pv(v[k]), pv(a[k]))))
        fds.append(back(pin.aba(model, data, pq(q[k]), pv(v[k]), pv(tau[k]))))
        M = pin.crba(model, data, pq(q[k]))
        Ms.append((np.triu(M) + np.triu(M, 1).T)[np.ix_(order, order)])
    return np.stack(ids), np.stack(fds), np.stack(Ms)


def _rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - b) / max(np.linalg.norm(b), 1.0))


@covers("bindings.GRiDDynamics")
@needs_gpu
def test_grid_matches_pinocchio_and_jax(g1):
    robot, model, perm, pin = g1
    gd = pyroffi.bindings.GRiDDynamics.from_robot(robot)
    q, v, a, tau = _g1_states(robot)
    ids, fds, Ms = _pin(model, pin, perm, q, v, a, tau)
    assert _rel(gd.inverse_dynamics(q, v, a), ids) < 1e-5
    assert _rel(gd.forward_dynamics(q, v, tau), fds) < 1e-5
    assert _rel(gd.mass_matrix(q), Ms) < 1e-5
    Minv_ref = np.linalg.inv(Ms)  # compare to the inverse: M^-1 M - I amplifies float32 by cond(M)
    assert np.linalg.norm(np.asarray(gd.mass_matrix_inv(q)) - Minv_ref) / np.linalg.norm(Minv_ref) < 1e-3
    # First-order gradients along a tangent direction, against the JAX dynamics.
    c0 = jnp.asarray(q[0])
    dq = jax.jvp(lambda e: FB.integrate(robot, c0, jnp.asarray(v[1]) * e), (0.0,), (1.0,))[1]
    for fg, fj in [(lambda c: gd.inverse_dynamics(c, v[0], a[0]), lambda c: robot.inverse_dynamics(c, v[0], a[0])),
                   (lambda c: gd.forward_dynamics(c, v[0], tau[0]), lambda c: robot.forward_dynamics(c, v[0], tau[0]))]:
        assert _rel(jax.jvp(fg, (c0,), (dq,))[1], np.asarray(jax.jvp(fj, (c0,), (dq,))[1])) < 1e-4
    q1, _ = gd.step(c0, jnp.asarray(v[0]), jnp.asarray(tau[0]), 0.005)
    assert abs(float(jnp.linalg.norm(q1[:4])) - 1.0) < 1e-5


@needs_gpu
def test_robot_dynamics_use_cuda(g1):
    robot, model, perm, pin = g1
    q, v, a, tau = _g1_states(robot, 4)
    ids, fds, Ms = _pin(model, pin, perm, q, v, a, tau)
    assert _rel(robot.inverse_dynamics(q, v, a, use_cuda=True), ids) < 1e-5
    assert _rel(robot.forward_dynamics(q, v, tau, use_cuda=True), fds) < 1e-5
    assert _rel(robot.mass_matrix(q, use_cuda=True), Ms) < 1e-5
    qn, _ = robot.step(jnp.asarray(q), jnp.asarray(v), jnp.asarray(tau), 0.005, use_cuda=True)
    assert np.allclose(np.linalg.norm(np.asarray(qn)[:, :4], axis=-1), 1.0, atol=1e-5)


def test_jacobian_matches_finite_differences(g1):
    robot = g1[0]
    q = jnp.asarray(_g1_states(robot, 1)[0][0])
    J, r = robot.jacobian(q)
    v = jnp.asarray(np.random.default_rng(5).normal(size=robot.nv), jnp.float32)
    urdf = robot.urdf
    links = jnp.array([robot.links.names.index(urdf.joint_map[n].child) for n in robot.dynamics.dof_names])
    eps = 1e-3
    T0 = jaxlie.SE3(robot.forward_kinematics(q)[links])
    Tp = jaxlie.SE3(robot.forward_kinematics(FB.integrate(robot, q, v, eps))[links])
    Tm = jaxlie.SE3(robot.forward_kinematics(FB.integrate(robot, q, v, -eps))[links])
    om = ((Tp.rotation() @ Tm.rotation().inverse()).log()) / (2 * eps)
    vl = (Tp.translation() - Tm.translation()) / (2 * eps)
    np.testing.assert_allclose(np.asarray(r), np.asarray(T0.translation()), atol=5e-4)  # float32
    np.testing.assert_allclose(np.asarray(jnp.einsum("bij,j->bi", J, v)),
                               np.asarray(jnp.concatenate([om, vl], -1)), atol=5e-3)


@covers("topp.jax_inverse_dynamics_fn")
def test_topp_inverse_dynamics_returns_joint_torques(g1):
    robot = g1[0]
    q, v, a, _ = _g1_states(robot, 3)
    fn = pyroffi.topp.jax_inverse_dynamics_fn(robot)
    curves = [_screw_curve(robot, jnp.asarray(q[i]), jnp.asarray(v[i]), 0.5) for i in range(3)]
    c, qd, qdd = (jnp.stack(x) for x in zip(*curves))
    tau = fn(c, qd, qdd)
    assert tau.shape == (3, robot.joints.num_actuated_joints)
    expect = robot.inverse_dynamics(c, jnp.asarray(v), 0.5 * jnp.asarray(v))[:, 6:]
    np.testing.assert_allclose(np.asarray(tau), np.asarray(expect), rtol=1e-3, atol=1e-2)


# ---------------------------------------------------------------------------
# Collision
# ---------------------------------------------------------------------------

def _jax_world(scene):
    robot, model, world = scene["robot"], scene["model"], scene["world"]
    return lambda c: jax.vmap(lambda q: model.compute_world_collision_distance(robot, q, world))(c)


def _jax_self(scene):
    robot, model = scene["robot"], scene["model"]
    return lambda c: jax.vmap(lambda q: model.compute_self_collision_distance(robot, q))(c)


@covers("collision.RobotCollisionSpherized", "collision.RobotCollision")
def test_jax_collision_models_move_with_the_base(scene):
    """The world seen from a floating base equals the fixed robot against the moved world."""
    urdf, robot, model, cfg = scene["urdf"], scene["robot"], scene["model"], scene["cfg"][:8]
    fixed = Robot.from_urdf(urdf)
    for m in (model, RobotCollision.from_urdf(urdf)):
        for c in cfg:
            T = jaxlie.SE3(c[:7])
            w = Sphere.from_center_and_radius(jnp.array([[0.4, 0.1, 0.4]]), jnp.array([0.15]))
            w_base = Sphere.from_center_and_radius(T.inverse().apply(jnp.array([[0.4, 0.1, 0.4]])), jnp.array([0.15]))
            a = np.asarray(m.compute_world_collision_distance(robot, c, w))
            b = np.asarray(m.compute_world_collision_distance(fixed, c[7:], w_base))
            live = np.isfinite(b)
            np.testing.assert_allclose(a[live], b[live], atol=1e-4)
        np.testing.assert_allclose(np.asarray(m.compute_self_collision_distance(robot, cfg[0])),
                                   np.asarray(m.compute_self_collision_distance(fixed, cfg[0, 7:])), atol=1e-5)


@covers("collision.FusedCUDACollisionChecker", "collision.make_fused_checker")
@needs_gpu
def test_fused_checker(scene):
    from pyroffi.collision import make_fused_checker
    from pyroffi.collision._esdf import ESDFWorldGeom, esdf_query_jax

    robot, model, cfg, dcfg = scene["robot"], scene["model"], scene["cfg"], scene["dcfg"]
    ch = make_fused_checker(robot, model)
    _agree(lambda c: ch.compute_self_collision_distance(robot, c), _jax_self(scene), cfg, dcfg)
    _agree(lambda c: ch.compute_world_collision_distance(robot, c, scene["world"]), _jax_world(scene), cfg, dcfg)
    _, z = ch.compute_self_collision_and_floor(robot, cfg)
    zref = jax.vmap(lambda q: jnp.min(jnp.where(model.at_config(robot, q).radius > 0,
                    model.at_config(robot, q).pose.translation()[..., 2] - model.at_config(robot, q).radius, jnp.inf)))(cfg)
    np.testing.assert_allclose(np.asarray(z), np.asarray(zref), atol=1e-4)
    ii, jj, kk = np.meshgrid(*(np.arange(n) for n in (60, 60, 60)), indexing="ij")
    origin = np.array([-0.8, -0.8, -0.4], np.float32)
    pts = origin + np.stack([ii, jj, kk], -1) * 0.025
    grid = (np.linalg.norm(pts - np.array([0.4, 0.1, 0.4]), axis=-1) - 0.15).astype(np.float32)
    esdf = ESDFWorldGeom(grid=jnp.asarray(grid), origin=jnp.asarray(origin), voxel_size=0.025)

    def ref(c):
        def one(q):
            g = model.at_config(robot, q)
            d = esdf_query_jax(g.pose.translation(), esdf.grid, esdf.origin, esdf.voxel_size) - g.radius
            return jnp.min(jnp.where(g.radius > 0.0, d, jnp.inf), axis=0)
        return jax.vmap(one)(c)
    _agree(lambda c: ch.compute_world_collision_distance(robot, c, esdf)[..., 0], ref, cfg, dcfg, rtol_jvp=5e-3)


@covers("collision.CUDADifferentiableSDFCollisionChecker", "collision.CUDARobotCollisionChecker",
        "collision.make_cuda_checker")
@needs_gpu
def test_sdf_checker(scene):
    from pyroffi.collision import make_cuda_checker

    robot, cfg, dcfg = scene["robot"], scene["cfg"], scene["dcfg"]
    ch = make_cuda_checker(scene["model"])
    _agree(lambda c: ch.compute_world_collision_distance(robot, c, scene["world"]), _jax_world(scene), cfg, dcfg)


def _verdicts(scene):
    dw, ds = _jax_world(scene)(scene["cfg"]), _jax_self(scene)(scene["cfg"])
    d = np.minimum(np.asarray(jnp.min(dw, axis=(1, 2))), np.asarray(jnp.min(jnp.where(jnp.isfinite(ds), ds, 1e9), axis=1)))
    return d > 0, np.abs(d) > 2e-3


@covers("collision.CUDABinaryCollisionChecker", "collision.make_cuda_binary_checker")
@needs_gpu
def test_binary_checker(scene):
    from pyroffi.collision import make_cuda_binary_checker

    free, clear = _verdicts(scene)
    got = np.asarray(make_cuda_binary_checker(scene["model"]).check_collision_free(
        scene["robot"], scene["cfg"], scene["world"]))
    assert (got == free)[clear].all()
    edges = jnp.stack([scene["cfg"], scene["cfg"]], axis=1)
    got_e = np.asarray(make_cuda_binary_checker(scene["model"]).check_edges_collision_free(
        scene["robot"], edges, scene["world"]))
    assert (got_e == free)[clear].all()


@covers("collision.RoboGPUCollisionChecker", "collision.make_robogpu_checker")
@needs_gpu
def test_robogpu_checker(scene):
    from pyroffi.collision import make_robogpu_checker

    try:
        ch = make_robogpu_checker(scene["model"])
    except (RuntimeError, OSError) as e:
        pytest.skip(f"roboGPU unavailable: {e}")
    free, clear = _verdicts(scene)
    got = np.asarray(ch.check_collision_free(scene["robot"], scene["cfg"], scene["world"])) != 0
    assert (got == free)[clear].all()


@covers("collision.NeuralRobotCollision", "collision.NeuralRobotCollisionSpherized")
def test_neural_collision_trains_and_evaluates(scene):
    from pyroffi.collision import NeuralRobotCollision

    robot, world = scene["robot"], scene["world"]
    net = NeuralRobotCollision.from_existing(scene["model"]).train(
        robot, world, num_samples=256, batch_size=128, epochs=1, layer_sizes=[32])
    d = net.compute_world_collision_distance(robot, scene["cfg"][:4], world)
    assert d.shape == (4, scene["model"].num_links, 2) and bool(jnp.all(jnp.isfinite(d)))
    np.testing.assert_allclose(np.asarray(net.compute_self_collision_distance(robot, scene["cfg"][0])),
                               np.asarray(scene["model"].compute_self_collision_distance(robot, scene["cfg"][0])), atol=1e-5)


def test_robot_collision_check(scene):
    robot, cfg, world = scene["robot"], scene["cfg"][:8], scene["world"]
    # The JAX path (capsules) takes one configuration: equal to the fixed robot against the world
    # moved into the base frame.
    T = jaxlie.SE3(cfg[0, :7])
    w_base = Sphere.from_center_and_radius(T.inverse().apply(world.pose.translation()), world.radius)
    fixed = Robot.from_urdf(scene["urdf"])
    a = np.asarray(robot.collision_check(cfg[0], world))
    b = np.asarray(fixed.collision_check(cfg[0, 7:], w_base))
    live = np.isfinite(b) & (np.abs(b) < 1e5)
    np.testing.assert_allclose(a[live], b[live], atol=1e-4)
    if GPU:
        # (use_cuda=True with the capsule model trips an XLA compiler error on fixed bases too.)
        assert robot.collision_check(cfg, world, method="binary").shape == (8,)


# ---------------------------------------------------------------------------
# Motion generators
# ---------------------------------------------------------------------------

@covers("motion_generators.TrajoptMotionGenerator", "motion_generators.StompMotionGenerator")
@needs_gpu  # both seed with the CUDA MPPI IK kernel
@pytest.mark.parametrize("kind", ["trajopt", "stomp"])
def test_motion_generators(scene, kind):
    from pyroffi.motion_generators import StompMotionGenerator, TrajoptMotionGenerator
    from pyroffi.optimization_engines._sco_optimization import ScoTrajOptConfig
    from pyroffi.optimization_engines._stomp_optimization import StompTrajOptConfig

    robot, model = scene["robot"], scene["model"]
    q0 = scene["cfg"][0].at[7:].set(jnp.asarray((scene["lo"] + scene["hi"]) / 2))
    ee = robot.links.names.index(EE)
    T0 = jaxlie.SE3(robot.forward_kinematics(q0)[ee])
    goal = jaxlie.SE3.from_translation(jnp.array([0.0, 0.2, -0.1])) @ T0
    Gen, cfg = ((TrajoptMotionGenerator, ScoTrajOptConfig(n_outer_iters=3, n_inner_iters=10)) if kind == "trajopt"
                else (StompMotionGenerator, StompTrajOptConfig()))
    far = (Sphere.from_center_and_radius(jnp.array([[9.0, 9.0, 9.0]]), jnp.array([0.1])),)
    for cuda in (False, True):
        g = Gen(robot=robot, robot_coll=model, world_geoms=far, ee_link_name=EE, n_timesteps=16,
                n_batch=4, trajopt_cfg=cfg, use_cuda=cuda)
        best, *_ = g.generate(T0, goal, jax.random.PRNGKey(0), prev_cfgs=jnp.broadcast_to(q0, (16, robot.nq)))
        assert best.shape == (16, robot.nq)
        np.testing.assert_allclose(np.linalg.norm(np.asarray(best[:, :4]), axis=-1), 1.0, atol=1e-5)
        pos = jaxlie.SE3(robot.forward_kinematics(best[-1])[ee]).translation()
        assert float(jnp.linalg.norm(pos - goal.translation())) < 1e-2


# ---------------------------------------------------------------------------
# Attachments and viewer
# ---------------------------------------------------------------------------

@covers("attachments.pose_attachments", "attachments.tool_frame", "attachments.Attachment",
        "attachments.compose_dynamics", "attachments.link_dof_bodies",
        "attachments.attachment_wrench_to_body")
def test_attachments(g1):
    from pyroffi.attachments import (Attachment, AttachmentSet, attachment_wrench_to_body,
                                     compose_dynamics, link_dof_bodies, pose_attachments, tool_frame)
    from pyroffi.collision import Sphere as S

    robot = g1[0]
    q = jnp.asarray(_g1_states(robot, 1)[0][0])
    T_LB = jaxlie.SE3.from_translation(jnp.array([0.1, 0.0, 0.05])).wxyz_xyz
    hand = robot.links.names.index("left_wrist_yaw_link")
    pelvis = robot.links.names.index("pelvis")
    geom = S.from_center_and_radius(jnp.zeros((1, 3)), jnp.array([0.03]))
    tool = Attachment.from_geom(geom, hand, T_LB, mass=jnp.array(0.5), name="tool")
    pack = Attachment.from_mass(jnp.array(2.0), pelvis, T_LB, name="pack")
    aset = AttachmentSet.empty().attach(tool).attach(pack)
    T_expect = jaxlie.SE3(robot.forward_kinematics(q)[hand]) @ jaxlie.SE3(T_LB)
    np.testing.assert_allclose(np.asarray(tool.T_world_body(robot, q).as_matrix()), np.asarray(T_expect.as_matrix()), atol=1e-5)
    np.testing.assert_allclose(np.asarray(tool_frame(robot, q, aset, "tool").as_matrix()), np.asarray(T_expect.as_matrix()), atol=1e-5)
    np.testing.assert_allclose(np.asarray(pose_attachments(robot, q, aset).pose.translation()[0]),
                               np.asarray(T_expect.translation()), atol=1e-5)
    regrasp = tool.grasp_from_current_pose(robot, q, T_expect.wxyz_xyz)
    np.testing.assert_allclose(np.asarray(regrasp.T_parent_body), np.asarray(T_LB), atol=1e-5)
    # A payload on the pelvis loads the free base body: it changes the base-row dynamics.
    assert link_dof_bodies(robot)["pelvis"][0] == -1
    loaded = compose_dynamics(robot, AttachmentSet.empty().attach(pack))
    assert float(jnp.abs(loaded.dynamics.I_base - robot.dynamics.I_base).max()) > 1.0
    v0 = jnp.zeros(robot.nv)
    assert float(jnp.abs(loaded.inverse_dynamics(q, v0, v0)[:6] - robot.inverse_dynamics(q, v0, v0)[:6]).max()) > 1.0
    idx, w = attachment_wrench_to_body(robot, aset, "pack", jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    assert idx == -1 and float(jnp.abs(w).max()) > 0.0


@covers("viewer.ee_path_positions", "viewer.ManipulabilityEllipse")
def test_viewer_helpers(scene):
    from pyroffi.viewer import ManipulabilityEllipse, ee_path_positions

    robot, cfg = scene["robot"], scene["cfg"][:3]
    pts = ee_path_positions(robot, np.asarray(cfg), EE)
    np.testing.assert_allclose(pts, np.asarray(robot.forward_kinematics(cfg)[:, robot.links.names.index(EE), 4:]), atol=1e-5)

    class _Scene:  # the smallest stand-in for a viser server the ellipse touches
        def add_mesh_simple(self, *a, **k):
            return type("Mesh", (), {"visible": True, "vertices": None, "remove": lambda s: None})()

    ell = ManipulabilityEllipse(type("Server", (), {"scene": _Scene()})(), robot, target_link_name=EE)
    ell.update(cfg[0])
    assert ell.manipulability > 0.0


@covers("costs.limit_residual", "costs.limit_velocity_residual", "costs.five_point_velocity_residual",
        "costs.manipulability_residual", "costs.pose_residual", "costs.self_collision_residual",
        "costs.world_collision_residual")
def test_residuals_accept_floating_configurations(scene):
    """PyRoKi's residuals (re-exported in pk.costs) on a floating base: limits apply to the joints."""
    robot, model, world, cfg = scene["robot"], scene["model"], scene["world"], scene["cfg"]
    a, b = pyroffi.Var(0), pyroffi.Var(1)
    vals = {a: cfg[0], b: cfg[1]}
    link = robot.links.num_links - 1
    T = jaxlie.SE3(robot.forward_kinematics(cfg[0])[link])
    c = pyroffi.costs
    n = robot.joints.num_actuated_joints
    out = {
        "limit": c.limit_residual(vals, robot, a),
        "limit_vel": c.limit_velocity_residual(vals, robot, a, b, 0.1),
        "five_vel": c.five_point_velocity_residual(vals, robot, a, a, b, b, 0.1),
        "manip": c.manipulability_residual(vals, robot, a, jnp.array([link])),
        "pose": c.pose_residual(vals, robot, a, T, link, 1.0, 1.0),
        "self": c.self_collision_residual(vals, robot, model, a, 0.01),
        "world": c.world_collision_residual(vals, robot, model, a, world, 0.01),
    }
    assert all(bool(jnp.all(jnp.isfinite(r))) for r in out.values())
    assert out["limit_vel"].shape == out["five_vel"].shape == (n,)
    assert float(jnp.abs(out["limit"]).max()) == 0.0          # cfg sampled inside the limits
    assert float(jnp.abs(out["pose"]).max()) < 1e-4


@covers("costs.world_collision_native")
def test_native_world_collision_rejects_floating_base(scene):
    robot, model, world = scene["robot"], scene["model"], scene["world"]
    with pytest.raises(NotImplementedError, match="fixed-base"):
        pyroffi.costs.world_collision_native(robot, model, None, world, 0.01)


# ---------------------------------------------------------------------------
# Robot methods (the public class)
# ---------------------------------------------------------------------------

@covers("Robot")
def test_robot_methods_accept_floating_configurations(g1):
    """Robot's configuration-taking methods: FK, IK, dynamics, jacobian, step, collision."""
    robot = g1[0]
    q, v, a, tau = _g1_states(robot, 2)
    assert robot.forward_kinematics(q).shape == (2, robot.links.num_links, 7)
    assert robot.inverse_dynamics(q, v, a).shape == (2, robot.nv)
    assert robot.forward_dynamics(q, v, tau).shape == (2, robot.nv)
    assert robot.mass_matrix(q).shape == (2, robot.nv, robot.nv)
    assert robot.jacobian(q)[0].shape == (2, robot.dynamics.num_dof, 6, robot.nv)
    qn, vn = robot.step(q, v, tau, 0.002)
    assert qn.shape == q.shape and vn.shape == v.shape


# ---------------------------------------------------------------------------
# Coverage: fails while any public robot-taking entry point lacks a test above
# ---------------------------------------------------------------------------

def _public_robot_entry_points():
    def params(obj):
        try:
            return list(inspect.signature(obj.__init__ if inspect.isclass(obj) else obj).parameters)
        except (TypeError, ValueError):
            return []

    names = {"Robot"}
    for mod_name in ("attachments", "bindings", "collision", "costs", "kinematics",
                     "motion_generators", "topp", "viewer"):
        mod = getattr(pyroffi, mod_name)
        for n in getattr(mod, "__all__", None) or [n for n in dir(mod) if not n.startswith("_")]:
            try:
                obj = getattr(mod, n)
            except Exception:
                continue
            if inspect.ismodule(obj):
                continue
            takes = {"robot", "cfg"} & set(params(obj))
            if inspect.isclass(obj):
                takes |= {m for m, f in inspect.getmembers(obj, inspect.isfunction)
                          if not m.startswith("_") and "robot" in params(f)}
            if takes:
                names.add(f"{mod_name}.{n}")
    return names


def test_every_public_entry_point_is_covered():
    missing = _public_robot_entry_points() - COVERED - EXEMPT
    assert not missing, f"public entry points with no floating-base test: {sorted(missing)}"
