"""The motion-generator default: collision-free, torque-feasible trajopt.

Panda at dt=0.05 (a 1.55 s move of joint 1 from -1.2 to 1.2 rad around a sphere):
kinematics-only SCO lands at ~0.55x the URDF torque limit, the dynamics-aware
engine at ~0.42x; both collision-free. The GRiD variant needs a CUDA device.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import numpy as np
import pytest
import yourdfpy

import pyroffi as pk
from pyroffi.collision import RobotCollision, RobotCollisionSpherized, Sphere
from pyroffi.motion_generators import TrajoptMotionGenerator
from pyroffi.motion_generators import _trajopt as trajopt_module
from pyroffi.optimization_engines import (
    ChompTrajOptConfig,
    DynamicsAwareTrajOptConfig,
    ScoTrajOptConfig,
    StompTrajOptConfig,
    dynamics_aware_trajopt,
    effort_limits,
)
from pyroffi.optimization_engines._contact_trajopt import _fd_vel_acc
from pyroffi.optimization_engines._sco_optimization import _collision_distances_all

RESOURCES = Path(__file__).resolve().parents[1] / "resources"
PANDA_URDF = "resources/panda/panda_spherized.urdf"
PANDA_SRDF = "resources/panda/panda.srdf"  # without it adjacent spheres always collide


@pytest.fixture(scope="module")
def scene():
    urdf = yourdfpy.URDF.load(PANDA_URDF, load_meshes=False)
    robot = pk.Robot.from_urdf(urdf)
    rc = RobotCollisionSpherized.from_urdf(urdf, srdf_path=PANDA_SRDF)
    world = (Sphere.from_center_and_radius(jnp.array([[0.45, 0.0, 0.45]]), jnp.array([0.12])),)
    start = jnp.array([-1.2, 0.3, 0.0, -1.8, 0.0, 2.1, 0.8])
    goal = start.at[0].set(1.2)
    T, B = 32, 8
    a = jnp.linspace(0.0, 1.0, T)[:, None]
    init = (start * (1 - a) + goal * a)[None] + 0.05 * jax.random.normal(
        jax.random.PRNGKey(0), (B, T, 7))
    return robot, rc, world, start, goal, init


def _torque_ratio(robot, traj, dt):
    qd, qdd = _fd_vel_acc(traj, dt)
    return float(jnp.max(jnp.abs(robot.inverse_dynamics(traj, qd, qdd)) / effort_limits(robot)))


def _clearance(robot, rc, world, traj):
    return float(jax.vmap(lambda q: jnp.min(_collision_distances_all(q, robot, rc, world)))(traj).min())


@pytest.mark.parametrize(
    "use_cuda",
    [False, pytest.param(True, marks=pytest.mark.skipif(
        jax.default_backend() != "gpu", reason="GRiD needs a CUDA device"))],
)
def test_default_is_collision_free_and_torque_feasible(scene, use_cuda):
    robot, rc, world, start, goal, init = scene
    dt = 0.05
    seed_ratio = max(_torque_ratio(robot, t, dt) for t in init)
    assert seed_ratio > 1.0  # the noisy seeds violate the limits

    best, costs, trajs = dynamics_aware_trajopt(
        init, start, goal, robot, rc, world, DynamicsAwareTrajOptConfig(dt=dt),
        use_cuda=use_cuda,
    )
    assert trajs.shape == init.shape and costs.shape == (init.shape[0],)
    assert jnp.allclose(best[0], start) and jnp.allclose(best[-1], goal)
    assert _torque_ratio(robot, best, dt) <= 1.0
    assert _clearance(robot, rc, world, best) >= 0.0
    lo, hi = robot.joints.lower_limits, robot.joints.upper_limits
    assert bool(jnp.all((best >= lo - 1e-3) & (best <= hi + 1e-3)))


def test_rejects_robot_without_dynamics(scene):
    robot, rc, world, *_ = scene
    robot = jdc.replace(robot, dynamics=None)
    q = jnp.zeros((1, 4, robot.joints.num_actuated_joints))
    with pytest.raises(ValueError, match="no dynamics model"):
        dynamics_aware_trajopt(q, q[0, 0], q[0, 0], robot, rc, world)


@pytest.fixture(scope="module")
def panda_description():
    """The robot_descriptions Panda PyRoki users load: its second finger is a mimic joint."""
    from robot_descriptions.loaders.yourdfpy import load_robot_description

    urdf = load_robot_description("panda_description")
    return urdf, pk.Robot.from_urdf(urdf)


def test_mimic_joint_gravity_matches_exact(panda_description):
    """Mimic joints are held rigid in dynamics. Against gravity from dU/dq over FK
    (which moves the mimic finger exactly), the error is the rigid 15 g finger:
    ~2e-3 N*m on arm joints and the coupled finger's weight (~0.15 N) on the finger."""
    urdf, robot = panda_description
    assert robot.dynamics is not None
    assert robot.dynamics.dof_names == robot.joints.actuated_names
    bodies = []
    for i, name in enumerate(robot.links.names):
        inertial = urdf.link_map[name].inertial
        if inertial is not None and inertial.mass:
            T = inertial.origin if inertial.origin is not None else np.eye(4)
            bodies.append((i, inertial.mass, jnp.asarray(T[:3, 3], jnp.float32)))

    def potential(q):
        Ts = robot.forward_kinematics(q)
        return sum(m * 9.81 * jaxlie.SE3(Ts[i]).apply(c)[2] for i, m, c in bodies)

    lo, hi = robot.joints.lower_limits, robot.joints.upper_limits
    z = jnp.zeros(robot.joints.num_actuated_joints)
    for k in range(5):
        q = jax.random.uniform(jax.random.PRNGKey(k), z.shape, minval=lo, maxval=hi)
        err = jnp.abs(robot.inverse_dynamics(q, z, z) - jax.grad(potential)(q))
        assert float(err[:7].max()) < 1e-2  # arm joints, N*m
        assert float(err[7]) < 0.2  # finger joint, N


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="GRiD needs a CUDA device")
def test_mimic_joint_grid_matches_jax(panda_description):
    _, robot = panda_description
    q, qd, qdd = (0.5 * jax.random.normal(k, (16, robot.joints.num_actuated_joints))
                  for k in jax.random.split(jax.random.PRNGKey(0), 3))
    np.testing.assert_allclose(
        robot.inverse_dynamics(q, qd, qdd, use_cuda=True),
        robot.inverse_dynamics(q, qd, qdd), atol=1e-3)


def test_default_runs_on_panda_description(panda_description):
    urdf, robot = panda_description
    sd = json.loads((RESOURCES / "panda_spheres.json").read_text())
    rc = RobotCollision.from_sphere_decomposition(sd, urdf)
    world = (Sphere.from_center_and_radius(jnp.array([[0.45, 0.0, 0.45]]), jnp.array([0.12])),)
    start = jnp.array([-1.2, 0.3, 0.0, -1.8, 0.0, 2.1, 0.8, 0.02])
    goal = start.at[0].set(1.2)
    T, B, dt = 32, 8, 0.1
    a = jnp.linspace(0.0, 1.0, T)[:, None]
    init = (start * (1 - a) + goal * a)[None] + 0.05 * jax.random.normal(
        jax.random.PRNGKey(0), (B, T, 8))
    best, _, _ = dynamics_aware_trajopt(
        init, start, goal, robot, rc, world, DynamicsAwareTrajOptConfig(dt=dt))
    assert _torque_ratio(robot, best, dt) <= 1.0
    world_clearance = jax.vmap(
        lambda q: rc.compute_world_collision_distance(robot, q, world[0]).min())(best)
    assert float(world_clearance.min()) >= 0.0


@pytest.mark.parametrize(
    "cfg, engine",
    [
        (None, "dynamics_aware_trajopt"),
        (ScoTrajOptConfig(), "sco_trajopt"),
        (ChompTrajOptConfig(), "chomp_trajopt"),
        (StompTrajOptConfig(), "stomp_trajopt"),
    ],
)
def test_generator_picks_engine_from_config_type(scene, monkeypatch, cfg, engine):
    robot, rc, world, start, goal, init = scene
    called = []

    def fake(name):
        def run(init_trajs, *args, **kwargs):
            called.append(name)
            return init_trajs[0], jnp.zeros(init_trajs.shape[0]), init_trajs
        return run

    for name in ("dynamics_aware_trajopt", "sco_trajopt", "chomp_trajopt", "stomp_trajopt"):
        monkeypatch.setattr(trajopt_module, name, fake(name))
    kwargs = {} if cfg is None else {"trajopt_cfg": cfg}
    gen = TrajoptMotionGenerator(robot, rc, world, "panda_hand", n_timesteps=32, n_batch=8, **kwargs)
    monkeypatch.setattr(gen, "_seed_trajectories", lambda *a, **k: (init, start, goal))
    if cfg is None:
        assert isinstance(gen.trajopt_cfg, DynamicsAwareTrajOptConfig)
    ident = jaxlie.SE3.identity()  # seeding is stubbed, poses are unused
    gen.generate(ident, ident, jax.random.PRNGKey(0))
    assert called == [engine]


def test_pyroki_wall_scene_fully_feasible(panda_description):
    """PyRoki example 07 (ground + 0.4 m wall, T=50, dt=0.02, endpoints from
    PyRoki's collision-aware IK). The straight line goes through the wall. PyRoki
    returns a path at ~2.4x the torque limit here; this engine must clear the
    wall along the whole path and respect joint, velocity and torque limits."""
    urdf, robot = panda_description
    sd = json.loads((RESOURCES / "panda_spheres.json").read_text())
    rc = RobotCollision.from_sphere_decomposition(sd, urdf)
    ground = pk.collision.HalfSpace.from_point_and_normal(jnp.zeros(3), jnp.array([0.0, 0.0, 1.0]))
    wall = pk.collision.Box.from_extent(jnp.array([0.4, 0.1, 0.4]), position=jnp.array([0.5, 0.0, 0.2]))
    start = jnp.array([-0.352321, 0.525767, -0.240692, -2.010926, -0.12214, 2.55608, -2.481458, 0.016184])
    goal = jnp.array([0.27369, 0.534774, 0.318855, -2.013845, 0.027342, 2.556996, -2.155888, 0.016184])
    dt = 0.02
    line = jnp.linspace(start, goal, 50)
    best, _, _ = dynamics_aware_trajopt(
        line[None], start, goal, robot, rc, (ground, wall),
        DynamicsAwareTrajOptConfig(dt=dt, collision_margin=0.05))

    a = jnp.linspace(0.0, 1.0, 9)[:-1, None, None]  # 8x, finer than the solver's 4
    dense = ((1 - a) * best[None, :-1] + a * best[None, 1:]).reshape(-1, best.shape[1])
    moving = jnp.arange(rc.num_links) != robot.links.names.index("panda_link0")
    wall_d = rc.compute_world_collision_distance(robot, dense, wall)
    assert float(jnp.where(moving[None, :, None], wall_d, jnp.inf).min()) >= 0.0
    assert float(jnp.where(moving[None, :, None], rc.compute_world_collision_distance(
        robot, dense, ground), jnp.inf).min()) >= 0.0
    lo, hi = robot.joints.lower_limits, robot.joints.upper_limits
    assert bool(jnp.all((best >= lo) & (best <= hi)))
    qd, _ = _fd_vel_acc(best, dt)
    assert float(jnp.max(jnp.abs(qd) / robot.joints.velocity_limits)) <= 1.0
    assert _torque_ratio(robot, best, dt) <= 1.0


def test_generator_default_falls_back_for_unsupported_robots(scene):
    """Auto default: dynamics-aware when supported, else SCO. An explicit
    DynamicsAwareTrajOptConfig on an unsupported robot raises a specific error
    (a floating-base robot is planned on a chart robot that has no dynamics, so
    the reason must be decided on the robot as given)."""
    robot, rc, world, *_ = scene
    assert isinstance(
        TrajoptMotionGenerator(robot, rc, world, "panda_hand").trajopt_cfg,
        DynamicsAwareTrajOptConfig)
    for bad, reason in ((jdc.replace(robot, dynamics=None), "no dynamics model"),
                        (jdc.replace(robot, floating_base=True), "floating-base")):
        gen = TrajoptMotionGenerator(bad, rc, world, "panda_hand")
        assert isinstance(gen.trajopt_cfg, ScoTrajOptConfig)
        with pytest.raises(ValueError, match=reason):
            TrajoptMotionGenerator(bad, rc, world, "panda_hand",
                                   trajopt_cfg=DynamicsAwareTrajOptConfig())
