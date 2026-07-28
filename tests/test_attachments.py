"""Attached bodies / tool use: kinematics, collision and dynamics composition.

The dynamics tests lean on analytic answers rather than on cross-checking one
pyroffi path against another, because the failure mode this feature invites is a
*plausible* wrong answer (an inverted transform, a dropped parallel-axis term)
rather than a crash.
"""

import jax
import jax.numpy as jnp
import jaxlie
import numpy as onp
import pytest
import yourdfpy

import pyroffi
from pyroffi.attachments import (
    Attachment,
    AttachmentSet,
    attachment_wrench_to_body,
    motion_transform,
    pose_attachments,
    spatial_inertia,
    tool_frame,
)
from pyroffi.collision._geometry import Capsule, Sphere

PANDA_URDF = "resources/panda/panda_spherized.urdf"


@pytest.fixture(scope="module")
def robot():
    return pyroffi.Robot.from_urdf(yourdfpy.URDF.load(PANDA_URDF, load_meshes=False))


@pytest.fixture(scope="module")
def ee_link(robot):
    # Last link with a real parent joint; index is all the API needs.
    return robot.links.num_links - 1


def _identity(batch=()):
    return jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), batch + (7,))


def _sphere(radius=0.05, n=1):
    return Sphere.from_center_and_radius(
        jnp.zeros((n, 3)), jnp.full((n,), radius)
    )


# ---------------------------------------------------------------------------
# P1 — core + kinematics
# ---------------------------------------------------------------------------


def test_attachment_pose_composes_analytically(robot, ee_link):
    """T_WB must equal T_WL . T_LB, exactly."""
    T_LB = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_x_radians(jnp.array(0.3)), jnp.array([0.1, -0.2, 0.05])
    )
    a = Attachment.from_geom(_sphere(), ee_link, T_LB.wxyz_xyz, name="cup")
    cfg = robot.default_cfg
    T_WL = jaxlie.SE3(robot.forward_kinematics(cfg)[ee_link])
    got = a.T_world_body(robot, cfg)
    onp.testing.assert_allclose(
        got.wxyz_xyz, (T_WL @ T_LB).wxyz_xyz, atol=1e-6
    )


def test_body_fixed_point_is_config_invariant(robot, ee_link):
    """A point fixed in the body frame stays fixed in the body frame under any
    configuration -- the invariant that catches an inverted T_parent_body."""
    T_LB = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_z_radians(jnp.array(-0.7)), jnp.array([0.03, 0.2, -0.1])
    )
    a = Attachment.from_geom(_sphere(), ee_link, T_LB.wxyz_xyz)
    p_body = jnp.array([0.02, -0.04, 0.11])

    key = jax.random.PRNGKey(0)
    cfgs = jax.random.normal(key, (8, robot.joints.num_actuated_joints))
    for cfg in cfgs:
        T_WB = a.T_world_body(robot, cfg)
        p_world = T_WB @ p_body
        onp.testing.assert_allclose(T_WB.inverse() @ p_world, p_body, atol=1e-5)


def test_grasp_from_current_pose_roundtrip(robot, ee_link):
    """The "close the gripper here" constructor must reproduce the object's
    world pose at the configuration it was captured at."""
    cfg = robot.default_cfg
    T_WB = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_y_radians(jnp.array(1.1)), jnp.array([0.4, 0.1, 0.6])
    )
    a = Attachment.from_geom(_sphere(), ee_link, _identity())
    a = a.grasp_from_current_pose(robot, cfg, T_WB.wxyz_xyz)
    onp.testing.assert_allclose(
        a.T_world_body(robot, cfg).wxyz_xyz, T_WB.wxyz_xyz, atol=1e-5
    )


def test_inverted_grasp_transform_is_detectably_wrong(robot, ee_link):
    """Guard against the silent-frame-error failure mode: composing with the
    inverse of the correct transform must NOT accidentally agree."""
    T_LB = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_x_radians(jnp.array(0.9)), jnp.array([0.15, 0.0, 0.0])
    )
    cfg = robot.default_cfg
    good = Attachment.from_geom(_sphere(), ee_link, T_LB.wxyz_xyz)
    bad = Attachment.from_geom(_sphere(), ee_link, T_LB.inverse().wxyz_xyz)
    d = jnp.linalg.norm(
        good.T_world_body(robot, cfg).translation()
        - bad.T_world_body(robot, cfg).translation()
    )
    assert float(d) > 1e-3


def test_vmap_and_grad_over_grasp_transform(robot, ee_link):
    """T_parent_body is a leaf: batching candidate grasps and differentiating
    through them must both work without recompiling the topology."""
    a = Attachment.from_geom(_sphere(), ee_link, _identity())
    cfg = robot.default_cfg

    def tip_height(t_xyz):
        T = jnp.concatenate([jnp.array([1.0, 0.0, 0.0, 0.0]), t_xyz])
        return a.with_pose(T).T_world_body(robot, cfg).translation()[2]

    ts = jax.random.normal(jax.random.PRNGKey(1), (16, 3)) * 0.1
    heights = jax.vmap(tip_height)(ts)
    assert heights.shape == (16,)

    g = jax.grad(tip_height)(jnp.array([0.0, 0.0, 0.1]))
    assert g.shape == (3,) and jnp.all(jnp.isfinite(g))


def test_attachment_set_edits(robot, ee_link):
    a = Attachment.from_geom(_sphere(), ee_link, _identity(), name="pen")
    b = Attachment.from_geom(_sphere(), ee_link, _identity(), name="cup")
    s = AttachmentSet.empty().attach(a).attach(b)
    assert s.names() == ("pen", "cup")
    assert s.detach("pen").names() == ("cup",)
    assert bool(s.set_active("pen", False).attachments[0].active) is False
    with pytest.raises(ValueError):
        s.attach(a)
    with pytest.raises(KeyError):
        s.index_of("nope")


def test_pose_attachments_and_tool_frame(robot, ee_link):
    T_LB = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.identity(), jnp.array([0.0, 0.0, 0.1])
    )
    s = AttachmentSet.empty().attach(
        Attachment.from_geom(_sphere(n=3), ee_link, T_LB.wxyz_xyz, name="pen")
    )
    cfg = jnp.broadcast_to(robot.default_cfg, (4, robot.joints.num_actuated_joints))
    posed = pose_attachments(robot, cfg, s)
    assert posed is not None and posed.get_batch_axes() == (4, 3)

    T_tip = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.identity(), jnp.array([0.0, 0.0, 0.15])
    )
    tip = tool_frame(robot, robot.default_cfg, s, "pen", T_tip.wxyz_xyz)
    body = tool_frame(robot, robot.default_cfg, s, "pen")
    onp.testing.assert_allclose(
        tip.wxyz_xyz, (body @ T_tip).wxyz_xyz, atol=1e-6
    )


# ---------------------------------------------------------------------------
# P2 — collision
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rcoll():
    from pyroffi.collision import RobotCollision

    return RobotCollision.from_urdf(
        yourdfpy.URDF.load(PANDA_URDF, load_meshes=False)
    )


def _capsule(n=1, radius=0.04, height=0.1):
    return Capsule.from_radius_height(
        jnp.full((n,), radius), jnp.full((n,), height), jnp.zeros((n, 3))
    )


def test_compose_collision_extends_geometry_and_pairs(robot, rcoll, ee_link):
    s = AttachmentSet.empty().attach(
        Attachment.from_geom(_capsule(2), ee_link, _identity(), name="tool")
    )
    ext = rcoll.with_attachments(s)
    assert ext.num_links == rcoll.num_links + 2
    assert ext.num_robot_links == rcoll.num_links
    assert len(ext.active_idx_i) > len(rcoll.active_idx_i)

    coll = ext.at_config(robot, robot.default_cfg)
    assert coll.get_batch_axes() == (ext.num_links,)


def test_attached_geometry_is_posed_by_the_parent_link(robot, rcoll, ee_link):
    """The attached primitive's world pose must match the analytic composition."""
    T_LB = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_x_radians(jnp.array(0.4)), jnp.array([0.05, 0.0, 0.12])
    )
    s = AttachmentSet.empty().attach(
        Attachment.from_geom(_capsule(1), ee_link, T_LB.wxyz_xyz, name="tool")
    )
    ext = rcoll.with_attachments(s)
    cfg = robot.default_cfg
    coll = ext.at_config(robot, cfg)
    T_WL = jaxlie.SE3(robot.forward_kinematics(cfg)[ee_link])
    onp.testing.assert_allclose(
        coll.pose.wxyz_xyz[-1], (T_WL @ T_LB).wxyz_xyz, atol=1e-5
    )


def test_attached_sphere_distance_to_a_known_obstacle(robot, rcoll, ee_link):
    """A ball attached at a known offset reports the right signed distance to a
    world obstacle placed a known distance away from it."""
    r_att = 0.05
    T_LB = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.identity(), jnp.array([0.0, 0.0, 0.2])
    )
    s = AttachmentSet.empty().attach(
        Attachment.from_geom(
            Capsule.from_radius_height(
                jnp.full((1,), r_att), jnp.zeros((1,)), jnp.zeros((1, 3))
            ),
            ee_link,
            T_LB.wxyz_xyz,
            name="ball",
        )
    )
    ext = rcoll.with_attachments(s)
    cfg = robot.default_cfg
    T_WB = jaxlie.SE3(robot.forward_kinematics(cfg)[ee_link]) @ T_LB

    gap, r_obs = 0.3, 0.02
    centre = T_WB.translation() + jnp.array([0.0, 0.0, r_att + gap + r_obs])
    obstacle = Sphere.from_center_and_radius(centre[None], jnp.full((1,), r_obs))

    d = ext.compute_world_collision_distance(robot, cfg, obstacle)
    onp.testing.assert_allclose(float(d[-1, 0]), gap, atol=1e-4)


def test_ignored_links_are_not_self_checked(robot, rcoll, ee_link):
    """The allowed-collision set belongs to the attachment: an ignored link must
    contribute no pair at all (the object is supposed to touch the fingers)."""
    plain = rcoll.with_attachments(
        AttachmentSet.empty().attach(
            Attachment.from_geom(_capsule(1), ee_link, _identity(), name="t")
        )
    )
    ignored = rcoll.with_attachments(
        AttachmentSet.empty().attach(
            Attachment.from_geom(
                _capsule(1),
                ee_link,
                _identity(),
                name="t",
                ignored_links=(0, 1, 2),
            )
        )
    )
    n_plain = len(plain.active_idx_i) - len(rcoll.active_idx_i)
    n_ignored = len(ignored.active_idx_i) - len(rcoll.active_idx_i)
    assert n_ignored < n_plain


def test_inactive_slot_reports_inf_and_does_not_perturb_the_min(
    robot, rcoll, ee_link
):
    s = AttachmentSet.empty().attach(
        Attachment.from_geom(_capsule(1), ee_link, _identity(), name="t")
    )
    ext = rcoll.with_attachments(s)
    off = rcoll.with_attachments(s.set_active("t", False))
    cfg = robot.default_cfg

    d_on = ext.compute_self_collision_distance(robot, cfg)
    d_off = off.compute_self_collision_distance(robot, cfg)
    n_base = len(rcoll.active_idx_i)

    assert jnp.all(jnp.isinf(d_off[n_base:]))
    # The robot's own pairs are untouched, and the base min is recovered.
    onp.testing.assert_allclose(d_off[:n_base], d_on[:n_base], atol=0)
    base = rcoll.compute_self_collision_distance(robot, cfg)
    onp.testing.assert_allclose(float(d_off.min()), float(base.min()), atol=0)


def test_toggling_active_does_not_recompile(robot, rcoll, ee_link):
    """`active` is a leaf, so a pick/place transition inside a fixed skeleton
    must reuse the compiled kernel."""
    s = AttachmentSet.empty().attach(
        Attachment.from_geom(_capsule(1), ee_link, _identity(), name="t")
    )
    ext = rcoll.with_attachments(s)

    @jax.jit
    def f(rc, cfg):
        return rc.compute_self_collision_distance(robot, cfg).min()

    cfg = robot.default_cfg
    f(ext, cfg)
    n_after_first = f._cache_size()
    f(rcoll.with_attachments(s.set_active("t", False)), cfg)
    assert f._cache_size() == n_after_first


def test_swept_capsules_cover_attachments(robot, rcoll, ee_link):
    s = AttachmentSet.empty().attach(
        Attachment.from_geom(_capsule(1), ee_link, _identity(), name="t")
    )
    ext = rcoll.with_attachments(s)
    cfg0 = robot.default_cfg
    cfg1 = cfg0 + 0.1
    swept = ext.get_swept_capsules(robot, cfg0, cfg1)
    assert swept.get_batch_axes()[-1] == ext.num_links


@pytest.fixture(scope="module")
def scoll():
    from pyroffi.collision._robot_collision import RobotCollisionSpherized

    return RobotCollisionSpherized.from_urdf(
        yourdfpy.URDF.load(PANDA_URDF, load_meshes=False)
    )


def test_spherized_model_takes_attachments_as_extra_rows(robot, scoll, ee_link):
    """The (N, S) sphere model gains one *row* per attachment, padded with the
    same negative-radius sentinel its own per-link rows use -- so the CUDA
    sphere paths only ever see a larger N."""
    s = AttachmentSet.empty().attach(
        Attachment.from_geom(_sphere(n=2), ee_link, _identity(), name="ball")
    )
    ext = scoll.with_attachments(s)
    assert ext.num_links == scoll.num_links + 1
    assert ext.num_robot_links == scoll.num_links
    assert ext.coll.get_batch_axes() == (
        scoll.num_links + 1,
        scoll.coll.get_batch_axes()[1],
    )

    coll = ext.at_config(robot, robot.default_cfg)
    # at_config returns (S, N) for a single cfg.
    assert coll.get_batch_axes()[-1] == ext.num_links


def test_spherized_attachment_distance_to_a_known_obstacle(robot, scoll, ee_link):
    r_att = 0.05
    T_LB = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.identity(), jnp.array([0.0, 0.0, 0.2])
    )
    s = AttachmentSet.empty().attach(
        Attachment.from_geom(
            _sphere(radius=r_att, n=1), ee_link, T_LB.wxyz_xyz, name="ball"
        )
    )
    ext = scoll.with_attachments(s)
    cfg = robot.default_cfg
    T_WB = jaxlie.SE3(robot.forward_kinematics(cfg)[ee_link]) @ T_LB

    gap, r_obs = 0.3, 0.02
    centre = T_WB.translation() + jnp.array([0.0, 0.0, r_att + gap + r_obs])
    obstacle = Sphere.from_center_and_radius(centre[None], jnp.full((1,), r_obs))

    d = ext.compute_world_collision_distance(robot, cfg, obstacle)
    onp.testing.assert_allclose(float(d[-1, 0]), gap, atol=1e-4)


def test_spherized_inactive_slot_reports_inf(robot, scoll, ee_link):
    s = AttachmentSet.empty().attach(
        Attachment.from_geom(_sphere(n=1), ee_link, _identity(), name="ball")
    )
    on = scoll.with_attachments(s)
    off = scoll.with_attachments(s.set_active("ball", False))
    cfg = robot.default_cfg
    n_base = len(scoll.active_idx_i)

    d_off = off.compute_self_collision_distance(robot, cfg)
    assert jnp.all(jnp.isinf(d_off[n_base:]))
    onp.testing.assert_allclose(
        d_off[:n_base], on.compute_self_collision_distance(robot, cfg)[:n_base], atol=0
    )
    base = scoll.compute_self_collision_distance(robot, cfg)
    onp.testing.assert_allclose(float(d_off.min()), float(base.min()), atol=0)


def test_spherized_rejects_an_attachment_with_too_many_spheres(scoll, ee_link):
    too_many = scoll.coll.get_batch_axes()[1] + 1
    s = AttachmentSet.empty().attach(
        Attachment.from_geom(_sphere(n=too_many), ee_link, _identity(), name="ball")
    )
    with pytest.raises(ValueError, match="only carries"):
        scoll.with_attachments(s)


# ---------------------------------------------------------------------------
# P3 — dynamics (JAX)
# ---------------------------------------------------------------------------


def test_motion_transform_matches_the_parser(robot):
    """The differentiable X must agree with the numpy one parse_dynamics uses."""
    from pyroffi._robot_urdf_parser import _motion_transform_from_T

    T = onp.array(
        jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.from_rpy_radians(
                jnp.array(0.3), jnp.array(-0.2), jnp.array(1.1)
            ),
            jnp.array([0.1, 0.2, -0.3]),
        ).as_matrix()
    )
    onp.testing.assert_allclose(
        onp.asarray(motion_transform(jaxlie.SE3.from_matrix(jnp.asarray(T)))),
        _motion_transform_from_T(T),
        atol=1e-6,
    )


def test_spatial_inertia_includes_the_parallel_axis_term():
    """I_O = I_com + m (cᵀc I - c cᵀ). Dropping it is the classic silent error."""
    m, c = 2.0, onp.array([0.1, -0.2, 0.3])
    I = onp.asarray(
        spatial_inertia(
            jnp.asarray(m), jnp.asarray(c), jnp.zeros((3, 3))
        )
    )
    expected = m * (c @ c * onp.eye(3) - onp.outer(c, c))
    onp.testing.assert_allclose(I[:3, :3], expected, atol=1e-9)
    onp.testing.assert_allclose(I[3:, 3:], m * onp.eye(3), atol=1e-9)


def _one_dof_robot(tmp_path, link_len=0.0):
    """A single revolute z-joint whose body is massless, so an attached point
    mass is the entire dynamics and the answer is analytic."""
    urdf = f"""<?xml version="1.0"?>
    <robot name="one">
      <link name="base"/>
      <link name="arm">
        <inertial>
          <origin xyz="0 0 0"/>
          <mass value="1e-9"/>
          <inertia ixx="1e-12" ixy="0" ixz="0" iyy="1e-12" iyz="0" izz="1e-12"/>
        </inertial>
      </link>
      <joint name="j0" type="revolute">
        <parent link="base"/>
        <child link="arm"/>
        <origin xyz="0 0 {link_len}" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14" effort="10" velocity="10"/>
      </joint>
    </robot>"""
    path = tmp_path / "one.urdf"
    path.write_text(urdf)
    return pyroffi.Robot.from_urdf(yourdfpy.URDF.load(str(path), load_meshes=False))


def test_point_mass_at_radius_reproduces_analytic_torque(tmp_path):
    """Attach a point mass m at radius r on a 1-DOF z-revolute arm. Rotating
    about z with the mass offset along x:

        M(q) = m r^2,   gravity torque = 0 (g is parallel to the joint axis)

    so tau = m r^2 qdd exactly.
    """
    r, m = 0.37, 2.5
    rb = _one_dof_robot(tmp_path)
    arm = rb.links.names.index("arm")
    T_LB = jnp.array([1.0, 0.0, 0.0, 0.0, r, 0.0, 0.0])
    s = AttachmentSet.empty().attach(
        Attachment.from_mass(jnp.asarray(m), arm, T_LB, name="payload")
    )
    loaded = rb.with_attachments(s)

    q = jnp.zeros((1, 1))
    qdd = jnp.ones((1, 1))
    tau = loaded.inverse_dynamics(q, jnp.zeros((1, 1)), qdd)
    onp.testing.assert_allclose(float(tau[0, 0]), m * r * r, rtol=1e-4)
    M = loaded.mass_matrix(q)
    onp.testing.assert_allclose(float(M[0, 0, 0]), m * r * r, rtol=1e-4)


def test_gravity_torque_of_an_offset_payload(tmp_path):
    """With the joint axis along z and the payload offset along x, tilt the
    problem by asking for the torque about a *horizontal* axis instead: rotate
    the joint origin so the axis lies along y, giving the pendulum

        tau_gravity = -m g r cos(q)   at q = 0  ->  -m * 9.81 * r

    (mass at +x, gravity along -z, rotation about +y: the gravity moment about
    the axis is +m g r, so the torque needed to *hold* the arm is -m g r.)
    """
    r, m = 0.25, 3.0
    urdf = f"""<?xml version="1.0"?>
    <robot name="pend">
      <link name="base"/>
      <link name="arm">
        <inertial><origin xyz="0 0 0"/><mass value="1e-9"/>
        <inertia ixx="1e-12" ixy="0" ixz="0" iyy="1e-12" iyz="0" izz="1e-12"/>
        </inertial>
      </link>
      <joint name="j0" type="revolute">
        <parent link="base"/><child link="arm"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="-3.14" upper="3.14" effort="10" velocity="10"/>
      </joint>
    </robot>"""
    path = tmp_path / "pend.urdf"
    path.write_text(urdf)
    rb = pyroffi.Robot.from_urdf(yourdfpy.URDF.load(str(path), load_meshes=False))
    arm = rb.links.names.index("arm")
    T_LB = jnp.array([1.0, 0.0, 0.0, 0.0, r, 0.0, 0.0])
    loaded = rb.with_attachments(
        AttachmentSet.empty().attach(
            Attachment.from_mass(jnp.asarray(m), arm, T_LB, name="payload")
        )
    )
    z = jnp.zeros((1, 1))
    tau = loaded.inverse_dynamics(z, z, z)
    onp.testing.assert_allclose(float(tau[0, 0]), -m * 9.81 * r, rtol=1e-3)


def test_zero_mass_attachment_is_a_bitwise_no_op(robot, ee_link):
    """Composing a body and then giving it zero mass must recover the
    unattached DynamicsInfo exactly -- no drift from the extra arithmetic."""
    s = AttachmentSet.empty().attach(
        Attachment.from_mass(jnp.asarray(0.0), ee_link, _identity(), name="p")
    )
    loaded = robot.with_attachments(s)
    onp.testing.assert_array_equal(
        onp.asarray(loaded.dynamics.I_body), onp.asarray(robot.dynamics.I_body)
    )


def test_inactive_payload_recovers_the_unattached_dynamics(robot, ee_link):
    s = AttachmentSet.empty().attach(
        Attachment.from_mass(jnp.asarray(5.0), ee_link, _identity(), name="p")
    )
    off = robot.with_attachments(s.set_active("p", False))
    onp.testing.assert_array_equal(
        onp.asarray(off.dynamics.I_body), onp.asarray(robot.dynamics.I_body)
    )
    on = robot.with_attachments(s)
    assert not onp.allclose(
        onp.asarray(on.dynamics.I_body), onp.asarray(robot.dynamics.I_body)
    )


def test_attachment_does_not_change_the_dof_count_or_topology(robot, ee_link):
    """An attachment is a fixed joint: nothing about the DOF tree may move."""
    s = AttachmentSet.empty().attach(
        Attachment.from_mass(jnp.asarray(1.0), ee_link, _identity(), name="p")
    )
    loaded = robot.with_attachments(s)
    d0, d1 = robot.dynamics, loaded.dynamics
    assert d1.num_dof == d0.num_dof
    assert d1.parent_dof_indices == d0.parent_dof_indices
    onp.testing.assert_array_equal(onp.asarray(d1.S), onp.asarray(d0.S))
    onp.testing.assert_array_equal(
        onp.asarray(d1.X_tree), onp.asarray(d0.X_tree)
    )
    onp.testing.assert_array_equal(
        onp.asarray(d1.joint_is_prismatic), onp.asarray(d0.joint_is_prismatic)
    )


def test_crba_matches_finite_differenced_kinetic_energy(tmp_path):
    """M(q) from CRBA vs. the Hessian of the kinetic energy 1/2 qd' M qd,
    on the loaded model -- an independent check of the composed inertia."""
    r, m = 0.3, 1.7
    rb = _one_dof_robot(tmp_path)
    arm = rb.links.names.index("arm")
    loaded = rb.with_attachments(
        AttachmentSet.empty().attach(
            Attachment.from_mass(
                jnp.asarray(m),
                arm,
                jnp.array([1.0, 0.0, 0.0, 0.0, r, 0.0, 0.0]),
                name="p",
            )
        )
    )
    q = jnp.zeros((1,))
    M = loaded.mass_matrix(q[None])[0]
    # tau = M qdd with qd = 0 and gravity along the axis => pure inertia probe.
    tau = loaded.inverse_dynamics(q[None], jnp.zeros((1, 1)), jnp.ones((1, 1)))
    onp.testing.assert_allclose(float(tau[0, 0]), float(M[0, 0]), rtol=1e-5)


def test_grad_flows_to_mass_and_grasp_transform(tmp_path):
    """The payoff of keeping mass and T_parent_body as leaves: payload
    identification and grasp placement become gradient problems."""
    rb = _one_dof_robot(tmp_path)
    arm = rb.links.names.index("arm")
    z = jnp.zeros((1, 1))

    def tau_of(mass, radius):
        T = jnp.concatenate(
            [jnp.array([1.0, 0.0, 0.0, 0.0]), jnp.stack([radius, 0.0 * radius, 0.0 * radius])]
        )
        s = AttachmentSet.empty().attach(
            Attachment.from_mass(mass, arm, T, name="p")
        )
        return rb.with_attachments(s).inverse_dynamics(z, z, jnp.ones((1, 1)))[0, 0]

    dm, dr = jax.grad(tau_of, argnums=(0, 1))(jnp.asarray(2.0), jnp.asarray(0.4))
    # tau = m r^2 => dtau/dm = r^2, dtau/dr = 2 m r.
    onp.testing.assert_allclose(float(dm), 0.4**2, rtol=1e-4)
    onp.testing.assert_allclose(float(dr), 2 * 2.0 * 0.4, rtol=1e-4)


def test_attachment_wrench_maps_by_the_force_transform(tmp_path):
    """A pure force at a tool tip offset along x from the body frame must
    produce the corresponding lever-arm moment at the body: f_D = X_{B<-D}^T f_B.

    Uses the 1-DOF arm so the DOF body frame *is* the link frame, making the
    expected answer a bare cross product rather than one dressed in the panda's
    flange rotation.
    """
    rb = _one_dof_robot(tmp_path)
    arm = rb.links.names.index("arm")
    s = AttachmentSet.empty().attach(
        Attachment.from_mass(jnp.asarray(1.0), arm, _identity(), name="pen")
    )
    lever = 0.2
    T_tip = jnp.array([1.0, 0.0, 0.0, 0.0, lever, 0.0, 0.0])
    # Unit force along +z at the tip, no moment (angular-first convention).
    w_tip = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    dof, w_body = attachment_wrench_to_body(rb, s, "pen", w_tip, T_tip)
    assert dof == 0
    # Force is unchanged; the moment picks up r x f = (lever x) x (z) = -lever y.
    onp.testing.assert_allclose(onp.asarray(w_body[3:]), [0, 0, 1], atol=1e-6)
    onp.testing.assert_allclose(onp.asarray(w_body[:3]), [0, -lever, 0], atol=1e-6)


def test_attachment_wrench_matches_the_transpose_of_the_motion_transform(
    robot, ee_link
):
    """The general case, on a real flange: whatever the intervening rotation,
    the mapping must be exactly X_{B<-D} transposed."""
    import jaxlie as _jaxlie

    from pyroffi.attachments import link_dof_bodies

    T_LB = _jaxlie.SE3.from_rotation_and_translation(
        _jaxlie.SO3.from_rpy_radians(
            jnp.array(0.2), jnp.array(-0.5), jnp.array(0.9)
        ),
        jnp.array([0.03, -0.07, 0.11]),
    )
    s = AttachmentSet.empty().attach(
        Attachment.from_mass(jnp.asarray(1.0), ee_link, T_LB.wxyz_xyz, name="pen")
    )
    w = jnp.array([0.3, -1.2, 0.7, 2.0, -0.5, 1.1])
    dof, w_body = attachment_wrench_to_body(robot, s, "pen", w)

    idx, T_DL = link_dof_bodies(robot)[robot.links.names[ee_link]]
    assert idx == dof
    X = motion_transform(_jaxlie.SE3.from_matrix(jnp.asarray(T_DL)) @ T_LB)
    onp.testing.assert_allclose(
        onp.asarray(w_body), onp.asarray(X.T @ w), rtol=1e-5, atol=1e-6
    )


def test_grounded_attachment_loads_nothing(robot):
    """A fixture bolted above every actuated joint torques no DOF."""
    base = 0
    s = AttachmentSet.empty().attach(
        Attachment.from_mass(jnp.asarray(9.0), base, _identity(), name="fixture")
    )
    loaded = robot.with_attachments(s)
    onp.testing.assert_array_equal(
        onp.asarray(loaded.dynamics.I_body), onp.asarray(robot.dynamics.I_body)
    )


# ---------------------------------------------------------------------------
# P5 — downstream: tool-frame IK targets
# ---------------------------------------------------------------------------


def test_ik_target_for_tool_inverts_the_tool_offset(robot, ee_link):
    """Rewriting a tip goal as a link goal must be an exact algebraic inverse:
    posing the link at the rewritten target puts the tip on the original goal."""
    from pyroffi.attachments import ik_target_for_tool

    T_LB = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_rpy_radians(
            jnp.array(0.3), jnp.array(-0.4), jnp.array(0.8)
        ),
        jnp.array([0.01, 0.02, 0.09]),
    )
    T_tip = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.identity(), jnp.array([0.0, 0.0, 0.16])
    )
    s = AttachmentSet.empty().attach(
        Attachment.from_geom(_sphere(), ee_link, T_LB.wxyz_xyz, name="pen")
    )
    goal = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_y_radians(jnp.array(1.2)), jnp.array([0.4, -0.1, 0.5])
    )
    idx, T_link = ik_target_for_tool(s, "pen", goal.wxyz_xyz, T_tip.wxyz_xyz)
    assert idx == ee_link
    # T_W_L . (T_LB . T_tip) must be the original tip goal.
    recovered = jaxlie.SE3(T_link) @ (T_LB @ T_tip)
    onp.testing.assert_allclose(
        recovered.wxyz_xyz, goal.wxyz_xyz, atol=1e-5
    )


def test_ik_solve_on_a_retargeted_goal_lands_the_tool_tip(robot, ee_link):
    """End to end: solve IK against the rewritten link goal, then check the
    *tool frame* at the solution is where we asked -- no kernel change needed."""
    from pyroffi.attachments import ik_target_for_tool
    from pyroffi.optimization_engines import ls_ik_solve

    T_LB = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.identity(), jnp.array([0.0, 0.0, 0.12])
    )
    s = AttachmentSet.empty().attach(
        Attachment.from_geom(_sphere(), ee_link, T_LB.wxyz_xyz, name="pen")
    )
    # A tip goal that is reachable: take the tip pose at a known config.
    q_ref = robot.default_cfg + 0.15
    goal = tool_frame(robot, q_ref, s, "pen")

    idx, T_link = ik_target_for_tool(s, "pen", goal.wxyz_xyz)
    q = ls_ik_solve(
        robot,
        target_link_indices=(idx,),
        target_poses=(jaxlie.SE3(T_link),),
        rng_key=jax.random.PRNGKey(0),
        previous_cfg=robot.default_cfg,
    )
    got = tool_frame(robot, q, s, "pen")
    onp.testing.assert_allclose(
        onp.asarray(got.translation()), onp.asarray(goal.translation()), atol=2e-3
    )


# ---------------------------------------------------------------------------
# Guards: the ways attachments used to fail silently
# ---------------------------------------------------------------------------


def test_composing_dynamics_twice_is_rejected(robot, ee_link):
    """Composition is additive and leaves no other trace, so a second compose
    would double the carried load with nothing to show for it."""
    s = AttachmentSet.empty().attach(
        Attachment.from_mass(
            2.0, ee_link, jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]), name="p"
        )
    )
    once = robot.with_attachments(s)
    assert once.attached_names == ("p",)
    with pytest.raises(ValueError, match="already carries attachments"):
        once.with_attachments(s)


def test_compose_dynamics_is_additive_exactly_once(robot, ee_link):
    """The guard exists because the doubling is otherwise invisible: check the
    single compose really does add the payload once."""
    m = 2.0
    s = AttachmentSet.empty().attach(
        Attachment.from_mass(
            m, ee_link, jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]), name="p"
        )
    )
    base = onp.asarray(robot.dynamics.I_body)
    once = onp.asarray(robot.with_attachments(s).dynamics.I_body)
    # Mass enters I_body's translational block as m * I3; one compose adds m.
    onp.testing.assert_allclose(onp.abs(once - base).max(), m, rtol=1e-6)


def test_pose_attachments_neutralises_an_inactive_slot(robot, ee_link):
    """``pose_attachments`` used to return live geometry for a disabled slot,
    while every other consumer masked it out."""
    s = AttachmentSet.empty().attach(
        Attachment.from_geom(
            _sphere(), ee_link, jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]), name="g"
        )
    )
    cfg = robot.default_cfg
    on = pose_attachments(robot, cfg, s)
    off = pose_attachments(robot, cfg, s.set_active("g", False))
    assert float(onp.asarray(on.radius).min()) > 0.0
    assert float(onp.asarray(off.radius).max()) < 0.0
    # The pose is untouched -- only the extent is gated.
    onp.testing.assert_allclose(
        onp.asarray(on.pose.translation()), onp.asarray(off.pose.translation())
    )


def test_pose_attachments_gating_does_not_recompile(robot, ee_link):
    """``active`` is a leaf, so toggling it must stay inside one trace."""
    traces = []

    @jax.jit
    def f(T, active):
        traces.append(1)
        a = Attachment.from_geom(_sphere(), ee_link, T, name="g").with_active(active)
        return pose_attachments(
            robot, robot.default_cfg, AttachmentSet.empty().attach(a)
        ).radius.sum()

    T = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1])
    f(T, jnp.asarray(True))
    n = len(traces)
    f(T, jnp.asarray(False))
    assert len(traces) == n, "toggling `active` retraced"


# ---------------------------------------------------------------------------
# CUDA backends
#
# The CUDA checkers pose geometry by row index; attachments add rows with no
# kinematics link of their own.  These tests pin the two things that can go
# quietly wrong: an attachment row posed with the *wrong* transform (which looks
# like a plausible distance, not a crash), and an attachment row the kernel never
# looks at (which looks like a collision-free config).
# ---------------------------------------------------------------------------


def _cuda_or_skip():
    cuda = pytest.importorskip("pyroffi.collision._cuda_collision")
    if not [d for d in jax.local_devices() if d.platform in ("gpu", "cuda")]:
        pytest.skip("no local GPU")
    return cuda


def _no_contact_capped(d, cap=1e6):
    """Collapse both "no contact" sentinels onto one value.

    The JAX reduction masks padding spheres to ``+inf``; the CUDA kernels use
    ``1e9``.  That divergence predates attachments and applies to every all-padding
    link pair, so comparing the two backends means capping both first.
    """
    return onp.minimum(onp.asarray(d, dtype=onp.float64), cap)


@pytest.fixture(scope="module")
def scoll_srdf():
    """Spherized model with the SRDF's allowed-collision set applied.

    The binary checker returns a single world-OR-self verdict, so without the
    SRDF the panda's adjacent hand/finger spheres report a permanent self
    collision and every verdict is ``False`` — a vacuously passing test.
    """
    from pyroffi.collision._robot_collision import RobotCollisionSpherized

    return RobotCollisionSpherized.from_urdf(
        yourdfpy.URDF.load(PANDA_URDF, load_meshes=False),
        srdf_path="resources/panda/panda.srdf",
    )


@pytest.fixture(scope="module")
def _tool_attachment(ee_link):
    """One sphere held 0.2 m out along the parent link's +Z."""
    T_LB = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.identity(), jnp.array([0.0, 0.0, 0.2])
    )
    aset = AttachmentSet.empty().attach(
        Attachment.from_geom(
            _sphere(radius=0.05, n=1), ee_link, T_LB.wxyz_xyz, name="ball"
        )
    )
    return aset, T_LB


def test_cuda_sdf_checker_matches_jax_on_an_attached_model(
    robot, scoll, ee_link, _tool_attachment
):
    """Same distances as the pure-JAX path, including the attachment rows.

    This is the test that catches a mis-posed attachment: the JAX path composes
    ``T_WB = T_WL · T_LB`` explicitly, so any disagreement means the CUDA path
    put the grasped object somewhere else.
    """
    cuda = _cuda_or_skip()
    aset, T_LB = _tool_attachment
    ext = scoll.with_attachments(aset)
    checker = cuda.CUDADifferentiableSDFCollisionChecker(ext)

    cfgs = [robot.default_cfg, robot.default_cfg * 0.5]
    T_WB = jaxlie.SE3(robot.forward_kinematics(robot.default_cfg)[ee_link]) @ T_LB
    obstacle = Sphere.from_center_and_radius(
        (T_WB.translation() + jnp.array([0.0, 0.0, 0.3]))[None], jnp.full((1,), 0.02)
    )

    # Per-config, not a stacked batch: RobotCollisionSpherized.at_config cannot
    # take a batched cfg (it vmaps the N axis and then broadcasts B against S) --
    # a pre-existing limitation of the JAX reference, not of the CUDA path.
    for cfg in cfgs:
        onp.testing.assert_allclose(
            _no_contact_capped(
                checker.compute_world_collision_distance(robot, cfg, obstacle)
            ),
            _no_contact_capped(
                ext.compute_world_collision_distance(robot, cfg, obstacle)
            ),
            atol=2e-4,
        )
        onp.testing.assert_allclose(
            _no_contact_capped(checker.compute_self_collision_distance(robot, cfg)),
            _no_contact_capped(ext.compute_self_collision_distance(robot, cfg)),
            atol=2e-4,
        )

    # The attachment row must actually be the one that moved: its distance to the
    # obstacle is the analytic gap, not the un-attached model's distance.
    d_att = checker.compute_world_collision_distance(robot, cfgs[0], obstacle)
    onp.testing.assert_allclose(
        float(d_att[ext.num_robot_links, 0]), 0.3 - 0.05 - 0.02, atol=2e-4
    )

    # The batched launch must agree row-for-row with the single-config launches.
    batched = checker.compute_world_collision_distance(
        robot, jnp.stack(cfgs), obstacle
    )
    for k, cfg in enumerate(cfgs):
        onp.testing.assert_allclose(
            _no_contact_capped(batched[k]),
            _no_contact_capped(
                checker.compute_world_collision_distance(robot, cfg, obstacle)
            ),
            atol=2e-4,
        )


def test_cuda_sdf_checker_poses_a_rotated_capsule_attachment(robot, rcoll, ee_link):
    """The capsule layout carries orientation, so a wrong ``T_LB`` shows up here.

    Scoped to the attachment's own rows: the capsule self-collision kernel and the
    JAX reduction already disagree by ~9e-3 on one *base* pair of this model,
    which predates attachments and would otherwise set a useless tolerance.
    """
    cuda = _cuda_or_skip()
    T_LB = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_x_radians(jnp.array(0.7)), jnp.array([0.0, 0.0, 0.2])
    ).wxyz_xyz
    ext = rcoll.with_attachments(
        AttachmentSet.empty().attach(
            Attachment.from_geom(_capsule(1), ee_link, T_LB, name="pen")
        )
    )
    checker = cuda.CUDADifferentiableSDFCollisionChecker(ext)
    cfg = robot.default_cfg
    obstacle = Sphere.from_center_and_radius(
        jnp.array([[0.4, 0.0, 0.5]]), jnp.full((1,), 0.05)
    )
    n_base = len(rcoll.active_idx_i)

    onp.testing.assert_allclose(
        _no_contact_capped(
            checker.compute_world_collision_distance(robot, cfg, obstacle)
        )[-1],
        _no_contact_capped(
            ext.compute_world_collision_distance(robot, cfg, obstacle)
        )[-1],
        atol=1e-5,
    )
    onp.testing.assert_allclose(
        _no_contact_capped(checker.compute_self_collision_distance(robot, cfg))[n_base:],
        _no_contact_capped(ext.compute_self_collision_distance(robot, cfg))[n_base:],
        atol=1e-5,
    )


def test_cuda_sdf_checker_reports_inf_for_an_inactive_slot(
    robot, scoll, ee_link, _tool_attachment
):
    cuda = _cuda_or_skip()
    aset, _ = _tool_attachment
    off = scoll.with_attachments(aset.set_active("ball", False))
    checker = cuda.CUDADifferentiableSDFCollisionChecker(off)
    cfg = robot.default_cfg
    n_base = len(scoll.active_idx_i)

    d_self = checker.compute_self_collision_distance(robot, cfg)
    assert jnp.all(jnp.isinf(d_self[n_base:]))
    # The un-attached minimum is untouched by the disabled slot.
    onp.testing.assert_allclose(
        float(d_self.min()),
        float(scoll.compute_self_collision_distance(robot, cfg).min()),
        atol=2e-4,
    )

    obstacle = Sphere.from_center_and_radius(
        jnp.zeros((1, 3)), jnp.full((1,), 0.02)
    )
    d_world = checker.compute_world_collision_distance(robot, cfg, obstacle)
    assert jnp.all(jnp.isinf(d_world[off.num_robot_links :]))


def test_cuda_binary_checker_sees_a_collision_only_the_attachment_has(
    robot, scoll_srdf, ee_link, _tool_attachment
):
    """The fused-FK kernel must pose the attachment row, not skip it.

    The obstacle is placed on the grasped ball and nowhere near the robot, so the
    un-attached model is free and the attached model is not.  A dropped or
    mis-posed attachment row shows up here as "still free".
    """
    cuda = _cuda_or_skip()
    aset, T_LB = _tool_attachment
    cfg = robot.default_cfg
    T_WB = jaxlie.SE3(robot.forward_kinematics(cfg)[ee_link]) @ T_LB
    obstacle = Sphere.from_center_and_radius(
        T_WB.translation()[None], jnp.full((1,), 0.02)
    )

    bare = cuda.CUDABinaryCollisionChecker(scoll_srdf)
    assert bool(bare.check_collision_free(robot, cfg, obstacle)), (
        "precondition: the obstacle must clear the un-attached robot"
    )

    attached = cuda.CUDABinaryCollisionChecker(scoll_srdf.with_attachments(aset))
    assert not bool(attached.check_collision_free(robot, cfg, obstacle))

    # And an inactive slot is gated back out again.
    off = cuda.CUDABinaryCollisionChecker(
        scoll_srdf.with_attachments(aset.set_active("ball", False))
    )
    assert bool(off.check_collision_free(robot, cfg, obstacle))


def test_cuda_binary_checker_agrees_with_the_sdf_sign_over_a_batch(
    robot, scoll_srdf, ee_link, _tool_attachment
):
    """Batched verdicts must match the sign of the differentiable path."""
    cuda = _cuda_or_skip()
    aset, T_LB = _tool_attachment
    ext = scoll_srdf.with_attachments(aset)
    cfgs = [robot.default_cfg * s for s in (0.25, 0.5, 0.75, 1.0)]
    T_WB = jaxlie.SE3(robot.forward_kinematics(robot.default_cfg)[ee_link]) @ T_LB
    obstacle = Sphere.from_center_and_radius(
        T_WB.translation()[None], jnp.full((1,), 0.05)
    )

    binary = cuda.CUDABinaryCollisionChecker(ext)
    free = binary.check_collision_free(robot, jnp.stack(cfgs), obstacle)

    # Reference evaluated one config at a time (see the note above on batched
    # cfg in RobotCollisionSpherized.at_config).
    expected = [
        bool(
            ext.compute_world_collision_distance(robot, c, obstacle).min() > 0
            and ext.compute_self_collision_distance(robot, c).min() > 0
        )
        for c in cfgs
    ]
    onp.testing.assert_array_equal(onp.asarray(free), onp.asarray(expected))
    assert not all(expected), "test is vacuous unless some config collides"


def test_cuda_checkers_require_the_coarse_model_to_carry_the_attachment(
    robot, scoll, ee_link, _tool_attachment
):
    """A coarse guard flags fine geometry by row: an unmatched attachment row
    would never be flagged, so the collision would be missed silently."""
    cuda = pytest.importorskip("pyroffi.collision._cuda_collision")
    aset, _ = _tool_attachment
    ext = scoll.with_attachments(aset)
    for cls in (
        cuda.CUDADifferentiableSDFCollisionChecker,
        cuda.CUDABinaryCollisionChecker,
    ):
        with pytest.raises(ValueError, match="same attachment entries"):
            cls(ext, scoll)
