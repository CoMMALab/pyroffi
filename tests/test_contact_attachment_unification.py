"""``dynamics._contact`` and ``pyroffi.attachments`` describe the same grasp.

The point of the unification is that grasp bookkeeping exists once. These tests
pin that: the offsets ``ContactSystem.from_attachments`` derives must be exactly
the ones ``capture_grasp_offsets`` used to capture independently, and the
resulting closure residual must be unchanged -- otherwise the two trajopt
solvers calibrated against the old path would quietly shift.
"""

import jax.numpy as jnp
import jaxlie
import numpy as onp
import pytest
import yourdfpy

import pyroffi

PANDA_URDF = "resources/panda/panda_spherized.urdf"

pytest.importorskip("jax")


@pytest.fixture(scope="module")
def pieces():
    from pyroffi.collision import Box
    from pyroffi.dynamics._contact import ManipulatorSpec

    urdf = yourdfpy.URDF.load(PANDA_URDF, load_meshes=False)
    robot = pyroffi.Robot.from_urdf(urdf)
    # ManipulatorSpec wants a GRiDDynamics only for num_dof / fext / jacobian;
    # the grasp bookkeeping under test is pure kinematics, so a stub keeps
    # these tests runnable without nvcc.
    class _StubGrid:
        num_dof = robot.joints.num_actuated_joints

    left = ManipulatorSpec(
        robot, _StubGrid(), "panda_hand", base_xy_yaw=(-0.4, 0.0, 0.0),
        p_local=(0.0, 0.0, 0.1),
    )
    right = ManipulatorSpec(
        robot, _StubGrid(), "panda_hand", base_xy_yaw=(0.4, 0.0, onp.pi),
        p_local=(0.0, 0.0, 0.1),
    )
    mid = (robot.joints.lower_limits + robot.joints.upper_limits) / 2
    box = Box.from_center_and_dimensions(
        center=jnp.zeros(3), length=0.12, width=0.12, height=0.12, mass=3.0
    )
    return robot, (left, right), mid, box


def test_derived_offsets_match_the_captured_ones(pieces):
    """``A_ref · A_i^-1`` must reproduce ``T_ref^-1 · T_i`` exactly."""
    from pyroffi.dynamics._contact import (
        ContactSystem,
        _gripper_world_pose,
        capture_attachments,
        capture_grasp_offsets,
    )

    _, manips, mid, box = pieces
    captured = capture_grasp_offsets(manips, (mid, mid))

    # Capture the object anywhere -- the derived offset must not depend on it.
    T_obj = _gripper_world_pose(manips[0], mid) @ jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_x_radians(jnp.array(0.4)), jnp.array([0.02, -0.05, 0.13])
    )
    atts = capture_attachments(manips, (mid, mid), T_obj, geom=box)
    sys_a = ContactSystem.from_attachments(manips, atts)

    for got, want in zip(sys_a.grasp_offsets, captured):
        onp.testing.assert_allclose(
            onp.asarray(got.wxyz_xyz), onp.asarray(want.wxyz_xyz), atol=1e-5
        )


def test_closure_residual_is_unchanged_by_the_attachment_path(pieces):
    from pyroffi.dynamics._contact import (
        ContactSystem,
        GraspedObject,
        _gripper_world_pose,
        capture_attachments,
        capture_grasp_offsets,
        grasp_closure_residual,
    )

    _, manips, mid, box = pieces
    q = jnp.concatenate([mid, mid])

    old = ContactSystem(
        manips, GraspedObject(geom=box), capture_grasp_offsets(manips, (mid, mid))
    )
    atts = capture_attachments(
        manips, (mid, mid), _gripper_world_pose(manips[0], mid), geom=box
    )
    new = ContactSystem.from_attachments(manips, atts)

    onp.testing.assert_allclose(
        onp.asarray(grasp_closure_residual(new, q)),
        onp.asarray(grasp_closure_residual(old, q)),
        atol=1e-6,
    )


def test_capture_accounts_for_the_manipulator_base_transform(pieces):
    """The classic silent error: capturing through raw FK instead of the world
    gripper pose gives every non-origin manipulator a wrong grasp transform."""
    from pyroffi.dynamics._contact import (
        _gripper_world_pose,
        capture_attachments,
    )

    robot, manips, mid, box = pieces
    T_obj = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.identity(), jnp.array([0.3, 0.1, 0.5])
    )
    atts = capture_attachments(manips, (mid, mid), T_obj, geom=box)

    for m, a in zip(manips, atts):
        # Reconstructing through the world gripper pose must recover T_obj...
        T = _gripper_world_pose(m, mid) @ jaxlie.SE3(a.T_parent_body)
        onp.testing.assert_allclose(
            onp.asarray(T.wxyz_xyz), onp.asarray(T_obj.wxyz_xyz), atol=1e-5
        )
    # ...and the right arm (base yaw = pi, x = +0.4) must differ from the left,
    # which it would not if the base transform had been dropped.
    assert not onp.allclose(
        onp.asarray(atts[0].T_parent_body), onp.asarray(atts[1].T_parent_body), atol=1e-3
    )


def test_object_pose_world_agrees_across_manipulators_at_the_grasp(pieces):
    """Both attachments predict one object pose iff the chain is closed; at the
    capture configuration the closure residual is zero, so they must agree."""
    from pyroffi.dynamics._contact import (
        ContactSystem,
        capture_attachments,
        grasp_closure_residual,
        object_pose_world,
    )

    _, manips, mid, box = pieces
    T_obj = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_y_radians(jnp.array(0.2)), jnp.array([0.0, 0.05, 0.55])
    )
    sys = ContactSystem.from_attachments(
        manips, capture_attachments(manips, (mid, mid), T_obj, geom=box)
    )
    q = jnp.concatenate([mid, mid])
    onp.testing.assert_allclose(
        onp.asarray(grasp_closure_residual(sys, q)), 0.0, atol=1e-5
    )
    onp.testing.assert_allclose(
        onp.asarray(object_pose_world(sys, q, 0).wxyz_xyz),
        onp.asarray(object_pose_world(sys, q, 1).wxyz_xyz),
        atol=1e-5,
    )
    onp.testing.assert_allclose(
        onp.asarray(object_pose_world(sys, q, 0).wxyz_xyz),
        onp.asarray(T_obj.wxyz_xyz),
        atol=1e-5,
    )


def test_loaded_manipulator_carries_the_payload(pieces):
    """The payoff: a manipulator built from the contact system's attachment has
    the object's inertia in its own dynamics, with no payload argument."""
    from pyroffi.dynamics._contact import (
        ContactSystem,
        _gripper_world_pose,
        capture_attachments,
    )

    robot, manips, mid, box = pieces
    sys = ContactSystem.from_attachments(
        manips,
        capture_attachments(
            manips, (mid, mid), _gripper_world_pose(manips[0], mid), geom=box
        ),
    )
    loaded = sys.loaded_manipulator_robot(0)
    assert loaded.dynamics.num_dof == robot.dynamics.num_dof
    assert not onp.allclose(
        onp.asarray(loaded.dynamics.I_body), onp.asarray(robot.dynamics.I_body)
    )
    # A 3 kg box must raise the holding torque somewhere.
    z = jnp.zeros((1, robot.joints.num_actuated_joints))
    assert not onp.allclose(
        onp.asarray(loaded.inverse_dynamics(z, z, z)),
        onp.asarray(robot.inverse_dynamics(z, z, z)),
    )


def test_from_attachments_rejects_a_mismatched_grip_link(pieces):
    from pyroffi.attachments import Attachment
    from pyroffi.dynamics._contact import ContactSystem

    _, manips, mid, box = pieces
    good = Attachment.from_geom(box.broadcast_to((1,)), manips[0].grip_link_index,
                                jnp.array([1.0, 0, 0, 0, 0, 0, 0]), name="a")
    wrong = Attachment.from_geom(box.broadcast_to((1,)), 0,
                                 jnp.array([1.0, 0, 0, 0, 0, 0, 0]), name="b")
    with pytest.raises(ValueError, match="grips with link"):
        ContactSystem.from_attachments(manips, (good, wrong))
    with pytest.raises(ValueError, match="one attachment per manipulator"):
        ContactSystem.from_attachments(manips, (good,))
