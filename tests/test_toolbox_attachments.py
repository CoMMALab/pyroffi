"""Toolbox / MCP / viewer surface for attached bodies.

The behaviour worth pinning here is the *bookkeeping*, not the geometry (which
``test_attachments.py`` covers): an object must be either a world obstacle or
part of the robot, never both and never neither, and the round trip has to be
exact -- an object that drifts each time it is picked up and put down would make
a multi-step plan quietly diverge.

Run:
    CUDA_VISIBLE_DEVICES=<free gpu> pytest tests/test_toolbox_attachments.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from pyroffi.toolbox import Session, Toolbox

ROBOT = "panda_spherized"
CUBE = {
    "name": "cube",
    "shape": "box",
    "position": [0.45, 0.0, 0.45],
    "params": {"length": 0.06, "width": 0.06, "height": 0.06},
}


@pytest.fixture()
def tb() -> Toolbox:
    session = Session(
        robot=ROBOT, max_objects=8, n_timesteps=16, calibrate_self_collision=False
    )
    return Toolbox(session)


def _add_cube(tb: Toolbox) -> None:
    tb.add_object(
        CUBE["name"], CUBE["shape"], position=CUBE["position"], params=CUBE["params"]
    )


def test_attach_moves_the_object_from_the_world_onto_the_robot(tb):
    _add_cube(tb)
    s = tb.session
    n_before = s.robot_coll.num_links
    assert "cube" in s.scene.names()

    r = tb.attach_object("cube")
    assert r["attached"] == "cube"
    # Exactly one of the two places, never both: an object that stayed in the
    # world pool while also riding the gripper would collide with itself.
    assert "cube" not in s.scene.names()
    assert [a["name"] for a in s.attached()] == ["cube"]
    assert s.robot_coll.num_links == n_before + 1
    assert s.robot_coll.num_robot_links == n_before


def test_attach_detach_round_trips_the_pose_exactly(tb):
    _add_cube(tb)
    tb.attach_object("cube")
    r = tb.detach_object("cube")

    np.testing.assert_allclose(r["position"], CUBE["position"], atol=1e-4)
    np.testing.assert_allclose(r["wxyz"], [1.0, 0.0, 0.0, 0.0], atol=1e-4)
    assert "cube" in tb.session.scene.names()
    assert tb.session.attached() == []
    # Shape and params survive the trip, not just the pose.
    obj = tb.session.scene.get_object("cube")
    assert obj.shape == "box"
    assert obj.params == CUBE["params"]


def test_detached_object_follows_the_arm_before_release(tb):
    """Carrying it and *then* releasing must drop it where the arm now is."""
    _add_cube(tb)
    tb.attach_object("cube")
    tb.session.robot_state = np.asarray(tb.session.robot_state) + 0.25
    r = tb.detach_object("cube")
    assert not np.allclose(r["position"], CUBE["position"], atol=1e-3)


def test_attached_object_moves_with_the_gripper(tb):
    """The collision geometry must actually track the link, not sit still."""
    _add_cube(tb)
    tb.attach_object("cube")
    s = tb.session
    q0 = s.as_array(s.robot_state)
    q1 = q0 + 0.3
    p0 = np.asarray(s.robot_coll.at_config(s.robot, q0).pose.translation())[-1]
    p1 = np.asarray(s.robot_coll.at_config(s.robot, q1).pose.translation())[-1]
    assert np.linalg.norm(p0 - p1) > 1e-3


def test_ignored_links_keep_the_gripper_from_colliding_with_what_it_holds(tb):
    _add_cube(tb)
    fingers = [
        n for n in ("panda_leftfinger", "panda_rightfinger") if n in tb.session.link_names
    ]
    tb.attach_object("cube", ignore_links=tuple(fingers))
    n_ignored = len(tb.session.robot_coll.active_idx_i)

    tb.detach_object("cube")
    tb.attach_object("cube")
    n_plain = len(tb.session.robot_coll.active_idx_i)
    assert n_ignored == n_plain - len(fingers)


def test_double_attach_and_unknown_detach_fail_loudly(tb):
    _add_cube(tb)
    tb.attach_object("cube")
    with pytest.raises(ValueError, match="already attached"):
        tb.attach_object("cube")
    with pytest.raises(KeyError, match="not attached"):
        tb.detach_object("nope")


def test_a_halfspace_cannot_be_attached(tb):
    """A half-space is unbounded, so there is no conservative bounding sphere."""
    with pytest.raises(ValueError, match="cannot be attached"):
        tb.attach_object("ground")


def test_dynamics_picks_up_the_payload_too(tb):
    """Attaching is not collision-only: the session's robot carries the load."""
    _add_cube(tb)
    s = tb.session
    if s.robot.dynamics is None:
        pytest.skip("URDF has no inertial data")
    before = np.asarray(s.robot.dynamics.I_body).copy()
    # Collision-only by default: scene objects carry no mass.
    tb.attach_object("cube")
    np.testing.assert_array_equal(np.asarray(s.robot.dynamics.I_body), before)
    tb.detach_object("cube")
    # With a mass, the payload reaches the dynamics too.
    tb.attach_object("cube", mass=1.5)
    assert not np.allclose(np.asarray(s.robot.dynamics.I_body), before)
    tb.detach_object("cube")
    np.testing.assert_array_equal(np.asarray(s.robot.dynamics.I_body), before)


# --- MCP surface -----------------------------------------------------------


def test_mcp_dispatches_the_attachment_tools(tb):
    from pyroffi.mcp._tools import TOOLS_BY_NAME, dispatch

    for name in ("attach_object", "detach_object", "list_attachments"):
        assert name in TOOLS_BY_NAME

    dispatch(
        tb,
        "add_object",
        {"name": "cube", "shape": "box", "position": CUBE["position"],
         "params": CUBE["params"]},
    )
    r = dispatch(tb, "attach_object", {"name": "cube"})
    assert r["attached"] == "cube"
    listed = dispatch(tb, "list_attachments", {})
    assert [a["name"] for a in listed["attachments"]] == ["cube"]
    assert listed["n_attached"] == 1

    d = dispatch(tb, "detach_object", {"name": "cube"})
    assert d["detached"] == "cube"
    assert dispatch(tb, "list_attachments", {})["n_attached"] == 0


def test_reset_scene_hands_back_an_empty_session_that_is_still_warm(tb):
    """The between-problems operation for a server that outlives the problem.

    What matters is the pair: everything the problem put there is gone, and the
    session it was put into is the same one -- a reset that recreated the
    session would be correct and useless, since the next problem would pay the
    cold-start compile.
    """
    s = tb.session
    session_id = id(s)
    _add_cube(tb)
    tb.add_object("shelf", "box", position=[0.6, 0.0, 0.2],
                  params={"length": 0.1, "width": 0.4, "height": 0.02})
    tb.attach_object("cube")
    tb.import_path([dict(zip(s.joint_names, s.robot.default_cfg))] * 4)
    s.robot_state = np.asarray(s.robot.default_cfg, dtype=np.float64) + 0.1

    r = tb.reset_scene()

    assert r["detached"] == ["cube"]
    # The cube is detached back into the world first and then removed with
    # everything else, so it shows up in both lists.
    assert sorted(r["removed_objects"]) == ["cube", "shelf"]
    assert s.scene.names() == ["ground"]   # the floor is the world's, not the problem's
    assert s.attached() == []
    assert len(s.handles) == 0
    assert np.allclose(s.robot_state, s.robot.default_cfg)
    assert id(tb.session) == session_id

    # And the reset does not leave the toolbox in a state where the next problem
    # trips over the last one's handles.
    tb.add_object("cube", "box", position=CUBE["position"], params=CUBE["params"])
    fresh = tb.check_collision(dict(zip(s.joint_names, s.robot.default_cfg)))
    assert fresh["success"] is True


def test_reset_scene_can_drop_the_ground_plane_when_asked(tb):
    _add_cube(tb)
    r = tb.reset_scene(keep_ground_plane=False)
    assert "ground" in r["removed_objects"]
    assert tb.session.scene.names() == []


def test_mcp_dispatches_reset_scene(tb):
    from pyroffi.mcp._tools import TOOLS_BY_NAME, dispatch

    assert "reset_scene" in TOOLS_BY_NAME
    _add_cube(tb)
    r = dispatch(tb, "reset_scene", {})
    assert r["removed_objects"] == ["cube"]
    assert r["handles_invalidated"] is True


def test_mcp_tool_schemas_all_map_to_real_methods():
    from pyroffi.mcp._tools import TOOLS
    from pyroffi.toolbox import Toolbox as _TB

    for spec in TOOLS:
        if spec.method == "recreate_session":
            continue  # handled by the server, not the toolbox
        assert hasattr(_TB, spec.method), f"{spec.name} -> missing {spec.method}"


# --- viewer ----------------------------------------------------------------


def test_viewer_keeps_drawing_a_carried_object(tb):
    """An attached object leaves the scene pool, so without explicit support it
    would vanish from the render exactly when you want to watch it."""
    from pyroffi.viewer._world import ToolboxSource

    _add_cube(tb)
    tb.attach_object("cube")
    src = ToolboxSource(tb.session)

    assert "cube" in [o.name for o in src.describe().objects]
    state = src.read()
    assert "cube" in state.object_poses
    assert state.extras["attached"] == ["cube"]

    p0 = np.asarray(state.object_poses["cube"].position)
    src.set_config(np.asarray(tb.session.robot_state) + 0.3)
    p1 = np.asarray(src.read().object_poses["cube"].position)
    assert np.linalg.norm(p0 - p1) > 1e-3


def test_viewer_colours_carried_objects_distinctly(tb):
    from pyroffi.viewer._world import ToolboxSource

    _add_cube(tb)
    tb.add_object("other", "box", position=[0.0, 0.5, 0.2],
                  params={"length": 0.1, "width": 0.1, "height": 0.1})
    tb.attach_object("cube")
    by_name = {o.name: o for o in ToolboxSource(tb.session).describe().objects}
    assert by_name["cube"].color != by_name["other"].color


# ── collision reporting with something attached ──────────────────────────────
#
# Attaching appends a row to the collision arrays, so every array whose length
# was "one per link" is now one short and every index-to-name table is one
# short. Nothing above exercised a *reporting* primitive while holding
# something, which is how a shape mismatch that breaks all of validate_path
# went unnoticed.


def test_collision_primitives_work_while_holding_something(tb):
    """check_collision / check_edge / validate_path must survive an attachment.

    Regression: ``world_link_mask`` was built over ``link_names`` while the
    collision model had grown an attachment row, so every one of these raised
    a broadcast error the moment the robot picked anything up.
    """
    _add_cube(tb)
    tb.attach_object(CUBE["name"])
    s = tb.session
    q = {n: float(v) for n, v in zip(s.joint_names, s.robot_state)}

    assert tb.check_collision(q)["success"]
    assert tb.check_edge(q, q)["success"]
    assert tb.validate_path([q, q])["success"]


def test_a_carried_object_is_reported_by_name(tb):
    """The payload's own collisions must come back named, not as an index error.

    This is the whole point of attaching: ``validate_path`` checks the robot,
    and an attached object is now part of the robot, so a carried cube driven
    into the floor has to be reported as the cube.
    """
    tb.add_object(
        "cube", "box", position=[0.45, 0.0, 0.02], params=CUBE["params"]
    )
    s = tb.session
    # Put the gripper on the cube so attaching grabs it where it sits, low
    # enough that its conservative bounding sphere intersects the ground.
    res = tb.solve_ik(
        pose={"position": [0.45, 0.0, 0.02 + 0.095], "wxyz": [0, 0, 1, 0]},
        num_seeds=64,
        num_restarts=3,
    )
    tb.set_robot_state(res["config_id"])
    tb.attach_object("cube")

    q = {n: float(v) for n, v in zip(s.joint_names, s.handles.get(res["config_id"]).values)}
    report = tb.check_collision(q)
    named = {c["link"] for c in report["world_collisions"]}
    assert "cube" in named, report["world_collisions"]

    # And the row names stay consistent with the pair table it indexes into.
    assert len(s.collision_row_names()) == len(s.link_names) + 1
    assert s.self_pair_names()          # must not raise on attachment rows


def test_ignore_objects_mutes_only_the_named_surface(tb):
    """A carried object may be told to overlap the surface it came from.

    The bounding sphere is a strict over-approximation -- radius 0.052 for the
    6 cm cube here, against a 0.03 half-height -- so a block still resting on
    the table intersects it by construction. Without a way to say so, the
    lift-off that resolves it validates as invalid at its first waypoint and an
    agent cannot tell that apart from a real fault.
    """
    tb.add_object("cube", "box", position=[0.45, 0.0, 0.02], params=CUBE["params"])
    tb.add_object("wall", "box", position=[0.45, 0.0, 0.02],
                  params={"length": 0.05, "width": 0.05, "height": 0.05})
    s = tb.session
    res = tb.solve_ik(
        pose={"position": [0.45, 0.0, 0.02 + 0.095], "wxyz": [0, 0, 1, 0]},
        num_seeds=64, num_restarts=3,
    )
    tb.set_robot_state(res["config_id"])
    q = {n: float(v) for n, v in zip(s.joint_names, s.handles.get(res["config_id"]).values)}

    tb.attach_object("cube", ignore_objects=["ground"])
    hits = {(c["link"], c["object"]) for c in tb.check_collision(q)["world_collisions"]}
    assert ("cube", "ground") not in hits
    # Muting is per (row, object): the cube's other overlaps still report, and
    # so do the robot's own collisions with the ignored surface.
    assert ("cube", "wall") in hits, hits


def test_ignore_objects_rejects_a_name_that_is_not_in_the_scene(tb):
    """A typo must not silently present as the report it was meant to mute."""
    _add_cube(tb)
    with pytest.raises(KeyError, match="no such scene object"):
        tb.attach_object("cube", ignore_objects=["tabel"])
