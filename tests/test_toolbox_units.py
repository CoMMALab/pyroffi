"""Unit tests for the pyroffi.toolbox layers that need no robot or GPU.

Covers the pieces where a silent bug is most expensive: the padded scene's
shape-staticness, the interop contract (joint ordering, quaternion convention),
retiming feasibility, handle bookkeeping, path bucketing, and the MCP tool table
(schemas and argument validation). All CPU, all fast.

Run:
    pytest tests/test_toolbox_units.py -q
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from pyroffi.toolbox import (
    HandleTable,
    Scene,
    bucket_length,
    config_from_payload,
    joint_dict,
    pad_path,
    path_from_payload,
    retime_path,
    se3_from_payload,
)
from pyroffi.toolbox._exchange import export_scene_urdf
from pyroffi.toolbox._retiming import default_acceleration_limits

JOINTS = ("j1", "j2", "j3")


# ── scene: fixed capacity, stable shapes ─────────────────────────────────────


def test_scene_geometry_shapes_are_independent_of_object_count():
    """The whole point of the padded scene: XLA must never see a shape change."""
    scene = Scene(max_objects=8, ground_plane=True)
    shapes_empty = [g.get_batch_axes() for g in scene.world_geoms()]

    for i in range(5):
        scene.add_object(
            f"box{i}", "box", position=(i * 0.1, 0.0, 0.2),
            params={"length": 0.1, "width": 0.1, "height": 0.1},
        )
    scene.add_object("ball", "sphere", position=(1.0, 0.0, 0.5), params={"radius": 0.05})
    shapes_full = [g.get_batch_axes() for g in scene.world_geoms()]

    assert shapes_empty == shapes_full
    assert all(s == (8,) for s in shapes_full)


def test_scene_version_advances_on_every_mutation():
    scene = Scene(max_objects=4, ground_plane=False)
    v0 = scene.version
    scene.add_object("a", "sphere", params={"radius": 0.1})
    assert scene.version > v0
    v1 = scene.version
    scene.add_object("a", "sphere", position=(1, 0, 0), params={"radius": 0.1})
    assert scene.version > v1  # moving an object also invalidates prior validations
    scene.remove_object("a")
    assert scene.version > v1 + 1


def test_scene_reuses_slot_when_object_is_moved():
    scene = Scene(max_objects=4, ground_plane=False)
    slot = scene.add_object("a", "sphere", params={"radius": 0.1}).slot
    assert scene.add_object("a", "sphere", position=(1, 1, 1), params={"radius": 0.2}).slot == slot
    assert len(scene.names()) == 1


def test_scene_capacity_is_enforced_with_an_actionable_message():
    scene = Scene(max_objects=2, ground_plane=False)
    scene.add_object("a", "sphere", params={"radius": 0.1})
    scene.add_object("b", "sphere", params={"radius": 0.1})
    with pytest.raises(ValueError, match="max_objects"):
        scene.add_object("c", "sphere", params={"radius": 0.1})


def test_scene_rejects_missing_shape_params():
    scene = Scene(max_objects=2, ground_plane=False)
    with pytest.raises(ValueError, match="radius"):
        scene.add_object("a", "sphere", params={})
    with pytest.raises(ValueError, match="unknown shape"):
        scene.add_object("a", "tetrahedron", params={})


def test_parked_slots_are_far_away_so_reductions_ignore_them():
    """Padding must not manufacture clearance results.

    Parked slots have to be far enough that any real clearance wins a min().
    """
    scene = Scene(max_objects=4, ground_plane=False)
    scene.add_object("a", "sphere", position=(0.5, 0.0, 0.3), params={"radius": 0.05})
    sphere_pool = scene.world_geoms()[0]
    positions = np.asarray(sphere_pool.pose.translation())
    named = scene.slot_names("sphere")
    for slot, name in enumerate(named):
        if name is None:
            assert np.linalg.norm(positions[slot]) > 1e3
        else:
            assert np.allclose(positions[slot], [0.5, 0.0, 0.3])


def test_parked_halfspace_is_below_the_workspace():
    scene = Scene(max_objects=3, ground_plane=True)
    pool = scene.world_geoms()[3]
    # The real ground is at z=0; parked planes must be far below, not at z=0,
    # or every configuration would report a floor collision.
    offsets = np.asarray(pool.pose.translation())[:, 2]
    assert (offsets[np.asarray([n is None for n in scene.slot_names("halfspace")])] < -1e3).all()


def test_export_scene_urdf_is_wellformed_and_lists_every_object():
    scene = Scene(max_objects=4, ground_plane=True)
    scene.add_object("shelf", "box", position=(0.5, 0, 0.3),
                     params={"length": 0.4, "width": 0.1, "height": 0.6})
    urdf = export_scene_urdf(scene, "panda")
    assert urdf.startswith("<?xml")
    assert urdf.rstrip().endswith("</robot>")
    assert 'name="shelf"' in urdf and 'name="ground"' in urdf
    assert urdf.count("<link") == 3  # world + ground + shelf

    import yourdfpy  # the point of emitting URDF is that a foreign tool can load it

    import io
    yourdfpy.URDF.load(io.StringIO(urdf), load_meshes=False, build_scene_graph=False)


# ── interop contract ─────────────────────────────────────────────────────────


def test_config_from_payload_is_name_keyed_not_positional():
    """Ordering must come from names, so a reordered payload still resolves."""
    forward = config_from_payload({"j1": 1.0, "j2": 2.0, "j3": 3.0}, JOINTS)
    shuffled = config_from_payload({"j3": 3.0, "j1": 1.0, "j2": 2.0}, JOINTS)
    assert np.allclose(forward, [1.0, 2.0, 3.0])
    assert np.allclose(forward, shuffled)


def test_config_from_payload_rejects_unknown_joints():
    with pytest.raises(ValueError, match="unknown joint names"):
        config_from_payload({"j1": 1.0, "elbow": 2.0}, JOINTS)


def test_config_from_payload_requires_all_joints_without_defaults():
    with pytest.raises(ValueError, match="missing joint values"):
        config_from_payload({"j1": 1.0}, JOINTS)
    # With defaults, a partial update is a legitimate request.
    out = config_from_payload({"j2": 9.0}, JOINTS, defaults=np.zeros(3))
    assert np.allclose(out, [0.0, 9.0, 0.0])


def test_positional_config_must_be_exactly_full_length():
    assert np.allclose(config_from_payload([1.0, 2.0, 3.0], JOINTS), [1, 2, 3])
    with pytest.raises(ValueError, match="exactly 3"):
        config_from_payload([1.0, 2.0], JOINTS)


def test_joint_dict_round_trips():
    values = np.array([0.1, -0.2, 0.3])
    assert np.allclose(config_from_payload(joint_dict(values, JOINTS), JOINTS), values)


def test_path_from_payload_stacks_waypoints():
    wps = [{"j1": 0.0, "j2": 0.0, "j3": 0.0}, {"j1": 1.0, "j2": 1.0, "j3": 1.0}]
    arr = path_from_payload(wps, JOINTS)
    assert arr.shape == (2, 3)
    with pytest.raises(ValueError, match="no waypoints"):
        path_from_payload([], JOINTS)


def test_quaternion_convention_is_rejected_not_reinterpreted():
    """An xyzw payload must fail loudly: silently reinterpreting it produces a
    plausible-looking rotation error, which is the worst possible outcome."""
    with pytest.raises(ValueError, match="quaternion_convention"):
        se3_from_payload(
            {"position": [0, 0, 0], "wxyz": [0, 0, 0, 1],
             "quaternion_convention": "xyzw"}
        )


def test_se3_from_payload_normalises_and_requires_position():
    pose = se3_from_payload({"position": [1.0, 2.0, 3.0], "wxyz": [0.0, 0.0, 2.0, 0.0]})
    assert np.allclose(np.asarray(pose.translation()), [1, 2, 3], atol=1e-6)
    assert np.isclose(float(np.linalg.norm(np.asarray(pose.rotation().wxyz))), 1.0, atol=1e-6)
    with pytest.raises(ValueError, match="position"):
        se3_from_payload({"wxyz": [1, 0, 0, 0]})
    with pytest.raises(ValueError, match="zero norm"):
        se3_from_payload({"position": [0, 0, 0], "wxyz": [0, 0, 0, 0]})


# ── handles ──────────────────────────────────────────────────────────────────


def test_handle_table_insert_and_typed_lookup():
    table = HandleTable()
    cfg = table.insert("config", np.zeros(3), JOINTS, scene_version=1)
    path = table.insert("path", np.zeros((5, 3)), JOINTS, scene_version=1)
    assert cfg.handle.startswith("cfg_") and path.handle.startswith("path_")
    assert table.get(cfg.handle, "config") is cfg
    assert path.n_waypoints == 5 and cfg.n_waypoints == 1
    with pytest.raises(KeyError, match="is a path"):
        table.get(path.handle, "config")
    with pytest.raises(KeyError, match="unknown handle"):
        table.get("cfg_9999")


def test_handle_table_rejects_mismatched_joint_names():
    """Guards the invariant the whole exchange layer rests on."""
    table = HandleTable()
    with pytest.raises(ValueError, match="joint arrays are always name-keyed"):
        table.insert("config", np.zeros(5), JOINTS, scene_version=0)


def test_handle_drop_and_scene_version_recorded():
    table = HandleTable()
    entry = table.insert("path", np.zeros((2, 3)), JOINTS, scene_version=7)
    assert entry.scene_version == 7
    assert not entry.is_retimed
    table.drop(entry.handle)
    assert entry.handle not in table


# ── bucketing ────────────────────────────────────────────────────────────────


def test_bucket_length_rounds_up_to_a_compiled_shape():
    assert bucket_length(1, (16, 32, 64)) == 16
    assert bucket_length(16, (16, 32, 64)) == 16
    assert bucket_length(47, (16, 32, 64)) == 64
    # Overflowing the largest bucket is allowed but returns the true length,
    # which compiles a one-off program rather than silently truncating.
    assert bucket_length(200, (16, 32, 64)) == 200
    with pytest.raises(ValueError):
        bucket_length(0, (16,))


def test_pad_path_repeats_the_final_waypoint():
    """Padding must be validity-neutral: repeated waypoints are zero-length edges."""
    path = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    padded = pad_path(path, 5)
    assert padded.shape == (5, 3)
    assert np.allclose(padded[:2], path)
    assert np.allclose(padded[2:], path[-1])
    with pytest.raises(ValueError, match="exceeds target"):
        pad_path(path, 1)


# ── retiming ─────────────────────────────────────────────────────────────────


def _straight_path(n=20, dof=3, span=1.0):
    t = np.linspace(0.0, 1.0, n)[:, None]
    return np.tile(t, (1, dof)) * span


def test_retiming_respects_velocity_and_acceleration_limits():
    vmax = np.array([1.0, 2.0, 0.5])
    amax = np.array([2.0, 4.0, 1.0])
    res = retime_path(_straight_path(), vmax, amax, joint_names=JOINTS)

    assert res.feasible
    assert res.duration > 0.0
    assert res.peak_velocity_ratio <= 1.0 + 1e-3
    assert res.peak_acceleration_ratio <= 1.0 + 1e-3
    # Times must be strictly increasing, or downstream dt is meaningless.
    assert np.all(np.diff(res.times) > 0.0)
    assert res.dt > 0.0
    # The slowest joint sets the schedule.
    assert res.limiting_joint == "j3"


def test_retiming_starts_and_ends_at_rest():
    res = retime_path(_straight_path(), np.ones(3), np.ones(3), joint_names=JOINTS)
    assert np.allclose(res.velocities[0], 0.0)
    assert np.allclose(res.velocities[-1], 0.0)


def test_duration_is_monotone_in_the_velocity_limit():
    """Tightening a limit must never yield a faster trajectory.

    This is the property an iterative stretching scheme silently violated:
    starting closer to feasible needed fewer (and gentler) passes, so a *lower*
    speed ceiling could come back with a *shorter* duration.
    """
    path = _straight_path()
    durations = [
        retime_path(path, np.ones(3) * scale, np.ones(3) * 4).duration
        for scale in (1.0, 0.75, 0.5, 0.25, 0.1)
    ]
    assert durations == sorted(durations), durations


def test_duration_is_monotone_in_the_acceleration_limit():
    path = _straight_path()
    durations = [
        retime_path(path, np.ones(3), np.ones(3) * scale).duration
        for scale in (4.0, 2.0, 1.0, 0.5)
    ]
    assert durations == sorted(durations), durations


def test_tighter_limits_take_longer():
    path = _straight_path()
    fast = retime_path(path, np.ones(3), np.ones(3) * 10)
    slow = retime_path(path, np.ones(3) * 0.1, np.ones(3) * 10)
    assert slow.duration > fast.duration


def test_retiming_is_always_feasible_under_fuzzing():
    """Feasibility is the one guarantee the whole simulate path depends on."""
    rng = np.random.default_rng(0)
    for _ in range(100):
        n = int(rng.integers(2, 40))
        path = np.cumsum(rng.normal(size=(n, 5)) * 0.2, axis=0)
        result = retime_path(path, np.ones(5) * 1.5, np.ones(5) * 3.0)
        assert result.feasible
        assert np.all(np.diff(result.times) > 0.0)


def test_retiming_velocity_profile_matches_the_waypoint_spacing():
    """The reported velocities must actually be the derivative of the schedule."""
    path = _straight_path(n=30, span=2.0)
    res = retime_path(path, np.ones(3), np.ones(3) * 5)
    fd = np.diff(path, axis=0) / np.diff(res.times)[:, None]
    interior = np.abs(fd).max(axis=1)[1:-1]
    assert np.all(interior <= 1.0 + 1e-3)


def test_retiming_handles_degenerate_paths():
    single = retime_path(np.zeros((1, 3)), np.ones(3), np.ones(3))
    assert single.duration == 0.0 and single.feasible
    # A stationary two-waypoint path must not divide by zero.
    stationary = retime_path(np.zeros((2, 3)), np.ones(3), np.ones(3))
    assert np.all(np.isfinite(stationary.times))


def test_retiming_treats_absent_limits_as_unbounded_not_as_zero():
    """A zero limit in a URDF would otherwise make every ratio infinite."""
    res = retime_path(_straight_path(), np.array([1.0, 0.0, 1.0]), np.ones(3) * 5)
    assert res.feasible and np.all(np.isfinite(res.times))


def test_default_acceleration_limits_scale_with_time_to_peak():
    v = np.array([2.0, 4.0])
    assert np.allclose(default_acceleration_limits(v, 0.5), [4.0, 8.0])
    assert np.allclose(default_acceleration_limits(v, 1.0), [2.0, 4.0])


# ── MCP tool table ───────────────────────────────────────────────────────────


def test_every_tool_has_a_valid_schema_and_a_real_handler():
    from pyroffi.mcp import TOOLS
    from pyroffi.toolbox import Toolbox

    assert len(TOOLS) >= 20
    for spec in TOOLS:
        assert spec.description and len(spec.description) > 40, spec.name
        assert spec.input_schema["type"] == "object", spec.name
        for field in spec.input_schema.get("required", []):
            assert field in spec.input_schema["properties"], (spec.name, field)
        # Every tool must map to something that actually exists, or the failure
        # only shows up when an agent calls it.
        assert spec.method == "recreate_session" or hasattr(Toolbox, spec.method), spec.name


def test_tool_names_are_unique():
    from pyroffi.mcp import TOOLS

    names = [t.name for t in TOOLS]
    assert len(names) == len(set(names))


def test_tool_payloads_are_json_serialisable_mcp_tools():
    import json

    import mcp.types as types

    from pyroffi.mcp import list_tool_payloads

    payloads = list_tool_payloads()
    json.dumps(payloads)
    for payload in payloads:
        types.Tool(**payload)  # would raise if the schema shape is wrong


def test_dispatch_validates_arguments_without_touching_a_session():
    """Argument errors must be caught before any GPU work happens."""
    from pyroffi.mcp import dispatch

    class Spy:
        def __init__(self):
            self.calls = []

        def check_collision(self, **kwargs):
            self.calls.append(kwargs)
            return {"ok": True}

    spy = Spy()
    assert dispatch(spy, "check_collision", {"config": "cfg_1"}) == {"ok": True}
    assert spy.calls == [{"config": "cfg_1"}]

    with pytest.raises(ValueError, match="unknown tool"):
        dispatch(spy, "teleport", {})
    with pytest.raises(ValueError, match="unexpected argument"):
        dispatch(spy, "check_collision", {"config": "cfg_1", "speed": 3})
    with pytest.raises(ValueError, match="missing required argument"):
        dispatch(spy, "check_collision", {})


def test_cost_hints_are_present_so_the_model_can_self_order():
    """The cheap-before-expensive story is carried entirely by descriptions."""
    from pyroffi.mcp import TOOLS_BY_NAME

    for name in ("check_edge", "check_collision", "solve_ik", "validate_path"):
        assert "ms" in TOOLS_BY_NAME[name].description
    for name in ("list_objects", "add_object", "import_path"):
        assert "FREE" in TOOLS_BY_NAME[name].description
    # The two honesty-critical descriptions.
    assert "compiled" in TOOLS_BY_NAME["optimize_path"].description
    assert "not a planner" in TOOLS_BY_NAME["optimize_between"].description
    assert "EXPENSIVE in context" in TOOLS_BY_NAME["export_path"].description


def test_server_error_payloads_are_structured_not_exceptions():
    """An agent can act on {"error": ...}; a crash just loses the warm session."""
    from pyroffi.mcp._server import PyroffiServer

    class Spy:
        def check_collision(self, **kwargs):
            raise RuntimeError("boom")

    server = PyroffiServer.__new__(PyroffiServer)  # no session needed for this path
    server.__dict__["toolbox"] = Spy()

    # A bad argument is caught before the primitive runs.
    bad_args = server.call("check_collision", {})
    assert bad_args["success"] is False
    assert bad_args["error"] == "ValueError"
    assert bad_args["tool"] == "check_collision"

    # A primitive that blows up is also reported rather than killing the server.
    blew_up = server.call("check_collision", {"config": "cfg_1"})
    assert blew_up["success"] is False
    assert blew_up["error"] == "RuntimeError"
    assert "boom" in blew_up["message"]
