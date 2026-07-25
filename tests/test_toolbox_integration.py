"""End-to-end tests for the pyroffi toolbox against a real robot and GPU.

These exercise the behaviour that unit tests cannot: that IK actually reaches
its target, that the collision model gives *actionable* answers, that trajopt
repairs a seed that goes through an obstacle, that retiming plus simulation
produce a physically sensible rollout, and that the scene stays shape-static
across mutation (no recompiles).

The session build (URDF parse + collision calibration) and the first trajopt
compile are slow, so the session is module-scoped and warmed once.

Run:
    CUDA_VISIBLE_DEVICES=<free gpu> pytest tests/test_toolbox_integration.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from pyroffi.toolbox import Session, Toolbox, configure_process

configure_process(x64=True)

ROBOT = "panda_spherized"
"""Primitive collision geometry, so self-collision results are meaningful.
The mesh-based `panda` is covered separately by the fallback test."""

# Two reachable poses on either side of the wall obstacle, hand pointing down.
DOWN = [0.0, 0.0, 1.0, 0.0]
POSE_A = {"position": [0.5, -0.35, 0.35], "wxyz": DOWN}
POSE_B = {"position": [0.5, 0.35, 0.35], "wxyz": DOWN}


@pytest.fixture(scope="module")
def tb() -> Toolbox:
    session = Session(robot=ROBOT, max_objects=8, n_timesteps=32)
    toolbox = Toolbox(session)
    toolbox.add_object(
        "wall",
        "box",
        position=(0.5, 0.0, 0.2),
        params={"length": 0.1, "width": 0.4, "height": 0.4},
    )
    return toolbox


@pytest.fixture(scope="module")
def endpoints(tb: Toolbox) -> tuple[str, str]:
    a = tb.solve_ik(pose=POSE_A, collision_free=True, num_seeds=64)
    b = tb.solve_ik(pose=POSE_B, collision_free=True, num_seeds=64, seed=1)
    assert a["success"] and b["success"], (a, b)
    return a["config_id"], b["config_id"]


@pytest.fixture(scope="module")
def blocked_path(tb: Toolbox, endpoints) -> str:
    """A straight line between the endpoints, which drives through the wall."""
    s = tb.session
    qa = s.handles.get(endpoints[0]).values
    qb = s.handles.get(endpoints[1]).values
    q = np.linspace(qa, qb, 24)
    return tb.import_path([dict(zip(s.joint_names, row)) for row in q])["path_id"]


# ── session and capabilities ─────────────────────────────────────────────────


def test_capabilities_report_the_contract(tb: Toolbox):
    caps = tb.create_scene_info()["capabilities"]
    assert caps["dof"] == len(caps["joint_names"]) == 7
    assert caps["quaternion_convention"] == "wxyz"
    assert caps["units"] == {"length": "m", "angle": "rad", "time": "s"}
    assert caps["x64"] is True, "x64 must be reported honestly; solvers are sensitive"
    assert caps["ee_link"] in caps["link_names"]
    assert set(caps["joint_limits"]) == set(caps["joint_names"])


def test_self_collision_calibration_makes_the_model_usable(tb: Toolbox):
    """Without pruning structurally-overlapping pairs, every configuration reports
    a self-collision and the validation tools are worthless."""
    report = tb.session.self_collision_report
    assert report["calibrated"]
    assert report["n_pairs_after"] <= report["n_pairs_before"]
    assert report["reliable"], report
    assert report["frac_random_configs_self_colliding"] < 0.5


def test_static_links_are_excluded_from_world_collision(tb: Toolbox):
    """The bolted-down base intersects the floor plane by construction; reporting
    it would make collision_free false for every configuration forever."""
    assert "panda_link0" in tb.session.static_link_names
    mid = (tb.session.lower_limits + tb.session.upper_limits) / 2
    result = tb.check_collision(mid)
    assert result["collision_free"], result
    assert result["min_clearance_m"] > 0.0


# ── scene ────────────────────────────────────────────────────────────────────


def test_scene_mutation_does_not_recompile(tb: Toolbox):
    """The padded-scene design exists to make this true. If it regresses,
    production looks like random multi-second stalls."""
    mid = (tb.session.lower_limits + tb.session.upper_limits) / 2
    tb.check_collision(mid)  # ensure this shape is compiled

    tb.add_object("ball", "sphere", position=(0.9, 0.9, 0.9), params={"radius": 0.05})
    after_add = tb.check_collision(mid)
    tb.add_object("ball", "sphere", position=(0.2, 0.2, 0.9), params={"radius": 0.05})
    after_move = tb.check_collision(mid)
    tb.remove_object("ball")
    after_remove = tb.check_collision(mid)

    assert not after_add["compiled"]
    assert not after_move["compiled"]
    assert not after_remove["compiled"]


def test_scene_version_advances_and_export_round_trips(tb: Toolbox):
    listing = tb.list_objects()
    assert "wall" in [o["name"] for o in listing["objects"]]

    primitives = tb.export_scene("primitives")
    assert primitives["scene_version"] == listing["scene_version"]
    assert primitives["quaternion_convention"] == "wxyz"

    urdf = tb.export_scene("urdf")
    assert "<robot" in urdf["urdf"] and "wall" in urdf["urdf"]


def test_object_over_the_base_is_flagged_rather_than_silently_ignored(tb: Toolbox):
    """Static-link collisions are excluded from validation, so the only chance to
    tell the caller is at add time."""
    result = tb.add_object(
        "on_base", "sphere", position=(0.0, 0.0, 0.02), params={"radius": 0.12}
    )
    assert result["intersects_static_links"], result
    assert "warning" in result
    tb.remove_object("on_base")


# ── kinematics ───────────────────────────────────────────────────────────────


def test_solve_ik_reaches_the_target(tb: Toolbox):
    result = tb.solve_ik(pose=POSE_A, num_seeds=64)
    assert result["success"], result
    assert result["pos_error_m"] < 1e-3
    assert result["rot_error_rad"] < 1e-2

    fk = tb.forward_kinematics(result["config_id"], links=[tb.session.ee_link])
    reached = fk["poses"][tb.session.ee_link]["position"]
    assert np.allclose(reached, POSE_A["position"], atol=1e-3)


def test_solve_ik_reports_a_distribution_not_just_a_winner(tb: Toolbox):
    result = tb.solve_ik(pose=POSE_A, num_seeds=32, num_restarts=4)
    assert result["num_restarts"] == 4
    assert 0 <= result["restarts_converged"] <= 4
    assert result["pos_error_worst_m"] >= result["pos_error_m"]


def test_solve_ik_honours_the_requested_solver(tb: Toolbox):
    """A primitive that silently substitutes a solver makes the agent's model of
    the world wrong."""
    for solver in ("hjcd", "ls"):
        result = tb.solve_ik(pose=POSE_A, solver=solver, num_seeds=64)
        assert result["solver"] == solver
        assert result["pos_error_m"] < 1e-2
    with pytest.raises(ValueError, match="unknown solver"):
        tb.solve_ik(pose=POSE_A, solver="magic")


def test_collision_free_ik_returns_a_clear_configuration(tb: Toolbox):
    result = tb.solve_ik(pose=POSE_A, collision_free=True, num_seeds=64)
    assert result["success"], result
    assert result["in_collision"] is False
    assert result["min_clearance_m"] > 0.0


def test_solve_ik_batch_dispatches_all_targets_at_once(tb: Toolbox):
    unreachable = {"position": [2.0, 2.0, 2.0], "wxyz": DOWN}
    result = tb.solve_ik_batch([POSE_A, POSE_B, unreachable], num_seeds=64)

    assert result["n_targets"] == 3
    assert result["n_padded"] >= 3  # bucketed so target count doesn't recompile
    assert len(result["results"]) == 3
    # Partial success is the honest answer: two reachable, one not.
    assert result["results"][0]["converged"] and result["results"][1]["converged"]
    assert not result["results"][2]["converged"]
    assert result["n_converged"] == 2


def test_solve_ik_batch_is_bucketed_so_target_count_does_not_recompile(tb: Toolbox):
    """Target count is a static shape. Without bucketing, an agent enumerating a
    varying number of grasps would pay a fresh compile on nearly every call."""
    first = tb.solve_ik_batch([POSE_A, POSE_B], num_seeds=32)
    again = tb.solve_ik_batch([POSE_A, POSE_B, POSE_A], num_seeds=32)
    assert first["n_padded"] == again["n_padded"] == 8
    assert not again["compiled"], "3 targets must reuse the 8-wide compiled program"


def test_check_reachable_prunes_without_leaving_a_handle(tb: Toolbox):
    n_before = len(tb.session.handles)
    good = tb.check_reachable(pose=POSE_A, num_seeds=64)
    bad = tb.check_reachable(pose={"position": [2.0, 2.0, 2.0], "wxyz": DOWN}, num_seeds=64)
    assert good["reachable"] and not bad["reachable"]
    assert len(tb.session.handles) == n_before


def test_fixed_joints_are_held(tb: Toolbox):
    s = tb.session
    result = tb.solve_ik(pose=POSE_A, fixed_joints=["panda_joint1"], num_seeds=64)
    cfg = s.handles.get(result["config_id"]).values
    assert np.isclose(cfg[s.joint_names.index("panda_joint1")],
                      s.robot_state[s.joint_names.index("panda_joint1")], atol=1e-6)
    with pytest.raises(ValueError, match="unknown joints"):
        tb.solve_ik(pose=POSE_A, fixed_joints=["elbow"])


# ── validation ───────────────────────────────────────────────────────────────


def test_check_collision_names_the_colliding_pair(tb: Toolbox):
    """A boolean is not actionable; the agent needs to know what it hit."""
    tb.add_object("blocker", "box", position=(0.3, 0.0, 0.5),
                  params={"length": 0.5, "width": 0.5, "height": 0.5})
    try:
        mid = (tb.session.lower_limits + tb.session.upper_limits) / 2
        result = tb.check_collision(mid)
        assert not result["collision_free"]
        assert result["world_collisions"], result
        hit = result["world_collisions"][0]
        assert hit["object"] == "blocker"
        assert hit["link"] in tb.session.link_names
        assert hit["distance_m"] < 0.0
    finally:
        tb.remove_object("blocker")


def test_check_edge_finds_where_the_motion_first_fails(tb: Toolbox, endpoints):
    result = tb.check_edge(endpoints[0], endpoints[1])
    assert not result["valid"], "the straight line runs through the wall"
    failure = result["first_failure"]
    assert 0.0 <= failure["fraction"] <= 1.0
    assert failure["world_collisions"][0]["object"] == "wall"


def test_check_edge_passes_for_a_clear_motion(tb: Toolbox, endpoints):
    result = tb.check_edge(endpoints[0], endpoints[0])
    assert result["valid"] and result["first_failure"] is None
    assert result["joint_distance_rad"] == 0.0


def test_validate_path_reports_per_waypoint_and_per_edge(tb: Toolbox, blocked_path):
    result = tb.validate_path(blocked_path)
    assert not result["valid"]
    assert result["n_invalid_waypoints"] > 0
    assert result["n_invalid_edges"] > 0
    assert result["min_clearance_m"] < 0.0
    assert result["n_waypoints"] == 24
    assert result["n_padded"] >= 24  # bucketed


def test_validate_path_detects_a_stale_scene(tb: Toolbox, blocked_path):
    """A path validated against obstacles that have since moved is exactly the
    silent failure the scene_version exists to catch."""
    fresh = tb.validate_path(blocked_path)
    assert fresh["stale_scene"] is None

    tb.add_object("temp", "sphere", position=(3.0, 3.0, 3.0), params={"radius": 0.01})
    try:
        stale = tb.validate_path(blocked_path)
        assert stale["stale_scene"] is not None
        assert stale["stale_scene"]["current_scene_version"] > stale["stale_scene"][
            "path_scene_version"
        ]
    finally:
        tb.remove_object("temp")


# ── exchange ─────────────────────────────────────────────────────────────────


def test_import_export_path_round_trips_by_name(tb: Toolbox, blocked_path):
    exported = tb.export_path(blocked_path)
    assert exported["joint_names"] == list(tb.session.joint_names)
    assert len(exported["waypoints"]) == 24
    assert set(exported["waypoints"][0]) == set(tb.session.joint_names)

    reimported = tb.import_path(exported["waypoints"], source="round_trip")
    original = tb.session.handles.get(blocked_path).values
    restored = tb.session.handles.get(reimported["path_id"]).values
    assert np.allclose(original, restored)


def test_import_path_rejects_foreign_joint_names(tb: Toolbox):
    with pytest.raises(ValueError, match="unknown joint names"):
        tb.import_path([{"shoulder_pan_joint": 0.0}])


def test_exported_waypoint_order_is_irrelevant(tb: Toolbox, blocked_path):
    """Name-keying must make a reordered payload identical, or two servers with
    different internal orderings silently disagree."""
    exported = tb.export_path(blocked_path)["waypoints"]
    reversed_keys = [dict(reversed(list(wp.items()))) for wp in exported]
    a = tb.import_path(exported)["path_id"]
    b = tb.import_path(reversed_keys)["path_id"]
    assert np.allclose(
        tb.session.handles.get(a).values, tb.session.handles.get(b).values
    )


# ── optimization ─────────────────────────────────────────────────────────────


def test_optimize_path_repairs_a_seed_that_goes_through_an_obstacle(
    tb: Toolbox, blocked_path
):
    """The headline pipeline: a foreign planner's path comes in, trajopt fixes it,
    validation confirms it."""
    before = tb.validate_path(blocked_path)
    assert not before["valid"]

    result = tb.optimize_path(blocked_path, n_batch=16)
    assert result["min_clearance_m"] > result["min_clearance_m_before"]

    after = tb.validate_path(result["path_id"])
    assert after["valid"], after
    assert after["min_clearance_m"] > 0.0


def test_optimize_path_reports_compilation_honestly(tb: Toolbox, blocked_path):
    """`compiled` is what lets an agent tell a 40 s answer from a 10 ms one."""
    first = tb.optimize_path(blocked_path, n_batch=16)
    second = tb.optimize_path(blocked_path, n_batch=16)
    assert not second["compiled"]
    assert second["solve_ms"] < first["solve_ms"]


def test_optimize_path_resamples_to_a_bucket(tb: Toolbox, endpoints):
    s = tb.session
    qa = s.handles.get(endpoints[0]).values
    qb = s.handles.get(endpoints[1]).values
    odd = np.linspace(qa, qb, 19)
    handle = tb.import_path([dict(zip(s.joint_names, r)) for r in odd])["path_id"]

    result = tb.optimize_path(handle, n_batch=16)
    assert result["n_waypoints_in"] == 19
    assert result["n_waypoints"] in s.path_buckets


def test_concat_paths_rejects_discontinuous_segments(tb: Toolbox, blocked_path):
    result = tb.concat_paths([blocked_path, blocked_path])
    assert not result["success"]
    assert result["discontinuities"][0]["worst_joint"] in tb.session.joint_names


def test_concat_paths_joins_continuous_segments(tb: Toolbox, endpoints):
    s = tb.session
    qa = s.handles.get(endpoints[0]).values
    qb = s.handles.get(endpoints[1]).values
    mid = (qa + qb) / 2
    first = tb.import_path([dict(zip(s.joint_names, r)) for r in np.linspace(qa, mid, 8)])
    second = tb.import_path([dict(zip(s.joint_names, r)) for r in np.linspace(mid, qb, 8)])

    joined = tb.concat_paths([first["path_id"], second["path_id"]])
    assert joined["success"], joined
    # The shared junction waypoint must not be duplicated.
    assert joined["n_waypoints"] == 15
    values = s.handles.get(joined["path_id"]).values
    assert np.allclose(values[0], qa) and np.allclose(values[-1], qb)


# ── retiming and simulation ──────────────────────────────────────────────────


def test_retime_produces_a_feasible_schedule(tb: Toolbox, blocked_path):
    result = tb.retime(blocked_path)
    assert result["success"] and result["feasible"]
    assert result["duration_s"] > 0.0
    assert result["peak_velocity_ratio"] <= 1.001
    assert result["peak_acceleration_ratio"] <= 1.001
    assert result["limiting_joint"] in tb.session.joint_names


def test_retime_duration_is_monotone_in_the_velocity_scale(tb: Toolbox, blocked_path):
    """Tightening the speed ceiling must never produce a faster trajectory.

    Equality is allowed: with the URDF's acceleration limits the acceleration
    bound often binds first, so moderate velocity scaling changes nothing until
    the velocity bound takes over.
    """
    durations = [
        tb.retime(blocked_path, velocity_scale=scale)["duration_s"]
        for scale in (1.0, 0.5, 0.25, 0.1, 0.05)
    ]
    assert durations == sorted(durations), durations
    assert durations[-1] > durations[0], "a 20x slower limit must eventually bind"


def test_simulate_requires_a_retimed_trajectory(tb: Toolbox, blocked_path):
    """A rollout needs dt; inventing one would make every reported number a lie."""
    with pytest.raises(ValueError, match="retime"):
        tb.simulate(blocked_path)


def test_simulate_tracks_a_retimed_trajectory(tb: Toolbox, endpoints):
    s = tb.session
    qa = s.handles.get(endpoints[0]).values
    qb = s.handles.get(endpoints[1]).values
    path = tb.import_path(
        [dict(zip(s.joint_names, r)) for r in np.linspace(qa, qb, 24)]
    )["path_id"]
    timed = tb.retime(path)

    result = tb.simulate(timed["trajectory_id"])
    assert result["success"] and not result["diverged"], result
    assert result["max_tracking_error_rad"] < 0.1
    assert result["peak_torque_joint"] in s.joint_names
    assert 0.0 < result["peak_torque_nm"] < 200.0
    assert result["final_ee_pose"]["quaternion_convention"] == "wxyz"


def test_simulate_derives_a_stable_control_rate(tb: Toolbox, endpoints):
    """The controller must run faster than the waypoint rate: torque is held
    constant across Robot.step's substeps, so closing the loop per waypoint is a
    sampled-data instability, not a trajectory problem."""
    s = tb.session
    qa = s.handles.get(endpoints[0]).values
    qb = s.handles.get(endpoints[1]).values
    path = tb.import_path(
        [dict(zip(s.joint_names, r)) for r in np.linspace(qa, qb, 24)]
    )["path_id"]
    timed = tb.retime(path)

    auto = tb.simulate(timed["trajectory_id"])
    assert auto["control_rate"]["auto"]
    assert auto["dt_control_s"] < auto["dt_waypoint_s"]
    assert auto["dt_control_s"] <= auto["control_rate"]["dt_stability_limit_s"]
    assert not auto["diverged"]

    # Forcing the loop to close at the waypoint rate is what used to diverge.
    forced = tb.simulate(timed["trajectory_id"], substeps=1)
    assert forced["control_substeps"] == 1


def test_simulate_feedforward_improves_tracking(tb: Toolbox, endpoints):
    s = tb.session
    qa = s.handles.get(endpoints[0]).values
    qb = s.handles.get(endpoints[1]).values
    path = tb.import_path(
        [dict(zip(s.joint_names, r)) for r in np.linspace(qa, qb, 24)]
    )["path_id"]
    timed = tb.retime(path)

    with_ff = tb.simulate(timed["trajectory_id"], substeps=16, feedforward=True)
    without = tb.simulate(timed["trajectory_id"], substeps=16, feedforward=False)
    assert with_ff["max_tracking_error_rad"] < without["max_tracking_error_rad"]


# ── failure reporting ────────────────────────────────────────────────────────


def test_explain_failure_gives_a_structured_cause(tb: Toolbox):
    failed = tb.solve_ik(pose={"position": [3.0, 3.0, 3.0], "wxyz": DOWN}, num_seeds=32)
    assert not failed["success"]

    explanation = tb.explain_failure(failed["request_id"])
    assert explanation["explained_request"] == failed["request_id"]
    assert explanation["cause"] == "ik_did_not_converge"
    assert "hint" in explanation


def test_explain_failure_on_an_unknown_request_is_not_an_error(tb: Toolbox):
    result = tb.explain_failure("req_9999")
    assert not result["success"]
    assert "no failure recorded" in result["note"]


def test_collision_failures_are_explainable(tb: Toolbox, blocked_path):
    result = tb.validate_path(blocked_path)
    assert not result["valid"]
    explanation = tb.explain_failure(result["request_id"])
    assert explanation["cause"] == "path_invalid"
    assert explanation["first_invalid_waypoint"] is not None


# ── handles ──────────────────────────────────────────────────────────────────


def test_handles_are_type_checked_at_use(tb: Toolbox, blocked_path, endpoints):
    with pytest.raises(ValueError, match="is a path"):
        tb.check_collision(blocked_path)
    with pytest.raises(KeyError, match="unknown handle"):
        tb.check_collision("cfg_99999")


def test_set_robot_state_validates_limits(tb: Toolbox, endpoints):
    result = tb.set_robot_state(endpoints[0])
    assert result["success"] and not result["joint_limit_violations"]

    s = tb.session
    beyond = dict(zip(s.joint_names, s.upper_limits + 1.0))
    violated = tb.set_robot_state(beyond)
    assert not violated["success"]
    assert violated["joint_limit_violations"]
    tb.set_robot_state("default")


# ── collision-model fallback ─────────────────────────────────────────────────


@pytest.mark.slow
def test_mesh_urdf_falls_back_to_capsules_and_says_the_model_is_unreliable():
    """A mesh URDF cannot use the spherized model. Falling back silently would
    leave an agent trusting a collision_free answer that is never true."""
    session = Session(robot="panda", max_objects=4)
    assert session.collision_model == "capsule"
    report = session.self_collision_report
    assert report["calibrated"]
    assert not report["reliable"]
    assert "note" in report and "coarse" in report["note"]
