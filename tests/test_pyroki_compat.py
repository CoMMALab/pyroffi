"""PyRoki (upstream 388e43e) drop-in compatibility for IK/FK, collision and viewer.

`pk.costs` and `Robot.joint_var_cls` (jaxls) are covered by tests/test_costs.py.
"""

import dataclasses
import importlib
import inspect
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pytest
from robot_descriptions.loaders.yourdfpy import load_robot_description

import pyroffi as pk
from pyroffi.collision import Box, RobotCollision, RobotCollisionSpherized, Sphere

RESOURCES = Path(__file__).resolve().parents[1] / "resources"

# Upstream callables -> their parameter lists (positional first, then "*", kw-only).
UPSTREAM_CALLABLES = {
    "Robot.from_urdf": ["urdf", "default_joint_cfg"],
    "Robot.forward_kinematics": ["self", "cfg", "unroll_fk"],
    "collision.collide": ["geom1", "geom2"],
    "collision.colldist_from_sdf": ["_dist", "activation_dist"],
    "collision.CollGeom.get_batch_axes": ["self"],
    "collision.CollGeom.broadcast_to": ["self", "shape"],
    "collision.CollGeom.reshape": ["self", "shape"],
    "collision.CollGeom.transform": ["self", "transform"],
    "collision.CollGeom.transform_from_wxyz_position": ["self", "wxyz", "position"],
    "collision.CollGeom.to_trimesh": ["self"],
    "collision.HalfSpace.from_point_and_normal": ["point", "normal"],
    "collision.Sphere.from_center_and_radius": ["center", "radius"],
    "collision.Capsule.from_radius_height": ["radius", "height", "position", "wxyz"],
    "collision.Capsule.from_trimesh": ["mesh"],
    "collision.Capsule.decompose_to_spheres": ["self", "n_segments"],
    "collision.Capsule.from_sphere_pairs": ["sph_0", "sph_1"],
    "collision.Box.from_extent": ["extent", "position", "wxyz"],
    "collision.Box.get_corners_local": ["self"],
    "collision.Box.get_corners_world": ["self"],
    "collision.Heightmap.from_height_data": ["height_data", "x_scale", "y_scale"],
    "collision.Heightmap.from_trimesh": ["mesh", "resolution", "x_bins", "y_bins"],
    "collision.Heightmap.project_points": ["self", "points"],
    "collision.RobotCollision.from_urdf": [
        "urdf", "user_ignore_pairs", "ignore_immediate_adjacents"],
    "collision.RobotCollision.from_sphere_decomposition": [
        "sphere_decomposition", "urdf", "user_ignore_pairs", "ignore_immediate_adjacents"],
    "collision.RobotCollision.at_config": ["self", "robot", "cfg"],
    "collision.RobotCollision.get_link_collision_meshes": ["self"],
    "collision.RobotCollision.get_swept_capsules": ["self", "robot", "cfg_prev", "cfg_next"],
    "collision.RobotCollision.compute_self_collision_distance": ["self", "robot", "cfg"],
    "collision.RobotCollision.compute_world_collision_distance": [
        "self", "robot", "cfg", "world_geom"],
    "viewer.WeightTuner": [
        "server", "default", "*", "folder_name", "min", "max", "step",
        "default_min", "default_max", "default_step"],
    "viewer.ManipulabilityEllipse": [
        "server", "robot", "root_node_name", "target_link_name", "scaling_factor",
        "visible", "wireframe", "color"],
    "utils.stopwatch": ["label"],
    "utils.jax_log": ["fmt"],
}
UPSTREAM_PROPERTIES = {
    "collision.HalfSpace": ["normal", "offset"],
    "collision.Sphere": ["radius"],
    "collision.Capsule": ["radius", "height", "axis"],
    "collision.Box": ["half_extent", "extent"],
    "collision.Heightmap": ["x_scale", "y_scale", "height_scale", "rows", "cols"],
}
UPSTREAM_FIELDS = {
    "Robot": ["joints", "links"],
    "collision.CollGeom": ["pose", "size"],
    "collision.Heightmap": ["height_data"],
    "collision.RobotCollision": [
        "num_links", "link_names", "coll", "active_idx_i", "active_idx_j"],
}


def _resolve(path):
    parts = path.split(".")
    obj = pk
    for i, p in enumerate(parts):
        obj = getattr(obj, p, None)
        if obj is None and i == 0:
            obj = importlib.import_module(f"pyroffi.{p}")
        assert obj is not None, f"pyroffi.{path} is missing"
    return obj


@pytest.mark.parametrize("path", sorted(UPSTREAM_CALLABLES))
def test_callable_signature_accepts_upstream_calls(path):
    obj = _resolve(path)
    params = list(inspect.signature(obj).parameters.values())
    upstream = UPSTREAM_CALLABLES[path]
    upstream_pos = upstream[: upstream.index("*")] if "*" in upstream else upstream
    ours_pos = [p.name for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    assert ours_pos[: len(upstream_pos)] == upstream_pos, (path, ours_pos)
    names = {p.name for p in params}
    has_var_kw = any(p.kind is p.VAR_KEYWORD for p in params)
    missing = [n for n in upstream if n != "*" and n not in names and not has_var_kw]
    assert not missing, (path, missing)
    new_required = [
        p.name for p in params
        if p.default is p.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        and p.name not in upstream
    ]
    assert not new_required, (path, new_required)


@pytest.mark.parametrize("path", sorted(UPSTREAM_PROPERTIES))
def test_properties(path):
    cls = _resolve(path)
    for name in UPSTREAM_PROPERTIES[path]:
        assert isinstance(inspect.getattr_static(cls, name, None), property), (path, name)


@pytest.mark.parametrize("path", sorted(UPSTREAM_FIELDS))
def test_dataclass_fields(path):
    fields = {f.name for f in dataclasses.fields(_resolve(path))}
    missing = set(UPSTREAM_FIELDS[path]) - fields
    assert not missing, (path, missing)


def test_box_aliases_match_native_names():
    b = Box.from_extent(jnp.array([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]]), position=jnp.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(b.half_extent, b.half_lengths)
    np.testing.assert_allclose(b.extent[0], [1.0, 2.0, 3.0])
    corners = b.get_corners_world()
    assert corners.shape == (2, 8, 3)
    np.testing.assert_allclose(corners[0].min(0), [-0.5, -1.0, -0.5])
    np.testing.assert_allclose(corners[0].max(0), [0.5, 1.0, 2.5])


@pytest.fixture(scope="module")
def panda():
    urdf = load_robot_description("panda_description")
    return urdf, pk.Robot.from_urdf(urdf)


def test_sphere_decomposition_loads_every_sphere(panda):
    urdf, robot = panda
    sd = json.loads((RESOURCES / "panda_spheres.json").read_text())
    rc = RobotCollision.from_sphere_decomposition(sd, urdf)
    assert isinstance(rc, RobotCollisionSpherized)
    n_spheres = sum(len(d["radii"]) for d in sd.values())
    assert int((rc.coll.size[..., 0] > 0).sum()) == n_spheres
    q = (robot.joints.lower_limits + robot.joints.upper_limits) / 2
    world = Sphere.from_center_and_radius(jnp.zeros((2, 3)), jnp.full((2,), 0.1))
    assert rc.compute_world_collision_distance(robot, q, world).shape == (rc.num_links, 2)
    meshes = rc.get_link_collision_meshes()
    assert set(meshes) == set(rc.link_names)
    assert sum(len(m.vertices) > 0 for m in meshes.values()) == sum(1 for d in sd.values() if d["radii"])


def test_vmap_over_stacked_robot_and_collision(panda):
    """Upstream #73's guarantee: Robot/RobotCollision pytrees stack and vmap."""
    urdf, robot = panda
    rc = RobotCollision.from_urdf(urdf)
    q = (robot.joints.lower_limits + robot.joints.upper_limits) / 2
    stack = lambda tree: jax.tree.map(lambda x: jnp.stack([x] * 3), tree)
    grow = jnp.array([0.0, 0.01, 0.02])
    rcs = stack(rc)
    rcs = dataclasses.replace(rcs, coll=dataclasses.replace(
        rcs.coll, size=rcs.coll.size.at[..., 0].add(grow[:, None])))
    d = jax.vmap(lambda r, c: c.compute_self_collision_distance(r, q))(stack(robot), rcs)
    assert d.shape[0] == 3
    # Fatter capsules can only bring links closer.
    assert jnp.all(d[1] <= d[0] + 1e-6) and jnp.all(d[2] <= d[1] + 1e-6)


@pytest.mark.skipif(importlib.util.find_spec("pyroki") is None, reason="upstream pyroki not installed")
def test_parity_with_upstream(panda):
    import pyroki

    urdf, robot = panda
    up_robot = pyroki.Robot.from_urdf(urdf)
    rng = np.random.default_rng(0)
    lo, hi = np.asarray(robot.joints.lower_limits), np.asarray(robot.joints.upper_limits)
    qs = jnp.asarray(rng.uniform(lo, hi, size=(64, lo.shape[0])), dtype=jnp.float32)

    # Compare as matrices: q and -q are the same rotation.
    as_mat = lambda wxyz_xyz: jaxlie.SE3(wxyz_xyz).as_matrix()
    np.testing.assert_allclose(
        as_mat(robot.forward_kinematics(qs)), as_mat(up_robot.forward_kinematics(qs)), atol=1e-5)

    rc, up_rc = RobotCollision.from_urdf(urdf), pyroki.collision.RobotCollision.from_urdf(urdf)
    np.testing.assert_allclose(
        rc.at_config(robot, qs).size, up_rc.at_config(up_robot, qs).size, atol=1e-5)
    world = Sphere.from_center_and_radius(jnp.array([[0.4, 0.0, 0.4]]), jnp.array([0.1]))
    up_world = pyroki.collision.Sphere.from_center_and_radius(jnp.array([[0.4, 0.0, 0.4]]), jnp.array([0.1]))
    # Upstream fails on a batched cfg vs a (1,) world, so compare per config.
    ours = rc.compute_world_collision_distance(robot, qs, world)
    theirs = jax.vmap(lambda q: up_rc.compute_world_collision_distance(up_robot, q, up_world))(qs)
    np.testing.assert_allclose(ours, theirs, atol=1e-5)
    # Same pair set; PyRoFFI orders pairs by first link for kernel locality.
    # PyRoFFI stores (i > j), upstream (i < j): same unordered pairs.
    key = lambda c: [tuple(sorted((int(i), int(j)))) for i, j in zip(c.active_idx_i, c.active_idx_j)]
    ours_pairs, up_pairs = key(rc), key(up_rc)
    assert sorted(ours_pairs) == sorted(up_pairs)
    order = [ours_pairs.index(p) for p in up_pairs]
    np.testing.assert_allclose(
        rc.compute_self_collision_distance(robot, qs)[:, order],
        up_rc.compute_self_collision_distance(up_robot, qs), atol=1e-5)

    sd = json.loads((RESOURCES / "panda_spheres.json").read_text())
    sp = RobotCollision.from_sphere_decomposition(sd, urdf).at_config(robot, qs)
    up_sp = pyroki.collision.RobotCollision.from_sphere_decomposition(sd, urdf).at_config(up_robot, qs)
    # PyRoFFI's posed sphere layout is (*batch, S, N); go link-major to match upstream.
    centers = np.swapaxes(np.asarray(sp.pose.translation()), 1, 2)  # (B, N, S, 3)
    valid = np.swapaxes(np.asarray(sp.size[0, ..., 0] > 0), 0, 1)  # (N, S)
    ours = centers[:, valid]
    np.testing.assert_allclose(ours, up_sp.pose.translation(), atol=1e-5)
