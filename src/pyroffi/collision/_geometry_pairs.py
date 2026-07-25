from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Float, Array

from ._geometry import HalfSpace, Sphere, Capsule, Heightmap, Box
from . import _utils


# --- HalfSpace Collision Implementations ---


def _halfspace_sphere_dist(
    halfspace_normal: Float[Array, "*batch 3"],
    halfspace_point: Float[Array, "*batch 3"],
    sphere_pos: Float[Array, "*batch 3"],
    sphere_radius: Float[Array, "*batch"],
) -> Float[Array, "*batch"]:
    """Helper: Calculates distance between a halfspace boundary plane and sphere center, minus radius."""
    dist = (
        jnp.einsum("...i,...i->...", sphere_pos - halfspace_point, halfspace_normal)
        - sphere_radius
    )
    return dist


def halfspace_sphere(halfspace: HalfSpace, sphere: Sphere) -> Float[Array, "*batch"]:
    """Calculates distance between a halfspace and a sphere."""
    dist = _halfspace_sphere_dist(
        halfspace.normal,
        halfspace.pose.translation(),
        sphere.pose.translation(),
        sphere.radius,
    )
    return dist


def halfspace_capsule(halfspace: HalfSpace, capsule: Capsule) -> Float[Array, "*batch"]:
    """Calculates distance between halfspace and capsule (closest end)."""
    halfspace_normal = halfspace.normal
    halfspace_point = halfspace.pose.translation()
    cap_center = capsule.pose.translation()
    cap_radius = capsule.radius
    cap_axis = capsule.axis
    segment_offset = cap_axis * capsule.height[..., None] / 2
    dist1 = _halfspace_sphere_dist(
        halfspace_normal, halfspace_point, cap_center + segment_offset, cap_radius
    )
    dist2 = _halfspace_sphere_dist(
        halfspace_normal, halfspace_point, cap_center - segment_offset, cap_radius
    )
    final_dist = jnp.minimum(dist1, dist2)
    return final_dist


# --- Sphere/Capsule Collision Implementations ---


def _sphere_sphere_dist(
    pos1: Float[Array, "*batch 3"],
    radius1: Float[Array, "*batch"],
    pos2: Float[Array, "*batch 3"],
    radius2: Float[Array, "*batch"],
) -> Float[Array, "*batch"]:
    """Helper: Calculates distance between two spheres."""
    _, dist_center = _utils.normalize_with_norm(pos2 - pos1)
    dist = dist_center - (radius1 + radius2)
    return dist


def sphere_sphere(sphere1: Sphere, sphere2: Sphere) -> Float[Array, "*batch"]:
    """Calculate distance between two spheres."""
    dist = _sphere_sphere_dist(
        sphere1.pose.translation(),
        sphere1.radius,
        sphere2.pose.translation(),
        sphere2.radius,
    )
    return dist


def sphere_capsule(sphere: Sphere, capsule: Capsule) -> Float[Array, "*batch"]:
    """Calculate distance between sphere and capsule."""
    cap_pos = capsule.pose.translation()
    sphere_pos = sphere.pose.translation()
    cap_axis = capsule.axis
    segment_offset = cap_axis * capsule.height[..., None] / 2
    cap_a = cap_pos - segment_offset
    cap_b = cap_pos + segment_offset
    pt_on_axis = _utils.closest_segment_point(cap_a, cap_b, sphere_pos)
    dist = _sphere_sphere_dist(sphere_pos, sphere.radius, pt_on_axis, capsule.radius)
    return dist


def capsule_capsule(capsule1: Capsule, capsule2: Capsule) -> Float[Array, "*batch"]:
    """Calculate distance between two capsules."""
    pos1 = capsule1.pose.translation()
    axis1 = capsule1.axis
    length1 = capsule1.height
    radius1 = capsule1.radius
    segment1_offset = axis1 * length1[..., None] / 2
    a1 = pos1 - segment1_offset
    b1 = pos1 + segment1_offset

    pos2 = capsule2.pose.translation()
    axis2 = capsule2.axis
    length2 = capsule2.height
    radius2 = capsule2.radius
    segment2_offset = axis2 * length2[..., None] / 2
    a2 = pos2 - segment2_offset
    b2 = pos2 + segment2_offset

    pt1_on_axis, pt2_on_axis = _utils.closest_segment_to_segment_points(a1, b1, a2, b2)
    dist = _sphere_sphere_dist(pt1_on_axis, radius1, pt2_on_axis, radius2)
    return dist


# --- Heightmap Collision Implementations ---


def heightmap_sphere(heightmap: Heightmap, sphere: Sphere) -> Float[Array, "*batch"]:
    """Calculate approximate distance between heightmap and sphere.

    Approximation: Considers the heightmap point directly below the sphere center
    using bilinear interpolation and calculates vertical distance minus radius.
    """
    batch_axes = jnp.broadcast_shapes(
        heightmap.get_batch_axes(), sphere.get_batch_axes()
    )

    sphere_pos_w = sphere.pose.translation()
    sphere_radius = sphere.radius
    interpolated_local_z = heightmap._interpolate_height_at_coords(sphere_pos_w)
    sphere_pos_h = heightmap.pose.inverse().apply(sphere_pos_w)
    sphere_local_z = sphere_pos_h[..., 2]
    dist = sphere_local_z - interpolated_local_z - sphere_radius

    assert dist.shape == batch_axes
    return dist


def heightmap_capsule(heightmap: Heightmap, capsule: Capsule) -> Float[Array, "*batch"]:
    """Calculate approximate distance between heightmap and capsule, by
    checking heightmap points below capsule endpoints.

    Note that this may miss collisions when capsule body intersects but endpoints are above heightmap!
    """
    batch_axes = jnp.broadcast_shapes(
        heightmap.get_batch_axes(), capsule.get_batch_axes()
    )

    cap_pos_w = capsule.pose.translation()
    cap_radius = capsule.radius
    cap_axis_w = capsule.axis  # World frame axis
    segment_offset_w = cap_axis_w * capsule.height[..., None] / 2

    # Calculate world positions of the two end-sphere centers.
    p1_w = cap_pos_w + segment_offset_w
    p2_w = cap_pos_w - segment_offset_w

    # Interpolate heightmap surface height (local Z) below each end-sphere center.
    h_surf1_local = heightmap._interpolate_height_at_coords(p1_w)
    h_surf2_local = heightmap._interpolate_height_at_coords(p2_w)

    # Get end-sphere centers Z coordinates in heightmap's local frame.
    p1_h = heightmap.pose.inverse().apply(p1_w)
    p2_h = heightmap.pose.inverse().apply(p2_w)
    z1_local = p1_h[..., 2]
    z2_local = p2_h[..., 2]

    # Calculate vertical distance for each end sphere.
    dist1 = z1_local - h_surf1_local - cap_radius
    dist2 = z2_local - h_surf2_local - cap_radius

    # Return the minimum distance.
    min_dist = jnp.minimum(dist1, dist2)
    assert min_dist.shape == batch_axes
    return min_dist


def heightmap_halfspace(
    heightmap: Heightmap, halfspace: HalfSpace
) -> Float[Array, "*batch"]:
    """Calculate approximate distance between heightmap and halfspace.

    Approximation: Finds the minimum signed distance between any heightmap vertex
    and the halfspace plane.
    """
    batch_axes = jnp.broadcast_shapes(
        heightmap.get_batch_axes(), halfspace.get_batch_axes()
    )

    # Heightmap vertices in world frame.
    verts_local = heightmap._get_vertices_local()  # (*batch, N, 3), N=H*W
    verts_world = heightmap.pose.apply(verts_local)  # (*batch, N, 3)

    # Halfspace plane properties (world frame).
    hs_normal_w = halfspace.normal  # (*batch, 3)
    hs_point_w = halfspace.pose.translation()  # (*batch, 3)

    # Ensure batch dimensions are compatible for broadcasting.
    batch_axes = jnp.broadcast_shapes(
        heightmap.get_batch_axes(), halfspace.get_batch_axes()
    )
    # Expand dims for broadcasting against vertices.
    hs_normal_w = jnp.broadcast_to(hs_normal_w, batch_axes + (3,))[..., None, :]
    hs_point_w = jnp.broadcast_to(hs_point_w, batch_axes + (3,))[..., None, :]
    verts_world = jnp.broadcast_to(verts_world, batch_axes + verts_world.shape[-2:])

    # Calculate signed distance for each vertex to the plane:
    vertex_distances = jnp.einsum(
        "...vi,...i->...v", verts_world - hs_point_w, hs_normal_w.squeeze(-2)
    )

    # Find the minimum distance among all vertices.
    min_dist = jnp.min(vertex_distances, axis=-1)
    assert min_dist.shape == batch_axes
    return min_dist

def box_box(box1: Box, box2: Box) -> Float[Array, "*batch"]:
    """Compute signed distance between two oriented boxes via the Separating
    Axis Theorem (SAT).

    Tests all 15 candidate separating axes for a pair of OBBs: each box's 3
    face normals, plus the 9 pairwise cross products of their edge
    directions. SAT guarantees this test is *exact* for box-box intersection
    (no false negatives) -- unlike a vertex-in-AABB check, which can miss
    "cross"/T-junction overlaps where two boxes interpenetrate but neither
    has a vertex inside the other (e.g. two thin perpendicular plates
    crossing like a "+").

    When the boxes are separated, returns the (positive) gap along the best
    separating axis found. When they interpenetrate (no separating axis
    exists), returns the negative of the minimum-translation distance -- the
    standard penetration-depth measure for convex polytopes.
    """
    hl1 = box1.half_lengths  # (*batch, 3)
    hl2 = box2.half_lengths  # (*batch, 3)
    t = box2.pose.translation() - box1.pose.translation()  # (*batch, 3)

    # Rows = each box's local X/Y/Z axis expressed in world coordinates.
    axes1 = jnp.moveaxis(box1.pose.rotation().as_matrix(), -1, -2)  # (*batch, 3, 3)
    axes2 = jnp.moveaxis(box2.pose.rotation().as_matrix(), -1, -2)  # (*batch, 3, 3)

    # 9 edge-cross-product axes; near-parallel edges give a ~zero cross
    # product, which can't certify separation -- mask those out below.
    cross_axes = jnp.cross(axes1[..., :, None, :], axes2[..., None, :, :])
    cross_axes = cross_axes.reshape(cross_axes.shape[:-3] + (9, 3))
    cross_norms = jnp.linalg.norm(cross_axes, axis=-1, keepdims=True)
    degenerate = cross_norms[..., 0] < 1e-8
    cross_axes = cross_axes / jnp.where(cross_norms < 1e-8, 1.0, cross_norms)

    all_axes = jnp.concatenate([axes1, axes2, cross_axes], axis=-2)  # (*batch, 15, 3)

    d = jnp.einsum("...i,...ai->...a", t, all_axes)  # center offset along each axis
    proj1 = jnp.einsum("...ai,...ki->...ak", all_axes, axes1)
    proj2 = jnp.einsum("...ai,...ki->...ak", all_axes, axes2)
    ra = jnp.sum(jnp.abs(proj1) * hl1[..., None, :], axis=-1)  # box1's projected half-extent
    rb = jnp.sum(jnp.abs(proj2) * hl2[..., None, :], axis=-1)  # box2's projected half-extent

    overlap = jnp.abs(d) - (ra + rb)  # (*batch, 15); > 0 => this axis separates the boxes

    degenerate_mask = jnp.concatenate(
        [jnp.zeros(degenerate.shape[:-1] + (6,), dtype=bool), degenerate], axis=-1
    )
    overlap = jnp.where(degenerate_mask, -jnp.inf, overlap)

    return jnp.max(overlap, axis=-1)

def box_sphere(box: Box, sphere: Sphere) -> Float[Array, "*batch"]:
    """Compute signed distance between an oriented box and a sphere.

    Uses the standard box SDF in the box's local frame.
    """
    # Sphere center in box local frame
    sph_pos_w = sphere.pose.translation()
    sph_pos_b = box.pose.inverse().apply(sph_pos_w)

    hl = box.half_lengths
    q = jnp.abs(sph_pos_b) - hl
    outside = jnp.sqrt(jnp.sum(jnp.maximum(q, 0.0) ** 2, axis=-1) + 1e-12)
    inside = jnp.minimum(jnp.max(q, axis=-1), 0.0)
    sdist_box = outside + inside

    dist = sdist_box - sphere.radius
    return dist


def box_capsule(box: Box, capsule: Capsule) -> Float[Array, "*batch"]:
    """
    Signed distance between an oriented box and a capsule.

    Steps:
      1. Convert capsule segment endpoints into box-local coordinates.
      2. Ternary-search the segment parameter t in [0, 1] that minimizes the
         box's (rounded-box) SDF -- this SDF is convex in t along any line
         (both outside the box, where it's a Euclidean distance, and inside,
         where it reduces to max_i(|p_i| - hl_i)), so ternary search finds
         the segment's true closest approach to the box. Projecting onto the
         point closest to the box *center* (the previous approach) is only
         exact when the box degenerates to a sphere; for a real box it can
         report the segment as separated when it actually penetrates a face
         off-center (confirmed by brute force: ~3% of random configs, worst
         case reporting +0.035 free when truly -0.23 penetrating).
      3. Subtract capsule radius to get capsule-box SDF.
    """

    cap_pos = capsule.pose.translation()
    cap_axis = capsule.axis
    half_h = capsule.height[..., None] * 0.5

    a_w = cap_pos - cap_axis * half_h
    b_w = cap_pos + cap_axis * half_h

    a = box.pose.inverse().apply(a_w)
    b = box.pose.inverse().apply(b_w)

    hl = box.half_lengths
    ab = b - a

    def box_sdf(p):
        q = jnp.abs(p) - hl
        outside = jnp.sqrt(jnp.sum(jnp.maximum(q, 0.0) ** 2, axis=-1) + 1e-12)
        inside = jnp.minimum(jnp.max(q, axis=-1), 0.0)
        return outside + inside

    def body(_, carry):
        lo, hi = carry
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        f1 = box_sdf(a + m1[..., None] * ab)
        f2 = box_sdf(a + m2[..., None] * ab)
        take_right = f1 > f2
        lo = jnp.where(take_right, m1, lo)
        hi = jnp.where(take_right, hi, m2)
        return lo, hi

    batch_shape = ab.shape[:-1]
    lo, hi = jax.lax.fori_loop(
        0, 30, body, (jnp.zeros(batch_shape), jnp.ones(batch_shape))
    )
    t = (lo + hi) / 2.0

    sdist_box = box_sdf(a + t[..., None] * ab)

    # Capsule SDF = box SDF - capsule radius
    return sdist_box - capsule.radius


def box_halfspace(box: Box, halfspace: HalfSpace) -> Float[Array, "*batch"]:
    """Compute signed distance between box and a halfspace plane.

    We evaluate the halfspace plane signed distance at all eight box vertices
    and return the minimum value. This gives a penetration depth that grows
    (in absolute value) as the box penetrates the halfspace.
    """
    # Box vertices in local frame: combinations of +/- half_lengths
    hl = box.half_lengths
    # Create array of shape (8,3) with vertex signs
    signs = jnp.array(
        [[sx, sy, sz] for sx in (1.0, -1.0) for sy in (1.0, -1.0) for sz in (1.0, -1.0)]
    )
    verts_local = signs[None, ...] * hl[..., None, :]
    # verts_local shape: (*batch_box, 8, 3)
    verts_world = box.pose.apply(verts_local)

    hs_n = halfspace.normal
    hs_pt = halfspace.pose.translation()

    # Broadcast for einsum: ensure hs_n and hs_pt have a vertices axis
    hs_n_bc = jnp.broadcast_to(hs_n, verts_world.shape[:-1] + (3,))[..., None, :]
    hs_pt_bc = jnp.broadcast_to(hs_pt, verts_world.shape[:-1] + (3,))[..., None, :]

    vertex_distances = jnp.einsum(
        "...vi,...i->...v", verts_world - hs_pt_bc, hs_n_bc.squeeze(-2)
    )
    min_dist = jnp.min(vertex_distances, axis=-1)
    return min_dist


def box_heightmap(box: Box, heightmap: Heightmap) -> Float[Array, "*batch"]:
    """Compute approximate signed distance between box vertices and heightmap.

    We check the heightmap surface under each of the box's eight vertices
    (after transforming them into world then heightmap local frame) and
    return the minimum vertical signed distance (vertex_z - surface_z).
    """
    hl = box.half_lengths
    signs = jnp.array(
        [[sx, sy, sz] for sx in (1.0, -1.0) for sy in (1.0, -1.0) for sz in (1.0, -1.0)]
    )
    verts_local = signs[None, ...] * hl[..., None, :]
    verts_world = box.pose.apply(verts_local)

    # verts_world shape: (*batch_box, 8, 3)
    batch_axes = jnp.broadcast_shapes(box.get_batch_axes(), heightmap.get_batch_axes())
    verts_world = jnp.broadcast_to(verts_world, batch_axes + verts_world.shape[-2:])

    # Interpolate heightmap at each vertex world position: flatten verts for call
    flat_verts = verts_world.reshape(batch_axes + (-1, 3))
    # heightmap._interpolate_height_at_coords expects (*batch, 3) -> returns (*batch)
    interp = heightmap._interpolate_height_at_coords(flat_verts)

    # Reshape back to per-vertex and compute vertex z in heightmap local frame
    interp = interp.reshape(batch_axes + (verts_world.shape[-2],))
    verts_h = heightmap.pose.inverse().apply(verts_world)
    vert_local_z = verts_h[..., 2]

    # Signed vertical distance for each vertex: vertex_z - surface_z
    vert_dists = vert_local_z - interp
    min_dist = jnp.min(vert_dists, axis=-1)
    return min_dist
