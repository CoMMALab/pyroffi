"""Sphere-world geometry for the TAMP benchmark, backed by pyroffi / SPaSM math.

The planning world is spheres (SPaSM's representation): every manipulable object
is a set of collision spheres in its own frame, and collision tests reuse
``spasm.tetris.solve.sphere_sphere_penetration`` verbatim so the classical-TAMP baseline
scores geometry with the *identical* function the differentiable SPaSM solver
uses. A "pose" is ``[x, y, z, yaw]`` (numpy, shape (4,)); object-local spheres are
``(K, 4)`` = ``[x, y, z, r]``.

Grasp/IK is the SPaSM motion primitive: a top-down grasp over the object centre,
lifted to a 7-DOF Franka config by ``spasm.conversions.grasp_to_q`` (pyroffi
analytic IK). Arm collision uses ``backend.fk`` spheres.
"""
from __future__ import annotations

import functools
import os

import jax
import jax.numpy as jnp
import numpy as np

from . import _setup  # noqa: F401  (path shim; must precede backend import)
from spasm import backend
from spasm.conversions import yaw_to_quat_xyz

# IK backend for the oracle's `s-ik` stream. `PYROFFI_ANALYTIC_IK=1` swaps
# SPaSM's hand-rolled Franka closed form for pyroffi's model-derived analytic
# solver (pyroffi.kinematics._analytic_ik). Identical signature, so nothing
# downstream changes; see spasm/pyroffi_ik.py.
#   spasm          SPaSM's hand-rolled Franka closed form (default)
#   pyroffi        pyroffi's model-derived analytic solver
#   pyroffi-cfree  as above, but picks a self-collision-free branch
IK_BACKEND = os.environ.get("PYROFFI_ANALYTIC_IK", "spasm")
if IK_BACKEND in ("1", "pyroffi"):
    IK_BACKEND = "pyroffi"
    from spasm.pyroffi_ik import grasp_to_q
elif IK_BACKEND == "pyroffi-cfree":
    from spasm.pyroffi_ik import grasp_to_q_self_free as grasp_to_q
elif IK_BACKEND in ("0", "spasm"):
    IK_BACKEND = "spasm"
    from spasm.conversions import grasp_to_q
else:
    raise ValueError(
        f"PYROFFI_ANALYTIC_IK={IK_BACKEND!r}; expected one of "
        "'spasm', 'pyroffi', 'pyroffi-cfree'")
from spasm.tetris.solve import sphere_sphere_penetration

# --------------------------------------------------------------------------- #
# Object -> collision spheres
# --------------------------------------------------------------------------- #

def cuboid_spheres(half_extents, n_per_axis=(2, 2, 1)):
    """Fill an axis-aligned cuboid (half-extents (hx,hy,hz)) with an
    ``n_per_axis`` grid of spheres whose union *contains* the box. Returns local
    spheres (K,4).

    Each grid cell is covered by a sphere of radius = the cell's half-diagonal,
    so the union is a conservative (over-approximate) enclosure of the AABB —
    the right bias for a collision test."""
    half = np.asarray(half_extents, dtype=float)
    n = np.asarray(n_per_axis, dtype=int)
    cell_half = half / np.maximum(n, 1)          # per-axis half-size of a cell
    r = float(np.linalg.norm(cell_half))          # covers each cell's corner
    axes = []
    for h, ni, ch in zip(half, n, cell_half):
        axes.append(np.array([0.0]) if ni <= 1
                    else np.linspace(-h + ch, h - ch, ni))
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)
    sph = np.concatenate([pts, np.full((pts.shape[0], 1), r)], axis=-1)
    return jnp.asarray(sph, dtype=jnp.float32)


# --------------------------------------------------------------------------- #
# Pose transforms (yaw about z)
# --------------------------------------------------------------------------- #

@jax.jit
def _transform_spheres(local, pose):
    """local (K,4), pose (4,)=[x,y,z,yaw] -> world spheres (K,4)."""
    c, s = jnp.cos(pose[3]), jnp.sin(pose[3])
    R = jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    xyz = local[:, :3] @ R.T + pose[:3]
    return jnp.concatenate([xyz, local[:, 3:4]], axis=-1)


def transform_spheres(local, pose):
    return _transform_spheres(jnp.asarray(local), jnp.asarray(pose, jnp.float32))


# --------------------------------------------------------------------------- #
# Collision predicates (reuse SPaSM's exact penetration math)
# --------------------------------------------------------------------------- #

@jax.jit
def _max_pair_penetration(sph1, sph2):
    return jnp.max(sphere_sphere_penetration(sph1, sph2))


def blocks_collide(local1, pose1, local2, pose2, eps=1e-3):
    """True if two placed objects interpenetrate beyond ``eps`` (metres)."""
    s1 = transform_spheres(local1, pose1)
    s2 = transform_spheres(local2, pose2)
    pen = float(_max_pair_penetration(s1, s2))
    # sphere_sphere_penetration folds in a 0.010 margin and abs()s it; the
    # collision-free floor is that margin, so compare against margin + eps.
    return pen > 0.010 + eps


# --------------------------------------------------------------------------- #
# Grasp / IK — the SPaSM motion primitive (pyroffi analytic Franka IK)
# --------------------------------------------------------------------------- #

# Neutral-ish seed used by SPaSM's tetris trajopt.
NEUTRAL_Q = jnp.array([0.0, -jnp.pi / 4, 0.0, -3 * jnp.pi / 4, 0.0,
                       jnp.pi / 2, jnp.pi / 4, 0.0, 0.0])

TOP_DOWN_APPROACH = 0.10  # metres above the object centre for the grasp point


@functools.partial(jax.jit, static_argnums=())
def _grasp_to_q_topdown(pose, grasp_yaw, approach):
    """pose (4,)=[x,y,z,yaw] object placement -> (q7, error).

    Builds a top-down grasp SE3 over the object centre (offset ``approach`` up
    in z) with the given wrist yaw, then calls the SPaSM analytic-IK primitive.
    error is 0.0 if reachable, jnp.inf otherwise."""
    grasp_xyzyaw = jnp.array([pose[0], pose[1], pose[2] + approach, grasp_yaw])
    grasp_quat = yaw_to_quat_xyz(grasp_xyzyaw)          # (7,) [xyz quat_xyzw]
    q9, err = grasp_to_q(grasp_quat)
    return q9[:7], err


def ik_topdown(pose, grasp_yaw=0.0, approach=TOP_DOWN_APPROACH):
    """Return (q (7,) np.ndarray, reachable bool) for a top-down grasp at pose."""
    q7, err = _grasp_to_q_topdown(
        jnp.asarray(pose, jnp.float32), float(grasp_yaw), float(approach))
    return np.asarray(q7), bool(np.isfinite(float(err)))


# --------------------------------------------------------------------------- #
# Arm collision along a joint path (pyroffi FK spheres vs floor + placed blocks)
# --------------------------------------------------------------------------- #

@jax.jit
def _arm_below_floor(q, floor_z):
    pos, rad = backend.fk(q)          # (K,3),(K,4)
    return jnp.min(pos[:, 2] - rad) < floor_z


# The Panda's static base collision spheres reach z-r == -0.030 at every config
# (mount geometry); the table top is z==0. Gate just below the base minimum so a
# static base always passes but a wrist/hand pushed into the table is caught.
FLOOR_Z = -0.035

#: Include self-collision in path validation. Needed for parity with the cuRobo
#: backend, which checks it. Off leaves the historical behaviour (floor + joint
#: limits only), which silently admits self-intersecting configurations.
CHECK_SELF_COLLISION = os.environ.get("PYROFFI_SELF_COLLISION", "1") == "1"


@functools.lru_cache(maxsize=1)
def _self_collision_model():
    """Collision model built WITH an SRDF.

    Without one the spherized model's conservative enclosure leaves adjacent
    links overlapping by construction — self-clearance is about -0.03m even at
    the neutral pose — so every configuration reads as self-colliding and path
    validation rejects everything. ``backend.ROBOT_COLL`` is built without an
    SRDF (it is only used for FK sphere positions, where this does not matter),
    so a separate SRDF-aware model is built here.
    """
    import os.path as osp

    import pyroffi as pk
    import yourdfpy

    from spasm.paths import PYROFFI_ROOT, SPASM_URDF

    srdf = osp.join(PYROFFI_ROOT, "resources", "panda", "panda.srdf")
    urdf = yourdfpy.URDF.load(SPASM_URDF, load_collision_meshes=True)
    if not osp.exists(srdf):
        return None
    return pk.collision.RobotCollisionSpherized.from_urdf(urdf, srdf_path=srdf)


def _make_path_validator():
    """Build the fused, jitted path predicate once.

    Fusing matters more than it looks. The original walked the path in Python
    with two blocking device syncs (``bool(jnp.all(...))`` then
    ``bool(jnp.any(...))``), serialised by an early return, and rebuilt the
    ``vmap`` closure on every call. Collapsing it to a single jitted call with
    one sync measured ~1.9x faster on an identical predicate — the cost was
    dispatch overhead, not geometry.
    """
    coll = _self_collision_model() if CHECK_SELF_COLLISION else None

    @jax.jit
    def _valid(q_path):
        q = q_path[:, :7]
        lo, hi = backend.get_joint_limits()
        ok = jnp.all((q >= lo - 1e-3) & (q <= hi + 1e-3))

        def floor_ok(qi):
            pos, rad = backend.fk(qi)
            return jnp.min(pos[:, 2] - rad) >= FLOOR_Z

        ok = ok & jnp.all(jax.vmap(floor_ok)(q))

        if coll is not None:
            n_act = len(backend.ROBOT.joints.actuated_names)

            def self_ok(qi):
                q_full = jnp.zeros((n_act,), qi.dtype).at[:7].set(qi)
                d = coll.compute_self_collision_distance(backend.ROBOT, q_full)
                return jnp.min(d) > 0.0
            ok = ok & jnp.all(jax.vmap(self_ok)(q))
        return ok

    return _valid


@functools.lru_cache(maxsize=1)
def _path_validator():
    return _make_path_validator()


@functools.lru_cache(maxsize=1)
def _fused_checker():
    """Fused FK+collision checker, built eagerly. None if unavailable.

    Must be constructed OUTSIDE any jit trace: it extracts concrete sphere
    geometry and pair tables from the collision model, and under tracing those
    arrive as tracers. Built once here at first use (from module scope, never
    inside a traced function), it is then perfectly usable *inside* jit -- the
    FFI call itself traces fine, and carries a custom_jvp so gradients work.
    """
    if os.environ.get("PYROFFI_FUSED_COLLISION", "1") != "1":
        return None
    try:
        import pyroffi as pk
        from pyroffi.collision import FusedCUDACollisionChecker

        coll = _self_collision_model()
        if coll is None or not FusedCUDACollisionChecker.available():
            return None
        return FusedCUDACollisionChecker(backend.ROBOT, coll)
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def _batched_path_validator():
    """Validate a whole batch of paths in one fused call.

    The per-path validator vmaps a predicate that internally recomputes FK and
    re-enters the collision model per waypoint. This instead flattens
    ``[N, T, 7]`` to ``N*T`` configurations and hands them to the fused kernel
    as a single batch, so FK runs once per waypoint inside the collision kernel
    and no intermediate sphere tensor is materialised.

    That matters because a parallel TAMP validator's whole job is exactly this
    shape: many candidate plans, each many waypoints, all independent.

    Floor clearance comes out of the same launch as the self-collision check.
    Deriving it instead from a separate ``vmap(backend.fk)`` -- as this did
    originally -- ran FK twice per waypoint, once in JAX to place spheres for
    the floor test and again inside the kernel. The redundant pass measured
    3.47 ms against the kernel's 1.93 ms at B=61440, so two thirds of the
    validator's time went to recomputing kinematics it already had. It is also
    the slower of the two FKs, because the JAX model pads every link out to the
    widest one (13x18 slots for 59 real spheres) while the kernel walks a ragged
    CSR layout.
    """
    ck = _fused_checker()
    if ck is None:
        return None

    @jax.jit
    def _valid(paths):                       # [N, T, 7] -> [N]
        N, T = paths.shape[0], paths.shape[1]
        q = paths[..., :7].reshape(N * T, 7)

        lo, hi = backend.get_joint_limits()
        in_lim = jnp.all((q >= lo - 1e-3) & (q <= hi + 1e-3), axis=-1)

        n_act = len(backend.ROBOT.joints.actuated_names)
        q_full = jnp.zeros((N * T, n_act), q.dtype).at[:, :7].set(q)
        d_self, min_z = ck.compute_self_collision_and_floor(backend.ROBOT, q_full)

        self_ok = jnp.min(d_self, axis=-1) > 0.0
        floor_ok = min_z >= FLOOR_Z

        return jnp.all((in_lim & floor_ok & self_ok).reshape(N, T), axis=1)

    return _valid


def arm_paths_valid(paths, floor_z=FLOOR_Z):
    """Batched path validation: ``[N, T, 7]`` -> ``[N]`` bool.

    Uses the fused CUDA kernel when available, else falls back to mapping the
    single-path validator. The fallback is explicit rather than silent-on-error:
    availability is decided once, up front, so a failure cannot masquerade as a
    performance result.
    """
    paths = jnp.asarray(paths, jnp.float32)
    fast = _batched_path_validator()
    if fast is not None:
        return np.asarray(fast(paths))
    v = _path_validator()
    return np.asarray(jax.vmap(v)(paths))


def arm_path_valid(q_path, floor_z=FLOOR_Z):
    """Path validity: joint limits, floor clearance, and self-collision.

    One jitted call and one device sync. ``floor_z`` is accepted for interface
    compatibility but is baked into the compiled predicate; pass a different
    value only by changing :data:`FLOOR_Z` before first use.
    """
    return bool(_path_validator()(jnp.asarray(q_path, jnp.float32)))


def interpolate(q1, q2, n=20):
    """Straight-line joint-space path (matches SPaSM trajopt's interp init)."""
    q1 = np.asarray(q1)[:7]
    q2 = np.asarray(q2)[:7]
    ts = np.linspace(0.0, 1.0, n)[:, None]
    return (1.0 - ts) * q1[None] + ts * q2[None]
