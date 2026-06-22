"""CUDA-accelerated collision checking backend, using the JAX FFI interface.

pRRTC Reference: https://github.com/lyf44/pRRTC

Supports both capsule-based (RobotCollision) and sphere-based
(RobotCollisionSpherized) robot geometry.  The API mirrors the
JAX-based classes so that either backend can be dropped in.

Distance convention (matches pyroffi):
  positive  →  geometries separated
  negative  →  penetration / collision

World geometry types handled by the CUDA kernel:
  Sphere, Capsule, Box, HalfSpace

Data layout passed into CUDA (all float32, row-major):
  world spheres     [Ms, 4]  (x, y, z, r)
  world capsules    [Mc, 7]  (x1, y1, z1, x2, y2, z2, r)
  world boxes       [Mb, 15] (cx, cy, cz,
                              a1x, a1y, a1z,   ← local X axis
                              a2x, a2y, a2z,   ← local Y axis
                              a3x, a3y, a3z,   ← local Z axis
                              hl1, hl2, hl3)
  world halfspaces  [Mh, 6]  (nx, ny, nz, px, py, pz)
                              ← unit outward normal + point on plane

Robot geometry passed into CUDA (SoA layout for coalesced loads):
  capsule-robot   [7, B, N]  — component-major: (x1,y1,z1,x2,y2,z2,r) × (B, N)
  sphere-robot centers [3, B, K]  — component-major: (x,y,z) × (B, K)
  sphere-robot radii   [B, K]

Requires the compiled shared library _collision_cuda_lib.so:
  bash build_kernels/build_collision_cuda.sh
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Optional, Union, cast

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
from jaxtyping import Array, Float
from loguru import logger

from ._geometry import Box, Capsule, CollGeom, HalfSpace, Sphere
from ._robot_collision import RobotCollision, RobotCollisionSpherized
from ..cuda_kernels.collision._collision_cuda_ffi import (
    _load_and_register,
    collision_world_sphere,
    collision_world_sphere_reduced,
    collision_world_capsule,
    collision_self_sphere,
    collision_self_capsule,
)
from ..cuda_kernels.collision._collision_binary_cuda_ffi import (
    _load_and_register as _load_and_register_binary,
    collision_binary,
)

if TYPE_CHECKING:
    from pyroffi._robot import Robot


# ---------------------------------------------------------------------------
# World geometry helper
# ---------------------------------------------------------------------------

def _pose_translation_np(pose) -> np.ndarray:
    """Extract translation (xyz) from a jaxlie.SE3 as a concrete numpy array.

    Avoids calling ``pose.translation()`` which performs a JAX slice operation
    that creates an abstract traced value inside ``jax.lax.scan`` bodies.
    """
    return np.asarray(pose.wxyz_xyz)[..., 4:7].astype(np.float32)


def _pose_rotation_matrix_np(pose) -> np.ndarray:
    """Extract rotation matrix from a jaxlie.SE3 as a concrete numpy array.

    Computes the quaternion → rotation-matrix conversion in pure numpy so that
    no JAX operations are performed (safe inside ``jax.lax.scan`` bodies).
    Convention: jaxlie stores quaternions as (w, x, y, z).
    """
    wxyz = np.asarray(pose.wxyz_xyz)[..., :4].astype(np.float64)
    w, x, y, z = wxyz[..., 0], wxyz[..., 1], wxyz[..., 2], wxyz[..., 3]
    R = np.empty(wxyz.shape[:-1] + (3, 3), dtype=np.float32)
    R[..., 0, 0] = (1 - 2*(y*y + z*z)).astype(np.float32)
    R[..., 0, 1] = (2*(x*y - w*z)).astype(np.float32)
    R[..., 0, 2] = (2*(x*z + w*y)).astype(np.float32)
    R[..., 1, 0] = (2*(x*y + w*z)).astype(np.float32)
    R[..., 1, 1] = (1 - 2*(x*x + z*z)).astype(np.float32)
    R[..., 1, 2] = (2*(y*z - w*x)).astype(np.float32)
    R[..., 2, 0] = (2*(x*z - w*y)).astype(np.float32)
    R[..., 2, 1] = (2*(y*z + w*x)).astype(np.float32)
    R[..., 2, 2] = (1 - 2*(x*x + y*y)).astype(np.float32)
    return R


def _extract_world_arrays(
    world_geom: CollGeom,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert a pyroffi CollGeom (last axis = M obstacles) to flat numpy arrays.

    All conversions use pure numpy (no JAX operations), making this function
    safe to call inside ``jax.lax.scan`` bodies.

    Returns:
        spheres    — float32 [Ms, 4]
        capsules   — float32 [Mc, 7]
        boxes      — float32 [Mb, 15]
        halfspaces — float32 [Mh, 6]
    """
    empty_s = np.zeros((0, 4),  dtype=np.float32)
    empty_c = np.zeros((0, 7),  dtype=np.float32)
    empty_b = np.zeros((0, 15), dtype=np.float32)
    empty_h = np.zeros((0, 6),  dtype=np.float32)

    axes = world_geom.get_batch_axes()

    # Normalize: scalar → single obstacle
    if len(axes) == 0:
        world_geom = world_geom.broadcast_to((1,))
        axes = (1,)

    if len(axes) > 1:
        raise ValueError(
            "CUDARobotCollisionChecker only supports world_geom without leading "
            f"batch dimensions; got shape {axes}. "
            "Pre-extract the world geometry manually."
        )

    if isinstance(world_geom, Sphere):
        centers = _pose_translation_np(world_geom.pose)             # (M, 3)
        radii   = np.asarray(world_geom.radius, dtype=np.float32)   # (M,)
        spheres = np.concatenate([centers, radii[:, None]], axis=-1) # (M, 4)
        return spheres, empty_c, empty_b, empty_h

    if isinstance(world_geom, Capsule):
        centers = _pose_translation_np(world_geom.pose)              # (M, 3)
        axes_v  = np.asarray(world_geom.axis,   dtype=np.float32)    # (M, 3)
        heights = np.asarray(world_geom.height, dtype=np.float32)    # (M,)
        radii   = np.asarray(world_geom.radius, dtype=np.float32)    # (M,)
        half_h  = heights[:, None] * 0.5
        a       = centers - axes_v * half_h                          # (M, 3)
        b_      = centers + axes_v * half_h                          # (M, 3)
        capsules = np.concatenate([a, b_, radii[:, None]], axis=-1)  # (M, 7)
        return empty_s, capsules, empty_b, empty_h

    if isinstance(world_geom, Box):
        centers = _pose_translation_np(world_geom.pose)              # (M, 3)
        R       = _pose_rotation_matrix_np(world_geom.pose)          # (M, 3, 3)
        ax1     = R[..., :, 0]                                       # (M, 3) local X
        ax2     = R[..., :, 1]                                       # (M, 3) local Y
        ax3     = R[..., :, 2]                                       # (M, 3) local Z
        hl      = np.asarray(world_geom.half_lengths, dtype=np.float32)  # (M, 3)
        boxes   = np.concatenate([centers, ax1, ax2, ax3, hl], axis=-1)  # (M, 15)
        return empty_s, empty_c, boxes, empty_h

    if isinstance(world_geom, HalfSpace):
        R       = _pose_rotation_matrix_np(world_geom.pose)          # (M, 3, 3)
        normals = R[..., :, 2].astype(np.float32)                    # (M, 3) — Z col = normal
        points  = _pose_translation_np(world_geom.pose)              # (M, 3)
        halfspaces = np.concatenate([normals, points], axis=-1)      # (M, 6)
        return empty_s, empty_c, empty_b, halfspaces

    raise NotImplementedError(
        f"CUDA backend does not support world_geom of type {type(world_geom).__name__}. "
        "Supported: Sphere, Capsule, Box, HalfSpace."
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CUDADifferentiableSDFCollisionChecker:
    """CUDA-accelerated drop-in replacement for RobotCollision / RobotCollisionSpherized.

    Uses CUDA kernels via the JAX FFI interface (XLA custom calls), matching
    pRRTC's parallelism strategy: each collision pair in a batch is dispatched
    to a dedicated CUDA thread.

    Wraps the original JAX-based collision object for forward kinematics and
    metadata, but replaces the distance computation with CUDA kernels.

    Performance notes:
        - Call ``set_world(world_geom)`` once before the main loop to pre-upload
          world geometry to the device.  compute_world_collision_distance does this
          lazily on the first call (keyed by object identity), but pre-caching
          avoids the numpy→device copy on the hot path.
        - Both public entry points are wrapped with ``jax.jit`` on the first call
          for a given (robot, world) pair, so the first call per unique input shape
          triggers tracing + XLA compilation.  Subsequent calls with the same shapes
          hit the compiled cache.
        - Robot capsule/sphere geometry is assembled inside the JIT boundary.
        - For RobotCollisionSpherized, the fused S-reduction kernel outputs [B, N, M]
          directly — no Python-side reshape+min overhead.

    World geometry types supported (all four are handled by CUDA kernels):
        Sphere, Capsule, Box, HalfSpace

    Usage::

        from pyroffi.collision import RobotCollisionSpherized
        from pyroffi.collision._cuda_collision import CUDARobotCollisionChecker

        robot_coll_jax  = RobotCollisionSpherized.from_urdf(urdf)
        robot_coll_cuda = CUDARobotCollisionChecker(robot_coll_jax)
        robot_coll_cuda.set_world(world_geom)          # pre-cache world geometry

        # Both APIs are identical from here:
        dist = robot_coll_cuda.compute_world_collision_distance(robot, cfg, world_geom)
        dist = robot_coll_cuda.compute_self_collision_distance(robot, cfg)

    Notes:
        - Requires the compiled _collision_cuda_lib.so.
          Build with: bash build_kernels/build_collision_cuda.sh
        - World geometry must be a flat collection (no leading batch dims).
        - FK is performed by the CUDA FK kernel (via ``robot.forward_kinematics(use_cuda=True)``);
          geometry transforms and distance computation both run on CUDA/device.
        - All arrays stay on-device as JAX arrays throughout — no host round-trips.
    """

    def __init__(
        self,
        inner: Union[RobotCollision, RobotCollisionSpherized],
        coarse_inner: Optional[Union[RobotCollision, RobotCollisionSpherized]] = None,
    ) -> None:
        _load_and_register()  # verify the .so exists and register FFI targets

        self._inner = inner

        # Active self-collision pair indices as JAX int32 arrays (device-resident)
        self._pair_i = jnp.array(inner.active_idx_i, dtype=jnp.int32)
        self._pair_j = jnp.array(inner.active_idx_j, dtype=jnp.int32)

        # Optional coarse collision model for two-phase checking.
        # When set, collision methods first check the cheaper coarse geometry;
        # if it is collision-free the fine kernel is skipped entirely.
        # NOTE: this breaks SDF differentiability — do not use inside trajopt.
        self._coarse_inner = coarse_inner
        if coarse_inner is not None:
            self._coarse_pair_i = jnp.array(coarse_inner.active_idx_i, dtype=jnp.int32)
            self._coarse_pair_j = jnp.array(coarse_inner.active_idx_j, dtype=jnp.int32)
        else:
            self._coarse_pair_i = None
            self._coarse_pair_j = None

        # World geometry cache — populated by set_world() or lazily on first call.
        # Stored as JAX device arrays so the device upload happens exactly once.
        self._ws = None          # [Ms, 4]  float32, device
        self._wc = None          # [Mc, 7]  float32, device
        self._wb = None          # [Mb, 15] float32, device
        self._wh = None          # [Mh, 6]  float32, device
        self._cached_world_id = None   # id() of the last world_geom object

        # JIT cache — keyed on robot object identity.
        self._cached_robot_id = None
        self._jit_world = None   # jax.jit'd _compute_world_impl (or coarse-first variant)
        self._jit_self  = None   # jax.jit'd _compute_self_impl  (or coarse-first variant)

        logger.info(
            f"CUDARobotCollisionChecker (JAX FFI) wrapping "
            f"{type(inner).__name__} with {inner.num_links} links"
            + (
                f", coarse model: {type(coarse_inner).__name__} "
                f"with {coarse_inner.num_links} links."
                if coarse_inner is not None
                else "."
            )
        )

    # ── Metadata forwarding ────────────────────────────────────────────────

    @property
    def num_links(self) -> int:
        return self._inner.num_links

    @property
    def link_names(self) -> tuple[str, ...]:
        return self._inner.link_names

    @property
    def active_idx_i(self):
        return self._inner.active_idx_i

    @property
    def active_idx_j(self):
        return self._inner.active_idx_j

    def at_config(self, robot: "Robot", cfg) -> CollGeom:
        """Forward to inner JAX model."""
        return self._inner.at_config(robot, cfg)

    def get_swept_capsules(self, robot, cfg_prev, cfg_next):
        """Forward to inner JAX model."""
        return self._inner.get_swept_capsules(robot, cfg_prev, cfg_next)

    # ── World geometry caching ─────────────────────────────────────────────

    def set_world(self, world_geom: CollGeom) -> None:
        """Pre-upload world geometry to the device (call once, before the loop).

        Converts world_geom to flat float32 numpy arrays and uploads them as
        JAX device arrays.  Subsequent calls to compute_world_collision_distance
        reuse these device arrays without any host→device copy.

        Also invalidates any cached JIT'd world-collision function so that the
        next compute_world_collision_distance call retraces with the new world
        shapes.
        """
        ws_np, wc_np, wb_np, wh_np = _extract_world_arrays(world_geom)
        self._ws = jnp.array(ws_np)
        self._wc = jnp.array(wc_np)
        self._wb = jnp.array(wb_np)
        self._wh = jnp.array(wh_np)
        self._cached_world_id = id(world_geom)
        # Invalidate JIT cache so the new world shapes trigger retracing.
        self._cached_robot_id = None
        self._jit_world = None

    def _ensure_world_cache(self, world_geom: CollGeom) -> None:
        """Lazily populate the world cache if world_geom changed."""
        if id(world_geom) != self._cached_world_id:
            self.set_world(world_geom)

    # ── JIT cache management ───────────────────────────────────────────────

    def _ensure_jit(self, robot: "Robot") -> None:
        """Build (or reuse) JIT'd impl functions closed over robot.

        When a coarse model is attached the JIT'd functions use the two-phase
        coarse-first logic; otherwise the plain fine-only path is compiled.
        """
        if id(robot) == self._cached_robot_id:
            return

        _robot = robot  # capture by value in local scope

        if self._coarse_inner is not None:
            def _world_impl(cfg, ws, wc, wb, wh):
                return self._compute_world_impl_coarse_first(_robot, cfg, ws, wc, wb, wh)

            def _self_impl(cfg):
                return self._compute_self_impl_coarse_first(_robot, cfg)
        else:
            def _world_impl(cfg, ws, wc, wb, wh):
                return self._compute_world_impl(_robot, cfg, ws, wc, wb, wh)

            def _self_impl(cfg):
                return self._compute_self_impl(_robot, cfg)

        self._jit_world = jax.jit(_world_impl)
        self._jit_self  = jax.jit(_self_impl)
        self._cached_robot_id = id(robot)

    # ── Internal FK helper ─────────────────────────────────────────────────

    def _apply_transforms_for(
        self,
        inner: Union[RobotCollision, RobotCollisionSpherized],
        Ts_world_link_arr,
        is_batched: bool,
    ) -> CollGeom:
        """Apply link transforms to the collision geometry of *inner*.

        Separated from FK so that coarse and fine models can share a single
        ``robot.forward_kinematics`` call (same joints → same transforms).
        """
        def _apply(T_arr):
            T = jaxlie.SE3(T_arr)
            if isinstance(inner, RobotCollisionSpherized):
                coll_n_s = jax.vmap(
                    lambda ts, c: c.transform(ts),
                    in_axes=(0, 0),
                    out_axes=0,
                )(T, inner.coll)
                return cast(CollGeom, jax.tree.map(
                    lambda x: jnp.swapaxes(x, 0, 1),
                    coll_n_s,
                ))
            else:
                return inner.coll.transform(T)

        if is_batched:
            return jax.vmap(_apply)(Ts_world_link_arr)
        else:
            return _apply(Ts_world_link_arr)

    def _at_config_batched(self, robot: "Robot", cfg) -> CollGeom:
        """Run CUDA FK then transform collision geometry (fine model).

        The CUDA FK kernel runs once for the full batch (efficient), then the
        geometry transform is vmapped per batch element.  This avoids the
        redundant JAX FK inside ``at_config`` while keeping the per-element
        geometry transform that ``jdc.copy_and_mutate`` requires (it enforces
        that pose shapes don't change between calls).
        """
        cfg_arr = jnp.asarray(cfg)
        is_batched = cfg_arr.ndim > 1
        Ts_world_link_arr = robot.forward_kinematics(cfg_arr, use_cuda=True)
        return self._apply_transforms_for(self._inner, Ts_world_link_arr, is_batched)

    # ── Internal geometry extraction ───────────────────────────────────────

    def _sphere_robot_arrays(
        self, coll: CollGeom, is_batched: bool
    ) -> tuple[Array, Array, int, int, tuple]:
        """Extract sphere geometry as JAX arrays in SoA layout.

        Returns:
            centers_soa — float32 [3, B, K]  component-first (SoA)
            radii       — float32 [B, K]
            S, N        — Python ints  (S spheres per link, N links; K = S*N)
            batch_shape — Python tuple
        """
        batch_axes = coll.get_batch_axes()
        N = int(batch_axes[-1])

        centers = coll.pose.translation().astype(jnp.float32)  # (*batch, S, N, 3)
        radii   = coll.radius.astype(jnp.float32)              # (*batch, S, N)

        if is_batched:
            B = int(batch_axes[0])
            S = int(batch_axes[1])
            batch_shape = (B,)
        else:
            S = int(batch_axes[0])
            B = 1
            batch_shape = ()
            centers = jnp.expand_dims(centers, axis=0)  # (1, S, N, 3)
            radii   = jnp.expand_dims(radii,   axis=0)  # (1, S, N)

        # AoS → SoA: [B, K, 3] → [3, B, K]
        centers_aoi = centers.reshape(B, S * N, 3)
        centers_soa = jnp.transpose(centers_aoi, (2, 0, 1))  # [3, B, K]

        return (
            centers_soa,
            radii.reshape(B, S * N),
            S, N, batch_shape,
        )

    def _capsule_robot_arrays(
        self, coll: CollGeom, is_batched: bool
    ) -> tuple[Array, int, tuple]:
        """Extract flat capsule array in SoA layout [7, B, N] as a JAX array."""
        batch_axes = coll.get_batch_axes()
        N = int(batch_axes[-1])

        centers = coll.pose.translation().astype(jnp.float32)  # (*batch, N, 3)
        axes_v  = coll.axis.astype(jnp.float32)                # (*batch, N, 3)
        heights = coll.height.astype(jnp.float32)              # (*batch, N)
        radii   = coll.radius.astype(jnp.float32)              # (*batch, N)

        if is_batched:
            B = int(batch_axes[0])
            batch_shape = (B,)
        else:
            B = 1
            batch_shape = ()
            centers = jnp.expand_dims(centers, axis=0)
            axes_v  = jnp.expand_dims(axes_v,  axis=0)
            heights = jnp.expand_dims(heights, axis=0)
            radii   = jnp.expand_dims(radii,   axis=0)

        half_h = heights[..., None] * 0.5
        a    = centers - axes_v * half_h
        b_   = centers + axes_v * half_h
        # AoS [B, N, 7] → SoA [7, B, N] for coalesced kernel loads
        caps_aoi = jnp.concatenate([a, b_, radii[..., None]], axis=-1)  # (B, N, 7)
        caps_soa = jnp.transpose(caps_aoi, (2, 0, 1))                   # (7, B, N)
        return caps_soa, N, batch_shape

    # ── Core implementations (JIT-able) ────────────────────────────────────

    def _compute_world_impl(self, robot, cfg, ws, wc, wb, wh):
        """World collision implementation — JIT'd via _ensure_jit.

        For RobotCollisionSpherized: uses the fused S-reduction kernel, which
        outputs [B, N, M] directly without a Python-side reshape+min.

        For RobotCollision (capsule): uses the type-split capsule kernel.

        Args:
            cfg — [*batch, DOF] or [DOF]
            ws  — [Ms, 4]   world spheres (device)
            wc  — [Mc, 7]   world capsules (device)
            wb  — [Mb, 15]  world boxes (device)
            wh  — [Mh, 6]   world halfspaces (device)
        """
        M = ws.shape[0] + wc.shape[0] + wb.shape[0] + wh.shape[0]
        is_batched = jnp.asarray(cfg).ndim > 1

        coll = self._at_config_batched(robot, cfg)

        if isinstance(self._inner, RobotCollisionSpherized):
            centers_soa, radii, S, N, batch_shape = self._sphere_robot_arrays(coll, is_batched)

            # Fused kernel: reduces S spheres per link internally → [B, N, M]
            out = collision_world_sphere_reduced(
                centers_soa, radii, ws, wc, wb, wh, n=N
            )
            return out.reshape(*batch_shape, N, M)

        else:  # RobotCollision (capsule-based)
            caps_soa, N, batch_shape = self._capsule_robot_arrays(coll, is_batched)

            out = collision_world_capsule(caps_soa, ws, wc, wb, wh)  # [B, N, M]
            return out.reshape(*batch_shape, N, M)

    def _compute_self_impl(self, robot, cfg):
        """Self-collision implementation — JIT'd via _ensure_jit."""
        P = len(self._inner.active_idx_i)
        is_batched = jnp.asarray(cfg).ndim > 1

        coll = self._at_config_batched(robot, cfg)

        if isinstance(self._inner, RobotCollisionSpherized):
            centers_soa, radii, S, N, batch_shape = self._sphere_robot_arrays(coll, is_batched)
            B = radii.shape[0]
            K = S * N

            # Self-collision kernel expects AoS [B, S, N, 3] — undo the SoA transpose.
            centers_aoi = jnp.transpose(centers_soa, (1, 2, 0))  # [B, K, 3]
            out = collision_self_sphere(
                centers_aoi.reshape(B, S, N, 3),
                radii.reshape(B, S, N),
                self._pair_i,
                self._pair_j,
            )

        else:  # RobotCollision (capsule-based)
            caps_soa, N, batch_shape = self._capsule_robot_arrays(coll, is_batched)
            B = caps_soa.shape[1]

            # Self-collision kernel expects AoS [B, N, 7] — undo the SoA transpose.
            caps_aoi = jnp.transpose(caps_soa, (1, 2, 0))  # [B, N, 7]
            out = collision_self_capsule(caps_aoi, self._pair_i, self._pair_j)

        return out.reshape(*batch_shape, P)

    # ── Coarse-first implementations (used when coarse_inner is set) ───────

    def _compute_world_impl_coarse_first(self, robot, cfg, ws, wc, wb, wh):
        """Two-phase world collision: coarse guard → fine kernel on collision.

        1. Run FK once for the whole batch.
        2. Apply coarse transforms and run the cheap coarse kernel.
        3. If every coarse distance is positive (collision-free) skip the fine
           kernel entirely and return an all-ones array (+1 SDF value).
        4. Otherwise run the fine kernel and return its result.

        ``jax.lax.cond`` ensures only one branch executes at runtime (no
        differentiability — do not use inside trajopt).
        """
        M = ws.shape[0] + wc.shape[0] + wb.shape[0] + wh.shape[0]
        cfg_arr = jnp.asarray(cfg)
        is_batched = cfg_arr.ndim > 1

        # Single FK call shared by both coarse and fine geometry transforms.
        Ts = robot.forward_kinematics(cfg_arr, use_cuda=True)

        # ── Coarse check ──────────────────────────────────────────────────
        coarse_coll = self._apply_transforms_for(self._coarse_inner, Ts, is_batched)

        if isinstance(self._coarse_inner, RobotCollisionSpherized):
            cc_soa, cr, _, N_c, _ = self._sphere_robot_arrays(coarse_coll, is_batched)
            coarse_out = collision_world_sphere_reduced(cc_soa, cr, ws, wc, wb, wh, n=N_c)
        else:
            cc_caps, _, _ = self._capsule_robot_arrays(coarse_coll, is_batched)
            coarse_out = collision_world_capsule(cc_caps, ws, wc, wb, wh)

        coarse_clear = jnp.all(coarse_out > 0)

        # ── Output shape for fine result ──────────────────────────────────
        N_fine = self._inner.num_links
        B = cfg_arr.shape[0] if is_batched else 1
        batch_shape = (B,) if is_batched else ()

        def _return_clear(_):
            return jnp.ones((B, N_fine, M), dtype=jnp.float32)

        def _run_fine(_):
            fine_coll = self._apply_transforms_for(self._inner, Ts, is_batched)
            if isinstance(self._inner, RobotCollisionSpherized):
                f_soa, f_r, _, N, _ = self._sphere_robot_arrays(fine_coll, is_batched)
                out = collision_world_sphere_reduced(f_soa, f_r, ws, wc, wb, wh, n=N)
            else:
                f_caps, _, _ = self._capsule_robot_arrays(fine_coll, is_batched)
                out = collision_world_capsule(f_caps, ws, wc, wb, wh)
            return out.reshape(B, N_fine, M)

        result = jax.lax.cond(coarse_clear, _return_clear, _run_fine, None)
        return result.reshape(*batch_shape, N_fine, M)

    def _compute_self_impl_coarse_first(self, robot, cfg):
        """Two-phase self-collision: coarse guard → fine kernel on collision.

        Same logic as ``_compute_world_impl_coarse_first`` but for self-pairs.
        If the coarse model has no active self-collision pairs the coarse check
        is trivially skipped and the fine kernel always runs.
        """
        P_fine = len(self._inner.active_idx_i)
        P_coarse = len(self._coarse_inner.active_idx_i)
        cfg_arr = jnp.asarray(cfg)
        is_batched = cfg_arr.ndim > 1

        # Degenerate: coarse model has no pairs — can't act as a guard.
        if P_coarse == 0:
            return self._compute_self_impl(robot, cfg)

        Ts = robot.forward_kinematics(cfg_arr, use_cuda=True)

        # ── Coarse self-collision check ───────────────────────────────────
        coarse_coll = self._apply_transforms_for(self._coarse_inner, Ts, is_batched)

        if isinstance(self._coarse_inner, RobotCollisionSpherized):
            cc_soa, cr, S_c, N_c, _ = self._sphere_robot_arrays(coarse_coll, is_batched)
            B = cr.shape[0]
            cc_aoi = jnp.transpose(cc_soa, (1, 2, 0))
            coarse_out = collision_self_sphere(
                cc_aoi.reshape(B, S_c, N_c, 3),
                cr.reshape(B, S_c, N_c),
                self._coarse_pair_i,
                self._coarse_pair_j,
            )
        else:
            cc_caps, _, _ = self._capsule_robot_arrays(coarse_coll, is_batched)
            B = cc_caps.shape[1]
            cc_aoi = jnp.transpose(cc_caps, (1, 2, 0))
            coarse_out = collision_self_capsule(
                cc_aoi, self._coarse_pair_i, self._coarse_pair_j
            )

        coarse_clear = jnp.all(coarse_out > 0)
        batch_shape = (B,) if is_batched else ()

        def _return_clear(_):
            return jnp.ones((B, P_fine), dtype=jnp.float32)

        def _run_fine(_):
            fine_coll = self._apply_transforms_for(self._inner, Ts, is_batched)
            if isinstance(self._inner, RobotCollisionSpherized):
                f_soa, f_r, S, N, _ = self._sphere_robot_arrays(fine_coll, is_batched)
                f_aoi = jnp.transpose(f_soa, (1, 2, 0))
                out = collision_self_sphere(
                    f_aoi.reshape(B, S, N, 3),
                    f_r.reshape(B, S, N),
                    self._pair_i,
                    self._pair_j,
                )
            else:
                f_caps, _, _ = self._capsule_robot_arrays(fine_coll, is_batched)
                f_aoi = jnp.transpose(f_caps, (1, 2, 0))
                out = collision_self_capsule(f_aoi, self._pair_i, self._pair_j)
            return out.reshape(B, P_fine)

        result = jax.lax.cond(coarse_clear, _return_clear, _run_fine, None)
        return result.reshape(*batch_shape, P_fine)

    # ── Public API ─────────────────────────────────────────────────────────

    def compute_world_collision_distance(
        self,
        robot: "Robot",
        cfg: Float[Array, "*batch_cfg actuated_count"],
        world_geom: CollGeom,
    ) -> Float[Array, "*batch_combined N M"]:
        """Compute signed distances between all robot links and world obstacles.

        Matches the signature of RobotCollision.compute_world_collision_distance.

        On the first call for a given (robot, world_geom) pair, this triggers
        XLA compilation.  Subsequent calls with the same shapes hit the compiled
        cache with near-zero Python overhead.

        Returns:
            Float array of shape (*batch_combined, N, M).
            Positive = separated, negative = penetration.
        """
        self._ensure_world_cache(world_geom)
        self._ensure_jit(robot)
        cfg = jnp.asarray(cfg)

        # The two-phase coarse checker uses lax.cond on a coarse pass and is
        # non-differentiable by design (see make_cuda_checker docstring).
        if self._coarse_inner is not None:
            return self._jit_world(cfg, self._ws, self._wc, self._wb, self._wh)

        # The FFI kernel is opaque to autodiff.  Expose a custom_jvp whose
        # primal is the CUDA distance matrix and whose tangent comes from the
        # differentiable pure-JAX inner model, which computes the same
        # (*batch, N, M) signed-distance quantity.  custom_jvp covers both
        # forward (jax.jvp) and reverse (jax.grad, via transposition) modes.
        @jax.custom_jvp
        def _world_dist(c: Array) -> Array:
            return self._jit_world(c, self._ws, self._wc, self._wb, self._wh)

        @_world_dist.defjvp
        def _world_dist_jvp(primals, tangents):
            (c,) = primals
            (dc,) = tangents
            # The jvp rule is pure JAX (no FFI): the differentiable inner model
            # provides both the primal value and its tangent, so JAX can
            # transpose for reverse mode.  The CUDA kernel is used only on
            # undifferentiated forward calls (the custom_jvp primal function).
            primal_out, tangent_out = jax.jvp(
                lambda q: self._inner.compute_world_collision_distance(
                    robot, q, world_geom
                ),
                (c,),
                (dc,),
            )
            f32 = jnp.float32
            return primal_out.astype(f32), tangent_out.astype(f32)

        return _world_dist(cfg)

    def compute_self_collision_distance(
        self,
        robot: "Robot",
        cfg: Float[Array, "*batch actuated_count"],
    ) -> Float[Array, "*batch num_active_pairs"]:
        """Compute signed distances for active self-collision pairs.

        Matches the signature of RobotCollision.compute_self_collision_distance.

        Returns:
            Float array of shape (*batch, num_active_pairs).
            Positive = separated, negative = penetration.
        """
        self._ensure_jit(robot)
        cfg = jnp.asarray(cfg)

        # Two-phase coarse checker is non-differentiable by design.
        if self._coarse_inner is not None:
            return self._jit_self(cfg)

        # primal = CUDA kernel; tangent = differentiable pure-JAX inner model.
        @jax.custom_jvp
        def _self_dist(c: Array) -> Array:
            return self._jit_self(c)

        @_self_dist.defjvp
        def _self_dist_jvp(primals, tangents):
            (c,) = primals
            (dc,) = tangents
            # Pure-JAX jvp rule (see _world_dist_jvp) so reverse mode transposes.
            primal_out, tangent_out = jax.jvp(
                lambda q: self._inner.compute_self_collision_distance(robot, q),
                (c,),
                (dc,),
            )
            f32 = jnp.float32
            return primal_out.astype(f32), tangent_out.astype(f32)

        return _self_dist(cfg)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_cuda_checker(
    inner: Union[RobotCollision, RobotCollisionSpherized],
    coarse_inner: Optional[Union[RobotCollision, RobotCollisionSpherized]] = None,
) -> CUDADifferentiableSDFCollisionChecker:
    """Wrap a JAX collision model in the differentiable CUDA/JAX-FFI SDF backend.

    Args:
        inner: Fine-resolution collision model (capsule or spherized).
        coarse_inner: Optional coarse collision model for two-phase checking.
            When provided, each call first runs the cheap coarse kernel; if the
            coarse result is collision-free the fine kernel is skipped and an
            all-positive (+1) distance matrix is returned.  This breaks SDF
            differentiability — do not pass this to trajopt.

            NOTE: this ``lax.cond`` guard operates at the *batch* level and
            cannot early-exit individual configurations.  For true per-config
            early-exit binary edge validation use
            :class:`CUDABinaryCollisionChecker` instead.

    Raises ``RuntimeError`` if the compiled library is not found.
    """
    return CUDADifferentiableSDFCollisionChecker(inner, coarse_inner=coarse_inner)


# Backwards-compatible alias: the differentiable SDF checker was historically
# named ``CUDARobotCollisionChecker``.
CUDARobotCollisionChecker = CUDADifferentiableSDFCollisionChecker


# ---------------------------------------------------------------------------
# Binary collision checker (pRRTC-style fused FK + early-exit edge validation)
# ---------------------------------------------------------------------------


def _spherized_local_geometry(model: RobotCollisionSpherized) -> np.ndarray:
    """Extract link-LOCAL sphere geometry from a spherized model.

    Returns a float32 array ``[K, 4]`` of (x, y, z, r) in link-local frames,
    laid out with ``k = s * N + n`` (sphere ``s`` of link ``n``; N = num_links),
    matching the binary kernel's expected layout.  Padding spheres keep their
    radius < 0 sentinel so the kernel skips them.
    """
    # coll has batch axes (N, S): centers [N, S, 3], radii [N, S].
    centers = np.asarray(model.coll.pose.translation(), dtype=np.float32)  # [N, S, 3]
    radii = np.asarray(model.coll.radius, dtype=np.float32)                # [N, S]
    N, S = radii.shape
    local = np.concatenate([centers, radii[..., None]], axis=-1)           # [N, S, 4]
    local = np.transpose(local, (1, 0, 2)).reshape(S * N, 4)               # [S*N, 4]
    return np.ascontiguousarray(local, dtype=np.float32)


class CUDABinaryCollisionChecker:
    """Fused FK + *binary* collision checker for fast edge validation.

    This is the binary counterpart to :class:`CUDADifferentiableSDFCollisionChecker`.
    It mirrors pRRTC's SIMT collision checker (https://github.com/CoMMALab/pRRTC):

      * **Fused FK** — forward kinematics is computed *inside* the collision
        kernel (one CUDA block per configuration); no intermediate sphere-position
        array is written to global memory.
      * **Two-stage approx→fine** — an optional coarse sphere model (one enclosing
        sphere per link) is checked first.  A coarse "clear" proves the fine
        geometry is clear and skips the fine pass; where the coarse model reports
        a possible collision, only the flagged links are re-checked with the fine
        geometry.
      * **Per-config early exit** — the instant any fine sphere is in collision
        the block stops and the configuration is reported invalid.

    Returns a boolean per configuration: ``True`` == collision-free (valid),
    ``False`` == in collision (world OR self).  This is NOT differentiable — it is
    intended for sampling-based planning / edge validation, not trajectory
    optimisation.  Use the SDF checker for anything that needs gradients.

    Only sphere-based models (:class:`RobotCollisionSpherized`) are supported, as
    in pRRTC.

    Soundness requirement: the coarse model must geometrically *enclose* the fine
    model (each coarse sphere covers the fine spheres of its link).  Otherwise the
    coarse guard can produce false negatives (missed collisions).  The
    ``panda_spherized_coarse.urdf`` style of one-enclosing-sphere-per-link models
    satisfy this.

    Usage::

        fine   = RobotCollisionSpherized.from_urdf(urdf, srdf_path=srdf)
        coarse = RobotCollisionSpherized.from_urdf(coarse_urdf, srdf_path=srdf)
        checker = CUDABinaryCollisionChecker(fine, coarse_inner=coarse)
        checker.set_world(world_geom)

        free = checker.check_collision_free(robot, cfg, world_geom)   # bool [*batch]
        ok   = checker.check_edges_collision_free(robot, edge_cfgs, world_geom)
    """

    def __init__(
        self,
        inner: RobotCollisionSpherized,
        coarse_inner: Optional[RobotCollisionSpherized] = None,
    ) -> None:
        _load_and_register_binary()

        if not isinstance(inner, RobotCollisionSpherized):
            raise NotImplementedError(
                "CUDABinaryCollisionChecker only supports sphere-based models "
                "(RobotCollisionSpherized), matching pRRTC.  Got "
                f"{type(inner).__name__}."
            )
        if coarse_inner is not None and not isinstance(
            coarse_inner, RobotCollisionSpherized
        ):
            raise NotImplementedError(
                "coarse_inner must also be a RobotCollisionSpherized; got "
                f"{type(coarse_inner).__name__}."
            )
        if coarse_inner is not None and coarse_inner.num_links != inner.num_links:
            raise ValueError(
                "Fine and coarse models must share the same link set "
                f"(num_links {inner.num_links} != {coarse_inner.num_links}); "
                "the coarse guard indexes flags by link."
            )

        self._inner = inner
        self._coarse_inner = coarse_inner

        # Static link-local sphere geometry (config-independent), device-resident.
        self._f_local = jnp.asarray(_spherized_local_geometry(inner))  # [Kf, 4]
        if coarse_inner is not None:
            self._c_local = jnp.asarray(_spherized_local_geometry(coarse_inner))
            self._c_pair_i = jnp.asarray(coarse_inner.active_idx_i, dtype=jnp.int32)
            self._c_pair_j = jnp.asarray(coarse_inner.active_idx_j, dtype=jnp.int32)
        else:
            # Empty coarse geometry → kernel runs the fine pass directly.
            self._c_local = jnp.zeros((0, 4), dtype=jnp.float32)
            self._c_pair_i = jnp.zeros((0,), dtype=jnp.int32)
            self._c_pair_j = jnp.zeros((0,), dtype=jnp.int32)

        self._f_pair_i = jnp.asarray(inner.active_idx_i, dtype=jnp.int32)
        self._f_pair_j = jnp.asarray(inner.active_idx_j, dtype=jnp.int32)

        # World geometry cache (mirrors the SDF checker).
        self._ws = None
        self._wc = None
        self._wb = None
        self._wh = None
        self._cached_world_id = None

        # JIT cache keyed on robot identity.
        self._cached_robot_id = None
        self._jit_binary = None

        logger.info(
            f"CUDABinaryCollisionChecker wrapping {type(inner).__name__} with "
            f"{inner.num_links} links"
            + (
                f", coarse model {type(coarse_inner).__name__} "
                f"with {coarse_inner.num_links} links."
                if coarse_inner is not None
                else " (no coarse guard — fine pass runs directly)."
            )
        )

    # ── Metadata forwarding ────────────────────────────────────────────────

    @property
    def num_links(self) -> int:
        return self._inner.num_links

    @property
    def link_names(self) -> tuple[str, ...]:
        return self._inner.link_names

    # ── World geometry caching ─────────────────────────────────────────────

    def set_world(self, world_geom: CollGeom) -> None:
        """Pre-upload world geometry to the device (call once, before the loop)."""
        ws_np, wc_np, wb_np, wh_np = _extract_world_arrays(world_geom)
        self._ws = jnp.array(ws_np)
        self._wc = jnp.array(wc_np)
        self._wb = jnp.array(wb_np)
        self._wh = jnp.array(wh_np)
        self._cached_world_id = id(world_geom)

    def _ensure_world_cache(self, world_geom: CollGeom) -> None:
        if id(world_geom) != self._cached_world_id:
            self.set_world(world_geom)

    # ── JIT cache ──────────────────────────────────────────────────────────

    def _ensure_jit(self, robot: "Robot") -> None:
        if id(robot) == self._cached_robot_id:
            return
        _robot = robot

        def _impl(cfg_flat, ws, wc, wb, wh):
            j = _robot.joints
            return collision_binary(
                cfg_flat,
                twists=j.twists,
                parent_tf=j.parent_transforms,
                parent_idx=j.parent_indices,
                act_idx=j.actuated_indices,
                mimic_mul=j.mimic_multiplier,
                mimic_off=j.mimic_offset,
                mimic_act_idx=j.mimic_act_indices,
                topo_inv=j._topo_sort_inv,
                link_parent_joint=_robot.links.parent_joint_indices,
                f_local=self._f_local,
                c_local=self._c_local,
                world_spheres=ws,
                world_capsules=wc,
                world_boxes=wb,
                world_halfspaces=wh,
                f_pair_i=self._f_pair_i,
                f_pair_j=self._f_pair_j,
                c_pair_i=self._c_pair_i,
                c_pair_j=self._c_pair_j,
            )

        self._jit_binary = jax.jit(_impl)
        self._cached_robot_id = id(robot)

    # ── Public API ─────────────────────────────────────────────────────────

    def check_collision_free(
        self,
        robot: "Robot",
        cfg: Float[Array, "*batch actuated_count"],
        world_geom: CollGeom,
    ) -> Array:
        """Return a boolean per configuration: ``True`` if collision-free.

        Checks both world and self collision in a single fused kernel with
        per-config early exit.  Output shape is ``cfg.shape[:-1]`` (the batch
        axes); a single un-batched config returns a scalar boolean array.
        """
        self._ensure_world_cache(world_geom)
        self._ensure_jit(robot)

        cfg = jnp.asarray(cfg)
        batch_axes = cfg.shape[:-1]
        n_act = cfg.shape[-1]
        B = int(np.prod(batch_axes)) if batch_axes else 1
        cfg_flat = cfg.reshape(B, n_act)

        free = self._jit_binary(cfg_flat, self._ws, self._wc, self._wb, self._wh)
        free = free.reshape(batch_axes) if batch_axes else free.reshape(())
        return free != 0

    def check_edges_collision_free(
        self,
        robot: "Robot",
        edge_cfgs: Float[Array, "*batch granularity actuated_count"],
        world_geom: CollGeom,
    ) -> Array:
        """Validate edges: ``True`` if *every* discretised point is collision-free.

        ``edge_cfgs`` holds the already-discretised configurations along each
        edge in its second-to-last axis (granularity G).  Returns a boolean of
        shape ``edge_cfgs.shape[:-2]`` — one verdict per edge (logical AND over
        the G points).
        """
        edge_cfgs = jnp.asarray(edge_cfgs)
        *edge_axes, G, n_act = edge_cfgs.shape
        flat = edge_cfgs.reshape(-1, n_act)
        free = self.check_collision_free(robot, flat, world_geom)
        free = free.reshape(*edge_axes, G)
        return jnp.all(free, axis=-1)


def make_cuda_binary_checker(
    inner: RobotCollisionSpherized,
    coarse_inner: Optional[RobotCollisionSpherized] = None,
) -> CUDABinaryCollisionChecker:
    """Build a pRRTC-style fused FK + binary collision checker.

    Args:
        inner: Fine-resolution spherized collision model.
        coarse_inner: Optional coarse (one-enclosing-sphere-per-link) spherized
            model used as the early-exit guard.  Must enclose ``inner`` and share
            its link set.  When omitted, the fine pass runs directly.

    Raises ``RuntimeError`` if the compiled library is not found.
    """
    return CUDABinaryCollisionChecker(inner, coarse_inner=coarse_inner)
