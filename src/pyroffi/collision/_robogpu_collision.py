"""GPU collision checker using OptiX ray-tracing cores for point-cloud queries.

RoboGPUCollisionChecker implements the RoboGPU architecture (arXiv:2603.01517)
adapted for sphere-based robot models:

  - The environment point cloud is built into an OptiX BVH (each point as a
    sphere of radius ``r_env``).  The BVH is built once and cached across calls
    that use the same point cloud; subsequent checks pay only the traversal cost.

  - Robot collision geometry comes from a :class:`RobotCollisionSpherized`
    model (the same input as :class:`CUDABinaryCollisionChecker`).

  - Per-call execution on a single CUDA stream (fully asynchronous):
      1. CUDA kernel: FK → world-frame robot spheres → regular world geometry
         check (spheres / capsules / boxes / halfspaces) + self-collision.
      2. OptiX kernel: for configs still free after Stage 1, each robot sphere
         queries the point-cloud BVH.  OptiX any-hit terminates BVH traversal
         on the first hit (per-sphere early exit), and the raygen loop breaks
         immediately (per-config early exit).

  - Public API deliberately mirrors both :class:`CUDABinaryCollisionChecker`
    (same constructor argument, same ``check_collision_free`` /
    ``check_edges_collision_free`` signatures) and :class:`VAMPCPUCollisionChecker`
    (same ``set_world`` keyword args for the CAPT-style point-cloud parameters).

Prerequisites:
  bash build_kernels/build_robogpu_collision.sh
  (Requires NVIDIA OptiX SDK 7.x and CUDA 11.2+.)

Note: HalfSpace obstacles are supported (via the CUDA stage only), since OptiX
is used exclusively for the point cloud.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float

from ._geometry import CollGeom
from ._robot_collision import RobotCollisionSpherized
from ._cuda_collision import _extract_world_arrays, _spherized_local_geometry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _r_robot_max(f_local: Array) -> float:
    """Maximum active (non-padding) robot sphere radius."""
    radii = np.asarray(f_local[:, 3])
    active = radii[radii > 0.0]
    return float(active.max()) if active.size > 0 else 0.01


# ---------------------------------------------------------------------------
# RoboGPUCollisionChecker
# ---------------------------------------------------------------------------

class RoboGPUCollisionChecker:
    """OptiX-accelerated sphere-octree collision checker for point-cloud worlds.

    Args:
        inner: Spherized robot collision model (same as CUDABinaryCollisionChecker).
        edge_granularity: Number of interpolation points per edge for
            ``check_edges_collision_free`` (pre-discretised like the CUDA binary
            checker; increase for denser edges).

    Example::

        checker = RoboGPUCollisionChecker(RobotCollisionSpherized.from_urdf(urdf))
        checker.set_world(world_geom, point_cloud=pc, r_env=0.02)
        free = checker.check_collision_free(robot, cfg)   # [B] int32
    """

    def __init__(
        self,
        inner: RobotCollisionSpherized,
        *,
        edge_granularity: int = 16,
    ) -> None:
        from ..cuda_kernels._robogpu_collision_ffi import _load_and_register
        _load_and_register()

        if not isinstance(inner, RobotCollisionSpherized):
            raise TypeError(
                "RoboGPUCollisionChecker requires a RobotCollisionSpherized model; "
                f"got {type(inner).__name__}."
            )

        self._inner = inner
        self._edge_granularity = int(edge_granularity)

        # Robot sphere geometry (link-local, static across configs).
        self._f_local   = jnp.asarray(_spherized_local_geometry(inner))  # [K, 4]
        self._f_pair_i  = jnp.asarray(inner.active_idx_i, dtype=jnp.int32)
        self._f_pair_j  = jnp.asarray(inner.active_idx_j, dtype=jnp.int32)
        self._r_robot_max = _r_robot_max(self._f_local)

        # World geometry cache (updated by set_world or lazily on first call).
        self._ws: Optional[Array] = None
        self._wc: Optional[Array] = None
        self._wb: Optional[Array] = None
        self._wh: Optional[Array] = None
        self._wp: Array = jnp.zeros((0, 3), dtype=jnp.float32)  # point cloud
        self._r_env: float = 0.01
        self._cached_world_id: Optional[int] = None

        # Per-robot JIT cache (keyed by robot object identity).
        self._cached_robot_id: Optional[int] = None
        self._jit_fn = None

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def num_links(self) -> int:
        return self._inner.num_links

    @property
    def link_names(self) -> tuple[str, ...]:
        return self._inner.link_names

    # ── World handling ───────────────────────────────────────────────────────

    def set_world(
        self,
        world_geom: CollGeom,
        point_cloud: Optional[Array] = None,
        *,
        r_env: float = 0.01,
        # CAPT-compatible aliases (ignored — r_env covers all points uniformly)
        capt_r_min: float = 0.0,
        capt_r_max: float = 1.0,
        capt_r_point: float = 0.0,
    ) -> None:
        """Cache world obstacles and optional point cloud.

        Args:
            world_geom: Regular world geometry (spheres, capsules, boxes,
                halfspaces) — checked in CUDA Stage 1.
            point_cloud: ``[Mp, 3]`` float32 array of environment points.
                Pass ``None`` or an empty array to use Stage 1 only.
            r_env: Radius of each environment point sphere.  Also used as
                ``capt_r_point`` equivalent.  All points share the same radius.
        """
        ws_np, wc_np, wb_np, wh_np = _extract_world_arrays(world_geom)
        self._ws = jnp.array(ws_np)
        self._wc = jnp.array(wc_np)
        self._wb = jnp.array(wb_np)
        self._wh = jnp.array(wh_np)
        self._cached_world_id = id(world_geom)

        if point_cloud is not None:
            self._wp = jnp.asarray(point_cloud, dtype=jnp.float32).reshape(-1, 3)
        else:
            self._wp = jnp.zeros((0, 3), dtype=jnp.float32)
        self._r_env = float(r_env if r_env > 0.0 else capt_r_point)

        # Invalidate robot JIT cache so new world args are picked up.
        self._cached_robot_id = None
        self._jit_fn = None

    def _ensure_world(self, world_geom: CollGeom) -> None:
        if id(world_geom) != self._cached_world_id:
            self.set_world(world_geom)

    # ── JIT cache ────────────────────────────────────────────────────────────

    def _ensure_jit(
        self,
        robot,
        world_geom: Optional[CollGeom],
        point_cloud: Optional[Array],
        r_env: float,
    ) -> None:
        """Build and cache a jax.jit'd call for this (robot, world) combination."""
        cache_id = (id(robot), id(world_geom), id(point_cloud), r_env)
        if cache_id == self._cached_robot_id:
            return

        from ..cuda_kernels._robogpu_collision_ffi import robogpu_collision

        _robot = robot
        _f_local   = self._f_local
        _f_pair_i  = self._f_pair_i
        _f_pair_j  = self._f_pair_j
        _r_robot   = self._r_robot_max

        # Resolve world arrays once at JIT build time.
        if world_geom is not None:
            ws_np, wc_np, wb_np, wh_np = _extract_world_arrays(world_geom)
            _ws = jnp.array(ws_np)
            _wc = jnp.array(wc_np)
            _wb = jnp.array(wb_np)
            _wh = jnp.array(wh_np)
        else:
            _ws = self._ws if self._ws is not None else jnp.zeros((0, 4), jnp.float32)
            _wc = self._wc if self._wc is not None else jnp.zeros((0, 7), jnp.float32)
            _wb = self._wb if self._wb is not None else jnp.zeros((0, 15), jnp.float32)
            _wh = self._wh if self._wh is not None else jnp.zeros((0, 6), jnp.float32)

        if point_cloud is not None:
            _pc = jnp.asarray(point_cloud, dtype=jnp.float32).reshape(-1, 3)
            _re = float(r_env)
        else:
            _pc = self._wp
            _re = self._r_env

        def _impl(cfg_flat):
            j = _robot.joints
            return robogpu_collision(
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
                f_local=_f_local,
                f_pair_i=_f_pair_i,
                f_pair_j=_f_pair_j,
                world_spheres=_ws,
                world_capsules=_wc,
                world_boxes=_wb,
                world_halfspaces=_wh,
                point_cloud=_pc,
                r_env=_re,
                r_robot_max=_r_robot,
            )

        self._jit_fn = jax.jit(_impl)
        self._cached_robot_id = cache_id

    # ── Public API ───────────────────────────────────────────────────────────

    def check_collision_free(
        self,
        robot,
        cfg: Float[Array, "*batch actuated_count"],
        world_geom: Optional[CollGeom] = None,
        point_cloud: Optional[Array] = None,
        r_env: float = 0.0,
    ) -> Array:
        """Return ``int32[*batch]``: 1 = collision-free, 0 = in-collision.

        Args:
            robot: Robot model (provides FK joint arrays).
            cfg: Configuration tensor, shape ``[*batch, n_act]``.
            world_geom: Override the cached world geometry for this call.
            point_cloud: Override the cached point cloud for this call
                (``[Mp, 3]`` float32).  Pass ``None`` to use the cached cloud.
            r_env: Override the env sphere radius for this call.
        """
        cfg = jnp.asarray(cfg, dtype=jnp.float32)
        batch_axes = cfg.shape[:-1]
        n_act = cfg.shape[-1]
        B = int(np.prod(batch_axes)) if batch_axes else 1
        cfg_flat = cfg.reshape(B, n_act)

        self._ensure_jit(robot, world_geom, point_cloud,
                         r_env if r_env > 0.0 else self._r_env)
        out = self._jit_fn(cfg_flat)
        return out.reshape(batch_axes) if batch_axes else out.reshape(())

    def check_edges_collision_free(
        self,
        robot,
        edge_cfgs: Float[Array, "*batch granularity actuated_count"],
        world_geom: Optional[CollGeom] = None,
        point_cloud: Optional[Array] = None,
        r_env: float = 0.0,
    ) -> Array:
        """Batch edge validation: ``int32[*batch]`` — 1 if all points free.

        ``edge_cfgs`` has shape ``[*batch, G, n_act]`` where G is the number of
        pre-discretised waypoints along each edge (``edge_granularity``).  The
        result is 1 only when ALL G points are collision-free.

        Unlike :class:`VAMPCPUCollisionChecker`, this checker does NOT
        discretise internally; the caller is responsible for interpolation::

            G = 16
            ts = jnp.linspace(0, 1, G)
            edges = a[:, None, :] * (1 - ts)[None, :, None] + b[:, None, :] * ts[None, :, None]
            free = checker.check_edges_collision_free(robot, edges, world)
        """
        cfg = jnp.asarray(edge_cfgs, dtype=jnp.float32)
        *edge_axes, G, n_act = cfg.shape
        E = int(np.prod(edge_axes)) if edge_axes else 1

        # Flatten [E, G, n_act] → [E*G, n_act], check all at once, then AND.
        cfg_flat = cfg.reshape(E * G, n_act)
        self._ensure_jit(robot, world_geom, point_cloud,
                         r_env if r_env > 0.0 else self._r_env)
        out_flat = self._jit_fn(cfg_flat)  # [E*G] int32
        # A config is free iff ALL G waypoints are free: min over the G axis.
        out_edges = out_flat.reshape(E, G).min(axis=1)  # [E] int32
        return out_edges.reshape(tuple(edge_axes)) if edge_axes else out_edges.reshape(())


def make_robogpu_checker(
    inner: RobotCollisionSpherized,
    **kwargs,
) -> RoboGPUCollisionChecker:
    """Convenience factory for :class:`RoboGPUCollisionChecker`."""
    return RoboGPUCollisionChecker(inner, **kwargs)
