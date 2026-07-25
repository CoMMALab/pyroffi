"""Session: the long-lived, warm, transport-agnostic home for a robot + scene.

pyroffi's cost model is inverted relative to a classical planning library. The
first trajopt call is dominated by XLA compilation — tens of seconds — while
steady-state solves are milliseconds.  A process-per-request server would pay
the compile every time and be useless, so the session is the unit of reuse:
one robot, one collision model, one scene, one set of warm jitted callables.

Two consequences shape everything here.

**Shapes are static.** Batch size, path length and obstacle *count* all feed
XLA's cache key. The scene handles obstacle count (see :mod:`._scene`); this
module handles the rest by bucketing path lengths and by tracking which shape
signatures have been compiled, so every response can honestly say whether the
caller just paid for a compile.

**Device and precision are process-level.** Both must be pinned before JAX
initialises its backend, which is why :func:`configure_process` exists
separately from the ``Session`` constructor.
"""

from __future__ import annotations

import dataclasses
import os
import sys
import time
from typing import Any, Callable, Iterator

import numpy as np
from loguru import logger

DEFAULT_PATH_BUCKETS: tuple[int, ...] = (16, 32, 64, 128)
"""Path lengths the toolbox will compile for. Anything else is padded up to the
next bucket, so an agent handing over a 47-waypoint RRT output reuses the
64-waypoint program instead of triggering a fresh 40-second compile."""


def configure_process(
    gpu: int | None = None,
    x64: bool = True,
    preallocate: bool = False,
) -> dict[str, Any]:
    """Pin GPU and precision for this process. Call before touching JAX.

    Args:
        gpu: Physical GPU index to expose, via ``CUDA_VISIBLE_DEVICES``. ``None``
            leaves the current selection alone. On a shared box, pick a free
            device — a long-lived server pins its memory for its whole lifetime.
        x64: Enable ``jax_enable_x64``. Recommended: several IK and trajopt
            paths in pyroffi are precision-sensitive, and running float32
            silently gives worse solutions than the examples do.
        preallocate: Leave JAX's greedy pre-allocation on. Off by default so the
            server coexists with other jobs on the same device.

    Returns:
        What was actually applied, including whether the request came too late
        to take effect (``late=True`` — the backend was already initialised).
    """
    late = _jax_backend_initialised()

    if gpu is not None and not late:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if not preallocate:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax

    if x64 != bool(jax.config.read("jax_enable_x64")):
        jax.config.update("jax_enable_x64", x64)

    if late and gpu is not None:
        logger.warning(
            "configure_process(gpu=...) was called after JAX initialised its "
            "backend; device selection had no effect. Set CUDA_VISIBLE_DEVICES "
            "before importing pyroffi, or start the server with --gpu."
        )

    return {
        "gpu_requested": gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "x64": bool(jax.config.read("jax_enable_x64")),
        "preallocate": preallocate,
        "late": late,
    }


def _jax_backend_initialised() -> bool:
    if "jax" not in sys.modules:
        return False
    try:  # private API; a wrong answer here only costs a warning
        from jax._src import xla_bridge

        return bool(xla_bridge._backends)
    except Exception:  # pragma: no cover - defensive
        return False


# ── robot loading ────────────────────────────────────────────────────────────

def _repo_resource(*parts: str) -> str:
    """Path to a file under this checkout's ``resources/`` directory."""
    here = os.path.dirname(os.path.abspath(__file__))          # src/pyroffi/toolbox
    root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    return os.path.join(root, "resources", *parts)


_ROBOT_ALIASES: dict[str, str] = {
    "panda": "panda_description",
    "panda_spherized": _repo_resource("panda", "panda_spherized.urdf"),
    "ur5": "ur5_description",
    "ur10": "ur10_description",
    "iiwa": "iiwa14_description",
    "iiwa14": "iiwa14_description",
    "yumi": "yumi_description",
}


def load_urdf(spec: str):
    """Resolve a robot spec to a ``yourdfpy.URDF``.

    Accepts a filesystem path to a URDF, a ``robot_descriptions`` name
    (``panda_description``), or a short alias (``panda``).
    """
    import yourdfpy

    spec = _ROBOT_ALIASES.get(spec, spec)

    if spec.endswith(".urdf"):
        path = os.path.abspath(spec)
        if not os.path.exists(path):
            raise FileNotFoundError(f"no URDF at {path}")
        mesh_dir = os.path.join(os.path.dirname(path), "meshes")
        if os.path.isdir(mesh_dir):
            return yourdfpy.URDF.load(path, mesh_dir=mesh_dir)
        return yourdfpy.URDF.load(path)

    from robot_descriptions.loaders.yourdfpy import load_robot_description

    return load_robot_description(spec)


def _build_collision_model(urdf, requested: str = "auto"):
    """Build the robot collision model, returning ``(model, name)``.

    ``"spherized"`` decomposes each link into spheres and is markedly more
    faithful for self-collision, but it only works on URDFs whose collision
    geometry is primitive-based — it raises on mesh collision geometry. ``"auto"``
    therefore tries it and falls back to one capsule per link, saying so, rather
    than either failing on ordinary mesh URDFs or silently using the coarse model
    on a URDF that deserves better.
    """
    from pyroffi.collision import RobotCollision, RobotCollisionSpherized

    if requested not in ("auto", "capsule", "spherized"):
        raise ValueError(
            f"unknown collision_model {requested!r}; expected 'auto', 'capsule' "
            "or 'spherized'"
        )

    if requested in ("auto", "spherized"):
        try:
            return RobotCollisionSpherized.from_urdf(urdf), "spherized"
        except Exception as exc:
            if requested == "spherized":
                raise RuntimeError(
                    "collision_model='spherized' needs primitive (non-mesh) collision "
                    f"geometry in the URDF: {exc}"
                ) from exc
            logger.info(
                "spherized collision model unavailable for this URDF "
                f"({type(exc).__name__}); falling back to one capsule per link, which "
                "is coarser for self-collision"
            )

    return RobotCollision.from_urdf(urdf), "capsule"


# ── compile bookkeeping ──────────────────────────────────────────────────────


@dataclasses.dataclass
class CompileLedger:
    """Which shape signatures this process has already compiled.

    ``compiled: bool`` on every response comes from here.  It is what lets an
    agent tell a 40-second answer from a 4-millisecond one, instead of
    concluding the library is slow.
    """

    seen: set[str] = dataclasses.field(default_factory=set)

    def visit(self, key: str) -> bool:
        """Record *key*; return True if this is its first use (i.e. a compile)."""
        first = key not in self.seen
        self.seen.add(key)
        return first

    def __contains__(self, key: object) -> bool:
        return key in self.seen


def bucket_length(n: int, buckets: tuple[int, ...] = DEFAULT_PATH_BUCKETS) -> int:
    """Smallest bucket that fits *n* waypoints (the largest bucket if none does).

    Overflowing the largest bucket is allowed but returns *n* itself, which will
    compile a one-off program — reported, never hidden.
    """
    if n < 1:
        raise ValueError("a path needs at least one waypoint")
    for b in buckets:
        if n <= b:
            return b
    return n


def pad_path(path: np.ndarray, target: int) -> np.ndarray:
    """Pad a ``(T, DOF)`` path up to *target* waypoints by repeating the last one.

    A repeated waypoint is a zero-length edge: always collision-consistent with
    its neighbour and always within limits, so padding cannot manufacture a
    validity result. Reporting still trims back to the true length.
    """
    path = np.asarray(path, dtype=np.float64)
    if path.shape[0] > target:
        raise ValueError(f"path of {path.shape[0]} waypoints exceeds target {target}")
    if path.shape[0] == target:
        return path
    tail = np.repeat(path[-1:], target - path.shape[0], axis=0)
    return np.concatenate([path, tail], axis=0)


# ── the session ──────────────────────────────────────────────────────────────


class Session:
    """A warm robot + collision model + scene, with jitted callables cached.

    Not thread-safe by design: JAX dispatch and the scene are both mutable
    shared state, and an MCP session is logically single-threaded (the agent
    shares state with itself across calls). The server serialises calls.
    """

    def __init__(
        self,
        robot: str = "panda",
        max_objects: int = 16,
        n_timesteps: int = 64,
        ee_link: str | None = None,
        ground_plane: bool = True,
        path_buckets: tuple[int, ...] = DEFAULT_PATH_BUCKETS,
        acceleration_scale: float = 2.0,
        session_id: str = "default",
        collision_model: str = "auto",
        calibrate_self_collision: bool = True,
    ) -> None:
        import jax
        import pyroffi as pk

        from ._handles import HandleTable
        from ._scene import Scene

        self.session_id = session_id
        self.robot_spec = robot
        self.urdf = load_urdf(robot)
        self.robot = pk.Robot.from_urdf(self.urdf)
        self.robot_coll, self.collision_model = _build_collision_model(
            self.urdf, collision_model
        )
        self.scene = Scene(max_objects=max_objects, ground_plane=ground_plane)
        self.handles = HandleTable()
        self.ledger = CompileLedger()
        self.path_buckets = tuple(sorted(path_buckets))
        self.n_timesteps = int(n_timesteps)
        self.created_at = time.time()

        self.joint_names: tuple[str, ...] = self.robot.joints.actuated_names
        self.dof = len(self.joint_names)
        self.link_names: tuple[str, ...] = self.robot.links.names
        self.lower_limits = np.asarray(self.robot.joints.lower_limits, dtype=np.float64)
        self.upper_limits = np.asarray(self.robot.joints.upper_limits, dtype=np.float64)
        self.velocity_limits = np.asarray(
            self.robot.joints.velocity_limits, dtype=np.float64
        )
        self.acceleration_limits = self.velocity_limits * float(acceleration_scale)

        self.ee_link = ee_link or self._guess_ee_link()
        self.robot_state = np.asarray(self.robot.default_cfg, dtype=np.float64)

        self.device = jax.devices()[0]
        self.x64 = bool(jax.config.read("jax_enable_x64"))
        self._jitted: dict[str, Callable] = {}

        self.static_link_indices = self._detect_static_links()
        self.static_link_names = tuple(
            self.link_names[i] for i in self.static_link_indices
        )

        self.self_collision_report: dict[str, Any] = {"calibrated": False}
        if calibrate_self_collision:
            self.self_collision_report = self._calibrate_self_collision()

        logger.info(
            f"session {session_id!r}: {robot} ({self.dof} DOF), ee={self.ee_link}, "
            f"device={self.device}, x64={self.x64}, max_objects={max_objects}"
        )

    # ── calibration ───────────────────────────────────────────────────────

    def _detect_static_links(self, n_samples: int = 8, seed: int = 1) -> tuple[int, ...]:
        """Links whose world pose does not depend on the configuration.

        Typically the base link and anything rigidly attached ahead of the first
        actuated joint. They matter because a *bolted-down* link cannot be moved
        out of the way: a base link that intersects the floor plane (which is
        normal — the mounting plate sits at z=0) would otherwise report a
        world collision in every single configuration, and no motion the agent
        chooses could ever clear it.

        Excluding them keeps ``collision_free`` actionable. A world object that
        does intersect a static link is a scene-authoring mistake rather than a
        motion decision, and is flagged at ``add_object`` time instead.
        """
        import numpy as _np

        rng = _np.random.default_rng(seed)
        q = rng.uniform(self.lower_limits, self.upper_limits, size=(n_samples, self.dof))
        try:
            poses = _np.asarray(
                [
                    _np.asarray(self.robot.forward_kinematics(self.as_array(qi)))
                    for qi in q
                ]
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"static-link detection failed: {exc}")
            return ()
        spread = _np.abs(poses - poses[:1]).max(axis=(0, 2))    # per link
        return tuple(int(i) for i in _np.argwhere(spread < 1e-9).reshape(-1))

    def world_link_mask(self):
        """``(n_links,)`` bool — True for links whose world collisions are worth
        reporting (i.e. everything the configuration can actually move)."""
        from jax import numpy as jnp

        mask = np.ones(len(self.link_names), dtype=bool)
        for i in self.static_link_indices:
            mask[i] = False
        return jnp.asarray(mask)

    # ── self-collision calibration ────────────────────────────────────────

    def _calibrate_self_collision(
        self, n_samples: int = 512, threshold: float = 0.99, seed: int = 0
    ) -> dict[str, Any]:
        """Disable link pairs that are in collision in *every* sampled configuration.

        A pair that never separates across the whole joint range is a property of
        the collision *model*, not of any configuration: two link geometries that
        overlap by construction, which a hand-written SRDF would have listed under
        ``disable_collisions``. Left in, they make ``check_collision`` answer
        "false" for every configuration, and the validation half of the toolbox
        becomes worthless.

        This is the empirical version of what MoveIt's setup assistant does. It
        costs one batched GPU call at session creation.

        The residual false-positive rate is *measured* and reported rather than
        assumed: a coarse whole-link-capsule model still over-reports after
        pruning, and the caller deserves to know that before trusting a
        ``collision_free`` answer.
        """
        import dataclasses as _dc

        import jax
        import numpy as _np

        rng = _np.random.default_rng(seed)
        q = rng.uniform(self.lower_limits, self.upper_limits, size=(n_samples, self.dof))

        try:
            fn = jax.jit(
                jax.vmap(
                    lambda c: self.robot_coll.compute_self_collision_distance(
                        self.robot, c
                    )
                )
            )
            dists = np.asarray(fn(self.as_array(q)), dtype=np.float64)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"self-collision calibration failed: {exc}")
            return {"calibrated": False, "error": str(exc)}

        frac_negative = (dists < 0.0).mean(axis=0)
        keep = frac_negative <= threshold
        pair_names = self.self_pair_names()
        pruned = [
            {"link_a": a, "link_b": b, "frac_configs_colliding": round(float(f), 4)}
            for (a, b), f, k in zip(pair_names, frac_negative, keep)
            if not k
        ]

        if pruned:
            idx = np.asarray(keep)
            self.robot_coll = _dc.replace(
                self.robot_coll,
                active_idx_i=self.robot_coll.active_idx_i[idx],
                active_idx_j=self.robot_coll.active_idx_j[idx],
            )
            # The pruned pair set changes the distance vector's length, so any
            # program compiled against the old one must go.
            self._jitted.clear()

        residual = float((dists[:, keep].min(axis=-1) < 0.0).mean()) if keep.any() else 0.0
        reliable = residual < 0.5
        report = {
            "calibrated": True,
            "n_samples": n_samples,
            "threshold": threshold,
            "n_pairs_before": len(pair_names),
            "n_pairs_after": int(keep.sum()),
            "pruned_pairs": pruned,
            "frac_random_configs_self_colliding": round(residual, 4),
            "reliable": reliable,
        }
        if not reliable:
            report["note"] = (
                f"after pruning, {residual:.0%} of uniformly-sampled configurations "
                f"still report self-collision under the {self.collision_model} model. "
                "That is a coarse-geometry artifact, not a property of the robot: "
                "treat self-collision results as advisory and prefer a URDF with "
                "primitive collision geometry (collision_model='spherized')."
            )
            logger.warning(report["note"])
        return report

    # ── introspection ─────────────────────────────────────────────────────

    def _guess_ee_link(self) -> str:
        """Pick a plausible end-effector link when the caller didn't name one."""
        for candidate in ("hand", "tool", "gripper", "ee", "tcp", "flange"):
            for name in self.link_names:
                if candidate in name.lower():
                    return name
        return self.link_names[-1]

    def link_index(self, name: str) -> int:
        try:
            return self.link_names.index(name)
        except ValueError:
            raise ValueError(
                f"unknown link {name!r}; this robot has {list(self.link_names)}"
            ) from None

    def capabilities(self) -> dict[str, Any]:
        """Everything an agent needs to plan its calls, reported once up front."""
        import jax

        return {
            "session_id": self.session_id,
            "robot": self.robot_spec,
            "dof": self.dof,
            "joint_names": list(self.joint_names),
            "joint_limits": {
                name: [float(lo), float(hi)]
                for name, lo, hi in zip(
                    self.joint_names, self.lower_limits, self.upper_limits
                )
            },
            "velocity_limits": {
                name: float(v)
                for name, v in zip(self.joint_names, self.velocity_limits)
            },
            "acceleration_limits": {
                name: float(a)
                for name, a in zip(self.joint_names, self.acceleration_limits)
            },
            "ee_link": self.ee_link,
            "link_names": list(self.link_names),
            "collision_model": self.collision_model,
            "static_links_excluded_from_world_collision": list(self.static_link_names),
            "n_self_collision_pairs": int(self.robot_coll.active_idx_i.shape[0]),
            "self_collision_calibration": self.self_collision_report,
            "max_objects": self.scene.max_objects,
            "n_timesteps": self.n_timesteps,
            "path_buckets": list(self.path_buckets),
            "scene_version": self.scene.version,
            "x64": self.x64,
            "device": str(self.device),
            "devices": [str(d) for d in jax.devices()],
            "backends": self.available_backends(),
            "ik_solvers": ["hjcd", "ls"],
            "units": {"length": "m", "angle": "rad", "time": "s"},
            "quaternion_convention": "wxyz",
        }

    def available_backends(self) -> dict[str, bool]:
        """Which accelerated backends this build can actually reach.

        Probed rather than assumed: the CUDA kernels and GRiD are built
        out-of-tree, so a wheel that imports fine may still have none of them.
        """
        import jax

        out = {"jax": True, "cuda_device": any(d.platform == "gpu" for d in jax.devices())}
        for name, module in (
            ("cuda_collision", "pyroffi.cuda_kernels._collision_cuda_ffi"),
            ("cuda_fk", "pyroffi.cuda_kernels._fk_cuda"),
            ("vamp", "vamp"),
        ):
            try:
                __import__(module)
                out[name] = True
            except Exception:
                out[name] = False
        try:
            out["grid_dynamics"] = self.robot.dynamics is not None
        except Exception:
            out["grid_dynamics"] = False
        return out

    # ── warm jitted callables ─────────────────────────────────────────────

    def jitted(self, key: str, build: Callable[[], Callable]) -> tuple[Callable, bool]:
        """Fetch a jitted callable by cache key, building it on first use.

        Returns ``(fn, first_use)``. ``first_use`` is the honest ``compiled``
        flag: on that call the caller pays tracing plus XLA compilation.
        """
        first = key not in self._jitted
        if first:
            self._jitted[key] = build()
        self.ledger.visit(key)
        return self._jitted[key], first

    def world_geoms(self) -> tuple:
        return self.scene.world_geoms()

    # ── device / precision helpers ────────────────────────────────────────

    def as_array(self, values: Any):
        """Host array → device array in the session's working precision."""
        from jax import numpy as jnp

        dtype = jnp.float64 if self.x64 else jnp.float32
        return jnp.asarray(np.asarray(values), dtype=dtype)

    def clip_to_limits(self, cfg: np.ndarray) -> np.ndarray:
        return np.clip(cfg, self.lower_limits, self.upper_limits)

    def limit_violations(self, cfg: np.ndarray) -> list[dict[str, Any]]:
        """Per-joint limit violations, named. Empty when the config is inside."""
        cfg = np.asarray(cfg, dtype=np.float64).reshape(-1)
        out = []
        for i, name in enumerate(self.joint_names):
            lo, hi = self.lower_limits[i], self.upper_limits[i]
            if cfg[i] < lo or cfg[i] > hi:
                out.append(
                    {
                        "joint": name,
                        "value": float(cfg[i]),
                        "limits": [float(lo), float(hi)],
                        "excess_rad": float(max(lo - cfg[i], cfg[i] - hi)),
                    }
                )
        return out

    def self_pair_names(self) -> list[tuple[str, str]]:
        """Named link pairs, aligned with the self-collision distance vector."""
        idx_i = np.asarray(self.robot_coll.active_idx_i)
        idx_j = np.asarray(self.robot_coll.active_idx_j)
        return [
            (self.link_names[int(i)], self.link_names[int(j)])
            for i, j in zip(idx_i, idx_j)
        ]

    def iter_world_slots(self) -> Iterator[tuple[int, int, str | None]]:
        """``(pool_index, slot, object_name)`` over every world pool slot."""
        for pool_idx, (_shape, names) in enumerate(self.scene.geom_names()):
            for slot, name in enumerate(names):
                yield pool_idx, slot, name

    def close(self) -> None:
        self.handles.clear()
        self._jitted.clear()
