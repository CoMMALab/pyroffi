"""CPU collision checking via VAMP, JIT-compiled per robot through cricket.

This is the CPU counterpart to :class:`CUDABinaryCollisionChecker`.  Where the
CUDA checker fuses a hand-written SIMT kernel, this checker reuses VAMP's
heavily-optimised SIMD ``fkcc`` collision routine
(https://github.com/KavrakiLab/vamp) and specialises it to a concrete robot at
*runtime*:

  1. When the checker is constructed (i.e. as soon as the robot is defined),
     cricket parses the URDF and emits a ``vamp::robots::<Robot>`` C++ struct
     (traced forward kinematics + spherized collision).
  2. That struct is stitched into a tiny translation unit
     (``_robot_edge_validation_tu.cc.in``) that instantiates the JAX FFI
     handlers in ``_edge_validation_ffi.hh`` for the robot.
  3. cricket's JIT (LLVM ORC) compiles the TU once, caching the compiled object
     on disk keyed by a content hash so subsequent constructions for the same
     robot reuse the cached binary.
  4. The resulting XLA FFI custom-call handlers are registered with JAX and
     invoked from :meth:`check_collision_free` / :meth:`check_edges_collision_free`.

The public method surface intentionally mirrors
:class:`CUDABinaryCollisionChecker` so call sites are interchangeable; the
constructor differs because cricket needs the URDF/SRDF (not a pre-spherized
pyroffi model).

Note: VAMP's edge validation supports point-cloud obstacles through its CAPT
(Collision-Affording Point Tree).  Pass ``point_cloud=`` (an ``[Mp, 3]`` array)
to :meth:`set_world` / the check methods to enable it.  HalfSpace obstacles are
not supported by the VAMP backend (use a large flat Box instead).
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float

from ._geometry import CollGeom
from ._cuda_collision import _extract_world_arrays, _extract_world_arrays_jax

_KERNELS_DIR = Path(__file__).resolve().parent.parent / "vamp_kernels"
_FFI_HEADER = _KERNELS_DIR / "_edge_validation_ffi.hh"
_TU_TEMPLATE = _KERNELS_DIR / "_robot_edge_validation_tu.cc.in"

# Repo-relative default include roots for the JIT.  These can be overridden via
# the ``include_dirs`` constructor argument (e.g. when vamp's CPM dependencies
# live somewhere else).  vamp's transitive header-only deps (pdqsort,
# SIMDxorshift, nigh) are fetched into vamp's build ``.cpm-cache`` — point at
# them with ``include_dirs`` if header resolution fails.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_VAMP_INCLUDE = _REPO_ROOT / "external" / "vamp" / "src" / "impl"


@lru_cache(maxsize=1)
def _xla_ffi_include() -> str:
    """Locate the XLA FFI headers shipped with the installed jaxlib."""
    import jaxlib

    root = Path(jaxlib.__file__).resolve().parent / "include"
    if not (root / "xla" / "ffi" / "api" / "ffi.h").exists():
        raise RuntimeError(
            f"Could not find xla/ffi/api/ffi.h under {root}. "
            "Pass include_dirs=[...] pointing at the XLA FFI headers."
        )
    return str(root)


@lru_cache(maxsize=1)
def _cricket_jit():
    """Import cricket's JIT submodule, with a helpful error if unavailable."""
    try:
        import cricket  # noqa: F401
        from cricket import _core_ext

        jit = getattr(_core_ext, "jit", None)
        if jit is None:
            raise RuntimeError(
                "cricket was built without JIT support "
                "(reconfigure with -DCRICKET_BUILD_JIT=ON)."
            )
        return cricket, jit
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "cricket is not importable. Build it from external/cricket with the "
            "Python extension and JIT enabled (CRICKET_BUILD_PYTHON=ON, "
            "CRICKET_BUILD_JIT=ON)."
        ) from exc


def _robot_name_from_urdf(urdf_path: Path) -> str:
    """Derive a valid C++ struct name from the URDF ``<robot name=...>`` tag.

    Falls back to the file stem.  The result is sanitised to a valid identifier
    and capitalised (e.g. ``panda`` -> ``Panda``).
    """
    import re

    name = None
    try:
        text = urdf_path.read_text()
        m = re.search(r"<robot[^>]*\bname\s*=\s*\"([^\"]+)\"", text)
        if m:
            name = m.group(1)
    except OSError:
        pass
    if not name:
        name = urdf_path.stem
    name = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    if name and name[0].isdigit():
        name = "_" + name
    return name[:1].upper() + name[1:] if name else "Robot"


def _find_pdqsort_dir() -> Optional[str]:
    """Locate a directory containing ``pdqsort.h`` (vamp's CAPT dependency).

    vamp fetches pdqsort via CPM into its build tree, so search a few likely
    roots: an explicit ``$VAMP_PDQSORT_DIR``, then any vamp ``_deps`` build
    cache under the user's work tree.
    """
    env = __import__("os").environ.get("VAMP_PDQSORT_DIR")
    if env and (Path(env) / "pdqsort.h").exists():
        return env
    for base in (_REPO_ROOT.parent, _REPO_ROOT, Path.home() / "Work"):
        hits = list(base.glob("**/_deps/pdqsort-src/pdqsort.h"))
        if hits:
            return str(hits[0].parent)
    return None


def _default_include_dirs() -> list[str]:
    import os

    dirs = [str(_DEFAULT_VAMP_INCLUDE), _xla_ffi_include()]
    conda = os.environ.get("CONDA_PREFIX")
    for cand in (
        os.path.join(conda, "include", "eigen3") if conda else None,
        os.path.join(conda, "include") if conda else None,
        "/usr/include/eigen3",
        "/usr/local/include/eigen3",
    ):
        if cand and Path(cand).exists():
            dirs.append(cand)
    pdq = _find_pdqsort_dir()
    if pdq:
        dirs.append(pdq)
    return dirs


@lru_cache(maxsize=1)
def _preload_runtime_libs() -> None:
    """Load libstdc++ and the OpenMP runtime with RTLD_GLOBAL.

    The cricket JIT resolves external symbols via ``dlsym(RTLD_DEFAULT)``, so the
    C++ standard library and OpenMP runtime that the JIT-compiled collision code
    calls into must be present in the global symbol scope.  We load them from the
    active conda prefix when available, else by soname.
    """
    import ctypes
    import os

    candidates = ["libstdc++.so.6", "libomp.so"]
    conda = os.environ.get("CONDA_PREFIX")
    libdir = Path(conda) / "lib" if conda else None
    for name in candidates:
        loaded = False
        if libdir is not None and (libdir / name).exists():
            try:
                ctypes.CDLL(str(libdir / name), mode=ctypes.RTLD_GLOBAL)
                loaded = True
            except OSError:
                pass
        if not loaded:
            try:
                ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass  # surfaced later as a clear "Symbols not found" JIT error


# Registered (target_name) per (robot hash, kind) so we only register once.
_REGISTERED: dict[str, str] = {}


def _is_traced(x) -> bool:
    """True if ``x`` (or any leaf of a CollGeom's arrays) is a JAX tracer.

    Used to pick between the eager host path (concrete arrays → device_put →
    cached FFI call) and the fully-traceable path (JAX-native world extraction →
    ffi_call on the traced arrays), so the VAMP checker can be called from inside
    jax.jit / vmap / pmap / scan.
    """
    if isinstance(x, jax.core.Tracer):
        return True
    # CollGeom: probe its pose (present on every supported primitive).
    pose = getattr(x, "pose", None)
    if pose is not None and isinstance(getattr(pose, "wxyz_xyz", None), jax.core.Tracer):
        return True
    return False


class VAMPCPUCollisionChecker:
    """JIT-compiled VAMP CPU collision checker with batch edge validation.

    Args:
        urdf_path: Path to the robot URDF (cricket parses it to emit FK + CC).
        srdf_path: Optional SRDF for self-collision pairs.
        end_effector: Optional end-effector link name (forwarded to cricket).
        cache_dir: On-disk JIT object cache directory (defaults to cricket's).
        include_dirs: Extra ``-I`` include roots for the JIT compile.  Appended
            to the auto-discovered vamp / Eigen / XLA-FFI roots.
        extra_flags: Extra clang flags (e.g. ``["-march=native", "-fopenmp"]``).
    """

    def __init__(
        self,
        urdf_path: str | Path,
        srdf_path: Optional[str | Path] = None,
        end_effector: Optional[str] = None,
        *,
        robot_name: Optional[str] = None,
        resolution: int = 32,
        cache_dir: Optional[str | Path] = None,
        include_dirs: Optional[list[str]] = None,
        extra_flags: Optional[list[str]] = None,
    ) -> None:
        cricket, jit = _cricket_jit()

        urdf_path = Path(urdf_path).resolve()
        srdf_path = Path(srdf_path).resolve() if srdf_path is not None else None

        # cricket's template needs a C++ struct `name` and a planning
        # `resolution`; neither is in RobotInfo.json(), so supply them via data.
        if robot_name is None:
            robot_name = _robot_name_from_urdf(urdf_path)

        # 1. Codegen: URDF -> vamp::robots::<Robot> source.
        gen = cricket.generate_robot_source(
            cricket.GenOptions(
                urdf=urdf_path,
                srdf=srdf_path,
                end_effector=end_effector,
                data={"name": robot_name, "resolution": int(resolution)},
            )
        )
        robot_type_name = gen.robot_name           # struct name, e.g. "Panda"
        robot_token = robot_type_name.lower()      # symbol suffix, e.g. "panda"
        self._dimension = int(gen.dimension)
        self._n_spheres = int(gen.n_spheres)

        # 2. Materialise the generated header so the TU can #include it, keyed by
        #    a content hash for cache stability.  The hash folds in the generated
        #    robot source AND the FFI handler header + TU template contents, so an
        #    edit to either busts the on-disk object cache (which otherwise only
        #    sees the TU string, not the headers it #includes by path).
        digest = hashlib.sha1()
        digest.update(gen.source.encode())
        digest.update(_FFI_HEADER.read_bytes())
        digest.update(_TU_TEMPLATE.read_bytes())
        src_hash = digest.hexdigest()[:16]
        work_dir = Path(cache_dir) if cache_dir is not None else Path(jit.default_cache_dir())
        work_dir.mkdir(parents=True, exist_ok=True)
        header_path = work_dir / f"vamp_robot_{robot_token}_{src_hash}.hh"
        if not header_path.exists():
            header_path.write_text(gen.source)

        # 3. Build the per-robot translation unit from the template.
        tu_source = (
            _TU_TEMPLATE.read_text()
            .replace("@ROBOT_HEADER@", str(header_path))
            .replace("@FFI_HEADER@", str(_FFI_HEADER))
            .replace("@ROBOT_TYPE@", f"vamp::robots::{robot_type_name}")
            .replace("@ROBOT_NAME@", robot_token)
        )

        # 4. JIT compile (object-cached on disk) and register the FFI handlers.
        opts = jit.CompileOptions()
        opts.std_flag = "-std=c++17"
        opts.opt_flag = "-O3"
        dirs = _default_include_dirs()
        if include_dirs:
            dirs.extend(include_dirs)
        opts.include_dirs = dirs
        opts.extra_flags = extra_flags or ["-march=native", "-fopenmp"]
        opts.module_id = f"vamp_{robot_token}_{src_hash}"

        # Cache the (source, opts) hash so re-registration is skipped.
        self._key = jit.hash_source(tu_source, opts)
        self._configs_target = f"vamp_configs_{robot_token}_{src_hash}"
        self._edges_target = f"vamp_edges_{robot_token}_{src_hash}"

        if self._key not in _REGISTERED:
            _preload_runtime_libs()
            session = jit.JitSession(work_dir)
            # Reuse a previously JIT-compiled binary for this robot if present,
            # skipping the (expensive) clang front-end; else compile + cache it.
            if not session.try_load_cached(opts.module_id):
                session.add_source(tu_source, opts)
            jax.ffi.register_ffi_target(
                self._configs_target,
                session.handler_capsule("pyroffi_get_validate_configs"),
                platform="cpu",
            )
            jax.ffi.register_ffi_target(
                self._edges_target,
                session.handler_capsule("pyroffi_get_validate_edges"),
                platform="cpu",
            )
            # Keep the session alive for the process lifetime: the JIT-owned code
            # must outlive every FFI call into it.
            _REGISTERED[self._key] = self._configs_target
            self._session = session
        else:
            self._session = None

        # World geometry cache (mirrors the CUDA checkers).
        self._ws = np.zeros((0, 4), dtype=np.float32)
        self._wc = np.zeros((0, 7), dtype=np.float32)
        self._wb = np.zeros((0, 15), dtype=np.float32)
        self._wp = np.zeros((0, 3), dtype=np.float32)
        self._capt = (0.0, 0.0, 0.0)

        # Per-call caches so repeated checks against the same world are cheap:
        #   * _world_cache: CPU-resident obstacle buffers, keyed by object id +
        #     capt, so we skip the (GPU->host) CollGeom re-extraction each call;
        #   * _jit_cache: jax.jit'd FFI calls keyed by (kind, capt) so the kernel
        #     runs jitted (~tens of us) instead of via slow eager dispatch.
        self._world_cache_key = None
        self._world_cache = None
        self._jit_cache: dict = {}

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def n_spheres(self) -> int:
        return self._n_spheres

    # ── World handling ──────────────────────────────────────────────────────

    def set_world(
        self,
        world_geom: CollGeom,
        point_cloud: Optional[Array] = None,
        *,
        capt_r_min: float = 0.0,
        capt_r_max: float = 1.0,
        capt_r_point: float = 0.0,
    ) -> None:
        """Cache the world obstacles (and optional CAPT point cloud)."""
        self._ws, self._wc, self._wb, wh = _extract_world_arrays(world_geom)
        if wh.shape[0] != 0:
            raise NotImplementedError(
                "The VAMP backend has no half-space primitive; represent a "
                "ground plane as a large flat Box instead."
            )
        if point_cloud is not None:
            self._wp = np.asarray(point_cloud, dtype=np.float32).reshape(-1, 3)
            self._capt = (float(capt_r_min), float(capt_r_max), float(capt_r_point))
        else:
            self._wp = np.zeros((0, 3), dtype=np.float32)
            self._capt = (0.0, 0.0, 0.0)
        self._world_cache_key = None  # invalidate the per-call cache

    @staticmethod
    @lru_cache(maxsize=1)
    def _cpu_device():
        # The handlers are registered for platform="cpu"; on a CUDA-default JAX
        # install we must place operands on the host so ffi_call dispatches to
        # the CPU target rather than CUDA.
        return jax.devices("cpu")[0]

    def _world_args(
        self,
        world_geom: Optional[CollGeom],
        point_cloud: Optional[Array],
        capt: Optional[tuple[float, float, float]],
    ):
        """Resolve CPU-resident obstacle buffers for a call (cached by identity).

        Per-call ``world_geom`` / ``point_cloud`` override the cached
        :meth:`set_world` state; when both are omitted the cached state is used.
        Providing a ``world_geom`` without a ``point_cloud`` keeps the cached
        cloud (it does not silently wipe it).

        Re-extracting a ``CollGeom`` means converting its (GPU-resident) JAX
        arrays to host numpy — a sync worth ~1 ms.  We therefore memoise the
        result keyed by the object identities + capt, so repeatedly checking
        against the same world (the planner / benchmark pattern) is free after
        the first call.
        """
        key = (id(world_geom), id(point_cloud), capt)
        if key == self._world_cache_key and self._world_cache is not None:
            return self._world_cache

        cpu = self._cpu_device()
        if world_geom is not None:
            ws, wc, wb, wh = _extract_world_arrays(world_geom)
            if wh.shape[0] != 0:
                raise NotImplementedError(
                    "The VAMP backend has no half-space primitive; represent a "
                    "ground plane as a large flat Box instead."
                )
        else:
            ws, wc, wb = self._ws, self._wc, self._wb

        if point_cloud is not None:
            wp = np.asarray(point_cloud, dtype=np.float32).reshape(-1, 3)
            capt_v = capt if capt is not None else (0.0, 1.0, 0.0)
        else:
            wp, capt_v = self._wp, self._capt

        arrays = (
            jax.device_put(np.asarray(ws, dtype=np.float32), cpu),
            jax.device_put(np.asarray(wc, dtype=np.float32), cpu),
            jax.device_put(np.asarray(wb, dtype=np.float32), cpu),
            jax.device_put(np.asarray(wp, dtype=np.float32), cpu),
            tuple(float(x) for x in capt_v),
        )
        self._world_cache_key = key
        self._world_cache = arrays
        return arrays

    def _world_args_traced(
        self,
        world_geom: Optional[CollGeom],
        point_cloud: Optional[Array],
        capt: Optional[tuple[float, float, float]],
    ):
        """Traceable analogue of :meth:`_world_args`.

        Returns the obstacle buffers as *jnp* arrays (JAX-native extraction, no
        host numpy round-trip and no ``device_put``), so they can be threaded
        straight into ``ffi_call`` inside a jit/vmap/pmap trace.  ``capt`` stays
        a tuple of concrete Python floats (it is a static FFI attribute).

        Note: obstacle identity/id caching is skipped here — under a trace the
        arrays are tracers, and the (cheap) JAX extraction is folded into the
        surrounding compilation anyway.
        """
        if world_geom is not None:
            ws, wc, wb, wh = _extract_world_arrays_jax(world_geom)
            if wh.shape[0] != 0:
                raise NotImplementedError(
                    "The VAMP backend has no half-space primitive; represent a "
                    "ground plane as a large flat Box instead."
                )
        else:
            ws = jnp.asarray(self._ws, dtype=jnp.float32)
            wc = jnp.asarray(self._wc, dtype=jnp.float32)
            wb = jnp.asarray(self._wb, dtype=jnp.float32)

        if point_cloud is not None:
            wp = jnp.asarray(point_cloud, dtype=jnp.float32).reshape(-1, 3)
            capt_v = capt if capt is not None else (0.0, 1.0, 0.0)
        else:
            wp = jnp.asarray(self._wp, dtype=jnp.float32)
            capt_v = self._capt

        return ws, wc, wb, wp, tuple(float(x) for x in capt_v)

    # ── Public API (mirrors CUDABinaryCollisionChecker) ─────────────────────

    def _jit_fn(self, kind: str, capt: tuple[float, float, float]):
        """A cached ``jax.jit`` wrapper of the FFI call for (kind, capt).

        Baking ``capt`` (a static FFI attribute) into the traced function and
        caching by (kind, capt) lets the kernel run jitted — ~tens of us — rather
        than through slow per-call eager dispatch.  jit retraces per operand
        shape automatically."""
        key = (kind, capt)
        fn = self._jit_cache.get(key)
        if fn is not None:
            return fn

        rmin, rmax, rpt = (np.float32(capt[0]), np.float32(capt[1]), np.float32(capt[2]))
        if kind == "configs":
            tgt = self._configs_target

            def impl(a, ws, wc, wb, wp):
                out = jax.ffi.ffi_call(
                    tgt, jax.ShapeDtypeStruct((a.shape[0],), jnp.bool_)
                )(a, ws, wc, wb, wp,
                  capt_r_min=rmin, capt_r_max=rmax, capt_r_point=rpt)
                return out
        else:
            tgt = self._edges_target

            def impl(ab, ws, wc, wb, wp):
                # ab: [E, 2, n] — slice the endpoints inside the jit so no eager
                # dispatch happens on the host side.
                a = ab[:, 0, :]
                b = ab[:, 1, :]
                out = jax.ffi.ffi_call(
                    tgt, jax.ShapeDtypeStruct((ab.shape[0],), jnp.bool_)
                )(a, b, ws, wc, wb, wp,
                  capt_r_min=rmin, capt_r_max=rmax, capt_r_point=rpt)
                return out

        fn = jax.jit(impl)
        self._jit_cache[key] = fn
        return fn

    def check_collision_free(
        self,
        robot,  # accepted for API parity; FK is baked into the JIT binary
        cfg: Float[Array, "*batch actuated_count"],
        world_geom: Optional[CollGeom] = None,
        point_cloud: Optional[Array] = None,
        capt: Optional[tuple[float, float, float]] = None,
    ) -> Array:
        """Return a boolean per configuration: ``True`` if collision-free.

        ``capt`` is the optional ``(r_min, r_max, r_point)`` triple for a
        ``point_cloud`` (point radius defaults to 0 — set ``r_point`` to give the
        cloud thickness).
        """
        # Traced call (inside jax.jit / vmap / pmap / scan): extract the world
        # with JAX ops and feed the tracers straight into ffi_call — no host
        # numpy conversion or CPU device_put (both illegal on tracers). The FFI
        # target is CPU-registered, so the enclosing trace must be compiled for
        # the CPU backend.
        if _is_traced(cfg) or _is_traced(world_geom):
            ws, wc, wb, wp, capt_v = self._world_args_traced(world_geom, point_cloud, capt)
            cfg = jnp.asarray(cfg, dtype=jnp.float32)
            batch_axes = cfg.shape[:-1]
            n_act = cfg.shape[-1]
            B = int(np.prod(batch_axes)) if batch_axes else 1
            out = self._jit_fn("configs", capt_v)(cfg.reshape(B, n_act), ws, wc, wb, wp)
            return out.reshape(batch_axes) if batch_axes else out.reshape(())

        ws, wc, wb, wp, capt = self._world_args(world_geom, point_cloud, capt)
        cfg = jnp.asarray(cfg, dtype=jnp.float32)
        batch_axes = cfg.shape[:-1]
        n_act = cfg.shape[-1]
        B = int(np.prod(batch_axes)) if batch_axes else 1
        cfg_flat = jax.device_put(cfg.reshape(B, n_act), self._cpu_device())

        out = self._jit_fn("configs", capt)(cfg_flat, ws, wc, wb, wp)
        return out.reshape(batch_axes) if batch_axes else out.reshape(())

    def check_edges_collision_free(
        self,
        robot,
        edge_cfgs: Float[Array, "*batch endpoint actuated_count"],
        world_geom: Optional[CollGeom] = None,
        point_cloud: Optional[Array] = None,
        capt: Optional[tuple[float, float, float]] = None,
    ) -> Array:
        """Batch edge validation: ``True`` if the whole edge is collision-free.

        Unlike the CUDA checker — which expects pre-discretised points along the
        ``granularity`` axis and AND-reduces them — VAMP discretises each edge
        internally at the robot's planning resolution.  ``edge_cfgs`` therefore
        holds the two endpoints in its second-to-last axis (shape ``[*batch, 2,
        n_act]``); the return shape is ``edge_cfgs.shape[:-2]``.

        VAMP samples the open interval ``(0, 1]``: the goal endpoint and the
        interior are validated, but the start is assumed pre-validated (the usual
        planner contract, matching VAMP's own ``validate_motion``).  A ``True``
        verdict therefore implies the *goal* endpoint and interior are
        collision-free; validate the start separately with
        :meth:`check_collision_free` if you need it.
        """
        # Traced call: JAX-native extraction + ffi_call on the tracers (see
        # check_collision_free for the platform caveat).
        if _is_traced(edge_cfgs) or _is_traced(world_geom):
            ws, wc, wb, wp, capt_v = self._world_args_traced(world_geom, point_cloud, capt)
            edge_cfgs = jnp.asarray(edge_cfgs, dtype=jnp.float32)
            *edge_axes, endpoints, n_act = edge_cfgs.shape
            if endpoints != 2:
                raise ValueError(
                    "VAMP edge validation expects exactly 2 endpoints per edge "
                    f"(got {endpoints}); it discretises internally at the robot "
                    "resolution."
                )
            E = int(np.prod(edge_axes)) if edge_axes else 1
            out = self._jit_fn("edges", capt_v)(
                edge_cfgs.reshape(E, 2, n_act), ws, wc, wb, wp
            )
            return out.reshape(tuple(edge_axes)) if edge_axes else out.reshape(())

        ws, wc, wb, wp, capt = self._world_args(world_geom, point_cloud, capt)
        edge_cfgs = jnp.asarray(edge_cfgs, dtype=jnp.float32)
        *edge_axes, endpoints, n_act = edge_cfgs.shape
        if endpoints != 2:
            raise ValueError(
                "VAMP edge validation expects exactly 2 endpoints per edge "
                f"(got {endpoints}); it discretises internally at the robot "
                "resolution."
            )
        E = int(np.prod(edge_axes)) if edge_axes else 1
        # Move the whole edge array to the host in a single transfer; the jitted
        # function slices the two endpoints internally.  (Slicing the GPU array
        # and transferring each strided half separately is what made small edge
        # batches slow.)
        flat = jax.device_put(edge_cfgs.reshape(E, 2, n_act), self._cpu_device())
        out = self._jit_fn("edges", capt)(flat, ws, wc, wb, wp)
        return out.reshape(tuple(edge_axes)) if edge_axes else out.reshape(())


def make_vamp_cpu_checker(
    urdf_path: str | Path,
    srdf_path: Optional[str | Path] = None,
    end_effector: Optional[str] = None,
    **kwargs,
) -> VAMPCPUCollisionChecker:
    """Build a JIT-compiled VAMP CPU collision checker for ``urdf_path``."""
    return VAMPCPUCollisionChecker(
        urdf_path, srdf_path=srdf_path, end_effector=end_effector, **kwargs
    )
