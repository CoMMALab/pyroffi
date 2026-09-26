"""Robot-specialized CUDA kernels: stock kernels compiled against cricket-traced kinematics.

The stock kernels walk the kinematic tree from runtime buffers on every FK/Jacobian
evaluation and size every loop by runtime DOF. For one fixed robot and end-effector, this
module builds a variant of a kernel with ``-DPYROFFI_TRACED_ROBOT`` against a generated
header holding

* cricket's ``language="cuda"`` output: straight-line EE pose/Jacobian and every joint's
  world pose, with the robot's transforms folded to constants;
* the maps between cricket's joint order and the kernel's solved/frozen variables;
* the collision tables of the call (world spheres, SRDF self-collision pairs).

and registers it as its own FFI target taking the stock operands. What a kernel does with
the header is up to its ``#ifdef PYROFFI_TRACED_ROBOT`` blocks (see ``_traced_robot.cuh``).

Builds are cached on disk (``$PYROFFI_TRACED_CACHE``, default ``~/.cache/pyroffi/traced``)
keyed by the header, kernel sources and flags, so each robot compiles once. GRiD's dynamics
libraries go through the same builder (:func:`build_shared_library`) and cache.

Requires cricket built with its Python extension (``external/cricket``) and ``nvcc``.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import jax
import numpy as np
from loguru import logger

_KERNELS_DIR = Path(__file__).parent
_REPO_ROOT = _KERNELS_DIR.parents[2]
_GLASS_DIR = _REPO_ROOT / "external" / "GLASS"

# Kernels with a traced variant: source (relative to cuda_kernels/), FFI handler symbols, and
# whether a collision-free build may solve over the end-effector chain alone (IK only: FK and
# collision kernels must see every joint).
KERNELS = {
    "sqp_ik": ("ik/_sqp_ik_cuda_kernel.cu", ("SqpIkCudaFfi",), True),
    "ls_ik": ("ik/_ls_ik_cuda_kernel.cu", ("LsIkCudaFfi",), True),
    "hjcd_ik": ("ik/_hjcd_ik_cuda_kernel.cu", ("HjcdIkCoarseCudaFfi", "HjcdIkLmCudaFfi"), True),
    "mppi_ik": ("ik/_mppi_ik_cuda_kernel.cu", ("MppiIkCudaFfi",), True),
    "fused_collision": ("collision/_fused_self_collision_kernel.cu",
                        ("FusedSelfCollisionFfi", "FusedWorldCollisionFfi", "FusedWorldEsdfFfi"), False),
    "robogpu": ("collision/_robogpu_collision_host.cu", ("RoboGPUCollisionFfi",), False),
    "sco_trajopt": ("trajopt/_sco_trajopt_cuda_kernel.cu", ("ScoTrajoptCudaFfi",), False),
}


def _optix_sdk() -> Path:
    env = os.environ.get("OPTIX_SDK")
    for root in ([Path(env)] if env else []) + sorted(_REPO_ROOT.glob("NVIDIA-OptiX-SDK*")):
        if (root / "include" / "optix.h").is_file():
            return root
    raise RuntimeError("OptiX SDK not found; set OPTIX_SDK (see build_robogpu_collision.sh).")


def _build_extras(kernel: str) -> tuple[list[str], list[Path]]:
    """Extra nvcc flags, and files that must sit next to the built .so, per kernel."""
    if kernel == "robogpu":
        # The host library locates its OptiX programs next to itself (dladdr).
        ptx = _KERNELS_DIR / "collision" / "_robogpu_optix_programs.ptx"
        return [f"-I{_optix_sdk() / 'include'}", "-ldl"], [ptx]
    return [], []

# Kinematics are all cricket is used for here, so an empty SRDF keeps it from spending
# seconds sampling self-collisions it would otherwise guess without one.
_EMPTY_SRDF = '<?xml version="1.0"?>\n<robot name="robot"></robot>\n'


def _cache_root() -> Path:
    env = os.environ.get("PYROFFI_TRACED_CACHE")
    if env:
        return Path(env)
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "pyroffi" / "traced"


def _robot_header(urdf_xml: str, ee_link: str, actuated_names: tuple[str, ...],
                  joint_names: tuple[str, ...], solved_names: tuple[str, ...]) -> str:
    try:
        import cricket
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Traced kernels need cricket's Python extension. Build it with "
            "`bash build_kernels/build_cricket_jit.sh`."
        ) from exc

    with tempfile.TemporaryDirectory(prefix="pyroffi_traced_") as tmp:
        urdf_path = Path(tmp) / "robot.urdf"
        srdf_path = Path(tmp) / "robot.srdf"
        urdf_path.write_text(urdf_xml)
        srdf_path.write_text(_EMPTY_SRDF)
        gen = cricket.generate_robot_source(cricket.GenOptions(
            urdf=urdf_path, srdf=srdf_path, end_effector=ee_link,
            # Every joint's frame, in pyroffi's joint order, becomes T_world.
            language="cuda", data={"name": "Robot", "trace_frames": list(joint_names)},
        ))

    cricket_names = list(gen.data["joint_names"])
    if sorted(cricket_names) != sorted(actuated_names):
        raise ValueError(
            "Traced kernel: cricket's joints do not match pyroffi's actuated joints "
            f"(cricket {cricket_names}, pyroffi {list(actuated_names)}). Joints pinocchio "
            "models with extra coordinates (continuous, planar, floating) are not supported."
        )
    # The kernel solves over `solved_names` (in pyroffi order) and carries every other
    # actuated joint unchanged from its seed ("frozen"). Both maps are emitted as
    # straight-line code, so cricket's joint order costs nothing at runtime.
    solved = [a for a, n in enumerate(actuated_names) if n in solved_names]
    frozen = [a for a, n in enumerate(actuated_names) if n not in solved_names]
    src = {a: f"cfg[{solved.index(a)}]" if a in solved else f"frz[{frozen.index(a)}]"
           for a in range(len(actuated_names))}
    gather = "".join(f"    q[{i}] = {src[actuated_names.index(n)]};\n"
                     for i, n in enumerate(cricket_names))
    scatter = "".join(
        f"    J[{row * len(solved) + solved.index(actuated_names.index(n))}] = Jt[{row * len(cricket_names) + i}];\n"
        for row in range(6) for i, n in enumerate(cricket_names) if actuated_names.index(n) in solved)
    idx = lambda xs: ", ".join(map(str, xs)) or "0"
    return (
        f"{gen.source}\n"
        "namespace pyroffi::traced {\n"
        "namespace robot = cricket::robots::robot;\n"
        f"constexpr int n_solved = {len(solved)};\n"
        f"constexpr int n_frozen = {len(frozen)};\n"
        # Actuated (pyroffi) index of solved variable i / frozen slot k.
        "static __host__ __device__ __forceinline__ int solved_idx(int i)\n"
        f"{{\n    constexpr int t[{max(len(solved), 1)}] = {{{idx(solved)}}};\n    return t[i];\n}}\n"
        "static __host__ __device__ __forceinline__ int frozen_idx(int k)\n"
        f"{{\n    constexpr int t[{max(len(frozen), 1)}] = {{{idx(frozen)}}};\n    return t[k];\n}}\n"
        "// Cricket's q from the solved variables and the frozen joints.\n"
        "static __device__ __forceinline__ void gather_q(const float* __restrict__ cfg,\n"
        "    const float* __restrict__ frz, float* __restrict__ q)\n"
        f"{{\n{gather}    (void)frz;\n}}\n"
        "// Cricket's (6, n_q) Jacobian into the kernel's (6, n_solved), pyroffi order.\n"
        "static __device__ __forceinline__ void scatter_jacobian(const float* __restrict__ Jt,\n"
        "    float* __restrict__ J)\n"
        f"{{\n{scatter}}}\n"
        "}  // namespace pyroffi::traced\n"
    )


def constant_table(ctype: str, name: str, values) -> str:
    """A ``__constant__`` array definition holding ``values`` (at least one element)."""
    values = np.asarray(values).reshape(-1)
    fmt = (lambda v: f"{float(v):.9e}f") if ctype == "float" else (lambda v: str(int(v)))
    body = ", ".join(fmt(v) for v in values) or "0"
    return f"__device__ __constant__ {ctype} {name}[{max(values.size, 1)}] = {{{body}}};\n"


def collision_tables_source(robot_spheres, robot_sphere_joint, self_tables,
                            world_counts=(0, 0, 0, 0)) -> str:
    """The call's collision structure as compile-time tables.

    ``robot_spheres``/``robot_sphere_joint`` are the world-collision spheres (joint frame) and
    ``self_tables`` the SRDF-filtered self-collision tables, exactly as the kernel receives them.
    ``world_counts`` is the number of world spheres, capsules, boxes and halfspaces: a scene's
    obstacle set is fixed while the obstacles move, so the counts are baked in and the poses
    stay runtime inputs. Empty inputs give empty tables, and the traced kernel then compiles
    that part of collision out entirely.
    """
    table = constant_table
    sph, start, link_joint, pair_i, pair_j = self_tables
    return (
        "namespace pyroffi::traced {\n"
        f"constexpr int n_robot_spheres = {len(np.asarray(robot_sphere_joint).reshape(-1))};\n"
        + table("float", "kRobotSpheres", robot_spheres)
        + table("int", "kRobotSphereJoint", robot_sphere_joint)
        + f"constexpr int n_self_pairs = {len(np.asarray(pair_i).reshape(-1))};\n"
        + f"constexpr int n_self_links = {max(len(np.asarray(start).reshape(-1)) - 1, 0)};\n"
        + table("float", "kSelfSph", sph)
        + table("int", "kSelfLinkStart", start)
        + table("int", "kSelfLinkJoint", link_joint)
        + table("int", "kSelfPairI", pair_i)
        + table("int", "kSelfPairJ", pair_j)
        + "".join(f"constexpr int n_world_{kind} = {int(n)};\n"
                  for kind, n in zip(("spheres", "capsules", "boxes", "halfspaces"), world_counts))
        + "}  // namespace pyroffi::traced\n"
    )


def build_shared_library(name: str, source: Path, generated: dict[str, str], flags: list[str],
                         include_dirs: tuple[Path, ...] = (), key_files: tuple[Path, ...] = (),
                         copy_files: tuple[Path, ...] = (), so_name: str | None = None) -> Path:
    """nvcc ``source`` against the ``generated`` headers into a disk-cached shared library.

    The one builder for every robot-specialized kernel (traced kernels and GRiD dynamics).
    The cache key covers the generated headers, the flags and every file in ``key_files``
    (``source`` is added when absent), so any input that changes the binary changes the key.
    The library is ``so_name`` (default ``<name>.so``) inside the key's directory.
    Builds land in a temporary directory that is renamed into place, so concurrent builds of
    the same library cannot leave a half-written one behind.
    """
    import jaxlib

    so_name = so_name or f"{name}.so"
    if source not in key_files:
        key_files = (source, *key_files)
    key = hashlib.sha1("\x00".join(
        [name, *generated.values(), " ".join(flags),
         *(p.read_text() for p in key_files)]).encode()).hexdigest()
    build_dir = _cache_root() / key
    if (build_dir / so_name).is_file():
        return build_dir / so_name

    _cache_root().mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=_cache_root()))
    try:
        for name, text in generated.items():
            (tmp / name).write_text(text)
        for f in copy_files:
            (tmp / f.name).write_bytes(f.read_bytes())
        cmd = ["nvcc", *flags, f"-I{tmp}", *(f"-I{d}" for d in include_dirs),
               f"-I{Path(jaxlib.__file__).parent / 'include'}", "-o", str(tmp / so_name), str(source)]
        logger.info(f"Compiling {so_name} (one-time, cached): {build_dir}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"nvcc failed to compile {so_name}:\n{result.stderr}")
        try:
            tmp.rename(build_dir)
        except OSError:  # another process finished the same build first
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return build_dir / so_name


def _compile(kernel: str, header: str, n_solved: int, n_joints: int) -> Path:
    source = _KERNELS_DIR / KERNELS[kernel][0]
    # Capacities sized exactly to this robot, and only its own solve size instantiated.
    flags = [
        "-O3", "-std=c++17", os.environ.get("PYROFFI_TRACED_GPU_ARCH", "-arch=native"),
        f"-DMAX_JOINTS={n_joints}", f"-DMAX_ACT={n_solved}", "-DMAX_EE=1",
        f"-DPYROFFI_SOLVE_N_BUCKETS(X)=X({n_solved})",
        "-DPYROFFI_TRACED_ROBOT", "--shared", "--compiler-options", "-fPIC",
    ]
    extra_flags, extra_files = _build_extras(kernel)
    flags += extra_flags
    # GLASS is part of the key too: a submodule bump must not reuse builds made against the old one.
    sources = (*sorted(_KERNELS_DIR.glob("*.cuh")), *sorted(source.parent.glob("*.cuh")), source,
               *extra_files, *sorted(_GLASS_DIR.glob("*.cuh")), *sorted((_GLASS_DIR / "src").rglob("*.cuh")))
    return build_shared_library(
        kernel, source, {"_traced_robot_gen.cuh": header}, flags,
        include_dirs=(_KERNELS_DIR, _GLASS_DIR), key_files=sources, copy_files=tuple(extra_files),
        so_name=f"{kernel}_traced.so")


@lru_cache(maxsize=None)
def traced_target(kernel: str, urdf_xml: str, ee_link: str, actuated_names: tuple[str, ...],
                  joint_names: tuple[str, ...], chain_names: tuple[str, ...],
                  collision_src: str, has_collision: bool) -> tuple[str, ...]:
    """Build (or load from cache) and register a traced kernel; return one FFI target name
    per handler symbol in ``KERNELS[kernel]``, in that order.

    ``collision_src`` comes from :func:`collision_tables_source`; the build is only valid for
    those tables, and the kernel rejects a launch whose table sizes differ.

    ``chain_names`` are the actuated joints on the end-effector's chain. Without collision
    every other joint has an identically zero Jacobian column and gradient, so it can never
    move; the build then solves over the chain alone and carries the rest from the seed.
    Collision gradients can move any joint, so collision builds solve over all of them.
    """
    solved = chain_names if KERNELS[kernel][2] and not has_collision else actuated_names
    header = (_robot_header(urdf_xml, ee_link, actuated_names, joint_names, solved)
              + collision_src)
    lib = ctypes.CDLL(str(_compile(kernel, header, len(solved), len(joint_names))))

    capsule_new = ctypes.pythonapi.PyCapsule_New
    capsule_new.restype = ctypes.py_object
    capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    tag = hashlib.sha1(header.encode()).hexdigest()[:16]
    names = []
    for symbol in KERNELS[kernel][1]:
        capsule = capsule_new(
            ctypes.cast(getattr(lib, symbol), ctypes.c_void_p), b"xla._CUSTOM_CALL_TARGET", None)
        names.append(f"{symbol}_traced_{tag}")
        jax.ffi.register_ffi_target(names[-1], capsule, platform="CUDA")
    return tuple(names)
