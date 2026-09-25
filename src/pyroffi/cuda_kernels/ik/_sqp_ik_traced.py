"""Robot-specialized SQP-IK: the stock kernel compiled against cricket-traced kinematics.

The stock ``sqp_ik_cuda`` kernel walks the kinematic tree from runtime buffers on every
FK/Jacobian evaluation. For one fixed robot and end-effector, cricket's ``language="cuda"``
backend emits the same pose and geometric Jacobian as straight-line code with every
transform folded to constants. This module generates that header, compiles the SQP kernel
against it with ``-DPYROFFI_TRACED_ROBOT``, and registers the result as its own FFI target.
Solver behaviour is otherwise identical; only the EE residual/Jacobian source changes.

Builds are cached on disk (``$PYROFFI_TRACED_CACHE``, default ``~/.cache/pyroffi/traced``)
keyed by the generated header, kernel sources and flags, so each robot compiles once.

Requires cricket built with its Python extension (``external/cricket``) and ``nvcc``.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import jax
import numpy as np
from loguru import logger


_IK_DIR = Path(__file__).parent
_KERNELS_DIR = _IK_DIR.parent
_REPO_ROOT = _KERNELS_DIR.parents[2]

# Kinematics are all this path uses, so an empty SRDF keeps cricket from spending
# seconds sampling self-collisions it would otherwise guess without one.
_EMPTY_SRDF = '<?xml version="1.0"?>\n<robot name="robot"></robot>\n'


def _cache_root() -> Path:
    env = os.environ.get("PYROFFI_TRACED_CACHE")
    if env:
        return Path(env)
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "pyroffi" / "traced"


def _generate_header(urdf_xml: str, ee_link: str, actuated_names: tuple[str, ...],
                     joint_names: tuple[str, ...], solved_names: tuple[str, ...]) -> str:
    try:
        import cricket
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Traced SQP-IK needs cricket's Python extension. Build it with "
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
            "Traced SQP-IK: cricket's joints do not match pyroffi's actuated joints "
            f"(cricket {cricket_names}, pyroffi {list(actuated_names)}). Mimic joints and "
            "joints pinocchio models with extra coordinates are not supported."
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
        f"static __host__ __device__ __forceinline__ int solved_idx(int i)\n"
        f"{{\n    constexpr int t[{max(len(solved), 1)}] = {{{idx(solved)}}};\n    return t[i];\n}}\n"
        f"static __host__ __device__ __forceinline__ int frozen_idx(int k)\n"
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


def collision_tables_source(robot_spheres, robot_sphere_joint, self_tables) -> str:
    """The robot's collision geometry as compile-time tables for the traced build.

    ``robot_spheres``/``robot_sphere_joint`` are the world-collision spheres (joint frame) and
    ``self_tables`` the SRDF-filtered self-collision tables, exactly as the kernel receives them.
    Empty inputs give empty tables, and the traced kernel then compiles collision out entirely.
    """
    def table(ctype: str, name: str, values) -> str:
        values = np.asarray(values).reshape(-1)
        fmt = (lambda v: f"{float(v):.9e}f") if ctype == "float" else (lambda v: str(int(v)))
        body = ", ".join(fmt(v) for v in values) or "0"
        return f"__device__ __constant__ {ctype} {name}[{max(values.size, 1)}] = {{{body}}};\n"

    sph, start, link_joint, pair_i, pair_j = self_tables
    return (
        "namespace pyroffi::traced {\n"
        f"constexpr int n_robot_spheres = {len(np.asarray(robot_sphere_joint).reshape(-1))};\n"
        + table("float", "kRobotSpheres", robot_spheres)
        + table("int", "kRobotSphereJoint", robot_sphere_joint)
        + f"constexpr int n_self_pairs = {len(np.asarray(pair_i).reshape(-1))};\n"
        + table("float", "kSelfSph", sph)
        + table("int", "kSelfLinkStart", start)
        + table("int", "kSelfLinkJoint", link_joint)
        + table("int", "kSelfPairI", pair_i)
        + table("int", "kSelfPairJ", pair_j)
        + "}  // namespace pyroffi::traced\n"
    )


def _compile(header: str, n_q: int, n_joints: int) -> Path:
    import jaxlib

    # Capacities sized exactly to this robot, and only its own solve size instantiated.
    flags = [
        "-O3", "-std=c++17", os.environ.get("PYROFFI_TRACED_GPU_ARCH", "-arch=native"),
        f"-DMAX_JOINTS={n_joints}", f"-DMAX_ACT={n_q}", "-DMAX_EE=1",
        f"-DPYROFFI_SOLVE_N_BUCKETS(X)=X({n_q})",
        "-DPYROFFI_TRACED_ROBOT", "--shared", "--compiler-options", "-fPIC",
    ]
    sources = [*sorted(_KERNELS_DIR.glob("*.cuh")), *sorted(_IK_DIR.glob("*.cuh")),
               _IK_DIR / "_sqp_ik_cuda_kernel.cu"]
    key = hashlib.sha1("\x00".join(
        [header, " ".join(flags), *(p.read_text() for p in sources)]).encode()).hexdigest()

    build_dir = _cache_root() / key
    so_path = build_dir / "sqp_ik_traced.so"
    if so_path.is_file():
        return so_path

    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "_traced_robot_gen.cuh").write_text(header)
    cmd = [
        "nvcc", *flags,
        f"-I{build_dir}", f"-I{_KERNELS_DIR}", f"-I{_REPO_ROOT / 'external' / 'GLASS'}",
        f"-I{Path(jaxlib.__file__).parent / 'include'}",
        "-o", str(so_path), str(_IK_DIR / "_sqp_ik_cuda_kernel.cu"),
    ]
    logger.info(f"Compiling traced SQP-IK kernel (one-time, cached): {build_dir}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"nvcc failed to compile the traced SQP-IK kernel:\n{result.stderr}")
    return so_path


@lru_cache(maxsize=None)
def traced_sqp_ik_target(urdf_xml: str, ee_link: str, actuated_names: tuple[str, ...],
                         joint_names: tuple[str, ...], chain_names: tuple[str, ...],
                         collision_src: str) -> str:
    """Build (or load from cache) and register the traced kernel; return its FFI target name.

    ``collision_src`` comes from :func:`collision_tables_source`; the build is only valid for
    those tables, and the kernel rejects a launch whose table sizes differ.

    ``chain_names`` are the actuated joints on the end-effector's chain. Without collision
    every other joint has an identically zero Jacobian column and gradient, so it can never
    move; the build then solves over the chain alone and carries the rest from the seed.
    Collision gradients can move any joint, so collision builds solve over all of them.
    """
    no_collision = "n_robot_spheres = 0;" in collision_src and "n_self_pairs = 0;" in collision_src
    solved = chain_names if no_collision else actuated_names
    header = (_generate_header(urdf_xml, ee_link, actuated_names, joint_names, solved)
              + collision_src)
    so_path = _compile(header, len(solved), len(joint_names))
    lib = ctypes.CDLL(str(so_path))

    capsule_new = ctypes.pythonapi.PyCapsule_New
    capsule_new.restype = ctypes.py_object
    capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    capsule = capsule_new(
        ctypes.cast(lib.SqpIkCudaFfi, ctypes.c_void_p), b"xla._CUSTOM_CALL_TARGET", None)

    name = f"sqp_ik_cuda_traced_{so_path.parent.name[:16]}"
    jax.ffi.register_ffi_target(name, capsule, platform="CUDA")
    return name
