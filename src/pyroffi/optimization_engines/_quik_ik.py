"""QuIK CPU inverse-kinematics backend, exposed through the pyroffi JAX API.

This wraps the QuIK C++ solver (external/QuIK) behind an XLA FFI custom call,
JIT-compiled once at runtime by cricket and registered for ``platform="cpu"``.
The robot's product-of-exponentials model is converted to standard DH once (via
:func:`pyroffi.kinematics._dh.extract_dh`) and the DH table is fed to the kernel
as runtime buffers, so a single compiled kernel serves every serial robot.

Intended for CPU-only accelerated planning (``JAX_PLATFORMS=cpu``): QuIK's
Halley's-method solver converges in a handful of iterations and, parallelised
across a batch of seeds with OpenMP, is a fast CPU alternative to the CUDA IK
solvers for simple serial arms.

Algorithms (``algorithm=`` / the ``QuIKAlgorithm`` enum):
    0  QuIK   — third-order Halley's method (default)
    1  NR     — Newton-Raphson / Levenberg-Marquardt (with ``lambda2``)
    2  BFGS   — quasi-Newton with Armijo line search
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._jit_ffi import (
    cpu_device,
    cricket_jit,
    eigen_include_dirs,
    register_handlers,
    xla_ffi_include,
)
from ..kinematics._dh import DHModel, extract_dh

if TYPE_CHECKING:
    from .._robot import Robot

_KERNELS_DIR = Path(__file__).resolve().parent.parent / "quik_kernels"
_FFI_HEADER = _KERNELS_DIR / "_quik_ik_ffi.hh"
_TU_TEMPLATE = _KERNELS_DIR / "_quik_ik_tu.cc.in"
_QUIK_INCLUDE = (
    Path(__file__).resolve().parents[3] / "external" / "QuIK" / "C++" / "QuIK"
)
# QuIK's headers #include "IK/..." and "Robot/..." relatively from these roots.
_QUIK_SUBDIRS = ("IK", "Robot")

QUIK_TARGET = "quik_ik_solve"


@lru_cache(maxsize=1)
def _ensure_registered() -> str:
    """JIT-compile the QuIK FFI kernel once and register it. Returns the target."""
    cricket, jit = cricket_jit()

    tu_source = _TU_TEMPLATE.read_text().replace("@FFI_HEADER@", str(_FFI_HEADER))

    opts = jit.CompileOptions()
    opts.std_flag = "-std=c++17"
    opts.opt_flag = "-O3"
    include_dirs = [
        str(_FFI_HEADER.parent),
        str(_QUIK_INCLUDE),
        *[str(_QUIK_INCLUDE / d) for d in _QUIK_SUBDIRS],
        xla_ffi_include(),
        *eigen_include_dirs(),
    ]
    opts.include_dirs = include_dirs
    opts.extra_flags = ["-march=native", "-fopenmp"]

    digest = hashlib.sha1()
    digest.update(_FFI_HEADER.read_bytes())
    digest.update(_TU_TEMPLATE.read_bytes())
    opts.module_id = f"quik_ik_{digest.hexdigest()[:16]}"

    work_dir = Path(jit.default_cache_dir())
    work_dir.mkdir(parents=True, exist_ok=True)
    register_handlers(
        tu_source, opts, work_dir, {QUIK_TARGET: "pyroffi_get_quik_ik_solve"}
    )
    return QUIK_TARGET


# Cache one DH model per (robot identity, ee link) so repeated solves are free.
_DH_CACHE: dict[tuple[int, str], DHModel] = {}

# Reuse one QuIKSolver per (robot identity, ee link) so the dispatcher does not
# re-extract DH / re-register the kernel on every IK call.
_SOLVER_CACHE: dict[tuple[int, str], "QuIKSolver"] = {}


def _dh_for(robot: "Robot", ee_link_name: str, **kwargs) -> DHModel:
    key = (id(robot), ee_link_name)
    model = _DH_CACHE.get(key)
    if model is None:
        model = extract_dh(robot, ee_link_name, **kwargs)
        _DH_CACHE[key] = model
    return model


class QuIKSolver:
    """A reusable QuIK IK solver bound to one robot + end-effector.

    Construction converts the robot to standard DH (validated against pyroffi FK)
    and compiles/registers the FFI kernel.  :meth:`solve` then runs a batch of
    seeds against a batch of target poses on the CPU.
    """

    @classmethod
    def from_robot(
        cls,
        robot: "Robot",
        ee_link_name: str,
        **kwargs,
    ) -> "QuIKSolver":
        """Build from a pyroffi :class:`~pyroffi.Robot` and end-effector link.

        The unified ``from_robot`` entry point shared by the external-binding
        backends (see :mod:`pyroffi.bindings`).  QuIK already consumes the
        in-memory ``Robot`` directly (via POE->DH extraction), so this is a thin
        alias over the constructor for API symmetry.
        """
        return cls(robot, ee_link_name, **kwargs)

    def __init__(
        self,
        robot: "Robot",
        ee_link_name: str,
        *,
        validate: bool = True,
        tol: float = 1e-4,
    ) -> None:
        self.robot = robot
        self.ee_link_name = ee_link_name
        self.model = _dh_for(robot, ee_link_name, validate=validate, tol=tol)
        self._target = _ensure_registered()
        self._dof = self.model.dof

        cpu = cpu_device()
        m = self.model
        self._dh = jax.device_put(np.asarray(m.dh, np.float32), cpu)
        self._link_type = jax.device_put(
            np.asarray(m.link_types, np.float32), cpu
        )
        self._qsign = jax.device_put(np.asarray(m.qsign, np.float32), cpu)
        self._tbase = jax.device_put(np.asarray(m.t_base, np.float32), cpu)
        self._ttool = jax.device_put(np.asarray(m.t_tool, np.float32), cpu)
        self._jit_cache: dict[tuple, object] = {}

    @property
    def dof(self) -> int:
        return self._dof

    def _jit_fn(self, attrs: tuple):
        fn = self._jit_cache.get(attrs)
        if fn is not None:
            return fn
        (
            algorithm,
            iter_max,
            exit_tol,
            min_step,
            rel_improve_tol,
            max_grad_fails,
            max_grad_fails_total,
            lambda2,
            max_lin_step,
            max_ang_step,
        ) = attrs
        target = self._target
        dof = self._dof

        def impl(q0, twt, dh, link_type, qsign, tbase, ttool):
            B = q0.shape[0]
            return jax.ffi.ffi_call(
                target,
                (
                    jax.ShapeDtypeStruct((B, dof), jnp.float32),
                    jax.ShapeDtypeStruct((B,), jnp.float32),
                    jax.ShapeDtypeStruct((B,), jnp.int32),
                ),
            )(
                q0,
                twt,
                dh,
                link_type,
                qsign,
                tbase,
                ttool,
                algorithm=np.int32(algorithm),
                iter_max=np.int32(iter_max),
                exit_tol=np.float64(exit_tol),
                min_step=np.float64(min_step),
                rel_improve_tol=np.float64(rel_improve_tol),
                max_grad_fails=np.int32(max_grad_fails),
                max_grad_fails_total=np.int32(max_grad_fails_total),
                lambda2=np.float64(lambda2),
                max_lin_step=np.float64(max_lin_step),
                max_ang_step=np.float64(max_ang_step),
            )

        fn = jax.jit(impl)
        self._jit_cache[attrs] = fn
        return fn

    def solve(
        self,
        target_poses: Array,
        seeds: Array,
        *,
        algorithm: int = 0,
        iter_max: int = 100,
        exit_tol: float = 1e-12,
        min_step: float = 1e-14,
        rel_improve_tol: float = 0.05,
        max_grad_fails: int = 5,
        max_grad_fails_total: int = 20,
        lambda2: float = 0.0,
        max_lin_step: float = 0.34,
        max_ang_step: float = 1.0,
    ) -> dict[str, Array]:
        """Solve a batch of IK problems.

        Args:
            target_poses: ``[B, 4, 4]`` desired end-effector homogeneous poses.
            seeds:        ``[B, dof]`` initial joint guesses (chain order — see
                          ``self.model.actuated_order`` to map from the robot's
                          actuated-joint vector).
            algorithm:    0 QuIK/Halley, 1 NR/LM, 2 BFGS.

        Returns:
            ``{"q": [B, dof], "error": [B], "iters": [B]}`` (chain-ordered q).
        """
        cpu = cpu_device()
        twt = jax.device_put(
            jnp.asarray(target_poses, jnp.float32).reshape(-1, 4, 4), cpu
        )
        q0 = jax.device_put(jnp.asarray(seeds, jnp.float32).reshape(-1, self._dof), cpu)
        attrs = (
            int(algorithm),
            int(iter_max),
            float(exit_tol),
            float(min_step),
            float(rel_improve_tol),
            int(max_grad_fails),
            int(max_grad_fails_total),
            float(lambda2),
            float(max_lin_step),
            float(max_ang_step),
        )
        q, err, iters = self._jit_fn(attrs)(
            q0, twt, self._dh, self._link_type, self._qsign, self._tbase, self._ttool
        )
        return {"q": q, "error": err, "iters": iters}

    def solve_to_actuated(
        self, target_poses: Array, seeds: Array, **kwargs
    ) -> dict[str, Array]:
        """Like :meth:`solve` but scatters ``q`` back into the robot's full
        actuated-joint vector order (chain joints placed at ``actuated_order``).
        """
        out = self.solve(target_poses, seeds, **kwargs)
        n_act = int(self.robot.joints.num_actuated_joints)
        order = jnp.asarray(self.model.actuated_order)
        q_full = jnp.zeros(out["q"].shape[:-1] + (n_act,), out["q"].dtype)
        q_full = q_full.at[..., order].set(out["q"])
        out["q_actuated"] = q_full
        return out


def quik_ik_solve(
    robot: "Robot",
    target_link_indices: Sequence[int],
    target_poses: Sequence,
    rng_key: Array,
    previous_cfg: Array,
    num_seeds: int = 32,
    continuity_weight: float = 1e-3,
    fixed_joint_mask: Array | None = None,
    *,
    algorithm: int = 0,
    iter_max: int = 100,
    lambda2: float = 0.0,
    validate_dh: bool = True,
    **_ignored,
) -> Array:
    """Dispatcher-shaped single-target QuIK solve (see ``_ik.inverse_kinematics``).

    Mirrors the JAX solvers' calling convention so the QuIK backend plugs into
    the shared dispatcher.  Only single-end-effector targets are supported
    (QuIK is a serial-chain solver); multi-EE requests raise.  Seeds a batch of
    ``num_seeds`` random configurations plus ``previous_cfg`` and returns the
    lowest-error solution, in the robot's actuated-joint order.
    """
    import jaxlie

    if len(target_link_indices) != 1:
        raise ValueError(
            "The QuIK backend solves a single serial chain; got "
            f"{len(target_link_indices)} end-effector targets."
        )
    ee_link_name = robot.links.names[target_link_indices[0]]
    key = (id(robot), ee_link_name)
    solver = _SOLVER_CACHE.get(key)
    if solver is None:
        solver = QuIKSolver(robot, ee_link_name, validate=validate_dh)
        _SOLVER_CACHE[key] = solver

    pose = target_poses[0]
    T = np.asarray(pose.as_matrix() if isinstance(pose, jaxlie.SE3) else pose)
    T = T.reshape(4, 4)

    # Seeds: previous_cfg + random within limits, restricted to the chain joints.
    order = solver.model.actuated_order
    lower = np.asarray(robot.joints.lower_limits)[order]
    upper = np.asarray(robot.joints.upper_limits)[order]
    lower = np.where(np.isfinite(lower), lower, -np.pi)
    upper = np.where(np.isfinite(upper), upper, np.pi)
    key = rng_key if rng_key is not None else jax.random.PRNGKey(0)
    rand = jax.random.uniform(
        key, (num_seeds, solver.dof), minval=jnp.asarray(lower), maxval=jnp.asarray(upper)
    )
    prev = jnp.asarray(np.asarray(previous_cfg)[order])[None]
    seeds = jnp.concatenate([prev, rand], axis=0)
    poses = jnp.broadcast_to(jnp.asarray(T, jnp.float32), (seeds.shape[0], 4, 4))

    out = solver.solve_to_actuated(
        poses, seeds, algorithm=algorithm, iter_max=iter_max, lambda2=lambda2
    )
    best = int(jnp.argmin(out["error"]))
    return out["q_actuated"][best]
