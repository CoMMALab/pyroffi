"""Analytic task Jacobian from CUDA, for the IK implicit-diff rule.

The IK counterpart of GRiD's gradient kernels: a solve kernel paired with a
separate analytic-derivative kernel, whose output feeds a ``custom_jvp`` tangent
rule on the JAX side. One kernel serves every IK solver, so no solve path had to
be modified to obtain the derivative.

Replaces a ``jax.jacobian`` of the residual that XLA was re-tracing and
re-differentiating on the host side on every gradient evaluation.

CONVENTION: the returned ``J`` is the Jacobian of the KERNEL's residual --
world-frame ``p_ee - p_tgt`` stacked with a quaternion error -- not of the SE(3)
local log-map. Callers must pair it with :func:`ik_residual_kernel_convention`
for the other Jacobian blocks; see that function for why mixing is a silent
correctness bug rather than a cosmetic mismatch.
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .._ffi_dtypes import as_robot_buffers

_LIB_NAME = "_ik_jacobian_lib.so"


@lru_cache(maxsize=1)
def _load_and_register() -> None:
    lib_path = Path(__file__).parent / _LIB_NAME
    if not lib_path.exists():
        raise RuntimeError(
            f"IK task-Jacobian CUDA library not found at {lib_path}.\n"
            "Compile it first with:  bash build_kernels/build_ik_jacobian_cuda.sh\n"
        )
    lib = ctypes.CDLL(str(lib_path))

    _PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    _PyCapsule_New.restype = ctypes.py_object
    _PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

    capsule = _PyCapsule_New(
        ctypes.cast(getattr(lib, "IkTaskJacobianFfi"), ctypes.c_void_p),
        b"xla._CUSTOM_CALL_TARGET",
        None,
    )
    jax.ffi.register_ffi_target("ik_task_jacobian", capsule, platform="CUDA")


def library_available() -> bool:
    """Whether the .so is present, so callers can fall back rather than fail."""
    return (Path(__file__).parent / _LIB_NAME).exists()


def ancestor_tables(robot, target_link_indices):
    """``(target_jnts, ancestor_masks)`` for the given end-effectors.

    Walks the kinematic chain with numpy, exactly as the solvers' own
    pre-computation does, so this must run OUTSIDE a trace. Callers cache it.
    """
    parent_joint_indices_np = np.array(robot.links.parent_joint_indices)
    parent_idx_np = np.array(robot.joints.parent_indices)
    n_joints = robot.joints.num_joints
    n_ee = len(target_link_indices)

    target_joints_np = np.zeros(n_ee, dtype=np.int32)
    ancestor_masks_np = np.zeros((n_ee, n_joints), dtype=np.int32)
    for i, link_idx in enumerate(target_link_indices):
        tgt = int(parent_joint_indices_np[int(link_idx)])
        target_joints_np[i] = tgt
        j = tgt
        while j >= 0:
            ancestor_masks_np[i, j] = 1
            j = int(parent_idx_np[j])
    return jnp.asarray(target_joints_np), jnp.asarray(ancestor_masks_np)


def task_jacobian(
    cfgs,
    robot_buffers,
    target_jnts,
    ancestor_masks,
    target_Ts,
):
    """Residual and its Jacobian at ``cfgs``, evaluated on the GPU.

    Args:
        cfgs: ``(n_problems, n_act)`` configurations -- normally the solver's
            returned ``q*``.
        robot_buffers: the shared 8-tuple (twists, parent_tf, parent_idx,
            act_idx, mimic_mul, mimic_off, mimic_act_idx, topo_inv).
        target_jnts: ``(n_ee,)`` target joint per end-effector.
        ancestor_masks: ``(n_ee, n_joints)``.
        target_Ts: ``(n_problems, n_ee, 7)`` wxyz_xyz target poses.

    Returns:
        ``(r, J)`` with shapes ``(n_problems, 6*n_ee)`` and
        ``(n_problems, 6*n_ee, n_act)``, both float32.
    """
    _load_and_register()

    cfgs = jnp.asarray(cfgs, jnp.float32)
    n_problems, n_act = cfgs.shape
    n_ee = int(np.shape(target_jnts)[0])
    rows = 6 * n_ee

    ops = (
        cfgs,
        *as_robot_buffers(robot_buffers),
        jnp.asarray(target_jnts, jnp.int32),
        jnp.asarray(ancestor_masks, jnp.int32),
        jnp.asarray(target_Ts, jnp.float32).reshape(n_problems, n_ee, 7),
    )

    return jax.ffi.ffi_call(
        "ik_task_jacobian",
        (
            jax.ShapeDtypeStruct((n_problems, rows), jnp.float32),
            jax.ShapeDtypeStruct((n_problems, rows, n_act), jnp.float32),
        ),
    )(*ops)
