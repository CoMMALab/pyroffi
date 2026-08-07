"""Rigid body dynamics: pure-JAX algorithms plus a CUDA-accelerated path
generated per-robot by GRiD's codegen (``GRiDDynamics``).

The pure-JAX functions are differentiable and run on any backend; the
``GRiDDynamics`` class JIT-generates, compiles, and registers per-robot CUDA
kernels (including analytic-gradient kernels) via the JAX FFI.
"""

from ._api import (
    forward_dynamics as forward_dynamics,
    inverse_dynamics as inverse_dynamics,
    jacobian as jacobian,
    mass_matrix as mass_matrix,
    step as step,
)
from ._dynamics_jax import (
    forward_dynamics_jax as forward_dynamics_jax,
    inverse_dynamics_jax as inverse_dynamics_jax,
    jacobian_jax as jacobian_jax,
    mass_matrix_jax as mass_matrix_jax,
)
from ._integrators import step_with_fd as step_with_fd


def __getattr__(name: str):
    # GRiDDynamics pulls in the codegen/vendor machinery; import lazily so
    # pure-JAX dynamics works without the external GRiD modules present.
    if name == "GRiDDynamics":
        from ._grid_dynamics import GRiDDynamics

        return GRiDDynamics
    if name in (
        "ManipulatorSpec",
        "GraspedObject",
        "ContactSystem",
        "capture_attachments",
        "capture_grasp_offsets",
        "object_pose_world",
    ):
        from . import _contact

        return getattr(_contact, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
