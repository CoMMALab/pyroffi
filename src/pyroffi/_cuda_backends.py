"""Lazy, per-robot registry of CUDA/URDF-dependent backends.

Some accelerated backends need URDF-derived state that cannot live in the
``Robot`` JAX pytree: the collision checkers need trimesh geometry and the
GRiD dynamics backend JIT-compiles a per-robot CUDA library.  ``CudaBackends``
holds that state at the *Python* level (where side effects belong) and builds
each backend lazily on first use.

An instance is carried in ``Robot`` as a ``jdc.Static`` field, so it is part of
the pytree's ``treedef`` aux-data.  It is therefore **identity-hashed and never
compared by value** — do not give it ``__eq__``/``__hash__`` overrides, and do
not store traced arrays on it.  Two value-equal ``Robot`` objects with distinct
backend holders simply don't share cached wrappers, which is harmless.

The stateless CUDA paths (FK, IK) do not appear here: they derive their buffers
directly from ``robot.joints.*`` arrays and load a process-global FFI ``.so``
singleton, so they need no per-robot object.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import numpy as onp

if TYPE_CHECKING:
    import yourdfpy
    from jax import Array


def _eager():
    """Context manager forcing eager (compile-time) evaluation of array ops.

    Lets the URDF-dependent backends be *constructed* on their lazy first use
    even when that call happens inside a caller's ``jax.jit`` trace: the build
    performs concrete array math on the URDF, which would otherwise be captured
    as tracers.  Falls back to a no-op on older JAX without the helper.
    """
    try:
        return jax.ensure_compile_time_eval()
    except AttributeError:  # pragma: no cover - very old JAX
        import contextlib

        return contextlib.nullcontext()


class CudaBackends:
    """Per-robot holder that lazily builds URDF-dependent backends.

    Args:
        urdf:                The source URDF, retained so backends can be built
                             on demand.
        parent_indices:      Concrete (host) joint parent-index array, kept so
                             the CUDA IK ancestor masks can be precomputed
                             without materialising the *traced* Robot pytree —
                             this is what makes CUDA IK usable inside ``jax.jit``.
        parent_joint_indices: Concrete (host) link → parent-joint index array.
        num_joints:          Total joint count (mask width).
    """

    def __init__(
        self,
        urdf: "yourdfpy.URDF",
        parent_indices=None,
        parent_joint_indices=None,
        num_joints: int | None = None,
    ) -> None:
        self._urdf = urdf
        # Lazily-populated caches (built on first use).
        self._robot_collision: Any = None            # pure-JAX RobotCollision
        self._robot_collision_spherized: Any = None  # pure-JAX RobotCollisionSpherized
        self._sdf_cuda_collision: Any = None         # CUDA differentiable SDF
        self._binary_cuda_collision: Any = None      # CUDA binary checker
        self._grid: dict[float, Any] = {}            # GRiDDynamics, keyed by gravity

        # Concrete kinematic structure for host-side (trace-free) precompute.
        self._parent_indices = (
            None if parent_indices is None else onp.asarray(parent_indices, dtype=onp.int32)
        )
        self._parent_joint_indices = (
            None
            if parent_joint_indices is None
            else onp.asarray(parent_joint_indices, dtype=onp.int32)
        )
        self._num_joints = num_joints
        # CUDA IK ancestor masks, cached per target-link-index tuple.
        self._ik_masks: dict[tuple[int, ...], tuple] = {}

    @property
    def urdf(self) -> "yourdfpy.URDF":
        return self._urdf

    # ── CUDA IK structural precompute ──────────────────────────────────────

    def ik_ancestor_masks(self, target_link_indices: tuple[int, ...]):
        """Return ``(target_jnts, ancestor_masks)`` as JAX arrays for the given
        end-effector link indices.

        Computed purely from the *concrete* kinematic structure cached at
        construction, so it is safe to call while the owning ``Robot`` is a JAX
        tracer — the CUDA IK solvers can then run inside ``jax.jit``.  Results
        are constant per ``target_link_indices`` and memoised.
        """
        from jax import numpy as jnp

        if self._parent_indices is None or self._parent_joint_indices is None:
            raise RuntimeError(
                "CUDA IK ancestor masks require the concrete kinematic structure; "
                "this Robot was not created via Robot.from_urdf."
            )

        key = tuple(int(i) for i in target_link_indices)
        cached = self._ik_masks.get(key)
        if cached is not None:
            return cached

        n_ee = len(key)
        target_joints = onp.zeros(n_ee, dtype=onp.int32)
        ancestor_masks = onp.zeros((n_ee, self._num_joints), dtype=onp.int32)
        for i, link_idx in enumerate(key):
            j = int(self._parent_joint_indices[link_idx])
            target_joints[i] = j
            while j >= 0:
                ancestor_masks[i, j] = 1
                j = int(self._parent_indices[j])

        out = (jnp.asarray(target_joints), jnp.asarray(ancestor_masks))
        self._ik_masks[key] = out
        return out

    # ── Collision backends ─────────────────────────────────────────────────

    def robot_collision(self) -> Any:
        """Pure-JAX :class:`RobotCollision` model (used when ``use_cuda=False``)."""
        if self._robot_collision is None:
            from .collision import RobotCollision

            # The build does array math on the (concrete) URDF; guard it so it
            # evaluates eagerly even when this lazy first-use happens to occur
            # inside a caller's jax.jit trace (otherwise the parser's arrays
            # become tracers and Python `if array:` checks raise).
            with _eager():
                self._robot_collision = RobotCollision.from_urdf(self._urdf)
        return self._robot_collision

    def robot_collision_spherized(self) -> Any:
        """Pure-JAX :class:`RobotCollisionSpherized` model (sphere-based)."""
        if self._robot_collision_spherized is None:
            from .collision import RobotCollisionSpherized

            with _eager():
                self._robot_collision_spherized = RobotCollisionSpherized.from_urdf(
                    self._urdf
                )
        return self._robot_collision_spherized

    def sdf_collision(self) -> Any:
        """Differentiable CUDA SDF collision checker (``method="sdf"``)."""
        if self._sdf_cuda_collision is None:
            from .collision import make_cuda_checker

            with _eager():
                self._sdf_cuda_collision = make_cuda_checker(self.robot_collision())
        return self._sdf_cuda_collision

    def binary_collision(self) -> Any:
        """Fused CUDA binary (collision-free) checker (``method="binary"``).

        The binary kernel only supports sphere-based models, so it is built from
        a :class:`RobotCollisionSpherized` inner (not the capsule
        :class:`RobotCollision` used by the SDF path).
        """
        if self._binary_cuda_collision is None:
            from .collision import make_cuda_binary_checker

            with _eager():
                self._binary_cuda_collision = make_cuda_binary_checker(
                    self.robot_collision_spherized()
                )
        return self._binary_cuda_collision

    # ── Dynamics backend ───────────────────────────────────────────────────

    def grid(self, gravity: float) -> Any:
        """GRiD CUDA dynamics backend for the given ``gravity``.

        Cached per gravity value, since ``gravity`` is baked into the
        generated kernels at construction time.
        """
        key = float(gravity)
        gd = self._grid.get(key)
        if gd is None:
            from .dynamics import GRiDDynamics

            with _eager():
                gd = GRiDDynamics(self._urdf, gravity=key)
            self._grid[key] = gd
        return gd
