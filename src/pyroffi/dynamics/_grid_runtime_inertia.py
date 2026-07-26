"""Runtime-mutable inertia for the GRiD path (upstream ``runtime_inertia``).

Why this exists
---------------

The GRiD codegen normally bakes each body's spatial inertia into the generated
header as immediates, so changing a payload would mean regenerating and
recompiling kernels.  ``A2R-Lab/GRiD`` adds a flag-gated **mutable inertia
table**: ``GRiDCodeGenerator(..., runtime_inertia=True)`` emits
``d_inertia_params`` plus a ``set_inertia_params`` host memcpy, and the kernels
rebuild the ``s_XImats`` I-region from it once per launch.  Uploading a new
payload is then a ``10 · NB``-float memcpy — 70 floats for a panda — with no
recompile.

The parameter basis is the win
------------------------------

The table is in the standard inertial-parameter (regressor) basis
``π = [m, h = m·c, I_O]``, with ``I_O`` the six independent entries of the
inertia about the *body origin*.  Spatial inertia is **linear** in ``π``, so for
two rigidly-connected bodies referred to the same origin the composition is a
plain 10-vector add::

    π_total = π_link + π_object(referred to the link origin)

That is cheaper than the 6x6 congruence, exactly linear in mass, and it sums
trivially over several attachments on one body.  It is also the basis sysID and
domain randomization use, so payload identification and tool-use attachment
become the same code path.

We obtain the object's referred ``π`` by building its 6x6 in the target body
frame (``Xᵀ I X``, which :func:`pyroffi.attachments.compose_dynamics` already
does) and reading ``[m, h, I_O]` back out of it — the same extraction upstream's
``Link.get_inertia_params`` performs, and therefore bit-compatible with the
on-device scatter by construction.

The real constraint is purity, not compile time
-----------------------------------------------

``set_inertia_params`` mutates *device-resident global model state* through a
blocking host memcpy.  That is not a traceable JAX value, so:

* set the table only at **grasp-topology boundaries** (pick / place / handoff),
  which is exactly where the static-topology rule already permits a recompile;
* :class:`GridModelState` records what is currently uploaded and **raises if a
  tracer reaches it**.  A silently-stale inertia table is an invisible
  wrong-dynamics bug, so it must fail loudly, keyed on tracer-ness rather than
  on a caller-supplied promise;
* **you cannot ``vmap`` over payloads on this path** — there is one table per
  model.  Batched grasp optimization, payload sweeps and domain randomization
  across a batch stay on the pure-JAX RNEA, which composes inertia as a pytree
  leaf and batches fine.  (A variant taking the params pointer as a *kernel
  argument* rather than reading ``d_robotModel->d_inertia_params`` would make
  this functional and ``vmap``-able; the rebuild already reads through a
  pointer, so it is a small upstream change and worth raising.)
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as onp


def inertia_params_from_spatial(I: onp.ndarray) -> onp.ndarray:
    """Extract ``[m, h(3), I_O(6)]`` from a 6x6 spatial inertia.

    Mirrors upstream ``URDFParser.Link.get_inertia_params`` exactly, reading the
    numbers verbatim out of the 6x6 so the on-device divide-free scatter
    reconstructs the identical matrix.  Layout::

        I = [[ I_O,        skew(h) ],
             [ skew(h)^T,  m * I3  ]]

    with ``I_O`` ordered ``[Ixx, Ixy, Ixz, Iyy, Iyz, Izz]``.
    """
    I = onp.asarray(I, dtype=onp.float64)
    if I.shape[-2:] != (6, 6):
        raise ValueError(f"expected a (..., 6, 6) spatial inertia, got {I.shape}")
    m = I[..., 3, 3]
    # top-right block is skew(h): hx = TR[2,1], hy = TR[0,2], hz = TR[1,0]
    h = onp.stack([I[..., 2, 4], I[..., 0, 5], I[..., 1, 3]], axis=-1)
    I_O = onp.stack(
        [
            I[..., 0, 0],
            I[..., 0, 1],
            I[..., 0, 2],
            I[..., 1, 1],
            I[..., 1, 2],
            I[..., 2, 2],
        ],
        axis=-1,
    )
    return onp.concatenate([m[..., None], h, I_O], axis=-1)


def _inside_a_trace() -> bool:
    """True when we are being staged out (``jit``) rather than run eagerly.

    JAX exposes no stable public predicate for this across versions
    (``jax.core.trace_state_clean`` is gone as of 0.10), so probe it: a trivial
    ``jnp`` operation on constants returns a concrete array at the top level and
    a ``Tracer`` inside a ``jit`` trace. Cheap, and it degrades safely — a
    version where the probe stops firing falls back to the leaf check below.
    """
    return isinstance(jnp.zeros(()) + onp.float32(0), jax.core.Tracer)


_TRACER_MESSAGE = (
    "GRiD's inertia table is device-resident *model* state written by a "
    "blocking host memcpy, so it cannot be set from inside a jit/vmap/grad "
    "trace. Set it at a grasp-topology boundary (pick / place / handoff) "
    "outside the traced region, or use the pure-JAX dynamics path "
    "(Robot.with_attachments), where inertia is a pytree leaf and "
    "batches/differentiates normally."
)


class GridModelState:
    """Guarded owner of one compiled model's device-resident inertia table.

    Holds the currently-uploaded ``π`` so callers can tell what the GPU is
    actually computing with, and refuses to upload traced values.
    """

    def __init__(self, so_path: Path, num_bodies: int, baseline: onp.ndarray):
        lib = ctypes.CDLL(str(so_path))
        try:
            self._set = lib.GridSetInertiaParams
            self._size_fn = lib.GridInertiaParamsSize
        except AttributeError as exc:  # pragma: no cover - build-config error
            raise RuntimeError(
                "This GRiD library was built without runtime_inertia; construct "
                "GRiDDynamics(..., runtime_inertia=True) to get a mutable "
                "inertia table."
            ) from exc
        self._set.restype = None
        self._set.argtypes = [ctypes.POINTER(ctypes.c_float)]
        self._size_fn.restype = ctypes.c_int

        self.num_bodies = int(num_bodies)
        expected = 10 * self.num_bodies
        actual = int(self._size_fn())
        if actual != expected:
            raise RuntimeError(
                f"GRiD inertia table is {actual} floats but this robot has "
                f"{self.num_bodies} bodies ({expected} expected)."
            )
        self.baseline = onp.asarray(baseline, dtype=onp.float64).reshape(
            self.num_bodies, 10
        )
        self.current = self.baseline.copy()

    @staticmethod
    def reject_tracers(*trees) -> None:
        """Raise the guard's error if traced values are anywhere in play.

        Two checks, because either alone leaks a case:

        * any leaf of ``trees`` being a ``Tracer`` catches ``vmap`` / ``grad``
          over a payload, where the traced value arrives as an *input*;
        * ``_inside_a_trace()`` catches ``jit``, where the inputs may all be
          concrete constants closed over from outside but every ``jnp``
          operation on them is nonetheless staged into a jaxpr — so the tracer
          only appears partway through the composition.

        Called at the *entry* of every path that ends in an upload, so the
        failure names the real constraint rather than surfacing deeper in as an
        incidental ``TracerArrayConversionError`` from a numpy conversion.
        """
        if _inside_a_trace() or any(
            isinstance(leaf, jax.core.Tracer)
            for tree in trees
            for leaf in jax.tree_util.tree_leaves(tree)
        ):
            raise TypeError(_TRACER_MESSAGE)

    def upload(self, params: onp.ndarray) -> None:
        """Push a ``(num_bodies, 10)`` parameter table to the device.

        Raises on a traced value rather than silently leaving the GPU running
        stale dynamics — see the module docstring.
        """
        self.reject_tracers(params)
        arr = onp.ascontiguousarray(
            onp.asarray(params, dtype=onp.float32).reshape(self.num_bodies, 10)
        )
        self._set(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        self.current = onp.asarray(params, dtype=onp.float64).reshape(
            self.num_bodies, 10
        )

    def reset(self) -> None:
        """Restore the URDF's own parameters (the baked-equivalent table)."""
        self.upload(self.baseline)

    def add_body_inertia(self, deltas: dict[int, onp.ndarray]) -> None:
        """Upload ``baseline + Σ delta`` for the given body rows.

        ``deltas`` maps a table row index to either a ``(10,)`` parameter vector
        or a ``(6, 6)`` spatial inertia (converted here).  Linearity in ``π`` is
        what makes this a plain add.
        """
        params = self.baseline.copy()
        for row, d in deltas.items():
            d = onp.asarray(d, dtype=onp.float64)
            if d.shape == (6, 6):
                d = inertia_params_from_spatial(d)
            if d.shape != (10,):
                raise ValueError(
                    f"delta for body {row} must be (10,) or (6, 6), got {d.shape}"
                )
            params[row] += d
        self.upload(params)
