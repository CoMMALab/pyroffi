"""Path-parameterised constraints in TOPP-RA's canonical form.

TOPP-RA's leverage comes from one substitution. Write the trajectory as
``q(s(t))`` and let

    x = sdot^2          (squared path velocity)
    u = sddot           (path acceleration)

Then, by the chain rule,

    qd  = q'(s) * sdot
    qdd = q'(s) * u + q''(s) * x

so joint acceleration is **linear** in ``(u, x)`` at every fixed ``s``. Joint
torque inherits the same structure, because rigid-body dynamics is affine in
``qdd`` and quadratic in ``qd``:

    tau = M(q) qdd + C(q, qd) qd + g(q)
        = [M q'] u + [M q'' + C(q, q') q'] x + g(q)

Both therefore fit ``a(s) u + b(s) x + c(s)`` with two-sided bounds, and the
per-gridpoint feasible set is a polygon in the ``(u, x)`` plane. That is the
entire reason the method is fast: an inherently nonlinear time-optimal control
problem becomes a chain of tiny 2-D linear programs.

Everything in this module produces the same canonical output, so constraint
types compose by concatenation:

    ``A @ [u, x] <= h``     with ``A`` of shape ``(N, m, 2)``, ``h`` of ``(N, m)``

Velocity is the exception and is deliberately *not* expressed that way. It
gives ``|q'| sqrt(x) <= vmax``, i.e. a pure upper bound on ``x`` with no ``u``
in it; carrying it as an explicit box bound rather than 2·DOF polygon rows
keeps the LP smaller and the bound exact.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from ._path import GeometricPath

_QS_EPS = 1e-8
"""Guards ``vmax / |q'|`` where a joint is momentarily stationary along ``s``."""


class Constraints(NamedTuple):
    """Canonical per-gridpoint constraint set.

    ``x_upper`` is separate from the polygon rows because it is the one
    constraint that is a plain interval; see the module docstring.
    """

    A: Float[Array, "*batch N m 2"]
    """Constraint normals, columns ordered ``[u, x]``."""
    h: Float[Array, "*batch N m"]
    """Right-hand sides."""
    x_upper: Float[Array, "*batch N"]
    """Upper bound on ``x = sdot^2`` from velocity limits (``inf`` if unused)."""

    @property
    def n_grid(self) -> int:
        # Right-indexed, so an unvmapped batched build (one GRiD launch over
        # every path at once) and a per-path vmapped one both work.
        return self.A.shape[-3]

    def merge(self, other: "Constraints") -> "Constraints":
        """Intersect two constraint sets defined on the same grid."""
        if self.n_grid != other.n_grid:
            raise ValueError(
                f"grid mismatch: {self.n_grid} vs {other.n_grid} gridpoints"
            )
        return Constraints(
            A=jnp.concatenate([self.A, other.A], axis=-2),
            h=jnp.concatenate([self.h, other.h], axis=-1),
            x_upper=jnp.minimum(self.x_upper, other.x_upper),
        )


def _two_sided(
    a: Float[Array, "*batch N DOF"],
    b: Float[Array, "*batch N DOF"],
    lower: Float[Array, "*batch N DOF"],
    upper: Float[Array, "*batch N DOF"],
) -> tuple[Float[Array, "*batch N m 2"], Float[Array, "*batch N m"]]:
    """Turn ``lower <= a u + b x + c <= upper`` into canonical rows.

    ``c`` is expected to be already folded into ``lower``/``upper`` by the
    caller. Produces ``2 * DOF`` rows per gridpoint: the upper bound as-is and
    the lower bound negated. Axes are addressed from the right so an optional
    leading batch dimension passes through untouched.
    """
    A_up = jnp.stack([a, b], axis=-1)  # (..., N, DOF, 2)
    A = jnp.concatenate([A_up, -A_up], axis=-2)  # (..., N, 2*DOF, 2)
    h = jnp.concatenate([upper, -lower], axis=-1)  # (..., N, 2*DOF)
    return A, h


# ---------------------------------------------------------------------------
# Kinematic constraints
# ---------------------------------------------------------------------------


def velocity_bound(
    path: GeometricPath,
    velocity_limits: Float[Array, " DOF"],
) -> Float[Array, " N"]:
    """Per-gridpoint upper bound on ``x`` from ``|qd| <= vmax``.

    ``|q'_j| sqrt(x) <= vmax_j`` for every joint, so
    ``x <= min_j (vmax_j / |q'_j|)^2``.
    """
    vmax = jnp.abs(jnp.asarray(velocity_limits))
    vmax = jnp.where(vmax > 0.0, vmax, jnp.inf)
    ratio = vmax / jnp.maximum(jnp.abs(path.qs), _QS_EPS)
    return jnp.min(ratio, axis=-1) ** 2


def acceleration_constraints(
    path: GeometricPath,
    acceleration_limits: Float[Array, " DOF"],
    velocity_limits: Float[Array, " DOF"] | None = None,
) -> Constraints:
    """Joint acceleration limits ``|q' u + q'' x| <= amax``.

    Args:
        path: Geometric path with derivatives on the TOPP-RA grid.
        acceleration_limits: ``(DOF,)`` positive ``|qdd|`` bounds.
        velocity_limits: ``(DOF,)`` positive ``|qd|`` bounds. Optional only
            because a torque-limited problem may not want a separate velocity
            cap; omitting it leaves ``x`` bounded solely by the polygon, which
            for a straight path segment means *unbounded*.
    """
    amax = jnp.abs(jnp.asarray(acceleration_limits))
    amax = jnp.where(amax > 0.0, amax, jnp.inf)
    lim = jnp.broadcast_to(amax, path.qs.shape)
    A, h = _two_sided(path.qs, path.qss, -lim, lim)

    if velocity_limits is None:
        x_upper = jnp.full(path.qs.shape[:-1], jnp.inf, dtype=path.q.dtype)
    else:
        x_upper = velocity_bound(path, velocity_limits)
    return Constraints(A=A, h=h, x_upper=x_upper)


# ---------------------------------------------------------------------------
# Torque constraints
# ---------------------------------------------------------------------------

InverseDynamicsFn = Callable[[Array, Array, Array], Array]
"""``(q, qd, qdd) -> tau``, batched over a leading gridpoint axis."""


def torque_constraints(
    path: GeometricPath,
    inverse_dynamics_fn: InverseDynamicsFn,
    torque_lower: Float[Array, " DOF"],
    torque_upper: Float[Array, " DOF"],
) -> Constraints:
    """Actuator torque limits, via three inverse-dynamics evaluations.

    The affine coefficients are recovered from RNEA alone — no mass matrix, no
    Coriolis matrix, no symbolic differentiation:

        c = ID(q, 0,  0  )                 gravity torque
        a = ID(q, 0,  q' ) - c             = M(q) q'
        b = ID(q, q', q'') - c             = M(q) q'' + C(q, q') q'

    The subtractions work because RNEA is affine in ``qdd`` and its
    velocity-product terms vanish at ``qd = 0``. Each call is evaluated at all
    ``N`` gridpoints at once, so a GPU dynamics backend sees three batched
    launches for the whole path rather than ``3N`` scalar calls — this is the
    hook that :func:`grid_inverse_dynamics_fn` plugs into.

    Args:
        path: Geometric path with derivatives on the TOPP-RA grid.
        inverse_dynamics_fn: Batched ``(q, qd, qdd) -> tau``. Anything with
            ``pyroffi``'s dynamics signature works: ``robot.inverse_dynamics``,
            ``GRiDDynamics.inverse_dynamics``, or a closure adding payload
            wrenches.
        torque_lower: ``(DOF,)`` minimum actuator torque (typically ``-effort``).
        torque_upper: ``(DOF,)`` maximum actuator torque.

    Returns:
        A :class:`Constraints` with ``2 * DOF`` rows and no velocity bound
        (``x_upper`` is ``inf``); merge it with
        :func:`acceleration_constraints` to get both.
    """
    zeros = jnp.zeros_like(path.q)
    c = inverse_dynamics_fn(path.q, zeros, zeros)
    a = inverse_dynamics_fn(path.q, zeros, path.qs) - c
    b = inverse_dynamics_fn(path.q, path.qs, path.qss) - c

    lower = jnp.asarray(torque_lower) - c
    upper = jnp.asarray(torque_upper) - c
    A, h = _two_sided(a, b, lower, upper)
    return Constraints(
        A=A, h=h, x_upper=jnp.full(c.shape[:-1], jnp.inf, dtype=path.q.dtype)
    )


def grid_inverse_dynamics_fn(grid_dynamics) -> InverseDynamicsFn:
    """Adapt a :class:`~pyroffi.dynamics.GRiDDynamics` to :func:`torque_constraints`.

    The GRiD FFI kernels are not ``vmap``-able — they batch through a leading
    dimension on their operands instead. This wrapper flattens whatever leading
    axes it is handed into that single dimension and restores the shape after,
    so a ``[B, N, DOF]`` batch of paths becomes **one** kernel launch of
    ``B * N`` states rather than ``B`` launches of ``N``.

    That is the whole point of the CUDA path. At ``B = 64`` paths and
    ``N = 128`` gridpoints each RNEA call covers 8192 states, which is finally
    enough work for the GPU to pay off: building the torque constraints for the
    whole batch takes ~20 ms here against ~1100 ms for the pure-JAX RNEA.
    """

    def _fn(q: Array, qd: Array, qdd: Array) -> Array:
        lead = q.shape[:-1]
        n = q.shape[-1]
        flat = tuple(a.reshape(-1, n).astype(jnp.float32) for a in (q, qd, qdd))
        tau = grid_dynamics.inverse_dynamics(*flat)
        return tau.reshape(*lead, n)

    return _fn


def jax_inverse_dynamics_fn(robot, gravity: float | None = None) -> InverseDynamicsFn:
    """Adapt a pure-JAX :class:`~pyroffi.Robot` to :func:`torque_constraints`.

    Portable and differentiable, and unlike the GRiD wrapper it composes with
    ``vmap`` directly — so a batched solve can either vmap this over paths or
    flatten them, whichever the surrounding code prefers.
    """
    from ..dynamics._api import _require_dynamics
    from ..dynamics._dynamics_jax import _DEFAULT_GRAVITY, inverse_dynamics_jax

    dyn = _require_dynamics(robot)
    g = _DEFAULT_GRAVITY if gravity is None else gravity

    def _fn(q: Array, qd: Array, qdd: Array) -> Array:
        return inverse_dynamics_jax(dyn, q, qd, qdd, g)

    return _fn
