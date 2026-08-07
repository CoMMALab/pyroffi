"""Time-optimal path parameterisation (TOPP-RA).

A geometric path says *where*; this says *when*. Given the ordered
configurations a sampling-based planner returns, TOPP-RA computes the fastest
timing that respects joint velocity, acceleration and actuator torque limits —
which is what makes a planned path executable at speed rather than crawled
through.

Two backends for the torque constraints:

* **pure JAX** — :func:`jax_inverse_dynamics_fn`, portable and differentiable;
* **CUDA via GRiD** — :func:`grid_inverse_dynamics_fn`, which folds a whole
  batch of paths into single RNEA kernel launches.

The solver itself is pure JAX either way, and is jit/vmap-compatible on fixed
tensor shapes: variable-length planner outputs go through :func:`pad_paths` and
carry an ``n_valid`` count.

Quick start::

    from pyroffi import topp

    result = topp.topp_ra(waypoints, vmax, amax, n_grid=128)
    print(result.duration, result.feasible)

See ``examples/19_00_batched_topp_ra.py`` for the batched, torque-limited path.
"""

from ._constraints import (
    Constraints as Constraints,
    acceleration_constraints as acceleration_constraints,
    grid_inverse_dynamics_fn as grid_inverse_dynamics_fn,
    jax_inverse_dynamics_fn as jax_inverse_dynamics_fn,
    torque_constraints as torque_constraints,
    velocity_bound as velocity_bound,
)
from ._path import (
    GeometricPath as GeometricPath,
    make_path as make_path,
    pad_paths as pad_paths,
)
from ._topp_ra import (
    TOPPResult as TOPPResult,
    sample_at_times as sample_at_times,
    solve_topp_ra as solve_topp_ra,
    topp_ra as topp_ra,
    topp_ra_batched as topp_ra_batched,
)
