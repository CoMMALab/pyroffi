"""Cost and constraint factories, named and shaped as in PyRoKi's ``pk.costs``.

Swapping ``import pyroki as pk`` for ``import pyroffi as pk`` keeps PyRoKi cost code
working: the factories return ordinary ``jaxls.Cost`` objects, so
``jaxls.LeastSquaresProblem(costs, vars).analyze().solve()`` runs as before.

To run the same problem as one CUDA kernel, pass it to :func:`compile`::

    solver = pk.costs.compile(jaxls.LeastSquaresProblem(costs, [joint_var]))
    sol = solver.solve()            # a jaxls.VarValues; sol[joint_var] as usual

Requires ``jaxls`` (``pip install pyroffi[costs]``).
"""

from ._path_simplify import simplify_paths_batched as simplify_paths_batched

try:
    from jaxls import Cost as _Cost
except ImportError:  # pragma: no cover - optional dependency
    _Cost = None

if _Cost is not None:
    from . import _residuals as _r
    from ._compile import CompiledProblem as CompiledProblem
    from ._compile import compile as compile

    # Pose costs
    pose_cost = _Cost.factory(_r.pose_residual)
    pose_cost_analytic_jac = _Cost.factory(_r.pose_masked_residual)
    pose_cost_numerical_jac = _Cost.factory(_r.pose_numerical_residual)
    pose_cost_with_base = _Cost.factory(_r.pose_with_base_residual)

    # Limit costs
    limit_cost = _Cost.factory(_r.limit_residual)
    limit_velocity_cost = _Cost.factory(_r.limit_velocity_residual)

    # Regularization costs
    rest_cost = _Cost.factory(_r.rest_residual)
    rest_with_base_cost = _Cost.factory(_r.rest_with_base_residual)
    smoothness_cost = _Cost.factory(_r.smoothness_residual)

    # Manipulability cost
    manipulability_cost = _Cost.factory(_r.manipulability_residual)

    # Collision costs
    self_collision_cost = _Cost.factory(_r.self_collision_residual)
    world_collision_cost = _Cost.factory(_r.world_collision_residual)

    # Finite difference costs
    five_point_velocity_cost = _Cost.factory(_r.five_point_velocity_residual)
    five_point_acceleration_cost = _Cost.factory(_r.five_point_acceleration_residual)
    five_point_jerk_cost = _Cost.factory(_r.five_point_jerk_residual)

    # Constraint factories (augmented Lagrangian): residual <= 0 is feasible.
    limit_constraint = _Cost.factory(kind="constraint_leq_zero")(_r.limit_residual)
    limit_velocity_constraint = _Cost.factory(kind="constraint_leq_zero")(_r.limit_velocity_residual)
    world_collision_constraint = _Cost.factory(kind="constraint_leq_zero")(_r.world_collision_residual)
else:  # pragma: no cover - optional dependency

    def __getattr__(name):
        raise ImportError(
            f"pyroffi.costs.{name} needs jaxls: pip install 'pyroffi[costs]' "
            "(jaxls @ git+https://github.com/brentyi/jaxls.git)."
        )
