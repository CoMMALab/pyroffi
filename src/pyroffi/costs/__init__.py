"""Cost and constraint factories, named and shaped as in PyRoKi's ``pk.costs``.

The factories return :class:`pyroffi.Cost` objects for pyroffi's jaxls-shaped API
(``pk.Var``, ``pk.VarValues``, ``pk.LeastSquaresProblem``, ...). PyRoKi code ports by
replacing ``import pyroki as pk`` with ``import pyroffi as pk`` and ``jaxls.X`` with
``pk.X``::

    problem = pk.LeastSquaresProblem(costs, [joint_var])
    sol = problem.analyze().solve()   # compiled to one CUDA kernel; sol[joint_var]

``analyze()`` compiles to CUDA; there is no JAX solver behind it, and no fallback.
"""

from . import _residuals as _r
from ._native import world_collision_native as world_collision_native  # pyroffi: device-code residual
from ._path_simplify import simplify_paths_batched as simplify_paths_batched
from ._problem import Cost as Cost
from ._residuals import (
    five_point_acceleration_residual as five_point_acceleration_residual,
    five_point_jerk_residual as five_point_jerk_residual,
    five_point_velocity_residual as five_point_velocity_residual,
    limit_residual as limit_residual,
    limit_velocity_residual as limit_velocity_residual,
    manipulability_residual as manipulability_residual,
    pose_residual as pose_residual,
    pose_with_base_residual as pose_with_base_residual,
    rest_residual as rest_residual,
    rest_with_base_residual as rest_with_base_residual,
    self_collision_residual as self_collision_residual,
    smoothness_residual as smoothness_residual,
    world_collision_residual as world_collision_residual,
)


# Pose costs
pose_cost = Cost.factory(_r.pose_residual)
pose_cost_analytic_jac = Cost.factory(_r.pose_masked_residual)
pose_cost_numerical_jac = Cost.factory(_r.pose_numerical_residual)
pose_cost_with_base = Cost.factory(_r.pose_with_base_residual)

# Limit costs
limit_cost = Cost.factory(_r.limit_residual)
limit_velocity_cost = Cost.factory(_r.limit_velocity_residual)

# Regularization costs
rest_cost = Cost.factory(_r.rest_residual)
rest_with_base_cost = Cost.factory(_r.rest_with_base_residual)
smoothness_cost = Cost.factory(_r.smoothness_residual)

# Manipulability cost
manipulability_cost = Cost.factory(_r.manipulability_residual)

# Collision costs
self_collision_cost = Cost.factory(_r.self_collision_residual)
world_collision_cost = Cost.factory(_r.world_collision_residual)

# Finite difference costs
five_point_velocity_cost = Cost.factory(_r.five_point_velocity_residual)
five_point_acceleration_cost = Cost.factory(_r.five_point_acceleration_residual)
five_point_jerk_cost = Cost.factory(_r.five_point_jerk_residual)

# Constraint factories (augmented Lagrangian): residual <= 0 is feasible.
limit_constraint = Cost.factory(kind="constraint_leq_zero")(_r.limit_residual)
limit_velocity_constraint = Cost.factory(kind="constraint_leq_zero")(_r.limit_velocity_residual)
world_collision_constraint = Cost.factory(kind="constraint_leq_zero")(_r.world_collision_residual)
