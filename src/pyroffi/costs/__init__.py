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
from ._path_simplify import simplify_paths_batched as simplify_paths_batched
from ._problem import Cost as _Cost

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
