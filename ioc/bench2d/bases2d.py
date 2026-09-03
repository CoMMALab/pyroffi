"""Dynamics-aware residual wrapper for the 2D synthetic-robot benchmarks.

Mirrors `ioc.robot.bases.dynamic`: call the benchmark's own kinematic/geometric
residual function (`field_residuals`, etc.) for the existing residual tuple,
then append one more residual from RNEA inverse dynamics -- velocities and
accelerations by central finite differences on `q`, same convention (`DT`) as
`ioc.robot.bases`.  `problems.py`'s residual functions are untouched; this only
wraps them.

`torque_backend` selects the same GRiD-CUDA/pure-JAX split as
`ioc.robot.bases.dynamic`, through `problem.robot.inverse_dynamics`'s
`use_cuda` flag (`Robot2DProblem.robot` is a `pyroffi.Robot`, not the raw
`GRiDDynamics` wrapper, so both backends are one call): `"grid"` for the fast
CUDA forward solve, `"jax"` for the pure-JAX reference. The implicit adjoint's
exact Hessian works through either -- see `ioc.robot.bases.dynamic`'s
docstring.
"""

from ioc.bench2d import problems as pb

DT = 0.1  # [s] timestep between waypoints -- same convention as ioc.robot.bases


def dynamic(problem, T, cfg, base_residual_fn, torque_backend="grid"):
    """Return `residual_fn(x, ctx)`: `base_residual_fn`'s residuals plus torque.

    `problem` is a `Robot2DProblem` (state-space == joint-space by
    construction, so `pb.unpack` gives `q` directly with no separate FK).
    """

    def residual_fn(x, ctx):
        rs = base_residual_fn(x, ctx, T, cfg)
        q = pb.unpack(x, ctx, T, problem.dof)
        qd = (q[2:] - q[:-2]) / (2.0 * DT)
        qdd = (q[2:] - 2.0 * q[1:-1] + q[:-2]) / (DT**2)
        qm = q[1:-1]
        tau = problem.robot.inverse_dynamics(
            qm, qd, qdd, use_cuda=(torque_backend == "grid"))
        return rs + (tau.reshape(-1),)

    return residual_fn
