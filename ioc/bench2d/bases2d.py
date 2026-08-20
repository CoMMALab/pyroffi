"""Dynamics-aware residual wrapper for the 2D synthetic-robot benchmarks.

Mirrors `ioc.robot.bases.dynamic`: call the benchmark's own kinematic/geometric
residual function (`field_residuals`, etc.) for the existing residual tuple,
then append one more residual from `problem.grid.inverse_dynamics(q, qd, qdd)`
-- velocities and accelerations by central finite differences on `q`, same
convention (`DT`) as `ioc.robot.bases`.  `problems.py`'s residual functions are
untouched; this only wraps them.
"""

from ioc.bench2d import problems as pb

DT = 0.1  # [s] timestep between waypoints -- same convention as ioc.robot.bases


def dynamic(problem, T, cfg, base_residual_fn):
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
        tau = problem.grid.inverse_dynamics(qm, qd, qdd)
        return rs + (tau.reshape(-1),)

    return residual_fn
