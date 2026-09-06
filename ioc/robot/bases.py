"""Cost bases for the robot experiments.

A basis is a pair `(residual_fn, feature_names)` where
`residual_fn(x_flat, scene)` returns one residual vector r_k per feature and
phi_k = ||r_k||^2.  Features are written as residuals, not scalars, so the inner
problem stays a nonlinear least-squares problem with a PSD Gauss-Newton Hessian
(see `ioc.inner`).

Kinematic bases
---------------
`k3`   effort, collision, smoothness -- the E1 basis.
`k9`   one effort weight per joint, plus collision and smoothness.
`k16`  per-joint effort *and* smoothness, plus collision and a posture term.

k9/k16 exist for the cost-dimension sweep: they grow K while the trajectory
dimension and the landscape geometry stay fixed, which is the regime where the
per-step solve count is the whole story.  Growing K by adding *structurally
different* features would confound cost dimension with problem difficulty.

`collinear` is a deliberate non-identifiability control: the smoothness residual
is replaced by a second copy of the effort residual.  No method can separate the
two, the recovered weights should be flat along their difference, and the Gram
certificate should show it.

Dynamic basis
-------------
`dynamic` adds an RNEA torque feature, which is what makes the demonstrations
depend on mass, inertia and payload rather than geometry alone.  E3 fits a
kinematic basis to those demonstrations and measures the regret -- the price of
ignoring dynamics.
"""

import jax
import jax.numpy as jnp

DT = 0.1  # [s] timestep between waypoints
GRAVITY = -9.81

K3_NAMES = ("effort", "collision", "smooth")
DYNAMIC_NAMES = ("effort", "collision", "smooth", "torque")

# Franka Panda per-joint torque limits [Nm] (arm joints 1-7).
PANDA_TAU_MAX = jnp.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])


def x_dtype():
    """float64 when x64 is enabled, else float32 -- the graph's working dtype."""
    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


def kinematic(problem, basis="k3", collinear=False):
    """Return `(residual_fn, names)` for one of the kinematic bases."""
    dof = problem.dof

    def residual_fn(x_flat, scene):
        q = problem.unpack(x_flat, scene)
        dq = q[1:] - q[:-1]
        ddq = q[2:] - 2.0 * q[1:-1] + q[:-2]
        r_coll = problem.clearance_residual(q, scene)

        if basis == "k3":
            r_effort, r_smooth = dq.reshape(-1), ddq.reshape(-1)
            if collinear:
                r_smooth = r_effort
            return (r_effort, r_coll, r_smooth)
        if basis == "k9":
            return tuple([dq[:, j] for j in range(dof)] + [r_coll, ddq.reshape(-1)])
        if basis == "k16":
            nominal = 0.5 * (scene.q_start + scene.q_goal)
            return tuple(
                [dq[:, j] for j in range(dof)]
                + [ddq[:, j] for j in range(dof)]
                + [r_coll, (q - nominal).reshape(-1)]
            )
        raise ValueError(basis)

    if basis == "k3":
        names = K3_NAMES
    elif basis == "k9":
        names = tuple(f"effort_j{j}" for j in range(dof)) + ("collision", "smooth")
    elif basis == "k16":
        names = (
            tuple(f"effort_j{j}" for j in range(dof))
            + tuple(f"smooth_j{j}" for j in range(dof))
            + ("collision", "posture")
        )
    else:
        raise ValueError(basis)
    return residual_fn, names


def make_payload(problem, payload_kg):
    """Constant downward force at the last body, as a [torque; force] wrench."""
    if payload_kg <= 0:
        return None
    return jnp.zeros((problem.dof, 6)).at[-1, 5].set(payload_kg * GRAVITY)


def dynamic(problem, payload_wrench=None, torque_backend="grid"):
    """Return `(residual_fn, names)` for the kinematic basis plus RNEA torque.

    `torque_backend="grid"` routes inverse dynamics through the GRiD CUDA FFI.
    GRiD computes in float32 internally, so torques carry ~2.6e-7 relative
    error even when the surrounding graph is float64.  The torque feature is
    whitened by a large scale (~5e5), which shrinks that error's contribution
    to the weighted residual; the remaining effect on the outer gradient is
    measured, not assumed.

    The implicit adjoint's exact Hessian (`ioc.inner`'s `adjoint_hessian="jax"`,
    the only mode) works straight through this FFI: `pyroffi.dynamics
    .GRiDDynamics`'s analytic-gradient kernels carry their own `custom_jvp`
    built from GRiD's `idsva_so` second-order kernel, so `jax.hessian` no
    longer raises on `inverse_dynamics` the way it used to. No float64 twin
    and no once-differentiable-only fallback are needed any more --
    `torque_backend="grid"` runs with x64 on or off, same as `"jax"`.
    """

    def torque_residual(q):
        """RNEA torques at interior knots, from central differences."""
        qd = (q[2:] - q[:-2]) / (2.0 * DT)
        qdd = (q[2:] - 2.0 * q[1:-1] + q[:-2]) / (DT**2)
        qm = q[1:-1]
        f_ext = None
        if payload_wrench is not None:
            f_ext = jnp.broadcast_to(
                payload_wrench, qm.shape[:1] + payload_wrench.shape
            )
        tau = problem.robot.inverse_dynamics(
            qm, qd, qdd, gravity=GRAVITY, use_cuda=(torque_backend == "grid"),
            f_ext=f_ext,
        )
        # Keep the surrounding graph in its own dtype regardless of what the
        # kernel returns, so a float32 backend cannot silently downcast the whole
        # trajectory optimization.
        return tau.reshape(-1).astype(x_dtype())

    def residual_fn(x_flat, scene):
        q = problem.unpack(x_flat, scene)
        r_effort = (q[1:] - q[:-1]).reshape(-1)
        r_smooth = (q[2:] - 2.0 * q[1:-1] + q[:-2]).reshape(-1)
        r_coll = problem.clearance_residual(q, scene)
        return (r_effort, r_coll, r_smooth, torque_residual(q))

    return residual_fn, DYNAMIC_NAMES


def torque_limit_constraint(problem, payload_wrench=None, torque_backend="grid",
                            tau_limit_scale=1.0, rho0=1.0, rho_max=1e4,
                            penalty_scale=3.0):
    """A theta-INDEPENDENT torque-limit inequality for `ioc.inner`'s constrained
    path: `constraints_fn(scene) -> (AugmentedLagrangianTerm,)` enforcing
    ``|tau(q)| <= tau_max`` per joint, where ``tau`` is computed by exactly the
    same central-difference RNEA (same DT, gravity, payload, backend) as the
    `dynamic` basis's torque *feature*, so the hard limit and the soft feature
    are consistent.  ``tau_max = PANDA_TAU_MAX * tau_limit_scale``.

    The limit is a property of the robot, not the cost model, so the same
    constraint is applied to every inner solver (full / kinematic / ref) and to
    demonstration generation."""
    from pyroffi.optimization_engines._trajopt_core import AugmentedLagrangianTerm

    tau_max = PANDA_TAU_MAX[: problem.dof] * tau_limit_scale

    def constraints_fn(scene):
        def residual(x_flat):
            q = problem.unpack(x_flat, scene)
            qd = (q[2:] - q[:-2]) / (2.0 * DT)
            qdd = (q[2:] - 2.0 * q[1:-1] + q[:-2]) / (DT**2)
            qm = q[1:-1]
            f_ext = None
            if payload_wrench is not None:
                f_ext = jnp.broadcast_to(
                    payload_wrench, qm.shape[:1] + payload_wrench.shape
                )
            tau = problem.robot.inverse_dynamics(
                qm, qd, qdd, gravity=GRAVITY,
                use_cuda=(torque_backend == "grid"), f_ext=f_ext,
            )
            return jnp.maximum(0.0, jnp.abs(tau) - tau_max).reshape(-1).astype(x_dtype())

        return (AugmentedLagrangianTerm(
            residual_fn=residual, kind="ineq",
            rho0=rho0, rho_max=rho_max, penalty_scale=penalty_scale,
            name="torque_limit"),)

    return constraints_fn


def rff(problem, M, key, lengthscale=1.0):
    """`M` random-Fourier-feature residuals of a squared-exponential kernel on
    the per-waypoint descriptor `u_t = [q_t, q_{t+1} - q_t]` (Rahimi & Recht).
    An unknown-cost / RKHS basis, for the `basis_size` diagnostic's "hand-
    engineered vs. RKHS" axis -- as `M` grows this is an over-complete
    dictionary the same way `k16` is for the kinematic family, but without a
    named feature identity, so the diagnostic can ask whether recovery quality
    depends on the dictionary being *interpretable* or just *expressive
    enough*.  Ported from `iosp/study3_identifiable_refit.py::make_rff_residual_fn`.
    """
    k1, k2 = jax.random.split(key)
    dof = problem.dof
    Omega = jax.random.normal(k1, (M, 2 * dof), dtype=jnp.float32) / lengthscale
    b = jax.random.uniform(k2, (M,), dtype=jnp.float32) * 2.0 * jnp.pi
    scale = jnp.sqrt(2.0 / M)

    def residual_fn(x_flat, scene):
        q = problem.unpack(x_flat, scene)
        u = jnp.concatenate([q[:-1], q[1:] - q[:-1]], axis=-1)  # (T-1, 2*dof)
        phi = scale * jnp.cos(u @ Omega.T + b)                  # (T-1, M)
        return tuple(phi[:, j] for j in range(M))

    return residual_fn, tuple(f"rff[{j}]" for j in range(M))
