"""Rigid body dynamics (pure-JAX)

Demonstrates PyRoFFI's pure-JAX rigid body dynamics: inverse dynamics (RNEA),
forward dynamics, and the joint-space mass matrix (CRBA). These run on any JAX
backend (CPU/GPU/TPU) and are fully differentiable and vmap/jit-compatible.

See ``14_01_dynamics_grid.py`` for the CUDA-accelerated GRiD backend.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pyroffi as pk
import yourdfpy


def main():
    # Use the local Panda URDF: it carries <inertial> data and (unlike
    # robot_descriptions' panda) has no mimic joints, which dynamics requires.
    urdf = yourdfpy.URDF.load(
        "resources/panda/panda_spherized.urdf", load_meshes=False
    )
    robot = pk.Robot.from_urdf(urdf)

    if robot.dynamics is None:
        raise SystemExit("URDF lacks <inertial> data; dynamics unavailable.")

    n = robot.dynamics.num_dof
    print(f"Loaded Panda with {n} actuated DOF.\n")

    # A sample state: configuration q, velocity qd, acceleration qdd.
    q = (robot.joints.lower_limits + robot.joints.upper_limits) / 2
    qd = jnp.zeros(n)
    qdd = jnp.zeros(n)

    # --- Inverse dynamics: torques that realize qdd at state (q, qd) -------
    # With qd = qdd = 0 this is exactly the gravity-compensation torque.
    tau_gravity = robot.inverse_dynamics(q, qd, qdd)
    print("Gravity-compensation torques (Nm):")
    print(np.asarray(tau_gravity), "\n")

    # --- Mass matrix M(q) (CRBA) ------------------------------------------
    M = robot.mass_matrix(q)
    print(f"Mass matrix M(q) is {M.shape}, symmetric-PD "
          f"(min eigenvalue {float(jnp.linalg.eigvalsh(M).min()):.4f}).\n")

    # --- Forward dynamics: accelerations produced by a torque -------------
    tau = tau_gravity + jnp.ones(n)  # gravity comp plus 1 Nm on every joint.
    qdd_out = robot.forward_dynamics(q, qd, tau)
    print("Accelerations from (gravity-comp + 1 Nm) torque (rad/s^2):")
    print(np.asarray(qdd_out), "\n")

    # Round-trip check: inverse of forward dynamics recovers the torque.
    tau_rt = robot.inverse_dynamics(q, qd, qdd_out)
    print(f"ID(FD(tau)) round-trip max error: "
          f"{float(jnp.abs(tau_rt - tau).max()):.2e}\n")

    # --- Batching with vmap ------------------------------------------------
    keys = jax.random.split(jax.random.PRNGKey(0), 8)
    q_batch = jax.vmap(lambda k: jax.random.uniform(
        k, (n,), minval=robot.joints.lower_limits,
        maxval=robot.joints.upper_limits))(keys)
    tau_batch = jax.vmap(lambda qi: robot.inverse_dynamics(
        qi, jnp.zeros(n), jnp.zeros(n)))(q_batch)
    print(f"vmap over 8 configs -> torques batch shape {tau_batch.shape}\n")

    # --- Differentiability -------------------------------------------------
    # d(torque)/d(configuration): the gradient of a scalar torque-effort cost.
    def effort(q_):
        return jnp.sum(robot.inverse_dynamics(q_, qd, qdd) ** 2)

    grad_q = jax.grad(effort)(q)
    print("d/dq of sum-of-squared gravity torques (differentiable):")
    print(np.asarray(grad_q))

    # --- Simple forward simulation (semi-implicit Euler) ------------------
    print("\nSimulating 0.5 s of free fall under gravity (no control)...")
    dt = 0.01
    state_q, state_qd = q, jnp.zeros(n)

    @jax.jit
    def step(carry, _):
        qc, qdc = carry
        acc = robot.forward_dynamics(qc, qdc, jnp.zeros(n))
        qdc = qdc + dt * acc
        qc = qc + dt * qdc
        return (qc, qdc), qc

    (state_q, state_qd), _ = jax.lax.scan(step, (state_q, state_qd), None, length=50)
    print(f"Final joint speed magnitude: {float(jnp.linalg.norm(state_qd)):.3f} rad/s")


if __name__ == "__main__":
    main()
