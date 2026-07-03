"""Rigid body dynamics (CUDA / GRiD-generated kernels)

Demonstrates PyRoFFI's CUDA-accelerated dynamics backend. GRiDDynamics
JIT-generates, compiles, and registers per-robot CUDA kernels (cached under
~/.cache/pyroffi/grid) exposed through the JAX FFI. Its inverse/forward
dynamics carry analytic-gradient backward passes — the main payoff for
trajectory-optimization inner loops.

Requires a CUDA GPU and nvcc. See ``14_00_dynamics_jax.py`` for the portable
pure-JAX backend and a walkthrough of the same operations.
"""

import time

import jax
import jax.numpy as jnp
import pyroffi as pk
import yourdfpy


def main():
    if not any(d.platform == "gpu" for d in jax.devices()):
        raise SystemExit("GRiD dynamics requires a CUDA device (jax GPU backend).")

    # Local Panda URDF: has <inertial> data and no mimic joints (dynamics req).
    urdf = yourdfpy.URDF.load(
        "resources/panda/panda_spherized.urdf", load_meshes=False
    )
    robot = pk.Robot.from_urdf(urdf)

    # Build the GRiD backend directly. The first construction JIT-compiles the
    # per-robot CUDA kernels (slow once, then cached on disk).
    from pyroffi.dynamics import GRiDDynamics

    print("Compiling GRiD CUDA kernels for Panda (cached after first run)...")
    t0 = time.perf_counter()
    gd = GRiDDynamics(urdf)
    print(f"  ready in {time.perf_counter() - t0:.2f} s, {gd.num_dof} DOF.\n")

    n = gd.num_dof
    key = jax.random.PRNGKey(0)
    kq, kqd, kqdd = jax.random.split(key, 3)

    # NOTE: the FFI kernels do not support jax.vmap; batch instead via a
    # leading dimension on the inputs (here a batch of 4096 states).
    B = 4096
    q = jax.random.uniform(kq, (B, n), minval=robot.joints.lower_limits,
                           maxval=robot.joints.upper_limits)
    qd = jax.random.normal(kqd, (B, n))
    qdd = jax.random.normal(kqdd, (B, n))

    # --- Inverse dynamics (batched) ---------------------------------------
    tau = gd.inverse_dynamics(q, qd, qdd)
    tau.block_until_ready()
    print(f"Batched inverse dynamics: {q.shape} -> torques {tau.shape}")

    # --- Forward dynamics + mass matrix inverse ---------------------------
    qdd_out = gd.forward_dynamics(q, qd, tau)
    Minv = gd.mass_matrix_inv(q)
    print(f"Forward dynamics -> {qdd_out.shape}, Minv -> {Minv.shape}")
    print(f"FD/ID round-trip max error: "
          f"{float(jnp.abs(qdd_out - qdd).max()):.2e}\n")

    # --- Agreement with the pure-JAX reference ----------------------------
    tau_ref = robot.inverse_dynamics(q, qd, qdd)  # pure-JAX path
    print(f"GRiD vs pure-JAX inverse dynamics max diff: "
          f"{float(jnp.abs(tau - tau_ref).max()):.2e}\n")

    # --- Analytic gradients ------------------------------------------------
    # The GRiD backend supplies analytic [d tau/dq | d tau/dqd] directly.
    G = gd.inverse_dynamics_gradient(q, qd, qdd)  # (B, n, 2n)
    print(f"Analytic inverse-dynamics gradient shape: {G.shape}")

    # ...and inverse/forward dynamics are reverse-mode differentiable, using
    # those analytic kernels under the hood.
    def effort(q_):
        return jnp.sum(gd.inverse_dynamics(q_, qd, qdd) ** 2)

    grad_q = jax.grad(effort)(q)
    print(f"jax.grad through GRiD inverse dynamics -> {grad_q.shape}\n")

    # --- Timing vs pure-JAX ------------------------------------------------
    f_cuda = jax.jit(gd.forward_dynamics)
    f_jax = jax.jit(lambda q_, qd_, u_: robot.forward_dynamics(q_, qd_, u_))
    for f in (f_cuda, f_jax):
        f(q, qd, tau).block_until_ready()  # warm-up / compile.

    def bench(f):
        t = time.perf_counter()
        for _ in range(50):
            out = f(q, qd, tau)
        out.block_until_ready()
        return (time.perf_counter() - t) / 50 * 1e3

    print(f"Forward dynamics over {B} states, mean of 50 calls:")
    print(f"  GRiD CUDA : {bench(f_cuda):.3f} ms")
    print(f"  pure-JAX  : {bench(f_jax):.3f} ms")


if __name__ == "__main__":
    main()
