"""Real-time IK tracking through GRiD CUDA dynamics (viser)

The same drag-a-target IK-tracking demo as ``14_02_dynamics_sim_viser``, but the
simulated plant (forward dynamics) and the computed-torque control law (inverse
dynamics) are evaluated with PyRoFFI's CUDA-accelerated GRiD kernels instead of
the pure-JAX backend. Each frame:

  1. solve IK for the dragged target -> desired joint configuration,
  2. compute torques with a computed-torque PD law (GRiD inverse dynamics),
  3. integrate GRiD forward dynamics to advance the simulated state.

Requires a CUDA GPU and nvcc (the first run JIT-compiles the per-robot kernels,
cached under ~/.cache/pyroffi/grid).
"""

import time

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroffi as pk
import yourdfpy
from viser.extras import ViserUrdf

import viser


def main():
    if not any(d.platform == "gpu" for d in jax.devices()):
        raise SystemExit("GRiD dynamics requires a CUDA device (jax GPU backend).")

    # Local Panda URDF: has <inertial> data and no mimic joints (dynamics req).
    urdf = yourdfpy.URDF.load(
        "resources/panda/panda_spherized.urdf", load_meshes=True,
        mesh_dir="resources/panda/meshes",
    )
    robot = pk.Robot.from_urdf(urdf)

    from pyroffi.dynamics import GRiDDynamics

    print("Compiling GRiD CUDA kernels for Panda (cached after first run)...")
    t0 = time.perf_counter()
    gd = GRiDDynamics(urdf)
    print(f"  ready in {time.perf_counter() - t0:.2f} s.")

    target_link_name = "panda_hand"
    target_link_index = robot.links.names.index(target_link_name)
    n = gd.num_dof
    lower, upper = robot.joints.lower_limits, robot.joints.upper_limits
    q0 = ((lower + upper) / 2).astype(jnp.float32)

    # --- Viser scene ------------------------------------------------------
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    ik_target = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.61, 0.0, 0.56), wxyz=(0, 0, 1, 0)
    )

    kp = server.gui.add_slider("Stiffness (Kp)", min=10.0, max=600.0, step=10.0,
                               initial_value=300.0)
    kd = server.gui.add_slider("Damping (Kd)", min=1.0, max=80.0, step=1.0,
                               initial_value=35.0)
    sim_speed = server.gui.add_slider("Sim speed", min=0.25, max=4.0, step=0.25,
                                      initial_value=1.0)
    substeps = server.gui.add_slider("Substeps / frame", min=1, max=128, step=1,
                                     initial_value=8)
    reset_btn = server.gui.add_button("Reset")
    pos_error_handle = server.gui.add_number("Position error (mm)", 0.0, disabled=True)
    rot_error_handle = server.gui.add_number("Rotation error (rad)", 0.0, disabled=True)

    ik_solve = jax.jit(
        lambda pose, key, prev: robot.inverse_kinematics(
            target_link_name=target_link_name,
            target_pose=pose,
            rng_key=key,
            previous_cfg=prev,
        )
    )

    state = {"q": q0, "qd": jnp.zeros(n, jnp.float32)}

    @reset_btn.on_click
    def _(_) -> None:
        state["q"], state["qd"] = q0, jnp.zeros(n, jnp.float32)

    # Computed-torque PD step, driven by the GRiD kernels (see 14_02 for the
    # control-law derivation).
    @jax.jit
    def step(q, qd, q_des, kp_, kd_, dt):
        qdd_des = kp_ * (q_des - q) - kd_ * qd
        tau = gd.inverse_dynamics(q, qd, qdd_des)
        qdd = gd.forward_dynamics(q, qd, tau)
        qd = qd + dt * qdd
        q = q + dt * qd
        q_clamped = jnp.clip(q, lower, upper)
        qd = jnp.where(q_clamped != q, 0.0, qd)
        return q_clamped, qd

    print("Open the viser URL above and drag the target — the arm tracks it")
    print("through GRiD-simulated dynamics (adjust Kp/Kd to feel stiff vs. springy).")

    rng_key = jax.random.PRNGKey(0)
    target_dt = 1.0 / 60.0
    while True:
        t0 = time.perf_counter()

        target_pose = jaxlie.SE3.from_rotation_and_translation(
            rotation=jaxlie.SO3(wxyz=jnp.array(ik_target.wxyz)),
            translation=jnp.array(ik_target.position),
        )
        rng_key, subkey = jax.random.split(rng_key)
        q_des = ik_solve(target_pose, subkey, state["q"]).astype(jnp.float32)

        # Advance sim_speed * (1/60) s of simulated time per rendered frame,
        # split into `substeps` integration steps for stability.
        n_sub = int(substeps.value)
        dt = jnp.float32(target_dt * float(sim_speed.value) / n_sub)
        q, qd = state["q"], state["qd"]
        for _ in range(n_sub):
            q, qd = step(q, qd, q_des, jnp.float32(kp.value), jnp.float32(kd.value), dt)
        state["q"], state["qd"] = q, qd

        actual_pose = jaxlie.SE3(robot.forward_kinematics(q)[target_link_index])
        pos_err = jnp.linalg.norm(actual_pose.translation() - target_pose.translation())
        rot_err = jnp.linalg.norm(
            (target_pose.rotation().inverse() @ actual_pose.rotation()).log()
        )
        pos_error_handle.value = float(pos_err) * 1000
        rot_error_handle.value = float(rot_err)

        urdf_vis.update_cfg(np.asarray(q))

        elapsed = time.perf_counter() - t0
        if elapsed < target_dt:
            time.sleep(target_dt - elapsed)


if __name__ == "__main__":
    main()
