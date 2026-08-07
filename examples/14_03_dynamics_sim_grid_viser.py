"""Real-time IK tracking through real MuJoCo dynamics, GRiD control law (mjviser)

The same drag-a-target IK-tracking demo as ``14_02_dynamics_sim_viser``, but the
computed-torque control law (inverse dynamics) is evaluated with PyRoFFI's
CUDA-accelerated GRiD kernels instead of the pure-JAX backend. As in ``14_02``,
the actual plant being tracked is a *real* MuJoCo simulation (stepped by
``mujoco.mj_step``), not pyroffi's own integrator — so this is a genuine
ground-truth test of GRiD's dynamics model against an independent physics
engine. Each physics step:

  1. solve IK for the dragged target -> desired joint configuration
     (recomputed whenever the target moves),
  2. compute torques with a computed-torque PD law (GRiD inverse dynamics),
  3. apply those torques to MuJoCo and step its own integrator.

Requires a CUDA GPU and nvcc (the first run JIT-compiles the per-robot kernels,
cached under ~/.cache/pyroffi/grid).
"""

import jax
import jax.numpy as jnp
import jaxlie
import mujoco
import numpy as np
import pyroffi as pk
import yourdfpy

import mjviser
import viser

URDF_PATH = "resources/panda/panda_spherized.urdf"


def main():
    if not any(d.platform == "gpu" for d in jax.devices()):
        raise SystemExit("GRiD dynamics requires a CUDA device (jax GPU backend).")

    # Local Panda URDF: has <inertial> data and no mimic joints (dynamics req).
    urdf = yourdfpy.URDF.load(
        URDF_PATH, load_meshes=True, mesh_dir="resources/panda/meshes",
    )
    robot = pk.Robot.from_urdf(urdf)

    from pyroffi.dynamics import GRiDDynamics

    print("Compiling GRiD CUDA kernels for Panda (cached after first run)...")
    import time
    t0 = time.perf_counter()
    gd = GRiDDynamics(urdf)
    print(f"  ready in {time.perf_counter() - t0:.2f} s.")

    # MuJoCo loads the *same* URDF as ground truth for the physics step.
    mj_model = mujoco.MjModel.from_xml_path(URDF_PATH)
    mj_data = mujoco.MjData(mj_model)
    assert mj_model.nq == gd.num_dof, (
        "MuJoCo and GRiD disagree on DOF count -- check the URDF for "
        "joints MuJoCo can't parse (mimic joints, unsupported joint types)."
    )

    target_link_name = "panda_hand"
    target_link_index = robot.links.names.index(target_link_name)
    lower, upper = robot.joints.lower_limits, robot.joints.upper_limits
    q0 = ((lower + upper) / 2).astype(jnp.float32)
    mj_data.qpos[:] = np.asarray(q0)
    mujoco.mj_forward(mj_model, mj_data)

    # --- Viser scene (shared between mjviser's Viewer and our own controls) --
    server = viser.ViserServer()
    ik_target = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.61, 0.0, 0.56), wxyz=(0, 0, 1, 0)
    )

    kp = server.gui.add_slider("Stiffness (Kp)", min=10.0, max=600.0, step=10.0,
                               initial_value=300.0)
    kd = server.gui.add_slider("Damping (Kd)", min=1.0, max=80.0, step=1.0,
                               initial_value=35.0)
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

    # Computed-torque PD step, driven by the GRiD kernels (see 14_02 for the
    # control-law derivation). MuJoCo integrates the real plant.
    control_step = jax.jit(
        lambda q, qd, q_des, kp_, kd_: gd.inverse_dynamics(
            q, qd, kp_ * (q_des - q) - kd_ * qd
        )
    )

    state = {"q_des": q0, "target_pose": None, "rng_key": jax.random.PRNGKey(0)}

    def _resolve_ik(_=None) -> None:
        target_pose = jaxlie.SE3.from_rotation_and_translation(
            rotation=jaxlie.SO3(wxyz=jnp.array(ik_target.wxyz)),
            translation=jnp.array(ik_target.position),
        )
        state["rng_key"], subkey = jax.random.split(state["rng_key"])
        q_des = ik_solve(target_pose, subkey, jnp.asarray(mj_data.qpos))
        state["q_des"] = q_des.astype(jnp.float32)
        state["target_pose"] = target_pose

    ik_target.on_update(_resolve_ik)
    _resolve_ik()

    def step_fn(model, data):
        q = jnp.asarray(data.qpos, dtype=jnp.float32)
        qd = jnp.asarray(data.qvel, dtype=jnp.float32)
        tau = control_step(
            q, qd, state["q_des"], jnp.float32(kp.value), jnp.float32(kd.value)
        )
        data.qfrc_applied[:] = np.asarray(tau)
        mujoco.mj_step(model, data)

        actual_pose = jaxlie.SE3(robot.forward_kinematics(q)[target_link_index])
        target_pose = state["target_pose"]
        pos_err = jnp.linalg.norm(actual_pose.translation() - target_pose.translation())
        rot_err = jnp.linalg.norm(
            (target_pose.rotation().inverse() @ actual_pose.rotation()).log()
        )
        pos_error_handle.value = float(pos_err) * 1000
        rot_error_handle.value = float(rot_err)

    def reset_fn(model, data):
        data.qpos[:] = np.asarray(q0)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

    print("Open the viser URL above and drag the target — the arm tracks it")
    print("through real MuJoCo physics, driven by GRiD (adjust Kp/Kd to feel")
    print("stiff vs. springy).")

    viewer = mjviser.Viewer(
        mj_model, mj_data, step_fn=step_fn, reset_fn=reset_fn, server=server,
    )
    viewer.run()


if __name__ == "__main__":
    main()
