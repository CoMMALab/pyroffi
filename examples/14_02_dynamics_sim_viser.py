"""Real-time IK tracking through real MuJoCo dynamics (mjviser)

Like the interactive IK examples, you drag a target pose and the arm follows
— but instead of *snapping* to the IK solution each frame, the robot is driven
there by its own equations of motion. Every physics step:

  1. solve IK for the dragged target to get a desired joint configuration
     (recomputed whenever the target moves),
  2. compute control torques with a computed-torque (inverse-dynamics) PD law,
     using pyroffi's pure-JAX dynamics model, and
  3. apply those torques to a *real* MuJoCo simulation (stepped by
     ``mujoco.mj_step``) — an independent physics engine, not pyroffi's own
     integrator. This makes the demo an actual ground-truth test of pyroffi's
     dynamics model: if the URDF's inertial parameters or pyroffi's
     Newton-Euler math disagreed with MuJoCo, the gravity/Coriolis
     compensation would visibly fail (the arm sags or drifts) instead of
     tautologically tracking pyroffi's own simulated plant.

MuJoCo loads the same URDF pyroffi uses, so both see the same kinematic tree,
masses, and inertias. Visualized with ``mjviser``. See ``14_00``/``14_01`` for
the non-visual dynamics API, and ``14_03`` for the GRiD-CUDA control law.
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
    # Local Panda URDF: has <inertial> data and no mimic joints (dynamics req).
    urdf = yourdfpy.URDF.load(
        URDF_PATH, load_meshes=True, mesh_dir="resources/panda/meshes",
    )
    robot = pk.Robot.from_urdf(urdf)
    if robot.dynamics is None:
        raise SystemExit("URDF lacks <inertial> data; dynamics unavailable.")

    # MuJoCo loads the *same* URDF as ground truth for the physics step.
    mj_model = mujoco.MjModel.from_xml_path(URDF_PATH)
    mj_data = mujoco.MjData(mj_model)
    assert mj_model.nq == robot.dynamics.num_dof, (
        "MuJoCo and pyroffi disagree on DOF count -- check the URDF for "
        "joints MuJoCo can't parse (mimic joints, unsupported joint types)."
    )

    target_link_name = "panda_hand"
    target_link_index = robot.links.names.index(target_link_name)
    lower, upper = robot.joints.lower_limits, robot.joints.upper_limits
    q0 = (lower + upper) / 2
    mj_data.qpos[:] = np.asarray(q0)
    mujoco.mj_forward(mj_model, mj_data)

    # --- Viser scene (shared between mjviser's Viewer and our own controls) --
    server = viser.ViserServer()
    ik_target = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.61, 0.0, 0.56), wxyz=(0, 0, 1, 0)
    )

    # PD tracking gains (computed-torque law). Higher Kp = stiffer/snappier
    # tracking; Kd damps overshoot. Kd ~ 2*sqrt(Kp) is roughly critically damped.
    kp = server.gui.add_slider("Stiffness (Kp)", min=10.0, max=600.0, step=10.0,
                               initial_value=300.0)
    kd = server.gui.add_slider("Damping (Kd)", min=1.0, max=80.0, step=1.0,
                               initial_value=35.0)
    pos_error_handle = server.gui.add_number("Position error (mm)", 0.0, disabled=True)
    rot_error_handle = server.gui.add_number("Rotation error (rad)", 0.0, disabled=True)

    # --- IK solve (jitted, warm-started from the current MuJoCo configuration) --
    ik_solve = jax.jit(
        lambda pose, key, prev: robot.inverse_kinematics(
            target_link_name=target_link_name,
            target_pose=pose,
            rng_key=key,
            previous_cfg=prev,
        )
    )

    # Computed-torque PD step, evaluated with pyroffi's dynamics model:
    #   qdd_des = Kp (q_des - q) - Kd qd
    #   tau     = inverse_dynamics(q, qd, qdd_des)  (feedback-linearizing +
    #                                                gravity/Coriolis comp)
    # MuJoCo then integrates the *real* plant with these torques applied.
    control_step = jax.jit(
        lambda q, qd, q_des, kp_, kd_: robot.inverse_dynamics(
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
        state["q_des"] = ik_solve(target_pose, subkey, jnp.asarray(mj_data.qpos))
        state["target_pose"] = target_pose

    ik_target.on_update(_resolve_ik)
    _resolve_ik()

    def step_fn(model, data):
        q = jnp.asarray(data.qpos)
        qd = jnp.asarray(data.qvel)
        tau = control_step(q, qd, state["q_des"], kp.value, kd.value)
        data.qfrc_applied[:] = np.asarray(tau)
        mujoco.mj_step(model, data)

        # Report how close the *MuJoCo-simulated* end effector is to the target.
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
    print("through real MuJoCo physics (adjust Kp/Kd to feel stiff vs. springy).")

    viewer = mjviser.Viewer(
        mj_model, mj_data, step_fn=step_fn, reset_fn=reset_fn, server=server,
    )
    viewer.run()


if __name__ == "__main__":
    main()
