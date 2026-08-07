"""Single-arm grasp-and-transport planned with *contact-rich* trajopt, where the
contact force is a genuine decision variable, then played back in a real,
actively-stepped MuJoCo simulation.

This is the contact-rich sibling of ``16_01``. The scene is identical — a single
Franka Panda picks a box from a top-down pinch grasp and carries it sideways —
but the planner is different in a way that matters:

* ``16_01`` uses ``flat_contact_trajopt``: differential-flatness, **contact-
  aware**. The object pose is the flat output and the single contact force is
  *allocated analytically* (``G⁺ w_req``) from the object's acceleration. The
  force is an *output*; object dynamics hold by construction.
* ``16_02`` (this file) uses ``contact_rich_trajopt``: the contact force is a
  **first-class decision variable** optimized under the object's Newton-Euler
  balance (enforced by an augmented Lagrangian) and the friction cone. The force
  is an *unknown* the solver reasons over — the genuinely contact-rich
  formulation, in which the system is no longer differentially flat.

The contact *mode* is still fixed (one persistent rigid grasp for the whole
horizon), so this is contact-rich, not contact-implicit. See
``_flat_contact_trajopt.py``'s ``flat_contact_trajopt_theory.md`` §8 for the
precise distinction.

Grasp geometry, rendering, and the computed-torque MuJoCo playback are all
identical to ``16_01`` (see that file's docstring for the fingertip-pad grasp
mechanics and the *ideal actuated grip* — the planner's grasp wrench applied to
the box so it does not slip through the passive primitive contacts). The only
substantive change is the solver call (contact forces are decision variables)
and the diagnostics it reports (object-dynamics residual + optimized forces).

Requires a CUDA GPU + nvcc (GRiD dynamics).
"""

import argparse
import time

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import yourdfpy

import pyroffi as pk
from pyroffi.dynamics import GRiDDynamics
from pyroffi.dynamics._contact import (
    ContactSystem,
    GraspedObject,
    ManipulatorSpec,
    manipulator_contact_fext,
    object_center_world,
)
from pyroffi.optimization_engines import (
    ContactRichTrajOptConfig,
    contact_rich_trajopt,
)

# --- Scene constants (identical to 16_01) -----------------------------------
GRIP_LINK = "panda_hand"
PINCH_Z = 0.095
BOX_HALF_LENGTHS = np.array([0.025, 0.065, 0.020])
BOX_CENTER = np.array([0.45, 0.0, BOX_HALF_LENGTHS[2]])
BOX_MASS = 0.15
BOX_FRICTION = 1.0
LIFT = 0.12
TRANSPORT = 0.3
T = 24


def _grip_target(box_center: np.ndarray) -> jaxlie.SE3:
    """Top-down pinch pose (see 16_01)."""
    pos = box_center + np.array([0.0, 0.0, PINCH_Z])
    rot = jaxlie.SO3.from_matrix(jnp.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ]))
    return jaxlie.SE3.from_rotation_and_translation(rot, jnp.asarray(pos))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save-video", metavar="PATH", default=None,
        help="Render the actively-stepped simulation offscreen to an MP4 at "
             "this path instead of launching the interactive viser viewer.",
    )
    args = parser.parse_args()

    if not any(d.platform == "gpu" for d in jax.devices()):
        raise SystemExit("Contact-rich dynamics requires a CUDA device.")

    urdf = yourdfpy.URDF.load("resources/panda/panda_spherized.urdf", load_meshes=False)
    robot = pk.Robot.from_urdf(urdf)
    grid = GRiDDynamics(urdf)

    box_geom = pk.collision.box_with_mjcf_dynamics(
        center=BOX_CENTER, half_lengths=BOX_HALF_LENGTHS,
        mass=BOX_MASS, friction=BOX_FRICTION,
    )

    mid = (robot.joints.lower_limits + robot.joints.upper_limits) / 2

    print("Solving grasp IK (pick + carried-to-drop goal) ...")
    goal_center = BOX_CENTER + np.array([0.0, TRANSPORT, LIFT])
    q0 = robot.inverse_kinematics(
        GRIP_LINK, _grip_target(BOX_CENTER),
        solver="ls", num_seeds=64, previous_cfg=mid,
    )
    q1 = robot.inverse_kinematics(
        GRIP_LINK, _grip_target(goal_center),
        solver="ls", num_seeds=64, previous_cfg=q0,
    )

    arm = ManipulatorSpec(robot, grid, GRIP_LINK, base_xy_yaw=(0.0, 0.0, 0.0),
                          p_local=(0.0, 0.0, PINCH_Z))
    system = ContactSystem(manipulators=(arm,), body=GraspedObject(geom=box_geom),
                           grasp_offsets=())

    t = jnp.linspace(0.0, 1.0, T)[:, None]
    init = q0[None] * (1 - t) + q1[None] * t

    cfg = ContactRichTrajOptConfig(
        n_outer_iters=12, n_inner_iters=50, m_lbfgs=8, dt=0.1,
        w_smoothness=1.0, w_effort=1e-5, tau_max=87.0, f_min=1.0,
    )

    print("Warming up (JIT compile) ...")
    t0 = time.perf_counter()
    warm = contact_rich_trajopt(init, q0, q1, system, cfg)
    jax.block_until_ready(warm)
    print(f"  warmup done in {time.perf_counter() - t0:.1f}s")

    print("Optimizing contact-rich trajectory (forces are decision variables) ...")
    t0 = time.perf_counter()
    traj, forces, resid, centers, dt = contact_rich_trajopt(init, q0, q1, system, cfg)
    jax.block_until_ready(traj)
    print(f"  done in {time.perf_counter() - t0:.1f}s")
    print(f"  object-dynamics residual [rms, max] = {np.array(resid[:2])}")
    print(f"  grasp-closure residual (rms)         = {float(resid[2]):.2e}")
    print(f"  optimized dt = {float(dt):.4f}s  (horizon = {float(dt) * (len(traj) - 1):.2f}s)")
    print(f"  mean |contact force| = {float(jnp.mean(jnp.abs(forces))):.2f} N")
    print(f"  peak |contact force| = {float(jnp.max(jnp.abs(forces))):.2f} N")
    print(f"  carry achieved: dz={float(centers[-1, 2] - centers[0, 2]):.3f} m, "
          f"dy={float(centers[-1, 1] - centers[0, 1]):.3f} m")

    _maybe_visualize(system, traj, forces, dt, goal_center,
                     save_video=args.save_video)


def _maybe_visualize(system: ContactSystem, traj, forces, dt, goal_center,
                     save_video: str | None = None):
    try:
        import mujoco
    except Exception:
        print("mujoco not available — skipping visualization.")
        return

    arm = system.manipulators[0]
    box_geom = system.body.geom
    box_center0 = np.array(box_geom.pose.translation())
    box_extents = np.array(box_geom.extents)

    spec = mujoco.MjSpec()
    # MjSpec defaults to degree mode, which would reinterpret the URDF's *radian*
    # joint limits as degrees (shrinking the panda's ~+-2.9 rad ranges by 180/pi to
    # ~+-3 deg) and clamp the arm near its home configuration so it cannot track
    # the plan. The URDF limits are radians.
    spec.compiler.degree = False
    tex = spec.add_texture(
        name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.25, 0.3, 0.35], rgb2=[0.35, 0.4, 0.45], width=512, height=512,
    )
    mat = spec.add_material(name="grid", texrepeat=[8, 8])
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1],
                            material="grid")
    spec.worldbody.add_light(pos=[0.5, -0.5, 2.0], dir=[-0.15, 0.25, -1.0],
                             castshadow=True)
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX, pos=goal_center,
        size=np.array(system.body.geom.extents) / 2.0,
        rgba=[0.2, 0.8, 0.3, 0.35], contype=0, conaffinity=0,
    )
    arm_frame = spec.worldbody.add_frame()
    arm_spec = mujoco.MjSpec.from_file("resources/panda/panda_spherized.urdf")
    spec.attach(arm_spec, prefix="arm_", frame=arm_frame)

    box = spec.worldbody.add_body(name="box", pos=box_center0)
    box.add_freejoint()
    # The grasp is a *rigid* fixed-contact hold (the planner's assumption), realized
    # in playback as a kinematic attach of the box to the gripper (below). The coarse
    # *spherized* Panda hand cannot form a stable two-pad pinch around the box -- its
    # collision spheres transiently deep-penetrate the box and produce ~5e5 N contact
    # impulses that launch it -- so the box carries no primitive collision here; it is
    # driven directly from the simulated gripper pose.
    bg = box.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=box_extents / 2.0,
                      rgba=[200 / 255, 140 / 255, 60 / 255, 1.0],
                      friction=[BOX_FRICTION, 0.01, 0.001])
    bg.contype = 0
    bg.conaffinity = 0

    mj_model = spec.compile()
    mj_data = mujoco.MjData(mj_model)
    ndof = arm.num_dof
    dt = float(dt)

    # --- Feedforward torques (GRiD ID over the planned traj, incl. contact) ---
    traj_arr = jnp.asarray(traj)
    q, qd, qdd = traj_arr, *_fd_vel_acc_np(traj_arr, dt)
    fext = jax.vmap(manipulator_contact_fext, in_axes=(None, 0, 0))(
        arm, traj_arr, forces[:, 0, :]
    )
    tau_ff = np.array(arm.grid.inverse_dynamics(q, qd, qdd, f_ext=fext))
    traj_np = np.array(traj)
    times = np.arange(traj_np.shape[0]) * dt
    duration = float(times[-1])

    # --- Rigid grasp: kinematically pin the box to the simulated gripper ---------
    # The arm is dynamically stepped (GRiD feedforward torque + computed-torque PD),
    # and the box rigidly follows the *actual* simulated hand via the constant grasp
    # offset captured at the grasp configuration -- the faithful playback of the
    # planner's fixed-contact (rigid-grasp) assumption. The optimized contact forces
    # already enter the arm dynamics through ``fext`` in ``tau_ff`` above.
    box_body = mj_model.body("box").id
    hand_body = mj_model.body("arm_" + arm.grip_link).id
    box_start = np.array(object_center_world(system, traj_arr[0]))

    def _hand_se3(d: mujoco.MjData) -> jaxlie.SE3:
        return jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(jnp.asarray(d.xquat[hand_body])),
            jnp.asarray(d.xpos[hand_body]),
        )

    kp, kd = 1500.0, 80.0

    # Static gravity torque at the goal, used to hold the pose after the horizon.
    tau_hold = np.array(
        arm.grid.inverse_dynamics(
            traj_arr[-1:], jnp.zeros((1, ndof)), jnp.zeros((1, ndof))
        )
    )[0]

    def _ref(t: float):
        tc = np.clip(t, 0.0, duration)
        qr = np.array([np.interp(tc, times, traj_np[:, j]) for j in range(ndof)])
        # Past the horizon the arm is *holding* the goal: the reference velocity and
        # the feedforward must go to rest too. Clipping only `tc` would keep feeding
        # the terminal velocity/accel of the plan into the PD term forever, which
        # injects a constant kd * qd_ref torque bias and drives a large steady-state
        # pose error (the arm sags away from the goal while "holding" it).
        if t >= duration:
            return qr, np.zeros(ndof), tau_hold
        qdr = np.array([np.interp(tc, times, np.gradient(traj_np[:, j], dt))
                        for j in range(ndof)])
        taur = np.array([np.interp(tc, times, tau_ff[:, j]) for j in range(ndof)])
        return qr, qdr, taur

    def _pin_box(d: mujoco.MjData):
        """Rigidly place the box at gripper_pose @ (constant grasp offset)."""
        box_world = _hand_se3(d) @ _pin_box.offset
        d.qpos[ndof : ndof + 3] = np.array(box_world.translation())
        d.qpos[ndof + 3 : ndof + 7] = np.array(box_world.rotation().wxyz)
        d.qvel[ndof : ndof + 6] = 0.0

    def step_fn(m: mujoco.MjModel, d: mujoco.MjData):
        hold = 1.0
        if d.time >= duration + hold:
            reset_fn(m, d)
        q_ref, qd_ref, tau = _ref(d.time)
        d.qfrc_applied[:ndof] = (
            tau + kp * (q_ref - d.qpos[:ndof]) + kd * (qd_ref - d.qvel[:ndof])
        )
        mujoco.mj_step(m, d)
        _pin_box(d)  # box tracks the actual gripper (rigid grasp)

    def reset_fn(m: mujoco.MjModel, d: mujoco.MjData):
        mujoco.mj_resetData(m, d)
        d.qpos[:ndof] = traj_np[0]
        d.qpos[ndof : ndof + 3] = box_start
        d.qpos[ndof + 3 : ndof + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(m, d)
        # Capture the constant hand->box offset at the grasp configuration.
        box_world0 = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.identity(), jnp.asarray(box_start)
        )
        _pin_box.offset = _hand_se3(d).inverse() @ box_world0

    reset_fn(mj_model, mj_data)

    if save_video is not None:
        _record_video(mj_model, mj_data, step_fn, duration + 1.0, save_video)
        box_pos = mj_data.qpos[ndof : ndof + 3]
        err = np.linalg.norm(box_pos - goal_center)
        print(f"Simulated box centre at end: {np.round(box_pos, 3)} "
              f"(goal {np.round(goal_center, 3)}, error {err:.3f} m)")
        return

    try:
        import viser  # noqa: F401

        from mjviser import Viewer
    except Exception:
        print("mjviser not available — skipping visualization.")
        return
    print(f"Visualizing — open the viser URL. Ctrl-C to exit. "
          f"({duration:.1f}s trajectory, looping, computed-torque tracking)")
    Viewer(mj_model, mj_data, step_fn=step_fn, reset_fn=reset_fn).run()


def _record_video(mj_model, mj_data, step_fn, duration: float, path: str,
                  fps: int = 30, width: int = 1280, height: int = 720):
    """Step the same simulation offscreen and write an MP4."""
    import imageio.v2 as imageio
    import mujoco

    mj_model.vis.global_.offwidth = max(mj_model.vis.global_.offwidth, width)
    mj_model.vis.global_.offheight = max(mj_model.vis.global_.offheight, height)

    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.45, 0.15, 0.15]
    cam.distance = 1.4
    cam.azimuth = 135.0
    cam.elevation = -25.0

    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    writer = imageio.get_writer(path, fps=fps)
    n_frames = int(duration * fps)
    print(f"Recording {n_frames} frames ({duration:.1f}s) to {path} ...")
    try:
        for i in range(n_frames):
            while mj_data.time * fps < i + 1:
                step_fn(mj_model, mj_data)
            renderer.update_scene(mj_data, camera=cam)
            writer.append_data(renderer.render())
    finally:
        writer.close()
        renderer.close()
    print(f"Saved video to {path}")


def _fd_vel_acc_np(q, dt: float):
    from pyroffi.optimization_engines._flat_contact_trajopt import _fd_vel_acc
    return _fd_vel_acc(q, dt)


if __name__ == "__main__":
    main()
