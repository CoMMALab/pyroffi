"""Contact-*rich* bimanual box lift (contact forces as decision variables).

The contact-rich parallel to ``16_00``. Two Franka Panda arms face each other,
pinch a box palm-to-palm, and lift it — the same scene and grasp mechanics as
``16_00`` — but planned with a different solver:

* ``16_00`` uses ``contact_sco_trajopt``: the original augmented-Lagrangian
  contact solver over a fixed timestep.
* ``16_03`` (this file) uses ``contact_rich_trajopt``: the per-contact forces
  are **first-class decision variables** optimized under the object's
  Newton-Euler balance (augmented Lagrangian) *and* the friction cone, with a
  minimum-time objective over an optimized shared timestep ``dt``. It shares its
  interface with the differential-flatness **contact-aware** solver
  (``flat_contact_trajopt``) so the two can be swapped directly — but unlike that
  one, it does not *allocate* the forces from a grasp-map pseudo-inverse; it
  *optimizes* them, so the system is genuinely contact-rich (not flat). The
  contact *mode* is still one persistent rigid grasp — contact-rich, not
  contact-implicit. See ``flat_contact_trajopt_theory.md`` §8.

Where a single contact fully determines the force (the single-arm ``16_02``),
contact-rich and contact-aware coincide. Here, with two contacts, the grasp map
has a genuine null space (internal squeeze), so the contact-rich solver has real
force freedom to trade off against the friction cone — this is the regime where
optimizing the forces actually buys something over allocating them.

Scene, grasp geometry, and the actuated-grip MuJoCo playback follow ``16_00`` /
``16_01``: the box is a real dynamic freejoint body, and because the palms are
non-actuated an *ideal actuated grip* (the planner's own net grasp wrench applied
to the box) carries it, so it does not slip through the passive primitive contact.

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
    capture_grasp_offsets,
    contact_points_world,
    manipulator_contact_fext,
    object_center_world,
)
from pyroffi.optimization_engines import (
    ContactRichTrajOptConfig,
    contact_rich_trajopt,
)
from pyroffi.optimization_engines._contact_trajopt import _fd_vel_acc

# --- Scene constants (identical to 16_00) -----------------------------------
GRIP_LINK = "panda_hand"
BASE_SEP = 0.4
BOX_CENTER = np.array([0.0, 0.0, 0.45])
BOX_HALF_LENGTHS = np.array([0.06, 0.12, 0.12])
BOX_MASS = 0.5
BOX_FRICTION = 0.6
LIFT = 0.15
T = 24
PALM_STANDOFF = 0.070


def _ik_pose(robot, base_xy_yaw, world_target: jaxlie.SE3, seed):
    x, y, yaw = base_xy_yaw
    base = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_z_radians(jnp.asarray(yaw)), jnp.array([x, y, 0.0])
    )
    local_target = base.inverse() @ world_target
    return robot.inverse_kinematics(GRIP_LINK, local_target, solver="ls",
                                    num_seeds=64, previous_cfg=seed)


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
    box_center = np.array(box_geom.pose.translation())
    box_half_x = float(box_geom.half_lengths[0])

    left_base = (-BASE_SEP, 0.0, 0.0)
    right_base = (BASE_SEP, 0.0, np.pi)

    def grip_target(sign, height):
        pos = box_center + np.array([sign * (box_half_x + PALM_STANDOFF), 0.0, height])
        rot = jaxlie.SO3.from_matrix(jnp.array([
            [0.0, 0.0, -sign],
            [0.0, -sign, 0.0],
            [-1.0, 0.0, 0.0],
        ]))
        return jaxlie.SE3.from_rotation_and_translation(rot, jnp.asarray(pos))

    mid = (robot.joints.lower_limits + robot.joints.upper_limits) / 2

    print("Solving grasp IK (start + lifted goal) ...")
    qL0 = _ik_pose(robot, left_base, grip_target(-1, 0.0), mid)
    qR0 = _ik_pose(robot, right_base, grip_target(+1, 0.0), mid)
    qL1 = _ik_pose(robot, left_base, grip_target(-1, LIFT), qL0)
    qR1 = _ik_pose(robot, right_base, grip_target(+1, LIFT), qR0)

    left = ManipulatorSpec(robot, grid, GRIP_LINK, base_xy_yaw=left_base,
                           p_local=(0.0, 0.0, PALM_STANDOFF))
    right = ManipulatorSpec(robot, grid, GRIP_LINK, base_xy_yaw=right_base,
                            p_local=(0.0, 0.0, PALM_STANDOFF))
    manipulators = (left, right)

    grasp_offsets = capture_grasp_offsets(manipulators, (qL0, qR0))
    system = ContactSystem(
        manipulators=manipulators,
        body=GraspedObject(geom=box_geom),
        grasp_offsets=grasp_offsets,
    )

    start = jnp.concatenate([qL0, qR0])
    goal = jnp.concatenate([qL1, qR1])
    t = jnp.linspace(0.0, 1.0, T)[:, None]
    init = start[None] * (1 - t) + goal[None] * t

    cfg = ContactRichTrajOptConfig(
        n_outer_iters=15, n_inner_iters=30, m_lbfgs=8, dt=0.1,
        rho_grasp=50.0, rho_obj=10.0, penalty_scale=1.8, tau_max=87.0,
        f_min=3.0, w_smoothness=1.0, w_effort=1e-3,
    )

    print("Warming up (JIT compile) ...")
    t0 = time.perf_counter()
    warm = contact_rich_trajopt(init, start, goal, system, cfg)
    jax.block_until_ready(warm)
    print(f"  warmup done in {time.perf_counter() - t0:.1f}s")

    print("Optimizing contact-rich trajectory (forces are decision variables) ...")
    t0 = time.perf_counter()
    traj, forces, resid, centers, dt = contact_rich_trajopt(init, start, goal, system, cfg)
    jax.block_until_ready(traj)
    print(f"  done in {time.perf_counter() - t0:.1f}s")
    print(f"  object-dynamics residual [rms, max] = {np.array(resid[:2])}")
    print(f"  grasp-closure residual (rms)         = {float(resid[2]):.2e}")
    print(f"  optimized dt = {float(dt):.4f}s  (horizon = {float(dt) * (len(traj) - 1):.2f}s)")
    print(f"  mean |contact force| = {float(jnp.mean(jnp.abs(forces))):.2f} N")
    print(f"  box lift achieved: {float(centers[-1, 2] - centers[0, 2]):.3f} m")

    goal_center = BOX_CENTER + np.array([0.0, 0.0, LIFT])
    _maybe_visualize(system, traj, forces, dt, goal_center,
                     save_video=args.save_video)


def _grip_wrenches(system: ContactSystem, traj_arr, forces):
    """Per-waypoint net grasp wrench (force, torque about the box centre) the
    optimized contact forces exert on the held object, in world axes."""
    centers = jax.vmap(object_center_world, in_axes=(None, 0))(system, traj_arr)
    pts = jax.vmap(lambda q: jnp.stack(contact_points_world(system, q)))(traj_arr)
    net_f = jnp.sum(forces, axis=1)
    net_tau = jnp.sum(jnp.cross(pts - centers[:, None, :], forces), axis=1)
    return np.array(net_f), np.array(net_tau)


def _maybe_visualize(system: ContactSystem, traj, forces, dt, goal_center,
                     save_video: str | None = None):
    try:
        import mujoco
    except Exception:
        print("mujoco not available — skipping visualization.")
        return

    left, right = system.manipulators
    nL, nR = left.num_dof, right.num_dof
    box_geom = system.body.geom
    box_center0 = np.array(box_geom.pose.translation())
    box_extents = np.array(box_geom.extents)
    dt = float(dt)

    spec = mujoco.MjSpec()
    spec.add_texture(
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
        size=np.array(box_geom.extents) / 2.0,
        rgba=[0.2, 0.8, 0.3, 0.35], contype=0, conaffinity=0,
    )

    lb, rb = left.base_se3(), right.base_se3()
    lframe = spec.worldbody.add_frame(
        pos=np.array(lb.translation()), quat=np.array(lb.rotation().wxyz)
    )
    rframe = spec.worldbody.add_frame(
        pos=np.array(rb.translation()), quat=np.array(rb.rotation().wxyz)
    )
    left_spec = mujoco.MjSpec.from_file("resources/panda/panda_spherized.urdf")
    right_spec = mujoco.MjSpec.from_file("resources/panda/panda_spherized.urdf")
    spec.attach(left_spec, prefix="left_", frame=lframe)
    spec.attach(right_spec, prefix="right_", frame=rframe)

    box = spec.worldbody.add_body(name="box", pos=box_center0)
    box.add_freejoint()
    box.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=box_extents / 2.0,
                rgba=[200 / 255, 140 / 255, 60 / 255, 1.0],
                friction=[BOX_FRICTION, 0.01, 0.001])

    mj_model = spec.compile()
    mj_data = mujoco.MjData(mj_model)
    nv_arms = nL + nR

    # --- Feedforward torques per arm (GRiD ID over the planned traj) ----------
    traj_arr = jnp.asarray(traj)
    qL, qR = traj_arr[:, :nL], traj_arr[:, nL:]
    qdL, qddL = _fd_vel_acc(qL, dt)
    qdR, qddR = _fd_vel_acc(qR, dt)
    fextL = jax.vmap(manipulator_contact_fext, in_axes=(None, 0, 0))(left, qL, forces[:, 0, :])
    fextR = jax.vmap(manipulator_contact_fext, in_axes=(None, 0, 0))(right, qR, forces[:, 1, :])
    tau_ff = np.concatenate([
        np.array(left.grid.inverse_dynamics(qL, qdL, qddL, f_ext=fextL)),
        np.array(right.grid.inverse_dynamics(qR, qdR, qddR, f_ext=fextR)),
    ], axis=1)
    traj_np = np.array(traj)
    times = np.arange(traj_np.shape[0]) * dt
    duration = float(times[-1])

    # --- Actuated grip: apply the planner's net grasp wrench to the box -------
    grip_f, grip_tau = _grip_wrenches(system, traj_arr, forces)
    box_body = mj_model.body("box").id

    kp, kd = 1500.0, 80.0

    def _ref(t: float):
        tc = np.clip(t, 0.0, duration)
        qr = np.array([np.interp(tc, times, traj_np[:, j]) for j in range(nv_arms)])
        qdr = np.array([np.interp(tc, times, np.gradient(traj_np[:, j], dt))
                        for j in range(nv_arms)])
        taur = np.array([np.interp(tc, times, tau_ff[:, j]) for j in range(nv_arms)])
        fr = np.array([np.interp(tc, times, grip_f[:, a]) for a in range(3)])
        tr = np.array([np.interp(tc, times, grip_tau[:, a]) for a in range(3)])
        return qr, qdr, taur, fr, tr

    def step_fn(m: mujoco.MjModel, d: mujoco.MjData):
        hold = 1.0
        if d.time >= duration + hold:
            reset_fn(m, d)
        q_ref, qd_ref, tau, grip_force, grip_torque = _ref(d.time)
        d.qfrc_applied[:nv_arms] = (
            tau + kp * (q_ref - d.qpos[:nv_arms]) + kd * (qd_ref - d.qvel[:nv_arms])
        )
        d.xfrc_applied[box_body, :3] = grip_force
        d.xfrc_applied[box_body, 3:] = grip_torque
        mujoco.mj_step(m, d)

    def reset_fn(m: mujoco.MjModel, d: mujoco.MjData):
        mujoco.mj_resetData(m, d)
        d.qpos[:nv_arms] = traj_np[0]
        d.qpos[nv_arms : nv_arms + 3] = box_center0
        d.qpos[nv_arms + 3 : nv_arms + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(m, d)

    reset_fn(mj_model, mj_data)

    if save_video is not None:
        _record_video(mj_model, mj_data, step_fn, duration + 1.0, save_video)
        box_pos = mj_data.qpos[nv_arms : nv_arms + 3]
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
    cam.lookat[:] = [0.0, 0.0, 0.45]
    cam.distance = 2.0
    cam.azimuth = 90.0
    cam.elevation = -20.0

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


if __name__ == "__main__":
    main()
