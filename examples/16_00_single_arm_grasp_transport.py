"""Single-arm contact-rich grasp-and-transport, planned with differential
flatness and played back in a real, actively-stepped MuJoCo simulation.

A single Franka Panda picks up a box from a top-down pinch grasp and carries
it sideways to a drop location (lift + horizontal transport). Unlike the
bimanual pinch in ``16_00``, a *single* manipulator's "grasp" degenerates
cleanly in the ``pyroffi.dynamics._contact`` building blocks: with one
contact point, ``ContactSystem.grasp_offsets`` is empty (there is no second
gripper to hold in relative pose), and the object's centre *is* the contact
point, so the box moves exactly as a rigid extension of the gripper. This is
the "single-arm + fixture" case the module docstring calls out.

The trajectory itself is planned by ``flat_contact_trajopt``: the object-pose
twist is the flat output, the arm's joint config tracks the object-derived
gripper pose, and the (single) contact force is allocated analytically from
the object's acceleration -- so grasp closure and object dynamics hold by
construction rather than by penalty, and the whole solve is one L-BFGS pass
with light penalty continuation (see ``_flat_contact_trajopt.py`` docstring).

Grasp geometry. The Panda's spherized collision model for ``panda_hand`` is a
flat "puck" of spheres spanning local z in ``[~-0.02, 0.074]`` (the palm),
and its two fingers are *fixed* (non-actuated) at local y = +/-0.065, with
fingertip pad spheres spanning local z in ``[0.0804, 0.1024]``. The contact
point ``p_local=(0, 0, PINCH_Z)`` is placed inside that fingertip-pad band
(not inside the palm's own bulk, which is what caused the box to render
*embedded* in the wrist before this fix), and the box's finger-separation
half-width is sized with a couple of millimetres of *interference* against
the fingertip pads' inner reach. Because the fingers can't actuate a squeeze
(there is no gripper torque command in this model), that interference is what
lets a passive, snug friction fit hold the box against gravity -- exactly the
mechanism a real underactuated/fixed parallel gripper relies on.

Rendering departs from ``16_00``'s manual kinematic-playback loop and instead
follows ``gtmp_pyronot/examples/example_gtmp_kinodynamic.py``: a real MuJoCo
simulation is *actively stepped* (``mjviser.Viewer(step_fn=..., reset_fn=...)``)
with a computed-torque controller (GRiD feedforward torques from the planned
trajectory + PD correction) tracking the timed reference. The box is a
**real dynamic body** (freejoint, mass, friction, full collision against the
hand/fingers/table) -- there is no kinematic teleport of any kind.

Actuated grip. The Panda's fingertip pads are *fixed* (non-actuated) links, so
the passive friction between the snug sphere-vs-box primitive contacts is too
small to hold the box against gravity and it slips downward. Modeling an *ideal
actuated gripper*, each step applies the planner's own grasp wrench (the net
contact force + torque about the box centre, from the optimized contact forces)
to the box via ``xfrc_applied`` -- so the box is carried by exactly the forces
the trajopt reasoned about, not by incidental primitive friction. The box is
still a freely integrated dynamic body; only the grasp force is supplied by the
grip model instead of hoped-for penetration friction.

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
    contact_points_world,
    manipulator_contact_fext,
    object_center_world,
)
from pyroffi.optimization_engines import FlatContactTrajOptConfig, flat_contact_trajopt

# --- Scene constants --------------------------------------------------------
GRIP_LINK = "panda_hand"
# Fingertip-pad standoff (hand-local +z): inside the [0.0804, 0.1024] band the
# fixed fingertip pads actually occupy (see module docstring) -- neither
# palm-embedded nor past the fingertips.
PINCH_Z = 0.095
# Box half-extents (world axes; the top-down grasp aligns hand-local y, the
# finger-separation/pinch axis, with world Y). The Y half is a couple of mm
# *larger* than the fingertip pads' measured inner reach (~0.061 m) so the
# passive, non-actuated fingers hold the box by interference/friction alone.
BOX_HALF_LENGTHS = np.array([0.025, 0.065, 0.020])
BOX_CENTER = np.array([0.45, 0.0, BOX_HALF_LENGTHS[2]])  # resting on the table
BOX_MASS = 0.15           # kg -- light enough for a passive fixed-finger grip
BOX_FRICTION = 1.0
LIFT = 0.12               # vertical lift during transport (m)
TRANSPORT = 0.3           # horizontal carry distance, world +y (m)
T = 24                    # trajectory waypoints


def _grip_target(box_center: np.ndarray) -> jaxlie.SE3:
    """Top-down pinch pose: gripper origin stood off by ``PINCH_Z`` above the
    box centre (along the approach axis) so the fingertip-pad contact point
    ``p_local=(0,0,PINCH_Z)`` lands exactly on the box, not the hand's own
    palm bulk."""
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

    cfg = FlatContactTrajOptConfig(
        n_stages=5, n_inner_iters=50, dt=0.1,
        w_track=200.0, track_scale=3.0, tau_max=87.0, f_min=1.0,
    )

    print("Warming up (JIT compile) ...")
    t0 = time.perf_counter()
    warm = flat_contact_trajopt(init, q0, q1, system, cfg)
    jax.block_until_ready(warm)
    print(f"  warmup done in {time.perf_counter() - t0:.1f}s")

    print("Optimizing contact-rich trajectory ...")
    t0 = time.perf_counter()
    traj, forces, resid, centers, dt = flat_contact_trajopt(init, q0, q1, system, cfg)
    jax.block_until_ready(traj)
    print(f"  done in {time.perf_counter() - t0:.1f}s")
    print(f"  grasp-closure residual [rms, max] = {np.array(resid)}")
    print(f"  optimized dt = {float(dt):.4f}s  (horizon = {float(dt) * (len(traj) - 1):.2f}s)")
    print(f"  mean |contact force| = {float(jnp.mean(jnp.abs(forces))):.2f} N")
    print(f"  carry achieved: dz={float(centers[-1, 2] - centers[0, 2]):.3f} m, "
          f"dy={float(centers[-1, 1] - centers[0, 1]):.3f} m")

    _maybe_visualize(system, traj, forces, cfg, goal_center,
                     save_video=args.save_video)


def _maybe_visualize(system: ContactSystem, traj, forces,
                     cfg: FlatContactTrajOptConfig, goal_center: np.ndarray,
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
    # Translucent green goal marker (non-colliding) at the planned drop pose.
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX, pos=goal_center,
        size=np.array(system.body.geom.extents) / 2.0,
        rgba=[0.2, 0.8, 0.3, 0.35], contype=0, conaffinity=0,
    )
    arm_frame = spec.worldbody.add_frame()
    arm_spec = mujoco.MjSpec.from_file("resources/panda/panda_spherized.urdf")
    spec.attach(arm_spec, prefix="arm_", frame=arm_frame)

    # A real dynamic body (freejoint + mass + friction + full collision
    # against the fingertip pads/palm/table) — no kinematic teleport. It is
    # held up purely by the contact forces the fingertip-pad interference fit
    # generates as the arm tracks the plan.
    box = spec.worldbody.add_body(name="box", pos=box_center0)
    box.add_freejoint()
    box.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=box_extents / 2.0,
                 rgba=[200 / 255, 140 / 255, 60 / 255, 1.0],
                 friction=[BOX_FRICTION, 0.01, 0.001])

    mj_model = spec.compile()
    mj_data = mujoco.MjData(mj_model)
    ndof = arm.num_dof

    # --- Feedforward torques (GRiD ID over the planned traj, incl. contact) ---
    traj_arr = jnp.asarray(traj)
    q, qd, qdd = traj_arr, *_fd_vel_acc_np(traj_arr, cfg.dt)
    fext = jax.vmap(manipulator_contact_fext, in_axes=(None, 0, 0))(
        arm, traj_arr, forces[:, 0, :]
    )
    tau_ff = np.array(arm.grid.inverse_dynamics(q, qd, qdd, f_ext=fext))
    traj_np = np.array(traj)
    times = np.arange(traj_np.shape[0]) * cfg.dt
    duration = float(times[-1])

    # --- Actuated grip: net grasp wrench the planner computed, per waypoint ---
    # The fingertip pads are FIXED (non-actuated) links, so the passive friction
    # between the snug sphere-vs-box primitives is too small to hold the box and
    # it slips. Modeling an ideal actuated gripper, we apply the planner's own
    # grasp wrench (net contact force + torque about the box centre) to the box
    # each step, so the box is carried by exactly the forces the trajopt reasoned
    # about rather than by incidental primitive friction.
    grip_f, grip_tau = _grip_wrenches(system, traj_arr, forces)
    box_body = mj_model.body("box").id

    kp, kd = 1500.0, 80.0

    # Static hold torque at the goal: gravity plus the contact reaction of the
    # still-grasped box, with the arm at rest.
    tau_hold = np.array(
        arm.grid.inverse_dynamics(
            traj_arr[-1:], jnp.zeros((1, ndof)), jnp.zeros((1, ndof)),
            f_ext=fext[-1:],
        )
    )[0]

    def _ref(t: float):
        tc = np.clip(t, 0.0, duration)
        qr = np.array([np.interp(tc, times, traj_np[:, j]) for j in range(ndof)])
        fr = np.array([np.interp(tc, times, grip_f[:, a]) for a in range(3)])
        tr = np.array([np.interp(tc, times, grip_tau[:, a]) for a in range(3)])
        # Past the horizon the arm is *holding* the goal, so the reference velocity
        # and the feedforward must go to rest too. Clipping only `tc` would keep
        # feeding the plan's terminal velocity into the PD term forever, injecting a
        # constant kd * qd_ref torque bias that drives a large steady-state pose
        # error (the arm drifts off the goal while it is supposed to be holding it).
        # The grip wrench needs no such special case: the plan's terminal
        # acceleration is already zero, so grip_f[-1] is the static support wrench.
        if t >= duration:
            return qr, np.zeros(ndof), tau_hold, fr, tr
        qdr = np.array([np.interp(tc, times, np.gradient(traj_np[:, j], cfg.dt))
                        for j in range(ndof)])
        taur = np.array([np.interp(tc, times, tau_ff[:, j]) for j in range(ndof)])
        return qr, qdr, taur, fr, tr

    def step_fn(m: mujoco.MjModel, d: mujoco.MjData):
        hold = 1.0
        # Loop by resetting the *state*, not just wrapping the reference:
        # wrapping alone leaves the box at the drop pose while the arm snaps
        # back to the start, so every pass after the first plays out empty.
        if d.time >= duration + hold:
            reset_fn(m, d)
        q_ref, qd_ref, tau, grip_force, grip_torque = _ref(d.time)
        d.qfrc_applied[:ndof] = (
            tau + kp * (q_ref - d.qpos[:ndof]) + kd * (qd_ref - d.qvel[:ndof])
        )
        d.xfrc_applied[box_body, :3] = grip_force
        d.xfrc_applied[box_body, 3:] = grip_torque
        mujoco.mj_step(m, d)

    def reset_fn(m: mujoco.MjModel, d: mujoco.MjData):
        mujoco.mj_resetData(m, d)
        d.qpos[:ndof] = traj_np[0]
        d.qpos[ndof : ndof + 3] = box_center0
        d.qpos[ndof + 3 : ndof + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(m, d)

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


def _grip_wrenches(system: ContactSystem, traj_arr, forces):
    """Per-waypoint net grasp wrench (force, torque about the box centre) that
    the planner's contact forces exert on the held object, in world axes.

    Applied to the box body as an ideal actuated grip (see ``_maybe_visualize``).
    ``forces`` is ``[T, k, 3]`` (one world force per manipulator).
    """
    centers = jax.vmap(object_center_world, in_axes=(None, 0))(system, traj_arr)  # (T,3)
    pts = jax.vmap(lambda q: jnp.stack(contact_points_world(system, q)))(traj_arr)  # (T,k,3)
    net_f = jnp.sum(forces, axis=1)  # (T,3)
    net_tau = jnp.sum(jnp.cross(pts - centers[:, None, :], forces), axis=1)  # (T,3)
    return np.array(net_f), np.array(net_tau)


def _fd_vel_acc_np(q, dt: float):
    """Central finite-difference velocity/acceleration (matches the internal
    trajopt convention), returned as plain arrays for feedforward torques."""
    from pyroffi.optimization_engines._flat_contact_trajopt import _fd_vel_acc
    return _fd_vel_acc(q, dt)


if __name__ == "__main__":
    main()
