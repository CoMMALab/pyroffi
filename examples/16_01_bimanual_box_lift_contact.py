"""Contact-rich bimanual box lift (dynamics-aware SCO).

Two Franka Panda arms face each other and pinch a box between their grippers
(pressing on two opposing walls), then lift it. The trajectory is planned by
``contact_sco_trajopt``: a Sequential-Convex-Optimization loop made
dynamics-aware with GRiD inverse dynamics and made contact-aware with an
augmented-Lagrangian *fixed-contact* (grasp-closure) constraint plus a
grasped-object Newton-Euler balance that solves for the contact forces.

The bimanual manipulator setup and the two-Panda scene are assembled *here*,
in the example, out of the fully general building blocks in
``pyroffi.dynamics._contact`` (``ManipulatorSpec``, ``GraspedObject``,
``ContactSystem``) — none of that machinery is specific to two arms or to a
box; a caller wanting three manipulators grasping a sphere would assemble the
same building blocks differently.

The box's inertia is **not** a hand-picked constant: given its mass and
dimensions, ``pyroffi.collision.box_with_mjcf_dynamics`` builds a throwaway
in-memory MJCF model (via ``mujoco.MjSpec`` — no ``.xml`` file written to
disk) and lets MuJoCo's compiler derive the solid-box inertia tensor, the
same physics an MJCF asset file would give you.

Pipeline
--------
1. Build the box's physical geometry (MJCF-derived inertia); place it between
   two Panda bases.
2. IK each arm to a pinch pose on its box face (start), and to the same pose
   translated straight up by ``LIFT`` (goal). Translating *both* gripper
   targets by the same vector preserves their relative transform, so the goal
   still satisfies the rigid-grasp constraint.
3. Capture the grasp offset at the start config and build a
   ``ContactSystem`` from the two ``ManipulatorSpec``s and the
   MJCF-sourced ``GraspedObject``.
4. Optimize the joint + contact-force trajectory with ``contact_sco_trajopt``.
5. Visualize the result in a real, actively-stepped MuJoCo simulation (via
   mjviser's ``Viewer(step_fn=..., reset_fn=...)``, following
   ``gtmp_pyronot/examples/example_gtmp_kinodynamic.py``): a computed-torque
   controller (GRiD feedforward torques from the planned trajectory + PD
   correction) tracks the timed reference for each arm, and the box is a
   **real dynamic body** (freejoint, mass, friction, full collision) —
   there is no kinematic teleport. It's held up by the actual contact forces
   the two hands' palms exert on it as the arms track the plan, the same
   mechanism the trajopt's contact-force allocation models: an *actuated*
   squeeze from each arm's own joint torques (not a passive finger pinch —
   see the geometry note below) (skipped automatically if mjviser is
   unavailable / running headless).

Grasp geometry. The Panda's spherized collision model for ``panda_hand`` is a
flat "puck" of spheres spanning local z in ``[~-0.02, 0.074]`` (the palm); its
fixed, non-actuated fingers sit further out at local z in ``[0.08, 0.10]`` but
are only ~0.13 m apart, far narrower than this box, so they play no part here
— the grip is a *palm-to-palm* squeeze, like compressing the box between two
flat plates. Each hand's contact point ``p_local=(0, 0, D)`` is placed at the
surface of the deepest palm sphere ring (``D`` = ring depth + sphere radius),
with the box positioned a hair *inside* that surface (a few mm of
interference) so real, actuated squeeze pressure — not a geometric
coincidence — is what holds the box; too little interference and the palms
never truly load the box, too much and the arms fight the box's own
stiffness instead of tracking the plan.

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
    manipulator_contact_fext,
    object_center_world,
)
from pyroffi.optimization_engines import ContactTrajOptConfig, contact_sco_trajopt
from pyroffi.optimization_engines._contact_trajopt import _fd_vel_acc

# --- Scene constants --------------------------------------------------------
GRIP_LINK = "panda_hand"
BASE_SEP = 0.4          # each base offset +/- this along world x
BOX_CENTER = np.array([0.0, 0.0, 0.45])
BOX_HALF_LENGTHS = np.array([0.06, 0.12, 0.12])
BOX_MASS = 0.5          # kg
BOX_FRICTION = 0.6
LIFT = 0.15             # vertical lift (m)
T = 24                  # trajectory waypoints
# Palm contact standoff (hand-local +z): deepest palm-sphere ring depth
# (0.05) + its radius (0.028) = the sphere's surface, minus a few mm of
# interference so the actuated squeeze actually loads the box (see module
# docstring's geometry note).
PALM_STANDOFF = 0.070


def _ik_pose(robot, base_xy_yaw, world_target: jaxlie.SE3, seed):
    """IK an arm (with a world base offset) to a world gripper pose."""
    x, y, yaw = base_xy_yaw
    base = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_z_radians(jnp.asarray(yaw)), jnp.array([x, y, 0.0])
    )
    local_target = base.inverse() @ world_target  # into the arm's base frame
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

    # Gripper faces inward toward the box: the hand's approach axis (local z)
    # points horizontally at the box centre, so the fingers/palm confront the
    # box face rather than pressing down onto it from above. The hand origin
    # is stood off by (box_half_x + PALM_STANDOFF) from the box centre —
    # *not* placed flush on the box face — so it's the palm's contact point
    # p_local=(0,0,PALM_STANDOFF) that reaches the box surface, not the
    # hand's own bulk (which is what caused the box to render embedded in the
    # wrist before this fix).
    def grip_target(sign, height):
        pos = box_center + np.array([sign * (box_half_x + PALM_STANDOFF), 0.0, height])
        # Orientation: z (approach) toward the box centre, x pointing down.
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

    cfg = ContactTrajOptConfig(
        n_outer_iters=15, n_inner_iters=25, dt=0.1,
        rho_grasp=50.0, penalty_scale=1.8, tau_max=87.0,
        f_min=3.0,  # mu_friction=None -> uses the MJCF box's own friction
    )

    # Warm up with the SAME cfg we time below. opt_cfg is a static JIT arg, so a
    # warmup config that differs in any field (e.g. n_outer_iters) compiles a
    # *different* function and the real solve then recompiles inside the timed
    # region -- which is what made the "execution" look like ~70s. Warming the
    # exact cfg moves all compilation here and reveals the true few-second solve.
    print("Warming up (JIT compile) ...")
    t0 = time.perf_counter()
    warm_traj, _, _ = contact_sco_trajopt(init, start, goal, system, cfg)
    warm_traj.block_until_ready()
    print(f"  warmup done in {time.perf_counter() - t0:.1f}s")

    print("Optimizing contact-rich trajectory ...")
    t0 = time.perf_counter()
    traj, forces, resid = contact_sco_trajopt(init, start, goal, system, cfg)
    traj.block_until_ready()
    print(f"  done in {time.perf_counter() - t0:.1f}s")
    print(f"  grasp-closure residual [rms, max] = {np.array(resid)}")
    print(f"  mean |contact force| = {float(jnp.mean(jnp.abs(forces))):.2f} N")

    centers = jax.vmap(object_center_world, in_axes=(None, 0))(system, traj)
    print(f"  box lift achieved: {float(centers[-1, 2] - centers[0, 2]):.3f} m")

    goal_center = BOX_CENTER + np.array([0.0, 0.0, LIFT])
    _maybe_visualize(system, traj, forces, cfg, goal_center,
                     save_video=args.save_video)


def _maybe_visualize(system: ContactSystem, traj, forces,
                     cfg: ContactTrajOptConfig, goal_center: np.ndarray,
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

    # Build a MuJoCo scene with both arms (at their real base poses) plus the
    # grasped box. The box is a real dynamic body (freejoint + mass + friction
    # + full collision) — no kinematic teleport; it's held up purely by the
    # contact forces the two palms exert on it as the arms track the plan.
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
    # Translucent green goal marker (non-colliding) at the planned lift pose.
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

    # --- Feedforward torques per arm (GRiD ID over the planned traj, incl. contact) ---
    traj_arr = jnp.asarray(traj)
    qL, qR = traj_arr[:, :nL], traj_arr[:, nL:]
    qdL, qddL = _fd_vel_acc(qL, cfg.dt)
    qdR, qddR = _fd_vel_acc(qR, cfg.dt)
    fextL = jax.vmap(manipulator_contact_fext, in_axes=(None, 0, 0))(left, qL, forces[:, 0, :])
    fextR = jax.vmap(manipulator_contact_fext, in_axes=(None, 0, 0))(right, qR, forces[:, 1, :])
    tau_ff = np.concatenate([
        np.array(left.grid.inverse_dynamics(qL, qdL, qddL, f_ext=fextL)),
        np.array(right.grid.inverse_dynamics(qR, qdR, qddR, f_ext=fextR)),
    ], axis=1)
    traj_np = np.array(traj)
    times = np.arange(traj_np.shape[0]) * cfg.dt
    duration = float(times[-1])

    kp, kd = 1500.0, 80.0

    # Static hold torque at the goal: gravity plus the contact reaction of the
    # still-grasped box, with both arms at rest.
    tau_hold = np.concatenate([
        np.array(left.grid.inverse_dynamics(
            qL[-1:], jnp.zeros((1, nL)), jnp.zeros((1, nL)), f_ext=fextL[-1:])),
        np.array(right.grid.inverse_dynamics(
            qR[-1:], jnp.zeros((1, nR)), jnp.zeros((1, nR)), f_ext=fextR[-1:])),
    ], axis=1)[0]

    def _ref(t: float):
        tc = np.clip(t, 0.0, duration)
        qr = np.array([np.interp(tc, times, traj_np[:, j]) for j in range(nv_arms)])
        # Past the horizon the arms are *holding* the goal, so the reference velocity
        # and the feedforward must go to rest too. Clipping only `tc` would keep
        # feeding the plan's terminal velocity into the PD term forever, injecting a
        # constant kd * qd_ref torque bias that drives a steady-state pose error.
        if t >= duration:
            return qr, np.zeros(nv_arms), tau_hold
        qdr = np.array([np.interp(tc, times, np.gradient(traj_np[:, j], cfg.dt))
                        for j in range(nv_arms)])
        taur = np.array([np.interp(tc, times, tau_ff[:, j]) for j in range(nv_arms)])
        return qr, qdr, taur

    def step_fn(m: mujoco.MjModel, d: mujoco.MjData):
        hold = 1.0
        # Loop by resetting the *state*, not just wrapping the reference:
        # wrapping alone leaves the box at the lifted pose while the arms snap
        # back to the start, so every pass after the first plays out empty.
        if d.time >= duration + hold:
            reset_fn(m, d)
        q_ref, qd_ref, tau = _ref(d.time)
        d.qfrc_applied[:nv_arms] = (
            tau + kp * (q_ref - d.qpos[:nv_arms]) + kd * (qd_ref - d.qvel[:nv_arms])
        )
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
