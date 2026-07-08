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
5. Visualize the playback (real MuJoCo bodies for both arms + box, via
   mjviser) and per-gripper contact-force arrows
   (skipped automatically if mjviser is unavailable / running headless).

Requires a CUDA GPU + nvcc (GRiD dynamics).
"""

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
    object_center_world,
)
from pyroffi.optimization_engines import ContactTrajOptConfig, contact_sco_trajopt

# --- Scene constants --------------------------------------------------------
GRIP_LINK = "panda_hand"
BASE_SEP = 0.4          # each base offset +/- this along world x
BOX_CENTER = np.array([0.0, 0.0, 0.45])
BOX_HALF_LENGTHS = np.array([0.06, 0.12, 0.12])
BOX_MASS = 0.5          # kg
BOX_FRICTION = 0.6
LIFT = 0.15             # vertical lift (m)
T = 24                  # trajectory waypoints


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
    # box face rather than pressing down onto it from above.
    def grip_target(sign, height):
        # Position: on the box face facing the arm.
        pos = box_center + np.array([sign * box_half_x, 0.0, 0.0])
        pos = pos + np.array([0.0, 0.0, height])
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
                           p_local=(0.0, 0.0, 0.1))
    right = ManipulatorSpec(robot, grid, GRIP_LINK, base_xy_yaw=right_base,
                            p_local=(0.0, 0.0, 0.1))
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

    print("Warming up (JIT compile) ...")
    t0 = time.perf_counter()
    warmup_cfg = ContactTrajOptConfig(
        n_outer_iters=1, n_inner_iters=1, dt=cfg.dt,
        rho_grasp=cfg.rho_grasp, penalty_scale=cfg.penalty_scale,
        tau_max=cfg.tau_max, mu_friction=cfg.mu_friction, f_min=cfg.f_min,
    )
    warm_traj, _, _ = contact_sco_trajopt(init, start, goal, system, warmup_cfg)
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

    _maybe_visualize(system, traj, forces, centers)


def _maybe_visualize(system, traj, forces, centers):
    try:
        import mujoco
        import viser

        import mjviser
    except Exception:
        print("mjviser not available — skipping visualization.")
        return

    # Build a MuJoCo scene with both arms (at their real base poses) plus the
    # grasped box, by attaching two copies of the Panda URDF and the box MJCF
    # into one spec. This is playback only (no physics stepping) -- we set
    # qpos/box pose per frame from the already-optimized trajectory and let
    # mjviser render it.
    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1])

    left, right = system.manipulators
    lb = left.base_se3()
    rb = right.base_se3()
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

    box_geom = system.body.geom
    box_center0 = np.array(box_geom.pose.translation())
    box_extents = np.array(box_geom.extents)
    box = spec.worldbody.add_body(name="box", pos=box_center0)
    box.add_freejoint()
    box.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=box_extents / 2.0,
                rgba=[200 / 255, 140 / 255, 60 / 255, 1.0])

    mj_model = spec.compile()
    mj_data = mujoco.MjData(mj_model)
    nL = system.manipulators[0].num_dof

    server = viser.ViserServer()
    scene = mjviser.ViserMujocoScene(server, mj_model, num_envs=1)

    play_btn = server.gui.add_button("Pause", icon=viser.Icon.PLAYER_PAUSE)
    traj = np.array(traj); forces = np.array(forces); centers = np.array(centers)
    box_quat = np.array([1.0, 0.0, 0.0, 0.0])
    playback = {"playing": True, "k": 0}

    @play_btn.on_click
    def _(_) -> None:
        playback["playing"] = not playback["playing"]
        play_btn.label = "Pause" if playback["playing"] else "Play"
        play_btn.icon = (
            viser.Icon.PLAYER_PAUSE if playback["playing"] else viser.Icon.PLAYER_PLAY
        )

    def update(k):
        mj_data.qpos[: 2 * nL] = np.concatenate([traj[k, :nL], traj[k, nL:]])
        mj_data.qpos[2 * nL : 2 * nL + 3] = centers[k]
        mj_data.qpos[2 * nL + 3 : 2 * nL + 7] = box_quat
        mujoco.mj_forward(mj_model, mj_data)
        scene.update_from_mjdata(mj_data)

        c = centers[k]
        for i, name in enumerate(("L", "R")):
            f = forces[k, i]
            server.scene.add_spline_catmull_rom(
                f"/force_{name}",
                np.stack([c, c + 0.01 * f]), color=(230, 40, 40), line_width=4,
            )

    update(0)
    print("Visualizing — open the viser URL. Ctrl-C to exit.")
    frame_dt = 0.15  # playback pacing, independent of trajopt's dt.
    while True:
        if playback["playing"]:
            playback["k"] = (playback["k"] + 1) % traj.shape[0]
            update(playback["k"])
        time.sleep(frame_dt)


if __name__ == "__main__":
    main()
