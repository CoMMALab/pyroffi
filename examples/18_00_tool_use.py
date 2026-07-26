"""Collision-free pick and place, with the grasped block carried as an
*attachment*, played back in an mjviser window.

The point of this example is that a grasped object is not a special case in the
planner. Attaching a body to a link adds a fixed joint, and a fixed joint is
something both halves of pyroffi already absorb:

* **Kinematics** — ``T_WB(q) = T_WL(q) · T_LB``. One SE(3) compose on the FK the
  robot already runs. No new configuration variable, so ``MAX_JOINTS`` /
  ``MAX_ACT``, the FK/IK CUDA kernels and the topological sort are untouched.
* **Collision** — the block's geometry is concatenated onto the robot's
  collision array and its pairs onto the pair table, so ``ls_trajopt`` sweeps
  the *carried* block against the world with no change to its signature: it is
  handed ``robot_coll.with_attachments(aset)`` instead of ``robot_coll``.

So the pick-and-place below is four calls to the same solver with two collision
models, not a planner that knows what a "payload" is:

1. **reach** — home → pre-grasp, planned with the block still a *world*
   obstacle (the arm must not knock it over on the way in).
2. **approach / grasp** — a short straight joint-space descent onto the block,
   then ``grasp_from_current_pose`` turns "the block is *here* and the gripper
   is *there*" into ``T_LB``, so nobody writes that transform by hand and gets
   it inverted. The block leaves the world pool the instant it joins the robot:
   an object is never both an obstacle and part of the robot, which is the bug
   that makes a grasped object collide with itself.
3. **transport** — pre-grasp → pre-place over a wall, planned with the block
   attached, and reported against two baselines: the straight joint-space
   interpolation between the same endpoints (which drags the block straight
   through the wall), and the *same solver call on the bare model* — identical
   arguments except the collision model. Whether the bare model's path happens
   to clear the block is luck; the attachment is what makes it checked. Every
   clearance is reported per obstacle, because the Panda's base spheres sit a
   few centimetres inside the ground half-space at every configuration and a
   lumped minimum would report that constant forever.
4. **place / retreat** — lower, ``detach`` (the exact inverse of the grasp: the
   block returns to the world at the pose FK says it is at), retreat.

The playback is a MuJoCo scene driven kinematically in an mjviser window: the
arm's ``qpos`` is the planned path, and the block is posed by ``T_WL(q) · T_LB``
exactly while it is attached — i.e. the viewer shows the same composition the
collision checker used, not a re-derived one.

CPU or GPU; ``--cuda`` puts the trajopt on the CUDA kernels. ``--no-view``
prints the plan and its clearances without opening a window.
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import yourdfpy

import pyroffi as pk
from pyroffi.attachments import Attachment, AttachmentSet
from pyroffi.collision import HalfSpace, Sphere
from pyroffi.optimization_engines import LsTrajOptConfig, ls_trajopt

URDF_PATH = "resources/panda/panda_spherized.urdf"
MESH_DIR = "resources/panda/meshes"
EE_LINK = "panda_hand"

# --- Scene ------------------------------------------------------------------
BLOCK_HALF = np.array([0.03, 0.03, 0.03])
PICK_XY = np.array([0.45, -0.30])
PLACE_XY = np.array([0.45, 0.30])
BLOCK_START = np.array([PICK_XY[0], PICK_XY[1], BLOCK_HALF[2]])
BLOCK_GOAL = np.array([PLACE_XY[0], PLACE_XY[1], BLOCK_HALF[2]])
BLOCK_MASS = 0.25

# A wall between pick and place. Tall enough that a block dangling below the
# fingers has to be *lifted over* it, not just swung past.
WALL_CENTER = np.array([0.48, 0.0, 0.0])
WALL_HEIGHT = 0.30
WALL_THICK = 0.06
WALL_LENGTH = 0.36

# Hand-local +z standoff from the flange origin to the fingertip pads: the
# top-down grasp puts the hand this far above the block centre.
GRASP_Z = 0.10
PREGRASP_LIFT = 0.22  # hand height above the block for the pre-grasp pose
DOWN_WXYZ = np.array([0.0, 0.0, 1.0, 0.0])  # gripper +z pointing at the table

N_WAYPOINTS = 48
N_SEEDS = 16
LERP_STEPS = 8  # waypoints in the straight approach/lift/lower/retreat moves


def _fmt(v) -> str:
    return "[" + ", ".join(f"{float(x): .3f}" for x in np.ravel(v)) + "]"


def _grip_pose(center: np.ndarray, height: float) -> jaxlie.SE3:
    """Top-down gripper pose ``height`` above ``center`` (x, y taken from it)."""
    return jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(jnp.asarray(DOWN_WXYZ, dtype=jnp.float32)),
        jnp.asarray([center[0], center[1], height], dtype=jnp.float32),
    )


def build_world():
    """The static obstacles, plus the block as a world obstacle (pre-grasp)."""
    ground = HalfSpace.from_point_and_normal(
        np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    )
    # The wall as an overlapping grid of spheres. A row of capsules is the more
    # natural description, but the robot's collision model here is *spherized*,
    # and sphere-vs-sphere is the cheap kernel: a handful of capsule obstacles
    # makes XLA's fusion of the trajopt's collision Jacobian explode into a
    # multi-minute compile, while the sphere grid compiles in seconds. The
    # spheres overlap, so the wall is still a conservative solid.
    xs = np.arange(
        WALL_CENTER[0] - WALL_LENGTH / 2, WALL_CENTER[0] + WALL_LENGTH / 2 + 1e-9, 0.06
    )
    zs = np.arange(WALL_THICK / 2, WALL_HEIGHT + 1e-9, 0.06)
    centers = np.array([[x, WALL_CENTER[1], z] for x in xs for z in zs])
    wall = Sphere.from_center_and_radius(
        centers, np.full((centers.shape[0],), float(WALL_THICK / 2 + 0.01))
    )
    # The block, while it is still on the table, is an obstacle like any other.
    block_world = Sphere.from_center_and_radius(
        BLOCK_START[None, :], np.full((1,), float(np.linalg.norm(BLOCK_HALF)))
    )
    return ground, wall, block_world


def grasp_the_block(robot, cfg):
    """Capture the grasp from where the block and the gripper actually are.

    ``grasp_from_current_pose`` computes ``T_LB = T_WL(q)^-1 · T_WB`` so the
    caller states world poses (which they can measure) rather than a
    link-relative transform (which they would get backwards).

    The collision geometry is the block's *bounding sphere*: the robot's
    spherized model carries exactly one primitive type, and over-approximating
    is the right direction — it can refuse a plan that was feasible, but it
    never lets the robot drive the carried block through the wall.
    """
    hand = robot.links.names.index(EE_LINK)
    fingers = tuple(
        robot.links.names.index(n)
        for n in ("panda_hand", "panda_leftfinger", "panda_rightfinger")
        if n in robot.links.names
    )
    r = float(np.linalg.norm(BLOCK_HALF))
    m = BLOCK_MASS
    block = Attachment.from_geom(
        Sphere.from_center_and_radius(jnp.zeros((1, 3)), jnp.full((1,), r)),
        hand,
        jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        mass=jnp.asarray(m),
        # Solid cuboid about its centre.
        inertia_com=jnp.diag(
            jnp.asarray(
                m
                / 12.0
                * np.array(
                    [
                        (2 * BLOCK_HALF[1]) ** 2 + (2 * BLOCK_HALF[2]) ** 2,
                        (2 * BLOCK_HALF[0]) ** 2 + (2 * BLOCK_HALF[2]) ** 2,
                        (2 * BLOCK_HALF[0]) ** 2 + (2 * BLOCK_HALF[1]) ** 2,
                    ]
                )
            )
        ),
        name="block",
        ignored_links=fingers,  # the block is *supposed* to touch the gripper
    )
    T_world_block = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.identity(), jnp.asarray(BLOCK_START, dtype=jnp.float32)
    )
    block = block.grasp_from_current_pose(robot, cfg, T_world_block.wxyz_xyz)
    return AttachmentSet.empty().attach(block)


def plan(robot, coll, world, q_start, q_goal, *, key, use_cuda: bool, label: str):
    """One collision-free segment: ``ls_trajopt`` over straight-line seeds.

    ``coll`` is either the bare collision model or the one returned by
    ``with_attachments`` — the call is otherwise identical, which is the whole
    claim being demonstrated.
    """
    t = jnp.linspace(0.0, 1.0, N_WAYPOINTS)[:, None]
    line = q_start[None] * (1 - t) + q_goal[None] * t
    noise = 0.15 * jax.random.normal(key, (N_SEEDS, N_WAYPOINTS, q_start.shape[0]))
    # Seeds must agree at the endpoints; only the interior is perturbed.
    taper = jnp.sin(jnp.pi * t)[None]
    init = line[None] + noise * taper

    t0 = time.perf_counter()
    traj, costs, _ = ls_trajopt(
        init,
        q_start,
        q_goal,
        robot,
        coll,
        tuple(g for _, g in world),
        LsTrajOptConfig(
            n_outer_iters=12,
            n_ls_iters=12,
            n_cg_iters=12,
            w_smooth=2.5,
            w_acc=0.8,
            w_jerk=0.25,
            w_collision=10.0,
            w_collision_max=100.0,
            penalty_scale=3.0,
            collision_margin=0.02,
            w_trust=0.5,
            w_limits=1.0,
            w_endpoint=100.0,
            smooth_min_temperature=0.05,
            max_delta_per_step=0.1,
        ),
        use_cuda=use_cuda,
        key=key,
    )
    traj.block_until_ready()
    # Endpoint error is worth printing: ``ls_trajopt`` holds the endpoints with a
    # penalty, not a constraint, so "did it actually end where I asked" is a
    # question with a number rather than an assumption.
    ends = float(
        max(jnp.linalg.norm(traj[0] - q_start), jnp.linalg.norm(traj[-1] - q_goal))
    )
    print(
        f"  {label}: {time.perf_counter() - t0:5.2f}s, "
        f"best cost {float(jnp.min(costs)):.3f}, endpoint err {ends:.4f} rad"
    )
    return traj


def lerp(q0, q1, n: int = LERP_STEPS):
    """A short straight joint-space move (approach / lift / lower / retreat)."""
    t = jnp.linspace(0.0, 1.0, n)[:, None]
    return q0[None] * (1 - t) + q1[None] * t


def clearances(robot, coll, world, path, *, row: int | None = None) -> dict[str, float]:
    """Worst signed distance along ``path``, reported *per obstacle*.

    Per obstacle and not as one number, because a single minimum is useless
    here: the Panda's base spheres sit a few centimetres inside the ground
    half-space at every configuration, so a lumped minimum is that constant
    forever and says nothing about the wall. ``world`` is a tuple of
    ``(name, geom)``.

    ``row`` selects one row of the distance matrix. The attachment rows live in
    the tail of the collision array (that is the layout: ``K' = num_links + Σ
    num_prims``), so ``row=-1`` is the carried block's own clearance.
    """

    def at(cfg):
        # The self term belongs to the whole model, so it is reported only for
        # the whole model -- printing it beside a single row would read as that
        # row's number.
        out = [] if row is not None else [coll.compute_self_collision_distance(robot, cfg).min()]
        for _, g in world:
            d = coll.compute_world_collision_distance(robot, cfg, g)
            out.append((d if row is None else d[row]).min())
        return jnp.stack(out)

    d = np.asarray(jax.vmap(at)(path)).min(axis=0)
    names = ([] if row is not None else ["self"]) + [n for n, _ in world]
    return {n: float(v) for n, v in zip(names, d)}


def _report(label: str, c: dict[str, float]) -> None:
    print(f"    {label:22s}" + "  ".join(f"{n} {v: .4f}" for n, v in c.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true", help="Trajopt on CUDA kernels.")
    parser.add_argument(
        "--no-view", action="store_true", help="Print the plan; skip the viewer."
    )
    args = parser.parse_args()

    urdf = yourdfpy.URDF.load(URDF_PATH, mesh_dir=MESH_DIR)
    robot = pk.Robot.from_urdf(urdf)
    robot_coll = pk.collision.RobotCollisionSpherized.from_urdf(urdf)
    ground, wall, block_world = build_world()
    # Named so the clearance report can say *which* obstacle, which matters:
    # "ground" is always slightly negative (the base spheres overlap the plane)
    # and would otherwise mask everything the wall does.
    static = (("ground", ground), ("wall", wall))
    world_with_block = static + (("block", block_world),)

    q_home = (robot.joints.lower_limits + robot.joints.upper_limits) / 2
    # What this collision model says about the *empty* arm, so the numbers below
    # have something to be read against.
    self_base = float(robot_coll.compute_self_collision_distance(robot, q_home).min())

    print("\n1. IK for the pick/place key poses")
    key = jax.random.PRNGKey(0)
    poses = {
        "pregrasp": _grip_pose(BLOCK_START, BLOCK_START[2] + PREGRASP_LIFT),
        "grasp": _grip_pose(BLOCK_START, BLOCK_START[2] + GRASP_Z),
        "preplace": _grip_pose(BLOCK_GOAL, BLOCK_GOAL[2] + PREGRASP_LIFT),
        "place": _grip_pose(BLOCK_GOAL, BLOCK_GOAL[2] + GRASP_Z),
    }
    q = {}
    prev = q_home
    for name, pose in poses.items():
        q[name] = robot.inverse_kinematics(
            EE_LINK, pose, solver="ls", num_seeds=64, previous_cfg=prev
        )
        T = jaxlie.SE3(robot.forward_kinematics(q[name])[robot.links.names.index(EE_LINK)])
        err = float(jnp.linalg.norm(T.translation() - pose.translation()))
        print(f"  {name:9s} hand at {_fmt(T.translation())}  ik err {err * 1e3:5.2f} mm")
        prev = q[name]

    print("\n2. reach (block is still a world obstacle)")
    key, sub = jax.random.split(key)
    reach = plan(
        robot, robot_coll, world_with_block, q_home, q["pregrasp"],
        key=sub, use_cuda=args.cuda, label="home -> pregrasp",
    )
    _report("arm", clearances(robot, robot_coll, world_with_block, reach))
    print(f"      (self baseline for the empty arm at home: {self_base: .4f} m)")

    approach = lerp(reach[-1], q["grasp"])

    print("\n3. grasp capture")
    aset = grasp_the_block(robot, approach[-1])
    held_coll = robot_coll.with_attachments(aset)
    held_robot = robot.with_attachments(aset)
    a = aset.attachments[0]
    print(f"  T_link_body (hand <- block) = {_fmt(a.T_parent_body)}")
    print(
        f"  collision rows: {robot_coll.num_links} -> {held_coll.num_links}; "
        f"self-collision pairs: {len(robot_coll.active_idx_i)} -> "
        f"{len(held_coll.active_idx_i)}"
    )
    z = jnp.zeros((1, robot.joints.num_actuated_joints))
    qg = approach[-1][None]
    dtau = held_robot.inverse_dynamics(qg, z, z)[0] - robot.inverse_dynamics(qg, z, z)[0]
    print(f"  holding-torque delta from the {BLOCK_MASS * 1e3:.0f} g block: {_fmt(dtau)}")

    lift = lerp(q["grasp"], q["pregrasp"])

    print("\n4. transport (block attached; it is no longer a world obstacle)")
    key, sub = jax.random.split(key)
    transport = plan(
        robot, held_coll, static, q["pregrasp"], q["preplace"],
        key=sub, use_cuda=args.cuda, label="pregrasp -> preplace (attachment-aware)",
    )
    _report("arm", clearances(robot, robot_coll, static, transport))
    _report("carried block", clearances(robot, held_coll, static, transport, row=-1))

    # What the move costs without a planner: the straight joint-space
    # interpolation between the same two configurations, measured with the same
    # held model. This is the baseline the plan is worth something against.
    straight = lerp(q["pregrasp"], q["preplace"], N_WAYPOINTS)
    print("  straight joint-space interpolation (no planner), same endpoints:")
    _report("arm", clearances(robot, robot_coll, static, straight))
    _report("carried block", clearances(robot, held_coll, static, straight, row=-1))

    # And the same solver call on the *bare* model: identical arguments except
    # the collision model, so the block is simply not among the things being
    # swept. Whether its path happens to clear the wall is luck; the attachment
    # is what turns that into something checked.
    bare = plan(
        robot, robot_coll, static, q["pregrasp"], q["preplace"],
        key=sub, use_cuda=args.cuda, label="pregrasp -> preplace (payload ignored)",
    )
    _report("arm", clearances(robot, robot_coll, static, bare))
    _report("carried block", clearances(robot, held_coll, static, bare, row=-1))
    print("      ^ measured after the fact; the bare model never checked it")

    lower = lerp(q["preplace"], q["place"])
    retreat = lerp(q["place"], q["preplace"])

    # Detach is the exact inverse of the grasp: the block goes back to the world
    # at the pose FK says it is at, so there is no second source of truth for
    # where it landed.
    T_hand_end = jaxlie.SE3(
        robot.forward_kinematics(lower[-1])[robot.links.names.index(EE_LINK)]
    )
    landed = (T_hand_end @ jaxlie.SE3(a.T_parent_body)).translation()
    print(f"\n5. place: block released at {_fmt(landed)} (goal {_fmt(BLOCK_GOAL)})")

    path = jnp.concatenate([reach, approach, lift, transport, lower, retreat])
    # True exactly while the block is part of the robot: from the grasp at the
    # end of the approach through the last waypoint of the lowering move.
    held_from = len(reach) + len(approach) - 1
    held_to = held_from + len(lift) + len(transport) + len(lower)
    held_mask = np.zeros(len(path), dtype=bool)
    held_mask[held_from:held_to] = True
    print(f"  full path: {len(path)} waypoints, block carried for {held_mask.sum()}")

    if args.no_view:
        return 0
    view(urdf, robot, aset, np.asarray(path), held_mask)
    return 0


def view(urdf, robot, aset, path: np.ndarray, held_mask: np.ndarray):
    """Kinematic playback of the plan in an mjviser window.

    The arm's ``qpos`` is the planned path and the block is posed by
    ``T_WL(q) · T_LB`` while it is held — the same composition the collision
    checker used, so the render cannot disagree with what was planned.
    """
    try:
        import mujoco

        import mjviser
    except ImportError as e:
        print(f"\n({e.name} not installed; skipping the render)")
        return

    hand = robot.links.names.index(EE_LINK)
    T_LB = aset.attachments[0].T_parent_body

    # Block pose at every waypoint: attached -> FK compose; before/after ->
    # parked at the pick/place pose on the table.
    Ts = jaxlie.SE3(jnp.asarray(robot.forward_kinematics(jnp.asarray(path))[:, hand]))
    carried = np.asarray((Ts @ jaxlie.SE3(T_LB)).wxyz_xyz)
    parked_start = np.concatenate([[1.0, 0.0, 0.0, 0.0], BLOCK_START])
    parked_goal = np.concatenate([[1.0, 0.0, 0.0, 0.0], BLOCK_GOAL])
    block_pose = np.where(held_mask[:, None], carried, parked_start)
    after = np.flatnonzero(held_mask)[-1] + 1 if held_mask.any() else len(path)
    block_pose[after:] = parked_goal

    spec = mujoco.MjSpec()
    spec.add_texture(
        name="grid",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.25, 0.3, 0.35],
        rgb2=[0.35, 0.4, 0.45],
        width=512,
        height=512,
    )
    mat = spec.add_material(name="grid", texrepeat=[8, 8])
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], material="grid"
    )
    spec.worldbody.add_light(pos=[0.5, -0.5, 2.0], dir=[-0.15, 0.25, -1.0], castshadow=True)
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[WALL_CENTER[0], WALL_CENTER[1], WALL_HEIGHT / 2],
        size=[WALL_LENGTH / 2, WALL_THICK / 2, WALL_HEIGHT / 2],
        rgba=[0.45, 0.45, 0.5, 1.0],
        contype=0,
        conaffinity=0,
    )
    # Translucent marker at the drop pose.
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=BLOCK_GOAL,
        size=BLOCK_HALF,
        rgba=[0.2, 0.8, 0.3, 0.3],
        contype=0,
        conaffinity=0,
    )
    arm_frame = spec.worldbody.add_frame()
    spec.attach(mujoco.MjSpec.from_file(URDF_PATH), prefix="arm_", frame=arm_frame)

    # Playback is kinematic, so the block is a mocap body: its pose is written
    # every frame rather than integrated.
    box = spec.worldbody.add_body(name="block", pos=BLOCK_START, mocap=True)
    box.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=BLOCK_HALF,
        rgba=[200 / 255, 140 / 255, 60 / 255, 1.0],
        contype=0,
        conaffinity=0,
    )

    mj_model = spec.compile()
    mj_data = mujoco.MjData(mj_model)
    ndof = int(robot.joints.num_actuated_joints)
    mocap_id = mj_model.body("block").mocapid[0]

    fps = 30.0
    dwell = 0.06  # seconds per waypoint
    state = {"t": 0.0}

    def _apply(m, d, t: float):
        # Loop, with a beat of hold at each end so the grasp and the release
        # are readable rather than flashing past.
        span = dwell * (len(path) - 1)
        k = float(np.clip(t, 0.0, span)) / dwell
        i, f = int(k), k - int(k)
        j = min(i + 1, len(path) - 1)
        d.qpos[:ndof] = (1 - f) * path[i, :ndof] + f * path[j, :ndof]
        pose = block_pose[i] if f < 0.5 else block_pose[j]
        d.mocap_pos[mocap_id] = pose[4:]
        d.mocap_quat[mocap_id] = pose[:4]
        mujoco.mj_forward(m, d)

    def step_fn(m, d):
        span = dwell * (len(path) - 1)
        state["t"] += 1.0 / fps
        if state["t"] > span + 1.0:
            state["t"] = 0.0
        _apply(m, d, state["t"])
        time.sleep(1.0 / fps)

    def reset_fn(m, d):
        mujoco.mj_resetData(m, d)
        state["t"] = 0.0
        _apply(m, d, 0.0)

    reset_fn(mj_model, mj_data)
    print(
        f"\nPlaying the plan ({len(path)} waypoints, looping) — open the viser URL. "
        "Ctrl-C to exit."
    )
    mjviser.Viewer(mj_model, mj_data, step_fn=step_fn, reset_fn=reset_fn).run()


if __name__ == "__main__":
    raise SystemExit(main())
