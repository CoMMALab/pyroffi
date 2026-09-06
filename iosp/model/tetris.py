"""Tetris-packing IOSP model: inverting a differentiable planner for
block-packing on a Panda, adapting SPaSM's tetris problem structure.

The forward model
-----------------
A Panda picks blocks from scattered initial positions and places them inside
a walled goal region.  Each block is a spherized L- or O-shaped tetromino
(matching SPaSM's `spasm.tetris.env`), and the trajectory for each block is a
two-phase composed plan:

  1. **Pick-to-place**: IK for the pick pose (block's current position +
     standoff), trajopt from q_start to q_pick, then from q_pick to q_place
     (target inside the goal region + standoff).
  2. **Return**: trajopt from q_place back to q_home.

For the IOSP paper, we use N=1 block as the base case (single pick-place with
wall + obstacle geometry) and N=3 for scaling.

Cost features (tied across all segments)
-----------------------------------------
  ``effort``     velocity norm ||q[t+1] - q[t]||^2
  ``smooth``     acceleration norm ||q[t+2] - 2q[t+1] + q[t]||^2
  ``clearance``  obstacle avoidance (walls + static blocks)
  ``orient``     EE tilt away from pointing downward
  ``skeleton``   deviation from the task skeleton (pick/place poses)

theta = softmax(z) over these 5 features, shared across all segments and the
refine pass -- same tied-model argument as `pickplace.py`.
"""

import dataclasses
import os

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np

jax.config.update(
    "jax_compilation_cache_dir",
    os.environ.get("IOSP_JAX_CACHE_DIR",
                   os.path.expanduser("~/.cache/jax_pyroffi_iosp")),
)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 5)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

from ioc.robot.problem import RobotProblem, Scene

from pyroffi.optimization_engines import _implicit_diff
_implicit_diff.CANONICAL_BY_DEFAULT = True
from pyroffi.optimization_engines._sqp_ik import sqp_ik_solve_cuda_batch

# ---------------------------------------------------------------------------
# Geometry: block and wall definitions
# ---------------------------------------------------------------------------

CLEARANCE_MARGIN = 0.05
SOFTMIN_TAU = 0.02
SOFTNESS = 60.0

SPH_RADIUS = 0.03

DOWN_WXYZ = jnp.array([0.0, 1.0, 0.0, 0.0])
UP_AXIS = jnp.array([0.0, 0.0, 1.0])
IK_RNG_KEY = jax.random.PRNGKey(0)
IK_CONTINUITY_WEIGHT = 1.0

TORQUE_DT = 0.1     # [s] waypoint spacing for the finite-difference qd/qdd
GRAVITY = -9.81


SELF_MARGIN = 0.01   # [m] self-collision clearance margin (hinge point)


def _self_collision_residual(robot_coll, robot, q):
    """Smooth arm SELF-collision residual (arm-vs-arm), the term SPaSM prices in
    `arm_collision_cost` but iosp's obstacle-only `clearance` lacks.

    `compute_self_collision_distance` returns per-pair distances over the active
    (SRDF-filtered, non-adjacent) self-collision pairs, so a soft-min over them
    -- not the built-in hard min -- keeps this smooth for the implicit-diff
    adjoint (cf. `clearance_residual`'s soft-min fix).  Returns (T,)."""
    d = jax.vmap(lambda qi: robot_coll.compute_self_collision_distance(robot, qi))(q)
    d_min = -SOFTMIN_TAU * jax.scipy.special.logsumexp(-d / SOFTMIN_TAU, axis=-1)
    return jax.nn.softplus(SOFTNESS * (SELF_MARGIN - d_min)) / SOFTNESS


def _torque_residual(robot, q, dt=TORQUE_DT, gravity=GRAVITY):
    """RNEA joint torques at the interior knots (GRiD inverse dynamics).

    The `torque` cost feature: prices dynamic effort (mass/gravity/Coriolis),
    not just kinematic velocity, so the demonstrations are dynamically -- not
    only geometrically -- meaningful.  Central differences for qd/qdd, matching
    `ioc.robot.bases.dynamic`; routed through GRiD's CUDA FFI (`use_cuda=True`),
    whose analytic `idsva_so` custom_jvp keeps `jax.hessian` (the implicit
    adjoint) working through it."""
    qd = (q[2:] - q[:-2]) / (2.0 * dt)
    qdd = (q[2:] - 2.0 * q[1:-1] + q[:-2]) / (dt ** 2)
    qm = q[1:-1]
    tau = robot.inverse_dynamics(qm, qd, qdd, gravity=gravity, use_cuda=True)
    return tau.reshape(-1)


def create_tetris_spheres(shape="L", sph_radius=SPH_RADIUS):
    """Spherized tetromino, matching SPaSM's `create_tetris_spheres`."""
    coords = {
        "L": jnp.array([(0, 0, 0), (0, 1, 0), (0, -1, 0), (1, -1, 0)], dtype=jnp.float32),
        "O": jnp.array([(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)], dtype=jnp.float32),
    }[shape]
    n = coords.shape[0]
    spheres = jnp.zeros((n + 2, 4), dtype=jnp.float32)
    spheres = spheres.at[:n, :3].set(coords * sph_radius * 2)
    spheres = spheres.at[:n, 3].set(sph_radius)
    stick = jnp.array([
        [0.0, 0.0, -sph_radius * 1.25, sph_radius / 2],
        [0.0, 0.0, -sph_radius * 2.0, sph_radius / 2],
    ], dtype=jnp.float32)
    spheres = spheres.at[n:, :].set(stick)
    z_offset = -spheres[-1, 2]
    spheres = spheres.at[:, 2].add(z_offset)
    return spheres


def block_pose_to_spheres(block_spheres, pose_xyzyaw):
    """Transform local block spheres by a pose (x, y, z, yaw) -> (K, 4)."""
    pos = pose_xyzyaw[:3]
    yaw = pose_xyzyaw[3]
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    R = jnp.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=jnp.float32)
    centers = block_spheres[:, :3] @ R.T + pos
    return jnp.concatenate([centers, block_spheres[:, 3:]], axis=-1)


def create_goal_walls(goal_center, goal_dims, wall_height=0.045,
                      wall_thickness=0.015):
    """AABB walls around the goal region, as (N_walls, 6) [x1,y1,z1,x2,y2,z2]."""
    cx, cy, cz = goal_center
    dx, dy, _ = goal_dims
    walls = jnp.array([
        [cx - dx/2, cy + dy/2, cz,
         cx + dx/2, cy + dy/2 + wall_thickness, cz + wall_height],
        [cx - dx/2, cy - dy/2 - wall_thickness, cz,
         cx + dx/2, cy - dy/2, cz + wall_height],
        [cx - dx/2 - wall_thickness, cy - dy/2, cz,
         cx - dx/2, cy + dy/2, cz + wall_height],
        [cx + dx/2, cy - dy/2, cz,
         cx + dx/2 + wall_thickness, cy + dy/2, cz + wall_height],
    ], dtype=jnp.float32)
    return walls


# ---------------------------------------------------------------------------
# Segment layout
# ---------------------------------------------------------------------------

PHASES = ("approach", "place_traj", "return_traj")
N_APPROACH = 8
N_PLACE_TRAJ = 10
N_RETURN = 6
SEGMENT_LEN = {"approach": N_APPROACH, "place_traj": N_PLACE_TRAJ,
                "return_traj": N_RETURN}

N_FULL = N_APPROACH + (N_PLACE_TRAJ - 1) + (N_RETURN - 1)

PHASE_SPAN, _s = {}, 0
for _p in PHASES:
    PHASE_SPAN[_p] = (_s, _s + SEGMENT_LEN[_p])
    _s += SEGMENT_LEN[_p] - 1
del _s, _p
assert PHASE_SPAN["return_traj"][1] == N_FULL

IDX_PICK = PHASE_SPAN["approach"][1] - 1
IDX_PLACE = PHASE_SPAN["place_traj"][1] - 1

FEATURE_NAMES = ("effort", "smooth", "clearance", "orient", "torque", "skeleton")
K = len(FEATURE_NAMES)
SEGMENT_FEATURES = ("effort", "smooth", "clearance", "orient", "torque")
K_SEG = len(SEGMENT_FEATURES)


# ---------------------------------------------------------------------------
# Scene dataclass
# ---------------------------------------------------------------------------

@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class TetrisScene:
    """Context for one tetris pick-place demonstration."""
    q_start: jnp.ndarray       # (dof,) home configuration
    pick_pos: jnp.ndarray      # (3,) block's current EE target (with standoff)
    place_pos: jnp.ndarray     # (3,) target inside goal (with standoff)
    obs_center: jnp.ndarray    # (N_obs, 3) obstacle sphere centers
    obs_radius: jnp.ndarray    # (N_obs,) obstacle sphere radii


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class TetrisFullScene:
    """Context for the stage-3 refine solve."""
    q_start: jnp.ndarray       # (dof,)
    q_goal: jnp.ndarray        # (dof,) = q_home for return
    obs_center: jnp.ndarray    # (N_obs, 3)
    obs_radius: jnp.ndarray    # (N_obs,)
    q_pick: jnp.ndarray        # (dof,)
    q_place: jnp.ndarray       # (dof,)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SegmentScene:
    """Minimal segment scene, compatible with RobotProblem.unpack/seed."""
    q_start: jnp.ndarray       # (dof,)
    q_goal: jnp.ndarray        # (dof,)
    obs_center: jnp.ndarray    # (N_obs, 3) or (3,) if single obs
    obs_radius: jnp.ndarray    # (N_obs,) or (1,) if single obs


# ---------------------------------------------------------------------------
# Collision helpers (smooth, jit-friendly)
# ---------------------------------------------------------------------------

def _multi_sphere_clearance(robot_coll, robot, q, obs_centers, obs_radii):
    """Smooth clearance residual against multiple obstacle spheres.

    Returns a (T,) vector of clearance violations, soft-min over all
    robot spheres × all obstacles.
    """
    coll = robot_coll.at_config(robot, q)
    coll_pos = coll.pose.translation()   # (..., T, S, 3)
    coll_rad = coll.radius               # (..., T, S)

    # (..., T, S, N_obs) pairwise distances — vmap-safe via expand_dims
    # coll_pos: (..., T, S, 3) -> (..., T, S, 1, 3)
    # obs_centers: (..., N_obs, 3) -> (..., 1, 1, N_obs, 3)
    cp = jnp.expand_dims(coll_pos, -2)
    oc = jnp.expand_dims(jnp.expand_dims(obs_centers, -3), -3)
    d = (jnp.linalg.norm(cp - oc, axis=-1)
        - jnp.expand_dims(coll_rad, -1)
        - obs_radii)
    d_flat = d.reshape(*d.shape[:-2], -1)   # (..., T, S*N_obs)
    d_min = -SOFTMIN_TAU * jax.scipy.special.logsumexp(
        -d_flat / SOFTMIN_TAU, axis=-1)
    return jax.nn.softplus(SOFTNESS * (CLEARANCE_MARGIN - d_min)) / SOFTNESS


def _orient_residual(robot, ee_index, q):
    """Tilt of the EE away from pointing straight down: (T, 2) -> flat."""
    quat = robot.forward_kinematics(q)[..., ee_index, 0:4]
    return quat[:, 1:3].reshape(-1)


# ---------------------------------------------------------------------------
# Problem class
# ---------------------------------------------------------------------------

class TetrisProblem:
    """Tetris-packing IOSP problem: pick-place with wall and block obstacles."""

    def __init__(self, base: RobotProblem, seg: dict):
        self.base = base
        self.seg = seg

    @property
    def dof(self):
        return self.base.dof

    @property
    def ee_index(self):
        return self.base.ee_index

    @staticmethod
    def load(urdf_path, srdf_path, mesh_dir):
        base = RobotProblem.load(urdf_path, srdf_path, mesh_dir, n_timesteps=2)
        seg = {p: dataclasses.replace(base, n_timesteps=SEGMENT_LEN[p])
               for p in PHASES}
        seg["full"] = dataclasses.replace(
            base, n_timesteps=N_FULL,
            pinned_rows=((IDX_PICK, "q_pick"), (IDX_PLACE, "q_place")))
        return TetrisProblem(base=base, seg=seg)

    # -- IK ------------------------------------------------------------------

    def pick_ik(self, pick_pos, refs):
        return _ik_batch(self, pick_pos, refs)

    def place_ik(self, place_pos, q_pick):
        return _ik_batch(self, place_pos, q_pick)

    # -- residuals -----------------------------------------------------------

    def segment_residual_fn(self, phase):
        problem = self.seg[phase]

        def residual_fn(x_flat, scene: SegmentScene):
            q = problem.unpack(x_flat, scene)
            v = q[1:] - q[:-1]
            a = q[2:] - 2.0 * q[1:-1] + q[:-2]
            clearance = jnp.concatenate([
                _multi_sphere_clearance(
                    self.base.robot_coll, self.base.robot, q,
                    scene.obs_center.reshape(-1, 3),
                    scene.obs_radius.reshape(-1)),
                _self_collision_residual(
                    self.base.robot_coll, self.base.robot, q)[..., None]], axis=-1)
            orient = _orient_residual(self.base.robot, self.ee_index, q)
            torque = _torque_residual(self.base.robot, q)
            return (v.reshape(-1), a.reshape(-1), clearance, orient, torque)

        return residual_fn

    def full_residual_fn(self):
        problem = self.seg["full"]

        def residual_fn(x_flat, scene: TetrisFullScene):
            q = problem.unpack(x_flat, scene)
            v = q[1:] - q[:-1]
            a = q[2:] - 2.0 * q[1:-1] + q[:-2]
            clearance = jnp.concatenate([
                _multi_sphere_clearance(
                    self.base.robot_coll, self.base.robot, q,
                    scene.obs_center.reshape(-1, 3),
                    scene.obs_radius.reshape(-1)),
                _self_collision_residual(
                    self.base.robot_coll, self.base.robot, q)[..., None]], axis=-1)
            orient = _orient_residual(self.base.robot, self.ee_index, q)
            torque = _torque_residual(self.base.robot, q)
            skel = jnp.concatenate([
                q[IDX_PICK] - scene.q_pick,
                q[IDX_PLACE] - scene.q_place])
            return (v.reshape(-1), a.reshape(-1), clearance, orient, torque, skel)

        return residual_fn

    def calibrate_segment(self, phase, residual_fn, scenes, key,
                          n_probe=16, jitter=0.15):
        problem = self.seg[phase]

        def raw(scene, k):
            x0 = problem.seed(scene)
            x = x0 + jitter * jax.random.normal(k, x0.shape)
            rs = residual_fn(x, scene)
            return jnp.stack([jnp.sum(r**2) for r in rs])

        keys = jax.random.split(key, n_probe)
        vals = jax.vmap(jax.vmap(raw, in_axes=(None, 0)),
                        in_axes=(0, None))(scenes, keys)
        scales = jnp.mean(jnp.abs(vals.reshape(-1, vals.shape[-1])), axis=0)
        assert bool(jnp.all(scales > 1e-8)), \
            f"{phase}: degenerate feature scale {scales}"
        return scales

    def calibrate_full(self, residual_fn, scenes, key,
                       n_probe=16, jitter=0.15):
        problem = self.seg["full"]

        def raw(scene, k):
            x0 = problem.seed(scene)
            x = x0 + jitter * jax.random.normal(k, x0.shape)
            rs = residual_fn(x, scene)
            return jnp.stack([jnp.sum(r**2) for r in rs])

        keys = jax.random.split(key, n_probe)
        vals = jax.vmap(jax.vmap(raw, in_axes=(None, 0)),
                        in_axes=(0, None))(scenes, keys)
        scales = jnp.mean(jnp.abs(vals.reshape(-1, vals.shape[-1])), axis=0)
        scales = jnp.where(scales > 1e-8, scales, 1.0)  # pinned feature -> benign 0 scale
        return scales

    def ee_positions(self, q):
        return self.base.ee_positions(q)

    # -- composed forward solve ----------------------------------------------

    def seeds(self, scenes: TetrisScene, standoff=0.06):
        """Compute IK and per-segment seeds."""
        q_pick = self.pick_ik(scenes.pick_pos, scenes.q_start)
        q_place = self.place_ik(scenes.place_pos, q_pick)

        seg_scenes = {
            "approach": SegmentScene(scenes.q_start, q_pick,
                                     scenes.obs_center, scenes.obs_radius),
            "place_traj": SegmentScene(q_pick, q_place,
                                       scenes.obs_center, scenes.obs_radius),
            "return_traj": SegmentScene(q_place, scenes.q_start,
                                        scenes.obs_center, scenes.obs_radius),
        }
        x0 = {p: jax.vmap(self.seg[p].seed)(seg_scenes[p]) for p in PHASES}
        return x0, seg_scenes, q_pick, q_place

    def solve(self, scenes, inner_by_phase, theta_seg, theta_full,
              refine_inner, *, stage2=True):
        """Full composed forward solve: IK -> per-segment -> refine."""
        x0, seg_scenes, q_pick, q_place = self.seeds(scenes)

        xs = {}
        for phase in PHASES:
            xs[phase] = jax.vmap(
                inner_by_phase[phase].solve_implicit,
                in_axes=(0, None, 0))(x0[phase], theta_seg, seg_scenes[phase])

        # Stage 3: refine
        full_sc = TetrisFullScene(
            scenes.q_start, scenes.q_start,
            scenes.obs_center, scenes.obs_radius,
            q_pick, q_place)

        if stage2:
            rows = []
            for i, ph in enumerate(PHASES):
                q = jax.vmap(self.seg[ph].unpack)(xs[ph], seg_scenes[ph])
                rows.append(q[:, 1:] if i > 0 else q)
            q_cat = jnp.concatenate(rows, axis=1)
            x0_full = q_cat[:, 1:-1, :].reshape(q_cat.shape[0], -1)
        else:
            x0_full = jax.vmap(self.seg["full"].seed)(full_sc)

        xs["full"] = jax.vmap(
            refine_inner.solve_implicit,
            in_axes=(0, None, 0))(x0_full, theta_full, full_sc)

        return xs, seg_scenes, full_sc, q_pick, q_place


def _target_pose_batch(pos_batch, wxyz=DOWN_WXYZ):
    n = pos_batch.shape[0]
    return jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3(wxyz=jnp.broadcast_to(wxyz, (n, 4)).astype(jnp.float32)),
        translation=pos_batch.astype(jnp.float32),
    )


def _ik_batch(problem, target_pos, refs):
    return sqp_ik_solve_cuda_batch(
        problem.base.robot, problem.ee_index,
        _target_pose_batch(target_pos),
        IK_RNG_KEY, refs.astype(jnp.float32),
        continuity_weight=IK_CONTINUITY_WEIGHT,
    ).astype(refs.dtype)


def make_tetris_forward_solver(n_iters=60, robot=None, method=None, gd_lr=0.1):
    from pyroffi.optimization_engines import DynamicsTrajOptConfig, dynamics_trajopt
    method = method or os.environ.get("IOSP_TRAJOPT", "lbfgs")
    if method == "projected_gd":
        lo = tuple(float(v) for v in np.asarray(robot.joints.lower_limits))
        hi = tuple(float(v) for v in np.asarray(robot.joints.upper_limits))
        cfg = DynamicsTrajOptConfig(n_iters=n_iters, method="projected_gd",
                                    gd_lr=gd_lr, q_lo=lo, q_hi=hi, dof=len(lo))
    else:
        cfg = DynamicsTrajOptConfig(n_iters=n_iters, early_stop=False, unroll_tail=0,
                                    soft_line_search=False, soft_curvature_gate=False)
    return lambda x0, cost_fn: dynamics_trajopt(x0, cost_fn, cfg)


# ---------------------------------------------------------------------------
# Scene sampling
# ---------------------------------------------------------------------------

GOAL_CENTER = jnp.array([0.3, 0.0, -0.005], dtype=jnp.float32)
GOAL_DIMS = jnp.array([0.18, 0.30, 0.01], dtype=jnp.float32)
BLOCK_Z = 0.06  # approximate resting z for blocks

Q_HOME = jnp.array([0.0, -0.785, 0.0, -1.571, 0.0, 1.571, 0.785],
                    dtype=jnp.float32)

STANDOFF = 0.06


def sample_tetris_scenes(rng, n, num_blocks=1):
    """Sample n tetris pick-place scenes.

    Each scene has:
    - A block at a random position on the table
    - A target position inside the goal region
    - Wall obstacles (converted to spheres for the clearance residual)
    - Previously-placed blocks as additional obstacle spheres

    For N=1, there are no inter-block obstacles.
    """
    walls = create_goal_walls(GOAL_CENTER, GOAL_DIMS)
    wall_spheres = _walls_to_spheres(walls)

    qs, picks, places, obs_c, obs_r = [], [], [], [], []
    for _ in range(n):
        q0 = np.asarray(Q_HOME) + rng.normal(scale=0.05, size=7).astype(np.float32)

        # Block starts on the table, away from goal
        bx = rng.uniform(0.35, 0.55)
        by = rng.uniform(-0.4, -0.15)
        pick = np.array([bx, by, BLOCK_Z + STANDOFF], dtype=np.float32)

        # Target inside goal region
        px = GOAL_CENTER[0] + rng.uniform(-GOAL_DIMS[0]/3, GOAL_DIMS[0]/3)
        py = GOAL_CENTER[1] + rng.uniform(-GOAL_DIMS[1]/3, GOAL_DIMS[1]/3)
        place = np.array([float(px), float(py), BLOCK_Z + STANDOFF],
                         dtype=np.float32)

        qs.append(q0)
        picks.append(pick)
        places.append(place)
        obs_c.append(np.asarray(wall_spheres[:, :3]))
        obs_r.append(np.asarray(wall_spheres[:, 3]))

    f32 = lambda a: jnp.asarray(np.stack(a), dtype=jnp.float32)
    return TetrisScene(
        q_start=f32(qs), pick_pos=f32(picks), place_pos=f32(places),
        obs_center=f32(obs_c), obs_radius=f32(obs_r))


def _walls_to_spheres(walls, n_per_edge=5, radius=0.02):
    """Approximate AABB walls as a set of spheres for the clearance residual."""
    spheres = []
    for w in walls:
        x1, y1, z1 = float(w[0]), float(w[1]), float(w[2])
        x2, y2, z2 = float(w[3]), float(w[4]), float(w[5])
        for i in range(n_per_edge):
            for j in range(max(1, n_per_edge // 2)):
                t1 = i / max(n_per_edge - 1, 1)
                t2 = j / max(n_per_edge // 2 - 1, 1) if n_per_edge > 2 else 0.5
                x = x1 + t1 * (x2 - x1)
                y = y1 + t1 * (y2 - y1)
                z = z1 + t2 * (z2 - z1)
                spheres.append([x, y, z, radius])
    return jnp.array(spheres, dtype=jnp.float32)
