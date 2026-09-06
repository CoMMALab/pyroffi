"""Block-stacking IOSP model: inverting a differentiable planner for
vertical tower construction on a Panda, adapting SPaSM's tower problem.

The forward model
-----------------
A Panda picks blocks from scattered initial positions and stacks them into a
vertical tower.  Each block is a 6cm cube (matching SPaSM's
`spasm.tower.env.TowerSimulation`).  The trajectory for each block is a
two-phase composed plan identical in structure to `tetris.py`:

  1. **Pick-to-place**: IK for the pick pose and place pose (atop the
     previous block or at the base), then per-segment trajopt.
  2. **Return**: trajopt back to home.

The stacking constraint distinguishes this from tetris: each block's place
target is at z = block_height * (stack_level + 0.5), and the cost includes a
z-alignment term that penalizes deviation from the target height.

Cost features (tied across all segments)
-----------------------------------------
  ``effort``     velocity norm
  ``smooth``     acceleration norm
  ``clearance``  obstacle avoidance (static obstacles + already-stacked blocks)
  ``orient``     EE tilt away from pointing downward
  ``z_align``    z-height deviation from the stacking target
  ``skeleton``   deviation from the task skeleton (pick/place poses)

theta = softmax(z) over these 6 features.
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
from iosp.model.tetris import (
    _multi_sphere_clearance, _orient_residual, _target_pose_batch,
    _ik_batch, _torque_residual, _self_collision_residual, CLEARANCE_MARGIN, SOFTMIN_TAU, SOFTNESS,
    DOWN_WXYZ, UP_AXIS, IK_RNG_KEY, IK_CONTINUITY_WEIGHT,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BLOCK_DIM = 0.06
BLOCK_HALF = BLOCK_DIM / 2.0

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

FEATURE_NAMES = ("effort", "smooth", "clearance", "orient", "z_align",
                 "torque", "skeleton")
K = len(FEATURE_NAMES)
SEGMENT_FEATURES = ("effort", "smooth", "clearance", "orient", "z_align",
                    "torque")
K_SEG = len(SEGMENT_FEATURES)

Q_HOME = jnp.array([0.0, -0.785, 0.0, -1.571, 0.0, 1.571, 0.785],
                    dtype=jnp.float32)

STANDOFF = 0.08


# ---------------------------------------------------------------------------
# Scene dataclasses
# ---------------------------------------------------------------------------

@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class TowerScene:
    """Context for one block-stacking pick-place demonstration."""
    q_start: jnp.ndarray        # (dof,)
    pick_pos: jnp.ndarray       # (3,) EE target at block's current position
    place_pos: jnp.ndarray      # (3,) EE target at stack position
    target_z: jnp.ndarray       # (1,) target z height for the placed block
    obs_center: jnp.ndarray     # (N_obs, 3) obstacle sphere centers
    obs_radius: jnp.ndarray     # (N_obs,) obstacle sphere radii


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class TowerFullScene:
    """Context for the stage-3 refine solve."""
    q_start: jnp.ndarray
    q_goal: jnp.ndarray         # = q_home
    obs_center: jnp.ndarray
    obs_radius: jnp.ndarray
    q_pick: jnp.ndarray
    q_place: jnp.ndarray
    target_z: jnp.ndarray       # (1,)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class TowerSegScene:
    """Segment scene with z-alignment target."""
    q_start: jnp.ndarray
    q_goal: jnp.ndarray
    obs_center: jnp.ndarray
    obs_radius: jnp.ndarray
    target_z: jnp.ndarray       # (1,)


# ---------------------------------------------------------------------------
# Problem class
# ---------------------------------------------------------------------------

class TowerProblem:
    """Block-stacking IOSP problem."""

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
        return TowerProblem(base=base, seg=seg)

    def pick_ik(self, pick_pos, refs):
        return _ik_batch(self, pick_pos, refs)

    def place_ik(self, place_pos, q_pick):
        return _ik_batch(self, place_pos, q_pick)

    # -- z-alignment residual -----------------------------------------------

    def _z_align_residual(self, q, target_z):
        """Penalty for EE z deviating from the target stacking height.
        Returns (T,) residual."""
        ee = self.base.ee_positions(q)  # (T, 3)
        return ee[:, 2] - target_z[0]

    # -- residuals -----------------------------------------------------------

    def segment_residual_fn(self, phase):
        problem = self.seg[phase]

        def residual_fn(x_flat, scene: TowerSegScene):
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
            z_align = self._z_align_residual(q, scene.target_z)
            torque = _torque_residual(self.base.robot, q)
            return (v.reshape(-1), a.reshape(-1), clearance, orient, z_align,
                    torque)

        return residual_fn

    def full_residual_fn(self):
        problem = self.seg["full"]

        def residual_fn(x_flat, scene: TowerFullScene):
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
            z_align = self._z_align_residual(q, scene.target_z)
            torque = _torque_residual(self.base.robot, q)
            skel = jnp.concatenate([
                q[IDX_PICK] - scene.q_pick,
                q[IDX_PLACE] - scene.q_place])
            return (v.reshape(-1), a.reshape(-1), clearance, orient,
                    z_align, torque, skel)

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

    def seeds(self, scenes: TowerScene):
        q_pick = self.pick_ik(scenes.pick_pos, scenes.q_start)
        q_place = self.place_ik(scenes.place_pos, q_pick)

        seg_scenes = {
            "approach": TowerSegScene(scenes.q_start, q_pick,
                                      scenes.obs_center, scenes.obs_radius,
                                      scenes.target_z),
            "place_traj": TowerSegScene(q_pick, q_place,
                                         scenes.obs_center, scenes.obs_radius,
                                         scenes.target_z),
            "return_traj": TowerSegScene(q_place, scenes.q_start,
                                          scenes.obs_center, scenes.obs_radius,
                                          scenes.target_z),
        }
        x0 = {p: jax.vmap(self.seg[p].seed)(seg_scenes[p]) for p in PHASES}
        return x0, seg_scenes, q_pick, q_place

    def solve(self, scenes, inner_by_phase, theta_seg, theta_full,
              refine_inner, *, stage2=True):
        x0, seg_scenes, q_pick, q_place = self.seeds(scenes)

        xs = {}
        for phase in PHASES:
            xs[phase] = jax.vmap(
                inner_by_phase[phase].solve_implicit,
                in_axes=(0, None, 0))(x0[phase], theta_seg, seg_scenes[phase])

        full_sc = TowerFullScene(
            scenes.q_start, scenes.q_start,
            scenes.obs_center, scenes.obs_radius,
            q_pick, q_place, scenes.target_z)

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


def make_tower_forward_solver(n_iters=60, robot=None, method=None, gd_lr=0.1):
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

BASE_XY = jnp.array([0.45, 0.0], dtype=jnp.float32)


def _block_obstacle_spheres(block_pos, block_half=BLOCK_HALF, n_spheres=8):
    """Approximate a cube at `block_pos` (x,y,z center) as collision spheres."""
    h = block_half
    offsets = np.array([
        [-h, -h, -h], [-h, -h, h], [-h, h, -h], [-h, h, h],
        [h, -h, -h], [h, -h, h], [h, h, -h], [h, h, h],
    ], dtype=np.float32)
    r = h * 0.5
    centers = np.asarray(block_pos) + offsets
    return centers, np.full(n_spheres, r, dtype=np.float32)


def sample_tower_scenes(rng, n, stack_level=0):
    """Sample n tower-stacking scenes for a single block at `stack_level`.

    stack_level=0: place on the table (base of tower)
    stack_level=k: place on top of the k-th block
    """
    table_obs_c = np.array([[0.5, 0.5, -0.05], [0.5, -0.5, -0.05],
                             [0.0, 0.0, -0.05]], dtype=np.float32)
    table_obs_r = np.array([0.15, 0.15, 0.2], dtype=np.float32)

    target_z = BLOCK_DIM * (stack_level + 0.5) + STANDOFF

    qs, picks, places, tgts, obs_c, obs_r = [], [], [], [], [], []
    for _ in range(n):
        q0 = np.asarray(Q_HOME) + rng.normal(scale=0.05, size=7).astype(np.float32)

        bx = rng.uniform(0.35, 0.55)
        by = rng.uniform(0.15, 0.45)
        pick = np.array([bx, by, BLOCK_HALF + STANDOFF], dtype=np.float32)

        px = float(BASE_XY[0]) + rng.normal(scale=0.01)
        py = float(BASE_XY[1]) + rng.normal(scale=0.01)
        place = np.array([px, py, target_z], dtype=np.float32)

        oc = table_obs_c.copy()
        orr = table_obs_r.copy()

        # Add already-stacked blocks as obstacles
        for lvl in range(stack_level):
            bz = BLOCK_DIM * (lvl + 0.5)
            bc, br = _block_obstacle_spheres(
                np.array([float(BASE_XY[0]), float(BASE_XY[1]), bz]))
            oc = np.concatenate([oc, bc], axis=0)
            orr = np.concatenate([orr, br], axis=0)

        qs.append(q0)
        picks.append(pick)
        places.append(place)
        tgts.append(np.array([target_z], dtype=np.float32))
        obs_c.append(oc)
        obs_r.append(orr)

    # Pad to uniform shape
    max_obs = max(c.shape[0] for c in obs_c)
    for i in range(n):
        pad_n = max_obs - obs_c[i].shape[0]
        if pad_n > 0:
            obs_c[i] = np.concatenate(
                [obs_c[i], np.full((pad_n, 3), 100.0, dtype=np.float32)])
            obs_r[i] = np.concatenate(
                [obs_r[i], np.full(pad_n, 0.01, dtype=np.float32)])

    f32 = lambda a: jnp.asarray(np.stack(a), dtype=jnp.float32)
    return TowerScene(
        q_start=f32(qs), pick_pos=f32(picks), place_pos=f32(places),
        target_z=f32(tgts), obs_center=f32(obs_c), obs_radius=f32(obs_r))
