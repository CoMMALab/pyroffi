"""Demonstration data: dynamics-aware Panda reach-around-an-obstacle.

Task class
----------
A 7-DoF Franka Panda moves between two joint-space endpoints past a single
spherical obstacle.  Each *context* is a fresh pair of endpoints plus a fresh
obstacle, anchored on the straight-line end-effector path so it is genuinely in
the way (``ioc.robot.problem.sample_scenes``: if the obstacle never blocks the
seed path the trajectory is the straight line regardless of the cost, and the
demonstrations carry no information).

The demonstrator is ``pyroffi``'s dynamics-aware trajectory optimizer -- the
same ``dynamics_trajopt`` engine DiffTORI optimizes through at training time --
minimising the E3 dynamic basis under known weights ``theta*``:

    effort  ||dq||^2 + collision  clearance^2 + smooth ||ddq||^2
                                 + torque  ||RNEA(q, dq, ddq)||^2

The torque term is what makes this worth imitating: the demonstrations depend on
the arm's mass and inertia, not on geometry alone, so a policy that only sees
states and actions has something non-trivial to recover.  This is deliberately
the *same* problem family the IOC experiments use, so their measured properties
carry over -- most importantly the stationarity screen.

Screening
---------
Three filters run before anything is written, each removing a distinct way a
"demonstration" can fail to be one.

**1. Valid scenes** (``sample_scenes_valid``, before solving).  The endpoints of
the inner problem are *clamped* boundary conditions -- the optimizer cannot move
them -- so if the obstacle overlaps the start or goal configuration, the
demonstration necessarily begins or ends in collision and no solver can fix it.
Measured on the first version of this dataset, which sampled scenes without this
check: 121/200 episodes penetrated, 84 at the start and 58 at the goal, while
only 7 penetrated anywhere in the interior.  Endpoints are also rejected for
self-collision and clipped to the joint limits.

**2. Stationarity** (``prob.screen_scenes``).  Contexts whose inner solve
plateaus are discarded.  A non-stationary solve is not an optimal demonstration:
it is wherever the L-BFGS budget ran out, and it depends on the solver path.

**3. Clear solutions** (``screen_solutions``, after solving).  Even from valid
endpoints the returned trajectory can clip the obstacle, because the teacher's
clearance term is a soft penalty, not a hard constraint.  Those episodes are
dropped rather than shipped.

Every discard rate is reported and stored in the dataset metadata.

Dataset format
--------------
Written as a **zarr replay buffer in the layout the authors' code expects**
(``diffusion_policy_3d/common/replay_buffer.py``), so their dataset, sequence
sampler and normaliser drop in unchanged:

    data/state          (N, 25)  float32
    data/action         (N, 7)   float32
    data/point_cloud    (N, P, 3) float32
    meta/episode_ends   (E,)     int64

One row per interior waypoint, one episode per context:

    state   q, dq, q_goal - q, obs_center - p_ee(q), obs_radius   (25,)
    action  (q_{t+1} - q_t) / action_scale, in [-1, 1]            (7,)

Everything in ``state`` is relative except the configuration itself, so the
policy sees the goal and obstacle in a frame that transfers across contexts.
Actions are scaled by one dataset-wide constant (stored in the metadata)
because DiffTORI's inner problem carries a unit-box barrier on the actions.

``point_cloud`` is the scene as the policy would perceive it: the robot's
collision spheres plus the obstacle sphere, sampled at each waypoint.  It exists
so the DP3 PointNet encoder path can be exercised on this data; the MLP encoder
path uses ``state`` alone and ignores it.

Usage
-----
    XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=2 \\
        python -m difftori.data.panda_reach --n-contexts 256

Run from the repository root (the default URDF paths are relative to it) with
``diffTORI`` on ``PYTHONPATH``.
"""

from __future__ import annotations

import json
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

DT = 0.1  # matches ioc.robot.bases.DT, the timestep the torque feature assumes


def _load_ioc():
    """Import the IOC problem definitions (this repo, not a package dependency)."""
    from ioc.inner import make_inner_solver
    from ioc.robot import bases, problem as prob

    return make_inner_solver, bases, prob


def build_solver(problem, bases, make_inner_solver, scenes_for_calibration,
                 torque_backend: str, n_iters: int, seed: int):
    """The dynamics-aware teacher, wired exactly as in `ioc.robot.e3_dynamics`."""
    from pyroffi.optimization_engines import DynamicsTrajOptConfig, dynamics_trajopt

    residual_fn, names = bases.dynamic(problem, torque_backend=torque_backend)
    cfg = DynamicsTrajOptConfig(n_iters=n_iters)

    def forward_solver(x0, cost_fn):
        return dynamics_trajopt(x0, cost_fn, cfg)

    scales = problem.calibrate(residual_fn, scenes_for_calibration,
                               jax.random.key(seed))
    inner = make_inner_solver(residual_fn, scales, forward_solver=forward_solver)
    return inner, names, scales


def sample_point_cloud(problem, q, scene, n_points: int, key) -> Array:
    """Scene point cloud at one configuration: robot spheres + the obstacle.

    Points are drawn on the surface of each collision sphere, which is what the
    robot and the obstacle actually present to a depth sensor.  This is a stand-
    in for MetaWorld's cropped depth back-projection, not a simulation of one.
    """
    coll = problem.robot_coll.at_config(problem.robot, q[None, :])
    centers = coll.pose.translation().reshape(-1, 3)
    radii = jnp.broadcast_to(coll.radius.reshape(-1), (centers.shape[0],))
    centers = jnp.concatenate([centers, scene.obs_center[None, :]], axis=0)
    radii = jnp.concatenate([radii, scene.obs_radius], axis=0)

    k_idx, k_dir = jax.random.split(key)
    idx = jax.random.randint(k_idx, (n_points,), 0, centers.shape[0])
    dirs = jax.random.normal(k_dir, (n_points, 3))
    dirs = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)
    return centers[idx] + radii[idx, None] * dirs


def trajectories_to_pairs(problem, demos, scenes) -> tuple[np.ndarray, np.ndarray]:
    """``(N, T, dof)`` trajectories -> flat ``(state, action)`` rows."""

    def per_context(q, scene):
        # dq_t: backward difference, zero at the first waypoint.
        dq = jnp.concatenate([jnp.zeros((1, q.shape[1]), q.dtype),
                              (q[1:] - q[:-1]) / DT], axis=0)
        p_ee = problem.ee_positions(q)
        obs = jnp.concatenate([
            q[:-1],
            dq[:-1],
            scene.q_goal[None, :] - q[:-1],
            scene.obs_center[None, :] - p_ee[:-1],
            jnp.broadcast_to(scene.obs_radius[None, :], (q.shape[0] - 1, 1)),
        ], axis=-1)
        return obs, q[1:] - q[:-1]

    obs, act = jax.vmap(per_context)(demos, scenes)
    return (np.asarray(obs).reshape(-1, obs.shape[-1]),
            np.asarray(act).reshape(-1, act.shape[-1]))


def endpoint_clearance(problem, q, obs_center, obs_radius):
    """Min robot-sphere clearance to the obstacle at one configuration."""
    coll = problem.robot_coll.at_config(problem.robot, q[None, :])
    d = (jnp.linalg.norm(coll.pose.translation() - obs_center, axis=-1)
         - coll.radius - obs_radius)
    return jnp.min(d)


def sample_scenes_valid(
    problem,
    rng,
    n: int,
    jitter: float = 0.35,
    endpoint_margin: float = 0.02,
    self_collision_margin: float = 0.0,
    max_tries: int = 40,
):
    """Scenes whose *endpoints* are collision-free, with independent jitter.

    Two changes from ``problem.sample_scenes``, which is written for the IOC
    experiments and is left untouched so those results stay reproducible:

    * **Endpoints are validated.**  See the module docstring -- they are clamped,
      so an obstacle overlapping them is unfixable by the solver and produces a
      demonstration that starts or ends inside the obstacle.
    * **Start and goal are jittered independently**, and by more.  The IOC
      sampler perturbs them antisymmetrically (``start + j``, ``goal - j``) by
      0.10 rad, which traces one narrow tube of end-effector paths.  Imitation
      learning needs the observation distribution to cover more than that; the
      identifiability argument for anchoring the *obstacle* on the seed path is
      unaffected and is kept.

    Rejection is per-scene with a retry budget, so the returned scene count is
    exact.  Raises if a scene cannot be placed within ``max_tries``.
    """
    from ioc.robot.problem import Q_GOAL, Q_START, Scene

    dof = problem.dof
    lower = np.asarray(problem.robot.joints.lower_limits)
    upper = np.asarray(problem.robot.joints.upper_limits)
    q_start0, q_goal0 = Q_START[:dof], Q_GOAL[:dof]

    starts, goals, centers, radii = [], [], [], []
    rejected = {"endpoint_obstacle": 0, "self_collision": 0}

    for _ in range(n):
        for _try in range(max_tries):
            qs = np.clip(q_start0 + rng.normal(scale=jitter, size=dof),
                         lower, upper)
            qg = np.clip(q_goal0 + rng.normal(scale=jitter, size=dof),
                         lower, upper)
            probe = Scene(q_start=jnp.asarray(qs), q_goal=jnp.asarray(qg),
                          obs_center=jnp.zeros(3), obs_radius=jnp.ones(1))

            # Self-collision at either endpoint makes the scene unusable
            # regardless of where the obstacle goes.
            sc = problem.robot_coll.compute_self_collision_distance(
                problem.robot, jnp.stack([probe.q_start, probe.q_goal]))
            if float(jnp.min(sc)) < self_collision_margin:
                rejected["self_collision"] += 1
                continue

            # Anchor the obstacle on the seed EE path (the identifiability
            # argument from `ioc.robot.problem`), then check the endpoints.
            q_seed = problem.unpack(problem.seed(probe), probe)
            p = np.asarray(problem.ee_positions(q_seed))
            t = rng.integers(problem.n_timesteps // 3,
                             2 * problem.n_timesteps // 3)
            r = rng.uniform(0.08, 0.14)
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            c = p[t] + direction * (r + rng.uniform(-0.05, 0.10))

            c_j, r_j = jnp.asarray(c), jnp.asarray([r])
            d_start = float(endpoint_clearance(problem, probe.q_start, c_j, r_j[0]))
            d_goal = float(endpoint_clearance(problem, probe.q_goal, c_j, r_j[0]))
            if min(d_start, d_goal) < endpoint_margin:
                rejected["endpoint_obstacle"] += 1
                continue

            starts.append(qs), goals.append(qg)
            centers.append(c), radii.append(np.array([r]))
            break
        else:
            raise RuntimeError(
                f"could not place a valid scene in {max_tries} tries "
                f"(rejections so far: {rejected}); loosen endpoint_margin or "
                "reduce jitter")

    scenes = Scene(q_start=jnp.asarray(np.stack(starts)),
                   q_goal=jnp.asarray(np.stack(goals)),
                   obs_center=jnp.asarray(np.stack(centers)),
                   obs_radius=jnp.asarray(np.stack(radii)))
    return scenes, rejected


def screen_solutions(problem, demos, scenes, margin: float = 0.0):
    """Keep only trajectories that clear the obstacle at every waypoint.

    The teacher's clearance term is a soft penalty weighted against effort,
    smoothness and torque, so a solved trajectory can still clip the obstacle.
    Returns ``(keep_indices, min_clearance_per_episode)``.
    """
    def worst(q, scene):
        coll = problem.robot_coll.at_config(problem.robot, q)
        d = (jnp.linalg.norm(coll.pose.translation() - scene.obs_center, axis=-1)
             - coll.radius - scene.obs_radius[0])
        return jnp.min(d.reshape(d.shape[0], -1), axis=-1).min()

    worst_per_ep = np.asarray(jax.vmap(worst)(demos, scenes))
    return np.flatnonzero(worst_per_ep >= margin), worst_per_ep


# -- multimodality ---------------------------------------------------------


def detour_seeds(problem, scene, n_dirs: int = 4, amplitude: float = 0.7):
    """Seeds that push the trajectory around *different sides* of the obstacle.

    The multimodality that matters in manipulation is homotopy: going left of an
    obstacle and going right are both locally optimal and neither is a small
    perturbation of the other.  Jittering the straight-line seed does not find
    them -- after a few L-BFGS steps every jittered start falls back into the
    basin nearest the seed, which is the point ``ioc.inner.solve``'s docstring
    makes about i.i.d. restarts resampling one basin.

    So the seeds are *structured*: displace the interior waypoints by a
    sinusoidal bump along +/-u and +/-v, an orthonormal basis of the plane
    perpendicular to the start->goal end-effector chord.  The task-space
    displacement is mapped to joints by the pseudo-inverse of ``d p_ee / dq``
    (autodiff through FK, so no assumption about body indexing), which only has
    to be good enough to land the seed in the right basin -- the solver does the
    rest.
    """
    x0 = problem.seed(scene)
    q = problem.unpack(x0, scene)                     # (T, dof)
    ee = problem.ee_positions(q)
    chord = ee[-1] - ee[0]
    chord = chord / (jnp.linalg.norm(chord) + 1e-9)

    # Any vector not parallel to the chord gives a first perpendicular.
    helper = jnp.where(jnp.abs(chord[0]) < 0.9,
                       jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0]))
    u = jnp.cross(chord, helper)
    u = u / (jnp.linalg.norm(u) + 1e-9)
    v = jnp.cross(chord, u)
    dirs = jnp.stack([u, -u, v, -v])[:n_dirs]

    T = q.shape[0]
    bump = jnp.sin(jnp.pi * jnp.arange(T) / (T - 1))[:, None]   # 0 at endpoints

    def seed_for(d):
        def per_waypoint(q_t, w):
            J = jax.jacobian(lambda qq: problem.ee_positions(qq[None, :])[0])(q_t)
            return q_t + jnp.linalg.pinv(J) @ (amplitude * w * d)

        q_new = jax.vmap(per_waypoint)(q, bump[:, 0])
        return q_new[1:-1].reshape(-1)   # interior waypoints only; endpoints clamped

    return jax.vmap(seed_for)(dirs)


def mode_signature(problem, q, scene):
    """Which way round the obstacle this trajectory goes, as an angle.

    Taken at the waypoint of closest approach: the obstacle-to-end-effector
    vector, projected onto the plane perpendicular to the chord.  Two solutions
    belong to the same mode when their signatures point the same way.
    """
    ee = problem.ee_positions(q)
    chord = ee[-1] - ee[0]
    chord = chord / (jnp.linalg.norm(chord) + 1e-9)
    helper = jnp.where(jnp.abs(chord[0]) < 0.9,
                       jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0]))
    u = jnp.cross(chord, helper)
    u = u / (jnp.linalg.norm(u) + 1e-9)
    v = jnp.cross(chord, u)

    offs = ee - scene.obs_center
    t_star = jnp.argmin(jnp.linalg.norm(offs, axis=-1))
    o = offs[t_star]
    o = o - jnp.dot(o, chord) * chord
    return jnp.arctan2(jnp.dot(o, v), jnp.dot(o, u))


def cluster_modes(signatures, costs, max_modes: int = 2,
                  separation_deg: float = 60.0):
    """Pick the cheapest solution from each distinct homotopy class.

    Greedy: take the lowest-cost solution, then the next one whose signature is
    at least ``separation_deg`` away from every mode already kept.  Returns the
    chosen indices.
    """
    sep = np.deg2rad(separation_deg)
    order = np.argsort(np.asarray(costs))
    sig = np.asarray(signatures)
    kept: list[int] = []
    for i in order:
        if len(kept) >= max_modes:
            break
        d = np.abs(np.angle(np.exp(1j * (sig[i] - sig[kept])))) if kept else None
        if not kept or float(np.min(d)) >= sep:
            kept.append(int(i))
    return kept


def solve_modes(problem, inner, scenes, theta_star, n_dirs, amplitude,
                max_modes, separation_deg, conv_tol):
    """Solve every scene from structured seeds; return per-mode trajectories.

    Returns ``(demos, scene_index, mode_index)`` where ``demos[k]`` is a
    trajectory, ``scene_index[k]`` says which scene it belongs to (several
    demonstrations share a scene -- that is the whole point) and ``mode_index[k]``
    which homotopy class it represents.
    """
    n = int(np.asarray(scenes.q_start).shape[0])
    demos, scene_idx, mode_idx = [], [], []

    for i in range(n):
        scene = jax.tree.map(lambda a: a[i], scenes)
        seeds = detour_seeds(problem, scene, n_dirs, amplitude)
        xs = jax.vmap(lambda s0: inner.solve(s0, theta_star, scene))(seeds)
        qs = jax.vmap(lambda x: problem.unpack(x, scene))(xs)

        costs = jax.vmap(lambda x: inner.cost(x, theta_star, scene))(xs)
        stat = jax.vmap(
            lambda x: jnp.linalg.norm(inner.grad_x(x, theta_star, scene)))(xs)
        sigs = jax.vmap(lambda q: mode_signature(problem, q, scene))(qs)

        # Only converged solutions are candidate modes: a plateaued solve is not
        # a local optimum and its "side" is an artefact of where it stopped.
        ok = np.flatnonzero(np.asarray(stat) < conv_tol)
        if len(ok) == 0:
            continue
        keep = cluster_modes(np.asarray(sigs)[ok], np.asarray(costs)[ok],
                             max_modes, separation_deg)
        for m, j in enumerate(keep):
            demos.append(qs[ok[j]])
            scene_idx.append(i)
            mode_idx.append(m)

    return (jnp.stack(demos), np.array(scene_idx, dtype=np.int64),
            np.array(mode_idx, dtype=np.int64))


def write_zarr(path, state, action, point_cloud, episode_ends, meta,
               extra=None):
    """Write the authors' replay-buffer layout (`ReplayBuffer.copy_from_path`)."""
    import zarr

    root = zarr.group(path, overwrite=True)
    data, meta_grp = root.create_group("data"), root.create_group("meta")
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    data.create_dataset("state", data=state.astype(np.float32),
                        dtype="float32", overwrite=True, compressor=compressor)
    data.create_dataset("action", data=action.astype(np.float32),
                        dtype="float32", overwrite=True, compressor=compressor)
    data.create_dataset("point_cloud", data=point_cloud.astype(np.float32),
                        dtype="float32", overwrite=True, compressor=compressor)
    meta_grp.create_dataset("episode_ends", data=episode_ends, dtype="int64",
                            overwrite=True, compressor=compressor)
    for k, v in (extra or {}).items():
        meta_grp.create_dataset(k, data=np.asarray(v), overwrite=True,
                                compressor=compressor)
    root.attrs["difftori"] = json.dumps(meta)


def main(
    urdf_path: str = "resources/panda/panda_spherized.urdf",
    srdf_path: str = "resources/panda/panda.srdf",
    mesh_dir: str = "resources/panda/meshes",
    n_timesteps: int = 16,
    n_contexts: int = 256,
    pool_factor: int = 3,
    torque_weight: float = 0.4,
    n_iters: int = 200,
    conv_tol: float = 1e-5,
    demo_noise: float = 0.0,
    torque_backend: str = "jax",
    jitter: float = 0.35,
    endpoint_margin: float = 0.02,
    solution_margin: float = 0.0,
    max_modes: int = 1,
    n_dirs: int = 4,
    seed_amplitude: float = 0.7,
    separation_deg: float = 60.0,
    n_points: int = 512,   # matches their MetaWorld demo generator
    seed: int = 0,
    chunk: int = 8,
    out: str = "diffTORI/data/panda_reach_expert.zarr",
):
    """Generate and write the demonstration set.

    ``demo_noise`` defaults to 0: unlike the IOC experiments, which perturb
    demonstrations to test robustness of weight recovery, imitation learning
    wants the teacher's actual optima.

    ``max_modes > 1`` turns on multimodal collection: each scene is solved from
    structured detour seeds and up to ``max_modes`` distinct homotopy classes
    are kept, so the dataset contains several *different* optimal action
    sequences for the same observation.  Without it every observation has one
    right answer, the CVAE latent has nothing to encode, and the posterior
    collapses (measured: 0.0015 nats/dim).
    """
    make_inner_solver, bases, prob = _load_ioc()

    if not jax.config.jax_enable_x64:
        print("WARNING: x64 is OFF; run with JAX_ENABLE_X64=1 for the teacher.")
    print(f"jax devices: {jax.devices()}")

    problem = prob.RobotProblem.load(urdf_path, srdf_path, mesh_dir, n_timesteps)
    rng = np.random.default_rng(seed)

    # theta*: the same split e3_dynamics uses, so the torque term carries real
    # weight -- otherwise the dynamic and kinematic problems coincide.
    rest = (1.0 - torque_weight) * np.array([0.5, 0.3, 0.2])
    theta_star = jnp.asarray(np.concatenate([rest, [torque_weight]]))
    print(f"theta* = {np.asarray(theta_star).round(3)}  {bases.DYNAMIC_NAMES}")

    t0 = time.time()
    pool, scene_rejects = sample_scenes_valid(
        problem, rng, n_contexts * pool_factor, jitter=jitter,
        endpoint_margin=endpoint_margin)
    print(f"sampled {n_contexts * pool_factor} valid scenes in "
          f"{time.time() - t0:.1f}s (rejected {scene_rejects})")

    inner, names, scales = build_solver(
        problem, bases, make_inner_solver, pool, torque_backend, n_iters, seed)
    print(f"feature scales = {np.asarray(scales)}")

    t0 = time.time()
    scenes, discard_rate, stat_vals = prob.screen_scenes(
        problem, pool, inner.stationarity, theta_star, conv_tol, n_contexts,
        chunk=chunk)
    print(f"screened {n_contexts}/{n_contexts * pool_factor} contexts "
          f"(discard rate {discard_rate:.2%}, "
          f"max ||grad|| kept = {stat_vals.max():.2e}) in {time.time() - t0:.1f}s")

    t0 = time.time()
    if max_modes > 1:
        demos, scene_idx, mode_idx = solve_modes(
            problem, inner, scenes, theta_star, n_dirs, seed_amplitude,
            max_modes, separation_deg, conv_tol)
        # Several demonstrations now share a scene, so the per-episode scene
        # arrays are expanded to match, one entry per demonstration.
        scenes = jax.tree.map(lambda a: a[scene_idx], scenes)
        counts = np.bincount(mode_idx, minlength=max_modes)
        print(f"modes found: " + ", ".join(
            f"mode {m}: {int(c)}" for m, c in enumerate(counts)))
        print(f"  {len(np.unique(scene_idx))} scenes yielded "
              f"{len(demos)} demonstrations "
              f"({len(demos) / max(len(np.unique(scene_idx)), 1):.2f} per scene)")
    else:
        _x0s, _x_star, demos = prob.make_demos(
            problem, inner.solve, scenes, theta_star, rng, demo_noise)
        scene_idx = np.arange(len(np.asarray(scenes.q_start)))
        mode_idx = np.zeros(len(scene_idx), dtype=np.int64)
    demos = jax.block_until_ready(demos)
    print(f"solved {len(np.asarray(scenes.q_start))} demonstrations in "
          f"{time.time() - t0:.1f}s")

    # Final filter: the clearance term is a soft penalty, so a solved
    # trajectory can still clip the obstacle.  Drop those rather than ship them.
    keep, worst = screen_solutions(problem, demos, scenes, margin=solution_margin)
    solution_discard = 1.0 - len(keep) / len(worst)
    print(f"clearance screen: kept {len(keep)}/{len(worst)} "
          f"(discard {solution_discard:.2%}); min clearance "
          f"mean {worst.mean():.4f} m, worst {worst.min():.4f} m")
    if len(keep) < n_contexts:
        print(f"WARNING: only {len(keep)} episodes survived; asked for "
              f"{n_contexts}. Raise --pool-factor.")
    if max_modes > 1:
        # Truncate by SCENE, not by demonstration: cutting the list at
        # `n_contexts` would drop the second mode of the last scenes and
        # silently make the tail of the dataset unimodal -- the exact property
        # the multimodal run exists to create.
        wanted = np.unique(scene_idx[keep])[:n_contexts]
        keep = keep[np.isin(scene_idx[keep], wanted)]
    else:
        keep = keep[:n_contexts]
    demos = demos[keep]
    scenes = jax.tree.map(lambda a: a[keep], scenes)
    worst_kept = worst[keep]
    scene_idx, mode_idx = scene_idx[keep], mode_idx[keep]
    n_contexts = len(keep)   # now counts demonstrations, not scenes

    obs, act = trajectories_to_pairs(problem, demos, scenes)
    action_scale = float(np.abs(act).max())
    act_norm = act / action_scale

    # Point clouds, one per row, matching the (state, action) ordering.
    n_steps = n_timesteps - 1
    keys = jax.random.split(jax.random.key(seed + 1), n_contexts)
    clouds = np.concatenate([
        np.asarray(jax.vmap(lambda q, k: sample_point_cloud(
            problem, q, jax.tree.map(lambda a: a[i], scenes), n_points, k))(
                demos[i, :n_steps], jax.random.split(keys[i], n_steps)))
        for i in range(n_contexts)
    ], axis=0)

    # One episode per context; episode_ends are cumulative row counts.
    episode_ends = np.arange(1, n_contexts + 1, dtype=np.int64) * n_steps

    meta = {
        "task": "panda_reach_obstacle_dynamic",
        "teacher": "pyroffi.optimization_engines.dynamics_trajopt",
        "basis": list(names),
        "theta_star": np.asarray(theta_star).tolist(),
        "feature_scales": np.asarray(scales).tolist(),
        "n_contexts": int(n_contexts),
        "n_timesteps": int(n_timesteps),
        "dt": DT,
        "discard_rate": float(discard_rate),
        "max_stationarity_kept": float(stat_vals.max()),
        "scene_rejections": scene_rejects,
        "solution_discard_rate": float(solution_discard),
        "min_clearance_mean": float(worst_kept.mean()),
        "min_clearance_worst": float(worst_kept.min()),
        "jitter": float(jitter),
        "endpoint_margin": float(endpoint_margin),
        "solution_margin": float(solution_margin),
        "demo_noise": float(demo_noise),
        "torque_backend": torque_backend,
        "action_scale": action_scale,
        "state_layout": ["q(7)", "dq(7)", "q_goal-q(7)", "obs_center-p_ee(3)",
                         "obs_radius(1)"],
        "n_points": int(n_points),
        "seed": int(seed),
    }

    meta["max_modes"] = int(max_modes)
    meta["mode_counts"] = np.bincount(mode_idx, minlength=max_modes).tolist()
    meta["scenes_used"] = int(len(np.unique(scene_idx)))
    write_zarr(out, obs, act_norm, clouds, episode_ends, meta,
               extra={"scene_index": scene_idx, "mode_index": mode_idx})
    print(f"wrote {out}: state {obs.shape}, action {act_norm.shape}, "
          f"point_cloud {clouds.shape}, {len(episode_ends)} episodes, "
          f"action_scale={action_scale:.4f}")
    return meta


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
