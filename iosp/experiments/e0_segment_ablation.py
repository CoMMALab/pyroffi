"""E0 — Segment-freeze ablation for the composed pick-and-place chain.

Freezes all but a growing subset of segments at ground-truth weights, isolating
where in the 4-segment chain recovery degrades.  A sudden cliff when a specific
segment joins the free set points at that segment's boundary-condition wiring.

The free/frozen split uses a runtime mask over a fixed-shape 9-dim z, so every
rung reuses a single compiled executable.

Usage:
    CUDA_VISIBLE_DEVICES=<idx> XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        python -m iosp.experiments.e0_segment_ablation
"""

import dataclasses
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np

from ioc import outer as outer_opt
from ioc.inner import make_inner_solver
from iosp.model import pickplace as pp
from iosp.config import (PICK_POS, PLACE_POS, Q_START, THETA_IK_STAR, Z_TRAJOPT_STAR,
                       URDF_PATH, SRDF_PATH, MESH_DIR)


N_STEPS = 12  # matches recovery_bench.N_STEPS_IMPLICIT
FULL_STAR = jnp.concatenate([THETA_IK_STAR, Z_TRAJOPT_STAR])  # (9,): [ik(2), trajopt(7)]


def _phase_index_ranges():
    idx, i = {}, 0
    for p in pp.PHASES:
        n = len(pp.SEGMENT_FEATURES[p])
        idx[p] = list(range(pp.K_IK + i, pp.K_IK + i + n))
        i += n
    return idx


PHASE_IDX = _phase_index_ranges()  # phase -> indices INTO the 9-dim z (offset by K_IK)


def _split_trajopt(theta_trajopt):
    out, i = {}, 0
    for p in pp.PHASES:
        n = len(pp.SEGMENT_FEATURES[p])
        out[p] = theta_trajopt[i : i + n]
        i += n
    return out


def mask_for(free_phases, free_ik):
    m = np.zeros(pp.K_IK + pp.K_TRAJOPT, dtype=bool)
    if free_ik:
        m[: pp.K_IK] = True
    for p in free_phases:
        for i in PHASE_IDX[p]:
            m[i] = True
    return jnp.asarray(m)


def anchor_obstacle_to_transport(prob, theta_ik, q_start, pick_pos, place_pos,
                                  radius=0.10, offset=0.02, seed=0):
    """Place the obstacle near the ACTUAL `transport`-segment EE path, the way
    `ioc.robot.problem.RobotProblem.sample_scenes` anchors obstacles for
    `e1_identifiability` -- see that function's docstring: "the anchoring is
    what makes the weights identifiable at all."

    `recovery_bench`'s fixed `OBS_CENTER=[0.3,0,0.4]` is NOT anchored to any
    segment's real path -- MEASURED (this investigation): the closest
    approach of the straight `pick_pos`->`place_pos` line to that point is
    ~0.091m, past `RobotProblem.CLEARANCE_MARGIN=0.05m`, so `transport.
    clearance`'s soft-hinge residual sits in its flat off-region along
    essentially the whole segment -- a likely cause of the flat/null
    direction measured in `study0_segment_ablation`'s single-segment rung
    (bad theta recovery, near-zero ee_rmse).

    Procedure (mirrors `sample_scenes`): take the segment's own straight-line
    joint-space seed, FK it to the real EE path, pick the midpoint, and place
    the obstacle `radius + offset` away from it along a (seeded, reproducible)
    random direction -- `offset` near 0 keeps the obstacle surface right at
    the clearance margin's turn-on threshold rather than deep in the
    always-off (too far) or always-saturated (on-path) regime.
    """
    scene1 = pp.PickPlaceScene(
        q_start=q_start, pick_pos=pick_pos, place_pos=place_pos,
        obs_center=jnp.zeros(3, dtype=jnp.float32), obs_radius=jnp.ones(1, dtype=jnp.float32),
    )
    scenes = jax.tree.map(lambda a: a[None], scene1)
    x0, phase_scenes, _, _ = prob.seeds(scenes, theta_ik)
    transport = prob.seg["transport"]
    tsc = jax.tree.map(lambda a: a[0], phase_scenes["transport"])
    q_seed = transport.unpack(x0["transport"][0], tsc)
    p = np.asarray(transport.ee_positions(q_seed))
    t = p.shape[0] // 2

    rng = np.random.default_rng(seed)
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    center = p[t] + direction * (radius + offset)

    closest = float(np.min(np.linalg.norm(p - center, axis=-1)) - radius)
    print(f"  [anchor] transport EE path closest-approach to obstacle surface: "
          f"{closest:.4f}m (CLEARANCE_MARGIN=0.05m, target ~{offset:.3f}m)")
    return jnp.asarray(center, dtype=jnp.float32), jnp.asarray([radius], dtype=jnp.float32)


def build_common(seed=0):
    """Everything that does NOT depend on `free_phases`/`free_ik`: the
    problem, the composed forward solver, the per-phase inner solvers
    (calibrated once at theta_star), and the synthetic ground-truth demo."""
    prob = pp.PickPlaceProblem.load(str(URDF_PATH), str(SRDF_PATH), str(MESH_DIR))
    obs_center, obs_radius = anchor_obstacle_to_transport(
        prob, THETA_IK_STAR, Q_START, PICK_POS, PLACE_POS, seed=seed)
    scene1 = pp.PickPlaceScene(
        q_start=Q_START, pick_pos=PICK_POS, place_pos=PLACE_POS,
        obs_center=obs_center, obs_radius=obs_radius,
    )
    scenes = jax.tree.map(lambda a: a[None], scene1)
    forward_solver = pp.make_composed_forward_solver(n_iters=60)
    theta_trajopt_star = jax.nn.softmax(Z_TRAJOPT_STAR)

    x0_star, phase_scenes_star, _, _ = prob.seeds(scenes, THETA_IK_STAR)
    inner_by_phase = {}
    for p in pp.PHASES:
        residual_fn, _ = prob.make_segment_inner(p, forward_solver)
        scales = prob.calibrate_segment(p, residual_fn, phase_scenes_star[p], jax.random.PRNGKey(seed))
        inner_by_phase[p] = make_inner_solver(residual_fn, scales, forward_solver=forward_solver)

    _, _, xs_gt, phase_scenes_gt = prob.solve(
        THETA_IK_STAR, _split_trajopt(theta_trajopt_star), scenes, inner_by_phase, x0_star)
    demo = prob.full_ee_path(scenes, xs_gt, phase_scenes_gt, batch_index=0)

    def unpack(z, mask):
        z_eff = jnp.where(mask, z, jax.lax.stop_gradient(FULL_STAR))
        theta_ik = z_eff[: pp.K_IK]
        return theta_ik, jax.nn.softmax(z_eff[pp.K_IK :])

    def loss(z, mask):
        theta_ik, theta_trajopt = unpack(z, mask)
        x0, phase_scenes, _, _ = prob.seeds(scenes, theta_ik)
        _, _, xs, phase_scenes2 = prob.solve(
            theta_ik, _split_trajopt(theta_trajopt), scenes, inner_by_phase, x0)
        path = prob.full_ee_path(scenes, xs, phase_scenes2, batch_index=0)
        return jnp.mean(jnp.sum((path - demo) ** 2, axis=-1))

    # ONE jit, shared by every schedule rung AND by the final ee_rmse readout
    # -- a second `jax.jit(loss)` (no grad) traces a structurally different
    # graph and pays its own full compile for no benefit, since `gf` already
    # returns the loss value alongside the gradient.
    gf = jax.jit(jax.value_and_grad(loss, argnums=0))
    return dict(prob=prob, unpack=unpack, gf=gf, theta_trajopt_star=theta_trajopt_star)


def run_ablation(common, free_phases, free_ik=False, n_steps=N_STEPS):
    mask = mask_for(free_phases, free_ik)
    gf, unpack = common["gf"], common["unpack"]
    theta_trajopt_star = common["theta_trajopt_star"]

    # z0 = FULL_STAR everywhere EXCEPT free entries, which start at 0 (matches
    # recovery_bench's z0 convention for the fitted coefficients). Frozen
    # entries never move (zero gradient via the mask), so their init must
    # already be correct -- it is, by construction.
    z0 = jnp.where(mask, jnp.zeros_like(FULL_STAR), FULL_STAR)

    t0 = time.perf_counter()
    loss0, _ = jax.block_until_ready(gf(z0, mask))
    compile_s = time.perf_counter() - t0

    def gf_masked(z):
        return gf(z, mask)

    t0 = time.perf_counter()
    z_hat, history = outer_opt.adam(gf_masked, z0, lr=0.05, n_steps=n_steps)
    steady_s = time.perf_counter() - t0

    theta_ik_hat, theta_trajopt_hat = unpack(z_hat, mask)
    free_traj_idx = jnp.asarray([i - pp.K_IK for i in range(pp.K_IK, pp.K_IK + pp.K_TRAJOPT) if mask[i]])
    free_err = float(jnp.linalg.norm(theta_trajopt_hat[free_traj_idx] - theta_trajopt_star[free_traj_idx])) \
        if free_traj_idx.shape[0] else 0.0
    ik_err = float(jnp.linalg.norm(theta_ik_hat - THETA_IK_STAR)) if free_ik else 0.0
    final_loss, _ = gf(z_hat, mask)
    ee_rmse = float(jnp.sqrt(final_loss))
    return dict(
        free_phases=free_phases, free_ik=free_ik, loss0=float(loss0), free_param_err=free_err,
        ik_err=ik_err, ee_rmse=ee_rmse, compile_s=compile_s, steady_s=steady_s,
        theta_trajopt_hat=np.asarray(theta_trajopt_hat), theta_ik_hat=np.asarray(theta_ik_hat),
    )


SCHEDULE = [
    ("transport",),
    ("transport", "approach"),
    ("transport", "approach", "place"),
    ("transport", "approach", "place", "grasp"),  # = all trajopt weights free
]


def main():
    print(f"theta_ik_star = {np.asarray(THETA_IK_STAR)}")
    print(f"theta_trajopt_star = {np.asarray(jax.nn.softmax(Z_TRAJOPT_STAR))}  "
          f"({', '.join(pp.THETA_TRAJOPT_NAMES)})")
    print()
    common = build_common()
    for free_phases in SCHEDULE:
        res = run_ablation(common, free_phases, free_ik=False)
        print(f"free={free_phases!r:55s} loss0={res['loss0']:.4f}  "
              f"free_param_err={res['free_param_err']:.4f}  ee_rmse={res['ee_rmse']:.4f}  "
              f"(compile {res['compile_s']:.1f}s, steady {res['steady_s']:.1f}s)", flush=True)

    # Final rung: everything free, INCLUDING theta_ik -- same fit
    # `recovery_bench.run()`'s `implicit` arm does, on the SAME compiled
    # executable, so the cliff (if any) between "all trajopt free" and "all
    # trajopt + theta_ik free" is visible in this one table.
    res = run_ablation(common, pp.PHASES, free_ik=True)
    print("free=ALL (+theta_ik)".ljust(55) + f" loss0={res['loss0']:.4f}  "
          f"free_param_err={res['free_param_err']:.4f}  ik_err={res['ik_err']:.4f}  "
          f"ee_rmse={res['ee_rmse']:.4f}  (compile {res['compile_s']:.1f}s, steady {res['steady_s']:.1f}s)",
          flush=True)


# ---------------------------------------------------------------------------
# Multi-demo coverage check: does adding independently-sampled, individually
# anchored demo contexts tighten recovery, or is the collinearity/
# zero-gradient degeneracy structural regardless of how much data covers it?
#
# Section 4 of THEORY.md is explicit that this is NOT guaranteed: more demos
# only help if they change the SPAN of the feature-gradient Gram matrix, not
# just its sample count -- and Section 7 found the `transport.smooth`/
# `transport.upright` collinearity holds "regardless of scene geometry across
# the demos tested".  So the honest prediction is: coverage may reduce noise/
# variance in the recovered weights, but should NOT resolve a direction that
# is collinear in every scene, because varying context doesn't change which
# features push the trajectory the same way.  This is the direct empirical
# test of that prediction, not an assumed conclusion.
#
# Each of the N_MAX demo contexts gets its OWN anchored obstacle (own jittered
# q_start/pick/place -> own transport seed path -> own obstacle placement),
# not one obstacle shared across contexts -- sharing one placement across
# jittered contexts would silently reintroduce exactly the "not anchored to
# THIS context's path" bug just fixed for the single-demo case.
#
# Compile discipline (same trick as `recovery_bench.sweep_demo_count` and
# `run_ablation` above, STACKED): batch is fixed at N_MAX, and BOTH which
# weights are free (`z_mask`) and which of the N_MAX demos are active
# (`demo_mask`) are runtime values passed into one `(z, z_mask, demo_mask)`
# jit -- one compile total for the entire (free_phases x demo_count) grid.
# ---------------------------------------------------------------------------


def anchor_obstacles_batch(prob, theta_ik, scenes, radius=0.10, offset=0.02, seed=0):
    """Batched version of `anchor_obstacle_to_transport`: ONE obstacle per
    scene, each anchored to THAT scene's own transport seed path (not one
    obstacle shared/tiled across contexts -- see module note above)."""
    m = scenes.q_start.shape[0]
    dummy = dataclasses.replace(
        scenes, obs_center=jnp.zeros((m, 3), jnp.float32), obs_radius=jnp.ones((m, 1), jnp.float32))
    x0, phase_scenes, _, _ = prob.seeds(dummy, theta_ik)
    transport = prob.seg["transport"]
    q_seed_batch = jax.vmap(transport.unpack)(x0["transport"], phase_scenes["transport"])  # (m, T, dof)
    p_batch = np.asarray(jax.vmap(transport.ee_positions)(q_seed_batch))  # (m, T, 3)
    t = p_batch.shape[1] // 2
    p_mid = p_batch[:, t, :]

    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(m, 3))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    centers = p_mid + directions * (radius + offset)

    closest = np.min(np.linalg.norm(p_batch - centers[:, None, :], axis=-1), axis=-1) - radius
    print(f"  [anchor-batch] per-scene closest-approach: {np.round(closest, 4)} "
          f"(CLEARANCE_MARGIN=0.05m, target ~{offset:.3f}m)")
    return jnp.asarray(centers, dtype=jnp.float32), jnp.asarray(np.full((m, 1), radius), dtype=jnp.float32)


def sample_scenes_anchored(prob, theta_ik, n, seed=0, jitter_pos=0.03, jitter_q=0.05):
    """`recovery_bench.sample_pickplace_scenes`'s jitter, but with EACH
    context's obstacle individually anchored to ITS OWN transport path
    (`sample_pickplace_scenes` instead tiles one fixed, unanchored obstacle
    across every context)."""
    rng = np.random.default_rng(seed)
    starts, picks, places = [], [], []
    for _ in range(n):
        starts.append(np.asarray(Q_START) + rng.normal(scale=jitter_q, size=Q_START.shape[0]))
        picks.append(np.asarray(PICK_POS) + rng.normal(scale=jitter_pos, size=3))
        places.append(np.asarray(PLACE_POS) + rng.normal(scale=jitter_pos, size=3))
    scenes = pp.PickPlaceScene(
        q_start=jnp.asarray(np.stack(starts), dtype=jnp.float32),
        pick_pos=jnp.asarray(np.stack(picks), dtype=jnp.float32),
        place_pos=jnp.asarray(np.stack(places), dtype=jnp.float32),
        obs_center=jnp.zeros((n, 3), dtype=jnp.float32),
        obs_radius=jnp.ones((n, 1), dtype=jnp.float32),
    )
    obs_center, obs_radius = anchor_obstacles_batch(prob, theta_ik, scenes, seed=seed + 1)
    return dataclasses.replace(scenes, obs_center=obs_center, obs_radius=obs_radius)


def build_common_multidemo(n_max=5, seed=0):
    prob = pp.PickPlaceProblem.load(str(URDF_PATH), str(SRDF_PATH), str(MESH_DIR))
    scenes = sample_scenes_anchored(prob, THETA_IK_STAR, n_max, seed=seed)
    forward_solver = pp.make_composed_forward_solver(n_iters=60)
    theta_trajopt_star = jax.nn.softmax(Z_TRAJOPT_STAR)

    x0_star, phase_scenes_star, _, _ = prob.seeds(scenes, THETA_IK_STAR)
    inner_by_phase = {}
    for p in pp.PHASES:
        residual_fn, _ = prob.make_segment_inner(p, forward_solver)
        scales = prob.calibrate_segment(p, residual_fn, phase_scenes_star[p], jax.random.PRNGKey(seed))
        inner_by_phase[p] = make_inner_solver(residual_fn, scales, forward_solver=forward_solver)

    _, _, xs_gt, phase_scenes_gt = prob.solve(
        THETA_IK_STAR, _split_trajopt(theta_trajopt_star), scenes, inner_by_phase, x0_star)
    demo_paths = prob.full_ee_paths(scenes, xs_gt, phase_scenes_gt)

    def unpack(z, z_mask):
        z_eff = jnp.where(z_mask, z, jax.lax.stop_gradient(FULL_STAR))
        theta_ik = z_eff[: pp.K_IK]
        return theta_ik, jax.nn.softmax(z_eff[pp.K_IK :])

    def loss(z, z_mask, demo_mask):
        theta_ik, theta_trajopt = unpack(z, z_mask)
        x0, phase_scenes, _, _ = prob.seeds(scenes, theta_ik)
        _, _, xs, phase_scenes2 = prob.solve(
            theta_ik, _split_trajopt(theta_trajopt), scenes, inner_by_phase, x0)
        paths = prob.full_ee_paths(scenes, xs, phase_scenes2)
        per_demo_mse = jnp.mean(jnp.sum((paths - demo_paths) ** 2, axis=-1), axis=-1)  # (n_max,)
        return jnp.sum(per_demo_mse * demo_mask) / jnp.sum(demo_mask)

    gf = jax.jit(jax.value_and_grad(loss, argnums=0))
    return dict(prob=prob, unpack=unpack, gf=gf, theta_trajopt_star=theta_trajopt_star, n_max=n_max)


def run_ablation_multidemo(common, free_phases, free_ik, n_active, n_steps=N_STEPS):
    z_mask = mask_for(free_phases, free_ik)
    n_max = common["n_max"]
    demo_mask = jnp.asarray(np.array([1.0] * n_active + [0.0] * (n_max - n_active), dtype=np.float32))
    gf, unpack = common["gf"], common["unpack"]
    theta_trajopt_star = common["theta_trajopt_star"]

    z0 = jnp.where(z_mask, jnp.zeros_like(FULL_STAR), FULL_STAR)

    t0 = time.perf_counter()
    loss0, _ = jax.block_until_ready(gf(z0, z_mask, demo_mask))
    compile_s = time.perf_counter() - t0

    def gf_masked(z):
        return gf(z, z_mask, demo_mask)

    t0 = time.perf_counter()
    z_hat, _ = outer_opt.adam(gf_masked, z0, lr=0.05, n_steps=n_steps)
    steady_s = time.perf_counter() - t0

    theta_ik_hat, theta_trajopt_hat = unpack(z_hat, z_mask)
    free_traj_idx = jnp.asarray([i - pp.K_IK for i in range(pp.K_IK, pp.K_IK + pp.K_TRAJOPT) if z_mask[i]])
    free_err = float(jnp.linalg.norm(theta_trajopt_hat[free_traj_idx] - theta_trajopt_star[free_traj_idx])) \
        if free_traj_idx.shape[0] else 0.0
    ik_err = float(jnp.linalg.norm(theta_ik_hat - THETA_IK_STAR)) if free_ik else 0.0
    final_loss, _ = gf(z_hat, z_mask, demo_mask)
    ee_rmse = float(jnp.sqrt(final_loss))
    return dict(
        free_phases=free_phases, free_ik=free_ik, n_active=n_active, loss0=float(loss0),
        free_param_err=free_err, ik_err=ik_err, ee_rmse=ee_rmse,
        compile_s=compile_s, steady_s=steady_s,
    )


def main_multidemo(n_max=5, demo_counts=(1, 3, 5)):
    print(f"theta_ik_star = {np.asarray(THETA_IK_STAR)}")
    print(f"theta_trajopt_star = {np.asarray(jax.nn.softmax(Z_TRAJOPT_STAR))}  "
          f"({', '.join(pp.THETA_TRAJOPT_NAMES)})")
    print()
    common = build_common_multidemo(n_max=n_max)
    print()
    rungs = [(("transport",), False), (pp.PHASES, True)]
    for free_phases, free_ik in rungs:
        for n_active in demo_counts:
            res = run_ablation_multidemo(common, free_phases, free_ik, n_active)
            label = f"N={n_active} free={free_phases!r}" + ("+ik" if free_ik else "")
            print(f"{label:45s} free_param_err={res['free_param_err']:.4f}  "
                  f"ik_err={res['ik_err']:.4f}  ee_rmse={res['ee_rmse']:.4f}  "
                  f"(compile {res['compile_s']:.1f}s, steady {res['steady_s']:.1f}s)", flush=True)
        print()


if __name__ == "__main__":
    main()
