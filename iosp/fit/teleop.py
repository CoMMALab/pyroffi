"""Path A on HUMAN demonstrations: the same bilevel forward map as
`iosp.fit.parametric`, fitted against teleoperated pick-and-place instead of
against a rollout of the model itself.

What changes, and why each change is forced
-------------------------------------------
`build_parametric` answers "can the fit recover the weights that generated this
demonstration", and it can only ask that because the demonstration IS a rollout
of the forward model at `THETA_*_STAR`.  There is no such vector here.  A human
at a GELLO leader is not running this planner, so:

  * `theta_star`/`u_star` are None, and every parameter-space metric
    (`param_err`, `captured_frac`, `z_err_*`) is undefined and NOT reported.
    `iosp.fit.procedure.run_procedure` already branches on `theta_star is None`.
    What remains is the behavioural criterion, which is the one that was worth
    reporting anyway: does the fitted cost REPRODUCE held-out demonstrations.
  * The loss no longer has a zero.  Model misspecification -- everything the
    human did that this four-phase planner cannot express -- is a floor under
    `fit_rmse` that no amount of optimization removes, so the number that means
    something is the DROP from `fit_rmse_init`, and the gap between `fit_rmse`
    and `gen_rmse`.
  * Fit and held-out are different EPISODES, not scene A and a jittered scene
    B.  Each episode carries its own randomised cube and bucket, so the demo set
    already spans the "sufficiently different environments" that Cao, Cohen &
    Szpruch make a precondition of identifiability (see `iosp.model.scenes`).

The scenes are the episodes' own
--------------------------------
The recorded scene is used verbatim: `pick_pos` is where THAT episode's cube
actually was, `place_pos` the drop point of THAT episode's bucket, `q_start` the
demo's own first waypoint.  Nothing is re-nominalised onto `iosp.config`'s
canonical task, which is the whole point -- a forward rollout on a scene the
demonstration did not happen in is not a reconstruction of anything.

Two consequences of the recorded scenes to read the results with:

  * `obs_center`/`obs_radius` are CONSTANT across every episode -- the recording
    scene has no obstacle and emits a fixed placeholder held clear of the
    workspace (see `sim_teleop/pickplace/scene.py::iosp_scene_fields`).  The
    `clearance` weight is therefore unidentifiable BY CONSTRUCTION here, and is
    expected to land in the Gram's null space.  If it lands in `U_r` instead,
    something is wrong with the spectrum, not with the demonstrator.
  * The robot is an FR3, not a Panda -- see `iosp.model.fr3`.

The standoff prior
------------------
`u = 0` has to be a sensible planner, because it is both the initialization AND
the value stage 4's refit PINS the null space at -- a `theta_ik` the demos leave
unidentified stays wherever `u = 0` puts it.  `iosp.fit.params`' implicit prior
of `z = 0` is not sensible here: the EE frame is `fr3_hand`, the flange, which
sits `TCP_OFFSET_M` = 0.1034 m behind the fingertips, so a zero standoff asks IK
to put the flange inside the cube.

So the standoff prior is MEASURED, on the fit episodes only, as the median
height of the hand above the target at the skeleton rows the exporter pins the
grasp and the release to (`pickplace.SKELETON_PICK` / `SKELETON_PLACE`).  On the
first recorded session that is 0.105 m at the grasp -- the TCP offset, to 2 mm,
across all ten episodes, which is an independent confirmation that the gripper
channel put the grasp on the right row -- and 0.24 m at the release, because a
human drops the cube into the bucket from above rather than lowering it to the
floor.  Held-out episodes are excluded from the median for the same reason they
are excluded from the feature-scale calibration.

The trajopt logits keep a flat prior (uniform softmax), which is genuinely
uninformative.  `z = z_prior + Z_SCALE * u` leaves `u` dimensionless exactly as
in `iosp.fit.params`.

What the standoff CANNOT absorb: `theta_ik` offsets the IK target along +z only,
and at the release the hand is also ~0.07 m LATERALLY off the bucket centre (the
bucket's inner radius is 0.065 m, so the operator lets go over the rim rather
than the axis).  That lateral term is misspecification and lands in the loss
floor; it is one waypoint of the 23, so it bounds `fit RMSE` from below at
roughly 0.07/sqrt(23) ~ 0.015 m in EE terms before anything else contributes.
"""

import dataclasses
import os
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

from ioc import identifiability as ident
from iosp.fit.params import z_scale
from iosp.fit.parametric import _build_inner, screen_stationarity
from iosp.model import fr3, pickplace as pp
from iosp.model.pickplace import split_trajopt as _split_trajopt

# `fr3_hand` (the EE frame) to `fr3_hand_tcp`, from the URDF's fixed joint.
TCP_OFFSET_M = 0.1034

DEFAULT_TELEOP_ROOT = pathlib.Path(
    os.environ.get("IOSP_TELEOP_ROOT",
                   pathlib.Path(__file__).resolve().parents[3] / "sim_teleop")
)
DEFAULT_DEMO_DIR = DEFAULT_TELEOP_ROOT / "data" / "demos"


def _import_exporter(teleop_root=DEFAULT_TELEOP_ROOT):
    """`sim_teleop.pickplace.iosp_export`, imported from the sibling checkout.

    Imported rather than vendored: it reads `N_FULL`/`PHASE_SPAN` from this
    package, so the episode->waypoint collapse and the forward model's phase
    layout cannot drift apart.  A copy here would be a second thing to keep
    correct, and the first segment-length change would silently break it.
    """
    root = str(pathlib.Path(teleop_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from pickplace import iosp_export
    except ImportError as e:
        raise ImportError(
            f"cannot import the teleop exporter from {root!r}. Point "
            "IOSP_TELEOP_ROOT at the sim_teleop checkout."
        ) from e
    return iosp_export


def find_episodes(demo_dir=DEFAULT_DEMO_DIR):
    """Every episode directory holding both files, sorted by name (= by time)."""
    demo_dir = pathlib.Path(demo_dir)
    eps = sorted(d for d in demo_dir.iterdir()
                 if (d / "state.jsonl").exists() and (d / "factors.json").exists())
    if not eps:
        raise FileNotFoundError(f"no episodes with state.jsonl + factors.json in {demo_dir}")
    return eps


def load_demos(demo_dir=DEFAULT_DEMO_DIR, teleop_root=DEFAULT_TELEOP_ROOT,
               prob=None, anchor_grasp=False, max_episodes=None):
    """-> (names, (B, N_FULL, dof) waypoints, batched `PickPlaceScene`).

    `anchor_grasp=True` fills `pick_wxyz`/`grasp_ref` AND `place_wxyz`/
    `place_ref` from each episode's own configuration at the skeleton grasp and
    release rows, which requires `prob` for the FK.
    See `PickPlaceScene.pick_wxyz` for the measured effect and for what it does
    to the claim -- the grasp pose stops being predicted and becomes an input.
    """
    ex = _import_exporter(teleop_root)
    eps = find_episodes(demo_dir)
    if max_episodes is not None:
        eps = eps[: int(max_episodes)]
    paths, fields = ex.build_demo_batch(
        [(d / "state.jsonl", d / "factors.json") for d in eps])
    demo_q = jnp.asarray(paths, dtype=jnp.float32)
    scenes = pp.PickPlaceScene(
        **{k: jnp.asarray(v, dtype=jnp.float32) for k, v in fields.items()})

    if anchor_grasp:
        if prob is None:
            raise ValueError("anchor_grasp=True needs `prob` for the FK")
        # Both events, from the rows the skeleton pins them to.  FK over the
        # whole batch is one batched call, so reading four anchors costs
        # nothing measurable next to a single IK solve.
        q_grasp = demo_q[:, pp.SKELETON_PICK[0]]
        q_place = demo_q[:, pp.SKELETON_PLACE[0]]
        fk = lambda q: prob.base.robot.forward_kinematics(q)[:, prob.ee_index, :4]
        scenes = dataclasses.replace(
            scenes,
            pick_wxyz=fk(q_grasp).astype(jnp.float32),
            grasp_ref=q_grasp.astype(jnp.float32),
            place_wxyz=fk(q_place).astype(jnp.float32),
            place_ref=q_place.astype(jnp.float32))
    return [d.name for d in eps], demo_q, scenes


def z_prior(K, n_ik, standoffs=None):
    """`u = 0` in `z` coordinates: `theta_ik` at `standoffs`, flat logits.

    `standoffs=None` falls back to the TCP offset on the two standoffs and zero
    in-plane offset, which is right for the grasp and wrong for the release --
    prefer `measure_standoffs`, which reads all four off the demonstrations.
    """
    p = np.zeros(K, dtype=np.float32)
    if standoffs is None:
        p[:2] = TCP_OFFSET_M
    else:
        p[:n_ik] = np.asarray(standoffs)
    return jnp.asarray(p)


def measure_standoffs(prob, demo_q, scenes, idx):
    """`theta_ik`'s prior, in metres, from episodes `idx` -- all four entries.

    Each is the median over those episodes of exactly the quantity the matching
    coordinate parameterises, read off the demonstrations instead of guessed:

      grasp.standoff     height of the EE frame above the cube at the pinned
                         grasp row -- comes out at the hand-to-TCP offset, which
                         is an independent check that the gripper channel put
                         the grasp on the right row
      place.standoff     the same above the bucket at the release row
      place.radial       in-plane displacement of the release point along the
                         base->bucket direction (negative = short of centre)
      place.tangential   the same along its left-hand normal

    Median, not mean: one fumbled approach should not move the initialization.
    """
    idx = np.asarray(idx)
    ee = np.asarray(jax.vmap(prob.ee_positions)(demo_q))[idx]
    pick = np.asarray(scenes.pick_pos)[idx]
    place = np.asarray(scenes.place_pos)[idx]
    r_pick, r_place = list(pp.SKELETON_PICK), list(pp.SKELETON_PLACE)

    grasp_z = np.median(ee[:, r_pick, 2] - pick[:, 2:3])
    place_z = np.median(ee[:, r_place, 2] - place[:, 2:3])

    # In-plane, in the scene's own base->bucket frame; see `pp._place_frame`.
    radial, tangential = pp._place_frame(jnp.asarray(place))
    d = ee[:, r_place[0], :2] - place[:, :2]
    rad = np.median((d * np.asarray(radial)[:, :2]).sum(-1))
    tan = np.median((d * np.asarray(tangential)[:, :2]).sum(-1))
    return np.array([grasp_z, place_z, rad, tan], dtype=np.float32)


def build_teleop(demo_dir=DEFAULT_DEMO_DIR, teleop_root=DEFAULT_TELEOP_ROOT,
                 n_fit=None, seed=0, n_iters=60, n_restarts=1, space="joint"):
    """The `built` dict `iosp.fit.procedure.run_procedure` consumes.

    `space` defaults to "joint", not "ee" as in `build_parametric`: the
    demonstration IS a joint-space path (`q_d`, the commanded configuration),
    and the 7-DOF arm's self-motion manifold is invisible to an EE loss -- a
    fitted rollout can match the demonstrated EE path to the millimetre through
    a completely different elbow.  The EE criterion is still computed and
    reported, from the same rollout, via `ee_*`.

    The first `n_fit` episodes (chronological) are the fit set and the rest are
    held out.  Chronological, not random: the split is then reproducible without
    carrying a seed, and it is the honest one for a human demonstrator whose
    technique drifts over a session -- a random split leaks late-session
    technique into the training set.
    """
    if space not in ("ee", "joint"):
        raise ValueError(f"space must be 'ee' or 'joint', got {space!r}")

    names, demo_q, scenes = load_demos(demo_dir, teleop_root)
    B = len(names)
    n_fit = B - max(1, B // 4) if n_fit is None else int(n_fit)
    if not 0 < n_fit < B:
        raise ValueError(f"n_fit must be in (0, {B}); got {n_fit}")
    fit_idx = np.arange(n_fit)
    gen_idx = np.arange(n_fit, B)

    urdf, srdf, mesh_dir, ee_link = fr3.paths()
    prob = pp.PickPlaceProblem.load(urdf, srdf, mesh_dir, ee_link=ee_link)
    forward_solver = pp.make_composed_forward_solver(n_iters=n_iters)

    K = pp.K_IK + pp.K_TRAJOPT
    standoffs = measure_standoffs(prob, demo_q, scenes, fit_idx)
    print(f"  [teleop] standoff prior from the {n_fit} fit episodes: "
          f"grasp {standoffs[0]:.4f} m, place {standoffs[1]:.4f} m "
          f"(hand-to-TCP is {TCP_OFFSET_M:.4f} m)", flush=True)
    S, P = z_scale(K, pp.K_IK), z_prior(K, pp.K_IK, standoffs)
    z_of = lambda u: P + S * u

    # Feature scales are calibrated on the FIT episodes only, for the same
    # reason `build_parametric` calibrates on scene A only: a scale fitted on
    # the held-out scenes lets them re-normalise the very features being tested.
    fit_scenes = jax.tree.map(lambda a: a[fit_idx], scenes)
    inner, _ = _build_inner(prob, fit_scenes, z_of(jnp.zeros(K))[: pp.K_IK],
                            forward_solver, seed, n_restarts=n_restarts)

    def _rollout(u):
        z = z_of(u)
        theta_ik, z_traj = z[: pp.K_IK], z[pp.K_IK:]
        x0, _, _, _ = prob.seeds(scenes, theta_ik)
        _, _, xs, ps = prob.solve(theta_ik, _split_trajopt(jax.nn.softmax(z_traj)),
                                  scenes, inner, x0)
        return xs, ps

    def ee_paths(u):
        xs, ps = _rollout(u)
        return prob.full_ee_paths(scenes, xs, ps)

    def joint_paths(u):
        xs, ps = _rollout(u)
        return prob.full_joint_paths(scenes, xs, ps)

    paths = joint_paths if space == "joint" else ee_paths
    paths_j = jax.jit(paths)
    ee_paths_j = jax.jit(ee_paths)

    demo = demo_q if space == "joint" else jax.vmap(prob.ee_positions)(demo_q)
    ee_demo = jax.vmap(prob.ee_positions)(demo_q)

    screen_stationarity(prob, fit_scenes, inner, z_of(jnp.zeros(K))[: pp.K_IK],
                        _split_trajopt(jax.nn.softmax(jnp.zeros(pp.K_TRAJOPT))),
                        "teleop (path A, human demos)")

    def loss_a(u):
        return jnp.mean(jnp.sum((paths(u)[fit_idx] - demo[fit_idx]) ** 2, axis=-1))

    def _rmse(P_, D, idx):
        return float(jnp.sqrt(jnp.mean(jnp.sum((P_[idx] - D[idx]) ** 2, axis=-1))))

    def theta_of(u):
        z = np.asarray(z_of(u))
        return np.concatenate([z[: pp.K_IK], np.asarray(jax.nn.softmax(z[pp.K_IK:]))])

    return dict(
        gf=jax.jit(jax.value_and_grad(loss_a)),
        # Value-only loss: FD/CMA-ES need the loss VALUE, never its gradient.
        # Routing them through `gf(u)[0]` made every probe also build the
        # implicit adjoint's dense Hessian (see ioc.inner), which is what drove
        # the FD stage to >108 GB host RAM and OOM.  `loss` gives byte-identical
        # values with none of that curvature work.
        loss=jax.jit(loss_a),
        paths_fn=paths_j, demo_paths=demo, space=space,
        ee_paths_fn=ee_paths_j, ee_demo_paths=ee_demo,
        jac_fn=ident.make_jac_fn(lambda u: paths(u)[fit_idx]),
        rmse_a=lambda u: _rmse(paths_j(u), demo, fit_idx),
        rmse_b=lambda u: _rmse(paths_j(u), demo, gen_idx),
        ee_rmse_a=lambda u: _rmse(ee_paths_j(u), ee_demo, fit_idx),
        ee_rmse_b=lambda u: _rmse(ee_paths_j(u), ee_demo, gen_idx),
        K=K, n_ik=pp.K_IK, theta_of=theta_of, standoff_prior=standoffs,
        # No ground-truth cost exists for a human demonstrator: `run_procedure`
        # skips every parameter-space metric on `theta_star is None`.
        theta_star=None, u_star=None,
        names=list(pp.THETA_IK_NAMES) + list(pp.THETA_TRAJOPT_NAMES),
        episodes=names, n_fit=n_fit, fit_idx=fit_idx, gen_idx=gen_idx,
        scenes=scenes, demo_q=demo_q, prob=prob,
    )
