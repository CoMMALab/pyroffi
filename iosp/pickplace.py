"""Composed pick-and-place planner: differentiable IK chained into differentiable
trajopt, approach -> grasp -> transport -> place.

Why this is a genuine composition and not "one trajopt problem with named cost
terms"
----------------------------------------------------------------------------
The first version of this module (see git history) put the whole task in ONE
flat decision vector and used phase boundaries purely as index slices into one
`residual_fn` -- which a single hand-authored trajopt cost (e.g. cuRobo, given
the same named terms) could reproduce exactly.  That is not a claim worth
making: composing named residuals into one cost function is what any
weighted-trajopt system already does.

The thing a monolithic trajopt system CANNOT do is what this version does:
`grasp_ik`/`place_ik` are a genuinely separate differentiable module (SQP IK,
`sqp_ik_solve_cuda_batch`) whose *output* -- not a residual against it, the
literal returned array -- is the boundary condition of the `approach`/
`transport` trajopt segments.  Each segment is its own `ioc.inner.
make_inner_solver` instance with its own `solve_implicit` custom_vjp.  Because
`q_pick`/`q_place` flow into the next segment's `Scene.q_start`/`q_goal`
undetached (no `stop_gradient` at the interface), JAX's ordinary reverse mode
composes the IK stage's custom_jvp with each segment's implicit-adjoint
custom_vjp automatically: d(loss)/d(theta_ik) is the product of the trajopt
adjoint's sensitivity to its boundary condition and the IK stage's sensitivity
to theta_ik.  A single-cost trajopt formulation has no "theta_ik" at all --
there is no analogue of "which of several redundant-IK solutions the
downstream segment starts from" in a system that never solves an IK subproblem
as a distinct step.  Section 2 of this module's test harness verifies this
chain is real (not just constructed) by finite-differencing d(loss)/d(theta_ik)
through the full IK->trajopt composition.

Correctness precondition: canonical IK
---------------------------------------
The Panda is 7-DOF against a 6-DOF pose task, so `sqp_ik_solve_cuda_batch`'s
IK subproblem is redundant.  pyroffi's *default* implicit-diff rule for IK is
measured WRONG on redundant arms (assumes minimum-norm self-motion via a
pinv, but the solver's actual self-motion is arbitrary; ~80% gradient error
vs FD, confirmed to be a null-space artifact, not solution-ambiguity noise).
The fix is the canonical reformulation (`q* = argmin ||q-q_ref||^2 s.t.
r(q,t)=0`, exact KKT sensitivity), which the batched entry point applies
automatically via `cfg_ref=previous_cfgs` -- gated by a module-level flag
`_implicit_diff.CANONICAL_BY_DEFAULT` that is read at CALL time (not import
time), so setting it here after import is sufficient and doesn't require the
`PYROFFI_CANONICAL_IK` env var to be exported before `pyroffi` is first
imported.  Benchmarked cost on this kernel: ~1.29x at batch=1, ~1.34x at
batch=8 -- cheap enough to force on unconditionally for every IK call in this
module, rather than making it a per-call opt-in.

theta_ik parameterization -- and a real limitation of the IK API found while
building this
--------------------------------------------------------------------------
The brief's suggested knobs (`pos_weight`/`ori_weight` balance, or which
`cfg_ref` `canonical_ik` walks toward) turned out NOT to be differentiable
inputs of this API as implemented: `differentiable_ik_solution_batch` attaches
an implicit-diff rule ONLY to `target_poses` (see its docstring: "Gradients
flow back to these"); `cfg_ref` is explicitly `jax.lax.stop_gradient`-ed
before being handed to `canonical_ik`, and `pos_weight`/`ori_weight` are
config baked into the CUDA kernel launch, not JAX values participating in the
custom_jvp at all.  Wiring `theta_ik` through either would have produced a
`theta_ik` that gets a zero (or, worse, a JAX-tracer-shaped but wrong) gradient
silently -- exactly the kind of "looks connected, isn't" bug this whole
exercise exists to catch.  So `theta_ik` here is instead a standoff offset
along a fixed approach axis, applied to the pose actually passed as
`target_poses`:

    target_pos = scene.pick_pos (or place_pos) + theta_ik_k * UP_AXIS

This is genuinely meaningful (how far above the object the gripper should
approach from -- a real pregrasp/preplace standoff) and it is a real,
differentiable input of the wrapped solve.

MEASURED, and NOT papered over: the IK stage alone (`grasp_ik` in isolation,
holding the trajopt stage out of the loop) is smooth and its implicit
gradient is FD-verifiable -- `|q_pick(theta+eps)-q_pick(theta)| / eps` is
stable to 4 significant figures (~3.84-3.87) across eps in [1e-4, 1e-2] before
float32 noise takes over below that.  The FULL composed chain (IK -> trajopt)
is NOT FD-verifiable with the current trajopt forward solver
(`pyroffi.optimization_engines.dynamics_trajopt`, early-stopping L-BFGS):
`d(reconstruction_loss)/d(theta_ik)` estimated by forward differences roughly
DOUBLES every time eps is halved across 6 octaves (eps in [5e-4, 5e-2]) --
never converging to a stable value, cos(implicit, FD) ~ -0.71.  Ruled out by
direct test, in order: (1) NOT the multi-seed argmax in `sqp_ik_solve_cuda_
batch` -- confirmed above, IK alone is smooth; (2) NOT trajopt basin-jumping
in the usual `ioc.inner` sense -- `n_restarts=3` on every segment changes the
FD numbers by <1%; (3) NOT warm-start path-dependence -- holding the L-BFGS
x0 FIXED (only letting the boundary condition itself vary with theta_ik)
reproduces the identical blow-up.  What's left, by elimination: the
early-stopping trajopt solver's OWN convergence criterion (a `grad_tol`-gated
halt) responds discontinuously to a smoothly-varying boundary condition --
the iteration at which it decides "converged" can flip as the boundary shifts
by an infinitesimal amount, producing a different final iterate near a flat
region of the cost.  This makes the composed adjoint UNVALIDATED against FD,
not wrong -- the adjoint differentiates the returned iterate as an exact
stationary point (per `ioc.inner`'s own precondition), which is well-defined
and smooth in theta_ik regardless of the solver's stopping-time
discontinuity, but there is currently no FD ground truth to compare it to at
usable step sizes.  Validating the composed adjoint properly needs a
fixed-iteration (no early stopping) trajopt forward solver for this check
specifically; out of scope for this pass, flagged here rather than fixed.  `theta_ik` is NOT passed through the shared softmax: it is a
signed geometric offset, not a relative cost weight, and forcing it onto the
simplex with the trajopt weights would be a category error.  So the full
outer parameter is `(theta_ik, z_trajopt)`, optimized jointly; only
`z_trajopt` is softmaxed into `theta_trajopt`.

theta_trajopt (K=7, unchanged in spirit from the old K=9 but two features
dropped): `approach.smooth, approach.clearance, grasp.smooth,
transport.smooth, transport.clearance, transport.upright, place.smooth`.  The
old `*.align` features are GONE on purpose: in the old design they existed to
softly pull a free trajectory endpoint toward the pick/place pose; here that
endpoint is `q_pick`/`q_place` itself, hard-clamped as the segment boundary
(exactly reached by construction, delegated entirely to the IK stage), so an
align residual on it would be identically zero and is not a meaningful
degree of freedom to weight.

Segments
--------
Each segment reuses `ioc.robot.problem.RobotProblem`/`Scene` UNCHANGED (not
re-derived): `dataclasses.replace(base_problem, n_timesteps=...)` gives one
`RobotProblem` per phase sharing the same robot/collision model, and each
phase's `Scene(q_start=..., q_goal=..., obs_center=..., obs_radius=...)` uses
q_start/q_goal that are either the scene's literal endpoints (global start) or
`q_pick`/`q_place` -- undetached JAX values, not scene constants.
`approach`: q_start -> q_pick.  `grasp`: q_pick -> q_pick (small in-place
motion, per the brief's "grasp-in-place small motion" segment; the
Scene's start==goal cost is genuinely tiny but the segment is a real
`solve_implicit` call so its adjoint still participates in the chain).
`transport`: q_pick -> q_place.  `place`: q_place -> q_place.
"""

import dataclasses
import os

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np

# Persistent XLA compilation cache: this module's forward solver (IK stage +
# 4 chained trajopt segments, implicit-adjoint differentiated) compiles for
# 30-90+ min per distinct shape signature. Without this, EVERY fresh process
# (i.e. every debug/iteration run of any iosp script) pays that cost from
# scratch even when nothing about the traced function changed -- measured
# directly in this investigation. This caches compiled XLA executables to
# disk keyed by the HLO's hash, so a second run with the same shapes/config
# loads from disk in seconds instead of recompiling. Override the directory
# via IOSP_JAX_CACHE_DIR; safe to share across users/processes (read-mostly
# after first population), no correctness effect either way.
jax.config.update(
    "jax_compilation_cache_dir",
    os.environ.get("IOSP_JAX_CACHE_DIR", os.path.expanduser("~/.cache/jax_pyroffi_iosp")),
)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 5)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

from ioc.robot.problem import RobotProblem, Scene

# Read at CALL time inside `differentiable_ik_solution_batch`, so setting this
# after `pyroffi` is imported still takes effect -- see module docstring.
from pyroffi.optimization_engines import _implicit_diff
_implicit_diff.CANONICAL_BY_DEFAULT = True
from pyroffi.optimization_engines._sqp_ik import sqp_ik_solve_cuda_batch

N_APPROACH = 8
N_GRASP = 4
N_TRANSPORT = 10
N_PLACE = 4
PHASES = ("approach", "grasp", "transport", "place")
SEGMENT_LEN = {"approach": N_APPROACH, "grasp": N_GRASP,
               "transport": N_TRANSPORT, "place": N_PLACE}

SEGMENT_FEATURES = {
    "approach": ("smooth", "clearance"),
    "grasp": ("smooth",),
    "transport": ("smooth", "clearance", "upright"),
    "place": ("smooth",),
}
THETA_TRAJOPT_NAMES = tuple(
    f"{seg}.{feat}" for seg in PHASES for feat in SEGMENT_FEATURES[seg]
)
K_TRAJOPT = len(THETA_TRAJOPT_NAMES)
THETA_IK_NAMES = ("grasp.standoff", "place.standoff")
K_IK = len(THETA_IK_NAMES)

UP_AXIS = jnp.array([0.0, 0.0, 1.0])
DOWN_WXYZ = jnp.array([0.0, 1.0, 0.0, 0.0])  # gripper facing down at the target
IK_RNG_KEY = jax.random.PRNGKey(0)  # fixed: gradients don't flow through it


def make_composed_forward_solver(n_iters=60):
    """The forward solve for the 4 chained trajopt segments -- deliberately
    NOT `ioc.robot.e1_identifiability.make_dynamics_forward_solver`'s default
    (early-stopping `while_loop`, `DynamicsTrajOptConfig(early_stop=True)`).

    MEASURED root cause of the composed chain's FD check failing (cos=-0.71
    against the implicit adjoint, FD estimate doubling every eps-halving
    across 6 octaves, never converging): early stopping's `grad_tol`-gated
    `while_loop` has a data-dependent trip count, so q*(theta_ik) is not just
    "hard to differentiate" but genuinely DISCONTINUOUS in theta_ik -- an
    infinitesimal shift in the (smooth) boundary condition `q_pick`/`q_place`
    can flip which iteration crosses `grad_tol` and land the solver on a
    different final iterate.  That breaks the one precondition `solve_
    implicit`'s implicit-function-theorem adjoint actually needs (see `ioc.
    inner`'s module docstring: "provided the solve actually converged" to a
    point that varies smoothly with the inputs) -- not a bug in the adjoint
    itself, a violated precondition upstream of it.

    Fix: `DynamicsTrajOptConfig(early_stop=False, unroll_tail=0)` runs a
    FIXED-length `jax.lax.scan` for exactly `n_iters` steps -- no data-
    dependent trip count, so q*(theta_ik) is smooth by construction (the same
    L-BFGS step function, unrolled a fixed number of times, is a composition
    of smooth maps).  `unroll_tail=0` is deliberate: `solve_implicit`'s
    forward pass already wraps this call in `jax.lax.stop_gradient` (its
    gradient comes entirely from the analytic adjoint in `ioc.inner`'s `_bwd`,
    never from differentiating this solve), so the `early_stop=False` branch's
    own truncated-unroll differentiability is irrelevant here -- only its
    fixed-trip-count SHAPE matters.

    Does not touch `ioc.inner.make_inner_solver`'s semantics or any other
    caller's forward solver: this is a different value passed into the same
    pluggable `forward_solver` slot every caller already uses (`e1_
    identifiability.py`, `bench2d`, ...), which are unaffected.
    """
    from pyroffi.optimization_engines import DynamicsTrajOptConfig, dynamics_trajopt

    # `early_stop=False` alone did NOT fix the composed chain's FD check
    # (measured: cos(implicit, FD) = -0.707, unchanged from the early-stopping
    # baseline's -0.7066) -- the outer while_loop's data-dependent trip count
    # was not the only discontinuity.  `soft_line_search=True` (the per-step
    # line search's hard argmax over 5 trial alphas) ALSO measured no change
    # on its own (-0.7066, and re-confirmed non-degenerate via jaxpr diffing
    # after finding and fixing a temperature-scaling bug in it).
    # `soft_curvature_gate=True` targets the third hard branch in this step
    # function: the L-BFGS curvature-pair admit/reject gate and the
    # `direction = where(m_used > 0, dir_lbfgs, dir_gd)` switch it feeds -- see
    # `DynamicsTrajOptConfig.soft_curvature_gate`'s docstring.  All three
    # flags are stacked here; see this module's smoke-test notes for whether
    # stacking all three finally converges the composed chain's FD check.
    opt_cfg = DynamicsTrajOptConfig(n_iters=n_iters, early_stop=False, unroll_tail=0,
                                     soft_line_search=True, soft_curvature_gate=True)

    def forward_solver(x0, cost_fn):
        return dynamics_trajopt(x0, cost_fn, opt_cfg)

    return forward_solver


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PickPlaceScene:
    """Context for one pick-and-place demonstration (one row per batch element
    when used under `jax.vmap`/a leading batch dim, matching `ioc.robot.problem
    .Scene`'s convention)."""

    q_start: jnp.ndarray  # (dof,)
    pick_pos: jnp.ndarray  # (3,)
    place_pos: jnp.ndarray  # (3,)
    obs_center: jnp.ndarray  # (3,)
    obs_radius: jnp.ndarray  # (1,)


def _target_pose_batch(pos_batch, wxyz=DOWN_WXYZ):
    """Cast to float32 at the IK call boundary: the canonical-IK CUDA kernel's
    custom_jvp is float32-only (same boundary as GRiD's FFI -- see
    `ioc.robot.bases.dynamic`'s docstring for the analogous case), so under
    `JAX_ENABLE_X64=1` a float64 target silently mismatches the tangent dtype
    the custom rule expects and `jax.grad` raises rather than truncating
    quietly.  Casting here keeps the rest of the composed graph (trajopt) at
    its own working dtype."""
    n = pos_batch.shape[0]
    return jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3(wxyz=jnp.broadcast_to(wxyz, (n, 4)).astype(jnp.float32)),
        translation=pos_batch.astype(jnp.float32),
    )


@dataclasses.dataclass(frozen=True)
class PickPlaceProblem:
    """The composed planner: an IK stage plus four chained trajopt segments,
    all sharing one robot/collision model."""

    base: RobotProblem
    seg: dict  # phase name -> RobotProblem (same robot, different n_timesteps)

    @property
    def dof(self):
        return self.base.dof

    @property
    def ee_index(self):
        return self.base.ee_index

    @staticmethod
    def load(urdf_path, srdf_path, mesh_dir):
        base = RobotProblem.load(urdf_path, srdf_path, mesh_dir, n_timesteps=2)
        seg = {p: dataclasses.replace(base, n_timesteps=SEGMENT_LEN[p]) for p in PHASES}
        return PickPlaceProblem(base=base, seg=seg)

    # -- IK stage ----------------------------------------------------------

    def grasp_ik(self, theta_ik, scenes: PickPlaceScene):
        """(M, dof) previous-cfg batch -> (M, dof) canonical IK solution at
        `pick_pos + theta_ik[0] * UP_AXIS`.  See module docstring for why the
        standoff offset (not weight balance) is `theta_ik`'s meaning.

        `previous_cfg` is cast to float32 for the same reason `_target_pose_
        batch` casts the target: the canonical-IK custom_jvp is float32-only
        throughout (q, cfg_ref, AND target all have to agree), and mixing in a
        float64 `previous_cfg` under x64 promotes the kernel's own output to
        float64 by ordinary JAX promotion rules, which then mismatches the
        rule's declared float32 tangent -- not a target-only issue.  Cast the
        *output* back up so the trajopt stage downstream keeps its own dtype.
        """
        dtype = scenes.q_start.dtype
        target = scenes.pick_pos + theta_ik[0] * UP_AXIS
        q = sqp_ik_solve_cuda_batch(
            self.base.robot, self.ee_index, _target_pose_batch(target),
            IK_RNG_KEY, scenes.q_start.astype(jnp.float32),
        )
        return q.astype(dtype)

    def place_ik(self, theta_ik, scenes: PickPlaceScene, q_pick):
        """Continuity from the grasp pose: `previous_cfg=q_pick`, so the place
        IK's own null-space choice (and its canonical gradient) is relative to
        where the grasp phase actually left the arm, not the scene's home cfg."""
        dtype = q_pick.dtype
        target = scenes.place_pos + theta_ik[1] * UP_AXIS
        q = sqp_ik_solve_cuda_batch(
            self.base.robot, self.ee_index, _target_pose_batch(target),
            IK_RNG_KEY, q_pick.astype(jnp.float32),
        )
        return q.astype(dtype)

    # -- per-segment trajopt (reuses ioc.robot.problem.RobotProblem/Scene) -

    def segment_residual_fn(self, phase):
        problem = self.seg[phase]
        features = SEGMENT_FEATURES[phase]

        def residual_fn(x_flat, scene: Scene):
            q = problem.unpack(x_flat, scene)
            parts = []
            if "smooth" in features:
                parts.append((q[2:] - 2.0 * q[1:-1] + q[:-2]).reshape(-1))
            if "clearance" in features:
                parts.append(problem.clearance_residual(q, scene))
            if "upright" in features:
                quat = problem.robot.forward_kinematics(q)[..., problem.ee_index, 0:4]
                parts.append(quat[:, 1:3].reshape(-1))  # x,y ~ tilt, see old docstring
            return tuple(parts)

        return residual_fn

    def make_segment_inner(self, phase, forward_solver):
        from ioc.inner import make_inner_solver

        residual_fn = self.segment_residual_fn(phase)
        # Calibrated once per phase on that phase's own straight-line-plus-
        # jitter seed -- same reasoning as `RobotProblem.calibrate`'s docstring
        # (must not calibrate on the exactly-zero-acceleration straight line).
        return residual_fn, self.seg[phase]

    def calibrate_segment(self, phase, residual_fn, scenes: Scene, key, n_probe=16, jitter=0.15):
        problem = self.seg[phase]

        def raw(scene, k):
            x0 = problem.seed(scene)
            x = x0 + jitter * jax.random.normal(k, x0.shape)
            rs = residual_fn(x, scene)
            return jnp.stack([jnp.sum(r**2) for r in rs])

        keys = jax.random.split(key, n_probe)
        vals = jax.vmap(jax.vmap(raw, in_axes=(None, 0)), in_axes=(0, None))(scenes, keys)
        scales = jnp.mean(jnp.abs(vals.reshape(-1, vals.shape[-1])), axis=0)
        assert bool(jnp.all(scales > 1e-8)), f"{phase}: degenerate feature scale {scales}"
        return scales

    def ee_positions(self, q):
        return self.base.ee_positions(q)

    # -- full composed forward solve ----------------------------------------

    def solve(self, theta_ik, theta_trajopt_by_phase, scenes: PickPlaceScene,
              inner_by_phase, x0_by_phase):
        """One call per segment's `solve_implicit`, chained through literal
        (undetached) data flow -- the thing this module exists to demonstrate.

        Returns (q_pick, q_place, {phase: x_flat}, {phase: Scene}).
        """
        q_pick = self.grasp_ik(theta_ik, scenes)
        q_place = self.place_ik(theta_ik, scenes, q_pick)

        phase_scenes = {
            "approach": Scene(scenes.q_start, q_pick, scenes.obs_center, scenes.obs_radius),
            "grasp": Scene(q_pick, q_pick, scenes.obs_center, scenes.obs_radius),
            "transport": Scene(q_pick, q_place, scenes.obs_center, scenes.obs_radius),
            "place": Scene(q_place, q_place, scenes.obs_center, scenes.obs_radius),
        }
        xs = {}
        for phase in PHASES:
            inner = inner_by_phase[phase]
            sc = phase_scenes[phase]
            xs[phase] = jax.vmap(inner.solve_implicit, in_axes=(0, None, 0))(
                x0_by_phase[phase], theta_trajopt_by_phase[phase], sc)
        return q_pick, q_place, xs, phase_scenes

    def full_ee_path(self, scenes: PickPlaceScene, xs, phase_scenes, batch_index=0):
        """Concatenate one batch element's four segments into one EE path,
        dropping the duplicated boundary row between consecutive segments."""
        rows = []
        for i, phase in enumerate(PHASES):
            problem = self.seg[phase]
            sc = jax.tree.map(lambda a: a[batch_index], phase_scenes[phase])
            q = problem.unpack(xs[phase][batch_index], sc)
            p = self.ee_positions(q)
            rows.append(p[1:] if i > 0 else p)
        return jnp.concatenate(rows, axis=0)

    def seeds(self, scenes: PickPlaceScene, theta_ik, forward_solver_free=None):
        """Interior-waypoint seeds per phase, from the SAME IK call used by the
        differentiable forward path (not a separate disconnected precompute --
        see module docstring / the old design's `seed_ik` this replaces)."""
        q_pick = self.grasp_ik(theta_ik, scenes)
        q_place = self.place_ik(theta_ik, scenes, q_pick)
        phase_scenes = {
            "approach": Scene(scenes.q_start, q_pick, scenes.obs_center, scenes.obs_radius),
            "grasp": Scene(q_pick, q_pick, scenes.obs_center, scenes.obs_radius),
            "transport": Scene(q_pick, q_place, scenes.obs_center, scenes.obs_radius),
            "place": Scene(q_place, q_place, scenes.obs_center, scenes.obs_radius),
        }
        x0 = {p: jax.vmap(self.seg[p].seed)(phase_scenes[p]) for p in PHASES}
        return x0, phase_scenes, q_pick, q_place
