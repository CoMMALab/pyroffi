"""Composed 4-phase pick-and-place planner: approach -> grasp -> transport -> place.

Inverts SPaSM-style sequential planning (see `tamp/spasm/tower|tetris` for the
task-skeleton/segment-cost terminology this borrows, read-only -- no code or
imports from there) at the smallest scope that still exercises multi-segment
IOC: one fixed, hardcoded skeleton, one object, on the Panda.  No task planner.

Segment composition -- the design decision this module hinges on
------------------------------------------------------------------
`ioc.inner.make_inner_solver` differentiates one flat decision vector `x`
against one context `ctx` through one `residual_fn(x, ctx)`.  It does not know
about "segments"; it only needs `x` to be an unconstrained R^n point and
`residual_fn` to be a JAX function of it.  So rather than solving four
`RobotProblem`-style segments with clamped-at-both-ends interiors and stitching
their boundary conditions together (which `ioc.inner` has no machinery for --
it would need shared-variable coupling across four separate inner problems),
this module represents the WHOLE composed trajectory as ONE trajectory of
`T = n_approach + n_grasp + n_transport + n_place` waypoints, with only the
very first waypoint (the robot's home config) clamped as a boundary condition.
Every other waypoint -- including the three phase-transition points -- is a
free decision variable of a single inner problem.  Phase boundaries are then
just index slices into that one trajectory when features are computed, and
continuity between segments is automatic (it is literally the same array)
rather than an explicit constraint.

This is a strict generalization of `RobotProblem.unpack`/`seed` (which clamp
BOTH endpoints because a segment there has a known start and goal): here only
the start is known a priori, and the pick/grasp/place poses that the other
phase boundaries should reach are expressed as soft cost residuals (the
`*_align` features below), exactly like every other feature in `ioc.robot.bases`.
The upshot: `ioc.inner.make_inner_solver` and all of `ioc.outer` (FD, CMA-ES,
Adam-on-the-implicit-gradient) apply completely unmodified -- confirmed by the
smoke test in `iosp/examples/`'s `--check-grads` path. `ioc.analytic`
(KKT/CIOC) is NOT extended here: both need the Hessian/gradient of a single
stationary point of a *quadratic-in-theta* cost with an explicit Gram matrix
per named feature, which is unaffected by this change, so in principle they
would still apply to the concatenated problem -- but validating that on a
4-phase composed cost was out of scope for this pass (see module scope note in
the caller); only FD / CMA-ES / implicit are wired up here.

Theta parameterization
-----------------------
Per-segment, named blocks, concatenated into ONE K-vector and normalized with
a single global `softmax` (matching `ioc.robot.problem`'s convention, and
reusable via `ioc.robot.problem.make_outer`/`ioc.outer` verbatim): weights are
not independently normalized per segment because the outer loss only pins down
the *product* of a segment's weights and how strongly its residuals are
excited, and a global simplex is the smaller, already-verified machinery.  Per-
segment normalization is flagged below as a natural follow-up if the recovered
weights turn out to trade off *across* phases in a way a shared budget hides.

Names (K=9): `approach.smooth`, `approach.clearance`, `grasp.smooth`,
`grasp.align`, `transport.smooth`, `transport.clearance`, `transport.upright`,
`place.smooth`, `place.align`.

Object/grasp model
-------------------
No physics: the object is treated as rigidly welded to the end-effector from
the end of `grasp` through the end of `place` (`object_pose(t) = ee_pose(t)`,
grasp offset = identity).  "Pick"/"place" are reaching a target EE pose near
the object -- there is no separate pick-up/put-down event, no contact model,
and no force/torque feature (out of scope; `ioc.robot.bases.dynamic` shows how
one would be added if this were extended to a dynamic basis).  "Keep object
upright" during transport is approximated as small quaternion x/y components
(roll/pitch) rather than a proper SO(3) log-map deviation, since the target
attitude here is simply "whatever yaw the grasp left it at, don't tip it" --
adequate at the small tilts a feasible trajectory produces, not a general
attitude-tracking term.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from ioc.robot.problem import RobotProblem

N_APPROACH = 8
N_GRASP = 4
N_TRANSPORT = 10
N_PLACE = 4

THETA_NAMES = (
    "approach.smooth", "approach.clearance",
    "grasp.smooth", "grasp.align",
    "transport.smooth", "transport.clearance", "transport.upright",
    "place.smooth", "place.align",
)
K = len(THETA_NAMES)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PickPlaceScene:
    """Context for one pick-and-place demonstration.

    `pick_pos`/`place_pos` are EE target positions (grasp offset folded in, see
    module docstring); `obs_center`/`obs_radius` reuse `ioc.robot.problem.Scene`'s
    single-sphere obstacle so the composed planner stays visually and
    numerically consistent with the single-segment IOC study.
    """

    q_start: jnp.ndarray  # (dof,)
    pick_pos: jnp.ndarray  # (3,)
    place_pos: jnp.ndarray  # (3,)
    obs_center: jnp.ndarray  # (3,)
    obs_radius: jnp.ndarray  # (1,)


def phase_slices():
    """Waypoint index ranges (inclusive start, exclusive end) per phase, into
    the T = N_APPROACH + N_GRASP + N_TRANSPORT + N_PLACE trajectory."""
    a0 = 0
    a1 = a0 + N_APPROACH
    g1 = a1 + N_GRASP
    t1 = g1 + N_TRANSPORT
    p1 = t1 + N_PLACE
    return {
        "approach": (a0, a1),
        "grasp": (a1, g1),
        "transport": (g1, t1),
        "place": (t1, p1),
    }


T_TOTAL = sum(phase_slices()[k][1] - phase_slices()[k][0] for k in phase_slices())


@dataclasses.dataclass(frozen=True)
class PickPlaceProblem:
    """The composed 4-phase problem: one robot, four chained segments.

    Wraps a `RobotProblem` for FK/collision rather than re-deriving them --
    only the trajectory parameterization and cost basis are new.
    """

    problem: RobotProblem

    @property
    def dof(self):
        return self.problem.dof

    @staticmethod
    def load(urdf_path, srdf_path, mesh_dir):
        return PickPlaceProblem(
            problem=RobotProblem.load(urdf_path, srdf_path, mesh_dir, T_TOTAL)
        )

    # -- trajectory parameterization ------------------------------------------

    def unpack(self, x_flat, scene: PickPlaceScene):
        """Free waypoints (T_TOTAL - 1 of them) -> full trajectory, q_start clamped."""
        rest = x_flat.reshape(T_TOTAL - 1, self.dof)
        return jnp.concatenate([scene.q_start[None, :], rest], axis=0)

    def seed(self, scene: PickPlaceScene):
        """Straight-line-ish seed: home -> pick (via IK-free linear joint blend
        toward a config that roughly reaches pick_pos) -> hold -> ... -> place.

        No IK is run for the seed (keeps this a pure function of the scene,
        matching `RobotProblem.seed`'s spirit); it only needs to be a reasonable
        starting point for Gauss-Newton, not a good trajectory.
        """
        q0 = scene.q_start
        alphas = jnp.linspace(0.0, 1.0, T_TOTAL)[1:, None]
        # A mild, deterministic joint-space wobble toward "reach forward" is
        # enough seed diversity for the solver; the cost features (align,
        # clearance) do the actual work of reaching the target poses.
        drift = jnp.zeros_like(q0).at[0].set(0.3).at[1].set(-0.2)
        rest = q0[None, :] + alphas * drift[None, :]
        return rest.reshape(-1)

    def seeds(self, scenes: PickPlaceScene):
        return jax.vmap(self.seed)(scenes)

    def ee_positions(self, q):
        return self.problem.ee_positions(q)

    def ee_quats(self, q):
        return self.problem.robot.forward_kinematics(q)[..., self.problem.ee_index, 0:4]

    # -- cost basis -------------------------------------------------------------

    def residual_fn(self, x_flat, scene: PickPlaceScene):
        """Named residual tuple, one entry per THETA_NAMES component."""
        q = self.unpack(x_flat, scene)
        slices = phase_slices()

        def smooth(lo, hi):
            qs = q[max(lo - 1, 0):hi + 1]
            return (qs[2:] - 2.0 * qs[1:-1] + qs[:-2]).reshape(-1)

        def clearance(lo, hi):
            return self.problem.clearance_residual(q[lo:hi], scene)

        def align(idx, target_pos):
            return self.ee_positions(q[idx:idx + 1])[0] - target_pos

        a_lo, a_hi = slices["approach"]
        g_lo, g_hi = slices["grasp"]
        t_lo, t_hi = slices["transport"]
        p_lo, p_hi = slices["place"]

        r_approach_smooth = smooth(a_lo, a_hi)
        r_approach_clear = clearance(a_lo, a_hi)

        r_grasp_smooth = smooth(g_lo, g_hi)
        r_grasp_align = align(g_hi - 1, scene.pick_pos)

        r_transport_smooth = smooth(t_lo, t_hi)
        r_transport_clear = clearance(t_lo, t_hi)
        quat_transport = self.ee_quats(q[t_lo:t_hi])
        r_transport_upright = quat_transport[:, 1:3].reshape(-1)  # x,y ~ tilt

        r_place_smooth = smooth(p_lo, p_hi)
        r_place_align = align(p_hi - 1, scene.place_pos)

        return (
            r_approach_smooth, r_approach_clear,
            r_grasp_smooth, r_grasp_align,
            r_transport_smooth, r_transport_clear, r_transport_upright,
            r_place_smooth, r_place_align,
        )

    # -- scenes / calibration (thin wrappers around RobotProblem's) ------------

    def calibrate(self, scenes, key, n_probe=16, jitter=0.15):
        def raw(scene, k):
            x0 = self.seed(scene)
            x = x0 + jitter * jax.random.normal(k, x0.shape)
            rs = self.residual_fn(x, scene)
            return jnp.stack([jnp.sum(r**2) for r in rs])

        keys = jax.random.split(key, n_probe)
        vals = jax.vmap(jax.vmap(raw, in_axes=(None, 0)), in_axes=(0, None))(
            scenes, keys
        )
        scales = jnp.mean(jnp.abs(vals.reshape(-1, vals.shape[-1])), axis=0)
        assert bool(jnp.all(scales > 1e-8)), f"degenerate feature scale: {scales}"
        return scales
