"""The 2D benchmarks: residual bases, contexts, seeds and feature whitening.

Each benchmark supplies `residual_fn(x, ctx, T, cfg)` returning residual vectors
r_k with phi_k = ||r_k||^2, matching the robot pipeline exactly (see
`ioc.inner`), so the same solver, adjoint and baselines are reused unchanged.

racing    Point mass through a closed circuit of *varying* curvature.  Weights
          trade lap time against staying inside the track and keeping the line
          smooth -- the classic racing-line tradeoff.  The circuit is not a
          circle: on a constant-curvature annulus the optimal line is just a
          smaller concentric circle, so the trajectory shows nothing about the
          weights.  Alternating tight and open corners make the line cut apexes
          and run wide on exits, which is what the weights actually control.

field     Point mass crossing a landscape of Gaussian cost bumps with unknown
          weights.  Recovering theta *is* recovering the cost field, so the
          result can be rendered as a map and compared to ground truth.  Bump
          centres are scattered and their widths vary: on a regular grid of
          equal-width bumps the per-bump features are near-duplicates and theta
          is not identifiable at all (measured Gram condition number 8.8e14 at
          width 0.9, i.e. numerically rank-deficient -- recovery then fails for
          reasons that have nothing to do with the method under test).

unicycle  Nonholonomic unicycle (x, y, phi) with controls (v, omega), driven
          through a slalom of alternating obstacles.  The canonical IOC testbed
          from the human-locomotion literature (Arechavaleta et al.; Mombaur,
          Truong & Laumond 2010, the closest prior work, which used
          derivative-free bilevel optimization on this system).  A single
          obstacle leaves the demonstrations nearly straight, so the turn and
          obstacle weights barely act; the slalom forces both.

segments  Point mass with one obstacle, with effort and smoothness weighted per
          *time segment*.  This is the benchmark for scaling K -- see
          `segment_residuals` for why `field` cannot play that role.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

SOFT = 40.0  # smoothing of hinge penalties (keeps the Hessian continuous)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Ctx:
    """One demonstration context (scene)."""

    start: jnp.ndarray  # (d,)
    goal: jnp.ndarray  # (d,)
    obstacles: jnp.ndarray  # (n_obs, 3): x, y, radius
    centers: jnp.ndarray  # (n_bump, 2) field-bump centres
    widths: jnp.ndarray  # (n_bump,)
    winding: jnp.ndarray  # () signed angle to travel round the circuit (racing)


def _softplus_hinge(v):
    return jax.nn.softplus(SOFT * v) / SOFT


def _soft_min(d, tau=0.05):
    """Soft-min over obstacles.

    Not `jnp.min`: a hard min is nondifferentiable where the closest obstacle
    changes identity, and the optimizer settles on exactly that ridge -- leaving
    the solve at a non-stationary point (measured ||grad|| ~ 5e-2 with a hard
    min, ~1e-10 with this soft-min).  Same failure mode as the hard max inside
    the robot collision checker.
    """
    return -tau * jax.scipy.special.logsumexp(-d / tau, axis=-1)


def track_radius_at(angle, cfg):
    """Centreline radius of the circuit at polar `angle`.

    The circuit is star-shaped about the origin, r(a) = R (1 + e2 cos 2a +
    e3 sin 3a), so the signed distance to the centreline is available in closed
    form as ||p|| - r(atan2(p)) -- no soft-min over a sampled polyline, and
    exactly differentiable everywhere on the corridor.  The 2nd and 3rd
    harmonics together give four corners of visibly different radius, so the
    apex of each is in a different place: that is what makes the racing line
    informative about the weights.

    The width is then measured radially rather than perpendicular to the
    centreline.  For the amplitudes used here (|e| <= 0.18) the two differ by
    under 3%, and it is the corridor as *defined* that the cost sees, so the
    benchmark is exactly the stated one either way.
    """
    e2, e3 = cfg["track_wave2"], cfg["track_wave3"]
    return cfg["track_radius"] * (
        1.0 + e2 * jnp.cos(2.0 * angle) + e3 * jnp.sin(3.0 * angle)
    )


# Six coherent lateral-detour modes about the straight start->goal line:
# sin(m*pi*t) for m=1,2,3, each with both signs.  m=1 bows the whole path to
# one side (goes around a cluster on the left or right); m=2,3 add an S-curve
# / double-S that can pass a near bump on one side and a far one on the other.
# This is a small, fixed basis for *topologically distinct* routes, not a
# random one: each mode is a different homotopy class relative to a point
# obstacle/bump near the line, so seeding one solve per mode and keeping the
# lowest-cost result (see `ioc.inner.make_inner_solver`'s `restart_seed_fn`)
# actually samples the field's distinct basins instead of resampling one.
_DETOUR_MODES = [(1, 1.0), (1, -1.0), (2, 1.0), (2, -1.0), (3, 1.0), (3, -1.0)]


def make_topo_seed_fn(T, d, amplitude=0.9):
    """Structured multistart seeds for 2D point-mass benchmarks (field, etc.).

    Returns `restart_seed_fn(x0, ctx, n_restarts) -> (n_restarts, (T-2)*d)`
    for `ioc.inner.make_inner_solver`.  Candidates are the straight-line seed
    plus up to six coherent lateral detours (see `_DETOUR_MODES`), amplitude
    scaled to the start-goal distance so it works across contexts without
    per-context tuning.  If `n_restarts - 1` exceeds the six structured modes,
    the remainder falls back to Gaussian jitter around x0 for extra coverage.
    """
    assert d == 2, "structured detour seeding is defined for 2D point masses"

    def restart_seed_fn(x0, ctx, n_restarts):
        direction = ctx.goal[:2] - ctx.start[:2]
        length = jnp.linalg.norm(direction) + 1e-9
        tangent = direction / length
        normal = jnp.array([-tangent[1], tangent[0]])
        t = jnp.linspace(0.0, 1.0, T)[1:-1]
        x0_pts = x0.reshape(T - 2, d)

        seeds = [x0]
        for m, sign in _DETOUR_MODES[: n_restarts - 1]:
            deflect = sign * amplitude * length * jnp.sin(m * jnp.pi * t)
            seeds.append((x0_pts + deflect[:, None] * normal[None, :]).reshape(-1))
        n_extra = n_restarts - len(seeds)
        if n_extra > 0:
            keys = jax.random.split(jax.random.key(0), n_extra)
            seeds += [x0 + 0.35 * jax.random.normal(k, x0.shape) for k in keys]
        return jnp.stack(seeds[:n_restarts])

    return restart_seed_fn


def unpack(x_flat, ctx, T, d):
    interior = x_flat.reshape(T - 2, d)
    return jnp.concatenate([ctx.start[None, :], interior, ctx.goal[None, :]], axis=0)


def seed_path(ctx, T, d, cfg=None):
    """Straight-line seed, except on the ring track, where it is an arc.

    A chord between two points on a circular corridor passes through the
    infield, which is a *different homotopy class* from going around the ring.  A
    local solver started there cannot cross to the arc, so every "racing"
    demonstration ends up cutting the infield no matter how heavily the boundary
    term is weighted -- the benchmark then poses nothing about racing lines.
    Seeding along the ring puts the solve in the intended class and lets time,
    boundary and curvature actually trade off.
    """
    al = jnp.linspace(0.0, 1.0, T)[1:-1, None]
    if cfg is not None and cfg.get("ring_seed", False):
        a0 = jnp.arctan2(ctx.start[1], ctx.start[0])
        # `ctx.winding` is the signed angle to travel, set when the context is
        # sampled.  Recovering it by unwrapping atan2 to the short way round
        # cannot represent an arc past pi -- a near-full lap would silently
        # become a short backwards one, in the wrong homotopy class.
        ang = a0 + al[:, 0] * ctx.winding
        # Seed on the centreline itself, not on a radius interpolation: the
        # corridor now has varying radius, so a straight radial blend leaves the
        # track in the corners and starts the solve outside the intended class.
        rad = track_radius_at(ang, cfg)
        return jnp.stack([rad * jnp.cos(ang), rad * jnp.sin(ang)], -1).reshape(-1)
    return ((1 - al) * ctx.start + al * ctx.goal).reshape(-1)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def racing_residuals(x, ctx, T, cfg):
    """Point mass in a corridor. Cost basis: time, boundary, smooth, curvature."""
    p = unpack(x, ctx, T, 2)
    v = p[1:] - p[:-1]
    a = p[2:] - 2 * p[1:-1] + p[:-2]

    r_time = v.reshape(-1)  # sum ||v||^2 ~ lap time at fixed T
    r_smooth = a.reshape(-1)

    # Corridor: the band of half-width W about the circuit centreline r(a).
    W = cfg["track_halfwidth"]
    ang = jnp.arctan2(p[:, 1], p[:, 0])
    r_bound = _softplus_hinge(
        jnp.abs(jnp.linalg.norm(p, axis=-1) - track_radius_at(ang, cfg)) - W
    )

    # Curvature: lateral acceleration, what actually limits a racing line.
    speed = jnp.linalg.norm(v, axis=-1) + 1e-6
    tang = v / speed[:, None]
    lat = a - jnp.sum(a * tang[:-1], axis=-1)[:, None] * tang[:-1]
    return (r_time, r_bound, r_smooth, lat.reshape(-1))


RACING_NAMES = ("time", "boundary", "smooth", "curvature")


def field_residuals(x, ctx, T, cfg):
    """Point mass over a Gaussian reward field; one weight per bump.

    Each bump contributes a residual whose squared norm is the accumulated
    exposure to that bump along the path, so K grows with the number of bumps
    while the trajectory dimension stays fixed.
    """
    p = unpack(x, ctx, T, 2)
    v = p[1:] - p[:-1]
    a = p[2:] - 2 * p[1:-1] + p[:-2]
    d2 = jnp.sum((p[:, None, :] - ctx.centers[None, :, :]) ** 2, axis=-1)  # (T, B)
    bumps = jnp.exp(-d2 / (2.0 * ctx.widths[None, :] ** 2))
    return tuple(
        [v.reshape(-1), a.reshape(-1)]
        + [bumps[:, b] for b in range(ctx.centers.shape[0])]
    )


def field_names(n_bump):
    return ("effort", "smooth") + tuple(f"bump{b}" for b in range(n_bump))


def unicycle_residuals(x, ctx, T, cfg):
    """Nonholonomic unicycle: state (x, y, phi), controls from differences.

    The nonholonomic constraint enters as a residual rather than a hard
    constraint, keeping the inner problem an unconstrained least-squares problem
    (and so Gauss-Newton solvable).
    """
    s = unpack(x, ctx, T, 3)
    xy, phi = s[:, :2], s[:, 2]
    d_xy = xy[1:] - xy[:-1]
    d_phi = phi[1:] - phi[:-1]

    heading = jnp.stack([jnp.cos(phi[:-1]), jnp.sin(phi[:-1])], axis=-1)
    r_v = jnp.sum(d_xy * heading, axis=-1)  # forward speed: effort / time
    r_omega = d_phi  # turning effort
    lateral = d_xy[:, 0] * jnp.sin(phi[:-1]) - d_xy[:, 1] * jnp.cos(phi[:-1])
    r_nonhol = lateral * cfg["nonhol_weight"]  # no side-slip
    r_jerk = d_phi[1:] - d_phi[:-1]  # steering smoothness

    obs = ctx.obstacles
    d = jnp.linalg.norm(xy[:, None, :] - obs[None, :, :2], axis=-1) - obs[None, :, 2]
    r_obs = _softplus_hinge(cfg["clearance"] - _soft_min(d))
    return (r_v, r_omega, r_jerk, r_obs, r_nonhol)


UNICYCLE_NAMES = ("speed", "turn", "steer_smooth", "obstacle", "nonholonomic")


def segment_residuals(x, ctx, T, cfg):
    """Point mass with one obstacle; effort, smoothness AND clearance weighted
    per time segment.

    K scales through the number of segments, and every added feature is a
    quadratic on a disjoint slice of the trajectory -- so the inner problem's
    landscape geometry is unchanged as K grows.  That is what makes this a clean
    test of cost dimension.  The Gaussian-field benchmark cannot do that job:
    there K is the number of bumps, so raising K also raises the number of local
    minima, and a K sweep there confounds dimension with multimodality (measured:
    the implicit gradient degrades from cos 1.0 to 0.37 as bumps are added).

    The obstacle-clearance term used to be a single global feature spanning the
    whole trajectory. Whenever the swerve needed to clear the obstacle fell
    mostly inside one segment (guaranteed at low S, e.g. S=2 where a segment
    spans half the trajectory), that segment's effort/smooth residuals became
    near-duplicates of the global obstacle residual -- both are just relabeling
    the same local deflection. That's the same near-duplicate-feature failure
    documented for the old equal-width `field` bump grid (Gram lambda_2/lambda_K
    -> 0), here showing up as demonstrations that won't converge to the
    stationarity screen at low K. Splitting clearance per segment, the same way
    effort/smooth already are, removes the aliasing: whichever segment owns the
    swerve gets its own clearance weight, geometrically distinct from that
    segment's effort/smooth cost, and segments the obstacle never touches simply
    get an inactive (near-zero) clearance residual.
    """
    p = unpack(x, ctx, T, 2)
    v = p[1:] - p[:-1]
    a = p[2:] - 2 * p[1:-1] + p[:-2]
    S = cfg["k_segments"]
    idx_groups = np.array_split(np.arange(v.shape[0]), S)

    obs = ctx.obstacles
    d = jnp.linalg.norm(p[:, None, :] - obs[None, :, :2], axis=-1) - obs[None, :, 2]
    clear = _softplus_hinge(cfg["clearance"] - _soft_min(d))  # (T,)

    out = []
    for idx in idx_groups:
        out.append(v[idx[0] : idx[-1] + 1].reshape(-1))
    for idx in idx_groups:
        out.append(a[idx[0] : idx[-1] + 1].reshape(-1))
    for idx in idx_groups:
        out.append(clear[idx[0] : idx[-1] + 2])  # +2: clear is indexed on p, not v
    return tuple(out)


def segment_names(S):
    return (
        tuple(f"effort_s{i}" for i in range(S))
        + tuple(f"smooth_s{i}" for i in range(S))
        + tuple(f"clear_s{i}" for i in range(S))
    )


# residual_fn, fixed feature names (None if K is configurable), state dimension
BENCHMARKS = {
    "racing": (racing_residuals, RACING_NAMES, 2),
    "field": (field_residuals, None, 2),
    "unicycle": (unicycle_residuals, UNICYCLE_NAMES, 3),
    "segments": (segment_residuals, None, 2),
}


def benchmark_names(benchmark, k_bumps, cfg):
    """Feature names for a benchmark, resolving the configurable-K cases."""
    if benchmark == "field":
        return field_names(k_bumps)
    if benchmark == "segments":
        return segment_names(cfg["k_segments"])
    return BENCHMARKS[benchmark][1]


def default_cfg(benchmark, **overrides):
    cfg = dict(
        ring_seed=(benchmark == "racing"),
        track_radius=1.5,
        track_halfwidth=0.42,
        track_wave2=0.16,
        track_wave3=0.10,
        n_obstacles=3 if benchmark == "unicycle" else 2,
        clearance=0.20,
        nonhol_weight=1.0,
        bump_width=0.45,
        k_segments=2,
    )
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# Contexts and whitening
# ---------------------------------------------------------------------------


def sample_contexts(rng, n, bench, T, d, n_bump, cfg):
    """Sample contexts whose features are actually excited by the demonstrations."""
    starts, goals, obs, cen, wid, wind = [], [], [], [], [], []
    for _ in range(n):
        if bench == "racing":
            # A long arc is essential: with a short one the chord stays inside the
            # corridor, so time, smoothness and curvature are all minimized by the
            # same path and no weight is identifiable.  Each context is a
            # multi-corner stint (>= ~170 deg) starting anywhere on the circuit,
            # so the demonstrations collectively cover the whole track and every
            # corner appears at several entry speeds.
            a0 = rng.uniform(-np.pi, np.pi)
            da = rng.uniform(3.0, 4.2) * rng.choice([-1.0, 1.0])
            starts.append(_on_centreline(a0, cfg))
            goals.append(_on_centreline(a0 + da, cfg))
            wind.append(da)
        elif bench == "unicycle":
            # Enter and leave the slalom pointing along it, with the lateral
            # offset and the exit heading varied: the car must weave, so the
            # turn, steering-smoothness and obstacle weights all act.
            starts.append(np.array([-2.3, rng.uniform(-0.5, 0.5),
                                    rng.uniform(-0.35, 0.35)]))
            goals.append(np.array([2.3, rng.uniform(-0.5, 0.5),
                                   rng.uniform(-0.35, 0.35)]))
        else:
            starts.append(np.array([-2.2, rng.uniform(-1.2, 1.2)]))
            goals.append(np.array([2.2, rng.uniform(-1.2, 1.2)]))
        if bench != "racing":
            wind.append(0.0)

        obs.append(_sample_obstacles(rng, bench, starts[-1], goals[-1], cfg))
        c, w = _sample_bumps(rng, n_bump, cfg)
        cen.append(c)
        wid.append(w)

    return Ctx(jnp.asarray(np.stack(starts)), jnp.asarray(np.stack(goals)),
               jnp.asarray(np.stack(obs)), jnp.asarray(np.stack(cen)),
               jnp.asarray(np.stack(wid)), jnp.asarray(np.asarray(wind)))


def _on_centreline(angle, cfg):
    r = float(track_radius_at(jnp.asarray(angle), cfg))
    return np.array([r * np.cos(angle), r * np.sin(angle)])


def _sample_obstacles(rng, bench, start, goal, cfg):
    """Obstacles that actually block the path.

    An obstacle the demonstration never comes near leaves the avoidance weight
    inactive, and an inactive feature is unidentifiable however many
    demonstrations are collected -- the same argument as
    `ioc.robot.problem.sample_scenes`.
    """
    n = max(1, cfg["n_obstacles"])
    if bench == "unicycle":
        # Slalom: gates alternating above and below the straight line from start
        # to goal, each one placed so the straight path is blocked and the car
        # has to commit to a side.  Evenly spaced in x, jittered so the contexts
        # are not copies of one another.
        xs = np.linspace(-1.15, 1.15, n)
        ob = []
        for i, x in enumerate(xs):
            side = -1.0 if i % 2 else 1.0
            r = rng.uniform(0.34, 0.46)
            ob.append(np.array([x + rng.uniform(-0.12, 0.12),
                                side * rng.uniform(0.0, 0.30), r]))
        return np.stack(ob)

    # Anchor the first obstacle on the seed path, offset by about its own radius:
    # close enough to block, varied enough that its influence differs by context.
    mid = 0.5 * (start[:2] + goal[:2])
    r0 = rng.uniform(0.30, 0.50)
    off = rng.normal(size=2)
    off = off / (np.linalg.norm(off) + 1e-9) * rng.uniform(0.0, 0.6) * r0
    ob = [np.array([mid[0] + off[0], mid[1] + off[1], r0])]
    for _ in range(n - 1):
        ob.append(np.array([rng.uniform(-1.0, 1.0), rng.uniform(-1.2, 1.2),
                            rng.uniform(0.25, 0.45)]))
    return np.stack(ob)


def _sample_bumps(rng, n_bump, cfg):
    """Scattered bump centres with varied widths.

    Not a regular grid of equal-width bumps.  Two bumps of the same width whose
    centres are close relative to that width induce nearly the same exposure
    along every path the solver produces, so their features are near-duplicates
    and only their *sum* is identifiable.  On the old 3x2 grid at width 0.9 the
    feature Gram reached condition number 8.8e14 -- numerically rank-deficient,
    i.e. theta was not recoverable by any method, which is not a property of the
    benchmark anyone intended to test.  Jittered centres on a coarse lattice
    (for coverage) plus a spread of widths (so bumps differ in *shape*, not only
    position) restores identifiability: lambda_2/lambda_K goes from 9.8e-8 to
    5.0e-2 at bump width 0.9.  `ioc.plots.fig_recovery` prints the certificate
    for the configuration it plots.

    The certificate is lambda_2, not lambda_min, of the trace-normalized Gram of
    `ioc.analytic.kkt_fit`.  At a demonstration the weighted feature gradients
    cancel by definition, so theta itself is always a null direction of that
    Gram and lambda_min ~ 0 is a sign of *convergence*, not of degeneracy.  It
    is the next eigenvalue that says whether some other feature combination
    leaves the demonstrations unchanged.
    """
    side = int(np.ceil(np.sqrt(n_bump)))
    gx, gy = np.meshgrid(np.linspace(-1.55, 1.55, side), np.linspace(-1.15, 1.15, side))
    pts = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    pts = pts[rng.permutation(pts.shape[0])[:n_bump]]
    cen = pts + rng.normal(scale=0.22, size=pts.shape)
    # Widths spread over a ~2.4x range about the nominal.  `bump_width` still
    # sets the regime (it is the axis of the multimodality study in
    # `collect.STAGES["bench2d_regime"]`), it just no longer forces every bump
    # to be the same shape.
    wid = cfg["bump_width"] * rng.uniform(0.62, 1.5, size=n_bump)
    return cen, wid


def calibrate(res_fn, ctxs, T, d, cfg, key, K, n_probe=12, jitter=0.25):
    """Nominal feature magnitudes, probed on perturbed trajectories.

    Whitening on the seed itself would collapse the smoothness scale to the
    numerical floor (the seed has zero acceleration); see
    `ioc.robot.problem.RobotProblem.calibrate`.
    """
    keys = jax.random.split(key, n_probe)

    def raw(ctx, k):
        x0 = seed_path(ctx, T, d, cfg)
        x = x0 + jitter * jax.random.normal(k, x0.shape)
        rs = res_fn(x, ctx, T, cfg)
        return jnp.stack([jnp.sum(r**2) for r in rs])

    vals = jax.vmap(jax.vmap(raw, in_axes=(None, 0)), in_axes=(0, None))(ctxs, keys)
    return jnp.maximum(jnp.mean(jnp.abs(vals.reshape(-1, K)), axis=0), 1e-8)
