"""E1 — Minimal, identifiability-clean pick-and-place recovery.

Question: with few features (K=3) and curated demos designed to excite each
feature differently, does implicit-adjoint recovery work well?

Reduced parameterization (K=3, transport phase only):
    transport.clearance, transport.smooth, transport.line_dev

All other parameters (theta_ik and non-transport trajopt weights) are held
fixed at ground truth.  `transport.line_dev` replaces `transport.upright`,
which was structurally collinear with `transport.smooth`.

Curated demo scenes (see `CURATED_SCENES`):
    "clear"   intrusive obstacle on the pick->place midline
    "smooth"  obstacle inactive, short transport
    "shape"   obstacle inactive, wide lateral spread

Procedure: certificate -> rank selection -> zero-prior subspace fit ->
generalization test.

Usage:
    CUDA_VISIBLE_DEVICES=<idx> XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        python -m iosp.experiments.e1_minimal_identifiable
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from ioc.inner import make_inner_solver
from ioc import outer as outer_opt
from iosp.model import pickplace as pp
from iosp.config import MESH_DIR, OBS_CENTER, OBS_RADIUS, PICK_POS, PLACE_POS, Q_START, SRDF_PATH, THETA_IK_STAR, URDF_PATH, Z_TRAJOPT_STAR
from iosp.model.pickplace import split_trajopt as _split_trajopt

# Indices of the 3 free (transport.*) logits within the shared 7-dim
# theta_trajopt logit vector -- see pp.THETA_TRAJOPT_NAMES for the fixed
# ordering this relies on.  Position 5 (originally "transport.upright") now
# feeds `_line_dev_residual` instead -- see module docstring, REVISION 1.
_TRANSPORT_IDX = jnp.array([3, 4, 5])
assert [pp.THETA_TRAJOPT_NAMES[i] for i in [3, 4]] == ["transport.smooth", "transport.clearance"]
K_FREE = 3
FREE_NAMES = ("transport.smooth", "transport.clearance", "transport.line_dev")


def _line_dev_residual(q, problem):
    """Perpendicular distance of the EE path from the straight line joining
    the transport segment's own pinned endpoints -- a position/shape residual,
    structurally unrelated to `transport.smooth`'s joint-space curvature
    penalty (unlike `transport.upright`'s quaternion tilt, which turned out to
    be near-exactly anti-collinear with it -- see module docstring)."""
    p = problem.ee_positions(q)  # (T, 3)
    p0, p1 = p[0], p[-1]
    line_dir = p1 - p0
    line_unit = line_dir / (jnp.linalg.norm(line_dir) + 1e-8)
    rel = p - p0[None, :]
    proj = jnp.sum(rel * line_unit[None, :], axis=-1, keepdims=True) * line_unit[None, :]
    perp = rel - proj
    return perp.reshape(-1)


def _transport_residual_fn(problem):
    """Local variant of `PickPlaceProblem.segment_residual_fn("transport")`
    with `line_dev` in place of `upright` -- built here, NOT by editing
    `iosp.model.pickplace`'s shared `SEGMENT_FEATURES`/`segment_residual_fn`, so
    every other caller of that module is unaffected (see module docstring)."""

    def residual_fn(x_flat, scene):
        q = problem.unpack(x_flat, scene)
        r_smooth = (q[2:] - 2.0 * q[1:-1] + q[:-2]).reshape(-1)
        r_clearance = problem.clearance_residual(q, scene)
        r_line_dev = _line_dev_residual(q, problem)
        return (r_smooth, r_clearance, r_line_dev)

    return residual_fn


# Curated scenes: deliberately designed, not sampled.  Each perturbs
# recovery_bench's nominal scene along one axis meant to load onto one
# feature; obstacle/pick/place stay within the Panda's reach envelope used
# throughout this investigation.
CURATED_SCENES = {
    "clear": dict(
        q_start=Q_START,
        pick_pos=PICK_POS, place_pos=PLACE_POS,
        obs_center=jnp.array([0.4, 0.0, 0.3], dtype=jnp.float32),  # on the pick->place midline
        # REVISION 1: attempt-0's radius (0.08) let the solver detour with
        # room to spare -- MEASURED exactly-zero clearance gradient, not a
        # near-zero one.  0.25 is large relative to the ~0.4m pick/place
        # separation, so a full detour costs more (via smooth/line_dev) than
        # accepting some margin violation -- verified below, not assumed.
        obs_radius=jnp.array([0.25], dtype=jnp.float32),
    ),
    "smooth": dict(
        q_start=Q_START,
        pick_pos=jnp.array([0.4, 0.1, 0.3], dtype=jnp.float32),
        place_pos=jnp.array([0.4, -0.1, 0.3], dtype=jnp.float32),  # short transport
        obs_center=jnp.array([0.65, 0.35, 0.6], dtype=jnp.float32),  # far away: inactive
        obs_radius=jnp.array([0.03], dtype=jnp.float32),
    ),
    "shape": dict(
        q_start=Q_START + jnp.array([0.3, 0.0, 0.0, -0.2, 0.0, 0.2, 0.0], dtype=jnp.float32),
        pick_pos=jnp.array([0.45, 0.3, 0.25], dtype=jnp.float32),
        place_pos=jnp.array([0.3, -0.3, 0.4], dtype=jnp.float32),  # large lateral+height spread
        obs_center=jnp.array([0.65, 0.35, 0.6], dtype=jnp.float32),  # far away: inactive
        obs_radius=jnp.array([0.03], dtype=jnp.float32),
    ),
}


def _full_joint_path(prob, xs, phase_scenes, batch_index=0):
    """Joint-space analogue of `PickPlaceProblem.full_ee_path` -- concatenates
    one batch element's four segments' joint CONFIGURATIONS (not their EE
    positions) into one path, dropping the duplicated boundary row between
    consecutive segments.  Built here, not added to `pickplace.py`, for the
    same reason `_line_dev_residual` was: it's an addition specific to this
    module's reconstruction target, not a change to shared code any other
    caller (`recovery_bench`, `generalization_check`, the viser example)
    should see.

    Why joint space instead of the SE(3) EE path `full_ee_path` was using:
    the Panda is 7-DOF against a 6-DOF pose task (see `pickplace.py`'s module
    docstring), so many joint configurations reproduce the SAME EE pose.  An
    EE-path reconstruction loss is therefore satisfied by ANY of those
    configurations -- it cannot distinguish "recovered the right cost" from
    "found a different redundant-arm solution that happens to trace the same
    path," which inflates the flat/near-flat region of the loss (many thetas
    give an EE-equivalent path) without that flatness reflecting genuine
    unidentifiability of theta.  The demo IS given as joint configurations
    (a teleop trajectory, not just an EE trace), so matching joint space
    directly is both the more available supervision signal and the tighter,
    more tractable target.
    """
    rows = []
    for i, phase in enumerate(pp.PHASES):
        problem = prob.seg[phase]
        sc = jax.tree.map(lambda a: a[batch_index], phase_scenes[phase])
        q = problem.unpack(xs[phase][batch_index], sc)
        rows.append(q[1:] if i > 0 else q)
    return jnp.concatenate(rows, axis=0)


def _split_trajopt_batched(theta_trajopt):
    """`_split_trajopt` with a leading candidate axis: (C, 7) -> {phase: (C, n_p)}.

    The split is by fixed feature counts, so it is the same slice applied on the
    last axis instead of the only one.
    """
    out, i = {}, 0
    for p in pp.PHASES:
        n = len(pp.SEGMENT_FEATURES[p])
        out[p] = theta_trajopt[:, i:i + n]
        i += n
    return out


def _full_joint_paths(prob, xs, phase_scenes):
    """(C, T, dof) -- `_full_joint_path` for the whole batch, vmapped.

    Same concatenation and same duplicated-boundary drop; row b equals
    `_full_joint_path(..., batch_index=b)`.  Exists so a flattened multistart
    can read out every candidate's path without unrolling the concat C times.
    """
    rows = []
    for i, phase in enumerate(pp.PHASES):
        q = jax.vmap(prob.seg[phase].unpack)(xs[phase], phase_scenes[phase])
        rows.append(q[:, 1:] if i > 0 else q)
    return jnp.concatenate(rows, axis=1)


def _make_scene(spec):
    return pp.PickPlaceScene(
        q_start=spec["q_start"], pick_pos=spec["pick_pos"], place_pos=spec["place_pos"],
        obs_center=spec["obs_center"], obs_radius=spec["obs_radius"],
    )


def _z_trajopt_full(z_free, background=Z_TRAJOPT_STAR):
    """Embed the 3 free transport logits into the full 7-dim logit vector,
    holding the other 4 fixed at their Study-0 ground-truth values."""
    return background.at[_TRANSPORT_IDX].set(z_free)


def _setup(prob, forward_solver, scene, seed=0):
    scenes = jax.tree.map(lambda a: a[None], scene)
    theta_trajopt_star = jax.nn.softmax(Z_TRAJOPT_STAR)
    x0_star, phase_scenes_star, _, _ = prob.seeds(scenes, THETA_IK_STAR)

    inner_by_phase, residual_fn_by_phase, scales_by_phase = {}, {}, {}
    for p in pp.PHASES:
        if p == "transport":
            residual_fn = _transport_residual_fn(prob.seg[p])
        else:
            residual_fn, _ = prob.make_segment_inner(p, forward_solver)
        residual_fn_by_phase[p] = residual_fn
        # jitter=0.3 (not calibrate_segment's 0.15 default): the curated
        # scenes deliberately move the obstacle close to ONE phase's path
        # (e.g. "clear" targets transport) and leave others far from it, so a
        # phase whose own path stays consistently clear under the default
        # jitter can get a degenerate (all-zero) clearance scale -- a
        # real side effect of scene curation, not a bug in the calibration
        # itself. Only the transport-phase scales are load-bearing for this
        # study's certificate; the other phases' scales just need to be
        # non-degenerate so `make_inner_solver` can be built at all.
        scales = prob.calibrate_segment(p, residual_fn, phase_scenes_star[p], jax.random.PRNGKey(seed), jitter=0.3)
        scales_by_phase[p] = scales
        inner_by_phase[p] = make_inner_solver(residual_fn, scales, forward_solver=forward_solver)
    return scenes, theta_trajopt_star, x0_star, inner_by_phase, residual_fn_by_phase, scales_by_phase


def _solve_all(prob, scenes, inner_by_phase, theta_trajopt, x0=None):
    theta_trajopt_by_phase = _split_trajopt(theta_trajopt)
    if x0 is None:
        x0, _, _, _ = prob.seeds(scenes, THETA_IK_STAR)
    _, _, xs, phase_scenes2 = prob.solve(THETA_IK_STAR, theta_trajopt_by_phase, scenes, inner_by_phase, x0)
    return xs, phase_scenes2


def run_certificate(seed=0, whiten=True, scene_specs=None, prob=None, forward_solver=None):
    """Step 3: identifiability certificate on the COMBINED gradient set across
    the given scenes (default: the 3 curated demos) -- only the 3 free
    `transport.*` features (the other 6 directions are fixed, not part of
    this recovery problem, so including them in the certificate would just
    restate identifiability_check.py's finding about them rather than test
    whether curation fixed the transport collinearity).

    Generic over `scene_specs` (a name->spec dict, same shape as
    `CURATED_SCENES`) so Study 2 can pass its own demo-regime scene sets
    through this SAME function rather than reimplementing it.  Accepts an
    existing `prob`/`forward_solver` to reuse across calls within one process
    (each fresh `PickPlaceProblem.load` reloads the URDF/collision model, and
    each new `forward_solver` config is cheap, but reusing them saves the
    reload and keeps the JIT cache warm across regimes with matching shapes).

    `whiten=True` (the methodology fix, now the default): divides each
    feature's raw gradient column by its OWN `calibrate_segment` scale before
    building the Gram matrix.  This is not optional polish -- the outer loop
    never sees raw-unit gradients: `ioc.inner.make_inner_solver.features`
    computes `sum(r_k**2) / scales[k]`, so theta_k's actual effect on the cost
    is through the WHITENED feature, and a certificate built on raw gradients
    partly measures an artifact of residual units (joint-space smoothness
    sumsq vs. a meters-scale clearance hinge vs. a meters-scale line-deviation
    term) rather than the identifiability the fitted parameterization actually
    has.  `whiten=False` reproduces the earlier (raw-gradient) numbers for
    comparison.
    """
    if prob is None:
        prob = pp.PickPlaceProblem.load(str(URDF_PATH), str(SRDF_PATH), str(MESH_DIR))
    if forward_solver is None:
        forward_solver = pp.make_composed_forward_solver(n_iters=60)
    scene_specs = CURATED_SCENES if scene_specs is None else scene_specs

    B_blocks = []
    per_demo_norms = {}
    for demo_name, spec in scene_specs.items():
        scene = _make_scene(spec)
        scenes, theta_trajopt_star, x0_star, inner_by_phase, residual_fn_by_phase, scales_by_phase = _setup(
            prob, forward_solver, scene, seed=seed)
        xs_star, ps_star = _solve_all(prob, scenes, inner_by_phase, theta_trajopt_star, x0_star)

        phase = "transport"
        sc = jax.tree.map(lambda a: a[0], ps_star[phase])
        residual_fn = residual_fn_by_phase[phase]
        x_star_phase = xs_star[phase][0]
        scales_transport = scales_by_phase[phase]  # order matches FREE_NAMES: smooth, clearance, line_dev

        cols = []
        norms = {}
        for idx, name in enumerate(FREE_NAMES):  # smooth, clearance, line_dev
            def phi(x, idx=idx):
                return jnp.sum(residual_fn(x, sc)[idx] ** 2)
            g = jax.grad(phi)(x_star_phase)
            if whiten:
                g = g / scales_transport[idx]
            cols.append(g)
            norms[name] = float(jnp.linalg.norm(g))
        per_demo_norms[demo_name] = norms
        B_blocks.append(jnp.stack(cols, axis=-1))  # (dim_transport_x, 3), order = FREE_NAMES

    Gs = [B.T @ B for B in B_blocks]
    G_combined = sum(Gs) / len(Gs)
    G_np = np.asarray(G_combined)
    G_normed = G_np / (np.trace(G_np) / K_FREE + 1e-30)
    eigvals, eigvecs = np.linalg.eigh(G_normed)

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))

    cosines_per_demo = {}
    for demo_name, B in zip(scene_specs, B_blocks):
        Bn = np.asarray(B)
        cosines_per_demo[demo_name] = {
            f"{FREE_NAMES[i]} vs {FREE_NAMES[j]}": cos(Bn[:, i], Bn[:, j])
            for i in range(3) for j in range(i + 1, 3)
        }

    return dict(
        per_demo_norms=per_demo_norms, G=G_np, eigvals=eigvals, eigvecs=eigvecs,
        cosines_per_demo=cosines_per_demo, names=FREE_NAMES, whitened=whiten,
        prob=prob, forward_solver=forward_solver,
    )


def _print_certificate(r):
    print(f"whitened={r['whitened']}")
    print("per-demo, per-feature gradient norms (transport phase only):")
    for demo_name, norms in r["per_demo_norms"].items():
        print(f"  [{demo_name}]", {k: round(v, 6) for k, v in norms.items()})
    print()
    print("combined Gram matrix (averaged over 3 demos):")
    print(np.array2string(r["G"], precision=4, suppress_small=True))
    print()
    print("eigenvalues (trace-normalized, ascending):", r["eigvals"])
    print()
    print("pairwise cosines PER DEMO (not averaged -- averaging could hide a")
    print("collinearity that only shows up strongly on one demo):")
    for demo_name, cosines in r["cosines_per_demo"].items():
        print(f"  [{demo_name}]")
        for pair, c in cosines.items():
            print(f"    {pair:40s} cos={c:.4f}")


# Held-out scene for generalization, kept OUT of `CURATED_SCENES` (and
# therefore out of the certificate) so it is a genuinely independent check,
# not one of the demos the certificate itself was accumulated over (a
# methodological wrinkle in the earlier attempts: "shape" was both a
# certificate demo AND the held-out generalization scene there).  Another
# obstacle-intrusion scenario (different position/radius than "clear"),
# chosen because the null eigenvector is dominated by the clearance feature
# (V[:,0] ~ [-0.34, -0.92, -0.21], ~84% of its squared norm is clearance), so
# a second, differently-shaped obstacle scenario is the natural place to
# check whether the zero-prior convention costs something behaviorally.
HELD_OUT_SCENES = {
    "heldout_clear": dict(
        q_start=Q_START,
        pick_pos=jnp.array([0.42, 0.22, 0.32], dtype=jnp.float32),
        place_pos=jnp.array([0.38, -0.18, 0.28], dtype=jnp.float32),
        obs_center=jnp.array([0.38, -0.02, 0.30], dtype=jnp.float32),
        obs_radius=jnp.array([0.20], dtype=jnp.float32),
    ),
}


def select_rank(eigvals, cumulative_threshold=0.95):
    """CANONICAL rank-selection rule: the smallest set of TOP eigendirections
    (by eigenvalue, descending) whose cumulative sum covers >=
    `cumulative_threshold` of the total trace.

    Chosen over a fixed epsilon threshold (e.g. "keep eigval >= 1e-2")
    because it is scale-free: it works the same way whether the spectrum
    looks like `[1e-4, 0.6, 2.4]` or has different absolute magnitudes
    entirely (different demo counts/regimes give differently-scaled Gram
    matrices), and needs no per-study hand-tuning -- exactly what Study 2
    needs to run unattended across several demo regimes with different
    spectra.  95% is a conventional, defensible cutoff (same spirit as e.g.
    "95% of variance explained" in PCA); it is fixed BEFORE looking at any
    of this codebase's specific spectra, not fit to make a particular result
    come out cleanly.  Generic over K -- operates on a plain eigenvalue
    array of any length.

    Returns `(k, selected_idx)`: `k` is the selected rank, `selected_idx` are
    indices into the ORIGINAL (ascending, `np.linalg.eigh`-order) eigval/
    eigvec arrays, ordered from most to least important.
    """
    eigvals = np.asarray(eigvals)
    order = np.argsort(eigvals)[::-1]  # descending, i.e. most important first
    total = float(np.sum(eigvals))
    cum = np.cumsum(eigvals[order])
    k = int(np.searchsorted(cum, cumulative_threshold * total) + 1)
    k = int(min(max(k, 1), len(eigvals)))
    return k, order[:k]


def _nonneg_transport(theta_raw):
    """Elementwise shifted-softplus: `max(0, softplus(x) - softplus(0))`.

    MOTIVATION (found via the grid/kkt/multistart/cma-es comparison in this
    module's history): every fitted `theta_transport_hat` under the raw
    (unconstrained) linear map `V_top @ alpha` came out with NEGATIVE
    components, even though `theta_transport_star` (a slice of a softmax
    output) is all-positive by construction. A negative weight on a squared
    residual term makes the INNER trajopt cost unbounded below along that
    residual -- not merely non-convex, ill-posed -- which is consistent with
    what was actually measured: wild loss spikes (9-140) during optimization,
    and CMA-ES (population-based, no local-basin commitment) converging to
    essentially the SAME loss floor (~0.46) as gradient multistart, which is
    the signature of an optimizer-independent pathology in the objective
    itself, not a hard-to-navigate but well-posed landscape.

    Properties this transform is chosen for, in order:
    (1) `theta_raw=0 -> output=0` EXACTLY (preserves the zero-prior
        convention's `alpha=0` == the "no-fit" baseline every comparison in
        this module is measured against -- softplus(x) alone does NOT have
        this property, `softplus(0)=log(2)`, so the naive version is wrong).
    (2) output >= 0 EVERYWHERE (not just at the origin), so the inner
        trajopt cost stays well-posed for every alpha the outer optimizer
        can reach, not just ones near zero.
    (3) monotonic and smooth for theta_raw > 0 (a smoothed ReLU), so the
        outer gradient doesn't vanish immediately off the zero boundary the
        way a hard `relu` clamp's zero-gradient region would.
    """
    return jnp.maximum(0.0, jax.nn.softplus(theta_raw) - jnp.log(2.0))


def theta_from_alpha_zero_prior(alpha, eigvecs, selected_idx, nonneg=False):
    """CANONICAL, real-inference-time reparameterization: theta_transport =
    sum_i alpha_i * eigvecs[:, selected_idx[i]] -- every eigendirection NOT
    selected (i.e. every near-null direction the certificate found the demos
    don't identify) gets EXACTLY ZERO weight, not the (in a real setting,
    UNKNOWN) ground-truth theta_star's own value.

    This is the fix for the earlier attempts' non-canonical convention
    (fixing the null component at theta_star -- usable only because
    theta_star was known in this synthetic study, not usable at real
    inference time).  Zero is the natural "no-belief" default in the
    calibrated/whitened feature basis: the outer loop's cost is `theta . phi`
    in this SAME whitened basis, so a zero coefficient means "this feature
    contributes nothing to the learned cost," the honest statement of "the
    demos gave no information about this direction," not an arbitrary
    placeholder.  `alpha=0` (i.e. no fitting at all) is therefore also the
    correct "no-fit baseline" for this convention: zero weight on every
    `transport.*` feature, not just the unidentified ones.

    `nonneg=True` (opt-in, default False so existing callers/behavior are
    unchanged) additionally passes the raw linear map through
    `_nonneg_transport` -- see its docstring for why: the RAW map lets
    `alpha` reach sign-indefinite `theta_transport`, which is where the
    grid/kkt/multistart/cma-es comparison found every method converging to
    the same non-improving loss floor.
    """
    V_top = eigvecs[:, selected_idx]  # (K, k)
    theta_raw = V_top @ alpha
    return _nonneg_transport(theta_raw) if nonneg else theta_raw


def kkt_seed_alpha(prob, forward_solver, fit_scene_spec, theta_trajopt_star, eigvecs, selected_idx,
                    target_theta_mag=(0.1, 0.3), seed=0, n_steps=200, lr=0.05):
    """Analytic (KKT-stationarity) seed for `alpha0`, mirroring `ioc.analytic.
    kkt_fit`: if the fit-scene's ground-truth rollout `x_demo` is (near)
    optimal under SOME cost in the 3 free `transport.*` features, it must be
    a near-stationary point of that cost, i.e. `B(x_demo)^T @ theta ~= 0`
    only for theta ALIGNED with the small-residual directions of
    `G = B^T @ B`.  Solving `min_theta theta^T G theta` on the simplex
    (exactly `kkt_fit`'s objective/parameterization) recovers theta's
    DIRECTION for free -- no forward solve of the outer loop needed -- since
    in this synthetic study `x_demo` genuinely IS the forward solver's own
    converged output at `theta_star` (KKT-fit's best case, sigma~0 in
    `ioc/analytic.py`'s terms, not an approximation here).

    This only gives a genuinely-informative direction within the CERTIFICATE's
    own well-conditioned subspace (`ioc/analytic.py::kkt_fit`'s docstring:
    G's near-null eigenvalues mean "no method can recover theta there," which
    is exactly why `identifiability_check.py`/this module's own certificate
    exists) -- consistent with, not a workaround for, the zero-prior
    convention: projecting the raw KKT direction onto `V_top` (`eigvecs[:,
    selected_idx]`) automatically drops whatever component falls in the
    near-null direction, same as `theta_from_alpha_zero_prior` does for the
    fitted alpha itself.

    `kkt_fit`'s own softmax/simplex parameterization fixes an ARBITRARY scale
    (weights sum to 1) that has no relationship to `alpha`'s signed,
    calibrated-feature scale -- so only the projected DIRECTION is trusted
    from the KKT solve; magnitude is re-derived the same way the existing
    grid-search picks it (matching `calibrate_segment`'s ~0.1-0.3 target
    range), not taken from the simplex weights themselves.
    """
    scene = _make_scene(fit_scene_spec)
    scenes, _, x0_star, inner_by_phase, residual_fn_by_phase, scales_by_phase = _setup(
        prob, forward_solver, scene, seed=seed)
    xs_gt, ps_gt = _solve_all(prob, scenes, inner_by_phase, theta_trajopt_star, x0_star)

    phase = "transport"
    sc = jax.tree.map(lambda a: a[0], ps_gt[phase])
    residual_fn = residual_fn_by_phase[phase]
    x_demo = xs_gt[phase][0]
    scales = scales_by_phase[phase]

    cols = []
    for idx in range(K_FREE):
        def phi(x, idx=idx):
            return jnp.sum(residual_fn(x, sc)[idx] ** 2)
        g = jax.grad(phi)(x_demo) / scales[idx]
        cols.append(g)
    B = jnp.stack(cols, axis=-1)  # (dim_x, K_FREE), order = FREE_NAMES
    G = B.T @ B

    def resid(z):
        th = jax.nn.softmax(z)
        return th @ G @ th

    obj = jax.jit(jax.value_and_grad(resid))
    z_hat, _ = outer_opt.adam(obj, jnp.zeros(K_FREE, dtype=jnp.float32), lr=lr, n_steps=n_steps)
    theta_kkt_dir = jax.nn.softmax(z_hat)  # simplex weights: DIRECTION only, scale is arbitrary

    # Project onto the identified subspace (drops whatever the KKT direction
    # placed in the near-null eigendirection -- see docstring), then rescale
    # to the calibrated magnitude target, matching the existing grid-search's
    # convention so the two seeding strategies are directly comparable.
    V_top = eigvecs[:, selected_idx]  # (K_FREE, k)
    alpha_dir = V_top.T @ theta_kkt_dir
    alpha_dir_norm = alpha_dir / (jnp.linalg.norm(alpha_dir) + 1e-12)
    target_mag = sum(target_theta_mag) / 2.0
    alpha0 = alpha_dir_norm * target_mag
    return alpha0, dict(G=np.asarray(G), theta_kkt_dir=np.asarray(theta_kkt_dir),
                         z_hat=np.asarray(z_hat))


def multistart_fit(gf, k, target_theta_mag, n_starts=16, n_steps=40, lr=0.05, seed=0,
                    extra_starts=(), loss_rows=None, chunk=None):
    """Multistart Adam over the (now only k=2-3 dimensional) alpha subspace --
    the cheap thing to try BEFORE reaching for a derivative-free method: if
    Adam is only getting trapped by ONE bad local plateau/wall structure
    reached from a single start, sampling many starts and keeping the best
    should recover most of what CMA-ES would buy, at a fraction of the cost
    (still gradient-based per start, just not committed to one basin).

    `ioc.outer.adam` already returns `best_z` (the lowest-loss iterate seen
    during that run, not just the final one -- see its docstring), so each
    start's own within-run wandering (e.g. the transient wall-crossing spike
    seen in the single-KKT-seed trace) is already accounted for; multistart
    only adds BETWEEN-run diversity on top of that.

    `extra_starts` lets the caller fold in already-computed seeds (grid,
    KKT) as some of the candidates, so multistart's own random starts are
    compared against them directly rather than redundantly re-deriving them.

    Two modes:

    * `loss_rows=None` (default): the starts are fitted one at a time, each its
      own `ioc.outer.adam` call.  Unchanged behaviour, and the only option when
      the caller cannot supply a per-row forward map.
    * `loss_rows` given: `loss_rows(A) -> (C,)` scores a whole STACK of alpha
      candidates, and all starts are fitted as one batched program via
      `ioc.outer.adam_multi`.  Each start's loss depends only on its own row, so
      one gradient of the summed loss is exactly every start's own gradient (see
      `adam_multi`).  This is the flattening `iosp.fit.multistart` uses for its
      (branch x seed) candidates, applied to the alpha subspace.

    `chunk` splits the candidate axis into groups of that many, evaluated one
    group at a time via `jax.lax.map` -- sequential on device, so only one
    group's intermediates are live at a time, but with no host round-trip
    between groups.  It changes no value (row independence again), and every
    group has the same shape so it costs one compilation.  Use it when the
    batched map does not fit: this path solves four trajopt segments per row,
    and peak memory is linear in the number of rows.  `chunk=None` keeps the
    whole batch in one call.
    """
    key = jax.random.PRNGKey(seed)
    starts = list(extra_starts)
    n_random = max(n_starts - len(extra_starts), 0)
    keys = jax.random.split(key, max(n_random, 1))
    for i in range(n_random):
        kdir, kmag = jax.random.split(keys[i])
        v = jax.random.normal(kdir, (k,))
        v = v / (jnp.linalg.norm(v) + 1e-8)
        mag = jax.random.uniform(kmag, (), minval=target_theta_mag[0], maxval=target_theta_mag[1])
        starts.append(v * mag)
    starts = [jnp.asarray(s, dtype=jnp.float32) for s in starts]

    if loss_rows is None:
        results = []
        for s in starts:
            a_hat, trace = outer_opt.adam(gf, s, lr=lr, n_steps=n_steps)
            best_val = min(v for _, v in trace) if trace else float("inf")
            results.append(dict(alpha0=np.asarray(s), alpha_hat=np.asarray(a_hat),
                                 best_val=best_val, trace=[float(v) for _, v in trace]))
            print(f"  [multistart] alpha0={np.asarray(s)}  best_loss={best_val:.5f}  "
                  f"alpha_hat={np.asarray(a_hat)}")
        return min(results, key=lambda r: r["best_val"]), results

    A0 = jnp.stack(starts)                                   # (C, k)
    C = A0.shape[0]
    rows = loss_rows if chunk is None else _chunked_rows(loss_rows, C, int(chunk))
    A_hat, traces = outer_opt.adam_multi(
        outer_opt.summed_grad_fn(rows), A0, lr=lr, n_steps=n_steps)
    results = []
    for i in range(C):
        best_val = min(v for _, v in traces[i]) if traces[i] else float("inf")
        results.append(dict(alpha0=np.asarray(A0[i]), alpha_hat=np.asarray(A_hat[i]),
                             best_val=best_val,
                             trace=[float(v) for _, v in traces[i]]))
        print(f"  [multistart] alpha0={np.asarray(A0[i])}  best_loss={best_val:.5f}  "
              f"alpha_hat={np.asarray(A_hat[i])}")
    return min(results, key=lambda r: r["best_val"]), results


def _chunked_rows(loss_rows, n_rows, chunk):
    """`loss_rows` applied one group of `chunk` rows at a time, via `lax.map`.

    `lax.map` rather than a Python list comprehension: it is sequential on
    device, so peak memory is one group's worth (which is the point) WITHOUT a
    host round-trip per group, and it keeps the whole thing inside one traced
    program so the caller can still differentiate straight through it.
    """
    if n_rows % chunk:
        raise ValueError(f"chunk={chunk} must divide the candidate count {n_rows}")

    def rows(A):
        out = jax.lax.map(loss_rows, A.reshape(n_rows // chunk, chunk, A.shape[-1]))
        return out.reshape(n_rows)

    return rows


def run_canonical(fit_scene_spec, held_out_specs=None, certificate_scene_specs=None,
                   prob=None, forward_solver=None, seed=0, n_steps=12,
                   cumulative_threshold=0.95, theta_transport_star_override=None,
                   alpha0_mode="grid", n_starts=16, nonneg_theta=False,
                   multistart_batched=False, multistart_chunk=None):
    """THE canonical Study 1/2 procedure: certificate -> rank selection (95%
    cumulative-trace rule) -> zero-prior subspace fit -> held-out
    generalization vs. the alpha=0 ("no fitting at all") baseline.

    Generic over which scenes are used for the certificate, the fit, and
    generalization -- this is what lets Study 2 reuse this exact function
    for every demo regime instead of reimplementing the procedure.

    `multistart_batched` opts the `alpha0_mode="multistart"` path into the
    flattened forward map (`loss_rows` below), fitting every start as one
    batched program instead of one at a time.  DEFAULT OFF, and the measurement
    is why -- 8 starts x 10 steps on a curated scene:

        serial        456s   per-start best = [1.69e-2 4.64e-4 3.06e-2 4.35e0 ...]
        flattened     334s   per-start best = [4.25e-1 3.80e-4 3.01e-2 7.79e1 ...]
        flat chunk=2  291s   per-start best = [4.28e-1 1.20e-4 1.46e-2 2.54e0 ...]

    Only 1.4x -- this fit's batch is ONE scene, but each row still runs four
    trajopt segments, so it is not the idle-GPU regime the trick pays off in
    (contrast `ioc.bench2d`, where the same change measured 4.0x end to end).
    And the per-start losses genuinely MOVE: folding candidates into the scene
    batch changes each row's position in the IK kernel's problem batch, and
    `pickplace._ik_batch`'s docstring records that a row "can land on a
    different IK branch depending on how it was batched".  The winning start's
    loss is comparable either way (1.0e-4 / 9.1e-5 / 4.3e-5), so this is a
    different-but-valid search, not a worse one -- but it is not the same run,
    so it must not silently replace results already recorded serially.

    `multistart_chunk` bounds the batched path's memory; note above that
    chunking was FASTER than the full batch here, consistent with the
    autotuning pressure `iosp.fit.multistart.run` documents.
    """
    if prob is None:
        prob = pp.PickPlaceProblem.load(str(URDF_PATH), str(SRDF_PATH), str(MESH_DIR))
    if forward_solver is None:
        forward_solver = pp.make_composed_forward_solver(n_iters=60)
    certificate_scene_specs = CURATED_SCENES if certificate_scene_specs is None else certificate_scene_specs
    held_out_specs = HELD_OUT_SCENES if held_out_specs is None else held_out_specs

    cert = run_certificate(seed=seed, whiten=True, scene_specs=certificate_scene_specs,
                            prob=prob, forward_solver=forward_solver)
    k, selected_idx = select_rank(cert["eigvals"], cumulative_threshold)

    fit_scene = _make_scene(fit_scene_spec)
    scenes, theta_trajopt_star, x0_star, inner_by_phase, residual_fn_by_phase, scales_by_phase = _setup(
        prob, forward_solver, fit_scene, seed=seed)
    theta_transport_star = (
        theta_transport_star_override if theta_transport_star_override is not None
        else theta_trajopt_star[_TRANSPORT_IDX]
    )
    theta_trajopt_star = theta_trajopt_star.at[_TRANSPORT_IDX].set(theta_transport_star)

    xs_gt, ps_gt = _solve_all(prob, scenes, inner_by_phase, theta_trajopt_star, x0_star)
    # Reconstruction target is JOINT CONFIGURATION, not EE pose -- see
    # `_full_joint_path`'s docstring: the demo is given as joint
    # configurations, and an EE-space loss can't distinguish "recovered the
    # right cost" from "found a different redundant-arm solution tracing the
    # same path," which is a tractability problem, not a real ambiguity in
    # what's being asked to reconstruct.
    demo_path = _full_joint_path(prob, xs_gt, ps_gt, batch_index=0)

    def loss(alpha):
        theta_transport = theta_from_alpha_zero_prior(alpha, cert["eigvecs"], selected_idx, nonneg=nonneg_theta)
        theta_trajopt = theta_trajopt_star.at[_TRANSPORT_IDX].set(theta_transport)
        xs, ps = _solve_all(prob, scenes, inner_by_phase, theta_trajopt, x0_star)
        path = _full_joint_path(prob, xs, ps, batch_index=0)
        return jnp.mean(jnp.sum((path - demo_path) ** 2, axis=-1))

    def loss_rows(A):
        """`loss`, for a whole STACK of alpha candidates: (C, k) -> (C,).

        The flattening.  This fit has a ONE-scene batch, so the batched program
        the scalar `loss` builds leaves the device almost entirely idle; folding
        the candidate axis into that batch fills it.  Each row keeps its own
        cost via `solve_batched_theta`'s per-row `in_axes`, so row c's loss
        depends only on `A[c]` -- which is what lets `multistart_fit` take one
        gradient of the summed loss and get every start's own gradient.

        `scenes` has a leading axis of 1 here, so replicating it to C rows is a
        `jnp.repeat`; if this fit ever carries several scenes, the rows become
        (candidate, scene) pairs and the reshape below has to fold both.
        """
        C = A.shape[0]
        tt = jax.vmap(lambda a: theta_from_alpha_zero_prior(
            a, cert["eigvecs"], selected_idx, nonneg=nonneg_theta))(A)     # (C, 3)
        theta_traj = jax.vmap(
            lambda t: theta_trajopt_star.at[_TRANSPORT_IDX].set(t))(tt)    # (C, 7)
        by_phase = _split_trajopt_batched(theta_traj)
        rep = lambda a: jnp.repeat(a, C, axis=0)
        sc_c = jax.tree.map(rep, scenes)
        x0_c = {p: rep(x0_star[p]) for p in pp.PHASES}
        _, _, xs, ps = prob.solve_batched_theta(
            THETA_IK_STAR, by_phase, sc_c, inner_by_phase, x0_c)
        paths = _full_joint_paths(prob, xs, ps)                            # (C, T, dof)
        return jnp.mean(jnp.sum((paths - demo_path) ** 2, axis=-1), axis=-1)

    # Two SEPARATE things were conflated in the first canonical attempt:
    # (1) the null (unselected) eigendirections are pinned at exactly zero --
    #     that IS the honest identifiability convention, unchanged here.
    # (2) the FITTED alpha's optimizer STARTING POINT is a purely numerical
    #     choice, independent of (1).  Starting it at (near-)zero lands
    #     theta_transport in/near the degenerate flat-cost corner (MEASURED:
    #     at theta=[0,0,0], `dot(theta,features)` is identically zero for
    #     every x, so the forward solve never leaves its seed and the
    #     adjoint's Hessian is exactly zero there, rescued only by
    #     `adjoint_ridge`) -- a bad initialization, not a property of (1).
    # Fixed by scaling alpha0 so theta_transport's magnitude lands near what
    # `calibrate_segment` actually calibrated against (~0.1-0.3), verified
    # directly below via gradient-norm probes at a few candidate scales
    # BEFORE trusting Adam to move, rather than assumed.
    gf = jax.jit(jax.value_and_grad(loss))
    # `ioc.outer.cma_es` calls its `loss` argument once per population member
    # per generation, RAW (not jitted) -- through the full IK->4-segment
    # composed chain each time. MEASURED (`scratch/logs/profile_compile.log`):
    # even a SECOND un-jitted call with identical shapes costs ~32s (eager
    # per-primitive dispatch overhead through ~6 chained custom_vjp/custom_jvp
    # ops, not a compile cost -- a jitted call is a single cached executable
    # instead). At lam~6, n_gens=30 that's ~180 raw calls -- 90-150 MINUTES on
    # this alone, which is what actually made the earlier CMA-ES run look
    # stuck (compounded by Python fully buffering stdout when redirected to a
    # file, so the certificate section can be computed and just not flushed
    # yet). Jitting once here turns population evaluation into cheap repeated
    # dispatch of one cached executable.
    loss_jit = jax.jit(loss)
    t0 = time.perf_counter()
    _, g_probe0 = gf(jnp.zeros(k, dtype=jnp.float32))
    compile_s = time.perf_counter() - t0
    print(f"  [grad probe] alpha=0:      |grad|={float(jnp.linalg.norm(g_probe0)):.6e}  (expect ~0, degenerate corner)")

    ALPHA0_CANDIDATES = (0.001, 0.05, 0.2)
    # Target ||theta_transport|| range calibrate_segment actually calibrated
    # against -- see comment below on why we select by proximity to this
    # range rather than by raw gradient magnitude.
    TARGET_THETA_MAG = (0.1, 0.3)
    grad_norms = {}
    theta_mags = {}
    for cand in ALPHA0_CANDIDATES:
        _, g_probe = gf(cand * jnp.ones(k, dtype=jnp.float32))
        gn = float(jnp.linalg.norm(g_probe))
        grad_norms[cand] = gn
        theta_mag = float(jnp.linalg.norm(theta_from_alpha_zero_prior(
            cand * jnp.ones(k, dtype=jnp.float32), cert["eigvecs"], selected_idx, nonneg=nonneg_theta)))
        theta_mags[cand] = theta_mag
        print(f"  [grad probe] alpha0={cand:.3f}  |grad|={gn:.6e}  ||theta_transport||={theta_mag:.4f}")

    # BUGFIX (was: `max(grad_norms, key=grad_norms.get)`): near the
    # degenerate alpha=0 corner the gradient is not small, it's ILL-
    # CONDITIONED and explodes (measured |grad|~1e7 at alpha=0), then falls
    # off monotonically as alpha0 grows -- so picking the raw-max-gradient
    # candidate always selected the SMALLEST candidate (0.001), landing back
    # in/near the same degenerate region this probe exists to avoid (that
    # bug reproduced exactly: alpha_hat stayed pinned near alpha0=0.001,
    # zero real fit, in the first canonical run). The comment directly above
    # this code always said the real selection target: pick the candidate
    # whose ||theta_transport|| lands closest to the calibrated 0.1-0.3
    # range -- implement that literally instead of relying on gradient
    # magnitude, which is not monotonic in identifiability, only in
    # closeness to the degenerate corner.
    target_center = sum(TARGET_THETA_MAG) / 2.0

    kkt_info = None
    if alpha0_mode == "grid":
        alpha0_scale = min(theta_mags, key=lambda c: abs(theta_mags[c] - target_center))
        alpha0 = alpha0_scale * jnp.ones(k, dtype=jnp.float32)
        print(f"  [grad probe] selected alpha0_scale={alpha0_scale} "
              f"(||theta_transport||={theta_mags[alpha0_scale]:.4f}, target~{target_center:.2f})")
    elif alpha0_mode == "kkt":
        alpha0, kkt_info = kkt_seed_alpha(
            prob, forward_solver, fit_scene_spec, theta_trajopt_star,
            cert["eigvecs"], selected_idx, target_theta_mag=TARGET_THETA_MAG, seed=seed)
        print(f"  [kkt seed] alpha0={np.asarray(alpha0)}  "
              f"theta_kkt_dir(simplex)={kkt_info['theta_kkt_dir']}")
    elif alpha0_mode in ("multistart", "cma"):
        # Fold in the grid and KKT seeds as informed candidates alongside
        # the random ones, so multistart is a strict superset of the other
        # two strategies rather than a separate, incomparable run.
        grid_scale = min(theta_mags, key=lambda c: abs(theta_mags[c] - target_center))
        grid_alpha0 = grid_scale * jnp.ones(k, dtype=jnp.float32)
        kkt_alpha0, kkt_info = kkt_seed_alpha(
            prob, forward_solver, fit_scene_spec, theta_trajopt_star,
            cert["eigvecs"], selected_idx, target_theta_mag=TARGET_THETA_MAG, seed=seed)
        alpha0 = grid_alpha0  # for logging only; the real result is alpha_hat below
    else:
        raise ValueError(f"unknown alpha0_mode={alpha0_mode!r}")

    t0 = time.perf_counter()
    if alpha0_mode == "multistart":
        best, ms_results = multistart_fit(
            gf, k, TARGET_THETA_MAG, n_starts=n_starts, n_steps=n_steps, lr=0.05, seed=seed,
            extra_starts=(grid_alpha0, kkt_alpha0),
            loss_rows=loss_rows if multistart_batched else None,
            chunk=multistart_chunk)
        alpha_hat = jnp.asarray(best["alpha_hat"], dtype=jnp.float32)
        adam_trace = [(0, v) for v in best["trace"]]
        # movement must be measured against the start alpha_hat ACTUALLY came
        # from, not the grid seed that is only logged
        alpha0 = jnp.asarray(best["alpha0"], dtype=jnp.float32)
        print(f"  [multistart] BEST start alpha0={best['alpha0']}  best_loss={best['best_val']:.5f}")
    elif alpha0_mode == "cma":
        # Population-based, derivative-free -- doesn't commit to one basin's
        # local gradient at all, unlike multistart (which is still N
        # independent local searches). sigma0 set to span the calibrated
        # target range so the initial population covers it, not just probes
        # near one seed.
        sigma0 = (TARGET_THETA_MAG[1] - TARGET_THETA_MAG[0])
        cma_alpha_hat, cma_trace = outer_opt.cma_es(loss_jit, grid_alpha0, sigma0=sigma0, seed=seed, n_gens=n_steps)
        alpha_hat = jnp.asarray(cma_alpha_hat, dtype=jnp.float32)
        adam_trace = cma_trace
        ms_results = None
        print(f"  [cma-es] alpha_hat={np.asarray(alpha_hat)}  final_loss={cma_trace[-1][1]:.5f}")
    else:
        alpha_hat, adam_trace = outer_opt.adam(gf, alpha0, lr=0.05, n_steps=n_steps, trace_best=False)
        ms_results = None
    fit_s = time.perf_counter() - t0
    print(f"  [adam trace] loss per step: {[round(v, 5) for _, v in adam_trace]}")

    # G1 gate: is this a real fit, or did alpha_hat just stay where it
    # started?  Twice now this procedure has reported an `alpha_hat` exactly
    # equal to its own initialization and been read as a result.  Check it
    # explicitly and print the verdict next to the number, rather than
    # leaving it to be eyeballed off the trace.
    a0_np, ah_np = np.asarray(alpha0, dtype=float), np.asarray(alpha_hat, dtype=float)
    move_abs = float(np.linalg.norm(ah_np - a0_np))
    move_rel = move_abs / (float(np.linalg.norm(a0_np)) + 1e-30)
    loss0, lossN = float(adam_trace[0][1]), float(min(v for _, v in adam_trace))
    loss_red = loss0 / max(lossN, 1e-30)
    degenerate = (not np.isfinite(move_rel)) or move_rel < 1e-3 or loss_red < 1.05
    print(f"  [G1 non-degeneracy] ||alpha_hat - alpha0||={move_abs:.6e} "
          f"(rel {move_rel:.3e});  loss {loss0:.5f} -> {lossN:.5f} "
          f"(reduction {loss_red:.2f}x)")
    print(f"  [G1 non-degeneracy] {'DEGENERATE -- alpha_hat did not move off its start; NOT a result' if degenerate else 'OK -- alpha_hat moved and loss improved'}")

    theta_transport_hat = theta_from_alpha_zero_prior(alpha_hat, cert["eigvecs"], selected_idx, nonneg=nonneg_theta)
    theta_trajopt_hat = theta_trajopt_star.at[_TRANSPORT_IDX].set(theta_transport_hat)
    theta_trajopt_nofit = theta_trajopt_star.at[_TRANSPORT_IDX].set(jnp.zeros(3, dtype=jnp.float32))

    param_err_hat = float(jnp.linalg.norm(theta_transport_hat - theta_transport_star))
    param_err_nofit = float(jnp.linalg.norm(theta_transport_star))  # ||0 - theta_star||
    fit_rmse = float(jnp.sqrt(loss(alpha_hat)))

    gen_rmse_hat, gen_rmse_nofit = {}, {}
    for gen_name, spec in held_out_specs.items():
        gen_scene = _make_scene(spec)
        scenes_gen = jax.tree.map(lambda a: a[None], gen_scene)
        x0_gen, _, _, _ = prob.seeds(scenes_gen, THETA_IK_STAR)

        # jitted: 3 calls per held-out scene (star/hat/nofit) all share the
        # same shapes, so this is one compile + 3 cheap dispatches instead of
        # 3 separate ~30-50s eager passes through the composed chain.
        @jax.jit
        def rollout(theta_trajopt, scenes_gen=scenes_gen, x0_gen=x0_gen):
            xs, ps = _solve_all(prob, scenes_gen, inner_by_phase, theta_trajopt, x0_gen)
            return _full_joint_path(prob, xs, ps, batch_index=0)

        path_star_gen = rollout(theta_trajopt_star)
        path_hat_gen = rollout(theta_trajopt_hat)
        path_nofit_gen = rollout(theta_trajopt_nofit)
        gen_rmse_hat[gen_name] = float(jnp.sqrt(jnp.mean(jnp.sum((path_star_gen - path_hat_gen) ** 2, axis=-1))))
        gen_rmse_nofit[gen_name] = float(jnp.sqrt(jnp.mean(jnp.sum((path_star_gen - path_nofit_gen) ** 2, axis=-1))))

    return dict(
        eigvals=cert["eigvals"], k=k, selected_idx=selected_idx,
        theta_transport_star=np.asarray(theta_transport_star),
        theta_transport_hat=np.asarray(theta_transport_hat),
        alpha_hat=np.asarray(alpha_hat), alpha0=a0_np, alpha0_mode=alpha0_mode,
        degenerate=bool(degenerate), move_rel=move_rel, loss_reduction=loss_red,
        kkt_info=kkt_info,
        adam_trace=[float(v) for _, v in adam_trace], ms_results=ms_results,
        param_err_hat=param_err_hat, param_err_nofit=param_err_nofit,
        fit_rmse=fit_rmse, gen_rmse_hat=gen_rmse_hat, gen_rmse_nofit=gen_rmse_nofit,
        compile_s=compile_s, fit_s=fit_s,
        prob=prob, forward_solver=forward_solver,  # returned for reuse by callers (e.g. Study 2)
    )


def _print_canonical(r):
    print(f"selected rank k = {r['k']}  (eigenvalues: {r['eigvals']})  alpha0_mode={r['alpha0_mode']}")
    print(f"theta_transport_star = {r['theta_transport_star']}")
    print(f"theta_transport_hat  = {r['theta_transport_hat']}  (alpha_hat={r['alpha_hat']})")
    print()
    print(f"param_err, no-fit baseline (alpha=0, i.e. theta=[0,0,0]): {r['param_err_nofit']:.4f}")
    print(f"param_err, after fitting:                                {r['param_err_hat']:.4f}")
    print(f"fit RMSE, JOINT SPACE (reconstruction of the FIT-scene demo):  {r['fit_rmse']:.4f}")
    print()
    print("held-out generalization RMSE, JOINT SPACE (fitted vs no-fit baseline):")
    for gen_name in r["gen_rmse_hat"]:
        print(f"  [{gen_name}]  fitted={r['gen_rmse_hat'][gen_name]:.4f}   "
              f"no-fit={r['gen_rmse_nofit'][gen_name]:.4f}")


# Deliberate demonstration theta_star: comparable magnitude along the null
# eigendirection AND the top-2 identifiable ones (same search as the earlier
# "non-vacuous" fix, same coefficients -- kept here as the CANONICAL
# demonstration case, clearly separate from the null result above, which
# stays documented as its own real data point, not superseded by this one).
_DEMO_C0, _DEMO_A1, _DEMO_A2 = -0.20, 0.10, 0.03

if __name__ == "__main__":
    cert = run_certificate()
    _print_certificate(cert)

    V = cert["eigvecs"]
    theta_star_demo = jnp.asarray(
        _DEMO_C0 * V[:, 0] + _DEMO_A1 * V[:, 1] + _DEMO_A2 * V[:, 2], dtype=jnp.float32)

    # Fit is cheap per step relative to the one-time forward-solver compile
    # (measured: 12-step fit was ~4.2-4.5s total) -- n_steps=12 was chosen
    # only to confirm movement off the degenerate corner, not because 12
    # steps was expected to be enough to actually converge.  Now that KKT
    # seeding gives a non-degenerate, direction-informed start, the open
    # question is convergence, not corner-escape -- so run substantially
    # longer and log the per-step loss trace to see it directly instead of
    # inferring under- vs. non-convergence from before/after numbers alone.
    # Sampling more, cheapest-first: multistart Adam over the (only 2-3 dim)
    # alpha subspace before reaching for a full derivative-free method -- if
    # single-start Adam was just getting unlucky about which plateau/basin
    # it fell into (not a landscape that defeats ALL local search), enough
    # starts should find a meaningfully lower basin. CMA-ES is the fallback
    # if multistart ALSO plateaus near the no-fit baseline -- that would
    # point at a landscape hard for local methods generally, not bad luck.
    # nonneg_theta=True: the grid/kkt/multistart/cma-es comparison found
    # EVERY method (including CMA-ES, which doesn't commit to a local basin)
    # converging to the same ~0.44-0.46 loss floor, worse than the no-fit
    # baseline, with every fitted theta_transport carrying NEGATIVE
    # components -- a sign no raw softmax-derived theta_star has. A negative
    # weight on a squared residual makes the INNER trajopt cost unbounded
    # below along that residual, which is consistent with the instability
    # seen (loss spikes to 9-140 mid-optimization) and with the
    # optimizer-independent shared floor. Testing whether constraining
    # theta_transport to stay nonnegative (`_nonneg_transport`) removes that
    # floor.
    # G1 requires BOTH theta_stars, not just the demonstration one:
    #   - "natural": theta_transport_star as `_setup` calibrates it (no
    #     override) -- the case that produced the documented null result.
    #   - "demo": the deliberate theta_star with comparable mass on the null
    #     AND top-2 identifiable directions.
    # Same procedure, same gates, both reported, so the null result stays a
    # real data point rather than being quietly replaced by the easier case.
    # `multistart` only: the grid/kkt/multistart/cma-es comparison already
    # found every method converging to the same loss floor, and multistart
    # folds the grid and KKT seeds in as candidates, so it is a strict
    # superset of the two cheaper strategies.
    for star_name, override in (("natural", None), ("demo", theta_star_demo)):
        print()
        print("=" * 60)
        print(f"CANONICAL PROCEDURE  theta_star={star_name!r}  "
              f"alpha0_mode='multistart', nonneg_theta=True, n_steps=40, n_starts=16")
        if override is not None:
            print("theta_star_demo =", np.asarray(override))
        print("=" * 60)
        r = run_canonical(
            CURATED_SCENES["clear"], prob=cert["prob"], forward_solver=cert["forward_solver"],
            theta_transport_star_override=override, alpha0_mode="multistart",
            nonneg_theta=True, n_steps=40, n_starts=16)
        _print_canonical(r)
        print(f"fit_s={r['fit_s']:.2f}s")
        print(f"G1 VERDICT [{star_name}]: "
              f"{'DEGENERATE (not a result)' if r['degenerate'] else 'non-degenerate'}"
              f"  move_rel={r['move_rel']:.3e}  loss_reduction={r['loss_reduction']:.2f}x")
