"""The five-stage identifiable-refit procedure, and the gate that has to pass
before its output means anything.

The procedure is `THEORY_IDENTIFIABLE_REFIT.md` executed literally:

    1  fit wide          -- all K weights free
    2  sensitivity Gram  -- G = J^T J at the wide fit, J = d(path)/du
    3  select r          -- `ioc.identifiability.select_rank`
    4  refit on U_r      -- null component pinned at the prior
    5  report in the U_r projection, plus a held-out scene

It is path-agnostic: it takes a `built` dict from `iosp.fit.parametric` (or
from the shelved RKHS builder) and knows nothing about pick-and-place.

`check_loss_rmse_consistency` is the G0 gate.  `loss_a(u)` and `rmse_a(u)**2`
are the same quantity by construction, computed through separately jitted
graphs; on one configuration they disagreed by 340x at the fitted point, which
is what made a wide fit report WORSE than its own initialization.  Stationarity
screening does not catch this -- gate on the identity itself.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from ioc import identifiability as ident
from iosp import config
from iosp.fit.params import _proj_norm, gauge_fix, gauge_vector

N_STEPS, LR, TRACE_FRAC = config.N_STEPS, config.LR, config.TRACE_FRAC

# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# G0 gate -- is the forward map a stable function of `u` at all?
# ---------------------------------------------------------------------------

def check_loss_rmse_consistency(built, n_probe=3, tol=1e-2, seed=0):
    """`loss_a(u)` and `rmse_a(u)**2` are the SAME quantity by construction --
    a mean over per-timestep squared EE displacement on the fit scenes.  They
    are computed through separately-jitted graphs (`gf` vs `paths_j`), and on
    the aligned-M=3 config those two graphs disagreed 1% at u0 and 340x at the
    fitted point, which is what made `wide fit` report WORSE than `init`.

    Stationarity at 1e-3 does NOT catch this.  So gate on the identity itself,
    at u0 and at a few ||u|| ~ 1 probes.  Returns the worst relative gap.
    """
    gf, K = built["gf"], built["K"]
    rng = np.random.default_rng(seed)
    probes = [jnp.zeros(K, dtype=jnp.float32)]
    for _ in range(n_probe):
        v = rng.normal(size=K)
        probes.append(jnp.asarray(v / np.linalg.norm(v), dtype=jnp.float32))

    worst = 0.0
    print(f"  [{built.get('label', '?')}] G0 loss-vs-rmse^2 consistency:", flush=True)
    for i, u in enumerate(probes):
        loss = float(gf(u)[0])
        r2 = float(built["rmse_a"](u)) ** 2
        rel = abs(loss - r2) / max(abs(loss), 1e-30)
        worst = float("nan") if not np.isfinite(rel) else (
            max(worst, rel) if np.isfinite(worst) else worst)
        tag = "u0" if i == 0 else f"||u||=1 #{i}"
        flag = ("  <-- NaN" if not np.isfinite(rel)
                else "  <-- INCONSISTENT" if rel > tol else "")
        print(f"    {tag:12s} loss={loss:.6e}  rmse^2={r2:.6e}  rel={rel:.3e}{flag}",
              flush=True)
    if not np.isfinite(worst) or worst > tol:
        print(f"  [{built.get('label', '?')}] G0 FAIL: worst rel gap {worst:.3e} > "
              f"{tol:.0e} -- every number below this line is untrustworthy",
              flush=True)
    else:
        print(f"  [{built.get('label', '?')}] G0 PASS (worst rel gap {worst:.3e})",
              flush=True)
    return worst


def run_procedure(built, label, n_steps=N_STEPS, lr=LR):
    """Stages 1-5 for either path."""
    gf, K, n_ik = built["gf"], built["K"], built["n_ik"]
    u0 = jnp.zeros(K, dtype=jnp.float32)

    t0 = time.perf_counter()
    jax.block_until_ready(gf(u0))
    print(f"  [{label}] compile {time.perf_counter() - t0:.1f}s", flush=True)

    g0 = check_loss_rmse_consistency({**built, "label": label})

    # -- stage 1: fit wide --------------------------------------------------
    t0 = time.perf_counter()
    u_wide, _ = ident.wide_fit(gf, u0, lr=lr, n_steps=n_steps)
    t_wide = time.perf_counter() - t0

    # -- stages 2-3: sensitivity spectrum at the wide fit, rank by trace rule
    t0 = time.perf_counter()
    eigvals, eigvecs = ident.sensitivity_spectrum(built["jac_fn"], u_wide)
    # G4: `ident.select_rank` defaults to rule="gap"; `frac` is used ONLY by
    # rule="trace" and is silently ignored here.  The log line used to claim
    # ">= 95% cumulative trace" regardless, which made ranks from before the
    # default changed look comparable to today's when they are not (same
    # spectrum -> trace r=2, gap r=6).  Name the rule actually used.
    RANK_RULE = "gap"
    top, null, r = ident.select_rank(eigvals, frac=TRACE_FRAC, rule=RANK_RULE)
    t_gram = time.perf_counter() - t0
    print(f"  [{label}] jac+svd {t_gram:.1f}s; eigenvalues (desc) = "
          f"{np.array2string(eigvals, precision=4, max_line_width=100)}", flush=True)
    rule_desc = (f">= {TRACE_FRAC:.0%} cumulative trace" if RANK_RULE == "trace"
                 else "largest consecutive-eigenvalue ratio")
    print(f"  [{label}] r = {r} of K = {K} (rule={RANK_RULE!r}: {rule_desc})", flush=True)

    # confound 1's check: is `U_r` just the badly-scaled coordinates?
    ident.report_loadings(eigvals, eigvecs, built["names"], n_show=min(r + 2, K))
    g_hat = gauge_vector(K, n_ik)
    print(f"  [{label}] |<u_1, gauge>| = {abs(eigvecs[:, 0] @ g_hat):.4f}; "
          f"gauge lands at eigen-index {int(np.argmax(np.abs(eigvecs.T @ g_hat)))} "
          f"(lam={eigvals[int(np.argmax(np.abs(eigvecs.T @ g_hat)))]:.3e})", flush=True)

    # -- stage 4: refit on span(U_r), null pinned at the prior ---------------
    t0 = time.perf_counter()
    u_refit, _ = ident.refit_on_subspace(gf, u0, eigvecs[:, top], n_steps=n_steps, lr=lr)
    t_refit = time.perf_counter() - t0

    # -- stage 5: report ----------------------------------------------------
    out = dict(label=label, r=r, K=K, eigvals=eigvals, g0_rel=g0,
               t_wide=t_wide, t_gram=t_gram, t_refit=t_refit)
    for name, u in (("wide", u_wide), ("refit", u_refit)):
        # The fitted `u` itself, not just its `theta`: a caller that wants to
        # score the fit on a SECOND criterion (E10 reports EE RMSE alongside a
        # joint-space fit) needs the vector, and re-deriving it from `theta`
        # means inverting a softmax.
        out[f"u_{name}"] = np.asarray(u)
        out[f"fit_rmse_{name}"] = built["rmse_a"](u)
        out[f"gen_rmse_{name}"] = built["rmse_b"](u)
        out[f"theta_{name}"] = built["theta_of"](u)
    out["fit_rmse_init"] = built["rmse_a"](u0)

    if built["theta_star"] is not None:
        # CONFOUND 2: every parameter-space metric is computed on GAUGE-FIXED
        # vectors, since `u` and `u + c*gauge` are the same cost and a raw
        # difference between them is not a well-defined quantity.
        Ur, Un = eigvecs[:, top], eigvecs[:, null]
        u_star_g = gauge_fix(built["u_star"], n_ik)
        for name, u in (("wide", u_wide), ("refit", u_refit)):
            d = gauge_fix(u, n_ik) - u_star_g
            out[f"z_err_{name}"] = float(np.linalg.norm(d))
            out[f"z_err_top_{name}"] = _proj_norm(d, Ur)
            out[f"z_err_null_{name}"] = _proj_norm(d, Un)
            out[f"param_err_{name}"] = float(
                np.linalg.norm(out[f"theta_{name}"] - built["theta_star"]))
        d_star = u_star_g - gauge_fix(u0, n_ik)
        out["captured_frac"] = _proj_norm(d_star, Ur) / (float(np.linalg.norm(d_star)) + 1e-30)
    return out


def _report(res):
    L = res["label"]
    print()
    print(f"=== {L}: r = {res['r']} of K = {res['K']} identifiable directions ===")
    print(f"{'':22s} {'fit RMSE':>13s} {'gen RMSE':>13s}")
    print(f"{'init (u=0)':22s} {res['fit_rmse_init']:13.4f} {'-':>13s}")
    print(f"{'wide fit (all K)':22s} {res['fit_rmse_wide']:13.4f} {res['gen_rmse_wide']:13.4f}")
    print(f"{'refit on U_r':22s} {res['fit_rmse_refit']:13.4f} {res['gen_rmse_refit']:13.4f}")
    if "param_err_wide" in res:
        print()
        print(f"{'':22s} {'||dtheta||':>11s} {'||dz||':>9s} {'top-r':>9s} {'null':>9s}")
        for n in ("wide", "refit"):
            print(f"{n:22s} {res['param_err_'+n]:11.4f} {res['z_err_'+n]:9.4f} "
                  f"{res['z_err_top_'+n]:9.4f} {res['z_err_null_'+n]:9.4f}")
        print()
        print(f"fraction of (z_star - z_prior) inside span(U_r): {res['captured_frac']:.3f}"
              "   <- the generalization ceiling this demo supports")
    g0 = res.get("g0_rel")
    if g0 is not None:
        ok = np.isfinite(g0) and g0 <= 1e-2
        print(f"\nG0 forward-map consistency: {'PASS' if ok else 'FAIL'} "
              f"(worst |loss - rmse^2|/loss = {g0:.3e})"
              + ("" if ok else "   <- numbers above are NOT trustworthy"))
    print(f"\nwall: wide {res['t_wide']:.1f}s, gram {res['t_gram']:.1f}s, "
          f"refit {res['t_refit']:.1f}s", flush=True)


