"""Gate: is the composed rollout a STABLE function of `u`?

`loss_a(u)` and `rmse_a(u)**2` are the same quantity written twice, but they go
through separately-jitted graphs (`gf` vs `paths_j`).  If the inner solve is
converged they agree to float32 noise; if it is not, the rollout depends on the
numerical path rather than on `u`, and the fit's reported loss is fiction.

Measured on aligned M=3: 1% disagreement at u0, blowing up to 340x (in loss) at
the fitted point -- which is what made `wide fit` look worse than `init`.

Checked at u0 and at random u with ||u|| ~ 1.26, the distance the M=3 fit
actually travelled.
"""
import sys, pathlib
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from iosp.study3_identifiable_refit import build_same_demo

built_a, built_b = build_same_demo(M=64, ls=3.0)

for b in (built_a, built_b):
    K = b["K"]
    print(f"\n=== {b['label']}  (K={K}) ===", flush=True)
    key = jax.random.PRNGKey(0)
    pts = [("u0", jnp.zeros(K, dtype=jnp.float32))]
    for i in range(3):
        key, k = jax.random.split(key)
        v = jax.random.normal(k, (K,), dtype=jnp.float32)
        pts.append((f"rand{i} (|u|=1.26)", 1.26 * v / jnp.linalg.norm(v)))
    worst = 0.0
    for name, u in pts:
        val = float(b["gf"](u)[0])
        r2 = b["rmse_a"](u) ** 2
        rel = abs(val - r2) / max(abs(val), 1e-30)
        worst = max(worst, rel)
        print(f"  {name:18s} loss={val:.6e}  rmse^2={r2:.6e}  rel_diff={rel:.3e}",
              flush=True)
    verdict = "STABLE" if worst < 1e-2 else "UNSTABLE -- reported RMSE is unreliable"
    print(f"  worst rel_diff = {worst:.3e}  -> {verdict}", flush=True)
