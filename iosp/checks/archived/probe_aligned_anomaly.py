"""`wide fit 0.3250` > `init 0.0503` should be IMPOSSIBLE.

`ioc/outer.py::adam` returns `best_z`, the argmin over visited iterates tracked
on the loss, and it evaluates `z0` on its first pass.  `rmse_a(u) == sqrt(loss_a(u))`
exactly, so the best iterate's RMSE is bounded by the init's.  Reported values
say otherwise, so one of these is false:

  H1  the forward solve is nondeterministic -> repeated rmse_a(u0) disagree
  H2  `gf`'s loss and `rmse_a` disagree    -> loss(u0) != rmse_a(u0)^2
  H3  the trace never improves on step 1   -> best_z == u0, and 0.3250 is
                                              something else entirely
"""
import sys, pathlib
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from ioc import identifiability as ident
from iosp.study3_identifiable_refit import build_rkhs_aligned, N_STEPS, LR

b = build_rkhs_aligned(M=3)
u0 = jnp.zeros(b["K"], dtype=jnp.float32)

print("\nH1 determinism: rmse_a(u0) x4 =",
      [round(b["rmse_a"](u0), 8) for _ in range(4)], flush=True)

val, g = b["gf"](u0)
print(f"H2 consistency: loss(u0)={float(val):.8e}  rmse_a(u0)^2={b['rmse_a'](u0)**2:.8e}"
      f"  |grad|={float(jnp.linalg.norm(g)):.4e}", flush=True)

u_wide, trace = ident.wide_fit(b["gf"], u0, lr=LR, n_steps=N_STEPS)
losses = [v for _, v in trace]
print(f"H3 trace: first={losses[0]:.6e} min={min(losses):.6e} "
      f"last={losses[-1]:.6e} argmin={int(np.argmin(losses))}/{len(losses)}")
print("   first 8:", [f"{v:.4e}" for v in losses[:8]])
print("   last  4:", [f"{v:.4e}" for v in losses[-4:]])
print(f"   rmse_a(u_wide)={b['rmse_a'](u_wide):.6f}  sqrt(min loss)={np.sqrt(min(losses)):.6f}")
print(f"   ||u_wide - u0||={float(jnp.linalg.norm(u_wide - u0)):.4e}")
