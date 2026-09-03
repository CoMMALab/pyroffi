"""G0 root-cause probe: does the loss-vs-rmse^2 gap shrink with inner budget?

`loss_a(u)` and `rmse_a(u)**2` are the same quantity computed through two
separately jitted graphs.  A gap means the ROLLOUT is not a stable function
of `u`: the two graphs' fusion/reassociation land the fixed-length inner
solve on different iterates.  If that is an under-converged inner solve,
more iterations shrink the gap; if it is float32 noise in the rollout
itself, it will not.  Fixed probe seed so all budgets see the same `u`.
"""
import sys, time
sys.path.insert(0, ".")
import iosp.study3_identifiable_refit as st

BUDGETS = [int(a) for a in sys.argv[1:]] or [60, 120, 240]
for n_iters in BUDGETS:
    t0 = time.perf_counter()
    built = st.build_parametric(n_iters=n_iters)
    print(f"\n### n_iters={n_iters}  (build {time.perf_counter() - t0:.0f}s)", flush=True)
    worst = st.check_loss_rmse_consistency(
        {**built, "label": f"parametric n_iters={n_iters}"}, n_probe=5, seed=0)
    print(f"### n_iters={n_iters}: worst rel gap {worst:.3e}", flush=True)
