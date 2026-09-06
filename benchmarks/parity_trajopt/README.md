# pyroffi vs cuRobo — trajopt (config→config motion) parity

Head-to-head of the trajectory optimizers on the Panda/Franka. Cross-process by
necessity: cuRobo and pyroffi live in separate conda envs (`curobo`, `pyroffi`)
with incompatible torch/JAX pins, so they never share an interpreter — they
share a **problem file** (`problems.npz`).

    conda activate pyroffi && python make_problems.py
    conda activate pyroffi && python run_pyroffi.py
    conda activate curobo  && python run_curobo.py
    conda activate pyroffi && python compare.py

## What is being compared

Apples-to-apples **local trajectory optimizers**, both seeded from the straight
line, both config→config (no IK confound):

- **pyroffi**: `dynamics_trajopt` (the shared L-BFGS `_trajopt_core`), cost =
  smoothness + velocity + world/self collision hinge, fixed endpoints, batched
  over all problems with `vmap`.
- **cuRobo**: `TrajOptSolver.solve_cspace` with its default
  `lbfgs_bspline_trajopt` optimizer (L-BFGS), `use_cuda_graph=True`, 4 seeds.

Deliberately **not** cuRobo's full `MotionGen` — that adds a global graph
planner, which pyroffi trajopt has no analogue for. Comparing the trajopt stages
keeps it fair; the problems are constructed so the straight line only *grazes*
the obstacle (a collision-free detour exists locally), staying in the regime a
local optimizer can solve.

## Rules that make the numbers mean something

- **Endpoints collision-free by construction**; the straight-line midpoint
  grazes the obstacle, so local avoidance matters but the task is solvable.
- **One shared metric** (`compare.py`): collision (world + self), endpoint
  accuracy and path length are recomputed with **pyroffi's** FK + collision
  model on both stacks' returned trajectories — neither grades its own homework.
- **Timing excludes compile / graph capture**; each stack is warmed once with
  the exact timed config, best-of-N reported.
- **cuRobo scored on its interpolated executable plan** (7 arm joints), which
  is pinned to the start/goal — not the raw bspline knots.

## Results (128 problems, Panda, one cuboid; A5000)

Both stacks CUDA-graph / JIT warmed, batched, timed excluding compile.

| solver  | single-problem latency | batch throughput (128) | collision-free | goal err |
|---------|------------------------|------------------------|----------------|----------|
| cuRobo  | **49.5 ms**            | 6.6 s → **51.6 ms/prob** | 84.4 %       | 0.0      |
| pyroffi | 165 ms                 | 14.0 s → 109 ms/prob     | **95.3 %**   | 0.0      |

**Read:** cuRobo is ~3.3× faster per-problem latency and ~2.1× on batch
throughput; pyroffi is slower but competitive, and lands +11 pp more
collision-free paths here. Both hit the goal exactly.

### Why pyroffi's number is what it is (profiling)
This solve is **collision-cost-bound**, not L-BFGS-driver-bound: `m_lbfgs` 2 vs 8
changed batch time only 8 %, and the L-BFGS **two-loop is ~90 % of a *cheap*-cost
solve but ~8 % here**. So:
- The **compact-representation L-BFGS** rewrite was prototyped and **measured to
  be a no-win** (1.03× isolated on GPU — batched triangular solves cost about
  what the sequential dots did), and irrelevant to this collision-bound solve.
  Not shipped.
- The real lever is **fewer collision evaluations**: `n_iters` 100→60 (early-stop
  still on) is ~1.67× faster with coll-free essentially unchanged (96.1→95.3 %).
  That is the config used above. Line-search width (5→2 pt) barely mattered on
  GPU (~7 %) — the vmap parallelizes the trial-alpha collision evals.
- The remaining structural lever is **fused FK+collision kernels** (a separate
  effort), since the cost function — not the optimizer — dominates.

### Caveats (honest)
- `safe(≥2 cm)` is 0 % for **both** — by construction the problems graze the
  obstacle, so nobody keeps a 2 cm margin; `coll-free (≥0)` is the meaningful
  metric here.
- cuRobo optimizes its **own** collision spheres, then is scored on pyroffi's
  spherized model — a small geometry mismatch that can only *understate*
  cuRobo's collision number, not inflate it.
- cuRobo also has a batch trajopt API not used here; its throughput row would
  improve with it. pyroffi's strength is the vmap batch; cuRobo's is
  single-problem CUDA-graph latency.
