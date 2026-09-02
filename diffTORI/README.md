# DiffTORI (JAX / Flax)

Implementation of **DiffTORI: Differentiable Trajectory Optimization for Deep
Reinforcement and Imitation Learning** (Wan, Wang, Wang, Erickson, Held —
NeurIPS 2024, [arXiv:2402.05421](https://arxiv.org/abs/2402.05421)), following
the authors' released code at
[wkwan7/DiffTORI](https://github.com/wkwan7/DiffTORI) wherever it and the paper
disagree.

The policy *is* a trajectory optimiser: actions are produced by solving an
optimisation problem whose cost is a neural network, and the imitation / policy
gradient loss is back-propagated **through the solver** to train it. The inner
solver here is `pyroffi.optimization_engines.dynamics_trajopt`, wired up the way
`ioc.inner` wires it.

## Layout

| file | contents |
|---|---|
| `difftori/solver.py` | Implicit-function-theorem gradients through the inner solve — same construction as `ioc.inner.make_inner_solver` |
| `difftori/pyroffi_trajopt.py` | `dynamics_trajopt` wrapped as the inner solver (+ optional FK / `ls_trajopt` teachers) |
| `difftori/policy_il.py` | Imitation learning: CVAE with a trajectory-optimisation decoder |
| `difftori/agent_rl.py` | Model-based RL on top of TD-MPC (Eq. 4–6) |
| `difftori/networks.py` | Flax (`flax.linen`) modules |
| `difftori/config.py` | Hyperparameters, sourced from the released configs |
| `difftori/train_il.py`, `train_rl.py` | Training loops (IL is complete; RL leaves the env loop to the caller) |
| `difftori/run_il.py` | CLI: train the IL policy on a generated dataset |
| `difftori/tblog.py` | TensorBoard logging + per-run config/commit dump |
| `difftori/benchmarks/pendulum.py` | Amos et al.'s pendulum swing-up: dynamics, MPC expert, Table 8 cost metric |
| `difftori/benchmarks/lstm_baseline.py` | The `LSTM policy` row of Table 8 |
| `difftori/run_pendulum.py` | CLI: reproduce Table 8 |
| `difftori/data/panda_reach.py` | Demonstration generator (pyroffi teacher → their zarr layout) |
| `difftori/data/dataset.py` | Replay buffer, sequence sampler, limits normaliser |
| `difftori/data/report.py` | Quality gates — run before spending GPU time |
| `difftori/data/visualize.py` | Dataset figures: EE paths, joint traces, action/clearance stats |
| `difftori/data/viser_playback.py` | Interactive 3D playback of the demonstrations in a browser |
| `tests/test_difftori.py` | Shape, gradient and released-code-fidelity tests |
| `tests/test_paper_fidelity.py` | Eq. 7 / Appendix D / Table 9 and the Table 8 benchmark constants |

```bash
conda activate pyroffi
cd diffTORI && JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python -m pytest tests -q   # ~60 s
```
Run under float64: the implicit adjoint inverts the inner Hessian.

## Two modes: the paper, and the released code

The released IL policy (`diffusion_policy_3d/policy/difftori.py`) is not what
Eq. 7 describes. Both are implemented and selected by config:

| | | |
|---|---|---|
| `ILConfig.paper()` | Eq. 7 + Appendix D + Table 9 | latent dynamics `d_θ`, three-network CVAE encoder, β=1, lr 3e-4 |
| `ILConfig()` (default) | the released code | no dynamics, flat action chunk, β=10, recon×3000, lr 1e-4 |

The default is still the released code, so existing runs and checkpoints keep
their meaning. `ILConfig.paper()` restores what the release drops:

* **`d_θ` exists.** `plan_cost` rolls the latent dynamics into the objective —
  `a(θ) = argmax_a Σ_l γ^l f_θ(z_l, a_l)` s.t. `z_{l+1} = d_θ(z_l, a_l)` —
  substituting the constraint rather than imposing it, as the paper does for
  Eq. 5. `planning_horizon = H` decides `H + 1` actions and applies `d_θ` `H`
  times. Previously `LatentDynamics` was defined in `networks.py` and never
  instantiated.
* **`h^a` exists.** Appendix D's action encoder, `z^a = h^a(a*)`, feeding the
  fusing encoder; the release feeds raw actions. Latent order is `z = [z̃, z^s]`.
* **Table 9 hyperparameters**, including the ELBO's unweighted reconstruction
  term (Eq. 9), against the release's re-tuned 3000/10 pair.

`tests/test_paper_fidelity.py` pins each of these, including that `d_θ`
actually receives gradient (an unused dynamics model would make Eq. 7
decorative) and that the implicit adjoint still agrees with finite differences
at cos > 0.999 now that `d_θ` sits inside the inner Hessian — measured 0.999998.

### Where the two differ, in detail

| | Paper | Released code (what we do) |
|---|---|---|
| Latent dynamics `d_θ` in IL | rolled into the objective | **absent** — one network scores a whole action chunk |
| CVAE encoder | `h^o`, `h^a`, `h^l` (App. D) | no `h^a`; raw actions into `h^l` |
| `horizon` in IL | planning horizon | action-**chunk** length, 4; `planning_horizon` is hardcoded to 1 and unused |
| Observation | one frame | `n_obs_steps = 2` frames, encoded and concatenated |
| KL coefficient | 1 | **10**; reconstruction weighted **3000** |
| Learning rate | 3e-4 | 1e-4, cosine-annealed to 1e-6 over 15k steps |
| `a_init` | — | a pretrained **DP3 policy's** action chunk (`use_zero_initial: False`) |
| RL terminal term | `Q(z_H, a_H)`, `a_H` optimised | `min(Q₁,Q₂)(z_H, π(z_H))` — so `H` decision variables, not `H+1` |

The RL side *does* roll out the latent dynamics, matching Eq. 5, and always
did.

## Where we deliberately differ from the released code

- **Gradient path.** They pass `backward_mode="truncated",
  backward_num_iterations=5` to Theseus — truncated unrolling, despite the paper
  describing implicit differentiation. We default to the implicit adjoint: exact
  at a stationary point and O(1) in memory. `solver.solve_unrolled` reproduces
  their path (`unroll_tail=5`) and the tests assert the two agree.
- **Inner solver.** Theseus' Levenberg–Marquardt accepts only nonlinear least
  squares, which is why their cost is `(1000 − reward)²`. `dynamics_trajopt`
  minimises an arbitrary scalar, so we minimise `−reward` directly; squaring a
  positively-shifted objective does not move the argmin, so this is the same
  problem without the contortion. Their `traj_opt_step` / `damping` have no
  analogue — the engine line-searches.
- **Per-sample solves.** They flatten the batch into one Theseus problem and
  average the residual over it, coupling every sample through one shared cost
  (and forcing their fixed 128-sample padding). We `vmap` one problem per
  sample.
- **Action bound.** Their `torch.clamp(a, -1, 1)` inside the cost makes it
  exactly flat outside the box, so a solver that steps out gets no gradient
  back. We use a one-sided quadratic barrier (`solver.action_penalty`), inactive
  on solutions already inside.
- **Line-search smoothing is on by default.** `dynamics_trajopt`'s hard `argmax`
  line search and hard curvature-pair gate can flip discretely as the encoder's
  `z` shifts infinitesimally, compounding into a discontinuous iterate and a
  silently wrong adjoint — the same failure `iosp.pickplace` hit with IK-derived
  boundary conditions. `pyroffi_trajopt` sets `soft_line_search` and
  `soft_curvature_gate` (not the engine's own defaults); `smooth=False` restores
  stock behaviour.

**Convergence is a precondition, not a detail.** The adjoint is exact only at a
stationary point. `solver.stationarity()` returns `‖∇ₓC‖` per problem; in `ioc`,
non-stationary contexts drop gradient agreement from cos 0.9999 to 0.59. Watch
it during training.

## Table 8: Amos et al.'s pendulum swing-up

The paper's Appendix A.3 compares DiffTORI against Amos et al. on that paper's
pendulum imitation task, in two settings. This is the one benchmark from the
paper reproducible here without a legacy `mujoco-py`/`gym 0.21` stack, and it
is the setting that actually exercises Eq. 7's dynamics rollout.

```bash
conda activate pyroffi
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=1 \
    PYTHONPATH=diffTORI python -m difftori.run_pendulum --setting both --seeds 3
```

Every constant in `difftori/benchmarks/pendulum.py` is transcribed from Amos'
released code, not inferred: `dt=0.05`, `max_torque=2`, `mpc_T=20`,
`goal_weights=(1,1,0.1)`, `ctrl_penalty=0.001`, params `(g,m,l)=(10,1,1)` and
the damped `(10,1,1,d=1.0,b=0.1)`, initial states `θ~U(-π/2,π/2)`,
`θ̇~U(-1,1)`, split 100/10/10.

**The metric.** Table 8 reports the *cost of the learned policy*: roll the
policy's open-loop nominal sequence `u_{1:20}` out on the true dynamics and
accumulate Amos' `QuadCost`, `Σ_t [½τᵀdiag(q)τ + pᵀτ]` with
`q=(1,1,0.1,0.001)`, `p=(-1,0,0,0)`. Note this is *not* the "weighted distance
to a goal state" `‖√q ∘ (τ−τ_g)‖²` that Amos' Sec. 5.3 prose describes — the
two are affinely related (`weighted = 2·quad + 1` per step, so same optimal
controls) but only the quadratic form reproduces Table 8's magnitudes. Reading
the prose instead of `get_true_obj()` gives 49.5 where the paper says 13.126.

**Calibration.** Our expert, solved with `dynamics_trajopt` rather than Amos'
box-constrained iLQR:

| | ours | Table 8 | seed spread |
|---|---|---|---|
| expert, w/o damping | 13.356 | 13.126 | 12.04 – 14.74 |
| expert, with damping | 10.998 | 10.132 | 9.72 – 12.53 |

Both land inside the seed-to-seed spread, which validates the dynamics, the
cost functional, the solver substitution and the metric together. The residual
gap is the initial-state draw: Amos seeds Torch's RNG and we cannot reproduce
that stream from JAX, so the *distribution* matches and the exact sample does
not — and cost varies from ~3.7 to ~77 across the `|θ₀|` range, so a different
draw of 120 uniforms moves the mean by a few units on its own.

`solve_expert` uses 8 random restarts because the swing-up objective is
non-convex and L-BFGS from a single zero initialisation hangs at the bottom on
a minority of initial states — the local minimum Amos' iLQR line search avoids.

**Not reimplemented:** the `Amos et al.` column itself. Reproducing it means
running their box-constrained iLQR with 10 learnable physical parameters, which
is a reimplementation of their method, not ours.

## Data

Their demonstrations are MetaWorld point clouds from scripted experts, which
need a py3.8 `mujoco-py<2.2` / `gym 0.21` / pytorch3d stack, *and* a pretrained
DP3 policy to supply `a_init` — the MetaWorld and Robomimic numbers are action
refinement on top of a base policy, so reproducing them means training that base
policy first. (The paper's **RL** benchmark is a different story: DMC runs on
`dm_control`, which is a pure-Python wheel on the modern `mujoco` bindings and
needs no legacy stack.) Instead we generate
demonstrations with `pyroffi` and write them in **their zarr layout**, so their
dataset/sampler/normaliser semantics carry over:

    data/state (N, 25)  data/action (N, 7)  data/point_cloud (N, 512, 3)
    meta/episode_ends (E,)

**Task class: dynamics-aware Panda reach-around-an-obstacle.** A 7-DoF Franka
Panda moves between two joint-space endpoints past one spherical obstacle,
anchored on the straight-line end-effector path so it is genuinely in the way.
The teacher is `dynamics_trajopt` minimising the `ioc.robot.bases.dynamic`
basis — effort, clearance, smoothness and an **RNEA torque** term — under known
weights. The torque term is what makes the demonstrations worth imitating: they
depend on the arm's mass and inertia, not geometry alone.

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=2 \
    PYTHONPATH=diffTORI python -m difftori.data.panda_reach --n-contexts 200
```

Three filters run before anything is written, each removing a distinct way a
"demonstration" can fail to be one:

1. **Valid scenes** (`sample_scenes_valid`, before solving). The inner problem's
   endpoints are *clamped* — the optimizer cannot move them — so an obstacle
   overlapping the start or goal produces a demonstration that begins or ends in
   collision, and no solver can fix it. The first version of this dataset had no
   such check: **121/200 episodes penetrated, 84 at the start and 58 at the
   goal, while only 7 penetrated in the interior.** Endpoints are also rejected
   for self-collision and clipped to the joint limits.
2. **Stationarity** (`ioc.robot.problem.screen_scenes`). A non-stationary solve
   is not an optimum, it is wherever the L-BFGS budget ran out. At
   `--n-iters 500` the discard rate is 0%; at 120 it was 58%.
3. **Clear solutions** (`screen_solutions`, after solving). The clearance term
   is a soft penalty weighted against effort, smoothness and torque, so a solved
   trajectory can still clip the obstacle. Those episodes are dropped.

**Scene diversity.** `sample_scenes_valid` jitters start and goal independently
at 0.35 rad, against `ioc`'s antisymmetric 0.10 rad. That sampler is tuned for
weight identifiability, not for covering an observation distribution, and its
paths trace one narrow tube. Measured over the jitter range: 0.35 rad gives
~3.5× the end-effector spread of 0.10, *fewer* endpoint rejections (wider
endpoints sit further from the obstacle), no joint-limit pinning, and the
obstacle still blocks the straight-line seed in 100% of scenes — so the
diversity costs nothing. The identifiability argument for anchoring the
*obstacle* on the seed path is kept.

```bash
PYTHONPATH=diffTORI python -m difftori.data.report --data diffTORI/data/panda_reach_expert_v2.zarr
```

checks the gates that make a dataset safe to train on — no penetration,
stationarity, the obstacle actually being active, actions inside the unit box,
enough training windows — and exits non-zero if any fails.

`state` is relative except for the configuration itself (`q, dq, q_goal − q,
obs_center − p_ee(q), obs_radius`), so the goal and obstacle transfer across
contexts. Actions are joint deltas divided by one dataset-wide constant, then
mapped to `[-1, 1]` by the loader's limits normaliser — the inner problem's
barrier assumes a unit box. `point_cloud` is the robot's collision spheres plus
the obstacle, sampled per waypoint, so the PointNet encoder path can be
exercised; the MLP encoder path ignores it.

## Runs and logging

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=2 \
    PYTHONPATH=diffTORI python -m difftori.run_il --steps 15000
tensorboard --logdir diffTORI/runs
```

`--steps 0` builds everything and exits — the cheapest shape check before
committing a GPU. Each run gets `runs/<name>-<timestamp>/` holding the event
file and a `config.json` with the fully resolved config and the git commit: a
scalar curve you cannot tie back to the hyperparameters that produced it is not
a result.

Logged every step: `train/loss`, `train/recon`, `train/kl` (the raw terms, not
the 3000-/10-weighted ones — those are what tell you which term is actually
driving the gradient), `train/lr`, `train/grad_norm`.

Logged every `--diag-every` steps on the validation split: `val/loss`,
`val/recon`, `val/kl`, and **`val/stationarity_max` / `val/stationarity_mean`**.
Watch the stationarity. The implicit adjoint is exact only at a stationary point
of the inner problem, and nothing in the loss curve reveals a violated
precondition — see the measured numbers below.

Writing event files needs only `tensorboardX`; the `tensorboard` package was
installed to *view* them.

## Looking at the data

```bash
PYTHONPATH=diffTORI python -m difftori.data.visualize --n-contexts 6   # PNGs
PYTHONPATH=diffTORI python -m difftori.data.viser_playback            # browser
```

The viser script replays the demonstrations on the robot with the obstacle at
its true radius and the EE path traced: pick an episode, play/pause, scrub the
waypoint, and toggle the collision spheres the clearance term is computed from.
Joint order is mapped between pyroffi and the URDF **by name**, following the
same contract as `pyroffi.toolbox._exchange` — a viewer that silently reorders
joints draws a wrong pose convincingly.

`visualize` writes `figures/dataset_{ee,joints,summary}.png`. Nothing is re-solved: every
quantity is reconstructed from `data/state` using its documented layout, so the
figures show exactly what the policy is trained on.

The summary figure reports **min robot-sphere clearance**, not end-effector
distance. The EE frame origin is not a collision sphere, so an EE-to-obstacle
distance goes negative on perfectly valid trajectories and means nothing; the
sphere clearance is the quantity the teacher's collision feature is actually
built from.

**Known issue it exposes.** 121/200 episodes have at least one waypoint
penetrating the obstacle — but only **7** penetrate in the *interior*, and 84
start / 58 finish in collision. The endpoints are clamped boundary conditions
that the optimizer cannot move, and `sample_scenes` anchors the obstacle near
the path without checking the endpoints against it. Interior min clearance
averages 0.048 m, right at `CLEARANCE_MARGIN`, so the teacher is doing its job
where it has freedom. For IOC this is harmless; for imitation learning it means
some training states are in collision. Fixing it means rejecting scenes whose
endpoints collide, at scene-sampling time.

## Not implemented

PER for the RL replay buffer, the DP3 PointNet / Robomimic RNN encoders
themselves (pass any module with the right signature as `obs_encoder`), and the
residual-policy ablations of Sec. 5.

## Dependencies

`pyroffi` (`dynamics_trajopt` *is* the inner solver) and `zarr` (added to the
`pyroffi` env for the dataset format).
