# IOSP — inverse optimal control on a composed pick-and-place task

`ioc` inverts **one** trajectory-optimization segment. `iosp` inverts a whole
**task skeleton**: IK → approach → grasp → transport → place, chained through
literal (undetached) data flow, so a single rollout of a complete pick-and-place
can be inverted for per-segment cost weights. All differentiation machinery —
implicit adjoint, finite differences, CMA-ES, subspace refit — is reused
unmodified from `ioc`.

The demonstrations are **synthetic**: every one is a rollout of this same model
at `config.THETA_IK_STAR` / `config.Z_TRAJOPT_STAR`. That is deliberate — it
makes "did recovery work?" a question with an exact answer instead of a
judgement call.

## Layout

```
iosp/
  config.py         every path, task constant, ground-truth weight and solver default
  model/
    pickplace.py    the composed IK -> 4-segment trajopt forward model
    scenes.py       scene A (fit), scene B (held out), multi-scene contexts
  fit/
    params.py       the u -> theta parameterization and its softmax gauge
    parametric.py   build_parametric: the path-A bilevel forward map
    procedure.py    the 5-stage fit-wide/Gram/select/refit/report driver + G0 gate
    multistart.py   many candidates (IK branch x cost start) as one batched program
  experiments/      one module per experiment; see the table below
  checks/           diagnostics that gate the experiments
  record/           run a fit, save every outer step to .npz
  viz/              renderers that read those .npz files
  analysis/         aggregate multi-run sweeps into tables
  shelved/          deliberately out of scope, kept for the negative results
  scripts/          multi-GPU sweep drivers
```

Nothing in `config.py` imports from `iosp`. Before this layout, `study3`
imported its URDF paths from `study0_segment_ablation`, its ground truth from
`recovery_bench` and its held-out offsets from `generalization_check` — three
experiment scripts had to be importable for a fourth to run.

## The core idea, in one paragraph

The inner problem is `x*(θ,c) = argmin_x Σ_k θ_k φ_k(x,c)`; the outer problem
fits θ so that `x*` matches the demonstration. Differentiating through the
inner **solution** needs `dx*/dθ`, which exists only where `x*(θ)` is smooth.
Two things break that smoothness on a redundant arm, and both are handled
structurally rather than by tuning: the redundant IK's winner selection (fixed
by `IK_CONTINUITY_WEIGHT`, and by pinning one branch per candidate) and the
trajopt's non-convexity (handled by running many candidates and selecting once,
at the end, on training loss). Everything else — rank deficiency, the softmax
gauge, the identifiable-subspace refit — is about *which directions of θ the
demonstration can resolve at all*, and lives in `fit/procedure.py`.

## Experiments

Run each as `python -m iosp.experiments.<name>`. All GPU work needs
`CUDA_VISIBLE_DEVICES=<idx> XLA_PYTHON_CLIENT_PREALLOCATE=false` — check
`nvidia-smi` first, these boxes are shared. The persistent compile cache
(`config.enable_compilation_cache()`) turns a ~25 min cold compile into ~2 min.

| module | old name | question | headline finding |
|---|---|---|---|
| `recovery_bench` | — | Can the implicit adjoint recover a known θ\*, and how does it compare to CMA-ES at equal solve count? | The reference benchmark. FD agreement is *not* the standard here; recovering a known θ\* is. |
| `e0_segment_ablation` | study0 | Freeze all-but-a-growing-subset of the 4 segments at ground truth — where in the chain does recovery break? | Composition is **not** the problem. Fitting only `transport` errs as much as fitting all four (~0.21–0.24). |
| `e0d_eigen_projection` | study0d | Split recovery error into its identifiable and null components. | The optimizer is behaving correctly: top-1 error 0.0202, null (8-dim) 0.2422. Raw ‖θ̂−θ\*‖ cannot tell these apart. |
| `e1_minimal_identifiable` | study1 | A deliberately identifiability-clean K=3 transport-only problem. | The minimal case where recovery *should* work; the loss floor is a basin, not a floor (multistart reaches 0.13128 vs 0.44543). |
| `e1b_multidemo` | study1_diagnostic_multidemo | Does adding demonstrations fix it? | Only mildly and non-monotonically — more demos help only if they change the *span* of the Gram. |
| `e1c_fd_check` | study1_diagnostic_fd_check | FD vs implicit adjoint on E1's K=3 loss. | Diagnostic only. See `checks/composed_fd.py` for the composed-chain version. |
| `e2_demo_quality` | study2 | Does recovery track demo **curation** rather than demo **count**? | Curation. Count barely moves it. |
| `e3_identifiable_refit` | study3 | Fit wide → Gram → select r → refit on `U_r`. Does rank deficiency become a *generalization* cost instead of a *reconstruction* cost? | The main path-A experiment. Refit pins the null component by construction; report in the `U_r` projection, never as raw L2. |
| `e4_three_stage` | study4 | Invert spasm's three-stage forward pass (segments + a whole-trajectory refine). | Not yet validated — do not use it for a figure. |
| `e5_tamp2d` | study5 | The same tied three-stage inversion on a cheap 2D TAMP benchmark. | The drawable sanity check for the composed method. |
| `e7_loss_space` | scratch/joint_loss_test (now e7) | Score the outer loss in **joint** space instead of EE space. | Reconstruction −18.5% on 5/5 seeds; **generalization not established** (−3.5% ± 21.9%, 4/5). Mechanism is reconditioning (λ₁/λ₂ 30 → 4.05), *not* added rank. |

Numbered names are kept as stable IDs because the logs, notes and memory all
refer to them; the old `study<n>` filenames map to the table above.

## Checks — run these when a result looks wrong

| module | asks |
|---|---|
| `checks/identifiability` | What is the Gram spectrum on this scene? Which features are resolvable at all? |
| `checks/generalization` | Does fit-demo RMSE actually indicate correct recovery, or only memorization? |
| `checks/fullchain` | Does a loss still differentiate through IK → 4 chained solves? |
| `checks/ik_branch` | Are the behavioural-loss spikes IK self-motion branch flips? (They were.) |
| `checks/branch_classes` | How many *distinct* IK branches does this arm actually have here? |
| `checks/composed_fd` | Do the single-segment soft-flag findings carry to the composed chain? |
| `checks/loss_rmse_consistency` | Is the rollout a stable function of `u`? (The G0 gate — catches disagreements stationarity screening misses.) |

## Reproducing the current results

**The multistart robustness result** (the strongest claim: 9 candidates,
held-out scene at 2× displacement, one selection on training loss):

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m iosp.record.multistart --space joint --n-branches 3 --n-starts 3 \
      --steps 40 --scene-b-scale 2.0 --out iosp/data/viz/multistart_behavior.npz
```

~2 h. Prints the per-candidate table and writes an `.npz` holding every
candidate's path at every outer step. Then render:

```bash
python -m iosp.viz.multistart iosp/data/viz/multistart_behavior.npz   # talk figure
python -m iosp.viz.behavior3d iosp/data/viz/multistart_behavior.npz   # 3D, single fit
python -m iosp.viz.behavior   iosp/data/viz/multistart_behavior.npz   # 2D, x-y projection
```

**The joint-vs-EE loss comparison**, paired by seed:

```bash
python -m iosp.experiments.e7_loss_space --space joint --seed 0
python -m iosp.experiments.e7_loss_space --space ee    --seed 0
```

~25 min each warm (add ~11 min for the sensitivity spectrum unless you pass
`--no-spectrum`).

**The whole multi-seed sweep** across every free GPU, then the tables:

```bash
bash iosp/scripts/queue_multiseed.sh          # ~3.5 h on 4 idle GPUs
python -m iosp.analysis.multiseed             # prints stage A and stage B tables
```

`IOSP_RESULTS=<dir>` repoints the aggregator if you keep results elsewhere.

## Reading the results honestly

Three rules, each of which was learned by getting it wrong first:

1. **Score behaviour, not parameters.** The spectrum is rank-deficient, so the
   fit reproduces behaviour while drifting along null directions. A weight-bar
   plot against ground truth looks like failure on a good fit.
2. **Report at a fixed step budget, never at each run's own minimum.** Taking
   the min is selection on the held-out criterion.
3. **The eigendecomposition is local to a basin.** `G = JᵀJ` needs `x*(θ)`
   differentiable, so it is blind to basin mismatch — a wrong-branch candidate
   can have a perfectly conditioned Gram and still sit 1 m from the demo. Rank
   diagnoses null-drift and identifiable error; only multistart diagnoses basin
   mismatch.

## Theory

`THEORY.md` (the bilevel problem and the three ways to get `dx*/dθ`),
`THEORY_IDENTIFIABLE_REFIT.md` (the rank-deficient refit derivation, path A and
path B), `THEORY_NONPARAMETRIC_GAPS.md`. `HANDOFF.md` is the running
investigation log.
