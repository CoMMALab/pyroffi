# AGENTS.md: PyRoFFI

Orientation for an AI agent (or a new human) working in this repo. This file is tracked and
authoritative for conventions. `CLAUDE.md` just includes it.

## What PyRoFFI is

A backward-compatible superset of [PyRoKi](https://github.com/chungmin99/pyroki): `import pyroffi
as pk` runs PyRoKi code unchanged (checked by `tests/test_pyroki_compat.py`). On top of that it
adds raw CUDA kernels reached through the JAX FFI, for performance past what XLA can generate:

- **FK / IK / collision / dynamics / trajopt kernels.** The motion kernels live in
  `external/GRiM` (`grim.motion`) and compile per robot on first use. pyroffi's own kernels
  (world-side collision, ESDF, roboGPU, TOPP-RA) are in `src/pyroffi/cuda_kernels/` and are built
  by `build_kernels/build_all.sh`.
- **The CUDA cost factory** (`src/pyroffi/costs/`). Users write costs in PyRoKi's jaxls-shaped API
  (`pk.Var`, `pk.Cost.factory`, `pk.LeastSquaresProblem`). `analyze()` compiles the whole problem
  (costs, kinematics, Jacobians) into one straight-line CUDA kernel, and `solve()` /
  `solve_batch()` is one launch with no JAX in the loop. **There is no JAX fallback.** An
  unsupported problem raises.
- **Solver suites.** IK: HJCD, LS, SQP, MPPI, region IK, IKFlow. Trajopt: dynamics-aware (the
  default), SCO, CHOMP, STOMP, L-BFGS, LS, and contact-aware/contact-rich tiers. Also SRDF
  parsing, neural SDFs, floating base and attachments.

JAX stays the outer layer. Kernels are FFI custom calls, so they compose with `jit`/`vmap` and
keep XLA's fusion around them.

## Layout

- `src/pyroffi/`: `_robot.py` (`pk.Robot`), `kinematics/`, `collision/`, `costs/` (cost compiler),
  `optimization_engines/` (IK solvers), `motion_generators/` (trajopt), `dynamics/`, `topp/`,
  `toolbox/`, `attachments/`, `_grim.py` (GRiM bridge), `_cuda_backends.py`.
- `external/`: vendored submodules (GRiM, which carries GRiD/GLASS/crocket). **Don't edit unless
  asked.** A change needed there goes to that repo first, then the submodule pointer gets
  bumped. Follow `external/GRiM/AGENTS.md` when you work inside GRiM.
- `OPEN_TASKS.md`: known-but-unfixed issues. Add what you find and remove what you fix.
- `tests/`: pytest suite. `examples/`: numbered runnable demos. `docs/`: design notes.
  `tools/` and `tests/bench_*.py`: perf tooling.

## Using PyRoFFI well

- **Opt in to CUDA.** The PyRoKi-compatible paths are JAX by default. Pass `use_cuda=True`, or
  pick a CUDA solver (`robot.inverse_kinematics(..., solver="hjcd" | "ls")`).
- **Batch everything.** The kernels pay off at large batches. Use `solve_batch` and batched
  targets, not Python loops. Thread/warp/block tier choice depends on DOF × batch together.
- **Runtime inputs vs baked constants.** In the cost compiler, `jax.Array` arguments (targets,
  obstacles, rest poses) are runtime inputs. NumPy arrays, Python scalars and robot models are
  baked into the build. To re-solve with new data without a rebuild, pass `jax.Array`s.
- **Warm up with the exact config you'll time.** A mismatched static config recompiles, and the
  recompile looks like a slow solve.
- **Limits:** cost compiler problems are Euclidean variables only, ≤32 scalars. Kernels have a
  hard `MAX_ACT ≤ 32` DOF ceiling. Dynamics-aware trajopt needs URDF inertials and a fixed base.
  Otherwise it falls back to SCO with a warning.
- **Precision:** x64 is off by default. Some grid/autodiff/flat tests are borderline under
  float32 and pass with `JAX_ENABLE_X64=1`. Set `jax.default_matmul_precision("highest")` when
  comparing against references, because TF32 breaks accuracy.
- **Differentiating through IK:** first order only. `jax.hessian` through IK returns a plausible
  but wrong number.

## Environment

```bash
conda activate pyroffi            # never the base env
export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
CUDA_VISIBLE_DEVICES=<free idx> python -m pytest tests -q
ruff format --check . && pyright   # CI: formatting.yml, pyright.yml, pytest.yml
```

GPUs are shared. Pick a free device. If all of them are busy, stop and say so, and never kill
other processes. Large jobs (IOSP/E10-style) need ~50–60 GB of host RAM each, so run them
serially.

## Rules for agents

1. **Keep PyRoKi compatibility.** Never break a PyRoKi call signature or name. Intentional
   differences go in the README's "Behaviour that intentionally differs" list. Run
   `tests/test_pyroki_compat.py` after any change to the public API.
2. **No silent fallbacks.** A CUDA path that can't handle an input raises. It never quietly
   drops to JAX or returns a degraded answer.
3. **Correctness lives in tests, not hot paths.** Don't add runtime validation or guards to
   kernels or solver loops. Every new kernel or cost gets a test against the JAX/float64
   reference (and a 0-iteration cost check for trajopt kernels).
4. **Fix, don't guard.** No `xfail`/`skip`/`nan_to_num`/clamping to hide a failure. Find the root
   cause. NaNs usually mean conditioning (zero-vector normalize, stiff integrator, col-scale) and
   rarely API misuse.
5. **Stay vectorized and `jit`-able.** Use `vmap`/batched kernels, not Python loops, and no
   data-dependent shapes. Keep everything differentiable end to end where it already is.
6. **Validate traced CUDA builds.** ptxas has silently miscompiled large straight-line traces at
   -O2/-O3. Check new kernels numerically, not just that they compile.
7. **Measure performance claims.** Report speedups and success rates only from runs you made,
   with the same config, seeds and idle GPU for both sides. Use ≥6 seeds for contact solvers,
   since their residuals vary ~40% run to run. Write it down when a benchmark is the wrong
   comparison.
8. **Ask before** adding dependencies (all pins in `pyproject.toml` are exact), restructuring
   modules, editing `external/`, or changing a default solver/config.
9. **Match the surrounding code.** Same naming, comment density and idioms. Run `ruff` and
   `pyright` clean. Update the README tables and examples when user-facing behaviour changes.
10. **Git:** path-scoped `git add` (never `-A`/`.`). Commit submodules before bumping the parent
    pointer. Commit or push only when asked. Verify a subagent's work yourself before trusting
    "done".
