# pyroffi vs cuRobo — IK parity

Head-to-head on the Panda/Franka. Cross-process by necessity: cuRobo and pyroffi
live in separate conda envs (`curobo`, `pyroffi`) with incompatible torch/JAX
pins, so the two solvers never share an interpreter. They share a **problem
file** instead.

    conda activate pyroffi && python benchmarks/parity/make_problems.py
    conda activate pyroffi && python benchmarks/parity/run_pyroffi.py
    conda activate curobo  && python benchmarks/parity/run_curobo.py
    conda activate pyroffi && python benchmarks/parity/compare.py

## What is being compared

`mppi_ik` is the intended cuRobo analogue: MPPI particles for gradient-free
exploration, L-BFGS as the convergence enforcer — the same architecture as
cuRobo's `particle_opt` + `newton_base`. `ls_ik` and `sqp_ik` are reported
alongside because they are what pyroffi would actually ship, but MPPI+L-BFGS is
the apples-to-apples row.

## Rules that make the numbers mean something

- **Targets are reachable by construction.** Poses come from forward kinematics
  on sampled in-limit configurations, so a solver cannot look good by failing
  fast on impossible problems.
- **Success is judged by one shared metric**, computed here from the returned
  configuration — never by either library's own convergence flag. Position and
  orientation error are recomputed with the same FK, plus collision.
- **Timing excludes compilation.** Both stacks JIT/warm up; each is run once
  with the exact timed configuration before measurement, and the best of N is
  reported.
- **Both stacks must use the SAME end-effector frame.** cuRobo's `franka.yml`
  sets `ee_link: panda_hand`; the URDF's last link is `panda_grasptarget`, a
  fixed offset away. Scoring against the wrong one reported cuRobo at 0.0% pose
  success while it was solving correctly. A frame mismatch does not look like a
  frame mismatch in the output -- it looks like the other library is broken, so
  treat any 0% row as a harness bug until proven otherwise.
- **Collision-free success is reported separately from pose success.** That
  split is the point of the comparison: cuRobo treats collision as a weighted
  cost, so its failures concentrate there rather than in pose error.
