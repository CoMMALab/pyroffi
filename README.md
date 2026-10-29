# `PyRoFFI`: A Python Library for Accelerated Rigid Body Motion Planning using the Foreign Function Interface


[![Format Check](https://github.com/CoMMALab/pyroffi/actions/workflows/formatting.yml/badge.svg)](https://github.com/CoMMALab/pyroffi/actions/workflows/formatting.yml)
[![Pyright](https://github.com/CoMMALab/pyroffi/actions/workflows/pyright.yml/badge.svg)](https://github.com/CoMMALab/pyroffi/actions/workflows/pyright.yml)
[![Pytest](https://github.com/CoMMALab/pyroffi/actions/workflows/pytest.yml/badge.svg)](https://github.com/CoMMALab/pyroffi/actions/workflows/pytest.yml)
[![PyPI - Version](https://img.shields.io/pypi/v/pyroffi)](https://pypi.org/project/pyroffi/)

By Weihang Guo, Sai Coumar

PyRoFFI is a toolkit optimized with Jax JIT tracing for accelerated motion planning research with NVIDIA GPUs. This repository is expands on [pyroki](https://github.com/chungmin99/pyroki) and is backward-compatible. PyRoFFI replaces core kinematics/dynamics kernels in PyRoKi with raw CUDA implementations through the Jax FFI interface to retain the benefits of Jax's compiler optimizations while adding the flexibility of CUDA for low-level GPU optimization of primitive operations. Additionally, we add an extended suite of optimization solvers for inverse kinematics and trajectory optimization. We provide accelerated kernels for the following kinematics functions:

- Forward Kinematics
- Inverse Kinematics: HJCD-IK, LS, SQP, MPPI, IKFlow
- Trajectory Optimization: dynamics-aware (default), SCO, CHOMP, STOMP, L-BFGS, LS


Additional Features:
- SRDF parsing
- Runtime Neural SDFs
- Improved Jax performance

## Coming from PyRoki

For IK, FK and collision code, swap the import:

```python
import pyroffi as pk  # was: import pyroki as pk
```

`pk.Robot`, `pk.collision` (geometry, `collide`, `RobotCollision`, including
`RobotCollision.from_sphere_decomposition` and the PyRoki `Box` names),
`pk.viewer` and `pk.utils` accept the same calls as PyRoki (upstream `388e43e`),
checked by `tests/test_pyroki_compat.py`. The fast paths are opt-in: pass
`use_cuda=True`.

PyRoki's cost layer is available under the same names, on pyroffi's own jaxls-shaped API
(no jaxls dependency). Port PyRoki cost code by swapping the import and writing `pk.X`
for `jaxls.X` (`pk.Var`, `pk.VarValues`, `pk.Cost.factory`, `pk.LeastSquaresProblem`,
`pk.TrustRegionConfig`, ...):

```python
joint_var = robot.joint_var_cls(0)
costs = [pk.costs.pose_cost_analytic_jac(robot, joint_var, target, link_idx, 50.0, 10.0),
         pk.costs.limit_constraint(robot, joint_var)]
sol = pk.LeastSquaresProblem(costs, [joint_var]).analyze().solve()
cfg = sol[joint_var]
```

`analyze()` compiles the problem to one CUDA kernel: costs, kinematics and Jacobians
become straight-line code, and `solve()` is a single launch with no JAX in the loop.
There is no JAX solver behind it: without a CUDA device and nvcc, or for a problem the
compiler does not support, `analyze()` raises. Robot and collision models, NumPy arrays
and Python scalars are baked into the build; `jax.Array` arguments (targets, rest poses,
obstacles) are runtime inputs, so re-solving with new targets reuses the build. Problems
may be built and solved inside `jax.jit` with the robot as a traced argument, as PyRoki
does. `analyze().solve_batch(params, x0)` solves many problems in one launch.

Constraints (`kind="constraint_eq_zero" | "constraint_leq_zero" | "constraint_geq_zero"`)
are solved by an augmented Lagrangian, as in jaxls, and land on a local constrained optimum
(KKT point). To get the CUDA solvers' speed with arbitrary constraints, seed the
constrained problem with an unconstrained CUDA solve through `initial_vals`, exactly as
PyRoKi seeds its own problems:

```python
q0 = robot.inverse_kinematics("panda_hand", target, solver="ls", use_cuda=True)
sol = pk.LeastSquaresProblem(costs + constraints, [joint_var]).analyze().solve(
    initial_vals=pk.VarValues.make([joint_var.with_value(q0)]),
    termination=pk.TerminationConfig(max_iterations=300),
    augmented_lagrangian=pk.AugmentedLagrangianConfig(max_iterations=50))
```

For trajectories, seed `traj_vars.with_value(...)` with a `TrajoptMotionGenerator` result.
`max_iterations` counts inner steps across all AL updates (as in jaxls), so the default of
100 can stop before a constrained problem is feasible; check `SolveSummary.constraint_violation`.
Batches go through `analyze().solve_batch(params, x0)`.

Current limits: Euclidean and Lie-group variables, at most 32 tangent dimensions at the
thread and warp tiers and 128 at the block tier; trajectories use the banded solver. The
built-in solvers remain available too:

| PyRoki pattern | PyRoFFI |
| --- | --- |
| `pose_cost` + `limit_cost` + `rest_cost` in a jaxls problem | `robot.inverse_kinematics(target_link_name, target_pose, solver="hjcd" \| "ls")` |
| collision costs / constraints in IK | `robot.inverse_kinematics(..., constraints=...)` with a PyRoFFI collision model |
| `pose_cost_analytic_jac(..., joint_mask=...)` | `robot.inverse_kinematics(..., fixed_joint_mask=...)` |
| trajectory costs in a jaxls problem | `pk.motion_generators.TrajoptMotionGenerator` |

Behaviour that intentionally differs:

- Sphere models are `RobotCollisionSpherized`. Distances are per link (min over
  its spheres), not per sphere, and the posed geometry is `(*batch, S, N)`.
- Self-collision pairs are stored `(i > j)` and sorted by first link, not
  `(i < j)`; use `active_idx_i` / `active_idx_j` to identify them.
- Kinematic index fields (`parent_indices`, ...) are integer arrays, not tuples.

### Trajectory optimization

`TrajoptMotionGenerator` picks its engine from the type of `trajopt_cfg`:

| `trajopt_cfg` | Engine |
| --- | --- |
| `DynamicsAwareTrajOptConfig()` (default) | Collision-free along the interpolated path and within URDF joint, velocity and torque limits (tier-1 `dynamics_trajopt`, two-stage). Needs a URDF with inertials; fixed base only. Mimic joints (e.g. the Panda gripper) are held rigid at mid-stroke in dynamics. |
| `ScoTrajOptConfig`, `ChompTrajOptConfig`, `StompTrajOptConfig` | Kinematics only; faster. |

Leaving `trajopt_cfg` unset picks `DynamicsAwareTrajOptConfig()`. For floating-base robots, or robots without a dynamics model, it falls back to `ScoTrajOptConfig()` with a warning.

For contact, call the tiers directly: `flat_contact_trajopt` (contact-aware) or
`contact_rich_trajopt` (contact forces as decision variables).

## Installation
```
git clone https://github.com/commalab/pyroffi.git
cd pyroffi
git submodule update --init --recursive  # external/GRiM and external/cudals
pip install -e .
```
