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
- VAMP Compatible Primitive Sets
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

PyRoki's jaxls cost layer is available under the same names (`pip install -e .[costs]`):
`pk.costs.*` factories and `robot.joint_var_cls` build ordinary jaxls problems, so
`jaxls.LeastSquaresProblem(costs, vars).analyze().solve()` works unchanged. To run the
same problem as a single CUDA kernel (costs, kinematics and Jacobians compiled to
straight-line code, no JAX inside the solve loop), compile it:

```python
solver = pk.costs.compile(jaxls.LeastSquaresProblem(costs, [joint_var]))
sol = solver.solve()                     # jaxls.VarValues: sol[joint_var]
sol = solver.solve(new_problem)          # same structure, new targets: no recompile
x, cost, viol = solver.solve_batch(params, x0)   # B problems in one launch
```

Robot and collision models, NumPy arrays and Python scalars are baked into the build;
`jax.Array` arguments (targets, rest poses, obstacles) stay runtime inputs. The compiled
solve supports Euclidean variables up to 32 scalars (IK-sized problems); larger or
Lie-group problems stay on jaxls. The built-in solvers remain available too:

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
pip install -e .
```
