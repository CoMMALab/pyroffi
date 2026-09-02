"""pyroffi's dynamics-aware trajopt engine as DiffTORI's inner solver.

``pyroffi.optimization_engines.dynamics_trajopt`` is an L-BFGS minimizer over a
flat decision vector with a caller-supplied ``cost_fn(x) -> scalar``.  It is the
same engine ``ioc.inner`` uses as its ``forward_solver``, and it is wrapped here
the same way (cf. ``ioc.robot.e1_identifiability.make_dynamics_forward_solver``).

Why this engine rather than the paper's Theseus LM layer:

* it minimizes an *arbitrary scalar* cost, so DiffTORI's discounted return
  (Eq. 4/7) is optimized directly instead of being contorted into a nonlinear
  least-squares residual just to satisfy Theseus;
* it ships both forms the two gradient paths need -- the cheap early-stopping
  ``while_loop`` form for the implicit adjoint's forward solve, and the
  fixed-iteration ``scan`` form with a ``unroll_tail`` for truncated unrolling;
* it is already the differentiation-tested path in this repo.

Smoothness defaults
-------------------
Both smoothing flags default **on** here, which is not the engine's own default.
DiffTORI's inner problem is driven by an upstream continuous input -- the latent
``z`` from the encoder -- which is exactly the situation the engine's docstring
warns about, and the same one ``iosp.pickplace`` hit with its IK-derived
boundary conditions: the line search's hard ``argmax`` over trial step sizes and
the hard curvature-pair admit/reject gate can each flip discretely as ``z``
shifts infinitesimally, compounding across iterations into a discontinuous
final iterate.  Implicit differentiation assumes ``x*`` varies smoothly with its
inputs, so those flips silently corrupt the gradient.  Pass
``smooth=False`` to recover the engine's stock behaviour.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import Array

__all__ = ["make_dynamics_forward_solver", "make_unrolled_forward_solver",
           "task_space_residual", "ls_trajopt_teacher"]


def _config(smooth: bool, **kw: Any):
    from pyroffi.optimization_engines import DynamicsTrajOptConfig

    return DynamicsTrajOptConfig(
        soft_line_search=smooth, soft_curvature_gate=smooth, **kw)


def make_dynamics_forward_solver(
    n_iters: int = 100,
    grad_tol: float = 1e-6,
    m_lbfgs: int = 8,
    smooth: bool = True,
) -> Callable[[Array, Callable], Array]:
    """Early-stopping form: the forward solve behind the implicit adjoint.

    Not reverse-mode differentiable (data-dependent trip count) -- which is
    fine, and intended: it is only ever called under ``stop_gradient``, with the
    gradient supplied analytically by ``solver.make_difftori_solver``.
    """
    from pyroffi.optimization_engines import dynamics_trajopt

    cfg = _config(smooth, n_iters=n_iters, grad_tol=grad_tol,
                  m_lbfgs=m_lbfgs, early_stop=True)

    def forward_solver(x0: Array, cost_fn: Callable) -> Array:
        return dynamics_trajopt(x0, cost_fn, cfg)

    return forward_solver


def make_unrolled_forward_solver(
    n_iters: int = 100,
    unroll_tail: int = 5,
    m_lbfgs: int = 8,
    smooth: bool = True,
) -> Callable[[Array, Callable], Array]:
    """Fixed-iteration form for truncated unrolling (``solve_unrolled``).

    Memory scales with ``unroll_tail``; use it to cross-check the adjoint on a
    small batch, not to train.
    """
    from pyroffi.optimization_engines import dynamics_trajopt

    cfg = _config(smooth, n_iters=n_iters, m_lbfgs=m_lbfgs,
                  early_stop=False, unroll_tail=unroll_tail)

    def forward_solver(x0: Array, cost_fn: Callable) -> Array:
        return dynamics_trajopt(x0, cost_fn, cfg)

    return forward_solver


# -- optional: grounding the inner problem in real kinematics ---------------


def task_space_residual(
    robot: Any,
    link_index: int,
    target_wxyz_xyz: Array,
    weight: float = 1.0,
) -> Callable[[Array], Array]:
    """Exact FK cost ``w * ||fk(q)[link] - target||^2`` for one configuration.

    An extension to the paper, off by default: when the action space *is* a
    robot configuration, adding this to the learned latent cost keeps the
    network from having to rediscover kinematics from demonstrations.  Uses the
    pure-JAX FK path; ``use_cuda=True`` is not used because that FFI kernel does
    not support ``vmap``.
    """

    def cost(cfg: Array) -> Array:
        poses = robot.forward_kinematics(cfg)
        return weight * jnp.sum((poses[link_index] - target_wxyz_xyz) ** 2)

    return cost


def ls_trajopt_teacher(
    init_trajs: Array,
    start: Array,
    goal: Array,
    robot: Any,
    robot_coll: Any,
    world_geoms: tuple,
    **kwargs: Any,
) -> Array:
    """Collision-aware teacher trajectory from ``pyroffi.ls_trajopt``.

    For generating demonstrations when no human data exists, or for warm-starting
    ``a_init`` with a smooth collision-free trajectory instead of zeros.
    """
    from pyroffi.optimization_engines import ls_trajopt

    best_traj, _costs, _all = ls_trajopt(
        init_trajs=init_trajs, start=start, goal=goal, robot=robot,
        robot_coll=robot_coll, world_geoms=world_geoms, **kwargs)
    return jax.lax.stop_gradient(jnp.asarray(best_traj))
