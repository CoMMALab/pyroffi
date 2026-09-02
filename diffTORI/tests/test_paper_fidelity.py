"""Tests for the Eq. 7 / Appendix D / Table 9 path and the Table 8 benchmark.

The existing ``test_difftori.py`` pins the *released-code* behaviour.  These
pin the parts of the paper the released code drops, which is what
``ILConfig.paper()`` restores.  Run under float64::

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python -m pytest tests -q
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from difftori.config import ILConfig, SolverConfig
from difftori.policy_il import DiffTORIPolicy, il_loss, make_solver

SMALL = dict(action_dim=1, obs_dim=3, n_obs_steps=1, mlp_hidden=32,
             encoder_hidden=32, obs_feature_dim=8, posterior_dim=8)


def _fixture(cfg, B=3, seed=0):
    m = DiffTORIPolicy(cfg=cfg)
    rng = jax.random.PRNGKey(seed)
    obs = jax.random.normal(rng, (B, cfg.n_obs_steps, cfg.obs_dim))
    act = jnp.tanh(jax.random.normal(jax.random.PRNGKey(seed + 1),
                                     (B, cfg.chunk_len, cfg.action_dim)))
    params = m.init(rng, obs, act)["params"]
    return m, make_solver(m), params, obs, act, rng


# -- Table 9 ----------------------------------------------------------------

def test_paper_preset_matches_table9():
    c = ILConfig.paper(action_dim=4, obs_dim=9)
    assert c.kl_coefficient == 1.0            # Table 9 "KL coefficient"
    assert c.learning_rate == 3e-4            # Table 9 "Learning rate"
    assert c.obs_feature_dim == 50            # Table 9 "Latent dimension"
    assert c.posterior_dim == 64              # Table 9 "Posterior Gaussian dim"
    assert c.solver.n_iters == 100            # Table 9 "Max planning iterations"
    assert c.planning_horizon == 1            # Table 9 "Planning horizon schedule"
    assert c.action_loss_weight == 1.0        # Eq. 9 is an unweighted ELBO


def test_released_defaults_are_unchanged():
    """The released-code path must keep its meaning; old runs depend on it."""
    c = ILConfig()
    assert (c.kl_coefficient, c.action_loss_weight) == (10.0, 3000.0)
    assert c.learning_rate == 1e-4 and c.horizon == 4
    assert c.use_dynamics is False and c.paper_cvae is False
    assert c.chunk_len == 4


# -- Eq. 7: the dynamics model actually exists and is in the graph ----------

def test_paper_mode_instantiates_dynamics_and_action_encoder():
    cfg = ILConfig.paper(**SMALL)
    _, _, params, *_ = _fixture(cfg)
    # Appendix D's three encoder networks plus the decoder's f and d.
    assert {"h_o", "h_a", "h_l", "f", "d"} <= set(params)


def test_released_mode_has_no_dynamics():
    cfg = ILConfig(**SMALL)
    _, _, params, *_ = _fixture(cfg)
    assert "d" not in params and "h_a" not in params


def test_dynamics_receives_gradient():
    """If d_theta got no gradient, Eq. 7 would be decorative."""
    cfg = ILConfig.paper(**SMALL, solver=SolverConfig(n_iters=200))
    m, solver, params, obs, act, rng = _fixture(cfg)
    g = jax.grad(lambda p: il_loss(m, solver, p, obs, act, rng)[0])(params)
    dn = jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree.leaves(g["d"])))
    assert jnp.isfinite(dn) and dn > 1e-8


def test_chunk_len_follows_planning_horizon():
    """Eq. 7 sums l = t..t+H, so H decides H+1 actions."""
    for H in (1, 3, 19):
        c = ILConfig.paper(**SMALL, planning_horizon=H)
        assert c.chunk_len == H + 1


def test_horizon_zero_reduces_to_no_dynamics_call():
    """With H=0 there is one action and d_theta is never applied."""
    cfg = ILConfig.paper(**SMALL, planning_horizon=0)
    m, _, params, obs, act, _ = _fixture(cfg)
    z = jnp.zeros((m.z_dim,))
    a = jnp.zeros((cfg.action_dim,))
    cost = m.apply({"params": params}, z, a, method=DiffTORIPolicy.plan_cost)
    f_only = m.apply({"params": params}, z, a, method=lambda s, z_, a_: s.f(z_, a_))
    assert np.allclose(float(cost), float(-f_only))


# -- the adjoint stays exact with d_theta inside the Hessian ----------------

def test_implicit_gradient_matches_finite_differences():
    cfg = ILConfig.paper(**SMALL, solver=SolverConfig(n_iters=400, grad_tol=1e-12))
    m, solver, params, obs, act, rng = _fixture(cfg)
    f = lambda p: il_loss(m, solver, p, obs, act, rng)[0]
    gflat, _ = ravel_pytree(jax.grad(f)(params))
    pflat, unflat = ravel_pytree(params)
    idx = np.random.default_rng(0).choice(pflat.size, size=10, replace=False)
    eps = 1e-6
    fd = np.array([float((f(unflat(pflat.at[i].add(eps)))
                          - f(unflat(pflat.at[i].add(-eps)))) / (2 * eps))
                   for i in idx])
    an = np.array([float(gflat[i]) for i in idx])
    cos = fd @ an / (np.linalg.norm(fd) * np.linalg.norm(an))
    assert cos > 0.999, f"adjoint disagrees with finite differences: cos={cos}"


def test_inner_solve_reaches_stationarity():
    """The adjoint is exact only at a stationary point -- check the premise."""
    from difftori.policy_il import _latent
    cfg = ILConfig.paper(**SMALL, solver=SolverConfig(n_iters=400, grad_tol=1e-12))
    m, solver, params, obs, act, rng = _fixture(cfg)
    z, _, _ = _latent(m, params, obs, act, rng, sample=True)
    n = cfg.chunk_len * cfg.action_dim
    stat = solver.stationarity(jnp.zeros((z.shape[0], n), z.dtype), params, z)
    assert float(jnp.max(stat)) < 1e-3, f"non-stationary solve: {stat}"


# -- Table 8 benchmark ------------------------------------------------------

def test_pendulum_dynamics_match_amos_constants():
    from difftori.benchmarks import pendulum as P
    assert (P.DT, P.MAX_TORQUE, P.T_HORIZON) == (0.05, 2.0, 20)
    assert P.CTRL_PENALTY == 0.001
    assert np.allclose(np.asarray(P.GOAL_STATE), [1., 0., 0.])
    assert np.allclose(np.asarray(P.GOAL_WEIGHTS), [1., 1., 0.1])
    assert (P.SIMPLE.g, P.SIMPLE.m, P.SIMPLE.l) == (10., 1., 1.)
    assert (P.COMPLEX.d, P.COMPLEX.b, P.COMPLEX.simple) == (1.0, 0.1, False)


def test_pendulum_goal_state_is_the_cost_minimum():
    from difftori.benchmarks import pendulum as P
    at_goal = P.trajectory_cost(jnp.tile(P.GOAL_STATE, (P.T_HORIZON, 1)),
                                jnp.zeros((P.T_HORIZON, 1)))
    off = P.trajectory_cost(jnp.tile(jnp.array([-1., 0., 0.]), (P.T_HORIZON, 1)),
                            jnp.zeros((P.T_HORIZON, 1)))
    assert float(at_goal) < float(off)


def test_damped_dynamics_differ_from_simple():
    """The 'with damping' row must not silently be the same system."""
    from difftori.benchmarks import pendulum as P
    x = jnp.array([np.cos(1.0), np.sin(1.0), 0.5])
    u = jnp.array([0.3])
    assert not np.allclose(np.asarray(P.step(x, u, P.SIMPLE)),
                           np.asarray(P.step(x, u, P.COMPLEX)))


@pytest.mark.slow
def test_expert_reproduces_table8_expert_cost():
    """Expert cost must land near Table 8's 13.126 (w/o damping).

    The initial states come from numpy, not Torch's RNG, so the exact sample
    differs from Amos'; the tolerance is the measured seed-to-seed spread.
    """
    from difftori.benchmarks import pendulum as P
    x = jnp.asarray(P.sample_xinit(60, seed=0))
    u = P.solve_expert(x, P.SIMPLE, restarts=8)
    c = float(np.mean(np.asarray(P.policy_cost(x, u, P.SIMPLE))))
    assert 11.0 < c < 16.0, f"expert cost {c} far from Table 8's 13.126"
