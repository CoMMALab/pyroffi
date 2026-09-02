"""Shape / gradient-correctness tests.  No environment or dataset required.

Run under float64 -- the implicit adjoint inverts the inner Hessian:

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python -m pytest tests -q
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difftori import (DiffTORIAgent, DiffTORIPolicy, ILConfig, RLConfig,
                      SolverConfig, StateCritic, act, difftori_loss, il_loss,
                      make_difftori_solver, make_il_solver, make_rl_solver,
                      plan, planning_horizon)
from difftori.pyroffi_trajopt import (make_dynamics_forward_solver,
                                      make_unrolled_forward_solver)

SOLVER = SolverConfig(n_iters=60)


def _il_cfg(**kw):
    kw.setdefault("horizon", 2)
    return ILConfig(action_dim=3, obs_dim=5, n_obs_steps=2,
                    obs_feature_dim=4, posterior_dim=4, mlp_hidden=8,
                    encoder_hidden=8, solver=SOLVER, **kw)


def _rl_cfg(**kw):
    return RLConfig(action_dim=2, obs_dim=5, latent_dim=4, mlp_hidden=8,
                    enc_hidden=8, solver=SOLVER, **kw)


# -- solver ---------------------------------------------------------------


def test_implicit_gradient_matches_closed_form():
    """C(x) = 1/2||x - theta - aux||^2 has the closed-form argmin x* = theta+aux."""
    solver = make_difftori_solver(
        lambda x, params, aux: 0.5 * jnp.sum((x - params - aux) ** 2),
        forward_solver=make_dynamics_forward_solver(n_iters=100))

    theta = jnp.array([0.3, -0.2, 1.1])
    aux = jnp.arange(6.0).reshape(2, 3)
    x0 = jnp.zeros((2, 3))

    x_star = solver.solve(x0, theta, aux)
    assert jnp.allclose(x_star, theta + aux, atol=1e-4)
    # dx*/dtheta = I, so d(sum x*)/dtheta = 1 per batch element.
    g = jax.grad(lambda t: jnp.sum(solver.solve_implicit(x0, t, aux)))(theta)
    assert jnp.allclose(g, 2.0, atol=1e-4)


def test_implicit_agrees_with_truncated_unrolling():
    """The two gradient paths must agree; if they don't, nothing downstream means
    anything.  Same precondition check as `ioc.robot.e1_identifiability`."""
    cost = lambda x, params, aux: jnp.sum(jnp.cosh(x - aux) * params)
    solver = make_difftori_solver(
        cost,
        forward_solver=make_dynamics_forward_solver(n_iters=200),
        unrolled_forward_solver=make_unrolled_forward_solver(
            n_iters=200, unroll_tail=8))

    aux = jnp.array([[0.5, -0.3]])
    x0 = jnp.zeros((1, 2))
    params = jnp.array(1.3)
    gi = jax.grad(lambda p: jnp.sum(solver.solve_implicit(x0, p, aux)))(params)
    gu = jax.grad(lambda p: jnp.sum(solver.solve_unrolled(x0, p, aux)))(params)
    assert jnp.allclose(gi, gu, atol=1e-4)


def test_stationarity_is_small_after_a_converged_solve():
    """The adjoint is exact only at a stationary point; this is the screen."""
    solver = make_difftori_solver(
        lambda x, params, aux: jnp.sum((x - aux) ** 4) + jnp.sum(x ** 2),
        forward_solver=make_dynamics_forward_solver(n_iters=200))
    aux = jnp.array([[0.4, -0.6]])
    s = solver.stationarity(jnp.zeros((1, 2)), None, aux)
    assert float(s[0]) < 1e-4


def test_initialisation_carries_no_gradient():
    solver = make_difftori_solver(
        lambda x, params, aux: 0.5 * jnp.sum((x - params) ** 2),
        forward_solver=make_dynamics_forward_solver(n_iters=50))
    g = jax.grad(lambda x0: jnp.sum(
        solver.solve_implicit(x0, jnp.ones(2), jnp.zeros((1, 2)))))(
            jnp.zeros((1, 2)))
    assert jnp.allclose(g, 0.0)


# -- imitation learning ---------------------------------------------------


@pytest.mark.parametrize("horizon", [1, 4])
def test_il_loss_and_gradients(horizon):
    cfg = _il_cfg(horizon=horizon)
    module = DiffTORIPolicy(cfg=cfg)
    solver = make_il_solver(module)
    rng = jax.random.PRNGKey(0)
    obs = jax.random.normal(rng, (4, cfg.n_obs_steps, cfg.obs_dim))
    expert = jax.random.normal(rng, (4, cfg.horizon, cfg.action_dim))
    params = module.init(rng, obs, expert)["params"]

    (loss, _), grads = jax.value_and_grad(
        lambda p: il_loss(module, solver, p, obs, expert, rng),
        has_aux=True)(params)
    assert jnp.isfinite(loss)
    assert all(jnp.all(jnp.isfinite(g)) for g in jax.tree.leaves(grads))
    # The cost function f must receive gradient THROUGH the optimiser -- this
    # is the whole point of the method.
    assert jnp.linalg.norm(jnp.concatenate(
        [g.ravel() for g in jax.tree.leaves(grads["f"])])) > 0
    # ...and so must the encoder, via the latent z.
    assert jnp.linalg.norm(jnp.concatenate(
        [g.ravel() for g in jax.tree.leaves(grads["h_o"])])) > 0


def test_act_shape_and_multimodality():
    cfg = _il_cfg()
    module = DiffTORIPolicy(cfg=cfg)
    solver = make_il_solver(module)
    rng = jax.random.PRNGKey(1)
    obs = jax.random.normal(rng, (2, cfg.n_obs_steps, cfg.obs_dim))
    params = module.init(
        rng, obs, jnp.zeros((2, cfg.horizon, cfg.action_dim)))["params"]

    a1 = act(module, solver, params, obs, jax.random.PRNGKey(2))
    a2 = act(module, solver, params, obs, jax.random.PRNGKey(3))
    assert a1.shape == (2, cfg.horizon, cfg.action_dim)
    # Different prior samples give different cost functions (Fig. 4).
    assert not jnp.allclose(a1, a2)


def test_actions_stay_inside_the_box():
    """The barrier is what makes the unconstrained inner problem well-posed."""
    cfg = _il_cfg()
    module = DiffTORIPolicy(cfg=cfg)
    solver = make_il_solver(module)
    rng = jax.random.PRNGKey(4)
    obs = jax.random.normal(rng, (4, cfg.n_obs_steps, cfg.obs_dim))
    params = module.init(
        rng, obs, jnp.zeros((4, cfg.horizon, cfg.action_dim)))["params"]
    a = act(module, solver, params, obs, rng)
    assert jnp.all(jnp.abs(a) < 5.0)


# -- model-based RL -------------------------------------------------------


def test_planning_horizon_schedule():
    cfg = RLConfig()
    assert planning_horizon(0, cfg) == 1
    assert planning_horizon(cfg.horizon_anneal_steps, cfg) == 5
    assert planning_horizon(10 * cfg.horizon_anneal_steps, cfg) == 5


def test_rl_loss_and_gradients():
    cfg = _rl_cfg()
    horizon = 2
    agent = DiffTORIAgent(cfg=cfg, horizon=horizon)
    solver = make_rl_solver(agent)
    critic = StateCritic(cfg=cfg)
    rng = jax.random.PRNGKey(0)
    obs = jax.random.normal(rng, (3, horizon + 1, cfg.obs_dim))
    action = jax.random.normal(rng, (3, horizon, cfg.action_dim))
    agent_params = agent.init(rng, obs[:, 0], action[:, 0])["params"]
    critic_params = critic.init(rng, obs[:, 0], action[:, 0])["params"]
    batch = {"obs": obs, "action": action,
             "reward": jnp.zeros((3, horizon)),
             "td_target": jnp.zeros((3, horizon))}

    (loss, _), grads = jax.value_and_grad(
        lambda p: difftori_loss(agent, solver, critic, p, agent_params,
                                critic_params, batch), has_aux=True)(agent_params)
    assert jnp.isfinite(loss)
    assert all(jnp.all(jnp.isfinite(g)) for g in jax.tree.leaves(grads))
    # The policy gradient must reach the encoder through the optimiser.
    assert jnp.linalg.norm(jnp.concatenate(
        [g.ravel() for g in jax.tree.leaves(grads["h"])])) > 0


def test_plan_shape():
    cfg = _rl_cfg()
    agent = DiffTORIAgent(cfg=cfg, horizon=3)
    solver = make_rl_solver(agent)
    rng = jax.random.PRNGKey(0)
    obs = jax.random.normal(rng, (2, cfg.obs_dim))
    params = agent.init(rng, obs, jnp.zeros((2, cfg.action_dim)))["params"]
    assert plan(agent, solver, params, obs).shape == (2, 3, cfg.action_dim)


# -- logging -------------------------------------------------------------


def test_logger_writes_scalars_and_config(tmp_path):
    from difftori.tblog import Logger, flatten_config

    cfg = _il_cfg()
    flat = flatten_config(cfg)
    assert flat["solver.n_iters"] == SOLVER.n_iters   # nested dataclass flattened

    with Logger("unittest", tmp_path, config=cfg) as log:
        log.log(0, {"loss": 1.0, "kl": 2.0})
        log.log(1, {"loss": 0.5, "kl": 1.5})
        run_dir = log.dir

    cfg_json = json.loads((run_dir / "config.json").read_text())
    assert cfg_json["config"]["horizon"] == cfg.horizon
    assert "git_commit" in cfg_json
    assert list(run_dir.glob("events.out.tfevents.*"))


def test_logger_disabled_is_a_no_op(tmp_path):
    from difftori.tblog import Logger

    with Logger("off", tmp_path, config=_il_cfg(), enabled=False) as log:
        log.log(0, {"loss": 1.0})
    assert not list(tmp_path.iterdir())


# -- dataset tooling -----------------------------------------------------


def test_joint_permutation_maps_by_name():
    """pyroffi and the URDF must be aligned by joint NAME, never positionally:
    a viewer that silently reorders joints draws a wrong pose convincingly."""
    from difftori.data.viser_playback import _joint_permutation

    pyroffi_names = ("j0", "j1", "j2")
    perm = _joint_permutation(pyroffi_names, ("j2", "j0", "j1"))
    q = np.array([10.0, 11.0, 12.0])
    assert list(q[perm]) == [12.0, 10.0, 11.0]

    with pytest.raises(RuntimeError, match="absent from the pyroffi model"):
        _joint_permutation(pyroffi_names, ("j0", "missing"))


# -- released-code fidelity ----------------------------------------------


def test_float32_data_solves_against_float64_params():
    """Regression: Flax defaults parameters to float32 and datasets are stored
    float32, but the L-BFGS engine's line-search constants are float64 under
    x64 -- so an unpromoted x0 changed dtype inside the while_loop carry and
    the engine raised several frames from the cause."""
    cfg = _il_cfg()
    module = DiffTORIPolicy(cfg=cfg)
    solver = make_il_solver(module)
    rng = jax.random.PRNGKey(0)
    obs = jnp.asarray(
        np.random.RandomState(0).randn(2, cfg.n_obs_steps, cfg.obs_dim),
        dtype=jnp.float32)
    expert = jnp.asarray(
        np.random.RandomState(1).randn(2, cfg.horizon, cfg.action_dim),
        dtype=jnp.float32)
    params = module.init(rng, obs, expert)["params"]
    loss, _ = il_loss(module, solver, params, obs, expert, rng)
    assert jnp.isfinite(loss)


def test_il_policy_has_no_latent_dynamics():
    """The released IL policy scores an action chunk with one network and has
    no dynamics model; `agent_rl` is the only side that rolls one out."""
    cfg = _il_cfg()
    module = DiffTORIPolicy(cfg=cfg)
    rng = jax.random.PRNGKey(0)
    obs = jax.random.normal(rng, (2, cfg.n_obs_steps, cfg.obs_dim))
    params = module.init(
        rng, obs, jnp.zeros((2, cfg.horizon, cfg.action_dim)))["params"]
    assert set(params) == {"h_o", "h_l", "f"}


def test_base_policy_initialisation_is_accepted_but_carries_no_gradient():
    """Their default initialises from a pretrained DP3 policy (`use_zero_initial:
    False`); the initialisation must still not leak gradient."""
    cfg = _il_cfg()
    module = DiffTORIPolicy(cfg=cfg)
    solver = make_il_solver(module)
    rng = jax.random.PRNGKey(7)
    obs = jax.random.normal(rng, (2, cfg.n_obs_steps, cfg.obs_dim))
    expert = jax.random.normal(rng, (2, cfg.horizon, cfg.action_dim))
    params = module.init(rng, obs, expert)["params"]
    base = 0.3 * jax.random.normal(rng, (2, cfg.horizon, cfg.action_dim))

    loss_with, _ = il_loss(module, solver, params, obs, expert, rng,
                           base_actions=base)
    assert jnp.isfinite(loss_with)
    g = jax.grad(lambda b: il_loss(module, solver, params, obs, expert, rng,
                                   base_actions=b)[0])(base)
    assert jnp.allclose(g, 0.0)


def test_rl_terminal_value_uses_the_policy_action():
    """`mbrl` evaluates the terminal Q at pi(z_H), so there are H decision
    variables, not H+1 as Eq. 4 of the paper writes."""
    cfg = _rl_cfg()
    agent = DiffTORIAgent(cfg=cfg, horizon=3)
    rng = jax.random.PRNGKey(0)
    obs = jax.random.normal(rng, (2, cfg.obs_dim))
    params = agent.init(rng, obs, jnp.zeros((2, cfg.action_dim)))["params"]
    z = agent.apply({"params": params}, obs, method=DiffTORIAgent.encode)
    cost = agent.apply({"params": params}, z[0],
                       jnp.zeros(3 * cfg.action_dim),
                       method=DiffTORIAgent.plan_cost)
    assert jnp.isfinite(cost)
