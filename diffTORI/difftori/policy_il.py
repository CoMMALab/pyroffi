"""DiffTORI for imitation learning, as released.

This follows ``diffusion_policy_3d/policy/difftori.py`` from the authors' repo
rather than Eq. 7 of the paper, because the two disagree and the code is what
produced the reported results.  The difference is not cosmetic:

    paper    a(theta) = argmax_a sum_l gamma^l f(z_l, a_l),  z_{l+1} = d(z_l, a_l)
    released a(theta) = argmax_a f(z, a_{0:H})           -- no dynamics model

The released policy has **no latent dynamics function at all**.  One network
scores an entire action *chunk* in one shot, and ``planning_horizon`` is fixed
at 1 and never used.  ``horizon`` (4) is the chunk length, not a planning
horizon; ``n_obs_steps`` (2) frames of observation are encoded and concatenated.

What remains from the paper is the part that matters: the policy is a CVAE
whose **decoder is a trajectory optimiser**, so actions come from test-time
optimisation of a learned cost and the imitation loss is back-propagated
through the solve.

    encoder:  z^s = h^o(o_{t-1:t}),  (mu, sigma) = h^l(z^s, a*_{0:H})
              z = [z^s, z~],  z~ ~ N(mu, sigma^2)
    decoder:  a(theta) = argmin_a  -f_theta(z, a) + barrier
    loss:     w * ||a(theta) - a*||^2 + beta * KL(N(mu,sigma^2) || N(0,I))

with ``w = 3000`` and ``beta = 10`` from the released code.

Two deliberate departures from the released implementation, both documented at
their call sites: we solve each batch element as its own problem rather than
coupling the whole batch through one averaged residual, and we bound the
actions with a smooth barrier rather than a hard ``clamp`` (which zeroes the
gradient outside the box exactly where the solver needs it).
"""

from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import Array

from .config import ILConfig
from .networks import (ActionEncoder, FusingEncoder, LatentDynamics,
                       LatentReward, StateEncoder)
from .solver import DiffTORISolver, make_difftori_solver

__all__ = ["DiffTORIPolicy", "make_solver", "il_loss", "act"]


class DiffTORIPolicy(nn.Module):
    """CVAE with a trajectory-optimisation decoder.

    Args:
        obs_encoder: replaces the MLP observation encoder.  The released code
            uses DP3's PointNet encoder (point clouds) or Robomimic's RNN
            encoder (images); anything mapping ``(B, obs_dim) -> (B,
            obs_feature_dim)`` drops in.
    """

    cfg: ILConfig
    obs_encoder: nn.Module | None = None

    @property
    def z_dim(self) -> int:
        """Width of the decoder latent ``z`` that ``f`` and ``d`` consume."""
        c = self.cfg
        return c.n_obs_steps * c.obs_feature_dim + c.posterior_dim

    def setup(self):
        c = self.cfg
        self.h_o = self.obs_encoder or StateEncoder(c.obs_feature_dim,
                                                    c.encoder_hidden)
        if c.paper_cvae:
            # Appendix D's third encoder network.  The released code has no
            # h^a and feeds raw actions straight into h^l.
            self.h_a = ActionEncoder(c.posterior_dim, c.encoder_hidden)
        self.h_l = FusingEncoder(c.posterior_dim, c.mlp_hidden)
        # f_theta ("reward_network" upstream).  Under Eq. 7 it scores ONE step,
        # (z_l, a_l); under released-code fidelity it scores the whole flat
        # chunk in a single call.
        self.f = LatentReward(c.mlp_hidden)
        if c.use_dynamics:
            # d_theta: the piece Eq. 7 and Appendix D require and the released
            # code omits entirely.  It maps the full decoder latent forward.
            self.d = LatentDynamics(self.z_dim, c.mlp_hidden)

    # -- CVAE encoder ----------------------------------------------------
    def encode_obs(self, obs: Array) -> Array:
        """``(B, n_obs_steps, obs_dim) -> (B, n_obs_steps * obs_feature_dim)``."""
        b, n = obs.shape[0], obs.shape[1]
        feats = self.h_o(obs.reshape(b * n, -1))
        return feats.reshape(b, n * self.cfg.obs_feature_dim)

    def posterior(self, obs_features: Array, expert_chunk: Array):
        """``h^l([z^s, z^a]) -> (mu, log_std)``.

        Appendix D: ``z^a = h^a(a*_i)``.  Released-code mode has no ``h^a``, so
        the flattened expert chunk enters ``h^l`` raw.
        """
        flat = expert_chunk.reshape(expert_chunk.shape[0], -1)
        z_a = self.h_a(flat) if self.cfg.paper_cvae else flat
        return self.h_l(obs_features, z_a)

    # -- decoder objective ------------------------------------------------
    def plan_cost(self, z: Array, actions_flat: Array) -> Array:
        """``C(a) = -(discounted return) + barrier``, for ONE problem.

        Eq. 7 (``use_dynamics=True``)::

            a(theta) = argmax_a  sum_{l=0..H} gamma^l f_theta(z_l, a_l)
                       s.t.      z_{l+1} = d_theta(z_l, a_l),  z_0 = z

        The constraint is substituted into the objective rather than imposed,
        exactly as the paper does for Eq. 5 -- Theseus forced them to eliminate
        it, and eliminating it is also what lets a plain unconstrained L-BFGS
        engine solve it.  ``H = planning_horizon``, so ``H + 1`` actions are
        decided and ``d_theta`` is applied ``H`` times.

        Released-code mode (``use_dynamics=False``) keeps the flat-chunk form:
        one ``f_theta`` call scoring all ``horizon`` actions at once, with no
        dynamics model anywhere.

        Either way the cost minimised is the negated return.  The released code
        minimises ``(1000 - f)^2`` because Theseus accepts only nonlinear least
        squares; squaring a positively-shifted objective leaves the argmin
        alone, and our solver minimises an arbitrary scalar, so we drop the
        shift-and-square.

        The barrier replaces their ``torch.clamp(a, -1, 1)`` inside the cost.  A
        clamp makes the objective exactly flat outside the box, so a solver that
        starts or steps outside gets no gradient back; a one-sided quadratic
        pushes it back in and is inactive on solutions already inside.
        """
        c = self.cfg
        barrier = jnp.sum(jnp.maximum(jnp.abs(actions_flat) - 1.0, 0.0) ** 2)

        if not c.use_dynamics:
            value = self.f(z, actions_flat)
            return -value + c.solver.action_penalty * barrier

        actions = actions_flat.reshape(c.chunk_len, c.action_dim)

        # Unrolled Python loop rather than lax.scan: the body calls Flax
        # submodules, and creating their parameters inside a JAX transform is
        # exactly what Flax refuses to do during `init`.  chunk_len is small
        # (2 at the paper's H = 1), so the unroll costs nothing.
        z_l, total = z, jnp.zeros((), z.dtype)
        for l in range(c.chunk_len):
            total = total + (c.discount ** l) * self.f(z_l, actions[l])
            if l + 1 < c.chunk_len:      # no d_theta call after the last action
                z_l = self.d(z_l, actions[l])
        return -total + c.solver.action_penalty * barrier

    def __call__(self, obs: Array, expert_chunk: Array):
        """Trace every submodule once so ``init`` sees all parameters."""
        c = self.cfg
        obs_features = self.encode_obs(obs)
        mu, _ = self.posterior(obs_features, expert_chunk)
        z = (jnp.concatenate([mu, obs_features], axis=-1) if c.paper_cvae
             else jnp.concatenate([obs_features, mu], axis=-1))
        return z, self.plan_cost(
            z[0], jnp.zeros(c.chunk_len * c.action_dim, z.dtype))


def make_solver(
    module: DiffTORIPolicy,
    forward_solver=None,
    unrolled_forward_solver=None,
) -> DiffTORISolver:
    """Build the inner solver; defaults to ``pyroffi``'s L-BFGS engine.

    Each batch element is solved as its own problem (``jax.vmap``).  The
    released code instead flattens the batch into a single Theseus problem and
    averages the residual over it, which couples every sample's actions through
    one shared cost -- an artefact of fitting a batch into one Theseus layer,
    not part of the method, and it forces their fixed 128-sample padding.
    """
    cfg = module.cfg
    if forward_solver is None:
        from .pyroffi_trajopt import make_dynamics_forward_solver

        forward_solver = make_dynamics_forward_solver(
            n_iters=cfg.solver.n_iters, grad_tol=cfg.solver.grad_tol,
            m_lbfgs=cfg.solver.m_lbfgs, smooth=cfg.solver.smooth)

    def cost(x, params, z):
        return module.apply({"params": params}, z, x,
                            method=DiffTORIPolicy.plan_cost)

    return make_difftori_solver(
        cost, forward_solver=forward_solver,
        unrolled_forward_solver=unrolled_forward_solver,
        adjoint_ridge=cfg.solver.adjoint_ridge)


def _latent(module, params, obs, expert_chunk, rng, sample: bool):
    """``z = [z^s, z~]``; ``z~`` from the posterior when training, else the prior."""
    obs_features = module.apply({"params": params}, obs,
                                method=DiffTORIPolicy.encode_obs)
    if sample:
        mu, log_std = module.apply({"params": params}, obs_features,
                                   expert_chunk,
                                   method=DiffTORIPolicy.posterior)
        std = jnp.exp(log_std)
        z_tilde = mu + std * jax.random.normal(rng, mu.shape, mu.dtype)
    else:
        mu = std = None
        z_tilde = jax.random.normal(
            rng, (obs_features.shape[0], module.cfg.posterior_dim),
            obs_features.dtype)
    # Appendix D writes z = [z~, z^s]; the released code concatenates the other
    # way round.  The order is arbitrary in itself -- it only has to agree
    # between il_loss and act, and with whatever d_theta was trained on.
    parts = ([z_tilde, obs_features] if module.cfg.paper_cvae
             else [obs_features, z_tilde])
    return jnp.concatenate(parts, axis=-1), mu, std


def _initial_actions(cfg: ILConfig, z: Array, base_actions, rng) -> Array:
    """``a_init``: a base policy's chunk, else zeros (their ``use_zero_initial``).

    The released default initialises from a *pretrained DP3 policy* and
    optionally perturbs it by ``expert_noise``, which makes DiffTORI a
    refinement of that policy rather than a policy trained from scratch.  Pass
    ``base_actions`` to reproduce that; the initialisation carries no gradient
    either way (see ``solver``).
    """
    n = cfg.chunk_len * cfg.action_dim
    if base_actions is None:
        return jnp.zeros((z.shape[0], n), z.dtype)
    x0 = base_actions.reshape(z.shape[0], n)
    if cfg.init_noise > 0:
        x0 = x0 + cfg.init_noise * jax.random.normal(rng, x0.shape, x0.dtype)
    return jax.lax.stop_gradient(x0)


def _plan(solver, cfg: ILConfig, params, z, base_actions, rng) -> Array:
    x0 = _initial_actions(cfg, z, base_actions, rng)
    x_star = solver.solve_implicit(x0, params, z)
    return x_star.reshape(z.shape[0], cfg.chunk_len, cfg.action_dim)


def il_loss(
    module: DiffTORIPolicy,
    solver: DiffTORISolver,
    params: Any,
    obs: Array,
    expert_chunk: Array,
    rng: Array,
    base_actions: Array | None = None,
) -> tuple[Array, dict[str, Array]]:
    """``w * MSE(a(theta), a*) + beta * KL``.

    Args:
        obs:          ``(B, n_obs_steps, obs_dim)``.
        expert_chunk: ``(B, horizon, action_dim)``, normalised to [-1, 1].
        base_actions: optional ``a_init`` from a pretrained base policy.

    The whole chunk is supervised, not just the first action: the released code
    compares all ``horizon`` steps.
    """
    cfg = module.cfg
    rng_z, rng_init = jax.random.split(rng)
    z, mu, std = _latent(module, params, obs, expert_chunk, rng_z, sample=True)
    actions = _plan(solver, cfg, params, z, base_actions, rng_init)

    recon = jnp.mean((actions - expert_chunk) ** 2)
    kl = jnp.mean(jnp.sum(0.5 * (mu ** 2 + std ** 2 - 1.0) - jnp.log(std),
                          axis=-1))
    loss = cfg.action_loss_weight * recon + cfg.kl_coefficient * kl
    return loss, {"loss": loss, "recon": recon, "kl": kl}


def act(
    module: DiffTORIPolicy,
    solver: DiffTORISolver,
    params: Any,
    obs: Array,
    rng: Array,
    base_actions: Array | None = None,
) -> Array:
    """Test-time action chunk: prior sample -> trajectory optimisation.

    Returns ``(B, horizon, action_dim)``; the runner executes the first
    ``n_action_steps`` of it.  The CVAE encoder is unused here, so different
    prior samples give different cost functions and hence multi-modal actions
    (Fig. 4).
    """
    cfg = module.cfg
    rng_z, rng_init = jax.random.split(rng)
    dummy = jnp.zeros((obs.shape[0], cfg.chunk_len, cfg.action_dim), obs.dtype)
    z, _, _ = _latent(module, params, obs, dummy, rng_z, sample=False)
    return _plan(solver, cfg, params, z, base_actions, rng_init)
