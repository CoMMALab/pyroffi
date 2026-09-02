"""Flax modules for DiffTORI.

Layer sizes and activations follow Appendix B of the paper verbatim
(ELU MLPs; the state-space twin Q uses a LayerNorm+Tanh trunk as in TD-MPC).
"""

from __future__ import annotations

from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import Array

__all__ = ["MLP", "StateEncoder", "ActionEncoder", "FusingEncoder",
           "LatentDynamics", "LatentReward", "LatentQ", "LatentPolicy",
           "StateQ", "TwinStateQ"]


class MLP(nn.Module):
    """ELU MLP; linear output layer."""

    hidden: Sequence[int]
    out_dim: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        for h in self.hidden:
            x = nn.elu(nn.Dense(h)(x))
        return nn.Dense(self.out_dim)(x)


class StateEncoder(nn.Module):
    """``h_theta^o``: one observation frame -> feature vector.

    The released code uses DP3's PointNet encoder (point clouds) or Robomimic's
    RNN encoder (images); anything with this signature can be substituted via
    the policy's ``obs_encoder`` argument.
    """

    latent_dim: int
    hidden: int = 256

    @nn.compact
    def __call__(self, obs: Array) -> Array:
        return MLP((self.hidden, self.hidden), self.latent_dim)(obs)


class ActionEncoder(nn.Module):
    """``h_theta^a``: expert action (chunk) -> latent feature vector.

    Appendix D lists this as one of the three CVAE encoder networks:
    ``z^a = h^a(a*_i)``, concatenated with ``z^s`` before the fusing encoder.
    The released code drops it and feeds the raw actions to ``h^l``, which is
    why ``ILConfig.paper_cvae=False`` bypasses this module.
    """

    latent_dim: int
    hidden: int = 256

    @nn.compact
    def __call__(self, action: Array) -> Array:
        return MLP((self.hidden, self.hidden), self.latent_dim)(action)


class FusingEncoder(nn.Module):
    """``h_theta^l``: [z^s, z^a] -> N(mu, sigma^2).

    Appendix D: the fusing encoder takes the concatenation of the state and
    action latent features.  Under ``paper_cvae`` the second argument is
    ``z^a = h^a(a*)``; under released-code fidelity it is the raw flattened
    expert chunk (they have no ``h^a``).  Outputs ``(mu, log_std)``; the
    released code splits one ``2 * z_dim`` head into mean and log-variance.
    """

    posterior_dim: int
    hidden: int = 256

    @nn.compact
    def __call__(self, z_s: Array, action: Array) -> tuple[Array, Array]:
        h = MLP((self.hidden, self.hidden), 2 * self.posterior_dim)(
            jnp.concatenate([z_s, action], axis=-1)
        )
        mu, log_std = jnp.split(h, 2, axis=-1)
        return mu, jnp.clip(log_std, -10.0, 2.0)


class LatentDynamics(nn.Module):
    """``d_theta(z, a) -> z'`` (App. B: 512-512 ELU)."""

    latent_dim: int
    hidden: int = 512

    @nn.compact
    def __call__(self, z: Array, action: Array) -> Array:
        return MLP((self.hidden, self.hidden), self.latent_dim)(
            jnp.concatenate([z, action], axis=-1)
        )


class LatentReward(nn.Module):
    """``R_theta(z, a) -> scalar`` (upstream: ``reward_network``).

    In imitation learning this is ``f_theta``: there is no ground-truth reward,
    and the function is defined only by the behaviour-cloning loss.  There it
    scores an entire action *chunk* at once, so ``action`` is the flattened
    chunk; in RL it scores one step.
    """

    hidden: int = 512

    @nn.compact
    def __call__(self, z: Array, action: Array) -> Array:
        out = MLP((self.hidden, self.hidden), 1)(
            jnp.concatenate([z, action], axis=-1)
        )
        return jnp.squeeze(out, axis=-1)


LatentQ = LatentReward  # same architecture, terminal value in Eq. 4


class LatentPolicy(nn.Module):
    """``pi_psi(z) -> a`` in [-1, 1]; supplies the solver's initialisation."""

    action_dim: int
    hidden: int = 512

    @nn.compact
    def __call__(self, z: Array) -> Array:
        return jnp.tanh(MLP((self.hidden, self.hidden), self.action_dim)(z))


class StateQ(nn.Module):
    """One head of the state-space Q used for the policy gradient (Eq. 6).

    Learned in the *original* state space S, not the latent space, so the
    policy gradient is not distorted by the encoder (Sec. 4.2).
    """

    latent_dim: int
    enc_hidden: int = 256
    hidden: int = 512

    @nn.compact
    def __call__(self, obs: Array, action: Array) -> Array:
        z = nn.Dense(self.latent_dim)(nn.elu(nn.Dense(self.enc_hidden)(obs)))
        x = jnp.concatenate([z, action], axis=-1)
        x = jnp.tanh(nn.LayerNorm()(nn.Dense(self.hidden)(x)))
        x = nn.elu(nn.Dense(self.hidden)(x))
        return jnp.squeeze(nn.Dense(1)(x), axis=-1)


class TwinStateQ(nn.Module):
    """Clipped double-Q: ``Q_s1``/``Q_s2`` of App. B."""

    latent_dim: int
    enc_hidden: int = 256
    hidden: int = 512

    @nn.compact
    def __call__(self, obs: Array, action: Array) -> tuple[Array, Array]:
        q1 = StateQ(self.latent_dim, self.enc_hidden, self.hidden, name="Q_s1")
        q2 = StateQ(self.latent_dim, self.enc_hidden, self.hidden, name="Q_s2")
        return q1(obs, action), q2(obs, action)
