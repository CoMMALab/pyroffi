"""The 'LSTM policy' row of Table 8 -- Amos et al.'s ``nn`` baseline.

Amos Sec. 5.3: "``nn`` is an LSTM that takes the state ``x`` as input and
predicts the nominal action sequence.  In this setting we optimize the imitation
loss directly."  Trained with Adam at ``1e-4`` (their Sec. 5.3), against the same
demonstrations DiffTORI sees, and scored with the same ``policy_cost``.

This is a *baseline*, so it deliberately has no trajectory optimisation and no
dynamics model: the whole point of the row is what a generic sequence model
achieves on the same data.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from jax import Array

__all__ = ["LSTMPolicy", "train_lstm"]


class LSTMPolicy(nn.Module):
    """``x_init -> u_{1:T}``, one torque per unrolled step."""

    horizon: int
    action_dim: int = 1
    hidden: int = 128

    @nn.compact
    def __call__(self, x: Array) -> Array:
        cell = nn.OptimizedLSTMCell(self.hidden)
        carry = cell.initialize_carry(jax.random.PRNGKey(0), (x.shape[0], self.hidden))
        head = nn.Dense(self.action_dim)
        feat = nn.relu(nn.Dense(self.hidden)(x))
        outs = []
        for _ in range(self.horizon):
            carry, y = cell(carry, feat)
            outs.append(jnp.tanh(head(y)))   # actions live in the unit box
        return jnp.stack(outs, axis=1)


def train_lstm(x_train, u_train, x_val, u_val, horizon: int, action_dim: int = 1,
               steps: int = 4000, lr: float = 1e-4, batch_size: int = 32,
               seed: int = 0, hidden: int = 128):
    """Adam at ``lr`` on the imitation loss; returns (module, best params).

    Selection is on validation MSE, mirroring Amos' "we select the best
    validation loss observed during the training run".
    """
    module = LSTMPolicy(horizon=horizon, action_dim=action_dim, hidden=hidden)
    rng = jax.random.PRNGKey(seed)
    params = module.init(rng, jnp.asarray(x_train[:1]))["params"]
    state = TrainState.create(apply_fn=module.apply, params=params,
                              tx=optax.adam(lr))

    x_train, u_train = jnp.asarray(x_train), jnp.asarray(u_train)
    x_val, u_val = jnp.asarray(x_val), jnp.asarray(u_val)

    @jax.jit
    def step(state, xb, ub):
        def loss_fn(p):
            return jnp.mean((state.apply_fn({"params": p}, xb) - ub) ** 2)
        loss, g = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=g), loss

    @jax.jit
    def val_loss(p):
        return jnp.mean((module.apply({"params": p}, x_val) - u_val) ** 2)

    n = x_train.shape[0]
    best, best_params = jnp.inf, state.params
    rs = np.random.default_rng(seed)
    for i in range(steps):
        idx = rs.choice(n, size=min(batch_size, n), replace=False)
        state, _ = step(state, x_train[idx], u_train[idx])
        if i % 50 == 0:
            v = val_loss(state.params)
            if v < best:
                best, best_params = v, state.params
    return module, best_params
