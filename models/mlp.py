import optax
import numpy as np
import jax.numpy as jnp

from jax import random
from flax import linen as nn
from jax.scipy.stats import norm
from flax.training import train_state

from typing import Callable, Sequence


def relu(x):
    return jnp.maximum(0, x)


def softplus(x, beta=1, threshold=20):
    softp = 1.0 / beta * jnp.log(1 + jnp.exp(beta * x))
    return jnp.where(beta * x > threshold, x, softp)


class MLPActor(nn.Module):
    act_dim: int
    hidden_layers: Sequence[int]
    activation_fn: Callable
    act_limit: float
    log_min_std: float = -20
    log_max_std: float = 2

    def setup(self):
        self.layers = [nn.Dense(s) for s in self.hidden_layers]
        self.mu_layer = nn.Dense(self.act_dim)
        self.log_std_layer = nn.Dense(self.act_dim)

    def __call__(self, x, deterministic=False, with_logprob=True):
        for layer in self.layers:
            x = layer(x)
            x = self.activation_fn(x)

        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = jnp.clip(log_std, self.log_min_std, self.log_max_std)
        std = jnp.exp(log_std)

        prob = mu
        if not deterministic:
            seed = np.random.randint(0, 100000)
            prob += std * random.normal(random.PRNGKey(seed))

        logprob = None
        if with_logprob:
            logprob = jnp.sum(norm.logpdf(prob, mu, std), axis=-1)
            logprob -= 2 * jnp.sum(jnp.log(2) - prob - softplus(-2*prob), axis=-1)
        pi_action = jnp.tanh(prob) * self.act_limit

        return (pi_action, logprob)


class QFunction(nn.Module):
    hidden_layers: Sequence[int]
    activation_fn: Callable

    def setup(self):
        self.layers = [nn.Dense(s) for s in self.hidden_layers + (1,)]

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = self.activation_fn(x)
        return x


class ActorCritic(object):
    def __init__(self, obs_dim, act_dim, hidden_layers, activation_fn, act_limit, learning_rate, seed):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.pi = MLPActor(
            act_dim, hidden_layers, activation_fn, act_limit
        )
        self.q1 = QFunction(hidden_layers, activation_fn)
        self.q2 = QFunction(hidden_layers, activation_fn)

        seed1, seed2 = random.split(seed)
        self.q_state = self.create_q_train_state(learning_rate, seed1)
        self.pi_state = self.create_pi_train_state(learning_rate, seed2)

    def create_q_train_state(self, learning_rate, seed):
        seed1, seed2, seed3 = random.split(seed, 3)
        init_tensor = random.uniform(seed1, (self.obs_dim + self.act_dim,))
        params_q1 = self.q1.init(seed2, init_tensor)
        params_q2 = self.q2.init(seed3, init_tensor)
        tx = optax.adam(learning_rate)

        return train_state.TrainState.create(
            apply_fn=self.q1.apply,
            params={"q1": params_q1, "q2": params_q2},
            tx=tx,
        )

    def create_pi_train_state(self, learning_rate, seed):
        seed1, seed2 = random.split(seed)
        init_tensor = random.uniform(seed1, (self.obs_dim,))
        params_pi = self.pi.init(seed2, init_tensor)
        tx = optax.adam(learning_rate)

        return train_state.TrainState.create(
            apply_fn=self.pi.apply,
            params=params_pi,
            tx=tx,
        )
