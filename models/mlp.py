import optax
import numpy as np
import jax.numpy as jnp

from jax import random
from flax import struct
from flax import linen as nn
from jax.scipy.stats import norm
from flax.training import train_state

from typing import Callable, Sequence, Optional


def relu(x):
    return jnp.maximum(0, x)


def softplus(x, beta=1, threshold=20):
    softp = 1.0 / beta * jnp.log(1 + jnp.exp(beta * x))
    return jnp.where(beta * x > threshold, x, softp)


@struct.dataclass
class Actor(nn.Module):
    act_dim: int
    hidden_layers: Sequence[int]
    activation_fn: Callable
    act_limit: float
    seed: int
    tx: Optional[optax.GradientTransformation]
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

    def create_pi_train_state(self):
        seed = random.PRNGKey(self.seed)
        seed1, seed2 = random.split(seed)
        init_tensor = random.uniform(seed1, (self.obs_dim,))
        params_pi = self.pi.init(seed2, init_tensor)

        return train_state.TrainState.create(
            apply_fn=self.pi.apply,
            params=params_pi,
            tx=self.tx,
        )

    def act(self, x, deterministic=False):
        a, _ = self.__call__(x, deterministic, False)
        return a

class Critic(nn.Module):
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


@struct.dataclass
class DoubleCritic(nn.Module):
    hidden_layers: Sequence[int]
    acitvation_fn: Callable
    seed: int
    opt_state: Optional[optax.OptState]
    tx: Optional[optax.GradientTransformation]

    def setup(self):
        self.q1 = Critic(
            hidden_layers=self.hidden_layers,
            activation_fn=self.activation_fn
        )
        self.q2 = Critic(
            hidden_layers=self.hidden_layers,
            activation_fn=self.activation_fn
        )

        self.opt_state = self.create_train_state()


    def __call__(self, states, actions):
        x = jnp.concatenate([states, actions], axis=-1)
        q1 = self.q1(x)
        q2 = self.q2(x)

        return (q1, q2)


    def create_train_state(self):
        seed = random.PRNGKey(self.seed)
        seed1, seed2, seed3 = random.split(seed, 3)
        init_tensor = random.uniform(seed1, (self.obs_dim + self.act_dim,))
        params_q1 = self.q1.init(seed2, init_tensor)
        params_q2 = self.q2.init(seed3, init_tensor)

        return train_state.TrainState.create(
            apply_fn=self.apply,
            params={"q1": params_q1, "q2": params_q2},
            tx=self.tx,
        )
