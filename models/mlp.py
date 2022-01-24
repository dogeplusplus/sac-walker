import jax.numpy as jnp

from jax import random
from flax import linen as nn
from jax.scipy.stats import norm


def relu(x):
    return jnp.maximum(0, x)


def softplus(x, beta=1, threshold=20):
    softp = 1.0 / beta * jnp.log(1 + jnp.exp(beta * x))
    return jnp.where(beta * x > threshold, x, softp)


class MLPActor(nn.Module):
    act_dim: int
    act_limit: float = 10
    log_min_std: float = -20
    log_max_std: float = 2

    def setup(self):
        self.layers = [
            nn.Dense(256),
            nn.Dense(256)
        ]
        self.mu_layer = nn.Dense(self.act_dim)
        self.log_std_layer = nn.Dense(self.act_dim)

    def __call__(self, x, key, deterministic=False, with_logprob=True):
        for layer in self.layers:
            x = layer(x)
            x = relu(x)

        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = jnp.clip(log_std, self.log_min_std, self.log_max_std)
        std = jnp.exp(log_std)

        prob = mu
        if not deterministic:
            prob += std * random.normal(key)

        logprob = None
        if with_logprob:
            logprob = jnp.sum(norm.logpdf(prob, mu, std), axis=-1)
            logprob -= 2 * jnp.sum(jnp.log(2) - prob - softplus(-2*prob), axis=-1)
        pi_action = jnp.tanh(prob) * self.act_limit

        return (pi_action, logprob)


class QFunction(nn.Module):
    def setup(self):
        self.layers = [
            nn.Dense(256),
            nn.Dense(256),
            nn.Dense(1)
        ]

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = relu(x)
        return x

class DoubleCritic(nn.Module):
    def setup(self):
        self.q1 = QFunction()
        self.q2 = QFunction()

    def __call__(self, x):
        return self.q1(x), self.q2(x)

