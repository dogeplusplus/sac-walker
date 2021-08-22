import numpy as np
import jax.numpy as jnp

from jax import grad, jit, vmap, random
from jax.scipy.stats import norm


def linear(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key,
                                 (n, m)), scale * random.normal(b_key, (n,))


def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [linear(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


def relu(x):
    return jnp.maximum(0, x)

def softplus(x, beta=1, threshold=20):
    if beta * x > threshold:
        return x
    return 1. / beta * jnp.log(1 + jnp.exp(beta * x))


def predict(params, image, activation_fn=relu):
    activations = image
    for w, b in params:
        outputs = jnp.dot(w, activations) + b
        activations = activation_fn(outputs)

    logits = activations
    return logits


batched_predict = vmap(predict, in_axes=(None, 0))


@jit
def update(params, x, y, loss, step_size):
    grads = grad(loss)(params, x, y)
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params, grads)]


class MLPActor(object):
    def __init__(self, obs_dim, act_dim, hidden_layers, activation_fn, act_limit, seed, log_min_std=-20, log_max_std=2):
        net_seed, mu_seed, std_seed = random.split(seed, 3)
        layer_sizes = [obs_dim] + hidden_layers
        self._activation_fn = activation_fn
        self._net_params = init_network_params(layer_sizes, net_seed)
        self._mu_params = init_network_params([layer_sizes[-1], act_dim], mu_seed)[0]
        self._log_std_params = init_network_params([layer_sizes[-1], act_dim], std_seed)[0]
        self._params = self._net_params + [self._mu_params, self._log_std_params]
        self._act_limit = act_limit

        self._log_min_std = log_min_std
        self._log_max_std = log_max_std


    @property
    def params(self):
        return self._params


    def __call__(self, x, deterministic=False, with_logprob=True):
        for w, b in self._net_params:
            x = jnp.dot(w, x) + b
            x = self._activation_fn(x)

        mu = jnp.dot(self._mu_params[0], x) + self._mu_params[1]
        log_std = jnp.dot(self._log_std_params[0], x) + self._log_std_params[1]
        log_std = jnp.clip(log_std, self._log_min_std, self._log_max_std)
        std = jnp.exp(log_std)

        prob = mu

        if not deterministic:
            seed = np.random.randint(0, 10000)
            prob += std * random.normal(random.PRNGKey(seed))

        logprob = None
        if with_logprob:
            logprob = norm.logpdf(prob, mu, std)
            logprob -= 2 * jnp.sum(jnp.log(2) - prob - softplus(-2*prob), axis=-1)

        pi_action = jnp.tanh(prob) * self._act_limit

        return (pi_action, logprob)


class QFunction(object):
    def __init__(self, obs_dim, act_dim, hidden_layers, activation_fn, seed):
        layer_sizes = [obs_dim + act_dim] + hidden_layers + [1]
        self._activation_fn = activation_fn
        self._net_params = init_network_params(layer_sizes, seed)
        self.batched_predict = vmap(self.predict, in_axes=(None,0))

    def __call__(self, x):
        if x.ndim == 1:
            return self.predict(x)
        else:
            return self.batched_predict(x)

    def predict(self, x):
        for w, b in self._net_params:
            x = jnp.dot(w, x) + b
            x = self._activation_fn(x)

        return x
