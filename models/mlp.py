import time
import flax
import jax.numpy as jnp
import tensorflow_datasets as tfds

from jax import grad, jit, vmap, random
from jax.scipy.special import logsumexp


def linear(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key,
                                 (n, m)), scale * random.normal(b_key, (n, ))


def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [linear(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


def relu(x):
    return jnp.maximum(0, x)


def predict(params, image, activation_fn=relu):
    activations = image
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = activation_fn(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return logits - logsumexp(logits)


batched_predict = vmap(predict, in_axes=(None, 0))


@jit
def update(params, x, y):
    grads = grad(loss)(params, x, y)
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params, grads)]


class MLP(object):
    def __init__(self, layer_sizes, seed):
        self.layer_sizes= layer_sizes
        self.params = init_network_params(layer_sizes, random.PRNGKey(seed))

    def __call__(self, x):
        if x.ndim == 2:
            return batched_predict(self.params, x)
        else:
            return predict(self.params, x)

    def update(self, x, y):
        update(self.params, x, y)


class MLPActor(MLP):
    def __init__(self, obs_dim, act_dim, hidden_layers, seed):
        layer_sizes = [obs_dim] + hidden_layers + [act_dim]
        super().__init__(layer_sizes, seed)


class QFunction(MLP):
    def __init__(self, obs_dim, act_dim, hidden_layers, seed):
        layer_sizes = [obs_dim + act_dim] + hidden_layers + [1]
        super().__init__(layer_sizes, seed)
