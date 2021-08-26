import jax.numpy as jnp

from jax import random

from models.mlp import (
    MLPActor,
    QFunction,
    relu,
    linear,
    init_network_params,
)


def test_linear():
    m, n = 64, 32
    layer = linear(m, n, random.PRNGKey(0))
    assert layer[0].shape == (n, m)
    assert layer[1].shape == (n,)


def test_relu():
    x = 4
    assert relu(x) == 4, "ReLU of positive is identity"
    y = -1
    assert relu(y) == 0, "ReLU of negative is 0"


def test_init_network_params():
    layers = [1, 2, 3]
    params = init_network_params(layers, key=random.PRNGKey(0))
    for i, param in enumerate(params):
        assert param[0].shape[::-1] == tuple(
            layers[i : i + 2]
        ), "Layer sizes not compatible"


def test_sac_actor():
    obs_dim = 1
    act_dim = 4
    hidden_sizes = [4, 8, 16]
    activation_fn = relu
    act_limit = 1e-2
    actor = MLPActor(
        obs_dim,
        act_dim,
        hidden_sizes,
        activation_fn,
        act_limit,
        random.PRNGKey(0)
    )

    single = jnp.zeros((obs_dim))
    mu, log_std = actor(single)
    assert mu.shape == (act_dim,)
    # TODO: verify that the log std of the probability should be dim 1
    assert log_std.shape == ()

    # Extra dimension for std deviation
    batch = jnp.zeros((2, obs_dim))
    mu, log_std = actor(batch)
    assert mu.shape == (2, act_dim)
    assert log_std.shape == (2,)


def test_sac_critic():
    obs_dim = 4
    act_dim = 5
    hidden_sizes = [4, 8, 16]
    activation_fn = relu
    critic = QFunction(obs_dim, act_dim, hidden_sizes, activation_fn, random.PRNGKey(0))

    input_dim = obs_dim + act_dim
    single = jnp.zeros((input_dim))
    pred_single = critic(single)
    assert pred_single.shape == (1,)

    batch = jnp.zeros((2, input_dim))
    pred_batch = critic(batch)
    assert pred_batch.shape == (2, 1)
