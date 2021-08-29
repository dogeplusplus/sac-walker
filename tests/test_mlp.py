import jax.numpy as jnp

from jax import random

from models.mlp import (
    MLPActor,
    QFunction,
    relu,
)


def test_relu():
    x = 4
    assert relu(x) == 4, "ReLU of positive is identity"
    y = -1
    assert relu(y) == 0, "ReLU of negative is 0"


def test_sac_actor():
    obs_dim = 1
    act_dim = 4
    hidden_layers= [4, 8, 16]
    activation_fn = relu
    act_limit = 1e-2
    actor = MLPActor(
        act_dim=act_dim,
        hidden_layers=hidden_layers,
        activation_fn=activation_fn,
        act_limit=act_limit,
    )

    seed1, seed2 = random.split(random.PRNGKey(0))
    x = random.uniform(seed1, (obs_dim,))
    params = actor.init(seed2, x)

    single = jnp.zeros((obs_dim))
    mu, log_std = actor.apply(params, single)
    assert mu.shape == (act_dim,)
    # TODO: verify that the log std of the probability should be dim 1
    assert log_std.shape == ()

    # Extra dimension for std deviation
    batch = jnp.zeros((2, obs_dim))
    mu, log_std = actor.apply(params, batch)
    assert mu.shape == (2, act_dim)
    assert log_std.shape == (2,)


def test_sac_critic():
    obs_dim = 4
    act_dim = 5
    hidden_layers = [4, 8, 16]
    activation_fn = relu
    critic = QFunction(
        hidden_layers=hidden_layers,
        activation_fn=activation_fn,
    )

    seed1, seed2 = random.split(random.PRNGKey(0))
    input_dim = obs_dim + act_dim
    x = random.uniform(seed1, (input_dim,))
    params = critic.init(seed2, x)

    single = jnp.zeros((input_dim))
    pred_single = critic.apply(params, single)
    assert pred_single.shape == (1,)

    batch = jnp.zeros((2, input_dim))
    pred_batch = critic.apply(params, batch)
    assert pred_batch.shape == (2, 1)
