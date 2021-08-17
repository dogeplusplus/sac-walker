import pytest
import jax.numpy as jnp

from models.mlp import MLP, relu, predict, linear


def test_relu():
    x = 4
    assert relu(x) == 4, "ReLU of positive is identity"
    y = -1
    assert relu(y) == 0, "ReLU of negative is 0"

def test_mlp():
    layer_sizes = [784, 512, 512, 10]
    model = MLP(layer_sizes)
    single = jnp.zeros((784))
    pred_single = model(single)
    assert pred_single.shape == (10,)

    batch = jnp.zeros((2, 784))
    pred_batch = model(batch)
    assert pred_batch.shape == (2, 10)

def test_training():
    # TODO:
    pass
