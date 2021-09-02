import gym
import pytest
import jax.numpy as jnp

from jax import random
from copy import deepcopy

from models.mlp import ActorCritic, relu
from sac.algorithm import SAC, TrainingParameters


def first_kernel(weights):
    return weights["params"]["layers_0"]["kernel"]


def first_bias(weights):
    return weights["params"]["layers_0"]["bias"]


@pytest.fixture
def sac():
    env = gym.make("BipedalWalker-v3")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    hidden_sizes = [256, 256]
    activation_fn = relu
    act_limit = 1e-2
    learning_rate = 1e-3
    ac = ActorCritic(
        obs_dim,
        act_dim,
        hidden_sizes,
        activation_fn,
        act_limit,
        learning_rate,
        random.PRNGKey(0),
    )

    params = TrainingParameters(
        epochs=1,
        steps_per_epoch=1,
        batch_size=2,
        start_steps=0,
        update_after=1,
        update_every=1,
        learning_rate=1e-3,
        alpha=0.5,
        gamma=0.99,
        polyak=0.5,
        buffer_size=100,
    )

    soft_actor_critic = SAC(env, ac, params)
    s = env.reset()
    for _ in range(params.buffer_size):
        a = env.action_space.sample()
        s2, r, d, _ = env.step(a)
        soft_actor_critic.buffer.store(s, a, r, s2, d)
    return soft_actor_critic


@pytest.fixture
def samples(sac):
    samples = sac.buffer.sample(100)
    return samples


def test_q_update(sac, samples):
    q1_old = deepcopy(sac.ac.q_state.params["q1"])
    q2_old = deepcopy(sac.ac.q_state.params["q2"])

    new_state, _ = sac.update_q(sac.ac.q_state, samples)

    q1_new = new_state.params["q1"]
    q2_new = new_state.params["q2"]

    q1_updated = jnp.any(first_kernel(q1_old) != first_kernel(q1_new))
    q2_updated = jnp.any(first_kernel(q2_old) != first_kernel(q2_new))
    assert q1_updated or q2_updated, "Q networks did not update."


def test_q_loss(sac, samples):
    loss_q = sac.q_loss(
        sac.ac.q_state.params,
        samples.states,
        samples.actions,
        samples.rewards,
        samples.next_states,
        samples.done,
    )

    assert loss_q >= 0, "Loss should be at least 0."


def test_pi_loss(sac, samples):
    pi_weights = sac.ac.pi_state.params
    loss_pi = sac.pi_loss(pi_weights, samples.states, samples.next_states)
    assert loss_pi != 0, "Loss should be non-zero 0 for gradient ascent."


def test_pi_update(sac, samples):
    pi_old = deepcopy(sac.ac.pi_state.params)

    new_state, _ = sac.update_pi(sac.ac.pi_state, samples)
    pi_new = new_state.params

    assert jnp.any(first_kernel(pi_old) != first_kernel(pi_new))
    assert jnp.any(first_bias(pi_old) != first_bias(pi_new))


def test_target_update(sac, samples):
    q1_targ_params = deepcopy(sac.ac_targ.q_state.params["q1"])
    q2_targ_params = deepcopy(sac.ac_targ.q_state.params["q2"])

    # Perform 1 step of gradient updates
    sac.ac.q_state, _ = sac.update_q(sac.ac.q_state, samples)

    q1_params = deepcopy(sac.ac.q_state.params["q1"])
    q2_params = deepcopy(sac.ac.q_state.params["q2"])
    sac.update_targets()

    expected_q1_weights = 0.5 * (first_kernel(q1_targ_params) + first_kernel(q1_params))
    expected_q2_weights = 0.5 * (first_kernel(q2_targ_params) + first_kernel(q2_params))

    actual_q1_weights = first_kernel(sac.ac_targ.q_state.params["q1"])
    actual_q2_weights = first_kernel(sac.ac_targ.q_state.params["q2"])

    assert jnp.allclose(expected_q1_weights, actual_q1_weights), "Polyak weight averaging failed for Q1"
    assert jnp.allclose(expected_q2_weights, actual_q2_weights), "Polyak weight averaging failed for Q2"

    expected_q1_biases = 0.5 * (first_bias(q1_targ_params) + first_bias(q1_params))
    expected_q2_biases = 0.5 * (first_bias(q2_targ_params) + first_bias(q2_params))

    actual_q1_biases = first_bias(sac.ac_targ.q_state.params["q1"])
    actual_q2_biases = first_bias(sac.ac_targ.q_state.params["q2"])

    assert jnp.allclose(expected_q1_biases, actual_q1_biases), "Polyak bias averaging failed for Q1"
    assert jnp.allclose(expected_q2_biases, actual_q2_biases), "Polyak bias averaging failed for Q2"
