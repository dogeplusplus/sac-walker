import gym
import pytest
import jax.numpy as jnp

from jax import random
from copy import deepcopy

from models.mlp import ActorCritic, relu
from sac.algorithm import SAC, TrainingParameters


@pytest.fixture
def sac():
    env = gym.make("BipedalWalker-v3")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    hidden_sizes = [256, 256]
    activation_fn = relu
    act_limit = 1e-2
    ac = ActorCritic(
        obs_dim, act_dim, hidden_sizes, activation_fn, act_limit, random.PRNGKey(1)
    )

    params = TrainingParameters(
        epochs=1,
        steps_per_epoch=1,
        batch_size=2,
        start_steps=0,
        update_every=1,
        num_updates=1,
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
    q1_old = deepcopy(sac.ac.q1.params)
    q2_old = deepcopy(sac.ac.q2.params)

    sac.update_q(0, samples)

    q1_new = sac.ac.q1.params
    q2_new = sac.ac.q2.params

    q1_updated = jnp.any(q1_old[0][0] != q1_new[0][0])
    q2_updated = jnp.any(q2_old[0][0] != q2_new[0][0])
    assert q1_updated or q2_updated, "Q networks did not update."


def test_q_loss(sac, samples):
    q_weights = {
        "q1": sac.ac.q1.params,
        "q2": sac.ac.q2.params,
    }
    loss_q = sac.q_loss(
        q_weights,
        samples.states,
        samples.actions,
        samples.rewards,
        samples.next_states,
        samples.done,
    )

    assert loss_q >= 0, "Loss should be at least 0."


def test_pi_loss(sac, samples):
    pi_weights = sac.ac.pi.params
    loss_pi = sac.pi_loss(pi_weights, samples.states, samples.next_states)
    assert loss_pi >= 0, "Loss should be at least 0."


def test_pi_update(sac, samples):
    pi_old = deepcopy(sac.ac.pi.params)

    sac.update_pi(0, samples)
    pi_new = sac.ac.pi.params
    for component in ["net", "mu", "log_std"]:
        pi_updated = jnp.any(pi_old[component][0][0] != pi_new[component][0][0])
        assert pi_updated, "Policy network did not update."

def test_target_update(sac, samples):
    q1_targ_params = deepcopy(sac.ac_targ.q1.params)
    q2_targ_params = deepcopy(sac.ac_targ.q2.params)

    sac.update_q(0, samples)

    q1_params = deepcopy(sac.ac.q1.params)
    q2_params = deepcopy(sac.ac.q2.params)

    sac.update_targets()

    expected_q1_weights = 0.5 * (q1_targ_params[0][0] + q1_params[0][0])
    expected_q2_weights = 0.5 * (q2_targ_params[0][0] + q2_params[0][0])

    actual_q1_weights = sac.ac_targ.q1.params[0][0]
    actual_q2_weights = sac.ac_targ.q2.params[0][0]

    assert jnp.allclose(expected_q1_weights, actual_q1_weights), "Polyak weight averaging failed for Q1"
    assert jnp.allclose(expected_q2_weights, actual_q2_weights), "Polyak weight averaging failed for Q2"

    expected_q1_biases = 0.5 * (q1_targ_params[0][1] + q1_params[0][1])
    expected_q2_biases = 0.5 * (q2_targ_params[0][1] + q2_params[0][1])

    actual_q1_biases = sac.ac_targ.q1.params[0][1]
    actual_q2_biases = sac.ac_targ.q2.params[0][1]

    assert jnp.allclose(expected_q1_biases, actual_q1_biases), "Polyak bias averaging failed for Q1"
    assert jnp.allclose(expected_q2_biases, actual_q2_biases), "Polyak bias averaging failed for Q2"
