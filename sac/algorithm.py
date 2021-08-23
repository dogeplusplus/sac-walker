import numpy as np

from jax.random import split
from dataclasses import dataclass

from sac.buffer import ReplayBuffer
from models.mlp import MLPActor, QFunction


@dataclass(frozen=True)
class TrainingParameters:
    epochs: int
    steps_per_epoch: int
    batch_size: int
    start_steps: int
    update_every: int
    num_updates: int
    alpha: float
    gamma: float
    polyak: float
    buffer_size: int = int(1e6)


class SAC(object):
    def __init__(self, env, policy, q1, q2, params: TrainingParameters):
        self.env = env
        self.policy = policy
        self.q1 = q1
        self.q2 = q2
        self.buffer = ReplayBuffer(params.buffer_size, env.observation_size,
                                   env.action_size)
        self.params = params


    def train(self):
        state = self.env.reset()
        params = self.params

        for i in range(params.epochs * params.steps_per_epoch):
            epoch = i // params.steps_per_epoch

            action = self.policy(state)

            next_state, reward, done, _ = self.env.step(action)
            self.buffer.store(state, action, reward, next_state, done)
            state = next_state

            timeout = ep_len == self.max_ep_len
            terminal = done or timeout
            epoch_ended = t == steps_per_epoch - 1

            state = self.env.reset()

            if i % params.update_every == 0:
                samples = self.buffer.sample(params.batch_size)
                self.update(samples, params.num_updates)

    def update(self, data, num_updates):
        for j in range(num_updates):
            reward = data.reward
            done = data.done
            states = data.state
            actions = data.actions

            state_action = np.concat([states, actions], axis=-1)
            min_target = min(self.q1(state_action), self.q2(state_action))
            target = reward + self.params.gamma * min_target - self.params.alpha



class Trainer(object):
    def __init__(self):
        self._env = None
        self._actor_1 = None
        self._actor_2 = None
        self._critic = None
        self._training_parameters = None

    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, env):
        # TODO: make sure environment is valid
        self._env = env

    @property
    def training_parameters(self):
        return self._training_parameters

    @training_parameters.setter
    def training_parameters(self, params: TrainingParameters):
        self._training_parameters = params

    def produce_actor(self, hidden_sizes, activation_fn, act_limit, seed):
        assert self.env is not None, "Environment must be set before networks."
        obs_dim = self.env.observation_size.size
        act_dim = self.env.action_size.size

        self._actor = MLPActor(obs_dim, act_dim, hidden_sizes, activation_fn, act_limit, seed)

    def produce_critics(self, hidden_sizes, activation_fn, seed):
        assert self.env is not None, "Environment must be set before networks."
        obs_dim = self.env.observation_size.size
        act_dim = self.env.action_size.size

        seed_1, seed_2 = split(seed)

        self._critic_1 = QFunction(obs_dim, act_dim, hidden_sizes, activation_fn, seed_1)
        self._critic_2 = QFunction(obs_dim, act_dim, hidden_sizes, activation_fn, seed_2)

    def train(self):
        assert self._env is not None, "Environment must be set before training."
        assert self._critic_1 is not None and self._critic_2 is not None, "Critics must be set before training."
        assert self._actor, "Actor must be set before training."
        assert self._training_parameters is not None, "No training parameters provided."

        sac = SAC(self._env, self._actor, self._critic_1, self._critic_2, self._training_parameters)
        sac.train()


def main():
    pass
