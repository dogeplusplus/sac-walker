import numpy as np
import jax.numpy as jnp

from dataclass import dataclass
from model.mlp import SACActor, SACCritic


@dataclass(frozen=True)
class Batch:
    state: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_state: np.ndarray
    done: np.ndarray


class ReplayBuffer(object):
    def __init__(self, size, obs_dim, act_dim):
        self.state = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((size, act_dim), dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.next_state = np.zeros((size, obs_dim), dtype=np.float32)
        self.done = np.zeros(size, dtype=np.bool)

        self.ptr = 0
        self.size = size

    def store(self, obs, act, rew, next_obs, done):
        assert self.ptr < self.size
        self.state[self.ptr] = obs
        self.actions[self.ptr] = act
        self.rewards[self.ptr] = rew
        self.next_state[self.ptr] = next_obs
        self.done[self.ptr] = done
        self.ptr += 1

    def sample(self):
        self.ptr = 0

        return Batch(
            self.state,
            self.actions,
            self.rewards,
            self.next_state,
            self.donei
        )


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


class SAC(object):
    def __init__(self, env, policy, q1, q2, buffer_size, params: TrainingParameters):
        self.env = env
        self.policy = = policy
        self.q1 = q1
        self.q2 = q2
        self.buffer = ReplayBuffer(buffer_size, env.observation_size,
                                   env.action_size)
        self.params = params


    def train(self):
        state = self.env.reset()
        params = self.params

        for i in range(params.epochs * params.steps_per_epoch):
            epoch = i // params.steps_per_epoch

            action = self.ac.step(state)

            next_state, reward, done, _ = self.env.step(action)
            self.buffer.store(state, action, reward, next_state, done)
            state = next_state

            timeout = ep_len == self.max_ep_len
            terminal = done or timeout
            epoch_ended = t == steps_per_epoch - 1

            state = self.env.reset()

            if i % update_every == 0:
                samples = buffer.sample()
                self.update(samples, params.num_updates)

    def update(self, data, num_updates):
        for j in range(num_updates):
            reward = data.reward
            done = data.done
            state = data.state
            actions = data.actions

            state_action = np.concat([state, action], axis=-1)
            min_target = min(self.q1(state_action), self.q2(state_action))
            target = reward + self.params.gamma * min_target - self.params.alpha



class Trainer(object):
    def __init__(self):
        self._env = None
        self._actor_1 = None
        self._actor_2 = None
        self._critic = None
        self._training_parameters = {}

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
        # TODO: Enforce checks of mandatory parameters
        self._training_parameters = params

    def produce_actors(self, hidden_sizes, seed):
        assert self.env is not None, "Environment must be set before networks."
        obs_dim = self.env.observation_size.size
        act_dim = self.env.action_size.size

        self._actor1 = SACActor(obs_dim, act_dim, hidden_sizes, seed)
        self._actor2 = SACActor(obs_dim, act_dim, hidden_sizes, seed + 1)

    def produce_critics(self, hidden_sizes, seed):
        assert self.env is not None, "Environment must be set before networks."
        obs_dim = self.env.observation_size.size

        self._critic = SACCritic(obs_dim, hidden_sizes, seed)

    def train(self):
        assert self._env is not None, "Environment must be set before training."
        assert self._critic is not None, "Critic must be set before training."
        assert self._actor1 is not None and self._actor2 is not None, "Actors must be set before training."
        assert self._training_parameters is not None, "No training parameters provided."

        sac = SAC(self._env, self._actor1, self._critic, self._training_parameters)
        sac.train()


def main():
    pass
