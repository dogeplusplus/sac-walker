import numpy as np

class ReplayBuffer(object):
    def __init__(self, size, obs_dim, act_dim):
        self.observations = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((size, act_dim), dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.next_observations = np.zeros((size, obs_dim), dtype=np.float32)
        self.done = np.zeros(size, dtype=np.bool)

        self.ptr = 0
        self.size = size


    def store(self obs, act, rew, next_obs, done):
        assert self.ptr < self.size
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = act
        self.rewards[self.ptr] = rew
        self.next_observations[self.ptr] = next_obs
        self.done[self.ptr] = done
        self.ptr += 1

    def sample(self):
        self.ptr = 0

        return (
            self.observations,
            self.actions,
            self.rewards,
            self.next_observations,
            self.done
        )

