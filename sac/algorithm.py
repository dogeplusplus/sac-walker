import sys
import jax
import gym
import tqdm
import jax.numpy as jnp

from jax import jit, grad, random
from copy import deepcopy
from functools import partial
from dataclasses import dataclass
from jax.experimental.optimizers import adam

sys.path.append(".")
from models.mlp import ActorCritic, relu
from sac.buffer import ReplayBuffer


@dataclass(frozen=True)
class TrainingParameters:
    epochs: int
    steps_per_epoch: int
    batch_size: int
    start_steps: int
    update_after: int
    update_every: int
    learning_rate: float
    alpha: float
    gamma: float
    polyak: float
    buffer_size: int = int(1e6)
    max_ep_len: int = 4000


class SAC(object):
    def __init__(self, env, ac, params: TrainingParameters):
        self.env = env
        self.ac = ac
        self.ac_targ = deepcopy(ac)
        self.buffer = ReplayBuffer(
            params.buffer_size,
            env.observation_space.shape[0],
            env.action_space.shape[0],
        )
        self.params = params
        self.pi_opt_init, self.pi_opt_update, self.pi_get_params = adam(params.learning_rate)
        self.pi_opt_state = self.pi_opt_init(self.ac.pi.params)
        self.q_opt_init, self.q_opt_update, self.q_get_params = adam(params.learning_rate)
        self.q_params = {"q1": self.ac.q1.params, "q2": self.ac.q2.params}
        self.q_opt_state = self.q_opt_init(self.q_params)

    @partial(jit, static_argnums=(0,))
    def q_loss(self, q_weights, s, a, r, s2, d):

        sa = jnp.concatenate([s, a], axis=-1)
        q1 = self.ac.q1.predict(q_weights["q1"], sa)
        q2 = self.ac.q2.predict(q_weights["q2"], sa)

        a2, logp_ac = self.ac.pi(s2)
        sa2 = jnp.concatenate([s2, a2], axis=-1)

        q1_target = self.ac_targ.q1(sa2)
        q2_target = self.ac_targ.q2(sa2)
        q_target = jnp.where(q1_target < q2_target, q1_target, q2_target)

        backup = r + self.params.gamma * (1 - d) * (
            q_target - self.params.alpha * logp_ac
        )

        loss_q1 = jnp.mean((q1 - backup) ** 2)
        loss_q2 = jnp.mean((q2 - backup) ** 2)
        loss_q = loss_q1 + loss_q2
        return loss_q

    @partial(jit, static_argnums=(0,))
    def pi_loss(self, pi_weights, s, s2):
        pi, logp_pi = self.ac.pi.predict(pi_weights, s2)
        sa2 = jnp.concatenate([s, pi], axis=-1)
        q1 = self.ac.q1(sa2)
        q2 = self.ac.q2(sa2)

        q_pi = jnp.min(jnp.concatenate([q1, q2], axis=-1), axis=-1)
        loss_pi = jnp.mean(self.params.alpha * logp_pi - q_pi)
        return loss_pi

    def update_q(self, step, samples):
        loss_q, grad_q = jax.value_and_grad(self.q_loss)(
            self.q_get_params(self.q_opt_state),
            samples.states,
            samples.actions,
            samples.rewards,
            samples.next_states,
            samples.done,
        )

        self.q_opt_state = self.q_opt_update(step, grad_q, self.q_opt_state)
        self.q_params = self.q_get_params(self.q_opt_state)

        self.ac.q1.params = self.q_params["q1"]
        self.ac.q2.params = self.q_params["q2"]

        return loss_q

    def update_pi(self, step, samples):
        loss_pi, grad_pi = jax.value_and_grad(self.pi_loss)(
            self.pi_get_params(self.pi_opt_state),
            samples.states,
            samples.next_states,
        )

        self.pi_opt_state = self.pi_opt_update(step, grad_pi, self.pi_opt_state)
        pi_params = self.pi_get_params(self.pi_opt_state)
        self.ac.pi.params = pi_params

        return loss_pi

    def update_targets(self):
        targ_q1 = self.ac_targ.q1.params
        targ_q2 = self.ac_targ.q2.params

        source_q1 = self.ac.q1.params
        source_q2 = self.ac.q2.params

        polyak = self.params.polyak

        for i, _ in enumerate(self.ac_targ.q1.params):
            # weights and biases
            self.ac_targ.q1.params[i] = (
                polyak * targ_q1[i][0] + (1 - polyak) * source_q1[i][0],
                polyak * targ_q1[i][1] + (1 - polyak) * source_q1[i][1],
            )
            self.ac_targ.q2.params[i] = (
                polyak * targ_q2[i][0] + (1 - polyak) * source_q2[i][0],
                polyak * targ_q2[i][1] + (1 - polyak) * source_q2[i][1],
            )


    def train(self):
        state = self.env.reset()
        params = self.params

        step = 0
        ep_ret = 0
        ep_len = 0

        for e in range(params.epochs):
            pbar = tqdm.tqdm(
                range(params.steps_per_epoch),
                desc=f"Epoch {e+1:>4}",
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"
            )
            cumulative_metrics = {
                "q_loss": 0,
                "pi_loss": 0,
            }
            for i in pbar:
                if step < self.params.start_steps:
                    action = self.env.action_space.sample()
                else:
                    action = self.ac.act(state)

                next_state, reward, done, _ = self.env.step(action)
                # self.env.render()
                ep_len += 1
                ep_ret += reward

                self.buffer.store(state, action, reward, next_state, done)
                state = next_state

                # TODO: Implement environment reset and epoch termination
                timeout = ep_len == params.max_ep_len
                terminal = done or timeout
                epoch_ended = i % params.steps_per_epoch == params.steps_per_epoch - 1

                if terminal or epoch_ended:
                    state = self.env.reset()
                    ep_ret = 0
                    ep_len = 0

                if step > params.update_after and step % params.update_every == 0:
                    for _ in range(params.update_every):
                        samples = self.buffer.sample(params.batch_size)

                        q_loss = self.update_q(step, samples)
                        pi_loss = self.update_pi(step, samples)
                        self.update_targets()

                        cumulative_metrics["pi_loss"] += pi_loss
                        cumulative_metrics["q_loss"] += q_loss

                        metrics = {
                            "Episode Length": ep_len,
                            "Cumulative Reward": ep_ret,
                            "Q Loss": f"{cumulative_metrics['q_loss'] / (i + 1):.4g}",
                            "PI Loss": f"{cumulative_metrics['pi_loss'] / (i + 1):.4g}",
                        }
                        pbar.set_postfix(metrics)

                step += 1
            state = self.env.reset()
            ep_ret = 0
            ep_len = 0


class Trainer(object):
    def __init__(self):
        self._env = None
        self._ac = None
        self._training_parameters = None

    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, env):
        self._env = env

    @property
    def training_parameters(self):
        return self._training_parameters

    @training_parameters.setter
    def training_parameters(self, params: TrainingParameters):
        self._training_parameters = params

    def create_actor_critic(self, hidden_sizes, activation_fn, act_limit, seed):
        assert self.env is not None, "Environment must be set before networks."
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        self._ac = ActorCritic(
            obs_dim, act_dim, hidden_sizes, activation_fn, act_limit, seed
        )

    def train(self):
        assert self._env is not None, "Environment must be set before training."
        assert self._ac is not None, "Actor must be set before training."
        assert self._training_parameters is not None, "No training parameters provided."

        sac = SAC(self._env, self._ac, self._training_parameters)
        sac.train()


def main():
    trainer = Trainer()
    env = gym.make("BipedalWalker-v3")
    trainer.env = env

    training_params = TrainingParameters(
        epochs=10,
        steps_per_epoch=4000,
        batch_size=100,
        start_steps=10000,
        update_after=1000,
        update_every=50,
        learning_rate=1e-3,
        alpha=0.2,
        gamma=0.99,
        polyak=0.99
    )
    trainer.training_parameters = training_params

    hidden_sizes = [256, 256]
    activation_fn = relu
    act_limit = 10
    seed = random.PRNGKey(1337)
    trainer.create_actor_critic(hidden_sizes, activation_fn, act_limit, seed)
    trainer.train()

if __name__ == "__main__":
    main()
