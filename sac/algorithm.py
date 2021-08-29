import sys
import jax
import gym
import flax
import tqdm
import optax
import jax.numpy as jnp

from copy import deepcopy
from jax import jit, random
from functools import partial
from dataclasses import dataclass
from flax import traverse_util
from flax.core import freeze, unfreeze

sys.path.append(".")
from sac.buffer import ReplayBuffer
from models.mlp import ActorCritic, relu


def flat_params(params):
    params = unfreeze(params)
    flat = {
        "/".join(k): v for k,
        v in traverse_util.flatten_dict(params).items()
    }
    return flat 


def unflat_params(flat_params):
    unflat = traverse_util.unflatten_dict({
        tuple(k.split('/')): v 
        for k, v in flat_params.items()
    })
    unflat = freeze(unflat)

    return unflat

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


    @partial(jit, static_argnums=(0,))
    def q_loss(self, q_params, s, a, r, s2, d):

        sa = jnp.concatenate([s, a], axis=-1)
        q1 = self.ac.q1.apply(q_params["q1"], sa)
        q2 = self.ac.q2.apply(q_params["q2"], sa)

        a2, logp_ac = self.ac.pi.apply(self.ac.pi_state.params, s2)
        sa2 = jnp.concatenate([s2, a2], axis=-1)

        q1_target = self.ac_targ.q1.apply(self.ac_targ.q_state.params["q1"], sa2)
        q2_target = self.ac_targ.q2.apply(self.ac_targ.q_state.params["q2"], sa2)
        q_target = jnp.where(q1_target < q2_target, q1_target, q2_target)

        backup = r + self.params.gamma * (1 - d) * (
            q_target - self.params.alpha * logp_ac
        )

        loss_q1 = jnp.mean((q1 - backup) ** 2)
        loss_q2 = jnp.mean((q2 - backup) ** 2)
        loss_q = loss_q1 + loss_q2
        return loss_q

    @partial(jit, static_argnums=(0,))
    def pi_loss(self, pi_params, s, s2):
        pi, logp_pi = self.ac.pi.apply(pi_params, s2)
        sa2 = jnp.concatenate([s, pi], axis=-1)
        q1 = self.ac.q1.apply(self.ac.q_state.params["q1"], sa2)
        q2 = self.ac.q2.apply(self.ac.q_state.params["q2"], sa2)

        q_pi = jnp.min(jnp.concatenate([q1, q2], axis=-1), axis=-1)
        loss_pi = jnp.mean(self.params.alpha * logp_pi - q_pi)

        return loss_pi

    def update_q(self, q_state, samples):
        loss_q, grad_q = jax.value_and_grad(self.q_loss)(
            q_state.params,
            samples.states,
            samples.actions,
            samples.rewards,
            samples.next_states,
            samples.done,
        )
        q_state = q_state.apply_gradients(grads=grad_q)

        return (q_state, loss_q)

    def update_pi(self, pi_state, samples):
        loss_pi, grad_pi = jax.value_and_grad(self.pi_loss)(
            pi_state.params,
            samples.states,
            samples.next_states,
        )
        pi_state = pi_state.apply_gradients(grads=grad_pi)

        return (pi_state, loss_pi)

    def update_targets(self):

        targ_params = self.ac_targ.q_state.params
        source_params = self.ac.q_state.params

        targ_flat = flat_params(targ_params)
        source_flat = flat_params(source_params)
        polyak = self.params.polyak

        for k in targ_flat.keys():
            targ_flat[k] = polyak * targ_flat[k] + (1 - polyak) * source_flat[k]

        new_targ_params = unflat_params(targ_flat)
        self.ac_targ.q_state.replace(params=new_targ_params)

    def train(self):
        state = self.env.reset()
        params = self.params

        step = 0

        for e in range(params.epochs):
            ep_ret = 0
            ep_len = 0

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
                if step < params.start_steps:
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

                        self.ac.q_state, q_loss = self.update_q(self.ac.q_state, samples)
                        self.ac.pi_state, pi_loss = self.update_pi(self.ac.pi_state, samples)
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
        assert self._training_parameters is not None, "No training parameters provided."

        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        learning_rate = self._training_parameters.learning_rate
        self._ac = ActorCritic(obs_dim,
            act_dim,
            hidden_sizes,
            activation_fn,
            act_limit,
            learning_rate,
            seed
        )

    def train(self):
        assert self._env is not None, "Environment must be set before training."
        assert self._ac is not None, "Actor must be set before training."
        assert self._training_parameters is not None, "No training parameters provided."

        sac = SAC(self._env, self._ac, self._training_parameters)
        sac.train()


def main():
    trainer = Trainer()
    env = gym.make("MountainCarContinuous-v0")
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
