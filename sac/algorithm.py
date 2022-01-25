import sys
import jax
import gym
import tqdm
import optax
import jax.numpy as jnp

from copy import deepcopy
from functools import partial
from dataclasses import dataclass
from haiku import PRNGSequence

sys.path.append(".")
from sac.buffer import ReplayBuffer
from models.mlp import MLPActor, DoubleCritic


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


@jax.jit
def apply_double_q(q_state, state_action):
    q1, q2= DoubleCritic().apply(q_state, state_action)
    return q1, q2


@partial(jax.jit, static_argnums=(3, 4))
def apply_actor(pi_state, state, key, act_limit, act_dim):
    actor_fn = MLPActor(act_dim, act_limit)
    action, logp_ac = actor_fn.apply(pi_state, state, key=key)
    return action, logp_ac


@partial(jax.jit, static_argnums=(9, 10, 11, 12))
def eval_q_loss(pi_state, q_state, q_state_targ, s, a, r, s2, d, key, alpha, gamma, act_limit, act_dim):
    sa = jnp.concatenate([s, a], axis=-1)
    q1, q2 = apply_double_q(q_state, sa)

    a2, logp_ac = apply_actor(pi_state, s2, key, act_limit, act_dim)
    sa2 = jnp.concatenate([s2, a2], axis=-1)

    q1_target, q2_target = apply_double_q(q_state_targ, sa2)
    q_target = jnp.where(q1_target < q2_target, q1_target, q2_target)

    backup = r + gamma * (1 - d) * (
        q_target - alpha * logp_ac
    )

    loss_q1 = jnp.mean((q1 - backup) ** 2)
    loss_q2 = jnp.mean((q2 - backup) ** 2)
    loss_q = loss_q1 + loss_q2

    return loss_q


def update_targets(source, target, polyak):
    new_target = jax.tree_multimap(
        lambda x, y: polyak * y + (1 - polyak) * x,
        source,
        target
    )

    return new_target

@partial(jax.jit, static_argnums=(5, 6, 7))
def eval_pi_loss(
    pi_state,
    q_state,
    s,
    s2,
    key,
    alpha,
    act_limit,
    act_dim,
):
    pi, logp_pi = apply_actor(pi_state, s2, key, act_limit, act_dim)
    sa2 = jnp.concatenate([s, pi], axis=-1)
    q1, q2 = apply_double_q(q_state, sa2)

    q_pi = jnp.min(jnp.concatenate([q1, q2], axis=-1), axis=-1)
    loss_pi = jnp.mean(alpha * logp_pi - q_pi)

    return loss_pi


class SAC(object):
    def __init__(self, env, params: TrainingParameters):
        self.env = env
        self.buffer = ReplayBuffer(
            params.buffer_size,
            env.observation_space.shape[0],
            env.action_space.shape[0],
        )
        self.params = params


    def train(self):
        state = self.env.reset()
        params = self.params

        step = 0
        act_limit = 10

        act_dim = self.env.action_space.shape[0]
        obs_dim = self.env.observation_space.shape[0]

        actor_init = jnp.ones((params.batch_size, obs_dim))
        critic_init = jnp.ones((params.batch_size, obs_dim + act_dim))

        rng = PRNGSequence(42)

        pi_state = MLPActor(act_dim, act_limit).init(jax.random.PRNGKey(0), actor_init, key=next(rng))
        q_state = DoubleCritic().init(jax.random.PRNGKey(1), critic_init)
        q_state_targ = deepcopy(q_state)

        pi_opt = optax.adam(params.learning_rate)
        q_opt = optax.adam(params.learning_rate)

        q_opt_state = q_opt.init(q_state)
        pi_opt_state = pi_opt.init(pi_state)

        pi_loss_grad_fn = jax.value_and_grad(eval_pi_loss, allow_int=True, argnums=(0, 1, 2, 3, 4))
        q_loss_grad_fn = jax.value_and_grad(eval_q_loss, allow_int=True, argnums=(0,1,2,3,4,5,6,7,8))

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
                    action = apply_actor(pi_state, state, next(rng), act_limit, act_dim)

                next_state, reward, done, _ = self.env.step(action)
                self.env.render()
                ep_len += 1
                ep_ret += reward

                self.buffer.store(state, action, reward, next_state, done)
                state = next_state

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

                        q_loss, grad_q = q_loss_grad_fn(
                            pi_state,
                            q_state,
                            q_state_targ,
                            samples.states,
                            samples.actions,
                            samples.rewards,
                            samples.next_states,
                            samples.done,
                            next(rng),
                            params.alpha,
                            params.gamma,
                            act_limit,
                            act_dim,
                        )
                        updates, q_opt_state = q_opt.update(grad_q[1], q_opt_state, q_state)
                        q_state = optax.apply_updates(q_state, updates)

                        pi_loss, grad_pi = pi_loss_grad_fn(
                            pi_state,
                            q_state,
                            samples.states,
                            samples.next_states,
                            next(rng),
                            params.alpha,
                            act_limit,
                            act_dim,
                        )
                        updates, pi_opt_state = pi_opt.update(grad_pi[0], pi_opt_state, pi_state)
                        pi_state = optax.apply_updates(pi_state, updates)

                        q_state_targ = update_targets(q_state, q_state_targ, params.polyak)

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

def main():
    env = gym.make("Humanoid-v2")

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

    sac = SAC(env, training_params)
    sac.train()


if __name__ == "__main__":
    main()
