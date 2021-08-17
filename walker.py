import gym
import numpy as np

env = gym.make("BipedalWalker-v3")
env.render()

def random_games():
    action_size = env.action_space.shape[0]
    for episode in range(10):
        env.reset()

        while True:
            env.render()
            action = np.random.uniform(-1., 1., size=action_size)

            next_state, reward, done, _ = env.step(action)

            if done:
                break

def main():
    random_games()

if __name__ == "__main__":
    main()
