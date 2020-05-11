import gym
import random
import numpy as np

env = gym.make('BipedalWalkerHardcore-v3')
env.reset()

action_low = env.action_space.low
action_high = env.action_space.high
action_shape = env.action_space.shape

for _ in range(200):
    action = np.random.uniform(action_low, action_high, action_shape)
    state, reward, done, _ = env.step(action)
    env.render()

env.close()

