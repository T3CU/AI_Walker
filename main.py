import gym
import Box2D
from tensorflow import keras

env = gym.make('BipedalWalkerHardcore-v3')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())
env.close()