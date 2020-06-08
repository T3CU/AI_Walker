from DDPG import Agent
import gym
import numpy as np
from gym import wrappers
import os

if __name__ == '__main__':
    env = gym.make('BipedalWalker-v3')
    agent = Agent(alpha=0.00005, beta=0.0005, input_dims=[24], tau=0.001, env=env,
                  n_actions=4, batch_size=64, layer1_size=400, layer2_size=300,
                  chkpt_dir='tmp/ddpg')
    np.random.seed(0)
    #agent.load_models()
    #env = wrappers.Monitor(env, "tmp/walker2D", video_callable=lambda episode_id: True, Force=True)
    score_history = []
    for i in range(5000):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, info = env.step(act)
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs = new_state
            # if i % 20 == 0:
            #    env.render()
        score_history.append(score)
        print('episode ', i, 'score%.2f' % score,
              'trailing 100 game avg %.2f' % np.mean(score_history[-100:]))

        if i % 25 == 0:
            agent.save_models()


