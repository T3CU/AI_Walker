import gym
import random
import numpy as np

env_name = 'MountainCar-v0'
i = 0
LR = 0.1
DISC = 0.95
Episodes = 10000
Show_Every = 1000

DISCRETE_GRID_SIZE = [10, 10]
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = Episodes//2


def calc_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/buckets
    return tuple(discrete_state.astype(np.int))


def run_game(q_table, render, should_update):
    done = False
    discrete_state = calc_discrete_state(env.reset())
    success = False

    while not done:

        if np.random.random() > epsilon:
            #Exploit
            action = np.argmax(q_table[discrete_state])
        else:
            #Explore
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action)
        new_state_disc = calc_discrete_state(new_state)
        if new_state[0] >= env.goal_position:
            success = True

        if should_update:
            max_future_q = np.max(q_table[new_state_disc])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LR) * current_q + LR * (reward + DISC * max_future_q)
            q_table[discrete_state + (action,)] = new_q

        discrete_state = new_state_disc

        if render:
            env.render()

    return success


env = gym.make(env_name)

epsilon = 1
epsilon_change = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)
buckets = (env.observation_space.high - env.observation_space.low)/DISCRETE_GRID_SIZE
q_table = np.random.uniform(low=-3, high=0, size=(DISCRETE_GRID_SIZE + [env.action_space.n]))
success = False

episode = 0
success_count = 0

while episode<Episodes:
    episode+=1
    done = False

    if episode % Show_Every == 0:
        print(f"Current episode: {episode}, success: {success_count} ({float(success_count)/Show_Every})")
        success = run_game(q_table, True, False)
        success_count = 0
    else:
        success = run_game(q_table, False, True)

    if success:
        success_count += 1

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_change

print(success)
env.close()


