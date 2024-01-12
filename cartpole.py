from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import time, math
from typing import Tuple
import gym

env = gym.make('CartPole-v1')
rng = np.random.default_rng()

# Q-learning setup
n_bins = (6, 12)
lower_bounds = [env.observation_space.low[2], -math.radians(50)]
upper_bounds = [env.observation_space.high[2], math.radians(50)]

# Discretize the continuous observation space
est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
est.fit([lower_bounds, upper_bounds])

def discretizer(obs_tuple) -> Tuple[int, ...]:
    # Extract the observation array from the tuple
    obs_array = obs_tuple[0]

    # Extract the relevant elements from the observation array
    angle, pole_velocity = obs_array[2], obs_array[3]

    # Proceed with your discretization
    return tuple(map(int, est.transform([[angle, pole_velocity]])[0]))



Q_table = np.zeros(n_bins + (env.action_space.n,))

def policy(state: tuple, e: int) -> int:
    """Epsilon-greedy policy"""
    if rng.random() < exploration_rate(e):
        return env.action_space.sample()
    return np.argmax(Q_table[state])

def new_q_value(reward: float, new_state: tuple, discount_factor=1.0) -> float:
    """Temporal difference for updating Q-value of state-action pair"""
    future_optimal_value = np.max(Q_table[new_state])
    return reward + discount_factor * future_optimal_value

def learning_rate(n: int, min_rate=0.01) -> float:
    """Decaying learning rate"""
    return max(min_rate, min(1.0, 1.0 - math.log10((n + 1) / 25)))

def exploration_rate(n: int, min_rate=0.1) -> float: 
    """Decaying exploration rate"""
    return max(min_rate, min(1, 1.0 - math.log10((n + 1) / 25)))

# Training process
n_episodes = 10000
for e in range(n_episodes):
    obs = env.reset()
    current_state = discretizer(obs)
    done = False

    while not done:
        action = policy(current_state, e)
        obs, reward, done, unexpected_bool, _ = env.step(action)  # Adjusted here
        new_state = discretizer(obs)
        lr = learning_rate(e)
        learned_value = new_q_value(reward, new_state)
        old_value = Q_table[current_state][action]
        Q_table[current_state][action] = (1 - lr) * old_value + lr * learned_value
        current_state = new_state

        # Render the environment
        env.render()

env.close()
