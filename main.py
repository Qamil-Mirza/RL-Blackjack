import gymnasium as gym
from strategy import basic_strategy
import numpy as np

# Create the Blackjack environment
env = gym.make('Blackjack-v1', render_mode='human')

# Play a single episode
observation, info = env.reset()
done = False

while not done:
    # Take a random action (hit or stick)
    action = basic_strategy(observation[0])
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        done = True
