import gymnasium as gym
from strategy import basic_strategy
from tqdm import tqdm

# Run multiple episodes
n_episodes = 1000
wins, losses, draws = 0, 0, 0
env = gym.make('Blackjack-v1', render_mode='human')

for _ in tqdm(range(n_episodes)):
    observation, info = env.reset()
    done = False

    while not done:
        action = basic_strategy(observation[0])
        observation, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            done = True
            if reward == 1:
                wins += 1
            elif reward == -1:
                losses += 1
            else:
                draws += 1

win_rate = wins / n_episodes
print(f"Win Rate: {win_rate * 100:.2f}%")