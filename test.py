import gymnasium as gym
from algorithms.rl_agent import QLearningAgent
from algorithms.basic_strategy import basic_strategy

def test_rl_agent():
    """Test the RL agent with a few games"""
    print("Testing RL Agent...")
    
    env = gym.make('Blackjack-v1')
    agent = QLearningAgent(learning_rate=0.1, epsilon=0.2)
    
    # Train for a few episodes
    print("Training RL agent for 10 episodes...")
    for episode in range(10):
        observation, info = env.reset()
        done = False
        
        while not done:
            action = agent.get_action(observation, training=True)
            next_observation, reward, terminated, truncated, info = env.step(action)
            
            agent.update(observation, action, reward, next_observation, terminated or truncated)
            
            observation = next_observation
            done = terminated or truncated
    
    # Test the trained agent
    print("Testing trained agent for 5 episodes...")
    wins = 0
    for episode in range(5):
        observation, info = env.reset()
        done = False
        
        while not done:
            action = agent.get_action(observation, training=False)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        if reward > 0:
            wins += 1
    
    win_rate = wins / 5
    print(f"RL Agent win rate: {win_rate:.2f}")
    
    env.close()

def test_basic_strategy():
    """Test the basic strategy"""
    print("\nTesting Basic Strategy...")
    
    env = gym.make('Blackjack-v1')
    wins = 0
    
    for episode in range(5):
        observation, info = env.reset()
        done = False
        
        while not done:
            action = basic_strategy(observation[0])
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        if reward > 0:
            wins += 1
    
    win_rate = wins / 5
    print(f"Basic Strategy win rate: {win_rate:.2f}")
    
    env.close()

if __name__ == "__main__":
    test_rl_agent()
    test_basic_strategy()
    print("\nTest completed!")