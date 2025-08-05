import gymnasium as gym
import numpy as np
from algorithms.basic_strategy import basic_strategy
from algorithms.rl_agent import QLearningAgent
from algorithms.enhanced_rl_agent import EnhancedQLearningAgent
import matplotlib.pyplot as plt

class GameRunner:
    def __init__(self, rl_agent_type='basic'):
        """
        Initialize GameRunner with specified RL agent type
        
        Args:
            rl_agent_type (str): Type of RL agent to use
                - 'basic': QLearningAgent (default)
                - 'enhanced': EnhancedQLearningAgent
        """
        self.env = gym.make('Blackjack-v1')
        self.rl_agent_type = rl_agent_type
        
        # Initialize the appropriate RL agent
        if rl_agent_type == 'enhanced':
            self.rl_agent = EnhancedQLearningAgent(learning_rate=0.1, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995)
        else:  # default to basic
            self.rl_agent = QLearningAgent(learning_rate=0.1, epsilon=0.1)
        
        self.results = {
            'rl_wins': [],
            'basic_wins': [],
            'rl_win_rates': [],
            'basic_win_rates': []
        }
        
    def play_episode(self, agent_type='rl', training=True):
        """Play a single episode with specified agent"""
        observation, info = self.env.reset()
        done = False
        total_reward = 0
        
        while not done:
            if agent_type == 'rl':
                action = self.rl_agent.get_action(observation, training=training)
            else:  # basic strategy
                action = basic_strategy(observation[0])
            
            next_observation, reward, terminated, truncated, info = self.env.step(action)
            
            if agent_type == 'rl' and training:
                self.rl_agent.update(observation, action, reward, next_observation, terminated or truncated)
            
            observation = next_observation
            total_reward = reward
            done = terminated or truncated
        
        return total_reward
    
    def run_games(self, num_games, agent_type='rl', training=True):
        """Run multiple games and return win rate"""
        wins = 0
        for _ in range(num_games):
            reward = self.play_episode(agent_type, training)
            if reward > 0:  # Win
                wins += 1
        
        return wins / num_games
    
    def collect_data(self, game_counts=[5, 10, 20, 50, 100]):
        """Collect win rate data for different numbers of games"""
        print("Collecting data for RL agent vs Basic Strategy...")
        
        for num_games in game_counts:
            print(f"\nRunning {num_games} games...")
            
            # Train RL agent first
            print("Training RL agent...")
            self.run_games(num_games, 'rl', training=True)
            
            # Test both agents
            print("Testing RL agent...")
            rl_win_rate = self.run_games(num_games, 'rl', training=False)
            
            print("Testing Basic Strategy...")
            basic_win_rate = self.run_games(num_games, 'basic', training=False)
            
            # Store results
            self.results['rl_win_rates'].append(rl_win_rate)
            self.results['basic_win_rates'].append(basic_win_rate)
            
            print(f"RL Agent Win Rate: {rl_win_rate:.3f}")
            print(f"Basic Strategy Win Rate: {basic_win_rate:.3f}")
        
        return self.results
    
    def plot_results(self):
        """Plot the comparison results"""
        # Use the same game counts that were used for data collection
        game_counts = [5, 10, 20, 50, 100, 200, 500, 1000]
        
        plt.figure(figsize=(12, 6))
        plt.plot(game_counts, self.results['rl_win_rates'], 'b-o', label='RL Agent', linewidth=2, markersize=8)
        plt.plot(game_counts, self.results['basic_win_rates'], 'r-s', label='Basic Strategy', linewidth=2, markersize=8)
        
        plt.xlabel('Number of Games')
        plt.ylabel('Win Rate')
        plt.title('RL Agent vs Basic Strategy Win Rate Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(game_counts)
        
        # Add win rate values as text on points
        for i, (rl_rate, basic_rate) in enumerate(zip(self.results['rl_win_rates'], self.results['basic_win_rates'])):
            plt.annotate(f'{rl_rate:.3f}', (game_counts[i], rl_rate), 
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
            plt.annotate(f'{basic_rate:.3f}', (game_counts[i], basic_rate), 
                        textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('./results/win_rate_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_summary(self):
        """Print a summary of the results"""
        game_counts = [5, 10, 20, 50, 100, 200, 500, 1000]
        
        print("\n" + "="*60)
        print("WIN RATE COMPARISON SUMMARY")
        print("="*60)
        print(f"{'Games':<8} {'RL Agent':<12} {'Basic Strategy':<15} {'Difference':<12}")
        print("-"*60)
        
        for i, num_games in enumerate(game_counts):
            rl_rate = self.results['rl_win_rates'][i]
            basic_rate = self.results['basic_win_rates'][i]
            diff = rl_rate - basic_rate
            
            print(f"{num_games:<8} {rl_rate:<12.3f} {basic_rate:<15.3f} {diff:<12.3f}")
        
        print("-"*60)
        
        # Overall averages
        avg_rl = np.mean(self.results['rl_win_rates'])
        avg_basic = np.mean(self.results['basic_win_rates'])
        avg_diff = avg_rl - avg_basic
        
        print(f"{'AVERAGE':<8} {avg_rl:<12.3f} {avg_basic:<15.3f} {avg_diff:<12.3f}")
        print("="*60) 