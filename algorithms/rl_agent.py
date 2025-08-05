import numpy as np
import random

class QLearningAgent:
    def __init__(self, learning_rate=0.1, epsilon=0.1, discount_factor=0.95):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        
        # Q-table: state -> action -> value
        # State: (player_sum, dealer_card, usable_ace)
        # Action: 0 (stick), 1 (hit)
        self.q_table = {}
        
    def get_state_key(self, observation):
        """Convert observation to state key for Q-table"""
        player_sum, dealer_card, usable_ace = observation
        return (player_sum, dealer_card, usable_ace)
    
    def get_action(self, observation, training=True):
        """Choose action using epsilon-greedy policy"""
        state = self.get_state_key(observation)
        
        # Initialize Q-values for new state
        if state not in self.q_table:
            self.q_table[state] = {0: 0.0, 1: 0.0}
        
        # Epsilon-greedy policy
        if training and random.random() < self.epsilon:
            return random.choice([0, 1])  # Random action
        else:
            # Choose best action based on Q-values
            return max(self.q_table[state], key=self.q_table[state].get)
    
    def update(self, observation, action, reward, next_observation, done):
        """Update Q-values using Q-learning update rule"""
        current_state = self.get_state_key(observation)
        next_state = self.get_state_key(next_observation)
        
        # Initialize Q-values if needed
        if current_state not in self.q_table:
            self.q_table[current_state] = {0: 0.0, 1: 0.0}
        if next_state not in self.q_table:
            self.q_table[next_state] = {0: 0.0, 1: 0.0}
        
        # Q-learning update rule
        current_q = self.q_table[current_state][action]
        
        if done:
            max_next_q = 0
        else:
            max_next_q = max(self.q_table[next_state].values())
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[current_state][action] = new_q
    
    def save_q_table(self, filename='q_table.npy'):
        """Save Q-table to file"""
        np.save(filename, self.q_table)
    
    def load_q_table(self, filename='q_table.npy'):
        """Load Q-table from file"""
        try:
            self.q_table = np.load(filename, allow_pickle=True).item()
        except FileNotFoundError:
            print(f"Q-table file {filename} not found. Starting with empty Q-table.") 