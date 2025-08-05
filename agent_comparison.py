#!/usr/bin/env python3
"""
Agent Comparison Script
Demonstrates how to switch between different RL agents and compare their performance.
"""

from game_runner import GameRunner
import matplotlib.pyplot as plt
import numpy as np

def compare_agents():
    """Compare different RL agents against basic strategy"""
    print("Comparing Different RL Agents")
    print("=" * 50)
    
    # Define agents to test
    agents = [
        ('Basic Q-Learning', 'basic'),
        ('Enhanced Q-Learning', 'enhanced')
    ]
    
    # Store results for each agent
    all_results = {}
    game_counts = [5, 10, 20, 50, 100, 200, 500, 1000]
    
    for agent_name, agent_type in agents:
        print(f"\nTesting {agent_name}...")
        
        # Create game runner with specific agent
        runner = GameRunner(rl_agent_type=agent_type)
        
        # Collect data
        results = runner.collect_data(game_counts)
        
        # Store results
        all_results[agent_name] = {
            'rl_win_rates': results['rl_win_rates'].copy(),
            'basic_win_rates': results['basic_win_rates'].copy()
        }
        
        print(f"{agent_name} testing completed!")
    
    # Plot comparison
    plot_agent_comparison(all_results, game_counts)
    
    return all_results

def plot_agent_comparison(all_results, game_counts):
    """Plot comparison of different agents"""
    plt.figure(figsize=(15, 8))
    
    colors = ['blue', 'green', 'red', 'purple']
    markers = ['o', 's', '^', 'D']
    
    # Plot Basic Strategy only once (it should be the same for all agents)
    basic_strategy_rates = list(all_results.values())[0]['basic_win_rates']
    plt.plot(game_counts, basic_strategy_rates, 
            color='red', marker='s', linestyle='--', label='Basic Strategy', 
            linewidth=2, markersize=8, alpha=0.8)
    
    # Plot each RL agent
    for i, (agent_name, results) in enumerate(all_results.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        plt.plot(game_counts, results['rl_win_rates'], 
                color=color, marker=marker, linestyle='-', label=f'{agent_name}', 
                linewidth=2, markersize=8)
    
    plt.xlabel('Number of Games')
    plt.ylabel('Win Rate')
    plt.title('Comparison of Different RL Agents vs Basic Strategy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(game_counts)
    plt.tight_layout()
    
    plt.savefig('./results/agent_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nComparison plot saved to 'results/agent_comparison.png'")

def print_agent_summary(all_results, game_counts):
    """Print summary of all agents' performance"""
    print("\n" + "="*80)
    print("AGENT PERFORMANCE SUMMARY")
    print("="*80)
    
    for agent_name, results in all_results.items():
        print(f"\n{agent_name}:")
        print("-" * 40)
        
        avg_rl = np.mean(results['rl_win_rates'])
        avg_basic = np.mean(results['basic_win_rates'])
        improvement = avg_rl - avg_basic
        
        print(f"Average RL Win Rate: {avg_rl:.3f}")
        print(f"Average Basic Strategy Win Rate: {avg_basic:.3f}")
        print(f"Improvement over Basic Strategy: {improvement:.3f}")
        
        # Find best performance
        best_rl_rate = max(results['rl_win_rates'])
        best_rl_games = game_counts[results['rl_win_rates'].index(best_rl_rate)]
        print(f"Best RL Performance: {best_rl_rate:.3f} ({best_rl_games} games)")

if __name__ == "__main__":
    # Run comparison
    results = compare_agents()
    
    # Print summary
    game_counts = [5, 10, 20, 50, 100, 200, 500, 1000]
    print_agent_summary(results, game_counts) 