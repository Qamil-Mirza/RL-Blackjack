from game_runner import GameRunner

def run_basic_agent():
    """Run the basic Q-learning agent"""
    print("=" * 60)
    print("RUNNING BASIC Q-LEARNING AGENT")
    print("=" * 60)
    
    # Create game runner with basic agent
    runner = GameRunner(rl_agent_type='basic')
    
    # Collect data for different game counts
    game_counts = [5, 10, 20, 50, 100, 200, 500, 1000]
    results = runner.collect_data(game_counts)
    
    # Print summary
    runner.print_summary()
    
    # Plot results
    print("\nGenerating visualization...")
    runner.plot_results()
    
    print("\nResults saved to 'win_rate_comparison.png'")
    print("Q-table saved to 'q_table.npy'")

def run_enhanced_agent():
    """Run the enhanced Q-learning agent"""
    print("=" * 60)
    print("RUNNING ENHANCED Q-LEARNING AGENT")
    print("=" * 60)
    
    # Create game runner with enhanced agent
    runner = GameRunner(rl_agent_type='enhanced')
    
    # Collect data for different game counts
    game_counts = [5, 10, 20, 50, 100, 200, 500, 1000]
    results = runner.collect_data(game_counts)
    
    # Print summary
    runner.print_summary()
    
    # Plot results
    print("\nGenerating visualization...")
    runner.plot_results()
    
    print("\nResults saved to 'win_rate_comparison.png'")
    print("Q-table saved to 'enhanced_q_table.npy'")

def main():
    """Main function to run the RL vs Basic Strategy comparison"""
    print("RL Blackjack Agent vs Basic Strategy Comparison")
    print("Choose which agent to run:")
    print("1. Basic Q-Learning Agent")
    print("2. Enhanced Q-Learning Agent")
    print("3. Run both and compare")
    
    choice = input("\nEnter your choice (1, 2, or 3): ").strip()
    
    if choice == '1':
        run_basic_agent()
    elif choice == '2':
        run_enhanced_agent()
    elif choice == '3':
        run_basic_agent()
        print("\n" + "="*80)
        run_enhanced_agent()
    else:
        print("Invalid choice. Running basic agent by default.")
        run_basic_agent()

if __name__ == "__main__":
    main()
