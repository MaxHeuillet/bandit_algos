import matplotlib.pyplot as plt
import os

def plot_regret_with_confidence(agents, regret, confidence_intervals, config, env_name):
    """
    Plot regret with confidence intervals and save to plots directory.
    
    Args:
        agents: List of agents
        regret: Dictionary of regret data
        confidence_intervals: Dictionary of confidence intervals
        config: Configuration dictionary
        env_name: Name of the environment (for file naming)
    """
    plt.figure(figsize=(10, 6))
    
    for agent in agents:
        plt.plot(regret[agent.name], label=agent.name)
        for ci_name, (lower, upper) in confidence_intervals[agent.name].items():
            plt.fill_between(range(len(regret[agent.name])), lower, upper, alpha=0.2,
                           label=f'{ci_name} CI')
    
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Regret')
    plt.title(f'Regret with Confidence Intervals - {env_name} Environment')
    plt.legend()
    
    # Create plots directory if it doesn't exist
    plots_dir = config['paths']['plots_dir']
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save plots with environment name in filename
    base_filename = f"regret_with_ci_{env_name.lower()}"
    plt.savefig(os.path.join(plots_dir, f"{base_filename}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(plots_dir, f"{base_filename}.pdf"), bbox_inches='tight')
    plt.close() 