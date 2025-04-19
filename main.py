# bandits/main.py
import numpy as np
from collections import OrderedDict
import xml.etree.ElementTree as ET
from environments import BernoulliBandit, GaussianBandit
from agents import EpsilonGreedyAgent, UCBAgent, ThompsonSamplingAgent
from utils.seeder import set_seed
from utils.confidence import compute_regret_confidence_intervals
from plots import plot_regret_with_confidence

def load_config(config_path='config/config.xml'):
    """Load configuration from XML file."""
    tree = ET.parse(config_path)
    root = tree.getroot()
    
    config = {
        'paths': {
            'plots_dir': root.find('paths/plots_dir').text,
            'agents_dir': root.find('paths/agents_dir').text,
            'environments_dir': root.find('paths/environments_dir').text,
        },
        'simulation': {
            'n_steps': int(root.find('simulation/n_steps').text),
            'n_trials': int(root.find('simulation/n_trials').text),
            'confidence_levels': [float(level.text) for level in root.findall('simulation/confidence_levels/level')]
        },
        'seeds': {
            'numpy': int(root.find('seeds/numpy').text),
            'random': int(root.find('seeds/random').text)
        },
        'environments': {
            'n_actions': int(root.find('environments/n_actions').text),
            'bernoulli': {
                'probs': [float(p.text) for p in root.findall('environments/bernoulli/probs/prob')] if root.find('environments/bernoulli/probs') is not None else None
            },
            'gaussian': {
                'means': [float(m.text) for m in root.findall('environments/gaussian/means/mean')] if root.find('environments/gaussian/means') is not None else None,
                'stds': [float(s.text) for s in root.findall('environments/gaussian/stds/std')] if root.find('environments/gaussian/stds') is not None else None
            }
        }
    }
    return config

def get_regret(env, agents, n_steps, n_trials):
    """
    Simulates the multi-armed bandit problem for a given environment and a set of agents.
    """
    scores = OrderedDict({
        agent.name: [0.0 for _ in range(n_steps)] for agent in agents
    })

    for trial in range(n_trials):
        env.reset()

        for agent in agents:
            agent.init_actions(env.action_count)

        for i in range(n_steps):
            optimal_reward = env.optimal_reward()

            for agent in agents:
                action = agent.get_action()
                reward = env.pull(action)
                agent.update(action, reward)
                scores[agent.name][i] += optimal_reward - reward

            env.step()

    for agent in agents:
        scores[agent.name] = np.cumsum(scores[agent.name]) / n_trials

    return scores

def main():
    # Load configuration
    config = load_config()
    
    # Set random seeds
    set_seed(config['seeds']['numpy'])
    
    # Initialize environments with configuration
    n_actions = config['environments']['n_actions']
    
    bernoulli_env = BernoulliBandit(
        n_actions=n_actions,
        probs=config['environments']['bernoulli']['probs']
    )
    
    gaussian_env = GaussianBandit(
        n_actions=n_actions,
        means=config['environments']['gaussian']['means'],
        stds=config['environments']['gaussian']['stds']
    )
    
    # Initialize agents
    agents = [
        EpsilonGreedyAgent(),
        UCBAgent(),
        ThompsonSamplingAgent()
    ]
    
    # Run simulations for both environments
    for env_name, env in [('Bernoulli', bernoulli_env), ('Gaussian', gaussian_env)]:
        # Get regret scores
        regret = get_regret(env, agents, config['simulation']['n_steps'], config['simulation']['n_trials'])
        
        # Compute confidence intervals
        confidence_intervals = compute_regret_confidence_intervals(
            regret, 
            config['simulation']['confidence_levels']
        )
        
        # Plot results with environment name
        plot_regret_with_confidence(agents, regret, confidence_intervals, config, env_name)

if __name__ == "__main__":
    main()

# Add seeder for the bandits algorithms
#Now do a confidence interval 99% for the regret
# Create a folder called agents and save the agents in there and their strategies , and __init__.py file to import the agents add base agent  
# create a folder for the plots and save the plots in there
# folder for configuration and save the configuration in there 

