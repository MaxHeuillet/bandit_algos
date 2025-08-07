# main.py
import numpy as np
import matplotlib.pyplot as plt
from environments.bernoulli_bandit import BernoulliBandit
from environments.gaussian_bandit import GaussianBandit, generate_configuration
# from agents.llm_agent import LLMAgent
from agents.epsilon import EpsilonGreedyAgent
from agents.ucb import UCBAgent
from agents.thompson import ThompsonSamplingAgent
from utils.confidence import compute_confidence_interval
from plots.plot_utils import plot_regret_with_confidence
import os
import sys
from omegaconf import OmegaConf
from agents.gaussian_epsilon_greedy import GaussianEpsilonGreedyAgent
from agents.gaussian_ucb import GaussianUCBAgent
from agents.gaussian_thompson_sampling import GaussianThompsonSamplingAgent
from agents.ucb_kl import KLUCBAgent

from agents.gradient_bandit import GradientBanditAgent

def load_config():
    """
    Load configuration from YAML files using OmegaConf.
    """
    config_file = 'configurations/default_config.yaml'
    cfg = OmegaConf.load(config_file)
    return cfg

def run_simulation(
    env, agent, n_steps: int, n_trials: int, confidence_levels: list[float]
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Run the bandit simulation with a single agent.
    """
    
    print(f"\nStarting simulation for {agent.name}...")
    regrets = np.zeros((n_trials, n_steps))
    cumulative_regrets = np.zeros((n_trials, n_steps))
    
    for trial in range(n_trials):
        print(f"\nTrial {trial + 1}/{n_trials}")
        env.reset()
        agent.reset()
        agent.init_actions(env.action_count)
            
        for t in range(n_steps):
            action = agent.get_action()
            reward, reward_optimal = env.pull(action)
            agent.update(action, reward)
                
            # Calculate regret
            # optimal_reward = env.optimal_reward()
            # print(optimal_reward, reward)
            regrets[trial, t] = reward_optimal - reward
            if t > 0:
                cumulative_regrets[trial, t] = cumulative_regrets[trial, t-1] + regrets[trial, t]
            else:
                cumulative_regrets[trial, t] = regrets[trial, t]

        print("regrets", regrets)
        print(f"Trial {trial + 1} complete.")
    
    return cumulative_regrets

def main():
    # Load configuration
    config = load_config()
    print("Configuration loaded successfully")
        
    # Set random seeds
    np.random.seed(config.seed)
        
    # Test Bernoulli environment
    print("\nTesting Bernoulli environment with all agents...")
    probs = np.array([0.2, 0.9])
    env = BernoulliBandit(n_actions=len(probs), probs=probs)

    agents = [
            EpsilonGreedyAgent(epsilon=0.1, environment_type='bernoulli'),
            # UCBAgent(),
            # ThompsonSamplingAgent(environment_type='bernoulli'),
            GradientBanditAgent(alpha=0.1, baseline=True),
            # LLMAgent(model="gpt-4.1-nano")
            # KLUCBAgent(n_arms=len(probs)),
     ]
    print(f"Initialized {len(agents)} agents")
        
    # Run simulations for Bernoulli environment
    print("Starting Bernoulli simulations...")
    all_regrets_bernoulli = {}
    all_intervals_bernoulli = {}
        
    for agent in agents:
        print(f"\nTesting {agent.name}...")

        horizon = 100
        n_runs = 1

        regrets = run_simulation(env, agent, horizon, n_runs, config.get('confidence_levels', [0.95])  )
        all_regrets_bernoulli[agent.name] = regrets
        print(f"Completed simulation for {agent.name}")
        print(f"Regrets shape: {regrets.shape}")

    print("Plotting Bernoulli results...")
    plot_regret_with_confidence( agents, all_regrets_bernoulli, config, "Bernoulli" )
        
    print("Done!")


if __name__ == "__main__":
    main()