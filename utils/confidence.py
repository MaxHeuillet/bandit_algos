import numpy as np
from scipy import stats

def compute_confidence_interval(data, confidence_level=0.95):
    """
    Compute confidence interval for the given data.
    
    Args:
        data (np.array): Array of data points
        confidence_level (float): Confidence level (e.g., 0.95 for 95% CI)
        
    Returns:
        tuple: (lower_bound, upper_bound)
    """
    mean = np.mean(data)
    std_err = stats.sem(data)
    ci = stats.t.interval(confidence_level, len(data)-1, loc=mean, scale=std_err)
    return ci

def compute_regret_confidence_intervals(regret_data, confidence_levels=[0.95, 0.99]):
    """
    Compute confidence intervals for regret data at multiple confidence levels.
    
    Args:
        regret_data (dict): Dictionary of regret data for each agent
        confidence_levels (list): List of confidence levels to compute
        
    Returns:
        dict: Dictionary of confidence intervals for each agent and confidence level
    """
    ci_data = {}
    
    for agent_name, regrets in regret_data.items():
        ci_data[agent_name] = {}
        for level in confidence_levels:
            ci_data[agent_name][f"{int(level*100)}%"] = compute_confidence_interval(regrets, level)
            
    return ci_data 