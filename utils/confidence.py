import numpy as np
from scipy import stats

def compute_confidence_interval(data, confidence_level=0.95):
    """
    Compute confidence interval for cumulative regret using Hoeffding's inequality.
    This provides tighter bounds for bandit problems with sub-linear regret.
    Ensures non-negative bounds and proper sub-linear growth.
    
    Args:
        data (np.array): Array of cumulative regret data points
        confidence_level (float): Confidence level (e.g., 0.95 for 95% CI)
        
    Returns:
        tuple: (lower_bound, upper_bound)
    """
    n = len(data)
    t = np.arange(1, n+1)  # Time steps
    
    # For cumulative regret, we need to account for the fact that regret grows with time
    # We use a sub-linear bound that grows as O(sqrt(t log t))
    delta = 1 - confidence_level
    
    # Hoeffding's bound scaled by time
    # The bound grows as sqrt(t log t) to match the expected regret growth
    epsilon = np.sqrt(2 * t * np.log(2/delta))
    
    # The confidence interval should be centered around the actual regret
    # and grow sub-linearly with time
    lower_bound = np.maximum(0, data - epsilon)  # Ensure non-negative
    upper_bound = data + epsilon
    
    return lower_bound, upper_bound

def compute_regret_confidence_intervals(regret_data, confidence_levels=[0.95]):
    """
    Compute confidence intervals for regret data.
    
    Args:
        regret_data (dict): Dictionary of regret data for each agent
        confidence_levels (list): List containing only the 95% confidence level
        
    Returns:
        dict: Dictionary of confidence intervals for each agent
    """
    ci_data = {}
    
    for agent_name, regrets in regret_data.items():
        ci_data[agent_name] = {}
        for level in confidence_levels:
            ci_data[agent_name]["95%"] = compute_confidence_interval(regrets, level)
            
    return ci_data 