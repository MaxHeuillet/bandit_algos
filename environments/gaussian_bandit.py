import numpy as np
from .base_environment import BaseEnvironment

class GaussianBandit(BaseEnvironment):
    def __init__(self, means=None, stds=None, n_actions=10):
        """
        Initialize a Gaussian bandit environment.
        
        Args:
            means (list or np.array): List of means for each arm. If None, random means are generated.
            stds (list or np.array): List of standard deviations for each arm. If None, random stds are generated.
            n_actions (int): Number of actions if means and stds are not provided.
        """
        super().__init__()
        if means is not None:
            # Convert means to numpy array if it's a list
            means = np.array(means, dtype=float)
            if len(means) != n_actions:
                raise ValueError(f"Number of means ({len(means)}) must match number of actions ({n_actions})")
            self.means = means
        else:
            self.means = np.random.normal(0, 1, size=n_actions)
            
        if stds is not None:
            # Convert stds to numpy array if it's a list
            stds = np.array(stds, dtype=float)
            if len(stds) != n_actions:
                raise ValueError(f"Number of standard deviations ({len(stds)}) must match number of actions ({n_actions})")
            if not np.all(stds > 0):
                raise ValueError("All standard deviations must be positive")
            self.stds = stds
        else:
            self.stds = np.random.uniform(0.1, 1, size=n_actions)
            
        self.action_count = n_actions
        self._initial_means = np.copy(self.means)
        self._initial_stds = np.copy(self.stds)
        
        # Store the optimal action index
        self._optimal_action = np.argmax(self.means)
        
    def pull(self, action):
        """
        Pull the specified arm and return the reward.
        
        Args:
            action (int): The index of the action to take.
            
        Returns:
            float: The reward from the selected action.
            
        Raises:
            ValueError: If the action is out of bounds.
        """
        if not (0 <= action < self.action_count):
            raise ValueError(f"Action {action} is out of bounds. Must be between 0 and {self.action_count - 1}")
        
        # Generate reward and ensure it's non-negative
        reward = np.random.normal(self.means[action], self.stds[action])
        return float(max(0, reward))  # Ensure non-negative rewards
        
    def optimal_reward(self):
        """
        Return the optimal reward (maximum mean).
        
        Returns:
            float: The maximum mean among all actions.
        """
        return float(self.means[self._optimal_action])
        
    def step(self):
        """
        Step the environment. In this implementation, it does nothing as it's a stationary environment.
        """
        pass
        
    def reset(self):
        """
        Reset the environment to its initial state.
        """
        self.means = np.copy(self._initial_means)
        self.stds = np.copy(self._initial_stds)
        self._optimal_action = np.argmax(self.means) 