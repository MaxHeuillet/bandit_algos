# bandits/bernoulli_bandit.py
import numpy as np
from .base_environment import BaseEnvironment

class BernoulliBandit(BaseEnvironment):
    """
    A Bernoulli bandit with K actions. Each action yields a reward of 1 with probability theta_k
    and 0 otherwise, where theta_k is unknown to the agent but fixed over time.
    """

    def __init__(self, n_actions=10, probs=None):
        """
        Initializes the Bernoulli bandit.

        Args:
            n_actions (int): The number of available actions (arms).
            probs (list or np.array): Optional array of probabilities for each action.
        """
        super().__init__()
        self._probs = probs
        self._initial_probs = np.copy(self._probs)
        self.action_count = n_actions
        
    def pull(self, action):
        """
        Simulates pulling a lever (taking an action) and returns both:
        - The reward from the selected action
        - The reward that would have been obtained by pulling the optimal arm
        (using a single random draw to avoid double sampling variance)
        """

        rand = np.random.random()  # One draw for both

        reward_action = float(rand < self._probs[action])
        reward_optimal = float(rand < np.max(self._probs))

        return reward_action, reward_optimal

    def optimal_reward(self):
        """
        Returns the expected reward of the optimal action. Used for regret calculation.

        Returns:
            float: The maximum probability among all actions.
        """
        return np.max(self._probs)

    def step(self, action):
        """
        Used in non-stationary versions of the bandit to change the probabilities.
        This implementation is stationary, so this method does nothing.
        """
        pass  # Stationary bandit, so no need to change probabilities

    def reset(self):
        """Resets the bandit to its initial state (initial probabilities)."""
        self._probs = np.copy(self._initial_probs)