# bandits/epsilon.py
from typing import Optional, Union
import numpy as np
from .base_agent import BaseAgent

class EpsilonGreedyAgent(BaseAgent):
    """
    An agent that explores with probability epsilon and exploits (chooses the best action so far)
    with probability 1-epsilon.
    """

    def __init__(self, config, ):
        """Initializes the EpsilonGreedyAgent."""
        super().__init__("EpsilonGreedy")
        self._epsilon = config.agents.epsilon_greedy.epsilon
        self._successes: Optional[np.ndarray] = None
        self._failures: Optional[np.ndarray] = None
        self.t: int = 0  # Time step counter

    def init_actions(self, n_actions: int) -> None:
        """Initializes the agent's internal state."""
        super().init_actions(n_actions)
        self._successes = np.zeros(n_actions)
        self._failures = np.zeros(n_actions)

    def get_action(self) -> int:
        """Chooses an action based on the epsilon-greedy strategy."""
        self.t += 1  # Increment time step
        
        if self._successes is None or self._failures is None:
            raise ValueError("Agent has not been initialized. Call init_actions() first.")
            
        current_epsilon = min(1.0, 1.0 / np.sqrt(self.t))
            
        if np.random.random() < current_epsilon:
            return np.random.randint(len(self._successes))
            
        total_counts = self._successes + self._failures
        values = np.where(total_counts > 0,
                              self._successes / total_counts,
                              np.zeros_like(self._successes))
        
        return np.argmax(values)

    def update(self, action: int, reward: Union[int, float]) -> None:
        """Updates the agent's internal state based on the action taken and reward received."""
        if reward == 1:
            self._successes[action] += 1
        else:
            self._failures[action] += 1
        

    @property
    def name(self) -> str:
        """Returns the name of the agent, including the epsilon value and environment type."""
        return f"{self._name}(epsilon={self._epsilon}, {self.environment_type})"
