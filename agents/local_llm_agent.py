import numpy as np
import random
import time
import os
import re
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta

from .base_agent import BaseAgent

from typing import Optional, List, Tuple
import numpy as np
from vllm import LLM, SamplingParams


class LLMAgent(BaseAgent):

    def __init__(self, config, ):

        super().__init__("LLM")

        self.model = LLM(
            model=config.agents.llm_agent.model,
            # tensor_parallel_size=4,
            dtype="auto",
            trust_remote_code=True,
            enforce_eager=True,
        )

        n_samples = 1
        temperature = 0.6
        top_p = 0.95
        max_tokens = 32000 
        max_test = 999999

        self.sampling_params = SamplingParams(
            n=n_samples,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            logprobs=2,
        )
        
        # Initialize state
        self._rewards: Optional[np.ndarray] = None
        self._counts: Optional[np.ndarray] = None
        self._successes: Optional[np.ndarray] = None
        self._failures: Optional[np.ndarray] = None
        self._action_history: List[int] = []
        self._reward_history: List[float] = []
        
    def reset(self):
        """Reset the agent's internal state."""
        self._rewards = np.zeros(self.action_count)
        self._counts = np.zeros(self.action_count)
        self._successes = np.zeros(self.action_count)
        self._failures = np.zeros(self.action_count)
        self._action_history = []
        self._reward_history = []
    
    def init_actions(self, n_actions):
        """Initialize the agent's internal state."""
        super().init_actions(n_actions)
        self._rewards = np.zeros(n_actions)
        self._counts = np.zeros(n_actions)
        self._successes = np.zeros(n_actions)
        self._failures = np.zeros(n_actions)
        self._action_history = []
        self._reward_history = []
    
    def _get_context_prompt(self) -> str:
        """
        Generate a detailed context prompt with action history and statistics.
        
        Returns:
            Formatted context string for the LLM
        """
        if not self._action_history:
            return "This is the first action. You should explore different actions to learn their reward probabilities."
        
        n_actions = len(self._rewards)
        total_steps = len(self._action_history)
        
        # Calculate statistics for each action
        action_stats = []
        for a in range(n_actions):
            if self._counts[a] > 0:
                rewards = [r for act, r in zip(self._action_history, self._reward_history) if act == a]
                mean_reward = np.mean(rewards)
                std_reward = np.std(rewards, ddof=1) if len(rewards) > 1 else 0
                success_rate = self._successes[a] / self._counts[a] if self._counts[a] > 0 else 0
                action_stats.append({
                    'action': a,
                    'count': int(self._counts[a]),
                    'successes': int(self._successes[a]),
                    'failures': int(self._failures[a]),
                    'mean_reward': mean_reward,
                    'std_reward': std_reward,
                    'success_rate': success_rate
                })
            else:
                action_stats.append({
                    'action': a,
                    'count': 0,
                    'successes': 0,
                    'failures': 0,
                    'mean_reward': None,
                    'std_reward': None,
                    'success_rate': None
                })
        
        # Generate context
        context = f"""You are an advanced decision-making AI playing a multi-armed bandit game with {n_actions} actions.
                      Your goal is to minimize cumulative regret by choosing actions that maximize expected reward.
                      Current statistics (step {total_steps}):"""
        
        # Add action statistics
        context += "\nAction Statistics:"
        context += "\n| Action | Count | Successes | Failures | Mean Reward | Std Dev | Success Rate |"
        context += "\n|--------|-------|-----------|----------|-------------|---------|--------------|"
        
        for stats in action_stats:
            if stats['count'] > 0:
                context += f"\n| {stats['action']:6d} | {stats['count']:5d} | {stats['successes']:9d} | {stats['failures']:8d} | {stats['mean_reward']:.4f}    | {stats['std_reward']:.4f}  | {stats['success_rate']:.4f}    |"
            else:
                context += f"\n| {stats['action']:6d} |     0 |         0 |        0 |      ?.????    |   ?.????  |      ?.????    |"
        
        # Add recent history
        context += "\n\nRecent History (last 5 steps):\n"
        context += "| Step | Action | Reward |\n"
        context += "|------|--------|--------|\n"
        
        # Calculate actual step numbers
        total_steps = len(self._action_history)
        for i in range(max(0, len(self._action_history)-5), len(self._action_history)):
            actual_step = total_steps - (len(self._action_history) - i)
            context += f"| {actual_step:4d} | {self._action_history[i]:6d} | {self._reward_history[i]:.4f} |\n"
        
        # Add guidance
        context += """
                        Guidance:
                        1. Your goal is to minimize cumulative regret by maximizing expected reward.
                        2. Consider both the mean reward and uncertainty (std dev) of each action.
                        3. Balance exploration (trying actions with high uncertainty) with exploitation (choosing actions with high mean rewards).
                        4. For untried actions, consider exploring them to gather more information.
                        5. Look for patterns in the recent history that might indicate changes in reward distributions.
                        6. Use the success rate and failure counts to assess the reliability of each action.

                        Please respond with just the action number (0-{}) that you think will minimize regret.""".format(n_actions-1)
        
        return context

    def _call_llm_api(self, prompt: str) -> str:
        """
        Call the LLM API with retries and error handling.
        """

        # Prepare messages
        messages = [
            {"role": "system", "content": "You are an AI assistant that helps with decision making in multi-armed bandit problems. Respond with just the action number."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.model.generate(messages, self.sampling_params)
                
        # Extract the response text
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content.strip()
        else:
            raise ValueError("Empty response from LLM")
                        
    def _parse_llm_response(self, response: str) -> int:
        """
        Parse the LLM's response to extract the chosen action.
        """
        if not response:
            return None
            
        # Clean the response
        response = response.strip()
        
        # Look for patterns like "Action X" or "Choose X"
        patterns = [
            r'action\s*(\d+)',
            r'choose\s*(\d+)',
            r'select\s*(\d+)',
            r'\b(\d+)\b'  # Any number
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    action = int(match.group(1))
                    if 0 <= action < self.action_count:
                        return action
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def choose_action(self) -> int:
        """Choose an action using the LLM's reasoning capabilities."""

        context = self._get_context_prompt()

        response = self._call_llm_api(context)
            
        action = self._parse_llm_response(response)
    
    def update(self, action: int, reward: float) -> None:
        """Update the agent's knowledge based on the action taken and reward received."""
        if self._counts is None or self._rewards is None or self._successes is None or self._failures is None:
            raise RuntimeError("Agent not initialized. Call init_actions() first.")
            
        self._rewards[action] += reward
        self._counts[action] += 1
        
        # Update successes and failures
        if reward > 0:
            self._successes[action] += 1
        else:
            self._failures[action] += 1
        
        # Use incremental update for numerical stability
        if self._counts[action] == 1:
            self._rewards[action] = reward
        else:
            # Update the mean incrementally
            self._rewards[action] += (reward - self._rewards[action]) / self._counts[action]
        
        # Update history
        self._action_history.append(action)
        self._reward_history.append(reward)
    
    @property
    def name(self):
        """Return the name of the agent."""
        return f"LLM ({self.model}, temp={self.temperature})"

    def get_action(self) -> int:
        """
        Required method from the abstract base class.
        This is an alias for choose_action to maintain compatibility.
        """
        return self.choose_action()
