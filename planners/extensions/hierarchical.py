# planners/extensions/hierarchical.py
"""
Implementation of a Hierarchical Hybrid approach.
"""

from planners.drop import DeterministicRobustOptimisticPlanner


class HierarchicalHybridPlanner:
    """
    Hybrid planner that switches between high-level DROP planning
    and simple reactive rules.
    """

    def __init__(self, make_env_fn, high_horizon=3, low_horizon=1):
        """
        Initialize the hierarchical planner.

        Args:
            make_env_fn: Function to create the environment
            high_horizon: Planning horizon for high-level planner
            low_horizon: Planning horizon for low-level planner
        """
        self.high_planner = DeterministicRobustOptimisticPlanner(
            make_env_fn=make_env_fn,
            horizon=high_horizon,
            budget=100,
            gamma=0.9,
            thetas=[0.0, 0.05, 0.1]
        )
        self.next_high = 0  # Initial action

    def act(self, obs, step):
        """
        Select an action based on the current observation and step number.

        Args:
            obs: Current observation
            step: Current step number

        Returns:
            int: The selected action
        """
        # Every 5 steps do a high-level DROP action
        if step % 5 == 0:
            self.next_high = self.high_planner.act(obs)

        # Between high decisions, use a simple rule:
        # If current velocity < threshold, choose ACCELERATE (action 2), else IDLE (action 0)
        vel = obs[0, 3]
        return self.next_high if step % 5 == 0 else (2 if vel < 0.5 else 0)