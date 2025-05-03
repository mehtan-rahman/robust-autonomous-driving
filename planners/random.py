# planners/random.py
"""
Implementation of a random planner - serves as a baseline.
"""


class RandomPlanner:
    """
    Trivial planner that samples actions randomly from the action space.
    Useful as a baseline for comparison.
    """

    def __init__(self, env_fn):
        self.env_fn = env_fn

    def act(self, obs):
        """Return a random action from the action space"""
        # Sample from a fresh env to get the correct action_space
        return self.env_fn().action_space.sample()