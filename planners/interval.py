# planners/interval.py
"""
Implementation of Interval-based Robust Control.
"""

import gymnasium as gym


class IntervalWrapper(gym.Wrapper):
    """
    Creates a robust gym environment wrapper that simulates two versions
    of the environment at extreme parameter values.
    """

    def __init__(self, make_env_fn, theta_low, theta_high):
        # Create two env instances for the extremes
        self.env_low = make_env_fn(theta_low)
        self.env_high = make_env_fn(theta_high)
        # We delegate metadata (action_space, etc.) to env_low
        super().__init__(self.env_low)

    def reset(self, **kwargs):
        """Reset both env_low and env_high to get their initial states"""
        # Grab obs and info from both extremes
        obs_low, info_low = self.env_low.reset(**kwargs)
        obs_high, info_high = self.env_high.reset(**kwargs)
        # Build your interval hull
        self.lo = obs_low.copy()
        self.hi = obs_high.copy()
        # Return ((lo, hi), info) so wrappers stay happy
        return (self.lo, self.hi), {}

    def step(self, action):
        """Step both env_low and env_high with the same action"""
        # Step both envs with the same action
        obs_low, r_low, done_low, trunc_low, info_low = self.env_low.step(action)
        obs_high, r_high, done_high, trunc_high, info_high = self.env_high.step(action)
        # Update interval hull to the new extrema
        self.lo = obs_low.copy()
        self.hi = obs_high.copy()
        # Worst-case reward
        r_robust = min(r_low, r_high)
        # Episode ends if either extreme ends
        terminated = done_low or done_high
        truncated = trunc_low or trunc_high
        # Return 5-tuple: (obs, reward, terminated, truncated, info)
        return (self.lo, self.hi), r_robust, terminated, truncated, {'low': info_low, 'high': info_high}


class IntervalPlanner:
    """
    Simple rule-based planner that works with interval observations.
    Makes decisions based on worst-case values.
    """

    def act(self, obs_tuple):
        """
        Select an action based on the interval observation.

        Args:
            obs_tuple: Tuple of (lower_bound, upper_bound) observations

        Returns:
            int: The selected action
        """
        obs_lo, obs_hi = obs_tuple
        vel_lo = obs_lo[0, 3]
        lateral_pos = obs_lo[0, 1]

        if vel_lo < 0.3:
            return 2  # Accelerate
        elif lateral_pos > 0.3:
            return 3  # Steer left
        elif lateral_pos < -0.3:
            return 1  # Steer right
        else:
            return 0  # Keep straight