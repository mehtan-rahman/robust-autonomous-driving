# environment.py
"""
Environment creation and wrapper classes for robust autonomous driving.
"""

import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper, Wrapper
import cv2


class NoisyObservation(ObservationWrapper):
    """
    Makes the agent more robust by simulating sensor noise.
    Adds Gaussian noise to velocity components of the observation.
    """

    def __init__(self, env, std_dev):
        super().__init__(env)
        self.std_dev = std_dev

    def observation(self, obs):
        # Unpack if needed
        if isinstance(obs, tuple):
            obs = obs[0]

        obs = np.array(obs)  # make a writable copy
        obs[:, 3:5] += np.random.normal(0, self.std_dev, size=obs[:, 3:5].shape)
        return obs


class MetricOverlayWrapper(Wrapper):
    """
    Superimposes total reward & collision count onto the rendered RGB frames.
    Useful for visualization and debugging.
    """

    def __init__(self, env):
        super().__init__(env)
        self.total_reward = 0.0
        self.collision_count = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.total_reward = 0.0
        self.collision_count = 0
        return obs, info

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        # update metrics
        self.total_reward += r
        if info.get("crashed", False) or info.get("collision", False):
            self.collision_count += 1
        # pass metrics along if you want
        info["total_reward"] = self.total_reward
        info["collision_count"] = self.collision_count
        return obs, r, terminated, truncated, info

    def render(self, *args, **kwargs):
        from PIL import Image

        frame = self.env.render(*args, **kwargs)

        # Ensure proper format
        if isinstance(frame, Image.Image):
            frame = np.array(frame.convert("RGB"))
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        if np.issubdtype(frame.dtype, np.floating):
            frame = np.clip(frame * 255, 0, 255).astype(np.uint8)

        # Fix: make frame contiguous in memory
        frame = np.ascontiguousarray(frame)

        # Add overlay
        text = f"Reward: {self.total_reward:.1f}  Collisions: {self.collision_count}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        return frame


def make_env(env_name="highway-v0", noise_std=0.0):
    """
    Creates and configures the highway environment.
    Optionally wraps it with the NoisyObservation wrapper.

    Args:
        env_name (str): Name of the environment to create
        noise_std (float): Standard deviation of the noise to add to observations

    Returns:
        gym.Env: The created and configured environment
    """
    config = {
        "observation": {"type": "Kinematics"},
        "policy_frequency": 5,
        "simulation_frequency": 15,
        "duration": 40,
        "vehicles_count": 15,
        "screen_width": 600,
        "screen_height": 150,
        "show_trajectories": True,
        "render_agent": True
    }
    env = gym.make(env_name, config=config, render_mode="rgb_array")
    if noise_std > 0.0:
        env = NoisyObservation(env, noise_std)
    return env