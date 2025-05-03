# utils/metrics.py
"""
Metrics tracking and visualization utilities.
"""

import inspect
import matplotlib.pyplot as plt
from IPython.display import clear_output, display


def run_and_dashboard(make_env_fn, planner, episodes, max_steps, model_name):
    """
    Runs multiple episodes of a planner in the environment and displays
    a real-time dashboard that tracks performance metrics.

    Args:
        make_env_fn: Function that creates the environment
        planner: The planner/agent to use for decision making
        episodes: Number of episodes to run
        max_steps: Maximum number of steps per episode
        model_name: Name of the model/planner for display purposes

    Returns:
        tuple: Lists of rewards and collisions per episode
    """
    rewards = []
    collisions = []
    env = make_env_fn()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Check if planner.act requires a `step` argument
    needs_step = 'step' in inspect.signature(planner.act).parameters

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        collision_count = 0
        step = 0  # <- step value starts here, as before

        while not done and step < max_steps:
            if needs_step:
                action = planner.act(obs, step)
            else:
                action = planner.act(obs)

            obs, r, term, trunc, info = env.step(action)
            total_reward += r
            collision_count = info.get("collision_count", collision_count)
            done = term or trunc
            step += 1  # increment step for next loop

        rewards.append(total_reward)
        collisions.append(collision_count)

        clear_output(wait=True)
        ax1.clear();
        ax2.clear()

        ax1.plot(range(1, ep + 1), rewards, marker='o')
        ax1.set_title(f"{model_name} - Episode Returns")
        ax1.set_xlabel("Episode");
        ax1.set_ylabel("Return")

        ax2.plot(range(1, ep + 1), collisions, marker='o')
        ax2.set_title(f"{model_name} - Collisions")
        ax2.set_xlabel("Episode");
        ax2.set_ylabel("Collision Count")

        display(fig)

    plt.close(fig)
    return rewards, collisions