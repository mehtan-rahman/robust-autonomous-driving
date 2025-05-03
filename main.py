# main.py
"""
Main script for running experiments on robust autonomous driving.
"""

import os
import torch
import warnings
import pandas as pd

# Suppress some warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Import project modules
from setup import setup_environment
from environment import make_env, MetricOverlayWrapper
from utils.visualization import record_and_show_successful_episode
from utils.metrics import run_and_dashboard

from planners.random import RandomPlanner
from planners.drop import DeterministicRobustOptimisticPlanner
from planners.interval import IntervalWrapper, IntervalPlanner
from planners.extensions.adaptive import UncertaintyNet, train_uncertainty_net
from planners.extensions.hierarchical import HierarchicalHybridPlanner
from planners.extensions.policy_conditioned import QNetDrop, PolicyConditionedUncPlanner, train_q_network


def run_drop_planner():
    """Run and evaluate the Deterministic Robust Optimistic Planning algorithm"""
    print("\n=== Running DROP Planner ===")

    # Create the planner
    planner = DeterministicRobustOptimisticPlanner(
        make_env_fn=lambda θ: make_env(noise_std=θ),
        horizon=5,
        budget=500,
        gamma=0.9,
        thetas=[0.0, 0.05, 0.1]
    )

    # Run the dashboard
    rewards, collisions = run_and_dashboard(
        make_env_fn=lambda: make_env(noise_std=0.05),
        planner=planner,
        episodes=20,
        max_steps=200,
        model_name="DROP Planner"
    )

    # Record a video
    record_and_show_successful_episode(
        make_env_fn=lambda: MetricOverlayWrapper(make_env(noise_std=0.05)),
        planner=planner,
        video_folder="videos_drop",
        max_steps=200,
        max_tries=5
    )

    return rewards, collisions


def run_interval_planner():
    """Run and evaluate the Interval-based Robust Control"""
    print("\n=== Running Interval Planner ===")

    # Create the planner
    interval_planner = IntervalPlanner()

    # Run the dashboard
    rewards, collisions = run_and_dashboard(
        make_env_fn=lambda: IntervalWrapper(lambda t: make_env(noise_std=t), 0.0, 0.1),
        planner=interval_planner,
        episodes=20,
        max_steps=200,
        model_name="Interval Robust Control"
    )

    # Record a video
    record_and_show_successful_episode(
        make_env_fn=lambda: MetricOverlayWrapper(
            IntervalWrapper(lambda t: make_env(noise_std=t), 0.0, 0.1)),
        planner=interval_planner,
        video_folder="videos_interval",
        max_steps=200,
        max_tries=5
    )

    return rewards, collisions


def run_adaptive_uncertainty():
    """Run and evaluate the Adaptive Uncertainty Quantification extension"""
    print("\n=== Running Adaptive Uncertainty Quantification ===")

    # Create the environment
    env = make_env(noise_std=0.1)

    # Create and train the uncertainty network
    input_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    model = UncertaintyNet(input_dim)
    model = train_uncertainty_net(env, model, episodes=50, epochs=100)

    # Create the planner
    adaptive_planner = RandomPlanner(env_fn=lambda: make_env(noise_std=0.1))

    # Run the dashboard
    rewards, collisions = run_and_dashboard(
        make_env_fn=lambda: MetricOverlayWrapper(make_env(noise_std=0.1)),
        planner=adaptive_planner,
        episodes=20,
        max_steps=200,
        model_name="Adaptive Uncertainty"
    )

    # Record a video
    record_and_show_successful_episode(
        make_env_fn=lambda: MetricOverlayWrapper(make_env(noise_std=0.1)),
        planner=adaptive_planner,
        video_folder="videos_ext1",
        max_steps=200,
        max_tries=10
    )

    return rewards, collisions


def run_hierarchical_hybrid():
    """Run and evaluate the Hierarchical Hybrid Approach extension"""
    print("\n=== Running Hierarchical Hybrid Approach ===")

    # Create the planner
    hier_planner = HierarchicalHybridPlanner(lambda θ=None: make_env(noise_std=0.05))

    # Run the dashboard
    rewards, collisions = run_and_dashboard(
        make_env_fn=lambda: make_env(noise_std=0.05),
        planner=hier_planner,
        episodes=20,
        max_steps=200,
        model_name="Hierarchical Hybrid"
    )

    # Record a video
    record_and_show_successful_episode(
        make_env_fn=lambda: MetricOverlayWrapper(make_env(noise_std=0.05)),
        planner=hier_planner,
        video_folder="videos_ext2",
        max_steps=250,
        max_tries=1
    )

    return rewards, collisions


def run_policy_conditioned():
    """Run and evaluate the Policy-Conditioned Uncertainty extension"""
    print("\n=== Running Policy-Conditioned Uncertainty ===")

    # Create the environment
    env = make_env(noise_std=0.05)

    # Create and train the Q-network
    input_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    q_net = QNetDrop(
        input_dim=input_dim,
        n_actions=env.action_space.n
    )
    q_net = train_q_network(env, q_net)

    # Create the planner
    pcu_planner = PolicyConditionedUncPlanner(lambda θ=None: make_env(noise_std=0.05), q_net)

    # Run the dashboard
    rewards, collisions = run_and_dashboard(
        make_env_fn=lambda: make_env(noise_std=0.05),
        planner=pcu_planner,
        episodes=20,
        max_steps=200,
        model_name="Policy-Conditioned Uncertainty"
    )

    # Record a video
    record_and_show_successful_episode(
        make_env_fn=lambda: MetricOverlayWrapper(make_env(noise_std=0.05)),
        planner=pcu_planner,
        video_folder="videos_ext3",
        max_steps=200,
        max_tries=1
    )

    return rewards, collisions


def compare_results(results):
    """Compare the results of different planners"""
    import matplotlib.pyplot as plt
    import numpy as np

    # Extract metrics
    planners = list(results.keys())
    avg_rewards = [np.mean(results[p]['rewards']) for p in planners]
    avg_collisions = [np.mean(results[p]['collisions']) for p in planners]

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Rewards
    ax1.bar(planners, avg_rewards)
    ax1.set_ylabel('Average Return')
    ax1.set_title('Average Returns by Planner')
    ax1.set_xticklabels(planners, rotation=45, ha='right')

    # Collisions
    ax2.bar(planners, avg_collisions)
    ax2.set_ylabel('Average Collisions')
    ax2.set_title('Average Collisions by Planner')
    ax2.set_xticklabels(planners, rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

    return fig


if __name__ == "__main__":
    # Create output directories
    os.makedirs("videos_drop", exist_ok=True)
    os.makedirs("videos_interval", exist_ok=True)
    os.makedirs("videos_ext1", exist_ok=True)
    os.makedirs("videos_ext2", exist_ok=True)
    os.makedirs("videos_ext3", exist_ok=True)

    # Run all planners and collect results
    results = {}

    # Baseline methods
    drop_rewards, drop_collisions = run_drop_planner()
    results['DROP'] = {'rewards': drop_rewards, 'collisions': drop_collisions}

    interval_rewards, interval_collisions = run_interval_planner()
    results['Interval'] = {'rewards': interval_rewards, 'collisions': interval_collisions}

    # Extensions
    adaptive_rewards, adaptive_collisions = run_adaptive_uncertainty()
    results['Adaptive'] = {'rewards': adaptive_rewards, 'collisions': adaptive_collisions}

    hierarchical_rewards, hierarchical_collisions = run_hierarchical_hybrid()
    results['Hierarchical'] = {'rewards': hierarchical_rewards, 'collisions': hierarchical_collisions}

    policy_conditioned_rewards, policy_conditioned_collisions = run_policy_conditioned()
    results['PolicyConditioned'] = {'rewards': policy_conditioned_rewards, 'collisions': policy_conditioned_collisions}

    # Compare results
    compare_fig = compare_results(results)

    print("All experiments completed successfully!")