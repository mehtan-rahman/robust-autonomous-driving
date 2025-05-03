# planners/extensions/policy_conditioned.py
"""
Implementation of Policy-Conditioned Uncertainty approach.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np


class QNetDrop(nn.Module):
    """
    Q-network with dropout for estimating uncertainty in action values.
    """

    def __init__(self, input_dim, hidden=64, dropout=0.5, n_actions=5):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden, n_actions)

    def forward(self, x):
        """Forward pass through the network"""
        return self.head(self.base(x))


class PolicyConditionedUncPlanner:
    """
    Planner that selects actions using lower confidence bounds from a Q-network.
    """

    def __init__(self, make_env_fn, q_net, samples=20):
        """
        Initialize the planner.

        Args:
            make_env_fn: Function to create the environment
            q_net: Q-network with dropout
            samples: Number of Monte Carlo samples for uncertainty estimation
        """
        self.env = make_env_fn(0.05)
        self.q_net = q_net
        self.samples = samples

    def act(self, obs):
        """
        Select an action based on lower confidence bounds of Q-values.

        Args:
            obs: Current observation

        Returns:
            int: The selected action
        """
        inp = torch.tensor(obs.flatten(), dtype=torch.float32).unsqueeze(0)
        self.q_net.train()  # Enable dropout for MC sampling
        qs = torch.stack([self.q_net(inp) for _ in range(self.samples)])  # [S,1,A]
        mean_q = qs.mean(0)
        std_q = qs.std(0)
        # Lower confidence bound: μ − 1.96σ
        lcb = mean_q - 1.96 * std_q
        return int(lcb.argmax())


def train_q_network(env, q_net, gamma=0.9, episodes=200, batch_size=64, learning_rate=1e-3, epochs=200):
    """
    Train a Q-network for policy-conditioned uncertainty.

    Args:
        env: Environment to collect data from
        q_net: Q-network model to train
        gamma: Discount factor
        episodes: Number of episodes to collect data from
        batch_size: Batch size for training
        learning_rate: Learning rate
        epochs: Number of training epochs

    Returns:
        The trained Q-network
    """
    optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    # Experience buffer
    buffer = deque(maxlen=10000)

    # Collect random-policy data
    for ep in range(episodes):
        o = env.reset()[0];
        done = False
        while not done:
            a = env.action_space.sample()
            no, r, term, trunc, _ = env.step(a)
            done = term or trunc
            buffer.append((o.flatten(), a, r, no.flatten(), done))
            o = no

    # Train from buffer
    for epoch in range(epochs):
        if len(buffer) < batch_size:
            continue

        batch_idx = np.random.choice(len(buffer), batch_size, replace=False)
        obs_b = torch.tensor([buffer[i][0] for i in batch_idx], dtype=torch.float32)
        act_b = torch.tensor([buffer[i][1] for i in batch_idx], dtype=torch.long)
        rew_b = torch.tensor([buffer[i][2] for i in batch_idx], dtype=torch.float32)
        next_b = torch.tensor([buffer[i][3] for i in batch_idx], dtype=torch.float32)
        done_b = torch.tensor([buffer[i][4] for i in batch_idx], dtype=torch.float32)

        # Current Q
        q_vals = q_net(obs_b)
        q_taken = q_vals.gather(1, act_b.unsqueeze(1)).squeeze(1)

        # Target Q
        with torch.no_grad():
            next_q = q_net(next_b)
            max_next_q = next_q.max(1)[0]
            target_q = rew_b + gamma * max_next_q * (1 - done_b)

        loss = loss_fn(q_taken, target_q)
        optimizer.zero_grad();
        loss.backward();
        optimizer.step()

    print(f"Q-network training complete with final loss: {loss.item():.4f}")
    return q_net
