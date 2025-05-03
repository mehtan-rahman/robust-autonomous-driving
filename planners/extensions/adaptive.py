# planners/extensions/adaptive.py
"""
Implementation of Adaptive Uncertainty Quantification.
"""

import torch
import torch.nn as nn


class UncertaintyNet(nn.Module):
    """
    Neural network with dropout to estimate return uncertainty from states.
    """

    def __init__(self, input_dim, hidden=64, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        """Forward pass through the network"""
        return self.net(x)


def train_uncertainty_net(env, model, episodes=50, epochs=100, lr=1e-3):
    """
    Train an uncertainty estimation network on random rollouts.

    Args:
        env: Environment to collect data from
        model: UncertaintyNet model to train
        episodes: Number of episodes to collect data from
        epochs: Number of training epochs
        lr: Learning rate

    Returns:
        The trained model
    """
    import torch.optim as optim
    from collections import deque

    # Collect data
    data_X, data_y = [], []
    for _ in range(episodes):
        o = env.reset()[0];
        done = False;
        total = 0
        while not done:
            a = env.action_space.sample()
            o, r, term, trunc, _ = env.step(a)
            total += r
            done = term or trunc
        data_X.append(o.flatten())
        data_y.append(total)

    X = torch.tensor(data_X, dtype=torch.float32)
    y = torch.tensor(data_y, dtype=torch.float32).unsqueeze(1)

    # Train model
    opt = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        pred = model(X)
        loss = nn.MSELoss()(pred, y)
        opt.zero_grad();
        loss.backward();
        opt.step()

    return model