# planners/drop.py
"""
Implementation of Deterministic Robust Optimistic Planning algorithm.
"""

import numpy as np


class TreeNode:
    """Node in the planning tree for Deterministic Robust Optimistic Planning"""

    def __init__(self, depth, parent=None, action=None):
        self.depth = depth  # depth in the planning tree
        self.parent = parent
        self.action = action  # action taken to reach this node
        self.children = []  # list of TreeNode
        self.u_r = -np.inf  # worst-case return estimate at this node
        self.b_r = np.inf  # optimistic bound = u_r + bonus

    def is_leaf(self):
        """Check if the node is a leaf (has no children)"""
        return len(self.children) == 0


class DeterministicRobustOptimisticPlanner:
    """
    Implementation of the Deterministic Robust Optimistic Planning algorithm.
    Performs tree search to find actions that maximize worst-case performance.
    """

    def __init__(self, make_env_fn, horizon=3, budget=100, gamma=0.9, thetas=None):
        """
        Initialize the planner.

        Args:
            make_env_fn: Function to create the environment
            horizon: Planning horizon (depth of the search tree)
            budget: Number of nodes to expand in the search tree
            gamma: Discount factor
            thetas: List of uncertainty parameters to consider
        """
        self.make_env_fn = make_env_fn
        self.H = horizon
        self.B = budget
        self.γ = gamma
        self.thetas = thetas or [None]

    def act(self, obs):
        """
        Perform tree search planning from the current state and return the best first action.

        Args:
            obs: Current observation

        Returns:
            int: The selected action
        """
        root = TreeNode(depth=0)
        root.u_r = 0.0
        root.b_r = 1.0 / (1 - self.γ)  # max possible

        # Run B expansions
        for _ in range(self.B):
            leaf = self._select_leaf(root)
            if leaf.depth < self.H:
                self._expand(leaf, obs)
            self._backup(leaf)

        # Choose child of root with largest b_r
        if not root.children:
            return 0
        best = max(root.children, key=lambda c: c.b_r)
        return best.action

    def _select_leaf(self, node):
        """
        Greedily descend from a node to a leaf with the highest b_r value.

        Args:
            node: Starting node

        Returns:
            TreeNode: Selected leaf node
        """
        if node.is_leaf():
            return node
        best = max(node.children, key=lambda c: c.b_r)
        return self._select_leaf(best)

    def _expand(self, node, root_obs):
        """
        Simulate all possible actions from a leaf node and add new children to the planning tree.

        Args:
            node: Node to expand
            root_obs: Root observation
        """
        for a in range(self.make_env_fn(self.thetas[0]).action_space.n):
            # Worst‐case return across all θ
            returns = []
            for θ in self.thetas:
                env = self.make_env_fn(θ)
                obs = env.reset()  # Resets to root_obs if you've stored env state there

                # Replay path to `node` from root
                path = []
                cur = node
                while cur.parent is not None:
                    path.append(cur.action)
                    cur = cur.parent

                for act in reversed(path):
                    obs, _, done, *_ = env.step(act)

                # Now take action `a` at this leaf
                total, γt = 0.0, 1.0
                obs2, r, done, *_ = env.step(a)
                total += γt * r
                γt *= self.γ

                # Simulate greedy default (action=0) until horizon
                for _ in range(node.depth + 1, self.H):
                    obs2, r, done, *_ = env.step(0)
                    total += γt * r
                    γt *= self.γ
                    if done:
                        break

                returns.append(total)

            u_r = min(returns)

            # Create child
            child = TreeNode(depth=node.depth + 1, parent=node, action=a)
            child.u_r = u_r

            # Optimistic bound: u_r + γ^(H−d)/(1−γ)
            child.b_r = u_r + (self.γ ** (self.H - child.depth)) / (1 - self.γ)
            node.children.append(child)

    def _backup(self, node):
        """
        Propagate the best optimistic return (b_r) estimates up the tree from a newly expanded leaf.

        Args:
            node: Starting node for backup
        """
        while node.parent is not None:
            p = node.parent
            p.b_r = max(c.b_r for c in p.children)
            node = p