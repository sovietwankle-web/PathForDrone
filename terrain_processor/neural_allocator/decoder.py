"""
Assignment decoders: convert score matrix to robot-waypoint assignments.

Three strategies:
- Hungarian: optimal one-to-one matching, iterated for many-to-one
- Greedy: fast, picks highest-scoring pairs
- Autoregressive: sequential, returns log-probs for RL training
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from scipy.optimize import linear_sum_assignment


def hungarian_decode(score_matrix: np.ndarray,
                     n_waypoints_actual: int,
                     n_robots_actual: int,
                     max_per_robot: Optional[int] = None
                     ) -> Dict[int, List[int]]:
    """
    Iterative Hungarian assignment for many-to-one allocation.

    Args:
        score_matrix: (n_r, n_w) — higher is better
        n_waypoints_actual: actual number of waypoints (before padding)
        n_robots_actual: actual number of robots
        max_per_robot: max waypoints per robot (None = unlimited)
    Returns:
        {robot_id: [waypoint_indices]}
    """
    S = score_matrix[:n_robots_actual, :n_waypoints_actual].copy()
    assignments = {r: [] for r in range(n_robots_actual)}
    remaining = set(range(n_waypoints_actual))

    while remaining:
        # Build cost matrix for remaining waypoints
        wp_list = sorted(remaining)
        cost = -S[:n_robots_actual, wp_list]  # negate for minimization

        # Mask out full robots
        if max_per_robot is not None:
            for r in range(n_robots_actual):
                if len(assignments[r]) >= max_per_robot:
                    cost[r, :] = 1e9

        row_ind, col_ind = linear_sum_assignment(cost)

        assigned_any = False
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= 1e9:
                continue
            wp_idx = wp_list[c]
            if max_per_robot is None or len(assignments[r]) < max_per_robot:
                assignments[r].append(wp_idx)
                remaining.discard(wp_idx)
                assigned_any = True

        if not assigned_any:
            break

    return assignments


def greedy_decode(score_matrix: np.ndarray,
                  n_waypoints_actual: int,
                  n_robots_actual: int,
                  max_per_robot: Optional[int] = None
                  ) -> Dict[int, List[int]]:
    """
    Greedy assignment: pick highest-scoring (robot, waypoint) pair iteratively.
    """
    S = score_matrix[:n_robots_actual, :n_waypoints_actual].copy()
    assignments = {r: [] for r in range(n_robots_actual)}
    assigned_wp = set()

    while len(assigned_wp) < n_waypoints_actual:
        # Mask assigned waypoints and full robots
        mask = np.full_like(S, -np.inf)
        for r in range(n_robots_actual):
            if max_per_robot is not None and len(assignments[r]) >= max_per_robot:
                continue
            for w in range(n_waypoints_actual):
                if w not in assigned_wp:
                    mask[r, w] = S[r, w]

        if np.all(np.isinf(mask) & (mask < 0)):
            break

        idx = np.unravel_index(np.argmax(mask), mask.shape)
        r, w = int(idx[0]), int(idx[1])
        assignments[r].append(w)
        assigned_wp.add(w)

    return assignments


def autoregressive_decode(scores: torch.Tensor,
                          n_waypoints_actual: int,
                          n_robots_actual: int
                          ) -> Tuple[List[Tuple[int, int]], torch.Tensor]:
    """
    Autoregressive decoding for RL: sequentially assigns waypoints.

    At each step, flatten (n_r, n_w) scores into a single distribution,
    sample one (robot, waypoint) pair, mask that waypoint out.

    Args:
        scores: (n_r, n_w) logits — single sample, no batch dim
    Returns:
        actions: list of (robot_id, waypoint_id) tuples
        log_probs: tensor of log probabilities for each action
    """
    S = scores[:n_robots_actual, :n_waypoints_actual].clone()
    assigned_mask = torch.zeros(n_waypoints_actual, dtype=torch.bool,
                                device=scores.device)
    actions = []
    log_probs = []

    for _ in range(n_waypoints_actual):
        # Mask assigned waypoints
        masked_S = S.clone()
        masked_S[:, assigned_mask] = float('-inf')

        # Flatten to single distribution over (robot, waypoint)
        flat_logits = masked_S.reshape(-1)

        # Check if all masked
        if torch.all(torch.isinf(flat_logits) & (flat_logits < 0)):
            break

        probs = F.softmax(flat_logits, dim=0)
        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)

        r = int(action_idx) // n_waypoints_actual
        w = int(action_idx) % n_waypoints_actual

        actions.append((r, w))
        log_probs.append(log_prob)
        assigned_mask[w] = True

    return actions, torch.stack(log_probs) if log_probs else torch.tensor([])
