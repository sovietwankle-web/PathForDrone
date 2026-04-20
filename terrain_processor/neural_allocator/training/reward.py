"""
Reward computation for RL training.
"""

import numpy as np
from typing import Dict, List, Tuple


def compute_robot_costs(assignments: Dict[int, List[int]],
                        cost_matrix: np.ndarray,
                        wp_to_wp_cost: np.ndarray) -> Dict[int, float]:
    """Compute total travel cost for each robot."""
    costs = {}
    for r, wp_list in assignments.items():
        if not wp_list:
            costs[r] = 0.0
            continue
        total = cost_matrix[r, wp_list[0]]
        for i in range(len(wp_list) - 1):
            total += wp_to_wp_cost[wp_list[i], wp_list[i + 1]]
        costs[r] = total
    return costs


def compute_makespan_reward(assignments, cost_matrix, wp_to_wp_cost):
    """Reward = -makespan."""
    costs = compute_robot_costs(assignments, cost_matrix, wp_to_wp_cost)
    if not costs:
        return 0.0
    return -max(costs.values())


def compute_step_reward(prev_max_cost, new_assignments, cost_matrix, wp_to_wp_cost):
    """Dense step reward: penalize increase in current max robot cost."""
    costs = compute_robot_costs(new_assignments, cost_matrix, wp_to_wp_cost)
    new_max = max(costs.values()) if costs else 0.0
    reward = -(new_max - prev_max_cost)
    return reward, new_max


def compute_balance_bonus(assignments, cost_matrix, wp_to_wp_cost):
    """Bonus for load balance: -std of robot costs."""
    costs = compute_robot_costs(assignments, cost_matrix, wp_to_wp_cost)
    if len(costs) < 2:
        return 0.0
    return -float(np.std(list(costs.values())))


def compute_composite_step_reward(prev_max_cost, assignments, cost_matrix,
                                  wp_to_wp_cost, alpha=0.7, beta=0.3):
    """Composite: makespan + balance bonus."""
    makespan_reward, new_max = compute_step_reward(
        prev_max_cost, assignments, cost_matrix, wp_to_wp_cost)
    balance = compute_balance_bonus(assignments, cost_matrix, wp_to_wp_cost)
    return alpha * makespan_reward + beta * balance, new_max
