"""
Optimal solver for generating training labels.

For small instances: enumerate partitions + TSP within each robot.
For larger instances: use iterative Hungarian as a strong baseline.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from itertools import permutations
from scipy.optimize import linear_sum_assignment


def compute_sequential_cost(waypoint_indices: List[int],
                            robot_idx: int,
                            cost_matrix: np.ndarray,
                            wp_to_wp_cost: np.ndarray) -> float:
    """
    Compute total travel cost for a robot visiting waypoints in order.

    Cost = C[robot, wp_0] + sum(wp_to_wp_cost[wp_i, wp_{i+1}])
    """
    if not waypoint_indices:
        return 0.0

    total = cost_matrix[robot_idx, waypoint_indices[0]]
    for i in range(len(waypoint_indices) - 1):
        total += wp_to_wp_cost[waypoint_indices[i], waypoint_indices[i + 1]]
    return total


def compute_makespan(assignments: Dict[int, List[int]],
                     cost_matrix: np.ndarray,
                     wp_to_wp_cost: np.ndarray) -> float:
    """Compute makespan = max robot completion time."""
    max_cost = 0.0
    for r, wp_list in assignments.items():
        cost = compute_sequential_cost(wp_list, r, cost_matrix, wp_to_wp_cost)
        max_cost = max(max_cost, cost)
    return max_cost


def _tsp_order_brute(waypoint_indices: List[int],
                     robot_idx: int,
                     cost_matrix: np.ndarray,
                     wp_to_wp_cost: np.ndarray) -> List[int]:
    """Find best visit order by brute-force TSP (for small sets)."""
    if len(waypoint_indices) <= 1:
        return waypoint_indices

    best_order = None
    best_cost = float('inf')
    for perm in permutations(waypoint_indices):
        cost = compute_sequential_cost(list(perm), robot_idx, cost_matrix, wp_to_wp_cost)
        if cost < best_cost:
            best_cost = cost
            best_order = list(perm)
    return best_order


def _tsp_order_greedy(waypoint_indices: List[int],
                      robot_idx: int,
                      cost_matrix: np.ndarray,
                      wp_to_wp_cost: np.ndarray) -> List[int]:
    """Greedy nearest-neighbor TSP ordering."""
    if len(waypoint_indices) <= 1:
        return waypoint_indices

    remaining = set(waypoint_indices)
    order = []

    # Start with nearest waypoint to robot
    first = min(remaining, key=lambda w: cost_matrix[robot_idx, w])
    order.append(first)
    remaining.remove(first)

    while remaining:
        last = order[-1]
        nearest = min(remaining, key=lambda w: wp_to_wp_cost[last, w])
        order.append(nearest)
        remaining.remove(nearest)

    return order


def solve_optimal(cost_matrix: np.ndarray,
                  wp_to_wp_cost: np.ndarray,
                  n_robots: int,
                  n_waypoints: int) -> Tuple[Dict[int, List[int]], float]:
    """
    Find assignment that minimizes makespan.

    For small instances (n_waypoints <= 12): enumerate partition assignments.
    For larger: use balanced Hungarian heuristic.

    Args:
        cost_matrix: (n_robots, n_waypoints) robot→waypoint costs
        wp_to_wp_cost: (n_waypoints, n_waypoints) waypoint→waypoint costs
        n_robots: number of robots
        n_waypoints: number of waypoints
    Returns:
        (assignments, makespan)
    """
    if n_waypoints <= 8 and n_robots <= 3:
        return _solve_exact(cost_matrix, wp_to_wp_cost, n_robots, n_waypoints)
    else:
        return _solve_balanced_hungarian(cost_matrix, wp_to_wp_cost, n_robots, n_waypoints)


def _solve_exact(cost_matrix: np.ndarray,
                 wp_to_wp_cost: np.ndarray,
                 n_robots: int,
                 n_waypoints: int) -> Tuple[Dict[int, List[int]], float]:
    """
    Enumerate all possible assignments of waypoints to robots.
    For each assignment, find best TSP order per robot, compute makespan.
    """
    best_assignment = None
    best_makespan = float('inf')

    # Each waypoint can go to any robot: n_robots^n_waypoints combinations
    # Cap at reasonable size
    total_combos = n_robots ** n_waypoints
    if total_combos > 500000:
        return _solve_balanced_hungarian(cost_matrix, wp_to_wp_cost, n_robots, n_waypoints)

    for combo in range(total_combos):
        # Decode combo into assignment
        assignments = {r: [] for r in range(n_robots)}
        temp = combo
        for w in range(n_waypoints):
            r = temp % n_robots
            assignments[r].append(w)
            temp //= n_robots

        # Find best order per robot
        ordered_assignments = {}
        for r in range(n_robots):
            if len(assignments[r]) <= 6:
                ordered_assignments[r] = _tsp_order_brute(
                    assignments[r], r, cost_matrix, wp_to_wp_cost)
            else:
                ordered_assignments[r] = _tsp_order_greedy(
                    assignments[r], r, cost_matrix, wp_to_wp_cost)

        makespan = compute_makespan(ordered_assignments, cost_matrix, wp_to_wp_cost)
        if makespan < best_makespan:
            best_makespan = makespan
            best_assignment = ordered_assignments

    return best_assignment, best_makespan


def _solve_balanced_hungarian(cost_matrix: np.ndarray,
                              wp_to_wp_cost: np.ndarray,
                              n_robots: int,
                              n_waypoints: int) -> Tuple[Dict[int, List[int]], float]:
    """
    Balanced assignment heuristic:
    1. Use Hungarian to assign ~equal waypoints per robot
    2. Refine with local swap search to minimize makespan
    """
    # Target: each robot gets about n_waypoints / n_robots
    per_robot = max(1, (n_waypoints + n_robots - 1) // n_robots)

    # Create expanded cost matrix: replicate each robot per_robot times
    expanded_cost = np.zeros((n_robots * per_robot, n_waypoints))
    for r in range(n_robots):
        for k in range(per_robot):
            expanded_cost[r * per_robot + k, :] = cost_matrix[r, :]

    # Pad if needed (more rows than columns is fine for linear_sum_assignment)
    if expanded_cost.shape[0] < n_waypoints:
        pad = np.full((n_waypoints - expanded_cost.shape[0], n_waypoints), 1e9)
        expanded_cost = np.vstack([expanded_cost, pad])

    row_ind, col_ind = linear_sum_assignment(expanded_cost[:, :n_waypoints])

    assignments = {r: [] for r in range(n_robots)}
    for row, col in zip(row_ind, col_ind):
        if col < n_waypoints and row < n_robots * per_robot:
            r = row // per_robot
            assignments[r].append(col)

    # TSP ordering per robot
    for r in range(n_robots):
        if len(assignments[r]) <= 8:
            assignments[r] = _tsp_order_brute(assignments[r], r, cost_matrix, wp_to_wp_cost)
        else:
            assignments[r] = _tsp_order_greedy(assignments[r], r, cost_matrix, wp_to_wp_cost)

    # Local swap refinement: try moving waypoints from busiest to least busy robot
    for _ in range(50):
        costs = {r: compute_sequential_cost(assignments[r], r, cost_matrix, wp_to_wp_cost)
                 for r in range(n_robots)}
        busiest = max(costs, key=costs.get)
        lightest = min(costs, key=costs.get)

        if busiest == lightest or len(assignments[busiest]) <= 1:
            break

        # Try swapping each waypoint from busiest to lightest
        improved = False
        for wp in list(assignments[busiest]):
            new_busiest = [w for w in assignments[busiest] if w != wp]
            new_lightest = assignments[lightest] + [wp]

            # Re-order
            if len(new_busiest) <= 8:
                new_busiest = _tsp_order_brute(new_busiest, busiest, cost_matrix, wp_to_wp_cost)
            else:
                new_busiest = _tsp_order_greedy(new_busiest, busiest, cost_matrix, wp_to_wp_cost)

            if len(new_lightest) <= 8:
                new_lightest = _tsp_order_brute(new_lightest, lightest, cost_matrix, wp_to_wp_cost)
            else:
                new_lightest = _tsp_order_greedy(new_lightest, lightest, cost_matrix, wp_to_wp_cost)

            new_cost_b = compute_sequential_cost(new_busiest, busiest, cost_matrix, wp_to_wp_cost)
            new_cost_l = compute_sequential_cost(new_lightest, lightest, cost_matrix, wp_to_wp_cost)
            new_makespan = max(new_cost_b, new_cost_l,
                               max(costs[r] for r in range(n_robots) if r not in (busiest, lightest)))

            if new_makespan < costs[busiest]:
                assignments[busiest] = new_busiest
                assignments[lightest] = new_lightest
                improved = True
                break

        if not improved:
            break

    makespan = compute_makespan(assignments, cost_matrix, wp_to_wp_cost)
    return assignments, makespan


def assignment_to_label_matrix(assignments: Dict[int, List[int]],
                               n_robots: int,
                               n_waypoints: int) -> np.ndarray:
    """
    Convert assignment dict to binary label matrix for supervised learning.
    Returns (n_robots, n_waypoints) where Y[r, w] = 1 if waypoint w assigned to robot r.
    """
    Y = np.zeros((n_robots, n_waypoints), dtype=np.float32)
    for r, wp_list in assignments.items():
        for w in wp_list:
            Y[r, w] = 1.0
    return Y
