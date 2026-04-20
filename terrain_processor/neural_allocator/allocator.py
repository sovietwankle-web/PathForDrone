"""
NeuralWaypointAllocator — drop-in replacement for MultiRobotWaypointAllocator.

Uses the cross-attention network to assign waypoints, then plans paths
using the existing HierarchicalPathPlanner.
"""

import numpy as np
import torch
import time
from typing import Tuple, Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from viscous_hierarchical_fmm import (
    ViscosityParams, GasDiffusionParams, WaypointAllocationResult,
    HierarchicalPathPlanner, ViscosityField, CompetitiveFMM,
)

from .config import NeuralAllocatorConfig
from .model import WaypointAllocationNet
from .features import FeatureExtractor
from .decoder import hungarian_decode, greedy_decode


class NeuralWaypointAllocator:
    """
    Neural network-based multi-robot waypoint allocator.

    Interface-compatible with MultiRobotWaypointAllocator.
    Uses cross-attention network to predict optimal assignments,
    then plans path segments with the existing HierarchicalPathPlanner.
    """

    def __init__(self,
                 terrain: np.ndarray,
                 fog_data: Optional[np.ndarray] = None,
                 n_levels: int = 3,
                 viscosity_params: Optional[ViscosityParams] = None,
                 gas_params: Optional[GasDiffusionParams] = None,
                 spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
                 model_path: Optional[str] = None,
                 config: Optional[NeuralAllocatorConfig] = None,
                 fallback_to_fmm: bool = True):
        """
        Args:
            terrain, fog_data, n_levels, viscosity_params, gas_params, spacing:
                Same as MultiRobotWaypointAllocator.
            model_path: Path to trained model checkpoint (.pt file).
            config: Neural allocator config.
            fallback_to_fmm: If True, fall back to CompetitiveFMM when model unavailable.
        """
        self.terrain = terrain
        self.gas_params = gas_params or GasDiffusionParams()
        self.viscosity_params = viscosity_params or ViscosityParams()
        self.spacing = spacing
        self.n_levels = n_levels
        self.fog_data = fog_data
        self.fallback_to_fmm = fallback_to_fmm
        self.config = config or NeuralAllocatorConfig()

        # Speed field
        vf = ViscosityField(self.viscosity_params)
        self.speed_field = vf.get_speed_field(terrain, fog_data, spacing=spacing)

        # Feature extractor
        self.feature_extractor = FeatureExtractor(self.speed_field, spacing)

        # Path planner (same as original)
        self.planner = HierarchicalPathPlanner(
            terrain=terrain, fog_data=fog_data, n_levels=n_levels,
            viscosity_params=viscosity_params, spacing=spacing,
        )

        # Load neural network
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if model_path and os.path.exists(model_path):
            self.model = WaypointAllocationNet(self.config)
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print(f"   [Neural Allocator] Model loaded from {model_path}")
        elif model_path:
            print(f"   [Neural Allocator] Model not found at {model_path}")
            if not fallback_to_fmm:
                raise FileNotFoundError(f"Model not found: {model_path}")

    def allocate(self) -> WaypointAllocationResult:
        """
        Execute waypoint allocation and path planning.

        Returns WaypointAllocationResult (same format as MultiRobotWaypointAllocator).
        """
        total_start = time.time()

        if self.model is None:
            if self.fallback_to_fmm:
                print("   [Neural Allocator] No model loaded, falling back to CompetitiveFMM")
                return self._fallback_fmm_allocate()
            else:
                raise RuntimeError("No neural model loaded and fallback disabled")

        # Step 1: Extract features
        print("   [Neural Allocator] Extracting features...")
        feat_start = time.time()

        robot_starts = self.gas_params.robot_starts
        waypoints = self.gas_params.waypoints
        n_robots = self.gas_params.n_robots
        n_waypoints = len(waypoints)

        cost_matrix = self.feature_extractor.compute_cost_matrix(robot_starts, waypoints)
        robot_feats = self.feature_extractor.extract_robot_features(robot_starts)
        waypoint_feats = self.feature_extractor.extract_waypoint_features(waypoints)
        pairwise_feats = self.feature_extractor.extract_pairwise_features(
            robot_starts, waypoints, cost_matrix)

        # Normalize
        robot_feats, waypoint_feats, pairwise_feats = \
            self.feature_extractor.normalize_features(robot_feats, waypoint_feats, pairwise_feats)

        feat_time = time.time() - feat_start
        print(f"   [Neural Allocator] Feature extraction: {feat_time:.4f}s")

        # Step 2: Neural network inference
        print("   [Neural Allocator] Running neural network inference...")
        infer_start = time.time()

        # Pad to model's max sizes
        max_r = max(self.config.max_robots, n_robots)
        max_w = max(self.config.max_waypoints, n_waypoints)

        r_feats_padded = np.zeros((1, max_r, robot_feats.shape[1]), dtype=np.float32)
        r_feats_padded[0, :n_robots] = robot_feats

        w_feats_padded = np.zeros((1, max_w, waypoint_feats.shape[1]), dtype=np.float32)
        w_feats_padded[0, :n_waypoints] = waypoint_feats

        p_feats_padded = np.zeros((1, max_r, max_w, pairwise_feats.shape[2]), dtype=np.float32)
        p_feats_padded[0, :n_robots, :n_waypoints] = pairwise_feats

        r_mask = np.zeros((1, max_r), dtype=bool)
        r_mask[0, :n_robots] = True
        w_mask = np.zeros((1, max_w), dtype=bool)
        w_mask[0, :n_waypoints] = True

        with torch.no_grad():
            scores = self.model(
                torch.from_numpy(r_feats_padded).to(self.device),
                torch.from_numpy(w_feats_padded).to(self.device),
                torch.from_numpy(p_feats_padded).to(self.device),
                torch.from_numpy(r_mask).to(self.device),
                torch.from_numpy(w_mask).to(self.device),
            )

        score_matrix = scores[0].cpu().numpy()
        infer_time = time.time() - infer_start
        print(f"   [Neural Allocator] Inference: {infer_time:.4f}s")

        # Step 3: Decode assignment
        if self.config.decode_method == "hungarian":
            wp_assignments = hungarian_decode(
                score_matrix, n_waypoints, n_robots,
                self.gas_params.max_waypoints_per_robot,
            )
        else:
            wp_assignments = greedy_decode(
                score_matrix, n_waypoints, n_robots,
                self.gas_params.max_waypoints_per_robot,
            )

        # Step 3b: Makespan refinement — swap waypoints from busiest to lightest robot
        wp_assignments = self._refine_makespan(
            wp_assignments, cost_matrix, robot_starts, waypoints, n_robots)

        # Convert waypoint indices to actual coordinates + TSP ordering
        assignments = {}
        assignment_order = []

        for r in range(n_robots):
            wp_indices = wp_assignments.get(r, [])
            wp_coords = [waypoints[idx] for idx in wp_indices]

            # TSP ordering: nearest-neighbor heuristic from robot start
            if len(wp_coords) > 1:
                wp_coords = self._tsp_nearest_neighbor(
                    self.gas_params.robot_starts[r], wp_coords)

            assignments[r] = wp_coords
            for wp in wp_coords:
                wp_idx = waypoints.index(tuple(wp))
                assignment_order.append((r, wp_idx))

        print(f"   [Neural Allocator] Assignment done:")
        for r in range(n_robots):
            print(f"     Robot {r}: {len(assignments[r])} waypoints assigned")

        # Step 4: Plan path segments and compute actual arrival times
        print("   [Path Planning] Planning path segments...")
        plan_start = time.time()
        robot_paths = {}
        arrival_times_dict = {}

        for robot_id, wp_list in assignments.items():
            if not wp_list:
                robot_paths[robot_id] = []
                arrival_times_dict[robot_id] = []
                continue

            chain = [self.gas_params.robot_starts[robot_id]] + wp_list
            segments = []
            times = []
            cumulative_cost = 0.0

            for i in range(len(chain) - 1):
                seg_start = tuple(chain[i])
                seg_goal = tuple(chain[i + 1])
                try:
                    path, stats_seg = self.planner.plan(seg_start, seg_goal)
                    if path:
                        segments.append(path)
                        # Use actual path cost from planner
                        seg_cost = stats_seg.get('path_cost', 0)
                        if seg_cost <= 0:
                            seg_cost = self._estimate_path_cost(path)
                    else:
                        segments.append([seg_start, seg_goal])
                        seg_cost = self._estimate_segment_cost(seg_start, seg_goal)
                except Exception:
                    segments.append([seg_start, seg_goal])
                    seg_cost = self._estimate_segment_cost(seg_start, seg_goal)

                cumulative_cost += seg_cost
                times.append(cumulative_cost)

            robot_paths[robot_id] = segments
            arrival_times_dict[robot_id] = times

        plan_time = time.time() - plan_start
        total_time = time.time() - total_start

        # Build territory map (approximate: assign each cell to nearest robot by FMM cost)
        territory_map = np.full(self.speed_field.shape, -1, dtype=np.int32)
        # Simple version: mark waypoint neighborhoods
        for r, wp_list in assignments.items():
            for wp in wp_list:
                wp = tuple(wp)
                territory_map[wp] = r

        stats = {
            'success': sum(len(v) for v in assignments.values()) > 0,
            'n_robots': n_robots,
            'n_waypoints_total': n_waypoints,
            'n_waypoints_assigned': sum(len(v) for v in assignments.values()),
            'feature_extraction_time': feat_time,
            'inference_time': infer_time,
            'path_planning_time': plan_time,
            'total_time': total_time,
            'method': 'neural',
            'per_robot': {
                r: {
                    'n_waypoints': len(assignments[r]),
                    'n_segments': len(robot_paths.get(r, [])),
                    'arrival_times': arrival_times_dict[r],
                }
                for r in range(n_robots)
            }
        }

        return WaypointAllocationResult(
            assignments=assignments,
            assignment_order=assignment_order,
            robot_paths=robot_paths,
            arrival_times=arrival_times_dict,
            territory_map=territory_map,
            stats=stats,
        )

    def _fallback_fmm_allocate(self) -> WaypointAllocationResult:
        """Fallback to original CompetitiveFMM allocation."""
        from viscous_hierarchical_fmm import MultiRobotWaypointAllocator
        original = MultiRobotWaypointAllocator(
            terrain=self.terrain, fog_data=self.fog_data,
            n_levels=self.n_levels, viscosity_params=self.viscosity_params,
            gas_params=self.gas_params, spacing=self.spacing,
        )
        return original.allocate()

    def _estimate_path_cost(self, path) -> float:
        """Estimate cost along a planned path by summing inverse speed."""
        cost = 0.0
        for i in range(len(path) - 1):
            p1 = tuple(int(round(c)) for c in path[i])
            p2 = tuple(int(round(c)) for c in path[i + 1])
            p1 = tuple(max(0, min(c, s - 1)) for c, s in zip(p1, self.speed_field.shape))
            speed = self.speed_field[p1]
            dist = sum(((a - b) * h) ** 2
                        for a, b, h in zip(path[i], path[i + 1],
                                            self.spacing[:self.speed_field.ndim])) ** 0.5
            cost += dist / max(speed, 1e-6)
        return cost

    def _estimate_segment_cost(self, start, goal) -> float:
        """Estimate cost for a direct segment (no planned path)."""
        return self.feature_extractor._line_integral_cost(start, goal)

    def _tsp_nearest_neighbor(self, start, waypoints_list):
        """Order waypoints by nearest-neighbor TSP heuristic from start."""
        if len(waypoints_list) <= 1:
            return waypoints_list

        remaining = list(waypoints_list)
        ordered = []
        current = tuple(start)

        while remaining:
            # Find nearest remaining waypoint
            best_idx = 0
            best_cost = float('inf')
            for i, wp in enumerate(remaining):
                cost = self.feature_extractor._line_integral_cost(current, tuple(wp))
                if cost < best_cost:
                    best_cost = cost
                    best_idx = i
            ordered.append(remaining.pop(best_idx))
            current = tuple(ordered[-1])

        return ordered

    def _compute_robot_cost(self, robot_idx, wp_indices, cost_matrix, robot_starts, waypoints):
        """Compute total sequential travel cost for a robot's waypoint chain."""
        if not wp_indices:
            return 0.0

        wp_coords = [waypoints[idx] for idx in wp_indices]
        start = robot_starts[robot_idx]

        # TSP order
        if len(wp_coords) > 1:
            wp_coords = self._tsp_nearest_neighbor(start, wp_coords)

        total = self.feature_extractor._line_integral_cost(tuple(start), tuple(wp_coords[0]))
        for i in range(len(wp_coords) - 1):
            total += self.feature_extractor._line_integral_cost(
                tuple(wp_coords[i]), tuple(wp_coords[i + 1]))
        return total

    def _refine_makespan(self, wp_assignments, cost_matrix, robot_starts, waypoints, n_robots,
                         max_iterations=50):
        """
        Local search refinement: iteratively move waypoints from the busiest
        robot to less busy robots to minimize makespan.
        """
        assignments = {r: list(wp_assignments.get(r, [])) for r in range(n_robots)}

        for iteration in range(max_iterations):
            # Compute costs
            costs = {}
            for r in range(n_robots):
                costs[r] = self._compute_robot_cost(
                    r, assignments[r], cost_matrix, robot_starts, waypoints)

            busiest = max(costs, key=costs.get)
            current_makespan = costs[busiest]

            if len(assignments[busiest]) <= 1:
                break

            improved = False
            for wp_idx in list(assignments[busiest]):
                for target_r in range(n_robots):
                    if target_r == busiest:
                        continue

                    new_busiest_list = [w for w in assignments[busiest] if w != wp_idx]
                    new_target_list = assignments[target_r] + [wp_idx]

                    new_busiest_cost = self._compute_robot_cost(
                        busiest, new_busiest_list, cost_matrix, robot_starts, waypoints)
                    new_target_cost = self._compute_robot_cost(
                        target_r, new_target_list, cost_matrix, robot_starts, waypoints)

                    new_makespan = max(new_busiest_cost, new_target_cost,
                                       max(costs[r] for r in range(n_robots)
                                           if r not in (busiest, target_r)))

                    if new_makespan < current_makespan * 0.99:
                        assignments[busiest] = new_busiest_list
                        assignments[target_r] = new_target_list
                        improved = True
                        break
                if improved:
                    break

            if not improved:
                break

        return assignments
