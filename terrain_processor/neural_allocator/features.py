"""
Feature extraction: bridges existing FMM cost computation with neural network inputs.

Supports both fast (vectorized line-integral) and exact (FMM) cost matrix computation.
Position features are normalized to [0,1] for generalization across terrain scales.
"""

import numpy as np
from typing import List, Tuple, Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from viscous_hierarchical_fmm import NarrowBandFMM


class FeatureExtractor:
    """Extracts features from terrain/robot/waypoint configuration for neural network."""

    def __init__(self, speed_field: np.ndarray,
                 spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)):
        self.speed_field = speed_field
        self.spacing = spacing
        self.ndim = speed_field.ndim
        self.shape = speed_field.shape

    def compute_cost_matrix(self, robot_starts: List[Tuple[int, ...]],
                            waypoints: List[Tuple[int, ...]],
                            fast: bool = True) -> np.ndarray:
        """Compute cost matrix. fast=True uses vectorized line-integral (~1000x faster)."""
        if fast:
            return self._compute_cost_matrix_fast(robot_starts, waypoints)
        return self._compute_cost_matrix_fmm(robot_starts, waypoints)

    def _compute_cost_matrix_fast(self, robot_starts, waypoints) -> np.ndarray:
        """Vectorized line-integral cost approximation."""
        robots = np.array(robot_starts, dtype=np.float64)   # (n_r, ndim)
        wps = np.array(waypoints, dtype=np.float64)          # (n_w, ndim)
        n_r, n_w = len(robots), len(wps)

        if n_r == 0 or n_w == 0:
            return np.zeros((n_r, n_w), dtype=np.float64)

        sp = np.array(self.spacing[:self.ndim], dtype=np.float64)

        # Broadcast: diff (n_r, n_w, ndim)
        diff = wps[None, :, :] - robots[:, None, :]
        dists = np.linalg.norm(diff * sp[None, None, :], axis=-1)  # (n_r, n_w)

        # Sample along each line
        n_samples = 30
        t = (np.arange(n_samples, dtype=np.float64) + 0.5) / n_samples  # (S,)

        # Interpolated points: (n_r, n_w, S, ndim)
        points = robots[:, None, None, :] + t[None, None, :, None] * diff[:, :, None, :]

        # Clamp to valid grid indices
        for d in range(self.ndim):
            np.clip(points[:, :, :, d], 0, self.shape[d] - 1, out=points[:, :, :, d])
        idx = points.astype(np.intp)

        # Look up speed field values (advanced indexing)
        if self.ndim == 2:
            speeds = self.speed_field[idx[:, :, :, 0], idx[:, :, :, 1]]
        elif self.ndim == 3:
            speeds = self.speed_field[idx[:, :, :, 0], idx[:, :, :, 1], idx[:, :, :, 2]]
        else:
            # Fallback for arbitrary ndim
            speeds = np.zeros((n_r, n_w, n_samples))
            for d_idx in np.ndindex(n_r, n_w, n_samples):
                pt = tuple(idx[d_idx])
                speeds[d_idx] = self.speed_field[pt]

        # Cost = mean(1/speed) * distance
        speeds = np.maximum(speeds, 1e-6)
        mean_slowness = np.mean(1.0 / speeds, axis=-1)  # (n_r, n_w)
        cost_matrix = mean_slowness * dists

        return cost_matrix

    def _line_integral_cost(self, p1: Tuple[int, ...], p2: Tuple[int, ...]) -> float:
        """Single pair line-integral cost (for TSP ordering etc.)."""
        n_samples = max(int(sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5), 5)
        n_samples = min(n_samples, 50)

        total_cost = 0.0
        for k in range(n_samples):
            t = (k + 0.5) / n_samples
            point = tuple(
                max(0, min(int(a + t * (b - a)), s - 1))
                for a, b, s in zip(p1, p2, self.speed_field.shape)
            )
            speed = self.speed_field[point]
            total_cost += 1.0 / max(speed, 1e-6)

        dist = sum(((a - b) * h) ** 2
                    for a, b, h in zip(p1, p2, self.spacing[:self.ndim])) ** 0.5
        return total_cost * dist / n_samples

    def _compute_cost_matrix_fmm(self, robot_starts, waypoints) -> np.ndarray:
        """Exact FMM cost matrix (slower, for inference)."""
        n_robots = len(robot_starts)
        n_waypoints = len(waypoints)
        cost_matrix = np.full((n_robots, n_waypoints), np.inf, dtype=np.float64)

        fmm = NarrowBandFMM(self.speed_field.shape, self.spacing[:self.ndim])
        fmm.set_speed_field(self.speed_field)

        for i, start in enumerate(robot_starts):
            start = tuple(start)
            fmm.reset()
            farthest_wp = max(waypoints,
                              key=lambda wp: sum((a - b) ** 2 for a, b in zip(wp, start)))
            fmm.solve(start, tuple(farthest_wp))
            for j, wp in enumerate(waypoints):
                phi_val = fmm.get_phi_value(tuple(wp))
                if phi_val is not None and phi_val < float('inf'):
                    cost_matrix[i, j] = phi_val

        finite_mask = np.isfinite(cost_matrix)
        if finite_mask.any():
            cost_matrix[~finite_mask] = cost_matrix[finite_mask].max() * 2.0
        return cost_matrix

    def extract_robot_features(self, robot_starts: List[Tuple[int, ...]],
                               velocities=None, current_loads=None) -> np.ndarray:
        """Robot features with normalized positions [0,1]."""
        n = len(robot_starts)
        feats = np.zeros((n, 7), dtype=np.float32)
        for i, pos in enumerate(robot_starts):
            for d in range(min(len(pos), 3)):
                feats[i, d] = pos[d] / max(self.shape[d] - 1, 1)  # normalize to [0,1]
            if velocities and i < len(velocities):
                for d in range(min(len(velocities[i]), 3)):
                    feats[i, 3 + d] = velocities[i][d]
            if current_loads and i < len(current_loads):
                feats[i, 6] = current_loads[i]
        return feats

    def extract_waypoint_features(self, waypoints: List[Tuple[int, ...]],
                                  q_values=None) -> np.ndarray:
        """Waypoint features with normalized positions [0,1]."""
        n = len(waypoints)
        feats = np.zeros((n, 5), dtype=np.float32)
        for i, wp in enumerate(waypoints):
            wp = tuple(wp)
            for d in range(min(len(wp), 3)):
                feats[i, d] = wp[d] / max(self.shape[d] - 1, 1)  # normalize to [0,1]
            speed = self.speed_field[wp]
            feats[i, 3] = 1.0 / max(speed, 1e-6)
            if q_values and i < len(q_values):
                feats[i, 4] = q_values[i]
        return feats

    def extract_pairwise_features(self, robot_starts, waypoints,
                                  cost_matrix: np.ndarray) -> np.ndarray:
        """Vectorized pairwise features: (fmm_cost, euclidean_dist)."""
        robots = np.array(robot_starts, dtype=np.float32)
        wps = np.array(waypoints, dtype=np.float32)
        # Vectorized Euclidean distance
        dist = np.linalg.norm(robots[:, None, :] - wps[None, :, :], axis=-1)
        feats = np.stack([cost_matrix.astype(np.float32), dist], axis=-1)
        return feats

    def normalize_features(self, robot_feats, waypoint_feats, pairwise_feats):
        """Normalize pairwise features only (positions already in [0,1])."""
        def _norm(arr):
            axes = tuple(range(arr.ndim - 1))
            mean = arr.mean(axis=axes, keepdims=True)
            std = arr.std(axis=axes, keepdims=True) + 1e-8
            return ((arr - mean) / std).astype(np.float32)

        # Robot and waypoint positions are already normalized to [0,1]
        # Only normalize pairwise features (cost, distance have varying scales)
        return robot_feats, waypoint_feats, _norm(pairwise_feats)
