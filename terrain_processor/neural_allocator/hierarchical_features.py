"""
Hierarchical Feature Extractor for large-scale (50km+) terrains.

Uses TerrainPyramid's coarse levels to compute cost matrices efficiently.
For a 1000x1000 grid with 4-level pyramid:
  Level 0: 1000x1000 (original)
  Level 1: 500x500
  Level 2: 250x250
  Level 3: 125x125  ← cost matrix computed here (~ms)
"""

import numpy as np
from typing import Tuple, List, Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from viscous_hierarchical_fmm import (
    TerrainPyramid, ViscosityField, ViscosityParams
)
from .features import FeatureExtractor


class HierarchicalFeatureExtractor(FeatureExtractor):
    """
    Feature extractor that uses multi-resolution terrain pyramid.
    Computes cost matrix on coarse level for speed, while maintaining
    accuracy through terrain-aware cost estimation.
    """

    def __init__(self,
                 terrain: np.ndarray,
                 fog_data: Optional[np.ndarray] = None,
                 n_levels: int = 4,
                 coarse_level: Optional[int] = None,
                 viscosity_params: Optional[ViscosityParams] = None,
                 spacing: Tuple[float, ...] = (1.0, 1.0)):
        """
        Args:
            terrain: Full-resolution terrain data
            fog_data: Optional fog field
            n_levels: Number of pyramid levels
            coarse_level: Which level to use for cost matrix (default: coarsest)
            viscosity_params: Viscosity parameters
            spacing: Grid spacing at original resolution
        """
        self.terrain = terrain
        self.fog_data = fog_data
        self.n_levels = n_levels
        self.original_spacing = spacing
        self.original_shape = terrain.shape

        # Build viscosity field and speed field at original resolution
        vparams = viscosity_params or ViscosityParams()
        vf = ViscosityField(vparams)
        self.full_speed_field = vf.get_speed_field(terrain, fog_data, spacing=spacing)

        # Build terrain pyramid
        self.pyramid = TerrainPyramid(
            terrain=terrain,
            n_levels=n_levels,
            spacing=spacing,
            viscosity_field=vf,
        )

        # Select coarse level for cost computation
        self.coarse_level = coarse_level if coarse_level is not None else n_levels - 1
        _, coarse_speed, coarse_spacing = self.pyramid.get_level(self.coarse_level)
        self.scale_factor = 2 ** self.coarse_level

        # Initialize parent with coarse speed field
        super().__init__(coarse_speed, coarse_spacing)

        # Keep reference to full-res for path planning
        self._full_fe = FeatureExtractor(self.full_speed_field, spacing)

    def _to_coarse(self, coord: Tuple[int, ...]) -> Tuple[int, ...]:
        """Map original-resolution coordinate to coarse level."""
        return self.pyramid.coord_to_level(coord, 0, self.coarse_level)

    def _to_original(self, coarse_coord: Tuple[int, ...]) -> Tuple[int, ...]:
        """Map coarse coordinate back to original resolution."""
        return self.pyramid.coord_from_level(coarse_coord, self.coarse_level, 0)

    def compute_cost_matrix(self, robot_starts, waypoints, fast=True):
        """
        Compute cost matrix on coarse level for speed.
        Maps coordinates to coarse resolution, computes costs, returns.
        """
        coarse_robots = [self._to_coarse(tuple(r)) for r in robot_starts]
        coarse_waypoints = [self._to_coarse(tuple(w)) for w in waypoints]

        # Use vectorized fast computation on coarse grid
        return self._compute_cost_matrix_fast(coarse_robots, coarse_waypoints)

    def extract_robot_features(self, robot_starts, velocities=None, current_loads=None):
        """Robot features with positions normalized to [0,1] using ORIGINAL shape."""
        n = len(robot_starts)
        feats = np.zeros((n, 7), dtype=np.float32)
        for i, pos in enumerate(robot_starts):
            for d in range(min(len(pos), 3)):
                # Normalize using original shape for consistency
                dim_size = self.original_shape[d] if d < len(self.original_shape) else 1
                feats[i, d] = pos[d] / max(dim_size - 1, 1)
            if velocities and i < len(velocities):
                for d in range(min(len(velocities[i]), 3)):
                    feats[i, 3 + d] = velocities[i][d]
            if current_loads and i < len(current_loads):
                feats[i, 6] = current_loads[i]
        return feats

    def extract_waypoint_features(self, waypoints, q_values=None):
        """Waypoint features with positions normalized using original shape."""
        n = len(waypoints)
        feats = np.zeros((n, 5), dtype=np.float32)
        for i, wp in enumerate(waypoints):
            wp = tuple(wp)
            for d in range(min(len(wp), 3)):
                dim_size = self.original_shape[d] if d < len(self.original_shape) else 1
                feats[i, d] = wp[d] / max(dim_size - 1, 1)
            # Use original-resolution speed for terrain cost
            speed = self.full_speed_field[wp]
            feats[i, 3] = 1.0 / max(speed, 1e-6)
            if q_values and i < len(q_values):
                feats[i, 4] = q_values[i]
        return feats

    def get_full_resolution_extractor(self) -> FeatureExtractor:
        """Get a FeatureExtractor at original resolution (for TSP ordering etc.)."""
        return self._full_fe

    def get_pyramid(self) -> TerrainPyramid:
        """Access the terrain pyramid for path planning."""
        return self.pyramid
