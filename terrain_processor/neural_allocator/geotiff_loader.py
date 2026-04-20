"""
GeoTIFF integration for the Neural Allocator.

Loads terrain from .tif files, handles coordinate transforms,
and provides a high-level interface for large-scale (50km+) planning.
"""

import numpy as np
from typing import Tuple, List, Optional
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from .config import NeuralAllocatorConfig
from .hierarchical_features import HierarchicalFeatureExtractor
from .allocator import NeuralWaypointAllocator

from viscous_hierarchical_fmm import (
    ViscosityParams, GasDiffusionParams, WaypointAllocationResult,
    HierarchicalPathPlanner, ViscosityField,
)


class GeoTIFFAllocator:
    """
    Multi-UAV waypoint allocator for GeoTIFF terrain data.

    Handles:
    - Loading terrain from .tif files
    - Geographic ↔ pixel coordinate conversion
    - Hierarchical feature extraction for large grids
    - Neural network-based allocation with path planning
    """

    def __init__(self,
                 terrain: np.ndarray,
                 spacing: Tuple[float, float],
                 transform=None,
                 model_path: Optional[str] = None,
                 config: Optional[NeuralAllocatorConfig] = None,
                 viscosity_params: Optional[ViscosityParams] = None,
                 n_pyramid_levels: int = 4):
        """
        Args:
            terrain: 2D array (height × width)
            spacing: (y_spacing, x_spacing) in meters
            transform: Affine transform from rasterio (for geo↔pixel conversion)
            model_path: Path to trained neural network checkpoint
            config: Neural allocator configuration
            viscosity_params: Viscosity parameters for cost computation
            n_pyramid_levels: Number of pyramid levels for hierarchical processing
        """
        self.terrain = terrain
        self.spacing = spacing
        self.transform = transform
        self.config = config or NeuralAllocatorConfig.large()
        self.viscosity_params = viscosity_params or ViscosityParams()

        # Build hierarchical feature extractor
        self.hier_fe = HierarchicalFeatureExtractor(
            terrain=terrain,
            n_levels=n_pyramid_levels,
            viscosity_params=viscosity_params,
            spacing=spacing,
        )

        # Path planner at original resolution
        self.planner = HierarchicalPathPlanner(
            terrain=terrain,
            n_levels=min(n_pyramid_levels, 3),
            viscosity_params=viscosity_params,
            spacing=spacing,
        )

        # Neural model
        self.model_path = model_path

    @classmethod
    def from_geotiff(cls, tif_path: str,
                     model_path: Optional[str] = None,
                     config: Optional[NeuralAllocatorConfig] = None,
                     **kwargs):
        """
        Create allocator from a GeoTIFF file.

        Args:
            tif_path: Path to .tif file
            model_path: Path to trained neural network
            config: Allocator configuration

        Returns:
            GeoTIFFAllocator instance
        """
        # Import tif reader
        tif_dir = os.path.join(os.path.dirname(__file__), '..', 'tif')
        sys.path.insert(0, tif_dir)
        from terrain_processor.reader import TIFReader

        reader = TIFReader(tif_path)
        terrain, metadata = reader.read_tif()

        # Extract spacing from metadata
        pixel_size = metadata.get('pixel_size', (1.0, 1.0))
        spacing = (abs(pixel_size[1]), abs(pixel_size[0]))  # (y, x) in meters

        # Handle NaN values
        if np.any(np.isnan(terrain)):
            terrain = np.nan_to_num(terrain, nan=np.nanmean(terrain))

        transform = metadata.get('transform', None)

        print(f"Loaded terrain: {terrain.shape} @ {spacing}m spacing")
        print(f"Coverage: {terrain.shape[0]*spacing[0]/1000:.1f}km × "
              f"{terrain.shape[1]*spacing[1]/1000:.1f}km")

        return cls(
            terrain=terrain,
            spacing=spacing,
            transform=transform,
            model_path=model_path,
            config=config,
            **kwargs,
        )

    def geo_to_pixel(self, geo_coords: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
        """Convert geographic coordinates to pixel coordinates."""
        if self.transform is None:
            raise ValueError("No affine transform available. Use pixel coordinates directly.")

        pixel_coords = []
        inv_transform = ~self.transform
        for lon, lat in geo_coords:
            col, row = inv_transform * (lon, lat)
            row = max(0, min(int(round(row)), self.terrain.shape[0] - 1))
            col = max(0, min(int(round(col)), self.terrain.shape[1] - 1))
            pixel_coords.append((row, col))
        return pixel_coords

    def pixel_to_geo(self, pixel_coords: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
        """Convert pixel coordinates to geographic coordinates."""
        if self.transform is None:
            raise ValueError("No affine transform available.")

        geo_coords = []
        for row, col in pixel_coords:
            lon, lat = self.transform * (col, row)
            geo_coords.append((lon, lat))
        return geo_coords

    def allocate(self,
                 robot_starts: List[Tuple[int, int]],
                 waypoints: List[Tuple[int, int]],
                 max_waypoints_per_robot: Optional[int] = None
                 ) -> WaypointAllocationResult:
        """
        Allocate waypoints to robots and plan paths.

        Args:
            robot_starts: List of (row, col) pixel coordinates for robot start positions
            waypoints: List of (row, col) pixel coordinates for waypoints
            max_waypoints_per_robot: Optional max waypoints per robot

        Returns:
            WaypointAllocationResult
        """
        n_robots = len(robot_starts)

        gas_params = GasDiffusionParams(
            n_robots=n_robots,
            robot_starts=robot_starts,
            waypoints=waypoints,
            max_waypoints_per_robot=max_waypoints_per_robot,
        )

        # Use NeuralWaypointAllocator with hierarchical feature extractor
        allocator = NeuralWaypointAllocator(
            terrain=self.terrain,
            gas_params=gas_params,
            spacing=self.spacing,
            model_path=self.model_path,
            config=self.config,
            viscosity_params=self.viscosity_params,
            fallback_to_fmm=True,
        )

        # Override feature extractor with hierarchical version
        allocator.feature_extractor = self.hier_fe

        return allocator.allocate()

    def allocate_geo(self,
                     robot_starts_geo: List[Tuple[float, float]],
                     waypoints_geo: List[Tuple[float, float]],
                     **kwargs) -> WaypointAllocationResult:
        """
        Allocate using geographic coordinates.

        Args:
            robot_starts_geo: List of (lon, lat) for robot starts
            waypoints_geo: List of (lon, lat) for waypoints
        """
        robot_starts = self.geo_to_pixel(robot_starts_geo)
        waypoints = self.geo_to_pixel(waypoints_geo)
        return self.allocate(robot_starts, waypoints, **kwargs)
