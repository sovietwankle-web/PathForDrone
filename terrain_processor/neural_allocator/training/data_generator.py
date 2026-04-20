"""
Random scenario generation for training data.

Generates random terrain + fog + robot starts + waypoints configurations.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from viscous_hierarchical_fmm import ViscosityField, ViscosityParams


@dataclass
class ScenarioData:
    """A single training scenario."""
    terrain: np.ndarray
    fog_data: Optional[np.ndarray]
    speed_field: np.ndarray
    robot_starts: List[Tuple[int, ...]]
    waypoints: List[Tuple[int, ...]]
    spacing: Tuple[float, ...]
    n_robots: int
    n_waypoints: int


def generate_random_terrain_3d(shape: Tuple[int, int, int]) -> np.ndarray:
    """Generate random 3D terrain with sinusoidal hills and obstacles."""
    z, y, x = np.meshgrid(
        np.linspace(0, 2 * np.pi, shape[0]),
        np.linspace(0, 2 * np.pi, shape[1]),
        np.linspace(0, 2 * np.pi, shape[2]),
        indexing='ij'
    )

    # Random frequency hills
    terrain = np.zeros(shape, dtype=np.float64)
    n_hills = np.random.randint(2, 6)
    for _ in range(n_hills):
        freq = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2 * np.pi, 3)
        amp = np.random.uniform(0.3, 1.5)
        terrain += amp * np.sin(freq * x + phase[0]) * np.cos(freq * y + phase[1])

    # Random obstacles (high-cost regions)
    n_obstacles = np.random.randint(0, 4)
    for _ in range(n_obstacles):
        center = [np.random.randint(s // 4, 3 * s // 4) for s in shape]
        radius = np.random.randint(2, max(3, min(shape) // 4))
        zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
        dist = ((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2) ** 0.5
        terrain += 5.0 * np.exp(-dist ** 2 / (2 * radius ** 2))

    return terrain


def generate_random_fog_3d(shape: Tuple[int, int, int]) -> np.ndarray:
    """Generate random fog field."""
    fog = np.zeros(shape, dtype=np.float64)
    n_clouds = np.random.randint(1, 4)
    for _ in range(n_clouds):
        center = [np.random.randint(0, s) for s in shape]
        sigma = np.random.uniform(3, min(shape) // 3)
        intensity = np.random.uniform(0.3, 1.0)
        zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
        dist2 = ((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2)
        fog += intensity * np.exp(-dist2 / (2 * sigma ** 2))
    return np.clip(fog, 0, 1)


def _generate_terrain_2d(shape: Tuple[int, int]) -> np.ndarray:
    """Generate 2D terrain with sinusoidal hills and random obstacles."""
    y, x = np.meshgrid(
        np.linspace(0, 2 * np.pi, shape[0]),
        np.linspace(0, 2 * np.pi, shape[1]),
        indexing='ij'
    )
    terrain = np.zeros(shape, dtype=np.float64)
    n_hills = np.random.randint(2, 6)
    for _ in range(n_hills):
        freq = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2 * np.pi, 2)
        amp = np.random.uniform(0.3, 1.5)
        terrain += amp * np.sin(freq * x + phase[0]) * np.cos(freq * y + phase[1])

    n_obstacles = np.random.randint(0, 4)
    for _ in range(n_obstacles):
        center = [np.random.randint(s // 4, 3 * s // 4) for s in shape]
        radius = np.random.randint(2, max(3, min(shape) // 4))
        yy, xx = np.ogrid[:shape[0], :shape[1]]
        dist = ((yy - center[0]) ** 2 + (xx - center[1]) ** 2) ** 0.5
        terrain += 5.0 * np.exp(-dist ** 2 / (2 * radius ** 2))
    return terrain


def _generate_fog_2d(shape: Tuple[int, int]) -> np.ndarray:
    """Generate 2D fog field."""
    fog = np.zeros(shape, dtype=np.float64)
    n_clouds = np.random.randint(1, 4)
    for _ in range(n_clouds):
        center = [np.random.randint(0, s) for s in shape]
        sigma = np.random.uniform(3, max(4, min(shape) // 3))
        intensity = np.random.uniform(0.3, 1.0)
        yy, xx = np.ogrid[:shape[0], :shape[1]]
        dist2 = ((yy - center[0]) ** 2 + (xx - center[1]) ** 2)
        fog += intensity * np.exp(-dist2 / (2 * sigma ** 2))
    return np.clip(fog, 0, 1)


def _random_valid_points(speed_field: np.ndarray, n: int,
                          min_dist: float = 3.0) -> List[Tuple[int, ...]]:
    """Sample n random points on the terrain where speed > threshold, with min separation."""
    shape = speed_field.shape
    threshold = speed_field.mean() * 0.3
    valid_mask = speed_field > threshold

    valid_coords = np.argwhere(valid_mask)
    if len(valid_coords) < n:
        valid_coords = np.argwhere(speed_field > 0)

    points = []
    indices = np.random.permutation(len(valid_coords))
    for idx in indices:
        pt = tuple(valid_coords[idx].tolist())
        # Check min distance to existing points
        if all(sum((a - b) ** 2 for a, b in zip(pt, existing)) ** 0.5 >= min_dist
               for existing in points):
            points.append(pt)
            if len(points) >= n:
                break

    # Fallback: just pick random points if distance constraint too tight
    while len(points) < n:
        idx = np.random.randint(len(valid_coords))
        pt = tuple(valid_coords[idx].tolist())
        if pt not in points:
            points.append(pt)

    return points


def generate_scenario(terrain_size_range: Tuple[int, int] = (32, 64),
                      n_robots_range: Tuple[int, int] = (2, 6),
                      n_waypoints_range: Tuple[int, int] = (4, 20),
                      ndim: int = 2) -> ScenarioData:
    """Generate a complete random scenario for training.

    Args:
        ndim: 2 for 2D terrain (default, faster), 3 for 3D.
    """
    size = np.random.randint(terrain_size_range[0], terrain_size_range[1] + 1)

    if ndim == 3:
        z_size = max(size // 3, 8)
        shape = (z_size, size, size)
        terrain = generate_random_terrain_3d(shape)
        fog = generate_random_fog_3d(shape)
    else:
        shape = (size, size)
        terrain = _generate_terrain_2d(shape)
        fog = _generate_fog_2d(shape)

    spacing = (1.0,) * len(shape)

    # Compute speed field
    vf = ViscosityField(ViscosityParams())
    speed_field = vf.get_speed_field(terrain, fog, spacing=spacing)

    n_robots = np.random.randint(n_robots_range[0], n_robots_range[1] + 1)
    n_waypoints = np.random.randint(n_waypoints_range[0], n_waypoints_range[1] + 1)

    total_points = n_robots + n_waypoints
    all_points = _random_valid_points(speed_field, total_points, min_dist=2.0)

    robot_starts = all_points[:n_robots]
    waypoints = all_points[n_robots:]

    return ScenarioData(
        terrain=terrain,
        fog_data=fog,
        speed_field=speed_field,
        robot_starts=robot_starts,
        waypoints=waypoints,
        spacing=spacing,
        n_robots=n_robots,
        n_waypoints=n_waypoints,
    )


def generate_dataset(n_scenarios: int, **kwargs) -> List[ScenarioData]:
    """Generate multiple scenarios."""
    scenarios = []
    for i in range(n_scenarios):
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{n_scenarios} scenarios")
        scenarios.append(generate_scenario(**kwargs))
    return scenarios
