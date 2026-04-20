"""
Data augmentation for training scenarios.

Applies geometric transforms (rotation, flip) and perturbations
(coordinate jitter, speed field noise) to multiply effective dataset size.
"""

import numpy as np
from typing import List
from .data_generator import ScenarioData


def augment_scenario(scenario: ScenarioData, n_augments: int = 4) -> List[ScenarioData]:
    """
    Generate augmented copies of a scenario.

    Augmentations:
    - Geometric: 90/180/270° rotation + horizontal/vertical flip
    - Speed field perturbation: multiply by uniform [0.9, 1.1]
    - Coordinate jitter: Gaussian noise sigma=0.5 on robot/waypoint positions

    Returns list of augmented ScenarioData (NOT including the original).
    """
    augmented = []

    for _ in range(n_augments):
        terrain = scenario.terrain.copy()
        speed = scenario.speed_field.copy()
        fog = scenario.fog_data.copy() if scenario.fog_data is not None else None

        ndim = terrain.ndim
        robots = [list(r) for r in scenario.robot_starts]
        waypoints = [list(w) for w in scenario.waypoints]
        shape = terrain.shape

        # 1. Random rotation (2D plane rotation for last 2 axes)
        k = np.random.randint(0, 4)  # 0, 90, 180, 270 degrees
        if k > 0:
            axes = (-2, -1) if ndim >= 2 else (0, 1)
            terrain = np.rot90(terrain, k, axes=axes).copy()
            speed = np.rot90(speed, k, axes=axes).copy()
            if fog is not None:
                fog = np.rot90(fog, k, axes=axes).copy()
            shape = terrain.shape
            robots = [_rotate_coord(r, k, scenario.terrain.shape, ndim) for r in robots]
            waypoints = [_rotate_coord(w, k, scenario.terrain.shape, ndim) for w in waypoints]

        # 2. Random flip
        if np.random.random() < 0.5:
            axis = -1  # horizontal flip
            terrain = np.flip(terrain, axis=axis).copy()
            speed = np.flip(speed, axis=axis).copy()
            if fog is not None:
                fog = np.flip(fog, axis=axis).copy()
            robots = [_flip_coord(r, axis, terrain.shape, ndim) for r in robots]
            waypoints = [_flip_coord(w, axis, terrain.shape, ndim) for w in waypoints]

        if np.random.random() < 0.5:
            axis = -2  # vertical flip
            terrain = np.flip(terrain, axis=axis).copy()
            speed = np.flip(speed, axis=axis).copy()
            if fog is not None:
                fog = np.flip(fog, axis=axis).copy()
            robots = [_flip_coord(r, axis, terrain.shape, ndim) for r in robots]
            waypoints = [_flip_coord(w, axis, terrain.shape, ndim) for w in waypoints]

        # 3. Speed field perturbation
        scale = np.random.uniform(0.9, 1.1)
        speed = speed * scale

        # 4. Coordinate jitter (small Gaussian noise, clamped to valid range)
        sigma = 0.5
        robots = [_jitter_coord(r, sigma, terrain.shape) for r in robots]
        waypoints = [_jitter_coord(w, sigma, terrain.shape) for w in waypoints]

        # Ensure no duplicates between robots and waypoints
        all_points = set()
        valid_robots = []
        for r in robots:
            t = tuple(r)
            if t not in all_points:
                all_points.add(t)
                valid_robots.append(t)
        valid_waypoints = []
        for w in waypoints:
            t = tuple(w)
            if t not in all_points:
                all_points.add(t)
                valid_waypoints.append(t)

        if len(valid_robots) < scenario.n_robots or len(valid_waypoints) < scenario.n_waypoints:
            continue  # Skip invalid augmentation

        augmented.append(ScenarioData(
            terrain=terrain,
            fog_data=fog,
            speed_field=speed,
            robot_starts=valid_robots,
            waypoints=valid_waypoints,
            spacing=scenario.spacing,
            n_robots=scenario.n_robots,
            n_waypoints=scenario.n_waypoints,
        ))

    return augmented


def _rotate_coord(coord, k, original_shape, ndim):
    """Rotate coordinate k*90 degrees in the last two axes."""
    coord = list(coord)
    h, w = original_shape[-2], original_shape[-1]
    y_idx, x_idx = ndim - 2, ndim - 1

    for _ in range(k % 4):
        old_y, old_x = coord[y_idx], coord[x_idx]
        coord[y_idx] = old_x
        coord[x_idx] = h - 1 - old_y
        h, w = w, h  # dimensions swap after 90° rotation

    return coord


def _flip_coord(coord, axis, shape, ndim):
    """Flip coordinate along given axis."""
    coord = list(coord)
    real_axis = axis % ndim
    coord[real_axis] = shape[real_axis] - 1 - coord[real_axis]
    return coord


def _jitter_coord(coord, sigma, shape):
    """Add Gaussian jitter, clamped to valid range."""
    coord = list(coord)
    for d in range(len(coord)):
        noise = np.random.normal(0, sigma)
        coord[d] = max(0, min(int(round(coord[d] + noise)), shape[d] - 1))
    return coord
